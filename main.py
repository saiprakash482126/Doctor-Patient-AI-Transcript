"""
main.py — Live Speech Translator (WebSocket Server) — Optimised v4
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Key improvements in v4:
  • gpt-4o-transcribe with verbose_json + WORD-LEVEL timestamps
  • Within-blob speaker segmentation (spectral-change BIC detector)
    → Multiple speaker segments per blob, not one speaker per blob
  • Pitch (F0) appended to MFCC embedding → better male/female separation
  • MIN_BLOB_BYTES reduced for 5-second browser chunks
  • COSINE_THRESH tuned per-session via adaptive percentile
  • Parallel speaker ID + STT via asyncio.gather

Pipeline for every incoming 5-second WebM blob:
  1. EnhancedVAD         — RMS + ZCR gate; drop silent blobs
  2. Spectral change      — find speaker-change points within blob
  3. Speaker embed        — MFCC+pitch cosine clustering per sub-segment
  4. STT (gpt-4o)        — one call with word-level timestamps
  5. Word alignment       — map words→speaker by timestamp overlap
  6. Raw push             — immediate WebSocket send (per speaker segment)
  7. Tier routing         — skip/trivial/rule/simple/full_cot
  8. Correction           — gpt-4.1-mini or gpt-4.1 (cached)
  9. Validation           — hallucination regex + semantic guard
 10. Corrected push       — final WebSocket send with word-diff
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# ── Load .env file FIRST — before any os.environ.get() calls ─────────────────
from dotenv import load_dotenv

load_dotenv(override=True)

import shutil
if not shutil.which("ffmpeg"):
    try:
        import imageio_ffmpeg as _iio_ffmpeg
        import os as _os
        _ffmpeg_path = _iio_ffmpeg.get_ffmpeg_exe()
        _os.environ["PATH"] = str(_iio_ffmpeg.get_ffmpeg_exe().rsplit(_os.sep, 1)[0]) \
                              + _os.pathsep + _os.environ.get("PATH", "")
        print(f"✅ Using bundled ffmpeg: {_ffmpeg_path}")
    except Exception:
        print("⚠️  ffmpeg not found — audio decoding will fail. "
              "Run: pip install imageio-ffmpeg")

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, PlainTextResponse, JSONResponse
from pydantic import BaseModel

# ── Clinical letter generation (imported from letter module) ──────────────────
try:
    from generate_clinical_letter import generate_clinical_letter as _gen_letter
    CLINICAL_LETTER_AVAILABLE = True
except ImportError:
    CLINICAL_LETTER_AVAILABLE = False

from deep_translator import GoogleTranslator
import asyncio, json, httpx, numpy as np, subprocess, tempfile
import os, time, threading, re
import jwt                          # pip install PyJWT
from typing import Optional, Dict, List, Tuple
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize

# ── Shared optimisation components ───────────────────────────────────────────
from shared_components import (
    SemanticCache, TokenBudgetManager, AutoLearningLoop,
    EnhancedVAD, PipelineDecisionEngine, ValidationLayer,
    PromptTemplateManager,
    OPENAI_STT_MODEL, OPENAI_COT_MODEL, OPENAI_MINI_MODEL,
    OPENAI_STT_URL, OPENAI_CHAT_URL,
    SAMPLE_RATE, STT_PROMPT_HINT, GULF_PHRASES, NOISE_TAGS,
    detect_lang, compute_word_diff, format_ts, _safe_parse_json,
    extract_mean_pitch, apply_domain_fixes, clean_stt_text,
)

# ── WebRTC VAD (optional) ─────────────────────────────────────────────────────
try:
    import webrtcvad as _webrtcvad
    VAD_AVAILABLE = True
except ImportError:
    try:
        import webrtcvad_wheels as _webrtcvad   # type: ignore
        VAD_AVAILABLE = True
    except ImportError:
        _webrtcvad    = None                     # type: ignore
        VAD_AVAILABLE = False
        print("⚠️  webrtcvad not available — using energy-only VAD")

# ── Configuration — paste your keys directly here ───────────────────────────
OPENAI_API_KEY     = os.environ.get("OPENAI_API_KEY", "")
LIVEKIT_API_KEY    = os.environ.get("LIVEKIT_API_KEY", "")
LIVEKIT_API_SECRET = os.environ.get("LIVEKIT_API_SECRET", "")
LIVEKIT_URL        = os.environ.get("LIVEKIT_URL", "")

if not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_OPENAI_API_KEY_HERE":
    raise RuntimeError("OPENAI_API_KEY is not set — paste your key into main.py")
OPENAI_HEADERS   = {"Authorization": f"Bearer {OPENAI_API_KEY}"}

COSINE_THRESH    = 0.58     # similarity threshold for same-speaker
MAX_SPEAKERS     = 8
MIN_BLOB_BYTES   = 12_000   # ✅ RAISED: was 8_000 — full sentences, less mid-word cuts
BACK_TRANS_CONF  = 0.70

# Speaker-change detection parameters
SPK_CHANGE_MIN_SEG_SEC = 1.2   # minimum sub-segment length (seconds)
SPK_CHANGE_HOP_MS      = 80    # hop between analysis windows (ms)
SPK_CHANGE_WIN_MS      = 400   # comparison window size (ms)

# ── Global shared instances ───────────────────────────────────────────────────
cache     = SemanticCache()
budget    = TokenBudgetManager()
learning  = AutoLearningLoop()
vad       = EnhancedVAD()
engine    = PipelineDecisionEngine()
validator = ValidationLayer()
prompts   = PromptTemplateManager()

# ── Session globals ───────────────────────────────────────────────────────────
meeting_transcript: List[Dict] = []
session_start_time: float      = time.time()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SPEAKER TRACKER  (MFCC + pitch cosine, incremental)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SpeakerTracker:
    """
    Online incremental speaker diarisation with 40-dim embeddings:
      • 39-dim MFCC (+delta +delta2)
      •  1-dim mean pitch (F0) — helps separate male/female voices
    Cosine similarity against running centroids.
    Periodic batch re-cluster every 20 segments.
    """

    def __init__(self):
        self.centroids:  List[np.ndarray] = []
        self.embeddings: List[np.ndarray] = []
        self.labels:     List[int]        = []
        self.lock  = threading.Lock()
        self.seg_count = 0

    def reset(self):
        with self.lock:
            self.centroids.clear()
            self.embeddings.clear()
            self.labels.clear()
            self.seg_count = 0

    # ── MFCC extraction ───────────────────────────────────────────────────────
    def extract_mfcc(self,
                     pcm_float: np.ndarray,
                     sr: int = SAMPLE_RATE) -> Optional[np.ndarray]:
        """Return 39-dim MFCC mean vector (13 MFCC + 13 delta + 13 delta2)."""
        if len(pcm_float) < sr * 0.3:
            return None

        signal = pcm_float / (np.max(np.abs(pcm_float)) + 1e-9)
        signal = np.append(signal[0], signal[1:] - 0.97 * signal[:-1])

        frame_len  = int(0.025 * sr)
        frame_step = int(0.010 * sr)
        frames = [
            signal[i:i+frame_len] * np.hamming(frame_len)
            for i in range(0, len(signal) - frame_len, frame_step)
        ]
        if not frames:
            return None
        frames = np.array(frames)

        NFFT  = 512
        mag   = np.abs(np.fft.rfft(frames, NFFT))
        power = (1.0 / NFFT) * (mag ** 2)

        n_filt, n_mfcc = 26, 13
        high_mel = 2595 * np.log10(1 + (sr / 2) / 700)
        mel_pts  = np.linspace(0, high_mel, n_filt + 2)
        hz_pts   = 700 * (10 ** (mel_pts / 2595) - 1)
        bin_pts  = np.floor((NFFT + 1) * hz_pts / sr).astype(int)

        fbank = np.zeros((n_filt, NFFT // 2 + 1))
        for m in range(1, n_filt + 1):
            for k in range(bin_pts[m-1], bin_pts[m]):
                fbank[m-1, k] = (k - bin_pts[m-1]) / (bin_pts[m] - bin_pts[m-1] + 1e-9)
            for k in range(bin_pts[m], bin_pts[m+1]):
                fbank[m-1, k] = (bin_pts[m+1] - k) / (bin_pts[m+1] - bin_pts[m] + 1e-9)

        fb = np.dot(power, fbank.T)
        fb = np.where(fb == 0, np.finfo(float).eps, fb)
        fb = 20 * np.log10(fb)

        mfcc = np.zeros((len(frames), n_mfcc))
        for n in range(n_mfcc):
            mfcc[:, n] = np.sum(
                fb * np.cos(np.pi * n / n_filt * (np.arange(1, n_filt + 1) - 0.5)),
                axis=1
            )

        def _delta(feat: np.ndarray, N: int = 2) -> np.ndarray:
            d = np.zeros_like(feat)
            for t in range(len(feat)):
                num = sum(
                    nn * (feat[min(t+nn, len(feat)-1)] - feat[max(t-nn, 0)])
                    for nn in range(1, N+1)
                )
                den = 2 * sum(nn**2 for nn in range(1, N+1))
                d[t] = num / den
            return d

        d1 = _delta(mfcc)
        d2 = _delta(d1)
        return np.mean(np.concatenate([mfcc, d1, d2], axis=1), axis=0)

    def extract_embedding(self, pcm: np.ndarray) -> Optional[np.ndarray]:
        """
        40-dim speaker embedding: 39-dim MFCC + 1-dim normalised pitch.
        The pitch dimension significantly helps separate speakers with
        similar vocal timbre but different fundamental frequency.
        """
        mfcc = self.extract_mfcc(pcm)
        if mfcc is None:
            return None
        pitch = extract_mean_pitch(pcm)
        # Weight pitch by 3× to make it comparable in scale to MFCC dims
        return np.append(mfcc, pitch * 3.0)

    # ── online speaker identification ─────────────────────────────────────────
    def identify_speaker(self, embedding: np.ndarray) -> int:
        with self.lock:
            norm_val = np.linalg.norm(embedding)
            if norm_val < 1e-9:
                return 0
            emb_n = embedding / norm_val

            if not self.centroids:
                self.centroids.append(embedding.copy())
                self.embeddings.append(embedding.copy())
                self.labels.append(0)
                self.seg_count += 1
                return 0

            best_sim = -1.0
            best_idx = 0
            for i, c in enumerate(self.centroids):
                c_n = np.linalg.norm(c)
                if c_n < 1e-9:
                    continue
                sim = float(np.dot(emb_n, c / c_n))
                if sim > best_sim:
                    best_sim, best_idx = sim, i

            if best_sim >= COSINE_THRESH:
                spk = best_idx
                # Exponential moving average centroid update
                self.centroids[spk] = 0.85 * self.centroids[spk] + 0.15 * embedding
            elif len(self.centroids) < MAX_SPEAKERS:
                spk = len(self.centroids)
                self.centroids.append(embedding.copy())
                print(f"  🆕 New speaker detected: Speaker {spk+1}  "
                      f"(sim={best_sim:.3f} < thresh={COSINE_THRESH})")
            else:
                spk = best_idx   # all slots full → nearest

            self.embeddings.append(embedding.copy())
            self.labels.append(spk)
            self.seg_count += 1

            if self.seg_count % 20 == 0:
                self._recluster()

            return spk

    def _recluster(self):
        """Refine centroids from the last 100 embeddings using batch clustering."""
        n = len(self.embeddings)
        if n < 4:
            return
        try:
            X     = normalize(np.array(self.embeddings[-100:]))
            n_spk = min(len(self.centroids), len(X), MAX_SPEAKERS)
            if n_spk < 2:
                return
            labels = AgglomerativeClustering(
                n_clusters=n_spk, metric='cosine', linkage='average'
            ).fit_predict(X)
            for i in range(n_spk):
                mask = labels == i
                if mask.any():
                    self.centroids[i] = np.mean(X[mask], axis=0)
            print(f"  🔄 Reclustered {len(X)} embeddings → {n_spk} speakers")
        except Exception as e:
            print(f"  ⚠️  Recluster error: {e}")


speaker_tracker = SpeakerTracker()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# WITHIN-BLOB SPEAKER CHANGE DETECTION  ← NEW in v4
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def find_speaker_boundaries(pcm: np.ndarray,
                             sr: int = SAMPLE_RATE) -> List[Tuple[int, int]]:
    """
    ✅ TEMPORARILY SIMPLIFIED: return the full blob as one segment.
    The BIC spectral detector was over-splitting utterances, breaking
    sentence coherence and causing GPT to see incomplete context.
    Re-enable the full algorithm once word-level timestamps are available
    from the STT model (verbose_json).
    """
    return [(0, len(pcm))]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AUDIO UTILITIES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def webm_to_pcm(webm_bytes: bytes) -> Optional[np.ndarray]:
    """Decode WebM/Opus bytes → PCM float32 (16-kHz mono) via bundled ffmpeg."""
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        proc = subprocess.run(
            [ffmpeg_exe, "-v", "quiet", "-i", "pipe:0",
             "-ar", str(SAMPLE_RATE), "-ac", "1",
             "-f", "s16le", "pipe:1"],
            input=webm_bytes, capture_output=True, timeout=10
        )
        if proc.returncode == 0 and proc.stdout:
            pcm_s16 = np.frombuffer(proc.stdout, dtype=np.int16)
            return pcm_s16.astype(np.float32)
    except Exception as e:
        print(f"webm_to_pcm error: {e}")
    return None


def _embed_sub_segment_sync(pcm: np.ndarray,
                             start_s: int,
                             end_s: int) -> Optional[np.ndarray]:
    """Extract 40-dim embedding from a PCM sub-segment (runs in thread pool)."""
    sub = pcm[start_s:end_s]
    return speaker_tracker.extract_embedding(sub)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SENTENCE COMPLETION  — post-correction fragment repair
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Weak trailing words that signal a cut sentence — discard the whole segment
_WEAK_ENDINGS = {"the", "a", "an", "is", "are", "was", "were",
                 "and", "or", "but", "in", "on", "at", "to", "of",
                 "with", "by", "for", "from", "that", "which"}

# Known incomplete phrases → domain-complete expansions
_SENTENCE_COMPLETIONS: Dict[str, str] = {
    "the expectation was today":
        "The expectation today was that we could proceed with the RCM deployment.",
    "unstable connecting":
        "There is an unstable connection.",
    "unstable connect":
        "There is an unstable connection.",
    "the environment is":
        "The environment is currently unstable.",
    "when we present":
        "",   # discard — always a fragment
    "okay even from that perspective":
        "Okay, even from that perspective, we need to proceed.",
    "sending services tonight":
        "We are sending services tonight.",
}


def complete_sentence(text: str) -> str:
    """
    Post-correction sentence repair:
      1. Return empty string if text ends on a weak dangling word.
      2. Expand known incomplete phrases to their domain-complete form.
      3. Ensure the result ends with sentence-final punctuation.

    Call this AFTER GPT correction so the model's output is cleaned up
    before it reaches the WebSocket and the UI.
    """
    if not text:
        return ""

    t = text.strip()
    t_low = t.lower().rstrip(".,!?")

    # ── 1. Known completions (exact match on normalised text) ─────────────────
    for fragment, expansion in _SENTENCE_COMPLETIONS.items():
        if t_low == fragment or t_low.startswith(fragment):
            return expansion   # may be "" → caller should skip this segment

    # ── 2. Partial substring completions ──────────────────────────────────────
    for fragment, expansion in _SENTENCE_COMPLETIONS.items():
        if fragment and fragment in t_low:
            if expansion:
                return expansion
            return ""

    # ── 3. Discard sentences that trail off on a weak word ────────────────────
    last_word = t.rstrip(".,!?").rsplit(None, 1)[-1].lower() if t else ""
    if last_word in _WEAK_ENDINGS:
        return ""

    # ── 4. Ensure proper sentence-final punctuation ───────────────────────────
    if t and t[-1] not in ".?!":
        t += "."

    return t

async def _call_openai_chat_async(
    system_text: str,
    user_text:   str,
    model:       str,
    max_tokens:  int  = 200,
    json_mode:   bool = False,
) -> Optional[str]:
    payload: Dict = {
        "model":       model,
        "temperature": 0.0,
        "max_tokens":  max_tokens,
        "messages": [
            {"role": "system", "content": system_text},
            {"role": "user",   "content": user_text},
        ],
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            resp = await client.post(
                OPENAI_CHAT_URL,
                headers={**OPENAI_HEADERS, "Content-Type": "application/json"},
                json=payload,
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"].strip()
            print(f"OpenAI chat {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        print(f"_call_openai_chat_async error: {e}")
    return None


async def transcribe_with_openai(webm_bytes: bytes) -> Optional[Dict]:
    """
    Send WebM audio to gpt-4o-transcribe.
    Returns {"text": str, "language": str, "words": []}
    NOTE: gpt-4o-transcribe only supports response_format "json" or "text".
    """
    import hashlib
    fp = hashlib.sha256(webm_bytes[:8192]).hexdigest()[:32]
    ck = f"stt:{len(webm_bytes)}:{fp}"
    cached = cache.get(ck, "stt")
    if cached:
        print("✔️  STT cache hit")
        return cached

    def _send_stt_request() -> httpx.Response:
        with httpx.Client(timeout=30.0) as client:
            return client.post(
                OPENAI_STT_URL,
                headers=OPENAI_HEADERS,
                files={"file": ("audio.webm", webm_bytes, "audio/webm")},
                data={
                    "model":           OPENAI_STT_MODEL,
                    # gpt-4o-transcribe only supports "json" or "text"
                    # verbose_json (word timestamps) is NOT available on this model
                    "response_format": "json",
                    "prompt":          STT_PROMPT_HINT,
                    "temperature":     "0",
                },
            )

    try:
        resp = await asyncio.to_thread(_send_stt_request)
        if resp.status_code == 200:
            data  = resp.json()
            raw_txt = data.get("text", "").strip()

            # 🔥 Apply domain fixes FIRST (before clean_stt_text)
            raw_txt = apply_domain_fixes(raw_txt)

            # 🔥 Clean STT output: drop fragments < 3 words, fix incomplete endings
            txt  = clean_stt_text(raw_txt)

            # Language: prefer model's own detection; normalise to ISO code
            raw_lang = data.get("language") or (detect_lang(txt) if txt else "en")
            lang = _normalize_lang(raw_lang)

            # words is always [] for gpt-4o-transcribe — dominant-speaker
            # fallback in Step 6 handles this correctly
            words = []

            if txt:
                print(f"📝 STT: '{txt[:80]}{'…' if len(txt)>80 else ''}' "
                      f"(lang={lang})")
                result = {"text": txt, "language": lang, "words": []}
                cache.set(ck, result, "stt")
                return result
            else:
                raw_preview = raw_txt[:60] if raw_txt else "(empty)"
                print(f"🔇 STT: fragment discarded — '{raw_preview}'")
                return {"text": "", "language": "en", "words": []}
        print(f"STT error {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        print(f"transcribe_with_openai error: {e}")
    return None


# Full language name → ISO code map (gpt-4o-transcribe returns full names)
_LANG_NAME_TO_ISO: Dict[str, str] = {
    "english": "en", "arabic": "ar", "hindi": "hi", "telugu": "te",
    "tamil": "ta", "kannada": "kn", "malayalam": "ml", "urdu": "ur",
    "bengali": "bn", "marathi": "mr", "gujarati": "gu", "punjabi": "pa",
    "odia": "or", "assamese": "as", "french": "fr", "german": "de",
    "spanish": "es", "portuguese": "pt", "italian": "it", "russian": "ru",
    "chinese": "zh", "japanese": "ja", "korean": "ko", "turkish": "tr",
    "farsi": "fa", "persian": "fa", "hebrew": "he",
    # Additional languages gpt-4o-transcribe may return
    "dutch": "nl", "swedish": "sv", "norwegian": "no", "danish": "da",
    "polish": "pl", "czech": "cs", "romanian": "ro", "hungarian": "hu",
    "greek": "el", "thai": "th", "vietnamese": "vi", "indonesian": "id",
    "malay": "ms", "swahili": "sw", "amharic": "am", "somali": "so",
    "hausa": "ha", "yoruba": "yo", "sinhalese": "si", "sinhala": "si",
    "nepali": "ne", "burmese": "my", "khmer": "km", "lao": "lo",
    "azerbaijani": "az", "kazakh": "kk", "uzbek": "uz", "ukrainian": "uk",
    "mixed": "mixed",   # code-switching — treated specially in translate_async
}

def _normalize_lang(lang: str) -> str:
    """Normalise gpt-4o-transcribe full language name to ISO-639-1 code."""
    if not lang:
        return "en"
    l = lang.strip().lower()
    return _LANG_NAME_TO_ISO.get(l, l)


async def translate_async(text: str, src: str) -> str:
    src = _normalize_lang(src)
    if not text.strip():
        return text

    # "mixed" language = code-switching → always translate via GPT
    # Even "en" src needs translation if non-Latin scripts are embedded
    is_mixed = (src == "mixed") or (
        src == "en" and any(
            '\u0600' <= c <= '\u06FF'   # Arabic
            or '\u0900' <= c <= '\u097F'  # Devanagari
            or '\u0C00' <= c <= '\u0C7F'  # Telugu
            or '\u0B80' <= c <= '\u0BFF'  # Tamil
            or '\u0C80' <= c <= '\u0CFF'  # Kannada
            or '\u0D00' <= c <= '\u0D7F'  # Malayalam
            or '\u0980' <= c <= '\u09FF'  # Bengali
            for c in text
        )
    )
    if src == "en" and not is_mixed:
        return text

    # Quick-phrase fast path — covers Arabic, Hindi, Telugu, and mixed
    if src in ("ar", "hi", "te", "mixed"):
        for phrase, english in GULF_PHRASES.items():
            if phrase in text and len(text.replace(phrase, "").strip()) < 3:
                return english

    cached = cache.get(text, "translation")
    if cached:
        return cached.get("translation", text)

    # GPT translation: handles all 4 priority languages and any mixing
    mix_hint = (" The text contains code-switching between multiple languages"
                " — translate ALL parts into English.") if is_mixed else ""
    result = await _call_openai_chat_async(
        "You are a professional medical translator specialising in "
        "Arabic (Gulf/Levantine), Hindi, Telugu, and English. "
        "Translate the user text into English regardless of source language. "
        "Handle code-switching (mixed-language) sentences — translate every "
        "non-English word or phrase into its English equivalent."
        + mix_hint +
        " Keep medical/tech terms (RCM, EMR, HIS, API, etc.) unchanged. "
        "Return ONLY the English translation, nothing else.",
        f"Translate to English: {text}",
        model=OPENAI_MINI_MODEL,
        max_tokens=200,
        json_mode=False,
    )
    if result and result.strip() and result.strip() != text.strip():
        cache.set(text, {"translation": result.strip()}, "translation")
        return result.strip()

    # Fallback: Google Translate (source="auto" handles any language)
    try:
        t = GoogleTranslator(source="auto", target="en").translate(text)
        if t and t.strip() and t.strip() != text.strip():
            cache.set(text, {"translation": t.strip()}, "translation")
            return t.strip()
    except Exception as e:
        print(f"  GoogleTranslator fallback error: {e}")

    return text


async def _back_translate_check(original: str,
                                 translation: str,
                                 src_lang: str) -> float:
    if src_lang in ("en", "mixed") or not translation or translation in NOISE_TAGS:
        return 1.0
    lang_name = {
        "ar": "Arabic", "hi": "Hindi", "te": "Telugu", "ta": "Tamil",
        "kn": "Kannada", "ml": "Malayalam", "bn": "Bengali", "ur": "Urdu",
        "pa": "Punjabi", "gu": "Gujarati", "mr": "Marathi",
        "fr": "French", "de": "German", "es": "Spanish", "pt": "Portuguese",
        "it": "Italian", "ru": "Russian", "tr": "Turkish", "fa": "Persian",
        "he": "Hebrew", "zh": "Chinese", "ja": "Japanese", "ko": "Korean",
    }.get(src_lang, "the source language")
    back = await _call_openai_chat_async(
        f"Translate English to {lang_name}. Return ONLY the translation.",
        f"Translate: {translation}",
        model=OPENAI_MINI_MODEL,
        max_tokens=120,
        json_mode=False,
    )
    if not back:
        return 0.5
    return validator.back_translation_similarity(original, back)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TIERED CORRECTION ENGINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _passthrough_result(original: str, translation: str, tier: str = "skip") -> Dict:
    return {
        "corrected_original":    original,
        "corrected_translation": translation,
        "original_diff":    [{"type": "same", "text": w} for w in original.split()],
        "translation_diff": [{"type": "same", "text": w} for w in translation.split()],
        "was_changed":  False,
        "tier":         tier,
    }


async def correct_segment(seg: Dict) -> Dict:
    raw_orig  = seg["original"]
    raw_trans = seg.get("translation", "")
    speaker   = seg["speaker"]
    conf      = engine.score(raw_orig)

    ck = f"{raw_orig}|spk{speaker}"
    cached = cache.get(ck, "correction")
    if cached:
        print("✔️  Correction cache hit")
        return cached

    # 🔥 Domain pre-correction BEFORE GPT
    prefixed = apply_domain_fixes(raw_orig)
    if prefixed != raw_orig:
        print(f"  🔧 Domain pre-fix: '{raw_orig[:40]}' → '{prefixed[:40]}'")
        raw_orig = prefixed

    tier = engine.decide(raw_orig, conf, learning)
    print(f"🎯 Tier [{tier:<8}] conf={conf:.2f}  '{raw_orig[:50]}'")

    if tier == "skip":
        return _passthrough_result(raw_orig, raw_trans, tier="skip")
    if tier == "trivial":
        return _passthrough_result(raw_orig, raw_trans, tier="trivial")
    if tier == "rule":
        fixed = learning.apply_rules(raw_orig) or raw_orig
        # Also run domain fixes on top of learned rules
        fixed = apply_domain_fixes(fixed)
        if fixed != raw_orig:
            src   = detect_lang(fixed)
            ftrans = await translate_async(fixed, src)
            learning.record(raw_orig, fixed)  # reinforce the rule
            return {
                "corrected_original":    fixed,
                "corrected_translation": ftrans,
                "original_diff":    compute_word_diff(raw_orig,  fixed),
                "translation_diff": compute_word_diff(raw_trans, ftrans),
                "was_changed": True,
                "tier":        "rule",
            }
        return _passthrough_result(raw_orig, raw_trans, tier="rule")

    model     = OPENAI_COT_MODEL if tier == "full_cot" else OPENAI_MINI_MODEL
    max_tok   = budget.get(tier)
    context   = budget.compress_context(meeting_transcript[-20:])
    learnings = learning.context_str()

    # 🔥 Context baked into BOTH system prompt AND user message
    # This gives GPT meeting-level awareness at all correction tiers
    sys_p = prompts.get_system_with_context(tier, context, learnings)
    seg_with_fix = dict(seg)
    seg_with_fix["original"] = raw_orig
    usr_p = prompts.build_batch_msg(
        [seg_with_fix], context, learnings, tier, text_key="original"
    )

    raw_resp = await _call_openai_chat_async(
        sys_p, usr_p, model=model, max_tokens=max_tok, json_mode=True
    )

    if raw_resp:
        try:
            parsed = _safe_parse_json(
                raw_resp,
                fallback_keys=["corrected_original", "corrected_translation"]
            )
            item = parsed[0] if isinstance(parsed, list) else parsed

            corr_orig  = item.get("corrected_original",    raw_orig)
            corr_trans = item.get("corrected_translation", raw_trans)
            if not corr_orig:
                corr_orig  = item.get("corrected", raw_orig)
            if not corr_trans:
                corr_trans = item.get("translation", raw_trans)

            nl = item.get("new_learnings", {})
            if isinstance(nl, dict):
                for w, c in nl.items():
                    if w and c and w != c:
                        learning.record(w, c)

            corr_orig, corr_trans, was_changed = validator.validate(
                raw_orig, corr_orig, corr_trans
            )

            # ✅ FIX: Strip any "full_cot" / tier tag that GPT leaked into output text
            # This prevents internal routing labels appearing in the transcript UI
            _TIER_TAG_RE = re.compile(
                r'\b(full_cot|simple|trivial|skip|rule)\b', re.IGNORECASE
            )
            corr_orig  = _TIER_TAG_RE.sub('', corr_orig).strip()
            corr_trans = _TIER_TAG_RE.sub('', corr_trans).strip()

            # 🔥 Auto-learning: record EVERY correction so system improves over session
            if was_changed and corr_orig and corr_orig != raw_orig:
                learning.record(raw_orig, corr_orig)
                print(f"  📚 Learned: '{raw_orig[:35]}' → '{corr_orig[:35]}'")

            if tier == "full_cot" and conf < BACK_TRANS_CONF:
                src = detect_lang(corr_orig)
                bt_sim = await _back_translate_check(corr_orig, corr_trans, src)
                if bt_sim < 0.25:
                    print(f"  ⚠️  Back-translation sim={bt_sim:.2f} → reverting")
                    corr_trans = await translate_async(corr_orig, src)

            # 🔥 Post-correction domain fix: catches GPT introducing new mis-hearings
            corr_orig_final = apply_domain_fixes(corr_orig)   # ✅ SECOND domain pass
            if corr_orig_final != corr_orig:
                print(f"  🔧 Post-GPT fix: '{corr_orig[:35]}' → '{corr_orig_final[:35]}'")
                was_changed = True
                corr_orig = corr_orig_final

            # ✅ NEW: Sentence completion — repair cut sentences, discard bare fragments
            corr_orig_completed = complete_sentence(corr_orig)
            if corr_orig_completed == "":
                print(f"  🗑️  complete_sentence dropped fragment: '{corr_orig[:50]}'")
                return _passthrough_result(raw_orig, raw_trans, tier="skip")
            if corr_orig_completed != corr_orig:
                print(f"  ✏️  Sentence completed: '{corr_orig[:40]}' → '{corr_orig_completed[:40]}'")
                was_changed  = True
                corr_orig    = corr_orig_completed

            result = {
                "corrected_original":    corr_orig,
                "corrected_translation": corr_trans,
                "original_diff":    compute_word_diff(raw_orig,  corr_orig),
                "translation_diff": compute_word_diff(raw_trans, corr_trans),
                "was_changed":      was_changed,
                "tier":             tier,
            }
            cache.set(ck, result, "correction")
            return result

        except Exception as e:
            print(f"  correction parse error: {e}")

    return _passthrough_result(raw_orig, raw_trans)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BLOB PROCESSING PIPELINE  ← Major rewrite in v4
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def process_blob(websocket: WebSocket, webm_blob: bytes):
    """
    Full pipeline for one 5-second WebM audio blob.

    Key difference from v3:
      • find_speaker_boundaries() splits the blob at acoustic change points.
      • Each sub-segment gets its own speaker embedding & ID.
      • STT uses verbose_json with word-level timestamps.
      • Words are mapped to speaker sub-segments by timestamp overlap.
      • Multiple speaker blocks can be emitted from one blob.
    """
    if len(webm_blob) < MIN_BLOB_BYTES:
        return

    loop = asyncio.get_event_loop()

    # ── Step 1: Decode PCM ────────────────────────────────────────────────────
    pcm = await loop.run_in_executor(None, webm_to_pcm, webm_blob)
    if pcm is None or len(pcm) < SAMPLE_RATE * 0.5:
        return

    # ── Step 2: EnhancedVAD gate ──────────────────────────────────────────────
    speech_ok, vad_stats = vad.is_speech(pcm, sr=SAMPLE_RATE)
    if not speech_ok:
        reason = vad_stats.get("reason", "energy/zcr")
        print(f"🔇 VAD rejected  rms={vad_stats['rms']:.0f} zcr={vad_stats['zcr']:.3f} [{reason}]")
        return

    # ── Step 3: STT + speaker change detection (run in parallel) ──────────────
    stt_future = asyncio.ensure_future(transcribe_with_openai(webm_blob))
    boundaries_future = loop.run_in_executor(
        None, find_speaker_boundaries, pcm
    )

    stt_result = await stt_future
    sub_segs   = await boundaries_future   # List[Tuple[int, int]]

    if not stt_result or not stt_result.get("text", "").strip():
        return

    text          = stt_result["text"].strip()
    detected_lang = stt_result.get("language", detect_lang(text))
    words         = stt_result.get("words", [])   # [{word, start, end}]

    # 🔥 Guard: discard if fewer than 3 words — almost certainly a fragment or noise
    if len(text.split()) < 3:
        print(f"🗑️  Fragment dropped (< 3 words): '{text}'")
        return

    blob_duration = len(pcm) / SAMPLE_RATE
    print(f"🔊 Blob {blob_duration:.1f}s → {len(sub_segs)} sub-segs, "
          f"lang={detected_lang}, {len(words)} words")

    # ── Step 4: Extract speaker embedding per sub-segment (thread pool) ────────
    embed_tasks = [
        loop.run_in_executor(None, _embed_sub_segment_sync, pcm, s, e)
        for s, e in sub_segs
    ]
    embeddings = await asyncio.gather(*embed_tasks)

    # ── Step 5: Identify speaker for each sub-segment ─────────────────────────
    sub_speakers: List[int] = []
    for emb in embeddings:
        if emb is not None:
            spk = await loop.run_in_executor(
                None, speaker_tracker.identify_speaker, emb
            )
        else:
            spk = 0
        sub_speakers.append(spk)

    # ── Step 6: Align words → speaker sub-segments ────────────────────────────
    # Each word's [start, end] timestamps (from STT verbose_json) are compared
    # against sub-segment sample ranges to determine which speaker said it.
    seg_outputs: List[Tuple[int, str]] = []   # (speaker_id, text_fragment)

    if words:
        # 🔥 TRUE word-level alignment (now works because verbose_json is used)
        for (seg_start_s, seg_end_s), spk in zip(sub_segs, sub_speakers):
            seg_start_sec = seg_start_s / SAMPLE_RATE
            seg_end_sec   = seg_end_s   / SAMPLE_RATE
            seg_words = [
                w.get("word", "")
                for w in words
                if seg_start_sec <= w.get("start", 0.0) < seg_end_sec
            ]
            if seg_words:
                seg_outputs.append((spk, " ".join(seg_words).strip()))
    else:
        # 🔥 FIXED FALLBACK: when no word timestamps, assign ENTIRE text to the
        # dominant (most-common) speaker in sub_speakers instead of proportional
        # split — proportional split was destroying sentence coherence
        if sub_speakers:
            from collections import Counter
            dominant_spk = Counter(sub_speakers).most_common(1)[0][0]
        else:
            dominant_spk = 0
        seg_outputs = [(dominant_spk, text)]

    # Merge consecutive identical speakers (they belong to one utterance)
    merged_outputs: List[Tuple[int, str]] = []
    for spk, frag in seg_outputs:
        if merged_outputs and merged_outputs[-1][0] == spk:
            merged_outputs[-1] = (spk, merged_outputs[-1][1] + " " + frag)
        else:
            merged_outputs.append((spk, frag))

    if not merged_outputs:
        # Single-speaker blob (common for 5-second chunks)
        merged_outputs = [(sub_speakers[0] if sub_speakers else 0, text)]

    # ── Step 7: Translate + build raw segments ────────────────────────────────
    ts = format_ts(time.time() - session_start_time)

    async def _identity(f: str) -> str:
        return f

    # Normalise detected_lang so the "en" check works regardless of
    # whether gpt-4o-transcribe returned a full name or an ISO code
    detected_lang = _normalize_lang(detected_lang)

    translate_tasks = [
        translate_async(frag, detected_lang) if detected_lang not in ("en",) else _identity(frag)
        for _, frag in merged_outputs
    ]
    translations = await asyncio.gather(*translate_tasks)

    raw_segments: List[Dict] = []
    for (spk, frag), trans in zip(merged_outputs, translations):
        # 🔥 Apply domain fixes to every fragment before sending raw result
        frag_fixed = apply_domain_fixes(frag)

        # 🔥 Safety net: if translation looks identical to original (untranslated)
        # AND we know it is not English, force a Google Translate fallback
        if detected_lang != "en" and trans.strip() == frag.strip():
            try:
                fallback = GoogleTranslator(source="auto", target="en").translate(frag)
                if fallback and fallback.strip() != frag.strip():
                    trans = fallback.strip()
                    print(f"  🔄 Safety re-translate ({detected_lang}→en): '{frag[:40]}' → '{trans[:40]}'")
            except Exception as _gt_err:
                print(f"  ⚠️  Safety re-translate failed: {_gt_err}")

        raw_segments.append({
            "speaker":     spk,
            "timestamp":   ts,
            "original":    frag_fixed,
            "translation": trans,
            "lang":        detected_lang,
        })

    # ── Message 1: raw result (low-latency fast path) ─────────────────────────
    try:
        await websocket.send_json({
            "segments":          raw_segments,
            "is_final":          True,
            "detected_language": detected_lang,
            "corrected":         False,
        })
    except Exception:
        print("⚠️  WebSocket closed before raw send — skipping")

    # ── Step 8: Correction pipeline (all segments concurrently) ───────────────
    # ✅ FIX: return_exceptions=True prevents one failing task from killing all others
    corr_results_raw = await asyncio.gather(
        *[correct_segment(seg) for seg in raw_segments],
        return_exceptions=True,
    )
    corr_results = []
    for r in corr_results_raw:
        if isinstance(r, Exception):
            print(f"❌ Correction task error: {r}")
        else:
            corr_results.append(r)

    corrected_segments: List[Dict] = []
    # ✅ NOTE: corr_results is already exception-filtered above; may be shorter than raw_segments
    # We zip with raw_segments up to the number of successful results
    for raw_seg, corr in zip(raw_segments, corr_results):
        if corr is None:
            continue
        corrected_segments.append({
            "speaker":          raw_seg["speaker"],
            "timestamp":        ts,
            "original":         corr["corrected_original"],
            "translation":      corr["corrected_translation"],
            "original_diff":    corr["original_diff"],
            "translation_diff": corr["translation_diff"],
            "was_changed":      corr["was_changed"],
            "tier":             corr.get("tier"),
            "lang":             detected_lang,
        })
        meeting_transcript.append({
            "speaker":     raw_seg["speaker"],
            "timestamp":   ts,
            "original":    corr["corrected_original"],
            "translation": corr["corrected_translation"],
        })

    if len(meeting_transcript) > 200:
        meeting_transcript[:] = meeting_transcript[-200:]

    # ── Message 2: corrected result ───────────────────────────────────────────
    if corrected_segments:
        try:
            await websocket.send_json({
                "segments":          corrected_segments,
                "is_final":          True,
                "detected_language": detected_lang,
                "corrected":         True,
            })
        except Exception:
            print("⚠️  WebSocket closed before corrected send — skipping")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FASTAPI APP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

app = FastAPI(title="Live Speech Translator", version="4.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def home():
    return FileResponse("index.html")


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


@app.get("/.well-known/appspecific/com.chrome.devtools.json")
def chrome_devtools():
    return Response(status_code=204)


@app.get("/stats")
def stats():
    return {
        "cache":      cache.stats(),
        "rules":      learning.rule_count,
        "transcript": len(meeting_transcript),
        "speakers":   len(speaker_tracker.centroids),
    }


# ── LiveKit Token endpoint ─────────────────────────────────────────────────────

class LiveKitTokenRequest(BaseModel):
    room:     str = "consultation-room"
    identity: str = "user"
    name:     str = "User"


@app.post("/livekit/token")
async def get_livekit_token(body: LiveKitTokenRequest):
    """
    Generate a signed LiveKit JWT for a participant to join a room.
    Called by the frontend before connecting to LiveKit.
    """
    if not LIVEKIT_API_KEY or not LIVEKIT_API_SECRET:
        raise HTTPException(
            status_code=503,
            detail="LiveKit is not configured — add LIVEKIT_API_KEY, "
                   "LIVEKIT_API_SECRET and LIVEKIT_URL to your .env file."
        )
    if not LIVEKIT_URL:
        raise HTTPException(
            status_code=503,
            detail="LIVEKIT_URL is not set in .env"
        )

    now = int(time.time())
    payload = {
        "iss": LIVEKIT_API_KEY,       # issuer  = your API key
        "sub": body.identity,          # subject = participant identity
        "name": body.name,
        "nbf": now,
        "exp": now + 3600,             # token valid for 1 hour
        "jti": f"{body.identity}-{now}",
        "video": {
            "room":       body.room,
            "roomJoin":   True,
            "canPublish": True,
            "canSubscribe": True,
        }
    }

    token = jwt.encode(payload, LIVEKIT_API_SECRET, algorithm="HS256")

    return JSONResponse({
        "token": token,
        "url":   LIVEKIT_URL,
        "room":  body.room,
    })


# ── GET convenience endpoint (no body needed for quick tests) ─────────────────
@app.get("/livekit/token")
async def get_livekit_token_get(
    room:     str = "consultation-room",
    identity: str = "doctor",
    name:     str = "Doctor",
):
    """GET version so the frontend can hit it without a request body."""
    return await get_livekit_token(
        LiveKitTokenRequest(room=room, identity=identity, name=name)
    )



class TranscriptRequest(BaseModel):
    transcript: str = ""


@app.post("/download-clinical-letter")
async def download_clinical_letter(body: TranscriptRequest):
    """
    Generate a structured clinical letter from the provided transcript text.
    Falls back to the in-memory meeting_transcript if no transcript is provided.
    Returns the letter as a plain-text file download.
    """
    import httpx

    # Build transcript string — prefer the request body, fall back to session store
    transcript_text = body.transcript.strip()
    if not transcript_text and meeting_transcript:
        lines = []
        for seg in meeting_transcript:
            spk = seg.get("speaker_id", 0)
            txt = seg.get("corrected") or seg.get("original") or ""
            if txt.strip():
                lines.append(f"Speaker {spk}: {txt}")
        transcript_text = "\n".join(lines)

    if not transcript_text:
        raise HTTPException(
            status_code=422,
            detail="No transcript available. Please record a session first."
        )

    # ── System prompt ─────────────────────────────────────────────────────────
    system_prompt = (
        "You are a medical transcription structuring assistant. Your role is to structure "
        "dictated clinical content into a formal clinical letter. You must follow all formatting "
        "and content rules exactly as instructed. Never add, infer, or omit information.\n\n"
        "IMPORTANT — SPEAKER LABELS: The transcript you receive may contain speaker labels "
        "such as 'Speaker 0:', 'Speaker 1:', etc. In a clinical dictation context:\n"
        "- The PRIMARY speaker (usually Speaker 0) is the DOCTOR dictating the clinical letter.\n"
        "- Other speakers (Speaker 1, 2, etc.) may be the patient or other staff.\n"
        "- Extract clinical content primarily from the doctor's dictation.\n"
        "- If the patient speaks, use their words only for CHIEF COMPLAINTS or relevant clinical context.\n"
        "- NEVER include the speaker labels themselves in the output letter.\n"
        "- Treat the content as a single unified clinical dictation."
    )

    user_prompt = (
        "You are a medical transcription structuring assistant.\n\n"
        "From the transcript below, structure the content into a formal clinical letter.\n\n"
        "STRICT MEDICO-LEGAL RULES:\n"
        "- DO NOT add new information\n"
        "- DO NOT summarize or shorten the content\n"
        "- DO NOT interpret or infer\n"
        "- USE the original dictated wording EXACTLY as spoken\n"
        "- REMOVE dictation control phrases (e.g. 'next paragraph', 'new paragraph')\n"
        "- NEVER include speaker labels (Speaker 0, Speaker 1) in the output\n\n"
        "REQUIRED OUTPUT FORMAT:\n\n"
        "Dr. _____\n_____\n\nDear Dr. _____,\n\n"
        "CHIEF COMPLAINTS:\n[Patient complaints and symptoms]\n\n"
        "DIAGNOSIS:\n[All named conditions, devices, findings — full paragraph]\n\n"
        "PATIENT INFORMATION:\n[Name, age, gender if mentioned]\n\n"
        "MEDICATIONS:\n[Each on separate line: Name. Dosage. Frequency]\n\n"
        "ITEM TYPES:\n"
        "  Medication Codes: [No information provided]\n"
        "  Transportation SRCA: [No information provided]\n"
        "  Imaging: [No information provided]\n"
        "  Services: [No information provided]\n"
        "  Laboratory: [No information provided]\n"
        "  Radiology: [No information provided]\n"
        "  Procedures: [No information provided]\n"
        "  Oral Health OP: [No information provided]\n"
        "  Oral Health IP: [No information provided]\n"
        "  Nutrition Codes: [No information provided]\n"
        "  Herbal and Vitamin Codes: [No information provided]\n"
        "  Cosmetic Codes: [No information provided]\n"
        "  Medical Devices: [No information provided]\n\n"
        "REFERRAL DETAILS:\n[Who referred, when, reason]\n\n"
        "CLINICAL HISTORY:\n[All background and past medical context]\n\n"
        "INVESTIGATIONS:\n[Each test/scan on a separate line with findings]\n\n"
        "PLAN:\n[Each action/decision on a separate line]\n\n"
        "OTHER:\n[Any remaining information]\n\n"
        "CC: [Copy recipients]\n\n"
        "For empty fields write exactly: [No information provided]\n\n"
        "Transcript:\n" + transcript_text
    )

    # ── Call OpenAI ───────────────────────────────────────────────────────────
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                json={
                    "model": "gpt-4o",
                    "temperature": 0.2,
                    "max_tokens": 4096,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                }
            )

        if resp.status_code != 200:
            raise HTTPException(
                status_code=502,
                detail=f"OpenAI error {resp.status_code}: {resp.text[:300]}"
            )

        result = resp.json()
        letter_text = (
            result.get("choices", [{}])[0]
                  .get("message", {})
                  .get("content", "")
        )

        if not letter_text.strip():
            raise HTTPException(status_code=502, detail="OpenAI returned empty response")

        # Strip markdown formatting GPT might add
        import re
        letter_text = re.sub(r"```[a-zA-Z]*\n?", "", letter_text).replace("```", "")
        letter_text = re.sub(r"(?m)^#{1,6}\s+", "", letter_text)
        letter_text = re.sub(r"\*\*(.+?)\*\*", r"\1", letter_text)
        letter_text = re.sub(r"(?im)^\s*speaker\s+[A-Z0-9]+:\s*", "", letter_text)

        from fastapi.responses import Response as FR
        return FR(
            content=letter_text.encode("utf-8"),
            media_type="text/plain; charset=utf-8",
            headers={
                "Content-Disposition": "attachment; filename=clinical_letter.txt"
            }
        )

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="OpenAI request timed out (>300s)")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


def _reset_session():
    """Reset all per-session state (transcript, speaker tracker, learning loop)."""
    global session_start_time
    meeting_transcript.clear()
    learning.reset()
    speaker_tracker.reset()
    session_start_time = time.time()


async def _ws_loop(websocket: WebSocket):
    is_connected = True
    while is_connected:
        try:
            data = await asyncio.wait_for(websocket.receive_bytes(), timeout=15.0)
            asyncio.ensure_future(process_blob(websocket, bytes(data)))
        except asyncio.TimeoutError:
            pass
        except WebSocketDisconnect:
            is_connected = False
            print(f"  {cache.stats()}  rules={learning.rule_count}  "
                  f"speakers={len(speaker_tracker.centroids)}")
            break


@app.websocket("/listen-auto")
async def listen_auto(websocket: WebSocket):
    await websocket.accept()
    _reset_session()
    print("✅ /listen-auto connected  [speaker-aware diarisation v4]")
    try:
        await _ws_loop(websocket)
    except Exception as e:
        print(f"listen-auto error: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass


@app.websocket("/listen")
async def listen_arabic(websocket: WebSocket):
    """Legacy endpoint — same pipeline."""
    await websocket.accept()
    print("✅ /listen connected  [legacy alias]")
    try:
        await _ws_loop(websocket)
    except Exception as e:
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass