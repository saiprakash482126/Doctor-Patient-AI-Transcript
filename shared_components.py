"""
shared_components.py — Shared Optimization Layer for Live Speech Translator
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Hybrid Tiered Pipeline — single source of truth for both main.py and
Transcribe_video.py.  No duplicate class definitions anywhere.

Tiers (low→high cost):
  skip      → silently discard (noise / known-bad tags)
  trivial   → identity pass   (no API: "ok", "yes", "طيب" …)
  rule      → instant local   (promoted corrections, no API)
  simple    → gpt-4.1-mini    (fast, cheap, 150-tok budget)
  full_cot  → gpt-4.1         (chain-of-thought, 1024-tok budget)

Components:
  [1] SemanticCache          LRU + fuzzy-match TTL cache
  [2] TokenBudgetManager     per-tier max_tokens + context compression
  [3] AutoLearningLoop       repeated-correction → promoted rule
  [4] EnhancedVAD            RMS energy + zero-crossing-rate gate
  [5] PipelineDecisionEngine confidence gate → tier routing
  [6] ValidationLayer        hallucination regex + semantic cross-check
                             + back-translation similarity helper
  [7] PromptTemplateManager  versioned, grounded, CoT-freedom-reduced prompts
  [8] Utilities              detect_lang, compute_word_diff,
                             format_ts, _safe_parse_json,
                             extract_mean_pitch (new)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations
import re, json, time, difflib
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── API endpoint constants ────────────────────────────────────────────────────
# ✅ FIXED: All three models now match what the UI advertises
OPENAI_STT_MODEL  = "gpt-4o-transcribe"   # WAS "whisper-1" — massive accuracy boost
OPENAI_COT_MODEL  = "gpt-4.1"             # WAS "gpt-4-turbo"
OPENAI_MINI_MODEL = "gpt-4.1-mini"        # WAS "gpt-4-turbo"
OPENAI_STT_URL    = "https://api.openai.com/v1/audio/transcriptions"
OPENAI_CHAT_URL   = "https://api.openai.com/v1/chat/completions"

SAMPLE_RATE = 16000

# 🔥 UPGRADED STT_PROMPT_HINT — domain-anchored, fully multilingual transcription
STT_PROMPT_HINT = (
    "High-stakes healthcare IT meeting transcription. "
    "PRIMARY LANGUAGES: Arabic (Gulf/Levantine), Hindi, Telugu, English. "
    "Speakers may use any of these languages or MIX them mid-sentence "
    "(e.g. Hindi+English 'Hinglish', Telugu+English, Arabic+English, "
    "or all four in a single conversation). "
    "STRICT RULES: "
    "1. Transcribe EXACT words spoken in whatever language(s) are used — do NOT translate. "
    "2. Preserve code-switching: if a speaker switches language mid-sentence, "
    "transcribe each part in the language it was spoken. "
    "3. For Hindi/Telugu romanised speech (Latin script), transcribe as spoken. "
    "4. NEVER output incomplete sentence fragments (e.g. 'when we present', 'the environment is'). "
    "5. If a sentence is cut mid-way, infer and COMPLETE it logically using domain context. "
    "6. NEVER output single isolated words — always full meaningful utterances. "
    "7. Prefer TECHNICAL TERMS over similar-sounding common words. "
    "DOMAIN VOCABULARY (preserve exactly): "
    "Nafis, Waseel, Clinicy, RCM, EMR, EHR, HIS, FHIR, HL7, IRP, DICOM, ERP, API, "
    "sandbox, staging, production, deployment, resubmission, eligibility, "
    "insurance claim, revenue cycle, radiology, laboratory, pharmacy, "
    "ICD-10, CPT, billing, authorization, UAT, QA, backend, frontend, database. "
    "PHONETIC CORRECTIONS: "
    "'radios'→radiology | 'radio'→radiology | 'efficient'→environment | "
    "'administrate'→process | 'respond'→resubmission | "
    "'insurance cycle'→revenue cycle | 'radius'→radiology. "
    "Multiple speakers may speak in sequence — transcribe ALL of them. "
    "If audio is unclear → attempt best domain guess, NEVER hallucinate or repeat words."
)
# Quick-translation phrase dictionary — exact matches bypass GPT (fast path)
# Covers Arabic (Gulf/Levantine), Hindi (Devanagari), Telugu, and common Hinglish
GULF_PHRASES: Dict[str, str] = {
    # ── Arabic / Gulf ─────────────────────────────────────────────────────────
    "يا ربي":    "Oh God",
    "يارب":      "Oh God",
    "والله":     "By God",
    "يلا":       "Let's go",
    "طيب":       "okay",
    "تمام":      "perfect",
    "ماشي":      "okay",
    "أيوه":      "yes",
    "لأ":        "no",
    "زين":       "good",
    "هيشورت":    "task shortcut",
    "التسليم":   "delivery",
    "الاستقبال": "reception",
    "الإنتاج":   "production",
    "البيئة":    "environment",
    "النظام":    "the system",
    "المستشفى":  "the hospital",
    "المريض":    "the patient",
    "الدكتور":   "the doctor",
    "نعم":       "yes",
    "لا":        "no",
    "شكراً":     "thank you",
    "من فضلك":   "please",
    "انتظر":     "wait",
    "صح":        "correct",
    "خطأ":       "error",
    # ── Hindi / Devanagari ────────────────────────────────────────────────────
    "हाँ":        "yes",
    "हां":        "yes",
    "नहीं":       "no",
    "ठीक है":     "okay",
    "ठीक":       "okay",
    "अच्छा":     "good",
    "बिल्कुल":   "absolutely",
    "धन्यवाद":   "thank you",
    "कृपया":     "please",
    "रुकिए":     "wait",
    "सही":       "correct",
    "गलत":       "wrong",
    "समझ गया":   "understood",
    "समझा":      "understood",
    "मरीज़":     "patient",
    "डॉक्टर":    "doctor",
    "अस्पताल":   "hospital",
    "दवा":       "medicine",
    "रिपोर्ट":   "report",
    "सिस्टम":    "system",
    # ── Telugu ───────────────────────────────────────────────────────────────
    "అవును":     "yes",
    "అవు":       "yes",
    "కాదు":      "no",
    "లేదు":      "no",
    "సరే":       "okay",
    "సరైనది":    "correct",
    "బాగుంది":   "good",
    "ధన్యవాదాలు": "thank you",
    "దయచేసి":    "please",
    "ఆగండి":     "wait",
    "తప్పు":     "wrong",
    "అర్థమైంది": "understood",
    "రోగి":      "patient",
    "డాక్టర్":   "doctor",
    "ఆసుపత్రి":  "hospital",
    "మందు":      "medicine",
    "నివేదిక":   "report",
    "వ్యవస్థ":   "system",
}

# Tags that represent recognised noise/error states (never send to CoT)
NOISE_TAGS: frozenset = frozenset({
    "[audio dropout]", "[unclear]", "[noise/silence]", "[hallucination]", "...",
})

# ── Domain pre-correction rules (applied BEFORE GPT) ─────────────────────────
COMMON_FIXES: Dict[str, str] = {
    # Radiology variants
    "radios": "radiology", "radiose": "radiology", "radio": "radiology",
    "radius": "radiology",
    "radio registration": "radiology",                      # ✅ NEW phrase fix
    # Environment variants
    "efficient": "environment", "invirment": "environment", "mirment": "environment",
    "the environment is somewhat": "the environment is unstable",
    "environment is somewhat": "environment is unstable",
    # Process/admin
    "administrate": "process", "i can't administrate": "I cannot manage the process",
    # Resubmission
    "respond": "resubmission", "re-submission": "resubmission",
    # Revenue cycle
    "insurance cycle": "revenue cycle",
    # System terms
    "sandbox is used for registration": "sandbox is used for testing",
    "h3": "HIS", "ncm": "RCM", "pcm": "RCM", "pc": "RCM",
    # Brand names
    "wallue": "Waseel", "wallee": "Waseel", "wazeel": "Waseel",
    "clinisey": "Clinicy", "clinicey": "Clinicy",
    # Deploy/staging
    "decloy": "deploy", "deply": "deploy",
    "stabox": "sandbox", "sandbok": "sandbox", "sandox": "sandbox",
    "6sting": "staging", "staiging": "staging",
    # Insurance/eligibility
    "anshurns": "insurance", "eligiblity": "eligibility",
    # ✅ NEW: API / connection fixes
    "cancel api": "continue API",
    "cancel the api": "continue the API",
    "unstable connecting": "unstable connection",
    "unstable connect": "unstable connection",
    # ✅ NEW: Common mis-heard phrases
    "sending services tonight": "sending services",
    "they use registration": "they use it for registration",
    "my mind": "in my opinion",
    "in my mind": "in my opinion",
    # ✅ NEW: Common Arabic phonetic near-misses (English output)
    "the expectation was today": "The expectation today was that we could proceed.",
    "noses are ready": "the system is ready",
    "imagine the atmosphere": "Oh God",
    # ✅ Hinglish phonetic fixes — Hindi-accented English STT errors
    "sistam": "system", "sistum": "system",
    "updait": "update", "apdet": "update",
    "reebmission": "resubmission", "reesubmission": "resubmission",
    "elisibility": "eligibility", "eligiblity": "eligibility",
    "biling": "billing", "beel": "billing",
    "doktar": "doctor", "daktar": "doctor",
    "marij": "patient", "mareej": "patient",
    "aspatal": "hospital", "haspataal": "hospital",
    "dava": "medicine", "dawai": "medicine",
    "riport": "report",
    "klame": "claim", "klaim": "claim",
    "athorizeshon": "authorization", "autherization": "authorization",
    "insurans": "insurance", "inshurance": "insurance",
    # ✅ Telugu-accented English STT fixes
    "sistam loo": "in the system",
    "update chesaanu": "updated",
    "cheyandi": "please do",
    "chustanu": "I will check",
    "cheskoni": "after taking",
    "vachindi": "it came / it arrived",
    "ayyindi": "it is done",
    "ledu": "not there / doesn't exist",
    "undi": "it is there",
    "cheyali": "need to do",
    "pampinchandi": "please send",
    # Sentence fragments to discard
    "when we present": "",
}

# Fragment patterns that indicate a broken/incomplete STT output — discard them
FRAGMENT_PATTERNS: List[str] = [
    r"^when we\b",
    r"^the environment is$",
    r"^the system is$",
    r"^and the\b.{0,15}$",
    r"^\w{1,4}$",            # single very short word — likely noise
]

TECH_WORDS: List[str] = [
    "RCM", "ERP", "radiology", "laboratory", "pharmacy", "claim",
    "staging", "sandbox", "production", "environment", "deployment",
    "resubmission", "eligibility", "insurance", "authorization",
    "Waseel", "Clinicy", "Nafis", "HIS", "EMR", "EHR", "FHIR",
    "ICD-10", "CPT", "UAT", "QA", "API", "DICOM",
]


def apply_domain_fixes(text: str) -> str:
    """
    Rule-based pre-correction — runs BEFORE GPT to catch common mis-hearings.
    Entries mapped to '' are FRAGMENTS — returns empty string to signal discard.
    """
    result = text
    for wrong, correct in COMMON_FIXES.items():
        if not wrong:
            continue
        result = re.sub(rf"\b{re.escape(wrong)}\b", correct, result, flags=re.IGNORECASE)
    return result.strip()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [1]  SEMANTIC CACHE  — LRU + fuzzy TTL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SemanticCache:
    """
    Two-level read path:
      1. Exact normalised key  (O(1))
      2. Fuzzy match on the last 40 stored keys  (difflib ratio ≥ fuzz)
    LRU eviction when the store exceeds max_size.
    All entries expire after `ttl` seconds.
    Namespaced by `kind` ('correction', 'translation', 'stt', …).
    """

    def __init__(self, max_size: int = 300,
                 fuzzy_threshold: float = 0.93,
                 ttl: int = 3600):
        self._store: Dict[str, OrderedDict] = {}
        self._ts:    Dict[str, Dict[str, float]] = {}
        self.max_size = max_size
        self.fuzz     = fuzzy_threshold
        self.ttl      = ttl
        self.hits     = 0
        self.misses   = 0

    def _norm(self, t: str) -> str:
        return re.sub(r'\s+', ' ', t.lower().strip())

    def _ensure(self, kind: str):
        if kind not in self._store:
            self._store[kind] = OrderedDict()
            self._ts[kind]    = {}

    def get(self, text: str, kind: str = "correction") -> Optional[Dict]:
        self._ensure(kind)
        key   = self._norm(text)
        store = self._store[kind]
        now   = time.time()
        ts    = self._ts[kind]

        if key in store and now - ts.get(key, 0) < self.ttl:
            store.move_to_end(key)
            self.hits += 1
            return store[key]

        for k in list(store.keys())[-40:]:
            if now - ts.get(k, 0) < self.ttl:
                if difflib.SequenceMatcher(None, key, k).ratio() >= self.fuzz:
                    store.move_to_end(k)
                    self.hits += 1
                    return store[k]

        self.misses += 1
        return None

    def set(self, text: str, val: Dict, kind: str = "correction"):
        self._ensure(kind)
        key   = self._norm(text)
        store = self._store[kind]
        if len(store) >= self.max_size:
            oldest = next(iter(store))
            store.pop(oldest)
            self._ts[kind].pop(oldest, None)
        store[key]          = val
        self._ts[kind][key] = time.time()

    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def stats(self) -> str:
        total = self.hits + self.misses
        return (f"cache {self.hits}/{total} hits "
                f"({100*self.hit_rate():.0f}%)")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [2]  TOKEN BUDGET MANAGER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TokenBudgetManager:
    BUDGETS: Dict[str, int] = {
        "trivial":  60,
        "simple":   150,
        "standard": 512,
        "full_cot": 1024,
    }

    def get(self, tier: str) -> int:
        return self.BUDGETS.get(tier, 512)

    def compress_context(self,
                         transcript: List[Dict],
                         max_entries: int = 5) -> str:
        if not transcript:
            return "(session start)"
        tail  = transcript[-max_entries:]
        older = transcript[:-max_entries] if len(transcript) > max_entries else []
        lines: List[str] = []
        if older:
            snips = [
                f"Spk{s['speaker']+1}:{' '.join(s['original'].split()[:5])}…"
                for s in older[-8:]
            ]
            lines.append("[Earlier] " + " | ".join(snips))
        for s in tail:
            ts = s.get("timestamp", "--:--")
            lines.append(
                f"  Speaker {s['speaker']+1} [{ts}]: "
                f"{s['original']} → {s.get('translation', '')}"
            )
        return "\n".join(lines)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [3]  AUTO LEARNING LOOP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class AutoLearningLoop:
    PROMOTE_THRESHOLD = 1   # instant: first correction becomes a session rule

    def __init__(self):
        self._corr:  Dict[str, str] = {}
        self._freq:  Dict[str, int] = {}
        self._rules: Dict[str, str] = {}

    def record(self, wrong: str, correct: str):
        if not wrong or not correct:
            return
        wrong   = wrong.strip()
        correct = correct.strip()
        if wrong == correct:
            return
        k = wrong.lower()
        self._corr[k] = correct
        self._freq[k] = self._freq.get(k, 0) + 1
        if self._freq[k] >= self.PROMOTE_THRESHOLD and k not in self._rules:
            self._rules[k] = correct
            print(f"  📌 Rule promoted: '{wrong}' → '{correct}'")

    def apply_rules(self, text: str) -> Optional[str]:
        if not self._rules:
            return None
        low     = text.lower()
        result  = text
        changed = False
        for wrong, correct in self._rules.items():
            if wrong in low:
                result  = re.sub(re.escape(wrong), correct,
                                 result, flags=re.IGNORECASE)
                changed = True
        return result if changed else None

    def context_str(self, limit: int = 10) -> str:
        items = list(self._corr.items())[-limit:]
        return "\n".join(f"  '{w}' = '{c}'" for w, c in items) or "(none)"

    def reset(self):
        self._corr.clear()
        self._freq.clear()
        self._rules.clear()

    @property
    def rule_count(self) -> int:
        return len(self._rules)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [4]  ENHANCED VAD  — audio level + ZCR filtering
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class EnhancedVAD:
    def __init__(self,
                 energy_floor: float = 180.0,   # ✅ LOWERED: was 200 — catches quieter speech
                 zcr_max: float = 0.92):         # ✅ RAISED: was 0.88 — prevents cutting off soft consonants
        self.energy_floor = energy_floor
        self.zcr_max      = zcr_max

    def is_speech(self, audio: np.ndarray, sr: int = SAMPLE_RATE) -> Tuple[bool, Dict]:
        if len(audio) == 0:
            return False, {"rms": 0.0, "zcr": 0.0}

        # 🔥 Minimum duration gate: discard clips shorter than 0.7 seconds
        if len(audio) < sr * 0.7:
            return False, {"rms": 0.0, "zcr": 0.0, "reason": "too_short"}

        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        if rms < self.energy_floor:
            return False, {"rms": rms, "zcr": 0.0}

        signs   = np.sign(audio)
        crosses = int(np.sum(signs[:-1] != signs[1:]))
        zcr     = float(crosses / max(len(audio), 1))
        if zcr > self.zcr_max:
            return False, {"rms": rms, "zcr": zcr}

        return True, {"rms": rms, "zcr": zcr}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [5]  PIPELINE DECISION ENGINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class PipelineDecisionEngine:
    TRIVIAL_SET = {
        # English
        "yes", "no", "ok", "okay", "hello", "hi", "bye", "thanks", "good",
        "sure", "right", "hmm", "yeah", "yep", "great", "fine", "alright",
        # Arabic (Gulf)
        "طيب", "ماشي", "أيوه", "لأ", "تمام", "زين", "مرحبا", "نعم", "لا",
        # Hindi (Devanagari)
        "हाँ", "हां", "नहीं", "ठीक", "अच्छा", "बिल्कुल", "धन्यवाद",
        # Telugu
        "అవును", "అవు", "కాదు", "లేదు", "సరే", "బాగుంది", "ధన్యవాదాలు",
        # Hinglish (transliterated Hindi — common in mixed speech)
        "haan", "nahi", "theek hai", "theek", "achha", "achcha", "bilkul",
        "shukriya", "dhanyavaad",
        # Telugu romanised
        "avunu", "kaadu", "sare", "bagundi",
    }

    TECH_TERMS = set(
        "RCM EMR EHR HIS FHIR HL7 UAT QA API staging sandbox "
        "production deployment environment Waseel Clinicy backend "
        "frontend database server sprint release build cycle IRP DICOM".split()
    )

    MIN_CONF         = 0.65   # skip low-confidence garbage segments
    SIMPLE_CONF      = 0.90   # ✅ RAISED: was 0.75 — only very confident short segments skip full_cot
    MAX_SIMPLE_WORDS = 5      # ✅ LOWERED: was 12 — only very short utterances use mini

    def score(self, text: str) -> float:
        words = text.split()
        if not words:
            return 0.0
        s = 1.0
        if len(words) >= 4:
            unique_ratio = len({w.lower() for w in words}) / len(words)
            if unique_ratio < 0.5:
                s -= 0.40
        # Only penalise truly unexpected scripts (Cyrillic, CJK) that appear in
        # an otherwise Latin/Arabic/Indic transcript — these are hallucinations.
        # Do NOT penalise Devanagari, Telugu, Tamil, Arabic, etc. — valid inputs.
        cyrillic_cjk = sum(
            1 for c in text
            if '\u0400' <= c <= '\u04FF'   # Cyrillic
            or '\u4E00' <= c <= '\u9FFF'   # CJK
        )
        if cyrillic_cjk > 3:
            s -= 0.35
        return max(0.0, min(1.0, s))

    def decide(self,
               text: str,
               conf: float = 1.0,
               learning: Optional[AutoLearningLoop] = None) -> str:
        s = text.strip()
        if not s:
            return "skip"
        if s in NOISE_TAGS:
            return "skip"
        if conf < self.MIN_CONF:
            return "trivial"
        words = s.split()
        if s.lower() in self.TRIVIAL_SET:
            return "trivial"
        if learning and learning.apply_rules(s) is not None:
            return "rule"
        has_en   = any(c.isalpha() and c.isascii() for c in s)
        # Detect ANY non-Latin script that co-exists with Latin → code-switching
        # (Arabic, Devanagari, Telugu, Tamil, Kannada, Malayalam, Bengali, etc.)
        has_non_latin = any(
            '\u0600' <= c <= '\u06FF'   # Arabic / Urdu
            or '\u0900' <= c <= '\u097F'  # Devanagari (Hindi/Marathi)
            or '\u0C00' <= c <= '\u0C7F'  # Telugu
            or '\u0B80' <= c <= '\u0BFF'  # Tamil
            or '\u0C80' <= c <= '\u0CFF'  # Kannada
            or '\u0D00' <= c <= '\u0D7F'  # Malayalam
            or '\u0980' <= c <= '\u09FF'  # Bengali
            or '\u0A00' <= c <= '\u0AFF'  # Punjabi
            or '\u0A80' <= c <= '\u0AFF'  # Gujarati
            for c in s
        )
        # ✅ ACCURACY FIX: ANY tech term → full_cot (gpt-4.1 handles domain terms far better)
        has_tech = any(w.upper() in self.TECH_TERMS for w in words)
        if has_tech:
            return "full_cot"
        # Mixed-language content always needs full CoT for correct translation
        if has_non_latin and has_en:
            return "full_cot"
        if (len(words) <= self.MAX_SIMPLE_WORDS
                and not has_non_latin
                and conf >= self.SIMPLE_CONF):
            return "simple"
        return "full_cot"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [6]  VALIDATION LAYER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class ValidationLayer:
    HALL_RE = re.compile(
        r'[\u0400-\u04FF]{4,}'       # Cyrillic (≥4 chars)
        r'|[\u4E00-\u9FFF]{3,}'      # CJK (≥3 chars)
        # Devanagari (Hindi) — valid input language, NOT a hallucination
        # Telugu — valid input language, NOT a hallucination
        r'|dziękuję|спасибо|merci|gracias'
    )
    DIFF_MIN = 0.12
    SIM_MIN  = 0.20

    def is_hallucination(self, t: str) -> bool:
        return bool(self.HALL_RE.search(t))

    def semantic_similarity(self, a: str, b: str) -> float:
        aw = set(a.lower().split())
        bw = set(b.lower().split())
        if not aw and not bw:
            return 1.0
        if not aw or not bw:
            return 0.0
        return len(aw & bw) / len(aw | bw)

    def back_translation_similarity(self,
                                    original: str,
                                    back_translated: str) -> float:
        return self.semantic_similarity(original, back_translated)

    def was_changed(self, orig: str, corr: str) -> bool:
        ratio = difflib.SequenceMatcher(
            None, orig.lower().split(), corr.lower().split()
        ).ratio()
        return (1.0 - ratio) >= self.DIFF_MIN

    def domain_score(self, text: str) -> int:
        """Count domain tech terms present in text (case-insensitive)."""
        t = text.lower()
        return sum(1 for w in TECH_WORDS if w.lower() in t)

    def validate(self, raw: str, corr: str,
                 trans: str) -> Tuple[str, str, bool]:
        if self.is_hallucination(corr):
            corr = raw
        if self.is_hallucination(trans):
            trans = "[noise/silence]"
            corr  = "..."
        sim = self.semantic_similarity(raw, corr)
        if sim < self.SIM_MIN and corr != "...":
            corr = raw
        # Domain guard: revert if correction drops tech terms present in original
        if self.domain_score(raw) > 0 and self.domain_score(corr) < self.domain_score(raw):
            print(f"  🛡️  Domain guard — reverting correction")
            corr = raw
        return corr, trans, self.was_changed(raw, corr)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [7]  PROMPT TEMPLATE MANAGER  v2.2
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class PromptTemplateManager:
    VERSION = "3.0"

    SYSTEM_SIMPLE = (
        "STT post-processor v{version}: Multilingual healthcare IT transcription.\n"
        "Input may be ANY language or MIX of languages "
        "(Arabic, Hindi, Telugu, Tamil, Urdu, French, etc. + English code-switching).\n"
        "DOMAIN GLOSSARY (never change these): "
        "Nafis, Waseel, Clinicy, RCM, EMR, EHR, HIS, FHIR, ERP, API, "
        "radiology, laboratory, pharmacy, resubmission, revenue cycle, "
        "ICD-10, CPT, UAT, QA, sandbox, staging, production.\n"
        "Fix ONLY: phonetic mis-spellings, tech-term errors, "
        "word-dropout loops (3+ identical consecutive words → collapse to once), "
        "and foreign-script hallucinations.\n"
        "PROTECT: never replace a domain term with a common word.\n"
        "NEVER output an incomplete sentence or fragment — if the input is already "
        "a fragment, return it verbatim as-is rather than completing it.\n"
        "LANGUAGE RULES: Preserve the original language(s) in 'corrected_original'. "
        "If input is non-English or mixed, provide the English translation in 'translation'.\n"
        "Only replace clearly random Cyrillic/CJK noise (≥4 chars) with '...'.\n"
        "DO NOT invent content. Unclear audio → '...'.\n"
        # ✅ FIX: double-braces escape {} so .format(version=...) does not crash
        'Return ONLY JSON array: [{{"idx":0,"corrected":"...","translation":"[MUST BE ENGLISH]"}}]\n'
        "CRITICAL: \"translation\" field MUST always be in English regardless of source language."
    )

    SYSTEM_FULL_COT = """\
Expert Multilingual STT post-processor — healthcare IT transcription.
Prompt version: {version}  |  STRICT GROUNDING: never invent content.

LANGUAGE SCOPE: Primary languages are Arabic, Hindi, Telugu, and English.
Input may be any single language or code-switched mix (Hinglish, Tenglish, Arabizi, etc.).
Preserve all original language(s) verbatim in "corrected" field.
Always produce a clean English translation in "translation" field.

SILENT CHECKS (compute internally, do NOT narrate):
  a) Collapse 3+ consecutive identical words → once  (dropout artefact)
  b) Input may be ANY language — keep the original text, do NOT replace with "..."
     Only true noise (random Cyrillic ≥4 chars / CJK ≥3 chars) → "..." / "[noise/silence]"
  c) Phonetic near-miss → apply MASTER DICT
  d) Translation cross-check: does EN translation faithfully match corrected original?
  e) Semantic guard: reject correction if Jaccard similarity < 0.20 vs raw

DOMAIN GLOSSARY — NEVER change these:
  Nafis, Waseel, Clinicy, RCM, EMR, EHR, HIS, FHIR, HL7, ERP, API,
  radiology, laboratory, pharmacy, resubmission, revenue cycle,
  ICD-10, CPT, UAT, QA, DICOM, IRP

MASTER DICT — apply phonetic near-miss corrections:
  [Arabic/Gulf]
  mirment / invirment → environment     6sting / sting      → staging
  decloy              → deploy          stabox / sandbok     → sandbox
  NCM / PCM / PC      → RCM             H3                   → HIS
  Wallue              → Waseel          Clinisey             → Clinicy
  Anshurns            → insurance       respond              → resubmission
  radios / radio      → radiology       efficient            → environment
  administrate        → process         insurance cycle      → revenue cycle
  "imagine the atmosphere" → يا ربي
  "noses are ready"        → النظام جاهز

  [Hindi/Hinglish STT errors]
  sistam / sistum   → system           updait / apdet      → update
  reebmission       → resubmission     elisibility         → eligibility
  biling / beel     → billing          klame / klaim       → claim
  doktar / daktar   → doctor           aspatal / haspataal → hospital
  dava / dawai      → medicine         riport              → report

  [Telugu-accented English]
  sistam loo        → in the system    update chesaanu     → updated
  ayyindi           → it is done       ledu                → does not exist
  undi              → it exists        cheyali             → needs to be done

PROTECT RULE: if original has a domain term and correction removes it → revert to original

STRICT OUTPUT RULES:
  1. Return ONLY a valid JSON array — no markdown, no preamble
  2. new_learnings: at most 3 HIGH-CONFIDENCE pairs — never guess
  3. Uncertain text → keep original verbatim
  4. NEVER alter: Waseel, Clinicy, RCM, EMR, HIS, UAT, QA
  5. CoT reasoning is SILENT — output only the JSON array

[{{"idx":0,"corrected":"...","translation":"...","new_learnings":{{}}}}]"""

    def get_system(self, tier: str) -> str:
        if tier == "full_cot":
            return self.SYSTEM_FULL_COT.format(version=self.VERSION)
        return self.SYSTEM_SIMPLE.format(version=self.VERSION)

    def get_system_with_context(self, tier: str, context: str,
                                learnings: str) -> str:
        """
        🔥 NEW: Returns a system prompt with session context BAKED IN.
        This gives GPT meeting-level awareness even for the simple tier,
        so it can resolve ambiguous terms using what was said earlier.
        """
        base = self.get_system(tier)
        if not context or context == "(session start)":
            return base
        ctx_block = (
            f"\n\n=== LIVE MEETING CONTEXT (use to resolve ambiguous words) ===\n"
            f"{context}\n"
            f"=== SESSION CORRECTIONS LEARNED ===\n"
            f"{learnings}\n"
            "Use the above context to pick the correct domain term when audio is ambiguous."
        )
        return base + ctx_block

    def build_batch_msg(self,
                        segments: List[Dict],
                        context: str,
                        learnings: str,
                        tier: str,
                        text_key: str = "original") -> str:
        lines = [
            f"[{i}] Speaker {s['speaker']+1} [{s.get('timestamp','--')}]: "
            f"{s.get(text_key, s.get('transcript', ''))}"
            for i, s in enumerate(segments)
        ]
        body = "\n".join(lines)
        # 🔥 ALL tiers now receive context — critical for sentence coherence
        if tier == "full_cot":
            return (
                f"=== MEETING CONTEXT ===\n{context}\n\n"
                f"=== SESSION CORRECTIONS LEARNED ===\n{learnings}\n\n"
                f"=== SEGMENTS TO CORRECT ===\n{body}\n\n"
                "Return JSON array. Apply checks a→e silently."
            )
        # simple tier: also gets context so GPT can resolve ambiguous terms
        if context and context != "(session start)":
            return (
                f"=== RECENT CONTEXT ===\n{context}\n\n"
                f"=== SEGMENTS ===\n{body}\n"
                "Return JSON array."
            )
        return f"Segments:\n{body}\nReturn JSON array."


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [8]  UTILITIES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def clean_stt_text(text: str) -> str:
    """
    Post-STT cleanup:
      • Collapse 3+ identical consecutive words (hallucination dropout artefact)
      • Strip known sentence fragments
      • Strip leaked tier labels (full_cot / simple / trivial …) from GPT output
      • Ensure sentence starts with capital, ends with punctuation
      • Returns empty string if text is too short / a fragment
    """
    if not text:
        return ""

    # ✅ FIX: Remove any tier routing label that GPT accidentally echoed
    _TIER_TAG_RE = re.compile(r'\b(full_cot|simple|trivial|skip|rule)\b', re.IGNORECASE)
    text = _TIER_TAG_RE.sub('', text).strip()

    # Collapse word loops: "the the the" → "the"
    text = re.sub(r'\b(\w+)(\s+\1){2,}\b', r'\1', text, flags=re.IGNORECASE)

    text = text.strip()
    if not text:
        return ""

    words = text.split()

    # Discard if fewer than 3 words (almost certainly a fragment)
    if len(words) < 3:
        return ""

    # Check against known fragment patterns
    for pat in FRAGMENT_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            # If it's a known bad fragment return empty
            if len(words) < 6:
                return ""

    # Remove incomplete trailing words (sentence ends with preposition / article)
    INCOMPLETE_ENDINGS = {"the", "a", "an", "is", "are", "was", "were",
                          "and", "or", "but", "in", "on", "at", "to", "of",
                          "with", "by", "for", "from"}
    while words and words[-1].lower().rstrip(".,!?") in INCOMPLETE_ENDINGS:
        words = words[:-1]

    if len(words) < 3:
        return ""

    text = " ".join(words)

    # Capitalise first letter
    text = text[0].upper() + text[1:]

    # Ensure sentence ends with punctuation
    if text and text[-1] not in ".?!":
        text += "."

    return text


def detect_lang(text: str) -> str:
    """
    Detect language from Unicode character ranges. Returns ISO-639-1 code.
    Returns "mixed" when two or more scripts are both meaningfully present
    (e.g. Arabic+Latin, Hindi+Latin, Telugu+Latin) — callers should treat
    "mixed" as requiring translation even if some words are English.
    """
    if not text:
        return "en"
    n   = max(len(text), 1)
    ar  = sum(1 for c in text if '\u0600' <= c <= '\u06FF')   # Arabic / Urdu
    hi  = sum(1 for c in text if '\u0900' <= c <= '\u097F')   # Devanagari (Hindi/Marathi)
    te  = sum(1 for c in text if '\u0C00' <= c <= '\u0C7F')   # Telugu
    ta  = sum(1 for c in text if '\u0B80' <= c <= '\u0BFF')   # Tamil
    kn  = sum(1 for c in text if '\u0C80' <= c <= '\u0CFF')   # Kannada
    ml  = sum(1 for c in text if '\u0D00' <= c <= '\u0D7F')   # Malayalam
    bn  = sum(1 for c in text if '\u0980' <= c <= '\u09FF')   # Bengali
    pa  = sum(1 for c in text if '\u0A00' <= c <= '\u0A7F')   # Punjabi (Gurmukhi)
    gu  = sum(1 for c in text if '\u0A80' <= c <= '\u0AFF')   # Gujarati
    lat = sum(1 for c in text if c.isalpha() and c.isascii()) # Latin / English

    # Ratios
    ar_r  = ar  / n
    hi_r  = hi  / n
    te_r  = te  / n
    ta_r  = ta  / n
    kn_r  = kn  / n
    ml_r  = ml  / n
    bn_r  = bn  / n
    pa_r  = pa  / n
    gu_r  = gu  / n
    lat_r = lat / n

    # Score non-Latin scripts
    non_latin = {
        "ar": ar_r, "hi": hi_r, "te": te_r, "ta": ta_r,
        "kn": kn_r, "ml": ml_r, "bn": bn_r, "pa": pa_r, "gu": gu_r,
    }
    dominant_nl_lang = max(non_latin, key=non_latin.get)
    dominant_nl_r    = non_latin[dominant_nl_lang]

    MIX_THRESHOLD = 0.08  # 8 % of chars is "meaningfully present"

    if dominant_nl_r > MIX_THRESHOLD:
        # Check if Latin is also meaningfully present → code-switching
        if lat_r > MIX_THRESHOLD:
            return "mixed"   # e.g. Arabic+English, Hindi+English
        return dominant_nl_lang

    return "en"   # all-Latin or nearly so


def extract_mean_pitch(pcm: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    """
    NEW UTILITY — Estimate mean fundamental frequency (F0) via autocorrelation.
    Returns normalised value in [0, 1]:
      • Low (~0.1) = deep male voice  (~80-120 Hz)
      • Mid (~0.5) = average voice    (~150-200 Hz)
      • High (~0.9) = female voice    (~220-280 Hz)
    Used as an extra dimension in speaker embeddings so that speakers with
    similar MFCC but different pitch (e.g. same-gender vs mixed) are separable.
    """
    FRAME_LEN = int(sr * 0.025)   # 25 ms
    HOP       = int(sr * 0.010)   # 10 ms hop
    MIN_LAG   = int(sr / 300)     # 300 Hz upper bound
    MAX_LAG   = int(sr / 60)      # 60  Hz lower bound

    pitches: List[float] = []
    for i in range(0, len(pcm) - FRAME_LEN, HOP):
        frame = pcm[i:i + FRAME_LEN].astype(np.float64)
        frame -= np.mean(frame)
        if np.max(np.abs(frame)) < 50:          # skip near-silent frames
            continue
        corr = np.correlate(frame, frame, mode='full')
        corr = corr[len(corr) // 2:]            # keep positive lags
        if MAX_LAG < len(corr):
            peak_idx = MIN_LAG + int(np.argmax(corr[MIN_LAG:MAX_LAG]))
            f0 = sr / peak_idx
            if 60.0 <= f0 <= 300.0:
                pitches.append(f0)

    if not pitches:
        return 0.5
    median_f0 = float(np.median(pitches))
    return float(np.clip((median_f0 - 60.0) / 240.0, 0.0, 1.0))


def compute_word_diff(raw: str, corrected: str) -> List[Dict]:
    raw_w = raw.split()
    cor_w = corrected.split()
    tokens: List[Dict] = []
    for op, i1, i2, j1, j2 in (
        difflib.SequenceMatcher(None, raw_w, cor_w).get_opcodes()
    ):
        if op == "equal":
            tokens.extend({"type": "same",    "text": w} for w in raw_w[i1:i2])
        elif op == "replace":
            tokens.extend({"type": "removed", "text": w} for w in raw_w[i1:i2])
            tokens.extend({"type": "added",   "text": w} for w in cor_w[j1:j2])
        elif op == "delete":
            tokens.extend({"type": "removed", "text": w} for w in raw_w[i1:i2])
        elif op == "insert":
            tokens.extend({"type": "added",   "text": w} for w in cor_w[j1:j2])
    return tokens


def format_ts(secs: float) -> str:
    s = max(0, int(secs))
    return f"{s // 60:02d}:{s % 60:02d}"


def format_ts_long(secs: float) -> str:
    s = max(0, int(secs))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def _safe_parse_json(raw: str,
                     fallback_keys: Optional[List[str]] = None) -> Any:
    cleaned = raw.strip().replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    for suffix in ['"}]', '"}}]', '"}', '"}}', '"},\"new_learnings\":{}}]']:
        try:
            return json.loads(cleaned + suffix)
        except json.JSONDecodeError:
            pass

    result: Dict[str, str] = {}
    if fallback_keys:
        for key in fallback_keys:
            m = re.search(
                rf'"{re.escape(key)}"\s*:\s*"((?:[^"\\]|\\.)*)', cleaned
            )
            if m:
                val = (m.group(1)
                         .replace('\\"', '"')
                         .replace('\\n', '\n')
                         .replace('\\\\', '\\'))
                result[key] = val

    if result:
        print(f"⚠️  JSON parse recovered via regex: {list(result.keys())}")
        return result

    raise json.JSONDecodeError(f"Cannot parse: {raw[:120]}", raw, 0)


def build_shared_bundle() -> Dict:
    return {
        "cache":     SemanticCache(),
        "budget":    TokenBudgetManager(),
        "learning":  AutoLearningLoop(),
        "vad":       EnhancedVAD(),
        "engine":    PipelineDecisionEngine(),
        "validator": ValidationLayer(),
        "prompts":   PromptTemplateManager(),
    }

# Public API — everything main.py and Transcribe_video.py need
__all__ = [
    "SemanticCache", "TokenBudgetManager", "AutoLearningLoop",
    "EnhancedVAD", "PipelineDecisionEngine", "ValidationLayer",
    "PromptTemplateManager", "build_shared_bundle",
    "OPENAI_STT_MODEL", "OPENAI_COT_MODEL", "OPENAI_MINI_MODEL",
    "OPENAI_STT_URL", "OPENAI_CHAT_URL",
    "SAMPLE_RATE", "STT_PROMPT_HINT", "GULF_PHRASES", "NOISE_TAGS",
    "COMMON_FIXES", "TECH_WORDS", "FRAGMENT_PATTERNS",
    "detect_lang", "compute_word_diff", "format_ts", "format_ts_long",
    "_safe_parse_json", "extract_mean_pitch",
    "apply_domain_fixes", "clean_stt_text",
]