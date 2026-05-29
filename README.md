# 🩺 Doctor-Patient AI Transcript — Intelligent Medical Conversation Analysis System

### 🚀 Real-Time Speech-to-Text | AI Medical Summaries | Clinical Insights | FastAPI | Docker | Kubernetes | AWS EKS

[![Python](https://img.shields.io/badge/Python-3.10+-blue)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-success)]()
[![OpenAI](https://img.shields.io/badge/OpenAI-LLM-green)]()
[![Docker](https://img.shields.io/badge/Docker-Containerized-blue)]()
[![Kubernetes](https://img.shields.io/badge/Kubernetes-Orchestrated-326CE5)]()
[![AWS](https://img.shields.io/badge/AWS-EKS-orange)]()
[![GitHub Actions](https://img.shields.io/badge/CI/CD-GitHub%20Actions-success)]()
[![Status](https://img.shields.io/badge/Status-Production%20Ready-success)]()


<img width="1920" height="1080" alt="Screenshot 2026-05-08 105730" src="https://github.com/user-attachments/assets/a8fc41ec-bdf5-4e53-8830-8e580b68e4c3" />

---

# 🚀 Overview

**Doctor-Patient AI Transcript** is an AI-powered healthcare documentation platform that automatically converts doctor-patient conversations into structured medical transcripts, clinical summaries, medication recommendations, and actionable healthcare insights.

The platform leverages modern Speech-to-Text technology, Generative AI, and Large Language Models (LLMs) to reduce documentation burden, improve consultation efficiency, and enhance clinical decision-making.

Designed for modern healthcare environments, the system supports cloud-native deployment using Docker, Kubernetes, GitHub Actions, and AWS EKS.

---

## 🎯 Problem Statement

Healthcare professionals spend a significant portion of their time on:

* Manual note-taking
* Patient documentation
* Consultation summaries
* Medication review
* Clinical record management
* Post-consultation reporting

Studies show physicians spend nearly as much time documenting as they do interacting with patients. AI-powered automation can significantly reduce this administrative workload while improving documentation quality.

---

## 💡 Solution

Doctor-Patient AI Transcript automates the entire consultation workflow:

1️⃣ Capture doctor-patient conversation

2️⃣ Convert speech into structured transcript

3️⃣ Identify clinical symptoms and conditions

4️⃣ Generate AI-powered consultation summaries

5️⃣ Extract medical entities and recommendations

6️⃣ Provide medication suggestions

7️⃣ Store structured healthcare records

---

## ✨ Key Features

### 🎤 Real-Time Audio Transcription

Converts doctor-patient conversations into highly accurate text transcripts using advanced speech recognition models.

### 🤖 AI-Powered Clinical Summaries

Automatically generates concise consultation summaries and patient visit notes.

### 💊 Medication Recommendation Support

Suggests relevant medications based on patient symptoms and consultation context.

### 📋 Structured Medical Documentation

Transforms unstructured conversations into organized clinical records.

### 🧠 LLM-Powered Medical Analysis

Uses Generative AI to extract healthcare insights and support decision-making.

### ⚡ FastAPI Backend

High-performance REST APIs for seamless integration.

### ☸️ Kubernetes & AWS Ready

Enterprise-grade deployment architecture.

### 🔄 CI/CD Automation

Automated deployment pipelines using GitHub Actions.

---

# 🏗️ System Architecture

```text
Doctor & Patient Conversation
                │
                ▼
       Audio Processing Layer
                │
                ▼
       Speech-To-Text Engine
                │
                ▼
        Transcript Generation
                │
                ▼
        AI Processing Layer
                │
                ├────────► Clinical Summary
                │
                ├────────► Symptom Extraction
                │
                ├────────► Medical Insights
                │
                └────────► Medication Suggestions
                │
                ▼
         Structured Response
```

---

# 🛠️ Technology Stack

## Backend

* Python
* FastAPI
* Uvicorn

## Artificial Intelligence

* OpenAI GPT Models
* Prompt Engineering
* Medical NLP
* Clinical Information Extraction

## Speech Processing

* Speech-to-Text
* Audio Processing
* Medical Conversation Analysis

## Infrastructure

* Docker
* Kubernetes
* AWS EKS
* Amazon ECR

## DevOps

* GitHub Actions
* CI/CD Pipelines
* Automated Deployments

---

# 📂 Project Structure

```text
Doctor-Patient-AI-Transcript/
│
├── app/
├── services/
├── routes/
├── models/
├── prompts/
├── audio/
├── transcripts/
│
├── Dockerfile
├── requirements.txt
├── main.py
│
├── .github/
│   └── workflows/
│       └── deploy-qa.yaml
│
├── .k8s/
│   ├── deployment.yaml
│   ├── service.yaml
│   └── namespace.yaml
│
└── README.md
```

---

# 📸 Application Screenshots

### 🎙️ Audio Recording Interface

*Add screenshot here*

### 📄 Generated Transcript

*Add screenshot here*

### 🤖 AI Summary Output

*Add screenshot here*

### 📊 Swagger API Documentation

*Add screenshot here*

### 🚀 GitHub Actions Deployment

*Add screenshot here*

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/saiprakash482126/Doctor-Patient-AI-Transcript.git
```

## Navigate To Project

```bash
cd Doctor-Patient-AI-Transcript
```

## Create Virtual Environment

```bash
python -m venv venv
```

## Activate Environment

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Application

```bash
uvicorn main:app --reload
```

Application URL:

```text
http://localhost:8000
```

Swagger Docs:

```text
http://localhost:8000/docs
```

---

# 🐳 Docker Deployment

```bash
docker build -t doctor-patient-ai .
```

```bash
docker run -p 8000:8000 doctor-patient-ai
```

---

# ☸️ Kubernetes Deployment

```bash
kubectl apply -f .k8s/
```

Verify deployment:

```bash
kubectl get pods
kubectl get svc
kubectl get deployments
```

---

# 🔄 CI/CD Pipeline

GitHub Actions automates:

✅ Build Validation

✅ Docker Image Creation

✅ Container Registry Push

✅ Kubernetes Deployment

✅ Rollout Verification

✅ Production Release Workflow

### Deployment Flow

```text
Developer Push
      │
      ▼
GitHub Repository
      │
      ▼
GitHub Actions
      │
      ▼
Docker Build
      │
      ▼
Container Registry
      │
      ▼
AWS EKS Cluster
      │
      ▼
Doctor-Patient AI Service
```

---

# 📈 Business Impact

* Reduce physician documentation workload
* Improve consultation efficiency
* Standardize clinical records
* Enable AI-assisted healthcare workflows
* Accelerate patient care delivery
* Improve healthcare data quality

---

# 🔮 Future Enhancements

* Multi-Language Consultation Support
* Real-Time Live Transcription
* Voice Biometrics
* Electronic Health Record (EHR) Integration
* Drug Interaction Detection
* Clinical Risk Prediction
* Automated SOAP Notes Generation
* Telemedicine Integration

---

# 👨‍💻 Author

### Sai Prakash

AI Engineer | Data Engineer | Cloud & DevOps Enthusiast

GitHub:
https://github.com/saiprakash482126

Repository:
https://github.com/saiprakash482126/Doctor-Patient-AI-Transcript

---

⭐ If you found this project useful, please consider giving it a star.
