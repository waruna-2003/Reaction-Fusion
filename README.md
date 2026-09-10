# ReactionFusion: Multimodal Sinhala Sentiment & Emotion Analysis Platform

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19-61DAFB?style=flat&logo=react&logoColor=black)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-3178C6?style=flat&logo=typescript&logoColor=white)](https://www.typescriptlang.org)
[![Chrome Extension](https://img.shields.io/badge/Chrome_Extension-Manifest_V3-4285F4?style=flat&logo=googlechrome&logoColor=white)](https://developer.chrome.com/docs/extensions/mv3)

**ReactionFusion** is an end-to-end multimodal AI platform designed to decode complex social media discourse in **Sinhala**. By jointly fusing **Sinhala text semantics** (posts and comments) with the **complete 7-dimensional Facebook reaction distribution** (`Like`, `Love`, `Care`, `Haha`, `Wow`, `Sad`, `Angry`), ReactionFusion eliminates ambiguity surrounding sarcasm, satirical mockery, and collective social sentiment.

---

## 👥 Group Project Work Breakdown & Contribution Flow

This repository serves as the **central Git repository** for the 4-member group project. Members initialize and contribute to the repository chronologically:

| Member | Role & Specialization | Key Deliverables & Code | Handoff Artifact |
| :---: | :--- | :--- | :--- |
| **K.A.A Dilshan** | **Data Engineering, Human Annotation & Sinhala NLP Lead** | Facebook post scraping, 3-annotator protocol, zero-leakage splits (`data/releases/emotion_22_v2/`), Sinhala normalization, 5,000-d TF-IDF extraction | Frozen Dataset Release (`v2`) & Feature Extractors |
| **L.W.L Silva** | **Deep Learning & Multimodal Neural Network Lead** | PyTorch `MultimodalEmotionNet` (dual-branch text + reaction fusion), multi-label weighted BCE training, threshold calibration (0.40–0.90), 22-emotion benchmarks | Trained Model Checkpoint (`best_emotion_model.pt`) & Configs |
| **K.A.P.B Himaranshi** | **Explainable AI & Inference Microservice Lead** | Phase 10 Decision Tree Sentiment Mapper, clinical evidence sums ($S_{\text{pos}}, S_{\text{neg}}$), sarcasm resolution engine, asynchronous FastAPI microservice (`port 8000`), test suite | Live REST API (`/api/v1/analyze`) & Explainability Engine |
| **Tharindu Kothalawala** | **Full-Stack Platform, Mock Facebook & Extension UI/UX Lead** | Mock Facebook React 19 frontend (`port 3000`), Express/Prisma/SQLite backend (`port 4000`, 36 users seed), Grammarly-style Manifest V3 Chrome Extension, DOM body portal overlay | Deployed Social Media Platform & Browser Extension |

---

## 🏛️ System Architecture

```
[ Sinhala Text (5000-d TF-IDF) ] --------+
                                         |--> [ Multimodal Fusion Layer ] --> [ 22 Sigmoid Emotion Heads ]
[ Facebook Reactions (7-d Normalized) ] -+                    |
                                                              v
                                              [ Phase 10 Decision Tree Mapper ]
                                              (Clinical Evidence Sums + Sarcasm Logic)
                                                              |
                                                              v
                                              [ 4-Class Explainable Sentiment ]
                                              (POSITIVE | NEGATIVE | MIXED | NEUTRAL)
```

---

## 📊 Dataset & Benchmark Performance

* **Dataset Release (`emotion_22_v2`)**: 4,997 authentic public Sri Lankan Facebook posts, human-annotated across 22 fine-grained emotional categories:
  * **Train**: 3,486 posts (69.8%)
  * **Validation**: 756 posts (15.1%)
  * **Test**: 755 posts (15.1%)
* **Zero-Leakage Guarantee**: Group-aware text stratification ensures identical reposted text strings never cross split boundaries.

### Benchmark Metrics on Unseen Test Partition ($N = 755$)

| Metric | Score | Description |
| :--- | :---: | :--- |
| **Macro-AP (Average Precision)** | **0.612** | Mean area under Precision-Recall curve across all 22 emotion heads |
| **Macro-F1 (Unweighted)** | **0.558** | Balanced harmonic mean treating low and high support classes equally |
| **Micro-F1 (Instance-Weighted)** | **0.632** | Global multi-label instance classification effectiveness |
| **Hamming Loss** | **0.122** | Fraction of incorrectly predicted binary emotion labels (lower is better) |
| **Multimodal F1 Gain** | **+14.5%** | Performance lift of multimodal fusion over text-only unimodal baseline |
| **Sarcasm Resolution F1** | **0.608** | Precision: 0.527, Recall: 0.718 under negative emotional context |

---

## ⚡ Quick Start & Setup Guide

### 1. Prerequisites
* **Python 3.10+**
* **Node.js v18+** and **npm**
* **Google Chrome** (for Manifest V3 browser extension)

---

### 2. Python Environment & Unified Dependency Installation

Clone the repository and install all project dependencies using the unified `requirements.txt`:

```bash
# Clone the repository
git clone https://github.com/waruna-2003/Reaction-Fusion.git
cd Reaction-Fusion

# Create and activate virtual environment
python -m venv .venv

# Windows (PowerShell):
.\.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

# Install all dependencies (PyTorch, FastAPI, Scikit-Learn, Pandas, etc.)
pip install -r requirements.txt
```

---

### 3. Launching the System (3 Terminals)

To run the complete live ecosystem, launch the three processes in separate terminal windows:

#### 🔹 Terminal 1: Launch FastAPI ML Inference Engine (Port 8000)
```bash
python scripts/run_plugin_backend.py
```
* **Status**: Live at `http://127.0.0.1:8000`
* **Interactive API Docs (Swagger UI)**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

#### 🔹 Terminal 2: Launch Mock Facebook Backend REST API (Port 4000)
```bash
cd mock-facebook-backend
npm install
npx prisma generate
npm run dev
```
* **REST API**: `http://localhost:4000/api/posts`
* **Interactive Developer Portal**: Open [http://localhost:4000/portal](http://localhost:4000/portal) in your browser to test endpoints visually!
* **Database Visualizer (Prisma Studio)**: Run `npx prisma studio` to inspect the SQLite `dev.db` table rows at `http://localhost:5555`.

#### 🔹 Terminal 3: Launch Mock Facebook React Web App (Port 3000)
```bash
cd "mock facebook"
npm install
npm run dev
```
* **Web App URL**: Open [http://localhost:3000](http://localhost:3000) in Google Chrome.

---

### 4. Install Chrome Extension (Manifest V3)

1. Open Google Chrome and go to `chrome://extensions/`.
2. Toggle on **Developer mode** in the top-right corner.
3. Click **Load unpacked** and select the folder:
   ```
   D:\ReactionFusion-Main\platform\plugin
   ```
4. Visit `http://localhost:3000` in Chrome:
   - Notice the glowing blue **ReactionFusion button** beside each post's header actions (`···`).
   - Click the button to inspect real-time multimodal sentiment classification, confidence scores, bilingual emotion triggers, and sarcasm detection.

---

## 📅 Chronological Git Contribution History

This section documents the step-by-step contributions made by the group members during development:

1. **Project Initialization** (`95230af`):
   - Initialized Git repository, `.gitignore`, and base documentation.
2. **Member 1: K.A.A Dilshan** (`3398d9d`):
   - Ingested 4,997 Facebook posts, created zero-leakage `emotion_22_v2` dataset release, and implemented 5,000-d TF-IDF + 7-d reaction feature extractors.
3. **Member 2: L.W.L Silva** (`1d27c1e`):
   - Implemented PyTorch `MultimodalEmotionNet`, trained model with class-weighted multi-label BCE, calibrated decision thresholds, and produced 22-emotion benchmarks.
4. **Member 3: K.A.P.B Himaranshi** (`3501ab9`):
   - Designed Phase 10 Transparent Sentiment Mapper ($S_{\text{pos}}, S_{\text{neg}}$), integrated sarcasm resolution, and engineered asynchronous FastAPI REST microservice on Port 8000.
5. **Member 4: Tharindu Kothalawala** (`ee2314f`, `d1eb672`, `364dc1a`):
   - Developed Mock Facebook React 19 frontend (Port 3000), Node/Express/Prisma backend with SQLite `dev.db` (Port 4000), and Grammarly-style Chrome Extension with DOM body portal overlays and interactive developer portal.

---

## 🎓 Academic Reference & Thesis Chapter Mapping

* **K.A.A Dilshan**: **Chapter 3** (*Data Acquisition, Corpus Linguistics & Annotation Governance*)
* **L.W.L Silva**: **Chapter 4** (*Multimodal Deep Neural Architecture & Emotion Classification*)
* **K.A.P.B Himaranshi**: **Chapter 5** (*Explainable AI Decision Systems & High-Performance Inference*)
* **Tharindu Kothalawala**: **Chapter 6** (*Platform Architecture, External Social Emulation & Browser Extensions*)

---

*(c) 2026 ReactionFusion Team. All rights reserved.*
