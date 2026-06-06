# 🧠 Smart Booking System - RAG Backend

A production-ready Retrieval-Augmented Generation (RAG) system for intelligent Q&A about smart parking booking applications. This system integrates the **Pinecone** vector database, a robust **FastAPI** backend, and remote-first LLM orchestration using **OpenRouter** and **Grok (xAI)**.

---

## 🚀 Key Features

- **Advanced Vector Database**: Powered by **Pinecone** for high-efficiency semantic retrieval and adaptive top-K context scanning.
- **FastAPI Core**: Highly performant asynchronous API featuring CORS, detailed health monitoring, and debug endpoints.
- **Remote-First Inference Engine**:
  - **Embeddings**: Dual-layer async embeddings via local SBERT or Hugging Face Inference API fallback.
  - **Generator**: Remote model invocation via OpenRouter (Llama 3.3 70B, Gemini) or Grok (xAI).
- **Robust Evaluation Suite**:
  - Multi-phase evaluations (Simple, Medium, Complex) across 150 curated questions.
  - LLM-Judge scoring (Faithfulness, Relevancy, Recall) and deterministic Statistical scoring (ROUGE-L, BERTScore).

---

## 📊 Evaluation Results

The system is evaluated against a curated benchmark of **150 questions** across three complexity phases.

### 🏆 Quality Evaluation (LLM-Judge / Ragas-Style)

| Metric | Combined | Simple | Medium | Complex |
| :--- | :---: | :---: | :---: | :---: |
| **Questions Scored** | **150 / 150** | 50 / 50 | 50 / 50 | 50 / 50 |
| **Overall Score** | **`0.838`** | `0.826` | `0.850` | `0.838` |
| **Faithfulness** | **`0.818`** | `0.798` | `0.836` | `0.820` |
| **Answer Relevancy** | **`0.888`** | `0.895` | `0.878` | `0.891` |
| **Context Recall** | **`0.736`** | `0.711` | `0.759` | `0.739` |
| **Context Precision** | **`0.915`** | `0.907` | `0.939` | `0.898` |

### 📈 Statistical Evaluation (Deterministic / No LLM)

| Metric | Simple | Medium | Complex |
| :--- | :---: | :---: | :---: |
| **ROUGE-L** | `0.239` | `0.221` | `0.241` |
| **BERTScore F1** | `0.777` | `0.777` | `0.782` |
| **GT Cosine Similarity** | `0.617` | `0.433` | `0.498` |
| **Context Cosine Similarity** | `0.512` | `0.325` | `0.429` |
| **Overall Weighted Score** | **`0.563`** | **`0.467`** | **`0.515`** |

---

## 🏗️ Architecture & Modules

```
  ┌────────────────────────────────────────────────────────┐
  │                      Client App                        │
  └──────────────────────────┬─────────────────────────────┘
                             │ (JSON query)
                             ▼
  ┌────────────────────────────────────────────────────────┐
  │                    FastAPI Backend                     │
  │                       (ask.py)                         │
  └───────┬──────────────────┬───────────────────┬─────────┘
          │                  │                   │
          ▼                  ▼                   ▼
    ┌───────────┐      ┌───────────┐      ┌──────────────┐
    │ Pinecone  │      │HuggingFace│      │  OpenRouter  │
    │ Vector DB │      │Embeddings │      │  or Grok LLM │
    └───────────┘      └───────────┘      └──────────────┘
```

### 📦 Key Code Modules

1. **`ask.py`**: Manages async FastAPI endpoints. Handles hybrid Pinecone search, Cross-Encoder reranking, intent-boosting, context compression, and LLM dispatch with extractive fallback.

2. **`phase_eval_common.py`**: Common orchestration for loading ground truths, calling evaluator models, and compiling NLI-based faithfulness and weighted metrics.

3. **`eval_ragas.py`**: LLM-based quality evaluation (Faithfulness, Relevancy, Recall, Precision) across all three phases.

4. **`eval_stats.py`**: Deterministic statistical evaluation using ROUGE-L, BERTScore, and SBERT Cosine Similarity.

---

## 📋 System Requirements

- **Python**: `3.8+`
- **API Keys** (free tiers available):
  - `HF_API_KEY` — Hugging Face embedding generation
  - `OPENROUTER_API_KEY` — LLM answering and evaluation
  - `XAI_API_KEY` — *(Optional)* Grok-based generation
  - `PINECONE_API_KEY` — Cloud vector search

---

## 🛠️ Installation & Setup

### 1. Clone & Navigate
```bash
cd RAG_LAST
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure `.env` File
```env
# API Access Keys
HF_API_KEY=your_huggingface_key
OPENROUTER_API_KEY=your_openrouter_key
XAI_API_KEY=your_xai_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=spsrag
PINECONE_ENVIRONMENT=us-east-1

# Server Settings
BACKEND_URL=http://localhost:8000

# LLM Provider
LLM_PROVIDER=openrouter   # 'openrouter' or 'grok'
EVAL_PROVIDER=openrouter  # 'openrouter' or 'grok'

# Model Configuration
EVAL_MODEL=meta-llama/llama-3.2-3b-instruct:free
GROK_MODEL=grok-beta
GROK_EVAL_MODEL=grok-beta
```

### 4. Start the Server
```bash
uvicorn ask:app --reload
```

---

## 📡 API Endpoints

### POST `/api/ask`
```json
// Request
{ "name": "Jane Doe", "email": "jane@example.com", "query": "How can I extend an active booking?" }

// Response
{
  "question": "How can I extend an active booking?",
  "answer": "Yes, you can extend your booking if the slot is not reserved...",
  "contexts": ["Users can request a booking extension before their slot checkout time..."],
  "matched": true
}
```

### GET `/health`
Returns status of services, active remote model providers, and vector db connections.

---

## 🧪 Running the Evaluations

```bash
# 1. Start backend server
uvicorn ask:app --reload

# 2. Generate answers across all phases
python eval_phase_simple.py
python eval_phase_medium.py
python eval_phase_complex.py

# 3. Score the answers
python eval_ragas.py   # LLM-Judge (saves to evaluation_outputs/phase_evals/)
python eval_stats.py   # Statistical (saves to evaluation_outputs/statistical/)

# 4. Consolidate results
python combine_phase_metrics.py
```

---

## 📄 License

Licensed under the MIT License.
