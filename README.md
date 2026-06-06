# 🧠 Smart Booking System - RAG Backend

A production-ready Retrieval-Augmented Generation (RAG) system for intelligent Q&A about smart parking booking applications. This system integrates the **Pinecone** vector database, a robust **FastAPI** backend, and remote-first LLM orchestration using **OpenRouter** and **Grok (xAI)**.

The entire local Ollama-based configuration has been deprecated and successfully migrated to high-performance, remote API-based models (Gemini 2.5/3.5, Grok-Beta) to guarantee flawless context reasoning, exact detail recall from complex tables, and high system availability.

---

## 🚀 Key Features

- **Advanced Vector Database**: Powered by **Pinecone** for high-efficiency semantic retrieval and adaptive top-K context scanning.
- **FastAPI Core**: Highly performant asynchronous API featuring CORS, detailed health monitoring, and system debug endpoints.
- **Remote-First Inference Engine**:
  - **Embeddings**: Dual-layer async embeddings fetched dynamically via Hugging Face Inference API.
  - **Generator**: Orchestrated remote model invocation utilizing OpenRouter free models (Gemini 3.5 Flash) or Grok via standard OpenAI-compatible connectors.
- **Robust Evaluation Suite**:
  - Resumable multi-phase evaluations spanning Simple, Medium, and Complex question pools.
  - Weighted overall performance scoring combining Faithfulness, Answer Relevancy, Context Recall, and Grounding alignment.
  - Custom LLM evaluator powered by Remote APIs.

---

## 📋 System Requirements

- **Python**: `3.8+`
- **Environment API Keys** (All free tiers available):
  - `HF_API_KEY`: For Hugging Face embedding generation.
  - `OPENROUTER_API_KEY`: For default LLM answering and evaluation scoring.
  - `XAI_API_KEY`: (Optional) To enable Grok-based generation and evaluation.
  - `PINECONE_API_KEY`: For cloud-based vector search.

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
Create a `.env` file in the project root containing your actual API credentials:

```env
# API Access Keys
HF_API_KEY=your_huggingface_key
OPENROUTER_API_KEY=your_openrouter_key
XAI_API_KEY=your_xai_key # Optional for Grok
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=spsrag
PINECONE_ENVIRONMENT=us-east-1

# Server Settings
BACKEND_URL=http://localhost:8000

# LLM Providers Configuration
LLM_PROVIDER=openrouter  # Supported options: 'openrouter' or 'grok'
EVAL_PROVIDER=openrouter # Supported options: 'openrouter' or 'grok'

# Model Configuration
EVAL_MODEL=meta-llama/llama-3.2-3b-instruct:free
GROK_MODEL=grok-beta
GROK_EVAL_MODEL=grok-beta
```

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

1. **`ask.py`**:
   - Manages asynchronous FastAPI endpoints (`/api/ask`, `/health`, `/`).
   - Handles hybrid semantic/lexical Pinecone search, Cross-Encoder reranking, metadata intent-boosting, context compression, and dynamic fallback.
   - Dispatches grounding prompt tasks to OpenRouter or Grok APIs.

2. **`phase_eval_common.py`**:
   - Contains common orchestration functions for loading ground truths, calling evaluator models, and compiling metrics.
   - Implements Natural Language Inference (NLI) faithfulness scoring and a weighted metric evaluation suite.

3. **`eval_ragas.py`**:
   - Runs LLM-based quality evaluation across Simple, Medium, and Complex question pools, checking faithfulness, answer relevancy, context precision, context recall, and ground truth alignment.

4. **`eval_stats.py`**:
   - Performs unbiased lexical and semantic verification using ROUGE-L, BERTScore, and SBERT Cosine Similarity.

---

## 📡 API Endpoints

### 1. POST `/api/ask`
Submit queries to retrieve context-grounded answers.

* **Request Payload**:
  ```json
  {
    "name": "Jane Doe",
    "email": "jane@example.com",
    "query": "How can I extend an active booking?"
  }
  ```

* **Response Payload**:
  ```json
  {
    "question": "How can I extend an active booking?",
    "answer": "Yes, you can extend your booking if the slot is not reserved by another user. Go to 'Active Bookings' in the app and select 'Extend'...",
    "contexts": [
      "Users can request a booking extension before their slot checkout time, provided no subsequent reservation is booked..."
    ],
    "matched": true,
    "source": "rag"
  }
  ```

### 2. GET `/health`
Returns the status of services, active remote model providers, and vector db connections.

---

## 🧪 Running the Evaluations

```bash
# 1. Start backend server
uvicorn ask:app --reload

# 2. Query the RAG system to generate answers (results saved to evaluation_outputs/backend_raw/):
python eval_phase_simple.py
python eval_phase_medium.py
python eval_phase_complex.py

# 3. Score the answers:
# LLM Ragas-style evaluation (saves to evaluation_outputs/phase_evals/):
python eval_ragas.py
# Statistical evaluation (saves to evaluation_outputs/statistical/):
python eval_stats.py

# 4. Consolidate results:
python combine_phase_metrics.py
```

---

## 📄 License

Licensed under the MIT License.
