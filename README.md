# Smart Booking System - RAG Backend

A production-ready Retrieval-Augmented Generation (RAG) system for intelligent Q&A about smart parking booking applications. This system uses Pinecone vector database, FastAPI backend, and free AI models (Hugging Face + OpenRouter) to provide accurate, context-aware answers to user queries about parking booking operations.

## 🚀 Key Features

- **Vector Database Integration**: Pinecone vector database for efficient semantic search and retrieval
- **FastAPI Backend**: High-performance RESTful API with CORS support for web/mobile integration
- **Intelligent RAG Pipeline**: Retrieval-Augmented Generation combining document context with LLM inference
- **Free AI Models**: 
  - **Embeddings**: Hugging Face Router API (sentence-transformers, BGE, paraphrase models)
  - **LLM**: OpenRouter API with fallback to free open-source models
- **Robust Fallback Mechanisms**: Multiple embedding and LLM endpoints for high availability
- **Comprehensive Evaluation Framework**: 
  - RAGAS metrics (Faithfulness, Answer Relevancy, Context Recall)
  - Custom LLM-based evaluation with GPT-3.5 Turbo via OpenRouter
  - 30-question test dataset for performance assessment
- **Cloud-Ready Deployment**: Pre-configured for Render.com with environment variable support
- **Performance Metrics**: Track system performance with detailed evaluation reports

## 📋 Requirements

- **Python**: 3.8 or higher
- **API Keys** (all free tiers available):
  - Hugging Face API key (for embeddings)
  - OpenRouter API key (for LLM inference)
  - Pinecone API key (for vector database)
- **Services**: Pinecone cloud account with active index

## 🛠️ Installation & Setup

### 1. Clone/Navigate to Project

```bash
cd RAG_LAST
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Key dependencies:**
- `fastapi` & `uvicorn`: Web server framework
- `pinecone-client`: Vector database integration
- `httpx`: Async HTTP client for API calls
- `python-dotenv`: Environment variable management
- `ragas`: RAGAS evaluation metrics
- `datasets`: Data handling for evaluation
- `requests`: HTTP requests library

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```env
# API Keys
HF_API_KEY=your_huggingface_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_index_name

# Backend Configuration
BACKEND_URL=http://localhost:8000
```

**How to get API keys:**

| Service | URL | Notes |
|---------|-----|-------|
| Hugging Face | https://huggingface.co/settings/tokens | Free tier includes API access for inference |
| OpenRouter | https://openrouter.ai/keys | Free tier has rate limits but sufficient for testing |
| Pinecone | https://console.pinecone.io | Free tier includes 1 index with limited storage |

## 🏗️ Architecture & Components

### System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   User/Frontend                         │
└────────────────┬────────────────────────────────────────┘
                 │
        ┌────────▼─────────────┐
        │   FastAPI Backend    │
        │    (ask.py)          │
        └────────┬─────────────┘
                 │
      ┌──────────┴──────────┐
      │                     │
 ┌────▼────────┐  ┌────────▼──────┐
 │   Pinecone   │  │   OpenRouter   │
 │   (Vector DB)│  │   LLM Server   │
 └──────────────┘  └────────────────┘
      │
 ┌────▼────────────────────────────┐
 │  Hugging Face Router API         │
 │  (Embeddings)                    │
 └─────────────────────────────────┘
```

### Core Modules

#### `ask.py` - Main FastAPI Application
- **Purpose**: RESTful API server for RAG queries
- **Key Components**:
  - `Query` model: Request schema (name, email, query)
  - `get_embedding()`: Async function to fetch embeddings from HF Router API with fallbacks
  - `call_openrouter_llm()`: LLM inference with multiple model fallbacks
  - `/api/ask`: POST endpoint for answering questions
  - `/health`: GET endpoint for system status
  - Pinecone integration for semantic search

#### `eval.py` - RAGAS Evaluation
- **Purpose**: Automated evaluation using RAGAS metrics
- **Features**:
  - Measures: Faithfulness, Answer Relevancy, Context Recall
  - Progress tracking and resumable evaluation
  - Retry logic with exponential backoff
  - Rate-limit handling
  - Generates `eval_progress.json` and `eval_results.json`

#### `RAG_LAST_EVAL.py` - Comprehensive LLM-Based Evaluation
- **Purpose**: Advanced evaluation using GPT-3.5 Turbo via OpenRouter
- **Components**:
  - 30-question test dataset (comprehensive parking booking scenarios)
  - Metrics: Answer Relevancy, Faithfulness, Harmfulness, Groundedness
  - Detailed per-question scoring
  - Summary statistics and performance analysis
  - Generates timestamped evaluation reports
- **Test Scenarios Cover**:
  - Booking cancellation
  - Finding active bookings
  - Payment issues & refunds
  - Car location
  - Parking violations
  - QR codes & confirmations

#### Configuration Files

- **`requirements.txt`**: Python package dependencies
- **`render.yaml`**: Deployment configuration for Render.com
- **`.env`**: Environment variables (not in repo, user-created)

## � Running the Application

### Start the FastAPI Server

```bash
uvicorn ask:app --reload
```

**Output Example:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete
✓ Connected to Pinecone index: parking-booking
🔑 HF_API_KEY present: True
🔑 OPENROUTER_API_KEY present: True
```

The API will be available at:
- **Main API**: `http://localhost:8000`
- **OpenAPI Docs**: `http://localhost:8000/docs`
- **Health Check**: `http://localhost:8000/health`

### Run RAGAS Evaluation

```bash
python eval.py
```

**Features**:
- Evaluates the RAG system using RAGAS framework
- Measures: Faithfulness, Answer Relevancy, Context Recall
- Progress is saved to `eval_progress.json` (resumable)
- Final results saved to `eval_results.json`
- Built-in retry logic and rate-limit handling

### Run Comprehensive LLM Evaluation

```bash
python RAG_LAST_EVAL.py
```

**Features**:
- Comprehensive 30-question evaluation
- Uses GPT-3.5 Turbo for scoring
- Metrics: Answer Relevancy, Faithfulness, Harmfulness, Groundedness
- Generates timestamped report: `rag_evaluation_[timestamp].json`
- Summary output: `rag_summary_[timestamp].txt`
- Overall performance score with detailed breakdown

## 📡 API Endpoints

### POST `/api/ask` - Query the RAG System

Ask questions about the smart parking booking system.

**Request:**
```bash
curl -X POST "http://localhost:8000/api/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "John Doe",
    "email": "john@example.com",
    "query": "How do I cancel a parking booking?"
  }'
```

**Request Schema:**
```json
{
  "name": "string",           // User name
  "email": "string",          // User email
  "query": "string"           // Question about parking booking
}
```

**Response Schema:**
```json
{
  "question": "string",       // Echo of the question
  "answer": "string",         // LLM-generated answer with context
  "contexts": ["string"],     // Retrieved context chunks from Pinecone
  "matched": true,            // Whether relevant context was found
  "source": "rag"             // Response source
}
```

**Response Example:**
```json
{
  "question": "How do I cancel a parking booking?",
  "answer": "To cancel a parking booking: 1) Open the app 2) Navigate to 'My Bookings' 3) Select the active booking 4) Click 'Cancel' 5) Confirm the cancellation. You will receive a confirmation notification...",
  "contexts": [
    "To cancel a parking booking, users can navigate to the 'My Bookings' section in the app...",
    "Cancellation confirmation is sent immediately to the registered email address..."
  ],
  "matched": true
}
```

### GET `/` - Root Endpoint

Basic health check.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0"
}
```

### GET `/health` - System Status

Detailed system health and configuration check.

**Response:**
```json
{
  "status": "healthy",
  "embedding_model": "Available",
  "llm_model": "Available",
  "database": "connected",
  "api_keys": {
    "HF_API_KEY": true,
    "OPENROUTER_API_KEY": true,
    "PINECONE_API_KEY": true
  }
}
```

Detailed system health information including API key status and loaded data.

## 🧪 Evaluation

Evaluate the system's performance using RAGAS metrics:

1. **Prepare evaluation data**:
   - Create `eval_data.json` with question-answer pairs
   - Format:
   ```json
   [
     {
       "question": "How can I cancel an existing parking booking?",
       "ground_truth": "Expected answer here..."
     }
   ]
   ```

2. **Run evaluation**:
   ```bash
   # Start the API server first
   uvicorn ask:app

   # In another terminal
   python eval.py
   ```

   This will evaluate faithfulness, answer relevancy, and context recall.

## 🚀 Deployment

### Render.com Deployment

1. **Connect your repository** to Render.com
2. **Use the provided `render.yaml`** configuration
3. **Set environment variables** in Render dashboard:
   - `HF_API_KEY`
   - `OPENROUTER_API_KEY`
4. **Deploy** the service

The service will be available at your Render URL.

## 🏗️ Architecture

```
PDF Document
    ↓
Text Extraction (pypdf)
    ↓
Text Chunking (LangChain)
    ↓
Embedding Generation (Sentence Transformers)
    ↓
embeddings.json
    ↓
FastAPI Server
    ↓
Query Processing → Retrieval → LLM Generation
    ↓
JSON Response
```

## 🔧 Configuration

### Embedding Model
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Dimensions**: 384
- **Processing**: 100% local (no API required)

### LLM Models (Free Tier)
- Primary: `google/gemini-2.0-flash-exp:free`
- Fallbacks: `meta-llama/llama-3.2-3b-instruct:free`, `nousresearch/hermes-3-llama-3.1-405b:free`

### Retrieval
- **Similarity**: Cosine similarity
- **Top-K**: 4 most similar chunks
- **Chunk Size**: 500 characters
- **Overlap**: 100 characters

## 🐛 Troubleshooting

### Common Issues

1. **"HF_API_KEY not configured"**
   - Ensure `.env` file exists with correct API key
   - Check key validity on Hugging Face

2. **"Model is loading" (503 error)**
   - Hugging Face models may take time to load on first use
   - Wait and retry the request

3. **"All LLM providers are temporarily unavailable"**
   - Free tier rate limits exceeded
   - Wait a few minutes before retrying

4. **Empty embeddings.json**
   - Ensure PDF file exists and is readable
   - Check PDF contains extractable text

### Performance Tips

- **Local Processing**: Embedding generation is fast and free
- **Rate Limits**: Free APIs have usage limits; consider upgrading for production
- **Chunk Size**: Adjust chunk_size in `preprocess_offline.py` for different document types

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For questions or issues:
- Check the troubleshooting section
- Review API documentation
- Open an issue on GitHub

---

**Built with**: FastAPI, Sentence Transformers, LangChain, RAGAS, Hugging Face, OpenRouter</content>
<parameter name="filePath">d:\VJTI\DEGRE\SEM_4\SE\SMART_BOOKING_SYSTEM\RAG_LAST\README.md



Test 

Example Setup
Question

Q: What is the capital of France?

Ground Truth

GT: The capital of France is Paris.

Retrieved Context
Doc 1: Paris is the capital and largest city of France.
Doc 2: France is a country in Western Europe.
Doc 3: Berlin is the capital of Germany.

Case 1: Good RAG Answer ✅
Generated Answer

A: The capital of France is Paris.

Metric-wise Evaluation
1️⃣ Answer Correctness → HIGH

Matches ground truth fact

No extra or missing info

✔ Correct

2️⃣ Faithfulness → HIGH

Answer supported by Doc 1

No hallucination

✔ Faithful

3️⃣ Answer Relevancy → HIGH

Directly answers the question

✔ Relevant

4️⃣ Context Recall → HIGH

Required info (“Paris is capital”) exists in context

✔ Retrieved what was needed

5️⃣ Context Precision → MEDIUM

Doc 3 (Berlin) is irrelevant noise

⚠ Some unnecessary context

Case 2: Hallucinated Answer ❌
Generated Answer

A: The capital of France is Paris and it has a population of 10 million.

Evaluation
Answer Correctness → LOW

Population info not in ground truth

❌ Extra fact

Faithfulness → LOW

Population not supported by context

❌ Hallucination

Answer Relevancy → MEDIUM

Partially relevant

Context Recall → HIGH

Capital info was retrieved

Context Precision → MEDIUM

Extra irrelevant docs

Case 3: Wrong Answer ❌
Generated Answer

A: The capital of France is Berlin.

Evaluation
Metric	Result	Why
Answer Correctness	❌ Low	Factually wrong
Faithfulness	❌ Low	Berlin supported only for Germany
Answer Relevancy	❌ Low	Wrong answer
Context Recall	✅ High	Correct info was present
Context Precision	❌ Low	Wrong doc influenced answer
Case 4: Correct but Unfaithful ⚠️
Generated Answer

A: The capital of France is Paris.

Retrieved Context
Doc 1: France is in Europe.
Doc 2: Paris is a city in France.

Evaluation
Answer Correctness → HIGH

Matches ground truth

Faithfulness → LOW

Context never says “capital”

⚠ Correct guess, but not grounded

One-line Summary (VERY IMPORTANT)
Metric	Checks
Answer Correctness	Matches ground truth
Faithfulness	Supported by context
Answer Relevancy	Answers the question
Context Recall	Context has required info
Context Precision	Context is not noisy