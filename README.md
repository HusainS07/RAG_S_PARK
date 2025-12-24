# Smart Booking System - RAG Backend

A Retrieval-Augmented Generation (RAG) system for intelligent Q&A about smart parking booking applications. This system preprocesses documentation, creates embeddings, and provides a FastAPI-based API for answering user queries using free AI models.

## 🚀 Features

- **Offline Preprocessing**: Generate embeddings locally from PDF documents without API calls
- **FastAPI Backend**: RESTful API with CORS support for web integration
- **Retrieval-Augmented Generation**: Combines document retrieval with LLM generation for accurate answers
- **Free AI Models**: Uses Hugging Face Inference API and OpenRouter for cost-free embeddings and LLM calls
- **Fallback Mechanisms**: Multiple model fallbacks ensure reliability
- **Evaluation Framework**: Built-in RAGAS evaluation for measuring system performance
- **Cloud Deployment Ready**: Configured for Render.com deployment

## 📋 Requirements

- Python 3.8+
- PDF document containing booking system documentation
- API keys for Hugging Face and OpenRouter (free tiers available)

## 🛠️ Installation

1. **Clone the repository** (if applicable) or navigate to the project directory

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file in the root directory:
   ```
   HF_API_KEY=your_huggingface_api_key
   OPENROUTER_API_KEY=your_openrouter_api_key
   ```

   - Get Hugging Face API key from [Hugging Face](https://huggingface.co/settings/tokens)
   - Get OpenRouter API key from [OpenRouter](https://openrouter.ai/keys)

## 📄 Data Preparation

1. **Prepare your PDF document**:
   - Place your documentation PDF as `sodapdf-converted.pdf` in the root directory
   - This should contain information about the smart booking system

2. **Generate embeddings**:
   ```bash
   python preprocess_offline.py
   ```
   This will:
   - Extract text from the PDF
   - Split text into chunks (500 characters with 100 overlap)
   - Generate embeddings using Sentence Transformers locally
   - Save results to `embeddings.json`

## 🚀 Running the Application

### Local Development

Start the FastAPI server:
```bash
uvicorn ask:app --reload
```

The API will be available at `http://localhost:8000`

### Health Check

Visit `http://localhost:8000/health` to check system status and configuration.

## 📡 API Endpoints

### POST `/api/ask`

Ask questions about the booking system.

**Request Body:**
```json
{
  "name": "John Doe",
  "email": "john@example.com",
  "query": "How do I cancel a parking booking?"
}
```

**Response:**
```json
{
  "question": "How do I cancel a parking booking?",
  "answer": "You can cancel a parking booking by opening the app, going to the My Bookings section...",
  "contexts": ["Context chunk 1", "Context chunk 2", ...],
  "matched": true
}
```

### GET `/`

Basic health check endpoint.

### GET `/health`

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