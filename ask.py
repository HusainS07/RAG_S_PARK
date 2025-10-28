# ask.py - Fixed with Hugging Face InferenceClient
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import os
from dotenv import load_dotenv
import httpx
import traceback
from huggingface_hub import InferenceClient

load_dotenv()

app = FastAPI(title="RAG Backend Serverless")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load precomputed embeddings
print("📦 Loading embeddings...")
try:
    with open("embeddings.json", "r") as f:
        data = json.load(f)
    chunks = data["chunks"]
    chunk_embeddings = data["embeddings"]
    print(f"✓ Loaded {len(chunks)} chunks")
except Exception as e:
    print(f"❌ Error loading embeddings: {e}")
    chunks = []
    chunk_embeddings = []

HF_API_KEY = os.getenv("HF_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

print(f"🔑 HF_API_KEY present: {bool(HF_API_KEY)}")
print(f"🔑 OPENROUTER_API_KEY present: {bool(OPENROUTER_API_KEY)}")

# Initialize Hugging Face InferenceClient
hf_client = InferenceClient(token=HF_API_KEY) if HF_API_KEY else None

class Query(BaseModel):
    name: str
    email: str
    query: str

def cosine_similarity(vec_a, vec_b):
    """Pure Python cosine similarity"""
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a ** 2 for a in vec_a) ** 0.5
    norm_b = sum(b ** 2 for b in vec_b) ** 0.5
    return dot_product / (norm_a * norm_b)

async def get_embedding(text: str):
    """Get FREE embedding from Hugging Face using InferenceClient"""
    if not hf_client:
        raise HTTPException(status_code=500, detail="HF_API_KEY not configured")
    
    try:
        print(f"🔄 Getting embedding for: {text[:50]}...")
        
        # Use InferenceClient's feature_extraction method
        embedding = hf_client.feature_extraction(
            text=text,
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Convert to list if it's a numpy array or tensor
        if hasattr(embedding, 'tolist'):
            embedding = embedding.tolist()
        
        # Handle nested lists
        if isinstance(embedding, list) and len(embedding) > 0:
            if isinstance(embedding[0], list):
                embedding = embedding[0]
        
        print(f"✓ Got embedding with {len(embedding)} dimensions")
        return embedding
            
    except Exception as e:
        print(f"❌ Embedding error: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")

async def call_free_llm(prompt: str):
    """Call FREE LLM via OpenRouter with fallback models"""
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY not configured")
    
    # Try multiple free models in order
    models = [
        "meta-llama/llama-3.2-3b-instruct:free",
        "nousresearch/hermes-3-llama-3.1-405b:free",
        "google/gemini-2.0-flash-exp:free"
    ]
    
    for model in models:
        try:
            print(f"🤖 Calling LLM with {model}...")
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "http://localhost:8000",
                        "X-Title": "RAG System"
                    },
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 500,
                        "temperature": 0.7
                    }
                )
                
                print(f"📡 OpenRouter Response status: {response.status_code}")
                
                if response.status_code == 429:
                    print(f"⚠️ Rate limited on {model}, trying next model...")
                    continue
                
                if response.status_code != 200:
                    error_detail = response.text
                    print(f"❌ LLM API Error: {error_detail}")
                    continue
                
                result = response.json()
                answer = result["choices"][0]["message"]["content"].strip()
                print(f"✓ Got LLM response from {model}: {answer[:100]}...")
                return answer
                
        except Exception as e:
            print(f"❌ Error with {model}: {str(e)}")
            continue
    
    # If all models failed
    raise HTTPException(
        status_code=503,
        detail="All LLM providers are temporarily unavailable. Please try again in a few moments."
    )

prompt_template = """You are a helpful AI assistant. Use the context below to answer the question accurately.

Context:
{context}

Question: {question}

Answer:"""

@app.post("/api/ask")
async def ask_question(query_obj: Query):
    try:
        print(f"\n📬 New query from {query_obj.name}: {query_obj.query}")
        
        if not chunks or not chunk_embeddings:
            raise HTTPException(status_code=500, detail="Embeddings not loaded")
        
        # Get query embedding
        query_embedding = await get_embedding(query_obj.query)

        # Find top 4 similar chunks
        print(f"🔍 Finding similar chunks...")
        similarities = [
            {"index": i, "similarity": cosine_similarity(query_embedding, emb)}
            for i, emb in enumerate(chunk_embeddings)
        ]
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        top_k = similarities[:4]
        
        print(f"✓ Top similarities: {[f'{s['similarity']:.3f}' for s in top_k]}")
        
        context = "\n\n".join([chunks[entry["index"]] for entry in top_k])

        # Build prompt
        prompt = prompt_template.format(context=context, question=query_obj.query)

        # Call FREE LLM
        answer = await call_free_llm(prompt)

        if not answer or len(answer) < 10:
            return JSONResponse({
                "answer": "Sorry, I couldn't generate a detailed answer.", 
                "matched": False
            })

        print(f"✅ Response sent\n")
        return JSONResponse({"answer": answer, "matched": True})

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {
        "status": "healthy", 
        "chunks": len(chunks),
        "embedding_model": "all-MiniLM-L6-v2 (384d)",
        "llm_model": "Gemini 2.0 Flash (FREE)"
    }

@app.get("/health")
async def health_check():
    has_hf_key = bool(HF_API_KEY)
    has_or_key = bool(OPENROUTER_API_KEY)
    return {
        "status": "healthy",
        "hf_api_key_configured": has_hf_key,
        "openrouter_api_key_configured": has_or_key,
        "chunks_loaded": len(chunks),
        "embedding_dimensions": len(chunk_embeddings[0]) if chunk_embeddings else 0,
        "embedding_model": "all-MiniLM-L6-v2 (FREE)",
        "llm_model": "Gemini 2.0 Flash (FREE)"
    }