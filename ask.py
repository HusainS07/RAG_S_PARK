# ask.py - FastAPI Backend with Pinecone (Updated HF API)
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import httpx
import traceback
from pinecone import Pinecone

load_dotenv()

app = FastAPI(title="RAG Backend with Pinecone")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment variables
HF_API_KEY = os.getenv("HF_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

print(f"🔑 HF_API_KEY present: {bool(HF_API_KEY)}")
print(f"🔑 OPENROUTER_API_KEY present: {bool(OPENROUTER_API_KEY)}")
print(f"🔑 PINECONE_API_KEY present: {bool(PINECONE_API_KEY)}")

# Initialize Pinecone
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    print(f"✓ Connected to Pinecone index: {PINECONE_INDEX_NAME}")
except Exception as e:
    print(f"❌ Error connecting to Pinecone: {e}")
    index = None

class Query(BaseModel):
    name: str
    email: str
    query: str

async def get_embedding(text: str):
    """Get FREE embedding from Hugging Face - NEW ROUTER API"""
    if not HF_API_KEY:
        raise HTTPException(status_code=500, detail="HF_API_KEY not configured")
    
    # NEW Hugging Face Router endpoints
    endpoints = [
        {
            "url": "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2",
            "payload": {"inputs": text},
            "name": "HF Router (MiniLM)"
        },
        {
            "url": "https://router.huggingface.co/hf-inference/models/BAAI/bge-small-en-v1.5",
            "payload": {"inputs": text},
            "name": "HF Router (BGE-small)"
        },
        {
            "url": "https://router.huggingface.co/hf-inference/models/sentence-transformers/paraphrase-MiniLM-L6-v2",
            "payload": {"inputs": text},
            "name": "HF Router (Paraphrase-MiniLM)"
        }
    ]
    
    last_error = None
    
    for endpoint_config in endpoints:
        try:
            endpoint = endpoint_config["url"]
            payload = endpoint_config["payload"]
            name = endpoint_config["name"]
            print(f"🔄 Trying {name}...")
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    endpoint,
                    headers={
                        "Authorization": f"Bearer {HF_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json=payload
                )
                
                print(f"📡 Response status: {response.status_code}")
                
                if response.status_code == 503:
                    result = response.json()
                    if "estimated_time" in result:
                        wait_time = result["estimated_time"]
                        print(f"⏳ Model loading, estimated time: {wait_time}s")
                        # Try next endpoint instead of failing
                        continue
                
                if response.status_code == 200:
                    embedding = response.json()
                    
                    # Handle different response formats
                    if isinstance(embedding, list):
                        # Direct list of embeddings
                        if len(embedding) > 0:
                            if isinstance(embedding[0], list):
                                # Nested list [[0.1, 0.2, ...]]
                                embedding = embedding[0]
                            elif isinstance(embedding[0], float):
                                # Already flat [0.1, 0.2, ...]
                                pass
                        
                        if len(embedding) > 0:
                            print(f"✓ Got embedding with {len(embedding)} dimensions using {name}")
                            return embedding
                    
                    elif isinstance(embedding, dict):
                        # Check for 'embedding' key
                        if 'embedding' in embedding:
                            result = embedding['embedding']
                            print(f"✓ Got embedding with {len(result)} dimensions using {name}")
                            return result
                    
                    raise ValueError(f"Unexpected embedding format: {type(embedding)}")
                
                last_error = f"{name} - Status {response.status_code}: {response.text[:200]}"
                print(f"⚠️ Failed: {last_error}")
                continue
                    
        except HTTPException:
            raise
        except Exception as e:
            last_error = str(e)
            print(f"⚠️ Error with endpoint: {last_error}")
            continue
    
    raise HTTPException(
        status_code=500,
        detail=f"Embedding failed on all endpoints. Last error: {last_error}"
    )

async def call_openrouter_llm(prompt: str):
    """Call LLM via OpenRouter with fallback models"""
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY not configured")
    
    # Try multiple free models in order
    models = [
        "mistralai/mistral-7b-instruct:free",
        "meta-llama/llama-3.3-70b-instruct:free",
        "google/gemini-2.0-flash-exp:free",
    ]
    
    for model in models:
        try:
            print(f"🤖 Calling LLM with {model}...")
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "http://localhost:8000",
                        "X-Title": "Smart Parking RAG System"
                    },
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 1000,
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
    
    raise HTTPException(
        status_code=503,
        detail="All LLM providers are temporarily unavailable. Please try again in a few moments."
    )

prompt_template = """You are a helpful assistant for a Smart Parking Booking System.
Use the context below to answer the question accurately and concisely.

Context:
{context}

Question: {question}

Instructions:
- Answer ONLY based on the provided context
- If the answer is not in the context, say "I could not find the answer in the provided document."
- Keep answers clear, concise, and helpful
- Use bullet points for step-by-step instructions

Answer:"""

@app.post("/api/ask")
async def ask_question(query_obj: Query):
    try:
        print(f"\n📬 New query from {query_obj.name}: {query_obj.query}")
        
        if not index:
            raise HTTPException(status_code=500, detail="Pinecone index not initialized")
        
        # Step 1: Get query embedding
        query_embedding = await get_embedding(query_obj.query)

        # Step 2: Search Pinecone for similar vectors
        print(f"🔍 Searching Pinecone for relevant documents...")
        search_results = index.query(
            vector=query_embedding,
            top_k=10,
            include_metadata=True
        )
        
        if not search_results.matches:
            return JSONResponse({
                "question": query_obj.query,
                "answer": "I could not find any relevant information to answer your question.",
                "contexts": [],
                "matched": False
            })
        
        # Extract context from top matches
        contexts = [match.metadata.get('text', '') for match in search_results.matches]
        context = "\n\n---\n\n".join(contexts[:4])  # Use top 4 matches
        
        print(f"✓ Found {len(search_results.matches)} relevant documents")
        print(f"📊 Top similarities: {[f'{match.score:.3f}' for match in search_results.matches[:4]]}")

        # Step 3: Build prompt
        prompt = prompt_template.format(context=context, question=query_obj.query)

        # Step 4: Call OpenRouter LLM
        answer = await call_openrouter_llm(prompt)

        if not answer or len(answer) < 10:
            return JSONResponse({
                "question": query_obj.query,
                "answer": "Sorry, I couldn't generate a detailed answer.", 
                "contexts": contexts[:4],
                "matched": False
            })

        print(f"✅ Response sent\n")
        return JSONResponse({
            "question": query_obj.query,
            "answer": answer,
            "contexts": contexts[:4],
            "matched": True
        })

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    stats = index.describe_index_stats() if index else {}
    return {
        "status": "healthy", 
        "vectors_in_index": stats.get('total_vector_count', 0) if stats else 0,
        "embedding_model": "all-MiniLM-L6-v2 (384d)",
        "llm_models": ["Gemini 2.0 Flash", "Llama 3.3 70B", "Mistral 7B"],
        "vector_db": "Pinecone",
        "api_version": "HF Router v2"
    }

@app.get("/health")
async def health_check():
    has_hf_key = bool(HF_API_KEY)
    has_or_key = bool(OPENROUTER_API_KEY)
    has_pc_key = bool(PINECONE_API_KEY)
    
    stats = {}
    if index:
        try:
            stats = index.describe_index_stats()
        except:
            pass
    
    return {
        "status": "healthy",
        "hf_api_key_configured": has_hf_key,
        "openrouter_api_key_configured": has_or_key,
        "pinecone_api_key_configured": has_pc_key,
        "pinecone_connected": bool(index),
        "vectors_in_index": stats.get('total_vector_count', 0) if stats else 0,
        "embedding_model": "all-MiniLM-L6-v2 (FREE)",
        "llm_models": "OpenRouter (FREE)",
        "hf_api_endpoint": "router.huggingface.co (NEW)"
    }