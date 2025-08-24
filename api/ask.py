from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="RAG Backend Serverless")

# Load precomputed embeddings
with open("embeddings.json", "r") as f:
    data = json.load(f)
chunks = data["chunks"]
chunk_embeddings = [np.array(emb) for emb in data["embeddings"]]

# Embedding model
embedder = SentenceTransformer("paraphrase-MiniLM-L3-v2", device="cpu")

# OpenRouter client
client = AsyncOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

class Query(BaseModel):
    name: str
    email: str
    query: str

# Cosine similarity
def cosine_similarity(vec_a, vec_b):
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))

prompt_template = """
You are an AI assistant. Use the following context to answer the question.
Context: {context}
Question: {question}
Answer:
"""
@app.post("/api/ask")
async def ask_question(query_obj: Query):
    try:
        # Embed user query
        query_embedding = embedder.encode(query_obj.query, convert_to_numpy=True)

        # Top 3 similar chunks
        similarities = [
            {"index": i, "similarity": cosine_similarity(query_embedding, emb)}
            for i, emb in enumerate(chunk_embeddings)
        ]
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        top_k = similarities[:4]
        context = "\n".join([chunks[entry["index"]] for entry in top_k])

        # Build prompt
        prompt = prompt_template.format(context=context, question=query_obj.query)

        # Call OpenRouter LLM
        response = await client.chat.completions.create(
            model="deepseek/deepseek-r1:free",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7
        )
        answer = response.choices[0].message.content.strip()

        if not answer or len(answer) < 10:
            return JSONResponse({"answer": "Sorry, no detailed ...so sorry answer found.", "matched": False})

        return JSONResponse({"answer": answer, "matched": True})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
