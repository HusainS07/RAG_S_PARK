from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Memory-Efficient RAG Backend")

# Load precomputed embeddings
with open("embeddings.json", "r") as f:
    data = json.load(f)
chunks = data["chunks"]
chunk_embeddings = [np.array(emb) for emb in data["embeddings"]]

# Initialize embedding model (for queries only)
embedder = SentenceTransformer("paraphrase-MiniLM-L3-v2")

# Initialize OpenRouter client
client = AsyncOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

class Query(BaseModel):
    name: str
    email: str
    query: str

# Cosine similarity function
def cosine_similarity(vec_a, vec_b):
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    return dot_product / (norm_a * norm_b)

# Prompt template
prompt_template = """
You are an AI assistant. Use the following context to answer the question.
Context: {context}
Question: {question}
Answer:
"""

@app.get("/")
def health():
    return {"status": "ok", "message": "RAG backend is running!"}

@app.post("/ask")
async def ask_question(data: Query):
    try:
        # Embed query
        query_embedding = embedder.encode(data.query, convert_to_numpy=True)

        # Find top 3 similar chunks
        similarities = [
            {"index": i, "similarity": cosine_similarity(query_embedding, emb)}
            for i, emb in enumerate(chunk_embeddings)
        ]
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        top_k = similarities[:3]
        context = "\n".join([chunks[entry["index"]] for entry in top_k])

        # Build prompt and call OpenRouter
        prompt = prompt_template.format(context=context, question=data.query)
        response = await client.chat.completions.create(
            model="deepseek/deepseek-r1:free",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=350,
            temperature=0.7
        )
        answer = response.choices[0].message.content.strip()

        if not answer or len(answer) < 10:
            return {
                "answer": "Sorry, we couldn't find a detailed answer. Try rephrasing or contact support.",
                "matched": False
            }
        return {"answer": answer, "matched": True}
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Something went wrong. Try again later.")

# Optional: Simple frontend for testing
@app.get("/ui", response_class=HTMLResponse)
async def get_ui():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>RAG Query App</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; }
            label { display: block; margin-top: 10px; }
            input, textarea { width: 100%; padding: 8px; margin-top: 5px; }
            button { padding: 10px 20px; margin-top: 10px; }
            #response { margin-top: 20px; }
        </style>
    </head>
    <body>
        <h1>RAG Query App</h1>
        <form id="queryForm">
            <label>Name:</label>
            <input type="text" id="name" required>
            <label>Email:</label>
            <input type="email" id="email" required>
            <label>Query:</label>
            <textarea id="query" required></textarea>
            <button type="submit">Submit</button>
        </form>
        <div id="response"></div>
        <script>
            document.getElementById("queryForm").addEventListener("submit", async (e) => {
                e.preventDefault();
                const name = document.getElementById("name").value;
                const email = document.getElementById("email").value;
                const query = document.getElementById("query").value;
                const responseDiv = document.getElementById("response");
                responseDiv.innerHTML = "Loading...";
                try {
                    const res = await fetch("/ask", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ name, email, query })
                    });
                    const data = await res.json();
                    responseDiv.innerHTML = `<h2>Response:</h2><p>${data.answer || data.detail}</p><p>Matched: ${data.matched ? "Yes" : "No"}</p>`;
                } catch (error) {
                    responseDiv.innerHTML = `<p>Error: ${error.message}</p>`;
                }
            });
        </script>
    </body>
    </html>
    """