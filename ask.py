from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import os
import re
from difflib import SequenceMatcher
from dotenv import load_dotenv
import httpx
import traceback
import hashlib
import time
from collections import OrderedDict
from pinecone import Pinecone

load_dotenv()

app = FastAPI(title="RAG Backend with Pinecone")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------------------------
# Environment variables
# ----------------------------------------------------------------------------
HF_API_KEY = os.getenv("HF_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openrouter").strip().lower()
XAI_API_KEY = os.getenv("XAI_API_KEY")
GROK_MODEL = os.getenv("GROK_MODEL", "grok-beta").strip()
LOCAL_EMBEDDING_MODEL = os.getenv(
    "LOCAL_EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local").strip().lower()

ENABLE_HYBRID = os.getenv("ENABLE_HYBRID", "true").strip().lower() in ("1", "true", "yes", "on")
ENABLE_PINNED = os.getenv("ENABLE_PINNED", "true").strip().lower() in ("1", "true", "yes", "on")
HYBRID_LEXICAL_WEIGHT = float(os.getenv("HYBRID_LEXICAL_WEIGHT", "0.35"))

OPENROUTER_MODELS = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "qwen/qwen3-coder:free",
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "openai/gpt-oss-120b:free",
    "openai/gpt-oss-20b:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
    "z-ai/glm-4.5-air:free",
    "cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "nvidia/nemotron-nano-12b-v2-vl:free",
    "nvidia/nemotron-nano-9b-v2:free",
    "liquid/lfm-2.5-1.2b-thinking:free",
    "liquid/lfm-2.5-1.2b-instruct:free",
    "poolside/laguna-m.1:free",
    "poolside/laguna-xs.2:free",
    "moonshotai/kimi-k2.6:free",
    "openrouter/free"
]

_local_embedding_model = None
_local_embedding_error = None
_reranker_model = None
_reranker_error = None


def _load_reranker_model():
    global _reranker_model, _reranker_error
    if _reranker_model is not None:
        return _reranker_model
    if _reranker_error is not None:
        return None
    try:
        from sentence_transformers import CrossEncoder
        print("Loading local Cross-Encoder reranker...")
        _reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        return _reranker_model
    except Exception as e:
        _reranker_error = str(e)
        print(f"Cross-Encoder unavailable: {_reranker_error}")
        return None


async def run_cross_encoder_rerank(query: str, matches, top_n: int = 20):
    if not matches:
        return []

    model = _load_reranker_model()
    if model is None:
        return matches[:top_n]

    try:
        loop = asyncio.get_running_loop()
        pairs = [[query, (m.metadata or {}).get("text", "")] for m in matches]
        scores = await loop.run_in_executor(
            None,
            lambda: model.predict(pairs, convert_to_numpy=True).tolist()
        )
        scored_matches = list(zip(scores, matches))
        scored_matches.sort(key=lambda x: x[0], reverse=True)
        for score, m in scored_matches:
            m.rerank_score = float(score)
        return [m for _, m in scored_matches[:top_n]]
    except Exception as e:
        print(f"Cross-Encoder reranking failed: {e}")
        return matches[:top_n]


print(f"HF_API_KEY present: {bool(HF_API_KEY)}")
print(f"OPENROUTER_API_KEY present: {bool(OPENROUTER_API_KEY)}")
print(f"XAI_API_KEY present: {bool(XAI_API_KEY)}")
print(f"PINECONE_API_KEY present: {bool(PINECONE_API_KEY)}")
print(f"Embedding provider: {EMBEDDING_PROVIDER}")
print(f"LLM provider: {LLM_PROVIDER} ({GROK_MODEL if LLM_PROVIDER == 'grok' else 'OpenRouter free models'})")
print(f"Hybrid rerank: {ENABLE_HYBRID} | Pinned chunks: {ENABLE_PINNED}")

try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    print(f"Connected to Pinecone index: {PINECONE_INDEX_NAME}")
except Exception as e:
    print(f"Error connecting to Pinecone: {e}")
    index = None


class Query(BaseModel):
    name: str
    email: str
    query: str


# ----------------------------------------------------------------------------
# Embeddings
# ----------------------------------------------------------------------------
def _load_local_embedding_model():
    global _local_embedding_model, _local_embedding_error
    if _local_embedding_model is not None:
        return _local_embedding_model
    if _local_embedding_error is not None:
        return None
    try:
        from sentence_transformers import SentenceTransformer
        print(f"Loading local embedding model: {LOCAL_EMBEDDING_MODEL}")
        _local_embedding_model = SentenceTransformer(LOCAL_EMBEDDING_MODEL)
        return _local_embedding_model
    except Exception as e:
        _local_embedding_error = str(e)
        print(f"Local sentence-transformers unavailable: {_local_embedding_error}")
        return None


async def get_local_embedding(text: str):
    if EMBEDDING_PROVIDER == "hf":
        return None
    model = _load_local_embedding_model()
    if model is None:
        if EMBEDDING_PROVIDER == "local":
            raise HTTPException(
                status_code=500,
                detail=(
                    "EMBEDDING_PROVIDER=local but sentence-transformers is not available. "
                    f"Last error: {_local_embedding_error}"
                )
            )
        return None
    loop = asyncio.get_running_loop()
    embedding = await loop.run_in_executor(
        None,
        lambda: model.encode(text, convert_to_numpy=True).tolist()
    )
    return embedding


async def get_hf_embedding(text: str):
    if not HF_API_KEY:
        raise HTTPException(status_code=500, detail="HF_API_KEY not configured")

    endpoints = [
        {
            "url": "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2",
            "payload": {"inputs": text},
            "name": "HF Router (MiniLM)",
        },
        {
            "url": "https://router.huggingface.co/hf-inference/models/BAAI/bge-small-en-v1.5",
            "payload": {"inputs": text},
            "name": "HF Router (BGE-small)",
        },
        {
            "url": "https://router.huggingface.co/hf-inference/models/sentence-transformers/paraphrase-MiniLM-L6-v2",
            "payload": {"inputs": text},
            "name": "HF Router (Paraphrase-MiniLM)",
        },
    ]

    last_error = None
    for endpoint_config in endpoints:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    endpoint_config["url"],
                    headers={
                        "Authorization": f"Bearer {HF_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json=endpoint_config["payload"],
                )

                if response.status_code == 503:
                    result = response.json()
                    if "estimated_time" in result:
                        continue

                if response.status_code == 200:
                    embedding = response.json()
                    if isinstance(embedding, list):
                        if len(embedding) > 0 and isinstance(embedding[0], list):
                            embedding = embedding[0]
                        if len(embedding) > 0:
                            return embedding
                    elif isinstance(embedding, dict) and "embedding" in embedding:
                        return embedding["embedding"]

                last_error = f"{endpoint_config['name']} - Status {response.status_code}: {response.text[:200]}"
        except Exception as e:
            last_error = str(e)

    raise HTTPException(status_code=500, detail=f"Embedding failed on all endpoints. Last error: {last_error}")


async def get_embedding(text: str):
    local_embedding = await get_local_embedding(text)
    if local_embedding is not None:
        return local_embedding
    return await get_hf_embedding(text)


# ----------------------------------------------------------------------------
# LLM providers
# ----------------------------------------------------------------------------
async def call_openrouter_llm(prompt: str, max_output_tokens: int = 600):
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY not configured")

    for retry_attempt in range(3):
        for model in OPENROUTER_MODELS:
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                            "Content-Type": "application/json",
                            "HTTP-Referer": "http://localhost:8000",
                            "X-Title": "Smart Parking RAG System",
                        },
                        json={
                            "model": model,
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": max_output_tokens,
                            "temperature": 0.1,
                        },
                    )

                    if response.status_code == 429:
                        await asyncio.sleep((2 ** retry_attempt) * 5)
                        continue

                    if response.status_code != 200:
                        continue

                    result = response.json()
                    if "choices" not in result or not result["choices"]:
                        continue

                    answer = result["choices"][0]["message"]["content"].strip()
                    if answer and len(answer) >= 10:
                        return answer
            except Exception:
                continue

    raise HTTPException(
        status_code=503,
        detail="All LLM providers are temporarily unavailable. Please try again in a few moments.",
    )


async def call_grok_llm(prompt: str, max_output_tokens: int = 600):
    if not XAI_API_KEY:
        raise HTTPException(status_code=500, detail="XAI_API_KEY not configured")

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "https://api.x.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {XAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROK_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_output_tokens,
                "temperature": 0.1,
            },
        )

    if response.status_code != 200:
        raise HTTPException(
            status_code=503,
            detail=f"Grok API returned {response.status_code}: {response.text[:300]}",
        )

    result = response.json()
    answer = result["choices"][0]["message"]["content"].strip()
    if not answer:
        raise HTTPException(status_code=503, detail="Grok API returned an empty response")
    return answer


async def call_llm(prompt: str, max_output_tokens: int = 600):
    if LLM_PROVIDER == "openrouter":
        return await call_openrouter_llm(prompt, max_output_tokens=max_output_tokens)
    if LLM_PROVIDER == "grok":
        return await call_grok_llm(prompt, max_output_tokens=max_output_tokens)
    raise HTTPException(status_code=500, detail="Invalid LLM_PROVIDER. Use `openrouter` or `grok`.")


# ----------------------------------------------------------------------------
# Retrieval configuration
# ----------------------------------------------------------------------------
DETAIL_TERMS = [
    "refund", "payment", "pay", "issue", "failed", "fail", "double", "violation",
    "dispute", "modify", "extend", "extension", "confirmation", "confirm",
    "support", "qr", "cancel", "cancellation", "booking", "book", "wallet",
    "slot", "active", "history", "vehicle", "notification", "alert", "email",
    "receipt", "invoice", "penalty", "fine", "overstay", "checkout", "check-in",
    "reschedule", "transfer", "account", "profile", "session", "expire", "expiry",
    "reservation", "parking reservation", "book parking", "reserve parking",
]

MULTI_INTENT_TERMS = [
    " and ", " or ", " also ", " as well ", " what if ", " if ",
    " after ", " before ", " while ", " then ", " plus ",
]


def get_retrieval_config(question: str):
    words = re.findall(r"\w+", question.lower())
    word_count = len(words)
    lower_question = f" {question.lower()} "

    score = 0
    if word_count > 7:
        score += 1
    if word_count > 14:
        score += 1
    if any(term in lower_question for term in MULTI_INTENT_TERMS):
        score += 1

    detail_hits = sum(
        1 for term in DETAIL_TERMS
        if re.search(r"\b" + re.escape(term) + r"\b", lower_question)
    )
    if detail_hits >= 1:
        score += 1
    if detail_hits >= 3:
        score += 1

    if question.count("?") > 1 or "," in question:
        score += 1

    if score <= 1:
        return {
            "complexity": "SIMPLE",
            "adaptive_k": 25,
            "context_limit": 6,
            "similarity_threshold": 0.25,
            "dedup_ratio": 0.82,
            "max_output_tokens": 600,
        }
    if score <= 3:
        return {
            "complexity": "MEDIUM",
            "adaptive_k": 25,
            "context_limit": 8,
            "similarity_threshold": 0.22,
            "dedup_ratio": 0.82,
            "max_output_tokens": 800,
        }
    return {
        "complexity": "COMPLEX",
        "adaptive_k": 25,
        "context_limit": 10,
        "similarity_threshold": 0.18,
        "dedup_ratio": 0.82,
        "max_output_tokens": 1000,
    }


# ----------------------------------------------------------------------------
# Hybrid rerank + pinned mandatory chunks
# ----------------------------------------------------------------------------
_STOPWORDS = {
    "the", "and", "for", "that", "with", "can", "how", "what", "are", "you",
    "your", "from", "this", "have", "will", "does", "did", "was", "were", "a",
    "an", "of", "to", "in", "on", "is", "it", "if", "do", "i", "my", "me",
}

def _tokenize(text: str):
    return [
        w for w in re.findall(r"\w+", (text or "").lower())
        if len(w) > 2 and w not in _STOPWORDS
    ]

def hybrid_rerank(question: str, matches, lexical_weight: float = HYBRID_LEXICAL_WEIGHT):
    if not matches:
        return matches

    q_tokens = set(_tokenize(question))
    if not q_tokens:
        return matches

    dense_scores = [getattr(m, "score", 0.0) or 0.0 for m in matches]
    dmin, dmax = min(dense_scores), max(dense_scores)
    dspan = (dmax - dmin) or 1.0

    reranked = []
    for m in matches:
        text = (m.metadata or {}).get("text", "")
        d_tokens = set(_tokenize(text))
        overlap = len(q_tokens & d_tokens) / (len(q_tokens) or 1)
        dense_norm = ((getattr(m, "score", 0.0) or 0.0) - dmin) / dspan
        blended = (1 - lexical_weight) * dense_norm + lexical_weight * overlap
        reranked.append((blended, m))

    reranked.sort(key=lambda t: t[0], reverse=True)
    return [m for _, m in reranked]


PINNED_SECTION_RULES = [
    {
        "name": "cancellation",
        "query_keywords": ["cancel", "cancellation", "refund", "after i cancel", "cancelled"],
        "section_markers": ["§3.4", "§3.6", "§8.2", "§12.1"],
        "text_keywords": [
            "cancellation email", "cancellation confirmation", "notification",
            "alert", "email is sent", "non-refundable", "refund policy",
        ],
    },
    {
        "name": "payment",
        "query_keywords": ["payment", "pay", "double payment", "payment fails", "failed payment"],
        "section_markers": ["§7.1", "§8.1", "§8.2"],
        "text_keywords": ["payment failed", "double payment", "retry payment", "refund", "wallet"],
    },
    {
        "name": "booking_confirm",
        "query_keywords": ["confirm", "confirming", "before confirming", "check before"],
        "section_markers": ["§3.2", "§7.1", "§8.1", "§10.3"],
        "text_keywords": ["vehicle details", "slot availability", "payment method", "confirmation"],
    },
    {
        "name": "booking_creation",
        "query_keywords": [
            "new booking", "new reservation", "new parking reservation",
            "parking reservation", "make a booking", "make a reservation",
            "how do i book", "how to book", "how do i reserve", "how to reserve",
            "create a booking", "create reservation", "reserve a slot",
            "book a slot", "book a parking", "reserve a parking",
            "find a parking", "find parking", "search parking",
            "book parking", "reserve parking", "make parking booking",
            "parking booking",
        ],
        "section_markers": ["§3.2", "§3.3", "§7.1"],
        "text_keywords": [
            "book page", "browse", "approved lots", "slot grid", "select a slot",
            "booking duration", "confirm booking", "step 1", "step 2", "step 3",
            "finding a parking lot", "selecting and booking", "initiate booking",
            "available slot", "razorpay", "wallet balance",
        ],
    },
]


def select_pinned_matches(question: str, matches, max_pinned: int = 4):
    if not ENABLE_PINNED or not matches:
        return []

    q = f" {question.lower()} "
    pinned = []
    seen_ids = set()

    for rule in PINNED_SECTION_RULES:
        if not any(kw in q for kw in rule["query_keywords"]):
            continue
        for m in matches:
            mid = getattr(m, "id", None) or id(m)
            if mid in seen_ids:
                continue
            text = ((m.metadata or {}).get("text", "") or "").lower()
            if any(mk.lower() in text for mk in rule["section_markers"]) or any(
                tk in text for tk in rule["text_keywords"]
            ):
                pinned.append(m)
                seen_ids.add(mid)
                if len(pinned) >= max_pinned:
                    return pinned
    return pinned


# ----------------------------------------------------------------------------
# Context compression
# ----------------------------------------------------------------------------
def normalize_context(text: str):
    return re.sub(r"\s+", " ", text.lower()).strip()


def compress_contexts(
    matches,
    context_limit: int,
    similarity_threshold: float = 0.30,
    dedup_ratio: float = 0.82,
    pinned_ids=None,
):
    pinned_ids = pinned_ids or set()

    raw_contexts = []
    for match in matches:
        if not match.metadata:
            continue
        text = (match.metadata.get("text", "") or "").strip()
        if not text:
            continue

        match_id = getattr(match, "id", None) or id(match)
        score = getattr(match, "score", 0.0) or 0.0
        is_pinned = match_id in pinned_ids

        if not is_pinned and score < similarity_threshold:
            continue

        raw_contexts.append((text, is_pinned))

    compressed = []
    compressed_norms = []
    duplicates_removed = 0
    score_filtered = sum(
        1 for m in matches
        if m.metadata and (m.metadata.get("text", "") or "").strip()
        and (getattr(m, "score", 0.0) or 0.0) < similarity_threshold
        and (getattr(m, "id", None) or id(m)) not in pinned_ids
    )

    for context, is_pinned in raw_contexts:
        normalized = normalize_context(context)
        is_duplicate = any(
            normalized == existing
            or SequenceMatcher(None, normalized, existing).ratio() >= dedup_ratio
            for existing in compressed_norms
        )
        if is_duplicate and not is_pinned:
            duplicates_removed += 1
            continue

        compressed.append(context)
        compressed_norms.append(normalized)
        if len(compressed) >= context_limit:
            break

    raw_chars = sum(len(c) for c, _ in raw_contexts)
    compressed_chars = sum(len(c) for c in compressed)
    compression_ratio = round(compressed_chars / raw_chars, 3) if raw_chars else 0.0

    return {
        "contexts": compressed,
        "raw_context_count": len(raw_contexts),
        "contexts_used": len(compressed),
        "duplicates_removed": duplicates_removed,
        "score_filtered": score_filtered,
        "compression_ratio": compression_ratio,
    }


# ----------------------------------------------------------------------------
# Extractive fallback
# ----------------------------------------------------------------------------
def build_extractive_fallback_answer(question: str, contexts):
    question_terms = {
        word
        for word in re.findall(r"\w+", question.lower())
        if len(word) > 2 and word not in _STOPWORDS
    }
    active_booking_query = any(
        term in question.lower()
        for term in ["active", "already started", "after booking start", "in-session", "during"]
    )

    candidates = []
    for context in contexts:
        snippets = re.split(r"(?<=[.!?])\s+|\n+", context.strip())
        for snippet in snippets:
            cleaned = re.sub(r"\s+", " ", snippet).strip(" -•")
            if len(cleaned) < 30:
                continue

            lowered = cleaned.lower()
            overlap = sum(1 for term in question_terms if term in lowered)
            policy_boost = 0
            if active_booking_query and (
                "after booking start" in lowered
                or "non-refundable once active" in lowered
                or "valid booking is active" in lowered
            ):
                policy_boost += 5
            if "cancel" in lowered or "cancellation" in lowered:
                policy_boost += 2
            if "refund" in lowered or "non-refundable" in lowered:
                policy_boost += 2
            if "profile > my bookings" in lowered or "cancel booking" in lowered:
                policy_boost += 1

            score = overlap + policy_boost
            if score > 0:
                candidates.append((score, cleaned))

    if not candidates:
        return (
            "I could not generate an LLM response, but relevant documents were retrieved. "
            + " ".join(contexts[:2])[:700]
        )

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = []
    seen = []
    for _, snippet in candidates:
        normalized = normalize_context(snippet)
        if any(SequenceMatcher(None, normalized, old).ratio() >= 0.88 for old in seen):
            continue
        selected.append(snippet)
        seen.append(normalized)
        if len(selected) >= 4:
            break

    if active_booking_query:
        intro = "Based on the retrieved documentation, once a booking is active it is non-refundable."
    else:
        intro = "Based on the retrieved documentation:"

    bullets = "\n".join(f"- {snippet}" for snippet in selected)
    return f"{intro}\n{bullets}"


prompt_template = """You are an expert assistant for a Smart Parking Booking System.
Answer the user's question STRICTLY and ONLY from the CONTEXT below.

ABSOLUTE RULES (follow exactly):

1. MATCH BY INTENT, NOT EXACT WORDS.
Question words may differ from context wording. Known equivalents:
- "remove booking" = "cancel booking"
- "delete booking" = "cancel booking"
- "deadline to cancel" = "cancellation policy time windows"
- "time limit" = "time before booking start"
- "cut-off" = "cancellation deadline"
Always use the context's own terminology in your answer.

2. IF THE QUESTION NAMES A SPECIFIC CONDITION OR SCENARIO, answer that specific
case first and directly before anything else.

3. USE ONLY FACTS EXPLICITLY STATED IN THE CONTEXT. Your training knowledge is FORBIDDEN.
  - If a detail is not explicitly stated, do not mention it.
  - Prefer a shorter answer over an inferred answer.
  - Do not add extra workflow steps, technical implementation details that arent needed from users point of view.

4. IF THE CONTEXT CONTAINS A TABLE, TIER LIST, FEE SCHEDULE, OR STEP LIST,
reproduce EVERY row/tier/step with its EXACT values.

5. IF AFTER CHECKING INTENT THE ANSWER IS GENUINELY NOT IN THE CONTEXT, reply
EXACTLY: "I could not find this information in the available documents."

6. WHERE IS / WHERE CAN I FIND questions: name the exact location using the
context's own words.
 Before generating the answer:
  - Identify the exact sentences in the context that support the answer.
  - Every statement in the answer must be supported by at least one context sentence.
  - Remove any statement that lacks direct support.

7. CANCELLED BOOKINGS REMAIN AS RECORDS.

8. If the question has multiple parts, answer each part separately.
9. Use bullet points for steps or lists.
10. Keep the tone professional and helpful and concise.


CONTEXT (Source Documents):
{context}

QUESTION: {question}

GROUNDED ANSWER (only from CONTEXT above):"""


def classify_intent_and_boost_matches(question: str, matches):
    if not matches:
        return matches

    q = question.lower()
    is_workflow_query = any(trig in q for trig in ["can i", "allowed", "possible", "unable", "how to", "how do i", "steps"])
    is_policy_query = any(trig in q for trig in ["policy", "refund", "fee", "penalty", "fine", "limit"])

    boosted = []
    for m in matches:
        text = ((m.metadata or {}).get("text", "") or "").lower()
        score = getattr(m, "score", 0.0) or 0.0

        if is_workflow_query:
            if "status reference" in text or "active" in text or "cancelled" in text or "completed" in text or "§7.2" in text or "transition" in text:
                score *= 1.15
        elif is_policy_query:
            if "policy" in text or "refund" in text or "fee" in text or "§8.1" in text:
                score *= 1.10

        m.score = score
        boosted.append(m)

    boosted.sort(key=lambda x: getattr(x, "score", 0.0) or 0.0, reverse=True)
    return boosted


# ----------------------------------------------------------------------------
# Answer Cache
# ----------------------------------------------------------------------------
_ANSWER_CACHE = OrderedDict()
CACHE_EXPIRATION_SECONDS = 3600
CACHE_MAX_SIZE = 100


def _cache_key(question: str) -> str:
    return hashlib.sha256(question.strip().lower().encode("utf-8")).hexdigest()


def get_cached_answer(question: str):
    key = _cache_key(question)
    if key in _ANSWER_CACHE:
        timestamp, response = _ANSWER_CACHE[key]
        if time.time() - timestamp < CACHE_EXPIRATION_SECONDS:
            _ANSWER_CACHE.move_to_end(key)
            return response
        del _ANSWER_CACHE[key]
    return None


def set_cached_answer(question: str, response: dict):
    key = _cache_key(question)
    if key in _ANSWER_CACHE:
        del _ANSWER_CACHE[key]
    _ANSWER_CACHE[key] = (time.time(), response)
    if len(_ANSWER_CACHE) > CACHE_MAX_SIZE:
        _ANSWER_CACHE.popitem(last=False)


# ----------------------------------------------------------------------------
# Async Pinned Chunk Fetcher
# ----------------------------------------------------------------------------
async def fetch_pinned_chunks_async(query: str):
    if not index:
        return []
    DIRECT_FETCH_IDS = {
        "booking_creation": [
            "8d53aba1-cf18-47bf-9800-308e6f791e74",
            "bac4c357-ced5-406f-882d-bcc202e1c04e",
            "0de2bf85-abf5-4ef7-b023-85b496712708",
            "ad13f32e-c446-4d81-8742-86f1ba191a33",
            "1a9f3987-8002-47f0-884e-51bcbc89724f",
            "1fc306c8-1791-4ac1-93be-edf24f733581",
            "7ea8c413-da23-4e48-b740-c375943f23b3",
        ],
    }
    q_lower = f" {query.lower()} "
    direct_fetched = []
    for rule in PINNED_SECTION_RULES:
        fetch_ids = DIRECT_FETCH_IDS.get(rule["name"])
        if not fetch_ids:
            continue
        if not any(kw in q_lower for kw in rule["query_keywords"]):
            continue
        try:
            loop = asyncio.get_running_loop()
            fetched = await loop.run_in_executor(None, lambda: index.fetch(ids=fetch_ids))
            if fetched and hasattr(fetched, "vectors"):
                for vid, vec in fetched.vectors.items():
                    class _PinnedMatch:
                        pass
                    pm = _PinnedMatch()
                    pm.id = vid
                    pm.score = 1.0
                    pm.metadata = vec.metadata if hasattr(vec, "metadata") else {}
                    pm.rerank_score = None
                    direct_fetched.append(pm)
        except Exception as e:
            print(f"Direct chunk fetch failed: {e}")
        break
    return direct_fetched


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------
# Toggle console printing of pipeline timings
PRINT_TIMINGS = True


@app.post("/api/ask")
async def ask_question(query_obj: Query):
    try:
        t_total_start = time.time()
        timings = {}
        print(f"\nNew query from {query_obj.name}: {query_obj.query}")

        cached = get_cached_answer(query_obj.query)
        if cached:
            print("  -> Cache HIT (0.00s)")
            return JSONResponse(cached)

        if not index:
            raise HTTPException(status_code=500, detail="Pinecone index not initialized")

        # Step 1: Start Embedding task and Direct Fetch task concurrently
        t0 = time.time()
        embedding_task = asyncio.create_task(get_embedding(query_obj.query))
        fetch_task = asyncio.create_task(fetch_pinned_chunks_async(query_obj.query))

        query_embedding = await embedding_task
        timings["1_embedding"] = round(time.time() - t0, 3)

        retrieval_config = get_retrieval_config(query_obj.query)

        # Step 2: Pinecone search
        t0 = time.time()
        loop = asyncio.get_running_loop()
        search_results = await loop.run_in_executor(
            None,
            lambda: index.query(
                vector=query_embedding,
                top_k=retrieval_config["adaptive_k"],
                include_metadata=True,
            )
        )
        matches = list(search_results.matches) if search_results.matches else []
        timings["2_pinecone_search"] = round(time.time() - t0, 3)

        # Step 3: Wait for Direct fetch task
        t0 = time.time()
        direct_fetched = await fetch_task
        if direct_fetched:
            fetched_ids = {m.id for m in direct_fetched}
            matches = [m for m in matches if getattr(m, "id", None) not in fetched_ids]
            matches = direct_fetched + matches
        timings["3_direct_fetch"] = round(time.time() - t0, 3)

        metadata = {
            "adaptive_k": retrieval_config["adaptive_k"],
            "complexity": retrieval_config["complexity"],
            "context_limit": retrieval_config["context_limit"],
            "similarity_threshold": retrieval_config["similarity_threshold"],
            "dedup_ratio": retrieval_config["dedup_ratio"],
            "max_output_tokens": retrieval_config["max_output_tokens"],
            "retrieved_contexts": len(matches),
            "contexts_used": 0,
            "raw_context_count": 0,
            "duplicates_removed": 0,
            "score_filtered": 0,
            "compression_ratio": 0.0,
            "hybrid_enabled": ENABLE_HYBRID,
            "pinned_enabled": ENABLE_PINNED,
            "pinned_count": 0,
            "top_similarities": [round(m.score, 3) for m in matches[:4]] if matches else [],
            "embedding_provider": "local" if _local_embedding_model is not None else "hf",
            "generation_provider": LLM_PROVIDER,
            "generation_fallback": None,
        }

        if not matches:
            response_data = {
                "question": query_obj.query,
                "answer": "I could not find any relevant information to answer your question.",
                "contexts": [],
                "matched": False,
                "metadata": metadata,
            }
            set_cached_answer(query_obj.query, response_data)
            return JSONResponse(response_data)

        # Step 4: Skip hybrid rerank (disabled - direct semantic pass)
        t0 = time.time()
        ranked = matches
        timings["4_hybrid_rerank"] = 0.0

        # Step 5: Cohere rerank (conditional)
        t0 = time.time()
        ranked = await maybe_rerank(query_obj.query, ranked)
        timings["5_cross_encoder"] = round(time.time() - t0, 3)

        # Step 6: Intent boost
        t0 = time.time()
        ranked = classify_intent_and_boost_matches(query_obj.query, ranked)
        timings["6_intent_boost"] = round(time.time() - t0, 3)

        # Step 7: Pinned selection
        t0 = time.time()
        pinned_matches = select_pinned_matches(query_obj.query, ranked)
        pinned_ids = {getattr(m, "id", None) or id(m) for m in pinned_matches}
        metadata["pinned_count"] = len(pinned_matches)

        if pinned_matches:
            ranked = pinned_matches + [
                m for m in ranked
                if (getattr(m, "id", None) or id(m)) not in pinned_ids
            ]
        timings["7_pinned_selection"] = round(time.time() - t0, 3)

        # Step 8: Dedup & compression
        t0 = time.time()
        compressed_context = compress_contexts(
            ranked,
            context_limit=retrieval_config["context_limit"],
            similarity_threshold=retrieval_config["similarity_threshold"],
            dedup_ratio=retrieval_config["dedup_ratio"],
            pinned_ids=pinned_ids,
        )

        contexts = compressed_context["contexts"]
        context = "\n\n---\n\n".join(contexts)
        metadata.update({
            "contexts_used": compressed_context["contexts_used"],
            "raw_context_count": compressed_context["raw_context_count"],
            "duplicates_removed": compressed_context["duplicates_removed"],
            "score_filtered": compressed_context["score_filtered"],
            "compression_ratio": compressed_context["compression_ratio"],
        })
        timings["8_dedup_compress"] = round(time.time() - t0, 3)

        if not contexts:
            response_data = {
                "question": query_obj.query,
                "answer": "I could not find this information in the available documents.",
                "contexts": [],
                "matched": False,
                "metadata": metadata,
            }
            set_cached_answer(query_obj.query, response_data)
            return JSONResponse(response_data)

        prompt = prompt_template.format(context=context, question=query_obj.query)

        # Step 9: LLM generation
        t0 = time.time()
        try:
            answer = await call_llm(
                prompt,
                max_output_tokens=retrieval_config["max_output_tokens"],
            )
        except Exception as e:
            metadata["generation_fallback"] = "extractive"
            metadata["generation_error"] = str(e)[:300]
            answer = build_extractive_fallback_answer(query_obj.query, contexts)
        timings["9_llm_generation"] = round(time.time() - t0, 3)

        if not answer or len(answer.strip()) < 5:
            summary = context[:800] if len(context) > 800 else context
            answer = (
                "I found relevant documentation but could not generate a detailed response. "
                f"Here's the relevant information: {summary}"
            )

        timings["TOTAL"] = round(time.time() - t_total_start, 3)

        if PRINT_TIMINGS:
            print("\n  ┌─────────────────────────────────────────────┐")
            print("  │         PIPELINE TIMING BREAKDOWN           │")
            print("  ├──────────────────────────┬──────────────────┤")
            for step, elapsed in timings.items():
                label = step.ljust(24)
                bar = "█" * int(min(elapsed / (timings["TOTAL"] or 1) * 20, 20))
                print(f"  │ {label} │ {elapsed:>7.3f}s {bar}")
            print("  └──────────────────────────┴──────────────────┘")

        metadata["timings"] = timings

        response_data = {
            "question": query_obj.query,
            "answer": answer,
            "contexts": contexts,
            "matched": True,
            "metadata": metadata,
        }
        set_cached_answer(query_obj.query, response_data)
        return JSONResponse(response_data)

    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    stats = index.describe_index_stats() if index else {}
    return {
        "status": "healthy",
        "vectors_in_index": stats.get("total_vector_count", 0) if stats else 0,
        "embedding_model": "all-MiniLM-L6-v2 (384d)",
        "llm_provider": LLM_PROVIDER,
        "llm_model": GROK_MODEL if LLM_PROVIDER == "grok" else OPENROUTER_MODELS[0],
        "vector_db": "Pinecone",
        "hybrid_rerank": ENABLE_HYBRID,
        "pinned_chunks": ENABLE_PINNED,
        "api_version": "HF Router v2",
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
        except Exception:
            pass

    return {
        "status": "healthy",
        "hf_api_key_configured": has_hf_key,
        "openrouter_api_key_configured": has_or_key,
        "pinecone_api_key_configured": has_pc_key,
        "pinecone_connected": bool(index),
        "vectors_in_index": stats.get("total_vector_count", 0) if stats else 0,
        "embedding_model": "all-MiniLM-L6-v2 (FREE)",
        "llm_provider": LLM_PROVIDER,
        "llm_model": GROK_MODEL if LLM_PROVIDER == "grok" else OPENROUTER_MODELS[0],
        "hybrid_rerank": ENABLE_HYBRID,
        "pinned_chunks": ENABLE_PINNED,
        "hf_api_endpoint": "router.huggingface.co (NEW)",
    }
