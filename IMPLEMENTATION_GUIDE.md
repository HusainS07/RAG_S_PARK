# 🧠 RAG System Optimization: Implementation & Architectural Guide

This guide consolidates the design decisions, core pipeline architecture, optimization strategies, and evaluation framework definitions for the **Smart Parking Booking System - RAG Backend**.

---

## 📋 Table of Contents
1. [Core RAG Architecture](#1-core-rag-architecture)
2. [RAG Pipeline Flowchart](#2-rag-pipeline-flowchart)
3. [Optimization Highlights (Problems Solved)](#3-optimization-highlights-problems-solved)
4. [Evaluation Framework Architecture](#4-evaluation-framework-architecture)

---

## 1. Core RAG Architecture

The RAG backend is built as a highly performant, asynchronous system designed to run locally or in resource-constrained environments (like Render).

*   **FastAPI Backend (`ask.py`)**: Handles async requests, CORS, complexity classification, and diagnostic health routing.
*   **Pinecone Cloud Vector DB**: Stores document embeddings. Queries are executed dynamically based on vector distances.
*   **Dual-Layer Embeddings**: Dynamically loads a local `all-MiniLM-L6-v2` transformer model. If unavailable (e.g., during deployment on Render's free tier), it automatically falls back to embedding generation via the **Hugging Face Inference API** to ensure 100% service uptime.
*   **Remote LLM Generation**: Uses OpenRouter (Llama 3.3 70B, Gemini) or Grok (xAI) APIs at low temperature (`0.1`) to ensure highly deterministic and grounded responses.

---

## 2. RAG Pipeline Flowchart

The following flowchart details how a query moves through parameter adaptation, forced context pinning, reranking, deduplication, and generation fallback:

```mermaid
graph TD
    A[User Query] --> B[Complexity Classification]
    B -->|Adaptive K & thresholds| C[Dual-Layer Embeddings]
    C --> D[Pinecone Semantic Search]
    E[Forced Pinning Rules] -->|Direct ID Injection| F[Context Blending]
    D --> F
    F --> G[Cross-Encoder Reranking]
    G --> H[Intent-Based Boosting]
    H --> I[Deduplication & Compression]
    I -->|No context found| J[Default Grounded Silence]
    I -->|Context available| K[Grounded Prompt Synthesis]
    K --> L[Remote LLM API Call]
    L -->|Success| M[Return Grounded Answer]
    L -->|Failure / Rate Limit| N[Extractive Fallback Engine]
    N --> M
```

---

## 3. Optimization Highlights (Problems Solved)

### ⚡ Problem 1: Retrieval Overhead vs. Insufficient Context
*   **The Issue**: Simple queries suffered from unnecessary retrieval latency and noise, while complex queries failed because the retrieved context was too shallow to answer multi-intent policy questions.
*   **The Solution: Adaptive K & Retrieval Ceilings**:
    We implemented a query complexity analyzer. Simple questions retrieve fewer chunks ($K=25$, limit 6, threshold 0.35) for fast response times. Complex queries scan deeper ($K=25$, limit 10, threshold 0.25) to provide the LLM with a comprehensive dataset.

### 🗑️ Problem 2: Token Bloat & Conflicting Duplicate Contexts
*   **The Issue**: Overlapping documentation sections created redundant chunks, causing high token costs, hitting LLM context ceilings, and degrading generation quality (due to conflicting duplicate text).
*   **The Solution: Context Deduplication & Compression**:
    We integrated a text sequence matcher (`SequenceMatcher` threshold = 0.82) to strip duplicate semantic blocks. A compression algorithm is applied to extract only highest-signal sentences, maximizing token efficiency.

### 🎯 Problem 3: Semantic Gaps in Critical Workflow Actions
*   **The Issue**: Semantic vector search frequently missed crucial procedure segments (such as cancellation policies or double payment handling) because the user's natural language queries shared zero token overlap with dense legal/system text.
*   **The Solution: Forced Vector Pinning**:
    We created a rule-based pinning engine (`PINNED_SECTION_RULES`). When specific intents (e.g., booking creation) are detected, the system bypasses semantic distance calculations and directly fetches targeted vector IDs to force-inject the correct policies into the LLM's prompt.

---

## 4. Evaluation Framework Architecture

To keep scoring completely objective, the codebase features two independent evaluation workflows:

### 🏆 A. Quality Scoring (LLM-Judge / Ragas-Style)
Executed via `eval_ragas.py`, this pipeline uses advanced LLMs to evaluate semantic accuracy and grounding across five metrics:

1.  **Faithfulness (NLI-based)**: Classifies generated claims against the retrieved context as *Entailed*, *Contradicted*, or *Not In Context* to identify hallucinations.
2.  **Answer Relevancy**: Assesses how directly and completely the answer addresses the user query.
3.  **Context Recall**: Determines what ratio of the reference ground truth is present in the retrieved context blocks.
4.  **Context Precision**: Evaluates the signal-to-noise ratio in retrieved context blocks.
5.  **Weighted Overall Score**:
    $$\text{Overall Score} = 0.30 \times \text{Faithfulness} + 0.25 \times \text{GT Alignment} + 0.20 \times \text{Answer Relevancy} + 0.15 \times \text{Context Recall} + 0.10 \times \text{Context Precision}$$
6.  **Smart Negative Override**: If a question cannot be answered from the docs, and the generator correctly says "information not found" (with the ground truth confirming this), the evaluator overrides the score to a perfect **`1.0`** so correct silence is not penalized.

### 📊 B. Statistical Validation (No LLM)
Executed via `eval_stats.py`, this provides a fully reproducible baseline without LLM generator bias:

*   **ROUGE-L**: Measures lexical overlap (longest common subsequence) between the generated answer and ground truth.
*   **BERTScore F1**: Measures semantic token-level similarity using contextual embeddings (`distilbert-base-uncased`).
*   **SBERT Cosine Similarity**: Employs sentence embeddings (`all-MiniLM-L6-v2`) to compute cosine similarity for two vectors:
    *   *Answer vs. Ground Truth* (factual alignment)
    *   *Answer vs. Context* (degree of grounding)
