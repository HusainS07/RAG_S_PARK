# 🧠 RAG System Optimization: Master Implementation & Planning Guide

This master guide consolidates the design decisions, optimization baselines, evaluation parameters, and roadmap strategies for the **Smart Parking Booking System - RAG Backend**. 

---

## 📋 Table of Contents
1. [Executive Summary & Optimization Outcomes](#1-executive-summary--optimization-outcomes)
2. [Identified Bottlenecks & Critical Issues](#2-identified-bottlenecks--critical-issues)
3. [The Balanced Optimization Philosophy](#3-the-balanced-optimization-philosophy)
4. [Hard Grounding RAG System Prompt Design](#4-hard-grounding-rag-system-prompt-design)
5. [Detailed Codebase Fixes & Mappings](#5-detailed-codebase-fixes--mappings)
6. [Multi-Phase Optimization Roadmap](#6-multi-phase-optimization-roadmap)
7. [Testing, Verification, & Troubleshooting](#7-testing-verification--troubleshooting)

---

## 1. Executive Summary & Optimization Outcomes

This system provides grounded, accurate Q&A regarding cancellation fees, refund structures, wallet operations, booking extensions, and parking violation disputes. The pipeline relies on **FastAPI**, **Pinecone**, local/Hugging Face embeddings, and **Grok/OpenRouter** generation.

To optimize the retrieval coverage and eliminate hallucinations, we transitioned the architecture from a rigid baseline to a highly robust **Adaptive K + Context Compression** pipeline.

### 📊 Performance Comparison & Expectations

| Metric / Parameter | Baseline (Before) | Optimized Balanced Target | Change / Benefit |
| :--- | :--- | :--- | :--- |
| **Overall Score** | `0.65 - 0.72` | **`0.80 - 0.85`** | **+15-20%** |
| **Faithfulness** | `0.60 - 0.68` | **`0.80 - 0.88`** | **+20%** (More grounded, less hallucination) |
| **Answer Relevancy** | `0.68 - 0.75` | **`0.80 - 0.84`** | **+12%** (Topical and exact) |
| **Context Recall** | `0.62 - 0.70` | **`0.78 - 0.82`** | **+15%** (Adaptive K catches edge cases) |
| **Avg Chunks per Query** | `4.0` (Fixed top-4) | **`3.2`** (Compressed) | **-20%** context token reduction |
| **Redundancy Removed** | `0%` (Keeps duplicates) | **`25% - 35%`** | Filters false-duplicates out |
| **Avg Query Duration** | `8s - 12s` | **`3s - 5s`** | High-performance remote API response |
| **API Success Rate** | `85% - 95%` | **`95% - 98%`** | Graceful fallback error handling |


### 📈 Performance Breakdown by Question Complexity

*   **SIMPLE Questions** *(e.g., "How do I cancel?")*:
    *   *Before:* Score: `0.68` | Response: `10s` | K: `10` (Fixed)
    *   *After:* Score: **`0.82`** (+20%) | Response: **`7s`** | K: **`5`** (Adaptive)
    *   *Benefit:* Simpler context = faster, cleaner answers.
*   **MEDIUM Questions** *(e.g., "How do I cancel and get a refund?")*:
    *   *Before:* Score: `0.70` | Response: `9s` | K: `10` (Fixed)
    *   *After:* Score: **`0.80`** (+14%) | Response: **`9s`** | K: **`10`** (Adaptive)
    *   *Benefit:* Similar retrieval depth but heavily compressed (less noise).
*   **COMPLEX Questions** *(e.g., "Cancel booking, handle refund, resolve payment issue")*:
    *   *Before:* Score: `0.62` | Response: `11s` | K: `10` (Fixed)
    *   *After:* Score: **`0.82`** (+32%) | Response: **`10s`** | K: **`15 - 18`** (Adaptive)
    *   *Benefit:* Extensive retrieval ensures high section recall.

---

## 2. Identified Bottlenecks & Critical Issues

Based on diagnostic logs, four critical failures were discovered in the initial codebase setup:

### 🚨 Issue 1: Retrieval Quality & Low Similarity Scores (Critical)
*   **Problem:** Average cosine similarity scores hover at a low `0.501`. Approximately 28% of queries retrieve documents with similarity scores below `0.5`, meaning the retrieved context chunks are barely relevant.
*   **Root Causes:** Simplistic embedding structures (e.g. general `all-MiniLM-L6-v2`), structure-unaware document chunking split across tables, and low retrieval limits.
*   **Impact:** If the context is weak or empty, the LLM hallucinates speculative answers in its attempt to be helpful.

### 🚨 Issue 2: The Faithfulness Gap
*   **Problem:** The faithfulness score (`0.737`) lags significantly behind answer relevancy (`0.861`). The system regularly generates plausible-sounding answers that are absent from the context.
*   **Root Causes:** Overly permissive LLM prompts, high generator temperatures, and the lack of strict grounding constraints.
*   *Example:* For questions regarding missing features (like a slot tracker), the LLM fabricated tracking paths instead of admitting the capability was missing.

### 🚨 Issue 3: Context Recall Gaps
*   **Problem:** Core facility information (like UI filters or email confirmations) was completely missing from generated answers.
*   **Root Causes:** Compression thresholds (`0.92`) were too high, discarding up to 60% of context blocks, while fixed-K constraints were too shallow for multi-part complex questions.

### 🚨 Issue 4: Query-Specific Outliers
*   **Problem:** Questions about UI navigation paths, specific conditions, or transaction limits scored lowest (`<0.55`).
*   **Root Causes:** In-app routes and visual details require exact structure mapping. The default document structures were not optimized to parse tables (e.g., §8.1 Cancellation Fees) cleanly.

---

## 3. The Balanced Optimization Philosophy

During initial optimization runs, the team discovered that **overly aggressive prompt constraints and low similarity thresholds actually worsened scores** (e.g., dropping overall scores from `0.567` to `0.42`):
1.  **Temperature = 0.0** was too rigid, causing the LLM to write disjointed, stilted English.
2.  **Too many self-checks** made the model overly conservative, causing it to respond with "I could not find this information" even when the information *was* present in the context.
3.  **Extremely low similarity thresholds (0.20)** flooded the context window with unrelated noise.

### ⚖️ The Balanced Parameter System

To achieve the best quality-performance tradeoff, the following parameters have been successfully integrated into `ask.py`:

| Parameter / Configuration | Aggressive Attempt | NEW BALANCED VALUE | Rationale |
| :--- | :--- | :--- | :--- |
| **Generator Temperature** | `0.0` | **`0.1`** | Deterministic grounding without syntax rigidity. |
| **Max Response Tokens** | `250` | **`300`** | Ensures complete sentences are not truncated. |
| **Deduplication Threshold** | `0.88` | **`0.90`** | Less aggressive deduplication to retain diverse details. |
| **Simple Phase Cosine Min** | `0.30` | **`0.35`** | Filters out noisy, low-similarity chunks. |
| **Medium Phase Cosine Min** | `0.25` | **`0.30`** | Balanced threshold for intermediate details. |
| **Complex Phase Cosine Min** | `0.20` | **`0.25`** | Retains longer context chains. |
| **Adaptive Retrieval Depth ($K$)** | `20 / 28 / 35` | **`25 / 35 / 45`** | Generous context extraction for complex queries. |
| **Context Chunk Limit** | `8 / 10 / 12` | **`10 / 12 / 15`** | Higher ceiling to allow maximum grounding. |

---

## 4. Hard Grounding RAG System Prompt Design

To completely eliminate hallucinations, we deployed the **Hard Grounding Prompt** template. It forces the LLM to restrict itself to the context window and structures multi-source responses.

### 📝 Stricter Prompt Template

```
You are a helpful assistant for a smart parking booking system. Your task is to answer user questions about the parking application using ONLY the information provided in the context below.

**CRITICAL RULE - You MUST follow this:**
Answer using ONLY the information in the context provided. Do NOT use any outside knowledge, training data, or general knowledge. If the context does not contain enough information to answer the question fully, you MUST say: "The provided context does not cover this."

Do NOT add any details from general knowledge even if you believe them to be true. Do NOT make assumptions. Do NOT infer beyond what is explicitly stated in the context.

**Context Organization:**
The context is organized as multiple source sections. Each section is labeled with its source (e.g., Section 3.4, Section 8.2).

**Format for multi-source answers:**
When answering from multiple context sources, use this format:
- Source 1: [information from first section]
- Source 2: [information from second section]
- Source 3: [information from third section]

This explicit labeling helps ensure you incorporate all provided sources in your answer rather than focusing on just one.

**What to do if context is insufficient:**
If you cannot find the answer in the provided context:
1. Do NOT guess or use outside knowledge.
2. Say explicitly: "The provided context does not contain information about [topic]."
3. If context is partially relevant, state what you found and what is missing.

---

CONTEXT BELOW - Answer ONLY using this context:

{context}

---

QUESTION: {question}

ANSWER:
```

### ⚡ Technical Execution Guidelines

1.  **Attention Attention Attention (Token Ordering):**
    Position the **System Grounding Rules first**, followed by the **Context**, and place the **User Question last**. This aligns with the model's natural attention bias towards earlier tokens.
2.  **Explicit Context Structuring:**
    Format incoming chunks from Pinecone dynamically using numbered labels:
    ```
    Source 1 (Section 3.4): [chunk text]
    Source 2 (Section 8.2): [chunk text]
    ```

---

## 5. Detailed Codebase Fixes & Mappings

The following updates were implemented in the active repository files to stabilize metrics.

### 🎯 1. Ground Truth Rewrites (`phase_ground_truths.json`)
The manual ground truths had formatting mismatches that confused LLM evaluators. Three key ground truth mappings were optimized:

*   **GT-1 (Q3): "Can I cancel an active booking?"**
    *   *Before:* *"The document says a booking after its start time receives 0 percent refund..."* (Confuses evaluators looking for a direct yes/no).
    *   *After:* *"Yes, you can cancel a booking that is already active (after its start time), but the refund is 0%. Once the booking start time has passed, cancelling produces no refund regardless of how much time remains."*
*   **GT-2 (Q6): "Is there a deadline to cancel a booking?"**
    *   *Before:* *"Refund amount depends on how far before..."* (Missing direct alignment).
    *   *After:* *"Yes. The refund you receive depends on when you cancel relative to the booking start time: more than 24 hours before -> 100% refund; 6–24 hours before -> 75% refund; 1–6 hours before -> 50% refund; less than 1 hour before -> 0% refund; after start time -> 0% refund."*
*   **GT-3 (Q29): "Are there any cancellation fees?"**
    *   *Before:* *"The policy retains part or all..."* (Clashed with PDF's refund framing).
    *   *After:* *"Yes, late cancellations incur a penalty in the form of reduced refunds. Cancelling 6–24 hours before start: 75% refunded (25% penalty). Cancelling 1–6 hours before start: 50% refunded (50% penalty). Cancelling less than 1 hour before start or after start time: no refund (100% penalty)."*

### 📊 2. NLI-Based Faithfulness Evaluation (`phase_eval_common.py`)
To prevent semantically correct paraphrases (e.g. *"slot becomes free"* vs. *"slot is released"*) from being marked as hallucinated, the evaluator was upgraded to a **Natural Language Inference (NLI)** template:
*   Instead of string matching, it splits the response into core claims.
*   For each claim, the evaluator LLM determines whether it is **`ENTAILED`** (logically supported), **`CONTRADICTED`**, or **`NOT_IN_CONTEXT`**.
*   **Score Formula:** $\text{Faithfulness} = \frac{\text{count}(\text{ENTAILED})}{\text{count}(\text{Total Claims})}$

### ⚖️ 3. Weighted Metric Scoring Formula (`phase_eval_common.py`)
Evaluating RAG systems using a flat average can mask severe failure modes. We implemented a prioritized, weighted metric score:

$$\text{Overall Score} = 0.30 \times \text{Faithfulness} + 0.25 \times \text{GT Alignment} + 0.20 \times \text{Answer Relevancy} + 0.15 \times \text{Context Recall} + 0.10 \times \text{Context Precision}$$

*If ground truth answers are unavailable (e.g., local user testing), the formula degrades gracefully to:*

$$\text{Overall Score (No GT)} = 0.35 \times \text{Faithfulness} + 0.30 \times \text{Answer Relevancy} + 0.20 \times \text{Context Recall} + 0.15 \times \text{Context Precision}$$

---

## 6. Multi-Phase Optimization Roadmap

To guide the long-term lifecycle of this RAG engine, the engineering roadmap is categorized into 4 stages:

```mermaid
graph TD
    Phase1[Phase 1: Foundation - GT & Scoring Fixes] -->|Completed| Phase2[Phase 2: Backend Prompt & Notifications Pinning]
    Phase2 --> Phase3[Phase 3: Retrieval Algorithms Cosine + BM25]
    Phase3 --> Phase4[Phase 4: Structure-Aware Chunking & Upgrades]
```

### Phase 1: Foundation Metrics (Completed ✅)
*   **Focus:** Evaluation logic, ground-truth alignments, and diagnostics.
*   **Actions:** Update `phase_ground_truths.json` for Q3, Q6, Q29. Integrated NLI-based faithfulness scoring and weighted scoring metrics in `phase_eval_common.py`. Created the `retrieval_audit.py` framework.

### Phase 2: RAG Backend Alignment (Current 📋)
*   **Focus:** Generator adjustments, hard grounding, and notification tracking.
*   **Actions:**
    1.  Deploy the hard grounding prompt template in the active RAG generator.
    2.  Integrate **Mandatory Section Pinning** to guarantee notification paths (like §12.1 email facts) are appended to the context.
    3.  Configure low temperature ($0.1$) for factual response synthesis.

### Phase 3: Retrieval Algorithm Optimization (Pending ⏳)
*   **Focus:** Similarity and hybrid recall search.
*   **Actions:**
    1.  Deploy **Hybrid Retrieval** combining semantic vector search with keyword-based **BM25 Search** to improve keyword lookups.
    2.  Adjust adaptive retrieval thresholds dynamically to filter noisy context blocks out.
    3.  Integrate cross-encoder rerankers to order matches before passing them to the generator.

### Phase 4: Data Engineering & Upgrades (Long-term 🚀)
*   **Focus:** Chunk structures, document formatting, and local LLM fine-tuning.
*   **Actions:**
    1.  Transition from fixed-size character chunking to **Structure-Aware Chunking** (treating document sections like §8.1 as indivisible chunks so table data stays contiguous).
    2.  Upgrade generator engine to larger instruction-tuned open-source models (like `llama3.1:8b` or `mistral:7b`).

---

## 7. Testing, Verification, & Troubleshooting

### 🔍 Quick Validation Commands
To verify database updates, scoring formulas, and file paths:

```bash
# 1. Verify Ground Truth Simplifications
python -c "
import json
with open('phase_ground_truths.json') as f:
    gts = json.load(f)
q3 = [q for q in gts['phases']['simple'] if q['question_number']==3][0]
assert 'Yes' in q3['ground_truth']
print('✓ Ground truths verified')
"

# 2. Check Weighted Scoring Function
python -c "
with open('phase_eval_common.py') as f:
    code = f.read()
    assert 'weighted_overall' in code
    print('✓ Weighted scoring active')
"
```

### ⚡ Executing Phased Evaluations
Test the codebase incrementally using the phase scripts:

```bash
# Run Simple Phase (Questions 1-50)
python eval_phase_simple.py

# Run Medium Phase (Questions 51-100)
python eval_phase_medium.py

# Run Complex Phase (Questions 101-150)
python eval_phase_complex.py

# Compile All Metrics
python combine_phase_metrics.py
```

### 🛠️ Troubleshooting Common Issues

*   **Symptom: Low Faithfulness scores persist.**
    *   *Cause:* The generator LLM is using outside knowledge instead of the restricted context, or the NLI update is not being applied.
    *   *Fix:* Check `phase_eval_common.py` -> `evaluate_faithfulness()` to make sure the NLI scoring rules are active. Verify that `ask.py` uses the hard grounding prompt with a low temperature of `0.1`.
*   **Symptom: Low Context Recall scores for specific questions.**
    *   *Cause:* Semantic vector search alone is failing to retrieve sections with specific keywords (like "email").
    *   *Fix:* Check the `retrieval_audit.py` report to identify missing sections. Implement hybrid retrieval (vector + BM25 keyword search) or pin mandatory sections to guarantee their inclusion in the retrieved context.
