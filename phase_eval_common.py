import json
import os
import re
import statistics
import time
from collections import defaultdict
from datetime import datetime

import httpx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
XAI_API_KEY = os.getenv("XAI_API_KEY")
EVAL_PROVIDER = os.getenv("EVAL_PROVIDER", os.getenv("LLM_PROVIDER", "openrouter")).strip().lower()
EVAL_MODEL = os.getenv("EVAL_MODEL", "meta-llama/llama-3.2-3b-instruct:free")
GROK_EVAL_MODEL = os.getenv("GROK_EVAL_MODEL", "grok-beta").strip()
QUERY_DELAY_SECONDS = 0.0
SCORE_DELAY_SECONDS = 0.0
REQUEST_TIMEOUT_SECONDS = float(os.getenv("PHASE_REQUEST_TIMEOUT_SECONDS", "180"))
MAX_SCORE_RETRIES = int(os.getenv("PHASE_MAX_SCORE_RETRIES", "3"))
EVAL_SKIP_SCORING = os.getenv("EVAL_SKIP_SCORING", "false").strip().lower() in {"1", "true", "yes", "on"}
GROUND_TRUTH_PATH = os.getenv("GROUND_TRUTH_PATH", "phase_ground_truths.json")

# Free LLM models on OpenRouter - multiple fallbacks
OPENROUTER_MODELS = [
    # Meta Llama Tiers
    "meta-llama/llama-3.3-70b-instruct:free",
    "meta-llama/llama-3.2-3b-instruct:free",

    # Qwen Tiers
    "qwen/qwen3-coder:free",
    "qwen/qwen3-next-80b-a3b-instruct:free",

    # Google Gemma & Lyria Tiers
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",

    # OpenAI & Hermes Tiers
    "openai/gpt-oss-120b:free",
    "openai/gpt-oss-20b:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",

    # GLM & Mistral Tiers
    "z-ai/glm-4.5-air:free",
    "cognitivecomputations/dolphin-mistral-24b-venice-edition:free",

    # NVIDIA Tiers
    "nvidia/nemotron-3-super-120b-a12b:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "nvidia/nemotron-nano-12b-v2-vl:free",
    "nvidia/nemotron-nano-9b-v2:free",

    # Liquid & Poolside & Moonshot Tiers
    "liquid/lfm-2.5-1.2b-thinking:free",
    "liquid/lfm-2.5-1.2b-instruct:free",
    "poolside/laguna-m.1:free",
    "poolside/laguna-xs.2:free",
    "moonshotai/kimi-k2.6:free",

    # Router
    "openrouter/free"
]

# Track which model is currently working
_current_model_idx = 0


# Ollama configuration removed


def load_phase_questions(start_index, limit=50):
    with open("expanded_eval_data.json", "r", encoding="utf-8") as f:
        questions = json.load(f)

    selected = questions[start_index:start_index + limit]
    return [
        {
            "number": start_index + idx + 1,
            "question": item["question"],
            "category": item.get("category", "unknown"),
        }
        for idx, item in enumerate(selected)
    ]


def load_ground_truth_map():
    if not os.path.exists(GROUND_TRUTH_PATH):
        print(f"Ground truth file not found: {GROUND_TRUTH_PATH}")
        return {}

    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)

    ground_truths = {}
    for rows in payload.get("phases", {}).values():
        for row in rows:
            ground_truths[row["question"]] = row

    print(f"Loaded {len(ground_truths)} ground truths from {GROUND_TRUTH_PATH}")
    return ground_truths


def make_llm(model=None):
    if EVAL_PROVIDER == "openrouter":
        if not OPENROUTER_API_KEY:
            raise RuntimeError("OPENROUTER_API_KEY not found in .env and EVAL_PROVIDER=openrouter")
        model_name = model or EVAL_MODEL
        return ChatOpenAI(
            model=model_name,
            openai_api_key=OPENROUTER_API_KEY,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0,
            max_tokens=50,
            request_timeout=45,
        )
    elif EVAL_PROVIDER == "grok":
        if not XAI_API_KEY:
            raise RuntimeError("XAI_API_KEY not found in .env and EVAL_PROVIDER=grok")
        model_name = model or GROK_EVAL_MODEL
        return ChatOpenAI(
            model=model_name,
            openai_api_key=XAI_API_KEY,
            openai_api_base="https://api.x.ai/v1",
            temperature=0,
            max_tokens=50,
            request_timeout=45,
        )
    else:
        raise RuntimeError("EVAL_PROVIDER must be `openrouter` or `grok`")


def extract_score(response_text):
    """Extract score from LLM response with multiple patterns."""
    try:
        # Try direct float conversion
        score = float(response_text.strip())
        return min(1.0, max(0.0, score))
    except Exception:
        pass

    # Try multiple regex patterns
    patterns = [
        r"(?:score|rating|result)?[:\s]*([0-1](?:\.\d+)?)",  # Score: 0.8
        r"\b([0-1](?:\.\d+)?)\b",  # Any number between 0-1
        r"(?:^|\s)([0-1](?:\.\d+)?)(?:\s|$)",  # Isolated number
    ]
    
    response_lower = response_text.lower()
    for pattern in patterns:
        match = re.search(pattern, response_lower)
        if match:
            try:
                score = float(match.group(1))
                return min(1.0, max(0.0, score))
            except Exception:
                continue

    # Default to 0.5 if we can't parse
    print(f"   ⚠️ Could not parse score from: {response_text[:100]}")
    return 0.5


def call_score_llm(llm, prompt, metric_name):
    """Call LLM for scoring with model fallback on rate limits."""
    global _current_model_idx
    last_error = None

    if EVAL_PROVIDER == "grok":
        # Grok scoring (no complex model list fallback needed, just standard retries)
        for attempt in range(1, MAX_SCORE_RETRIES + 1):
            try:
                response = llm.invoke(prompt)
                score = extract_score(response.content)
                print(f"   ✓ {metric_name} score: {score:.3f}")
                return score
            except Exception as e:
                last_error = str(e)
                error_msg = str(e)[:120]
                wait_time = SCORE_DELAY_SECONDS * (attempt ** 1.5)
                print(f"   Grok score error on {metric_name}: {error_msg}")
                if attempt < MAX_SCORE_RETRIES:
                    print(f"   Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
        print(f"   ❌ Failed to get {metric_name} from Grok: {last_error}")
        return 0.5
    
    # OpenRouter: Try current model, then fallback to others only on rate limit
    for attempt in range(1, MAX_SCORE_RETRIES + 1):
        try:
            model_name = OPENROUTER_MODELS[_current_model_idx]
            llm = make_llm(model_name)
            response = llm.invoke(prompt)
            score = extract_score(response.content)
            print(f"   ✓ {metric_name} score: {score:.3f}")
            return score
        except Exception as e:
            last_error = str(e)
            error_msg = str(e)[:90]
            is_rate_limit = "429" in error_msg or "rate" in error_msg.lower()
            
            if is_rate_limit:
                # Switch to next model on rate limit
                if _current_model_idx < len(OPENROUTER_MODELS) - 1:
                    _current_model_idx += 1
                    print(f"   Rate limited, switching to {OPENROUTER_MODELS[_current_model_idx]}...")
                    # Retry immediately with next model
                    try:
                        llm = make_llm(OPENROUTER_MODELS[_current_model_idx])
                        response = llm.invoke(prompt)
                        score = extract_score(response.content)
                        print(f"   ✓ {metric_name} score: {score:.3f}")
                        return score
                    except Exception as retry_e:
                        last_error = str(retry_e)
                        continue
                else:
                    print(f"   No more models available, all rate limited")
                    break
            else:
                # Non-rate-limit errors: retry with backoff
                wait_time = SCORE_DELAY_SECONDS * (attempt ** 1.5)
                print(f"   Score error on {metric_name}: {error_msg}")
                if attempt < MAX_SCORE_RETRIES:
                    print(f"   Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)

    print(f"   ❌ Failed to get {metric_name}: {last_error}")
    return 0.5


def query_rag_system(question):
    try:
        response = httpx.post(
            f"{BACKEND_URL}/api/ask",
            json={
                "name": "Phase Eval",
                "email": "phase-eval@test.com",
                "query": question,
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

        if response.status_code == 200:
            data = response.json()
            metadata = data.get("metadata", {})
            generation_fallback = metadata.get("generation_fallback")
            return {
                "answer": data.get("answer", ""),
                "contexts": data.get("contexts", []),
                "matched": data.get("matched", False),
                "metadata": metadata,
                "status": "llm_fallback" if generation_fallback else "success",
            }

        print(f"   API error: {response.status_code} {response.text[:120]}")
        return {
            "answer": "",
            "contexts": [],
            "matched": False,
            "metadata": {},
            "status": f"error_{response.status_code}",
        }
    except Exception as e:
        print(f"   Connection error: {str(e)[:120]}")
        return {
            "answer": "",
            "contexts": [],
            "matched": False,
            "metadata": {},
            "status": "connection_error",
        }


def evaluate_faithfulness(llm, question, answer, contexts):
    """
    Evaluate faithfulness using Natural Language Inference (NLI) approach.
    For each claim in the answer, determine if it is entailed by, contradicted by,
    or missing from the context. Paraphrases and synonyms are treated as entailment.
    """
    if not contexts or all(not context.strip() for context in contexts):
        return 0.3

    context_str = "\n\n---\n\n".join(contexts[:4])
    prompt = f"""You are an evaluator for a Q&A system. Your task is to assess whether the answer's claims are supported by the provided context using Natural Language Inference (NLI) logic.

INSTRUCTIONS - IMPORTANT:
For EACH major claim in the ANSWER, determine whether it is:
(A) ENTAILED — the context logically supports this claim, even if worded differently
(B) CONTRADICTED — the context explicitly contradicts this claim
(C) NOT_IN_CONTEXT — the context does not contain information about this claim

SCORING:
- Paraphrasing, synonyms, and different phrasings of the same fact count as ENTAILED
- For example: "slot is released back to the available pool" and "slot becomes available again" are both ENTAILED for the same concept
- Only score 0.0 for genuine hallucinations (CONTRADICTED) or critical facts missing (NOT_IN_CONTEXT when essential)

CONTEXT:
{context_str[:1200]}

QUESTION: {question}

ANSWER: {answer}

EVALUATION STEPS:
1. Identify the main claims in the answer
2. For each claim, check if context contains similar information even with different wording
3. Determine the entailment score: count(ENTAILED) / count(total_claims)
4. Adjust down only if there are CONTRADICTED claims or critical missing information

Return ONLY a decimal number between 0.0 and 1.0 representing the faithfulness score."""
    return call_score_llm(llm, prompt, "faithfulness")


def evaluate_relevancy(llm, question, answer, ground_truth=None):
    ground_truth_text = f"\nGROUND TRUTH ANSWER:\n{ground_truth}\n" if ground_truth else ""
    prompt = f"""You are an evaluator assessing answer quality. Judge how directly and completely the answer addresses the question.

INSTRUCTIONS:
- Score 1.0 if answer fully and directly answers the question
- Score 0.8-0.9 if answer mostly answers with minor gaps
- Score 0.6-0.7 if answer partially addresses the question
- Score 0.3-0.5 if answer barely addresses the question
- Score 0.0-0.2 if answer is largely irrelevant

QUESTION: {question}
{ground_truth_text}

ANSWER: {answer}

EVALUATION CRITERIA:
1. Does the answer directly address what was asked? (yes/partial/no)
2. Is the answer complete or missing key details? (complete/partial/minimal)
3. Would the user be satisfied with this answer? (yes/maybe/no)

Return ONLY a number between 0 and 1."""
    return call_score_llm(llm, prompt, "answer_relevancy")


def evaluate_ground_truth_alignment(llm, question, answer, ground_truth):
    if not ground_truth:
        return 0.5

    prompt = f"""You are evaluating a RAG answer against a curated ground-truth answer from the product document.

INSTRUCTIONS:
- Score 1.0 if the answer fully matches the ground truth with no contradictions
- Score 0.8-0.9 if it captures most ground-truth facts with minor omissions
- Score 0.6-0.7 if it captures the main idea but misses important details
- Score 0.3-0.5 if it only partially matches or is vague
- Score 0.0-0.2 if it contradicts or ignores the ground truth

QUESTION: {question}

GROUND TRUTH:
{ground_truth}

ANSWER:
{answer}

Return ONLY a number between 0 and 1."""
    return call_score_llm(llm, prompt, "ground_truth_alignment")


def evaluate_context_recall(llm, question, contexts, ground_truth=None):
    if not contexts or all(not context.strip() for context in contexts):
        return 0.0

    context_str = "\n\n---\n\n".join(contexts[:4])
    ground_truth_text = f"\nGROUND TRUTH NEEDED TO ANSWER:\n{ground_truth}\n" if ground_truth else ""
    prompt = f"""You are evaluating whether provided context contains sufficient information to answer a question.

INSTRUCTIONS:
- Score 1.0 if context comprehensively covers all information needed to answer the question
- Score 0.8-0.9 if context has most necessary information with minor gaps
- Score 0.6-0.7 if context covers key aspects but missing some details
- Score 0.3-0.5 if context only partially addresses the question
- Score 0.0-0.2 if context lacks relevant information

QUESTION: {question}
{ground_truth_text}

CONTEXT AVAILABLE:
{context_str[:1200]}

ANALYSIS:
1. What information does the question require?
2. Is that information present in the context?
3. Are there critical gaps or missing details?
4. Can someone answer the question using only this context?

Return ONLY a number between 0 and 1."""
    return call_score_llm(llm, prompt, "context_recall")


def evaluate_context_precision(llm, question, answer, contexts):
    if not contexts or all(not context.strip() for context in contexts):
        return 0.0

    context_str = "\n\n---\n\n".join(contexts[:4])
    prompt = f"""You are evaluating the precision of retrieved context - how many retrieved contexts are actually relevant to answering the question.

INSTRUCTIONS:
- Score 1.0 if almost all retrieved contexts are relevant to the question
- Score 0.8-0.9 if most retrieved contexts are relevant with few irrelevant ones
- Score 0.6-0.7 if majority of contexts are relevant but some noise present
- Score 0.4-0.5 if roughly half the contexts are relevant
- Score 0.2-0.3 if few contexts are relevant, mostly noise
- Score 0.0-0.1 if almost no contexts are relevant

QUESTION: {question}

RETRIEVED CONTEXTS:
{context_str[:1200]}

EVALUATION:
1. How many of these contexts are directly relevant to the question?
2. Are there irrelevant or noisy documents in the retrieval?
3. Is the signal-to-noise ratio good?
4. Would removing low-quality contexts improve the answer?

Return ONLY a number between 0 and 1."""
    return call_score_llm(llm, prompt, "context_precision")


def mean(values):
    return round(sum(values) / len(values), 3) if values else 0.0


def format_duration(seconds):
    seconds = int(seconds)
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {sec}s"
    if minutes:
        return f"{minutes}m {sec}s"
    return f"{sec}s"


def print_progress_line(phase_name, completed, total, start_time, completed_at_start):
    percent = (completed / total) * 100 if total else 0.0
    remaining = max(total - completed, 0)
    elapsed = time.time() - start_time
    completed_this_run = max(completed - completed_at_start, 0)

    if completed_this_run:
        avg_seconds = elapsed / completed_this_run
        eta = avg_seconds * remaining
        eta_text = format_duration(eta)
    else:
        eta_text = "calculating"

    print(
        f"   Progress: {phase_name.upper()} {completed}/{total} "
        f"({percent:.1f}%) | remaining={remaining} | "
        f"elapsed={format_duration(elapsed)} | ETA={eta_text}"
    )


def score_result_item(llm, question, answer, contexts, ground_truth=None):
    faithfulness = evaluate_faithfulness(llm, question, answer, contexts)
    time.sleep(SCORE_DELAY_SECONDS)
    relevancy = evaluate_relevancy(llm, question, answer, ground_truth)
    time.sleep(SCORE_DELAY_SECONDS)
    context_recall = evaluate_context_recall(llm, question, contexts, ground_truth)
    time.sleep(SCORE_DELAY_SECONDS)
    context_precision = evaluate_context_precision(llm, question, answer, contexts)
    scores = {
        "faithfulness": faithfulness,
        "answer_relevancy": relevancy,
        "context_recall": context_recall,
        "context_precision": context_precision,
    }
    if ground_truth:
        time.sleep(SCORE_DELAY_SECONDS)
        scores["ground_truth_alignment"] = evaluate_ground_truth_alignment(
            llm, question, answer, ground_truth
        )

    # WEIGHTED SCORING - prioritizes faithfulness and answer correctness (GT alignment)
    # Weights: faithfulness (0.30) + GT alignment (0.25) + answer relevancy (0.20) +
    #          context recall (0.15) + context precision (0.10)
    weighted_overall = (
        scores.get("faithfulness", 0.5) * 0.30 +
        scores.get("ground_truth_alignment", 0.5) * 0.25 +
        scores.get("answer_relevancy", 0.5) * 0.20 +
        scores.get("context_recall", 0.5) * 0.15 +
        scores.get("context_precision", 0.5) * 0.10
    )
    
    # Fallback to simple average if ground truth not available
    if "ground_truth_alignment" not in scores:
        weighted_overall = (
            scores.get("faithfulness", 0.5) * 0.35 +
            scores.get("answer_relevancy", 0.5) * 0.30 +
            scores.get("context_recall", 0.5) * 0.20 +
            scores.get("context_precision", 0.5) * 0.15
        )
    
    scores["overall"] = weighted_overall
    return scores


def summarize_results(phase_name, results, total_questions, total_time):
    successful = [item for item in results if item["status"] == "success"]
    score_rows = [item for item in successful if "scores" in item]

    scores = {
        "faithfulness": [item["scores"]["faithfulness"] for item in score_rows],
        "answer_relevancy": [item["scores"]["answer_relevancy"] for item in score_rows],
        "context_recall": [item["scores"]["context_recall"] for item in score_rows],
        "context_precision": [item["scores"].get("context_precision", 0.5) for item in score_rows],
        "ground_truth_alignment": [
            item["scores"]["ground_truth_alignment"]
            for item in score_rows
            if "ground_truth_alignment" in item["scores"]
        ],
        "overall": [item["scores"]["overall"] for item in score_rows],
    }

    metadata_rows = [item.get("metadata", {}) for item in results]
    adaptive_k_by_complexity = defaultdict(list)
    for metadata in metadata_rows:
        complexity = metadata.get("complexity", "UNKNOWN")
        adaptive_k_by_complexity[complexity].append(metadata.get("adaptive_k", 0))

    response_times = [item["response_time"] for item in results]
    compression_ratios = [
        item.get("metadata", {}).get("compression_ratio", 0.0)
        for item in results
    ]
    contexts_used = [
        item.get("metadata", {}).get("contexts_used", 0)
        for item in results
    ]

    return {
        "phase": phase_name,
        "total_questions": total_questions,
        "completed_questions": len(results),
        "successful_queries": len(successful),
        "failed_queries": len(results) - len(successful),
        "api_success_rate": round((len(successful) / total_questions) * 100, 2) if total_questions else 0,
        "total_time_seconds": round(total_time, 2),
        "avg_response_time_seconds": mean(response_times),
        "scores": {
            "faithfulness_avg": mean(scores["faithfulness"]),
            "answer_relevancy_avg": mean(scores["answer_relevancy"]),
            "context_recall_avg": mean(scores["context_recall"]),
            "context_precision_avg": mean(scores["context_precision"]),
            "ground_truth_alignment_avg": mean(scores["ground_truth_alignment"]),
            "overall_avg": mean(scores["overall"]),
            "overall_min": round(min(scores["overall"]), 3) if scores["overall"] else 0.0,
            "overall_max": round(max(scores["overall"]), 3) if scores["overall"] else 0.0,
            "overall_std": round(statistics.pstdev(scores["overall"]), 3) if len(scores["overall"]) > 1 else 0.0,
        },
        "retrieval": {
            "adaptive_k_by_complexity": {
                key: {
                    "avg": mean(values),
                    "min": min(values) if values else 0,
                    "max": max(values) if values else 0,
                    "count": len(values),
                }
                for key, values in adaptive_k_by_complexity.items()
            },
            "avg_compression_ratio": mean(compression_ratios),
            "min_compression_ratio": round(min(compression_ratios), 3) if compression_ratios else 0.0,
            "max_compression_ratio": round(max(compression_ratios), 3) if compression_ratios else 0.0,
            "avg_contexts_used": mean(contexts_used),
            "min_contexts_used": min(contexts_used) if contexts_used else 0,
            "max_contexts_used": max(contexts_used) if contexts_used else 0,
        },
    }


def print_summary(summary):
    print("\n" + "=" * 80)
    print(f"{summary['phase'].upper()} PHASE SUMMARY")
    print("=" * 80)
    print(f"Completed: {summary['completed_questions']}/{summary['total_questions']}")
    print(f"API success: {summary['api_success_rate']:.1f}%")
    print(f"Avg response time: {summary['avg_response_time_seconds']:.2f}s")
    print(f"Overall score: {summary['scores']['overall_avg']:.3f}")
    print(f"Faithfulness: {summary['scores']['faithfulness_avg']:.3f}")
    print(f"Relevancy: {summary['scores']['answer_relevancy_avg']:.3f}")
    print(f"Context recall: {summary['scores']['context_recall_avg']:.3f}")
    print(f"Context precision: {summary['scores']['context_precision_avg']:.3f}")
    print(f"Ground truth alignment: {summary['scores']['ground_truth_alignment_avg']:.3f}")
    print(f"Avg compression ratio: {summary['retrieval']['avg_compression_ratio']:.3f}")
    print(f"Avg contexts used: {summary['retrieval']['avg_contexts_used']:.1f}")


def write_outputs(phase_name, payload):
    metrics_path = f"phase_metrics_{phase_name}.json"
    summary_path = f"phase_summary_{phase_name}.txt"

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    summary = payload["summary"]
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"{phase_name.upper()} PHASE SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {payload['generated_at']}\n")
        f.write(f"Completed: {summary['completed_questions']}/{summary['total_questions']}\n")
        f.write(f"API success: {summary['api_success_rate']:.1f}%\n")
        f.write(f"Overall score: {summary['scores']['overall_avg']:.3f}\n")
        f.write(f"Faithfulness: {summary['scores']['faithfulness_avg']:.3f}\n")
        f.write(f"Relevancy: {summary['scores']['answer_relevancy_avg']:.3f}\n")
        f.write(f"Context recall: {summary['scores']['context_recall_avg']:.3f}\n")
        f.write(f"Context precision: {summary['scores']['context_precision_avg']:.3f}\n")
        f.write(f"Ground truth alignment: {summary['scores']['ground_truth_alignment_avg']:.3f}\n")
        f.write(f"Avg compression ratio: {summary['retrieval']['avg_compression_ratio']:.3f}\n")
        f.write(f"Avg contexts used: {summary['retrieval']['avg_contexts_used']:.1f}\n")

    print(f"\nSaved metrics: {metrics_path}")
    print(f"Saved summary: {summary_path}")


def load_progress(progress_path):
    """Load phase progress, ignoring empty or corrupt resume files."""
    if not os.path.exists(progress_path):
        return []

    try:
        if os.path.getsize(progress_path) == 0:
            print(f"Progress file {progress_path} is empty; starting fresh.")
            return []

        with open(progress_path, "r", encoding="utf-8") as f:
            progress = json.load(f)

        if not isinstance(progress, list):
            print(f"Progress file {progress_path} is not a JSON list; starting fresh.")
            return []

        return progress
    except json.JSONDecodeError as e:
        print(f"Could not parse {progress_path}: {e}; starting fresh.")
        return []


def save_json_atomic(path, payload):
    """Write JSON through a temp file so interrupted runs do not leave 0-byte progress."""
    temp_path = f"{path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp_path, path)


def run_phase(phase_name, start_index, limit=50):
    phase_questions = load_phase_questions(start_index, limit)
    ground_truths = load_ground_truth_map()
    progress_path = f"phase_progress_{phase_name}.json"

    results = load_progress(progress_path)
    if results:
        completed_questions = {item["question"] for item in results if "question" in item}
        print(f"Resuming {phase_name}: {len(results)} completed questions loaded")
    else:
        completed_questions = set()
    result_by_question = {item.get("question"): item for item in results}

    llm = None if EVAL_SKIP_SCORING else make_llm()
    start_time = time.time()
    completed_at_start = len(results)

    print("\n" + "=" * 80)
    print(f"STARTING {phase_name.upper()} PHASE ({len(phase_questions)} questions)")
    print("=" * 80)
    print(f"Backend: {BACKEND_URL}")
    print(f"Eval provider: {EVAL_PROVIDER}")
    print(f"Eval model: {GROK_EVAL_MODEL if EVAL_PROVIDER == 'grok' else EVAL_MODEL}")
    print(f"Scoring enabled: {not EVAL_SKIP_SCORING}")
    print(f"Ground truth file: {GROUND_TRUTH_PATH} ({len(ground_truths)} loaded)")
    print(f"Query delay: {QUERY_DELAY_SECONDS}s | Score delay: {SCORE_DELAY_SECONDS}s")
    print(f"Progress file: {progress_path}")
    print_progress_line(phase_name, len(results), len(phase_questions), start_time, completed_at_start)

    for idx, item in enumerate(phase_questions, 1):
        ground_truth_entry = ground_truths.get(item["question"], {})
        ground_truth = ground_truth_entry.get("ground_truth")
        existing_result = result_by_question.get(item["question"])
        if existing_result:
            needs_scoring = (
                not EVAL_SKIP_SCORING
                and existing_result.get("status") == "success"
                and "scores" not in existing_result
            )
            if not needs_scoring:
                continue

            print(f"\nScoring saved answer for Q{item['number']}: {item['question'][:90]}")
            if ground_truth:
                existing_result["ground_truth"] = ground_truth
                existing_result["ground_truth_source"] = ground_truth_entry.get("source", {})
            existing_result["scores"] = score_result_item(
                llm,
                item["question"],
                existing_result.get("answer", ""),
                existing_result.get("contexts", []),
                ground_truth,
            )
            existing_result.pop("scores_pending", None)
            save_json_atomic(progress_path, results)
            print(
                "   Scores: "
                f"faith={existing_result['scores']['faithfulness']:.2f} | "
                f"rel={existing_result['scores']['answer_relevancy']:.2f} | "
                f"recall={existing_result['scores']['context_recall']:.2f} | "
                f"prec={existing_result['scores']['context_precision']:.2f} | "
                f"gt={existing_result['scores'].get('ground_truth_alignment', 0.0):.2f} | "
                f"overall={existing_result['scores']['overall']:.2f}"
            )
            print(f"   Saved scored progress to {progress_path}")
            print_progress_line(
                phase_name,
                len([result for result in results if "scores" in result]),
                len(phase_questions),
                start_time,
                0,
            )
            continue

        next_completed = len(results) + 1
        percent = (next_completed / len(phase_questions)) * 100 if phase_questions else 0.0
        print(
            f"\n[{next_completed:02d}/{len(phase_questions)} | {percent:.1f}%] "
            f"Q{item['number']} ({item['category']}): {item['question'][:90]}"
        )

        if results:
            print(f"   Waiting {QUERY_DELAY_SECONDS}s before next backend query...")
            time.sleep(QUERY_DELAY_SECONDS)

        query_start = time.time()
        rag_response = query_rag_system(item["question"])
        response_time = time.time() - query_start

        answer = rag_response["answer"]
        contexts = rag_response["contexts"]
        metadata = rag_response["metadata"]
        status = rag_response["status"]

        print(f"   Response time: {response_time:.2f}s")
        print(
            "   Retrieval: "
            f"k={metadata.get('adaptive_k', 0)} | "
            f"complexity={metadata.get('complexity', 'UNKNOWN')} | "
            f"compression={metadata.get('compression_ratio', 0.0):.3f} | "
            f"contexts={metadata.get('contexts_used', len(contexts))}"
        )

        result_item = {
            "question_number": item["number"],
            "question": item["question"],
            "category": item["category"],
            "phase": phase_name,
            "status": status,
            "matched": rag_response["matched"],
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truth,
            "ground_truth_source": ground_truth_entry.get("source", {}),
            "metadata": metadata,
            "response_time": round(response_time, 3),
        }

        if status == "success" and EVAL_SKIP_SCORING:
            result_item["scores_pending"] = True
            print("   Scoring skipped; answer saved for later evaluation.")
        elif status == "success":
            print("   Evaluating quality metrics...")
            result_item["scores"] = score_result_item(llm, item["question"], answer, contexts, ground_truth)
            print(
                "   Scores: "
                f"faith={result_item['scores']['faithfulness']:.2f} | "
                f"rel={result_item['scores']['answer_relevancy']:.2f} | "
                f"recall={result_item['scores']['context_recall']:.2f} | "
                f"prec={result_item['scores']['context_precision']:.2f} | "
                f"gt={result_item['scores'].get('ground_truth_alignment', 0.0):.2f} | "
                f"overall={result_item['scores']['overall']:.2f}"
            )
        else:
            print(f"   Skipping scoring because status={status}")

        results.append(result_item)
        completed_questions.add(item["question"])
        result_by_question[item["question"]] = result_item

        save_json_atomic(progress_path, results)
        print(f"   Saved progress to {progress_path}")
        print_progress_line(
            phase_name,
            len(results),
            len(phase_questions),
            start_time,
            completed_at_start,
        )

    total_time = time.time() - start_time
    summary = summarize_results(phase_name, results, len(phase_questions), total_time)
    payload = {
        "generated_at": datetime.now().isoformat(),
        "phase": phase_name,
        "question_range": {
            "start_index": start_index,
            "end_index_exclusive": start_index + limit,
            "human_range": f"{start_index + 1}-{start_index + len(phase_questions)}",
        },
        "config": {
            "backend_url": BACKEND_URL,
            "eval_provider": EVAL_PROVIDER,
            "eval_model": GROK_EVAL_MODEL if EVAL_PROVIDER == "grok" else EVAL_MODEL,
            "eval_skip_scoring": EVAL_SKIP_SCORING,
            "ground_truth_path": GROUND_TRUTH_PATH,
            "query_delay_seconds": QUERY_DELAY_SECONDS,
            "score_delay_seconds": SCORE_DELAY_SECONDS,
        },
        "summary": summary,
        "results": results,
    }

    print_summary(summary)
    write_outputs(phase_name, payload)
