"""
eval_statistical.py
====================
Alternative RAG evaluation using three fully unbiased methods:

  1. ROUGE-L       — lexical overlap between answer and ground truth
  2. BERTScore     — semantic similarity using contextual embeddings
  3. Cosine Sim    — sentence-transformer embedding similarity (answer vs ground truth,
                     answer vs context)

No LLM judge. No generator bias. Fully reproducible.

Install deps:
    pip install rouge-score bert-score sentence-transformers torch

Usage:
    python eval_statistical.py
"""

import json
import os
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Lazy imports with friendly errors ────────────────────────────────────────

def _require(pkg, install):
    try:
        return __import__(pkg)
    except ImportError:
        raise SystemExit(f"Missing package: `{install}`. Run: pip install {install}")

# ── Constants ─────────────────────────────────────────────────────────────────

PHASES          = ["simple", "medium", "complex"]
SBERT_MODEL     = "all-MiniLM-L6-v2"          # same model already used in your pipeline
BERTSCORE_LANG  = "en"
BERTSCORE_MODEL = "distilbert-base-uncased"    # lightweight; swap to roberta-large for precision

# Weighted overall (no LLM, so no faithfulness — proxy via context cosine)
WEIGHTS = {
    "rouge_l":          0.20,   # lexical match vs ground truth
    "bertscore_f1":     0.30,   # semantic match vs ground truth
    "gt_cosine":        0.25,   # embedding cosine vs ground truth
    "context_cosine":   0.25,   # answer relevance to retrieved context
}


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_progress(phase: str) -> list:
    path = f"phase_progress_{phase}.json"
    if not Path(path).exists():
        print(f"  Skipping {phase}: {path} not found.")
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Metric helpers ────────────────────────────────────────────────────────────

def compute_rouge_l(predictions: list[str], references: list[str]) -> list[float]:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    for pred, ref in zip(predictions, references):
        if not ref.strip():
            scores.append(0.0)
            continue
        result = scorer.score(ref, pred)
        scores.append(round(result["rougeL"].fmeasure, 4))
    return scores


def compute_bertscore(predictions: list[str], references: list[str]) -> list[float]:
    import bert_score
    # Filter out empty references
    valid_idx = [i for i, r in enumerate(references) if r.strip()]
    if not valid_idx:
        return [0.0] * len(predictions)

    valid_preds = [predictions[i] for i in valid_idx]
    valid_refs  = [references[i]  for i in valid_idx]

    _, _, F1 = bert_score.score(
        valid_preds,
        valid_refs,
        lang=BERTSCORE_LANG,
        model_type=BERTSCORE_MODEL,
        verbose=False,
    )
    f1_list = F1.tolist()

    # Re-insert 0.0 for rows with empty reference
    result = [0.0] * len(predictions)
    for out_idx, orig_idx in enumerate(valid_idx):
        result[orig_idx] = round(f1_list[out_idx], 4)
    return result


def compute_cosine_similarity(
    embedder,
    texts_a: list[str],
    texts_b: list[str],
) -> list[float]:
    """Cosine similarity between paired text lists using sentence-transformers."""
    from torch.nn.functional import cosine_similarity
    import torch

    # Filter empty pairs
    scores = []
    for a, b in zip(texts_a, texts_b):
        if not a.strip() or not b.strip():
            scores.append(0.0)
            continue
        emb_a = embedder.encode(a, convert_to_tensor=True)
        emb_b = embedder.encode(b, convert_to_tensor=True)
        sim = cosine_similarity(emb_a.unsqueeze(0), emb_b.unsqueeze(0)).item()
        scores.append(round(max(0.0, sim), 4))
    return scores


def context_to_string(contexts: list) -> str:
    """Flatten context list to a single string for embedding."""
    if not contexts:
        return ""
    return " ".join(c.strip() for c in contexts if isinstance(c, str))[:1500]


# ── Per-phase evaluation ──────────────────────────────────────────────────────

def evaluate_phase(phase: str, embedder) -> dict | None:
    items = load_progress(phase)
    if not items:
        return None

    successful = [
        item for item in items
        if item.get("status") == "success"
        and item.get("answer", "").strip()
        and item.get("ground_truth", "").strip()
    ]

    no_gt = [
        item for item in items
        if item.get("status") == "success"
        and item.get("answer", "").strip()
        and not item.get("ground_truth", "").strip()
    ]

    print(f"\n{'='*60}")
    print(f"Phase: {phase.upper()}")
    print(f"  Total items     : {len(items)}")
    print(f"  With ground truth: {len(successful)}")
    print(f"  Without GT (skipped for GT metrics): {len(no_gt)}")
    print(f"{'='*60}")

    if not successful:
        print("  No items with ground truth — skipping statistical eval.")
        return None

    answers      = [item["answer"]       for item in successful]
    ground_truths = [item["ground_truth"] for item in successful]
    contexts_raw  = [item.get("contexts", []) for item in successful]
    context_strs  = [context_to_string(c) for c in contexts_raw]

    # ── 1. ROUGE-L ────────────────────────────────────────────────────────────
    print("  Computing ROUGE-L ...", end=" ", flush=True)
    rouge_scores = compute_rouge_l(answers, ground_truths)
    print(f"avg={sum(rouge_scores)/len(rouge_scores):.3f}")

    # ── 2. BERTScore ──────────────────────────────────────────────────────────
    print("  Computing BERTScore F1 ...", end=" ", flush=True)
    bert_scores = compute_bertscore(answers, ground_truths)
    print(f"avg={sum(bert_scores)/len(bert_scores):.3f}")

    # ── 3a. Cosine: answer vs ground truth ────────────────────────────────────
    print("  Computing GT cosine similarity ...", end=" ", flush=True)
    gt_cosine = compute_cosine_similarity(embedder, answers, ground_truths)
    print(f"avg={sum(gt_cosine)/len(gt_cosine):.3f}")

    # ── 3b. Cosine: answer vs context ─────────────────────────────────────────
    print("  Computing context cosine similarity ...", end=" ", flush=True)
    ctx_cosine = compute_cosine_similarity(embedder, answers, context_strs)
    print(f"avg={sum(ctx_cosine)/len(ctx_cosine):.3f}")

    # ── Weighted overall ──────────────────────────────────────────────────────
    overall = [
        WEIGHTS["rouge_l"]        * r
        + WEIGHTS["bertscore_f1"] * b
        + WEIGHTS["gt_cosine"]    * g
        + WEIGHTS["context_cosine"] * c
        for r, b, g, c in zip(rouge_scores, bert_scores, gt_cosine, ctx_cosine)
    ]

    # ── Attach scores back to items ───────────────────────────────────────────
    scored_items = []
    gt_iter = iter(range(len(successful)))
    for item in successful:
        i = next(gt_iter)
        item["statistical_scores"] = {
            "rouge_l":        rouge_scores[i],
            "bertscore_f1":   bert_scores[i],
            "gt_cosine":      gt_cosine[i],
            "context_cosine": ctx_cosine[i],
            "overall":        round(overall[i], 4),
        }
        scored_items.append(item)

    def _avg(lst): return round(sum(lst) / len(lst), 4) if lst else 0.0
    def _mn(lst):  return round(min(lst), 4) if lst else 0.0
    def _mx(lst):  return round(max(lst), 4) if lst else 0.0

    summary = {
        "phase":             phase,
        "total_items":       len(items),
        "evaluated_items":   len(successful),
        "skipped_no_gt":     len(no_gt),
        "weights_used":      WEIGHTS,
        "scores": {
            "rouge_l_avg":          _avg(rouge_scores),
            "rouge_l_min":          _mn(rouge_scores),
            "rouge_l_max":          _mx(rouge_scores),
            "bertscore_f1_avg":     _avg(bert_scores),
            "bertscore_f1_min":     _mn(bert_scores),
            "bertscore_f1_max":     _mx(bert_scores),
            "gt_cosine_avg":        _avg(gt_cosine),
            "gt_cosine_min":        _mn(gt_cosine),
            "gt_cosine_max":        _mx(gt_cosine),
            "context_cosine_avg":   _avg(ctx_cosine),
            "context_cosine_min":   _mn(ctx_cosine),
            "context_cosine_max":   _mx(ctx_cosine),
            "overall_avg":          _avg(overall),
            "overall_min":          _mn(overall),
            "overall_max":          _mx(overall),
        },
    }

    _print_summary(summary)
    return {"summary": summary, "results": scored_items}


# ── Pretty print ──────────────────────────────────────────────────────────────

def _print_summary(s: dict):
    sc = s["scores"]
    print(f"\n  ── {s['phase'].upper()} Results ──────────────────────────────")
    print(f"  Evaluated : {s['evaluated_items']} / {s['total_items']}")
    print(f"  ROUGE-L   : avg={sc['rouge_l_avg']:.3f}  "
          f"min={sc['rouge_l_min']:.3f}  max={sc['rouge_l_max']:.3f}")
    print(f"  BERTScore : avg={sc['bertscore_f1_avg']:.3f}  "
          f"min={sc['bertscore_f1_min']:.3f}  max={sc['bertscore_f1_max']:.3f}")
    print(f"  GT Cosine : avg={sc['gt_cosine_avg']:.3f}  "
          f"min={sc['gt_cosine_min']:.3f}  max={sc['gt_cosine_max']:.3f}")
    print(f"  Ctx Cosine: avg={sc['context_cosine_avg']:.3f}  "
          f"min={sc['context_cosine_min']:.3f}  max={sc['context_cosine_max']:.3f}")
    print(f"  ── Overall: avg={sc['overall_avg']:.3f}  "
          f"min={sc['overall_min']:.3f}  max={sc['overall_max']:.3f}")


# ── Output writers ────────────────────────────────────────────────────────────

def write_outputs(phase: str, payload: dict):
    metrics_path = f"phase_statistical_metrics_{phase}.json"
    summary_path = f"phase_statistical_summary_{phase}.txt"
    sc = payload["summary"]["scores"]

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"{phase.upper()} STATISTICAL EVALUATION SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated  : {payload['generated_at']}\n")
        f.write(f"Evaluated  : {payload['summary']['evaluated_items']} "
                f"/ {payload['summary']['total_items']}\n\n")
        f.write(f"ROUGE-L    avg : {sc['rouge_l_avg']:.4f}\n")
        f.write(f"BERTScore  avg : {sc['bertscore_f1_avg']:.4f}\n")
        f.write(f"GT  Cosine avg : {sc['gt_cosine_avg']:.4f}\n")
        f.write(f"Ctx Cosine avg : {sc['context_cosine_avg']:.4f}\n")
        f.write(f"Overall    avg : {sc['overall_avg']:.4f}\n")
        f.write(f"\nWeights used:\n")
        for k, v in payload["summary"]["weights_used"].items():
            f.write(f"  {k:<20}: {v}\n")

    print(f"\n  Saved: {metrics_path}")
    print(f"  Saved: {summary_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Lazy-load heavy deps once
    _require("rouge_score", "rouge-score")
    _require("bert_score",  "bert-score")
    _require("sentence_transformers", "sentence-transformers")

    from sentence_transformers import SentenceTransformer
    print(f"Loading sentence-transformer model: {SBERT_MODEL} ...")
    embedder = SentenceTransformer(SBERT_MODEL)
    print("Model loaded.\n")

    all_results = {}

    for phase in PHASES:
        result = evaluate_phase(phase, embedder)
        if result is None:
            continue

        payload = {
            "generated_at": datetime.now().isoformat(),
            "eval_method":  "statistical (ROUGE-L + BERTScore + Cosine Similarity)",
            "llm_judge":    "none",
            "summary":      result["summary"],
            "results":      result["results"],
        }
        write_outputs(phase, payload)
        all_results[phase] = result["summary"]["scores"]

    # ── Cross-phase comparison ─────────────────────────────────────────────────
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("CROSS-PHASE COMPARISON")
        print(f"{'='*60}")
        header = f"{'Metric':<22}" + "".join(f"{p.upper():>12}" for p in all_results)
        print(header)
        print("-" * len(header))
        metrics = [
            ("ROUGE-L avg",        "rouge_l_avg"),
            ("BERTScore F1 avg",   "bertscore_f1_avg"),
            ("GT Cosine avg",      "gt_cosine_avg"),
            ("Context Cosine avg", "context_cosine_avg"),
            ("Overall avg",        "overall_avg"),
        ]
        for label, key in metrics:
            row = f"{label:<22}"
            for phase_scores in all_results.values():
                row += f"{phase_scores[key]:>12.3f}"
            print(row)
        print("=" * 60)

    print("\nDone.")


if __name__ == "__main__":
    main()