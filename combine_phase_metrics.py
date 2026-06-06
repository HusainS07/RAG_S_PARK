import json
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path


PHASE_FILES = [
    Path("phase_metrics_simple.json"),
    Path("phase_metrics_medium.json"),
    Path("phase_metrics_complex.json"),
]


def mean(values):
    return round(sum(values) / len(values), 3) if values else 0.0


def load_phase_file(path):
    if not path.exists():
        print(f"Missing {path}; run its phase script first.")
        return None

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def aggregate(phases):
    all_results = []
    phase_summaries = {}

    for phase in phases:
        phase_name = phase["phase"]
        phase_summaries[phase_name] = phase["summary"]
        all_results.extend(phase.get("results", []))

    successful = [item for item in all_results if item.get("status") == "success"]
    scored = [item for item in successful if item.get("scores")]

    faithfulness = [item["scores"]["faithfulness"] for item in scored]
    relevancy = [item["scores"]["answer_relevancy"] for item in scored]
    recall = [item["scores"]["context_recall"] for item in scored]
    overall = [item["scores"]["overall"] for item in scored]

    response_times = [item.get("response_time", 0.0) for item in all_results]
    compression_ratios = [
        item.get("metadata", {}).get("compression_ratio", 0.0)
        for item in all_results
    ]
    contexts_used = [
        item.get("metadata", {}).get("contexts_used", 0)
        for item in all_results
    ]

    adaptive_k_by_complexity = defaultdict(list)
    category_scores = defaultdict(list)
    phase_scores = defaultdict(list)

    for item in all_results:
        metadata = item.get("metadata", {})
        complexity = metadata.get("complexity", "UNKNOWN")
        adaptive_k_by_complexity[complexity].append(metadata.get("adaptive_k", 0))

        if item.get("scores"):
            category_scores[item.get("category", "unknown")].append(item["scores"]["overall"])
            phase_scores[item.get("phase", "unknown")].append(item["scores"]["overall"])

    return {
        "generated_at": datetime.now().isoformat(),
        "source_files": [str(path) for path in PHASE_FILES],
        "total_questions": sum(phase["summary"]["total_questions"] for phase in phases),
        "completed_questions": len(all_results),
        "successful_queries": len(successful),
        "failed_queries": len(all_results) - len(successful),
        "api_success_rate": round((len(successful) / len(all_results)) * 100, 2) if all_results else 0.0,
        "scores": {
            "faithfulness_avg": mean(faithfulness),
            "answer_relevancy_avg": mean(relevancy),
            "context_recall_avg": mean(recall),
            "overall_avg": mean(overall),
            "overall_min": round(min(overall), 3) if overall else 0.0,
            "overall_max": round(max(overall), 3) if overall else 0.0,
            "overall_std": round(statistics.pstdev(overall), 3) if len(overall) > 1 else 0.0,
        },
        "retrieval": {
            "avg_response_time_seconds": mean(response_times),
            "avg_compression_ratio": mean(compression_ratios),
            "min_compression_ratio": round(min(compression_ratios), 3) if compression_ratios else 0.0,
            "max_compression_ratio": round(max(compression_ratios), 3) if compression_ratios else 0.0,
            "avg_contexts_used": mean(contexts_used),
            "min_contexts_used": min(contexts_used) if contexts_used else 0,
            "max_contexts_used": max(contexts_used) if contexts_used else 0,
            "adaptive_k_by_complexity": {
                key: {
                    "avg": mean(values),
                    "min": min(values) if values else 0,
                    "max": max(values) if values else 0,
                    "count": len(values),
                }
                for key, values in adaptive_k_by_complexity.items()
            },
        },
        "phase_performance": {
            phase: {
                "overall_avg": mean(scores),
                "questions_scored": len(scores),
            }
            for phase, scores in phase_scores.items()
        },
        "category_performance": {
            category: {
                "overall_avg": mean(scores),
                "questions_scored": len(scores),
            }
            for category, scores in sorted(category_scores.items())
        },
        "phase_summaries": phase_summaries,
    }


def write_summary_text(report):
    path = Path("phase_metrics_combined_summary.txt")
    with path.open("w", encoding="utf-8") as f:
        f.write("COMBINED RAG PHASE METRICS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {report['generated_at']}\n")
        f.write(f"Completed: {report['completed_questions']}/{report['total_questions']}\n")
        f.write(f"API success: {report['api_success_rate']:.1f}%\n\n")

        f.write("QUALITY SCORES\n")
        f.write("-" * 80 + "\n")
        f.write(f"Overall: {report['scores']['overall_avg']:.3f}\n")
        f.write(f"Faithfulness: {report['scores']['faithfulness_avg']:.3f}\n")
        f.write(f"Answer relevancy: {report['scores']['answer_relevancy_avg']:.3f}\n")
        f.write(f"Context recall: {report['scores']['context_recall_avg']:.3f}\n\n")

        f.write("RETRIEVAL METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Avg response time: {report['retrieval']['avg_response_time_seconds']:.2f}s\n")
        f.write(f"Avg compression ratio: {report['retrieval']['avg_compression_ratio']:.3f}\n")
        f.write(f"Avg contexts used: {report['retrieval']['avg_contexts_used']:.1f}\n\n")

        f.write("PHASE PERFORMANCE\n")
        f.write("-" * 80 + "\n")
        for phase, data in report["phase_performance"].items():
            f.write(f"{phase}: {data['overall_avg']:.3f} ({data['questions_scored']} scored)\n")

    return path


def main():
    phases = [load_phase_file(path) for path in PHASE_FILES]
    phases = [phase for phase in phases if phase]

    if not phases:
        print("No phase metric files found.")
        return

    report = aggregate(phases)

    output_path = Path("phase_metrics_combined.json")
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    summary_path = write_summary_text(report)

    print("\nCombined metrics saved:")
    print(f"  {output_path}")
    print(f"  {summary_path}")
    print(f"\nOverall score: {report['scores']['overall_avg']:.3f}")
    print(f"API success: {report['api_success_rate']:.1f}%")
    print(f"Avg compression ratio: {report['retrieval']['avg_compression_ratio']:.3f}")


if __name__ == "__main__":
    main()
