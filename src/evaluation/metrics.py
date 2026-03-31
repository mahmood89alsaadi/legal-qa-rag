"""
Evaluation module for Legal MCQ QA.
Computes accuracy, per-option breakdown, and robustness metrics.
"""

import json
from typing import List, Dict
from collections import Counter


def compute_accuracy(predictions: List[Dict]) -> float:
    correct = sum(
        1 for p in predictions
        if p.get("predicted_answer") == p.get("gold_answer")
    )
    return correct / len(predictions) if predictions else 0.0


def compute_per_option_accuracy(predictions: List[Dict]) -> Dict[str, float]:
    """Accuracy breakdown by gold answer label (A, B, C, ...)."""
    buckets: Dict[str, List] = {}
    for p in predictions:
        gold = p.get("gold_answer", "UNKNOWN")
        buckets.setdefault(gold, []).append(
            p.get("predicted_answer") == gold
        )
    return {label: sum(v) / len(v) for label, v in buckets.items()}


def compute_prompt_robustness(results_by_style: Dict[str, List[Dict]]) -> Dict:
    """
    Measure accuracy variance across prompt styles.
    High variance = low robustness (sensitive model).
    """
    accuracies = {
        style: compute_accuracy(preds)
        for style, preds in results_by_style.items()
    }
    values = list(accuracies.values())
    variance = sum((v - sum(values) / len(values)) ** 2 for v in values) / len(values)
    return {
        "per_style_accuracy": accuracies,
        "variance": round(variance, 4),
        "robust": variance < 0.01,
    }


def mcq_difficulty_analysis(mcq4_preds: List[Dict], mcq20_preds: List[Dict]) -> Dict:
    """Compare performance drop from MCQ-4 to MCQ-20."""
    acc4 = compute_accuracy(mcq4_preds)
    acc20 = compute_accuracy(mcq20_preds)
    return {
        "mcq4_accuracy": round(acc4, 4),
        "mcq20_accuracy": round(acc20, 4),
        "drop": round(acc4 - acc20, 4),
        "relative_drop_pct": round((acc4 - acc20) / acc4 * 100, 2) if acc4 > 0 else 0,
    }


def generate_report(predictions: List[Dict], output_path: str = None) -> Dict:
    report = {
        "total": len(predictions),
        "accuracy": round(compute_accuracy(predictions), 4),
        "per_option_accuracy": compute_per_option_accuracy(predictions),
        "predicted_distribution": dict(
            Counter(p.get("predicted_answer") for p in predictions)
        ),
    }

    if output_path:
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {output_path}")

    return report
