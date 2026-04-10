import argparse
import json
from pathlib import Path

import numpy as np


PLANNED_COMPARISONS = (
    ("full_hybrid", "repeat_last"),
    ("full_hybrid", "item_markov"),
    ("full_hybrid", "transition_cf"),
    ("full_hybrid", "hybrid_cf_topsis"),
    ("full_hybrid", "gru4rec"),
    ("full_hybrid", "sasrec"),
)


def paired_cohens_d(challenger_values, baseline_values):
    diffs = np.asarray(challenger_values, dtype=float) - np.asarray(baseline_values, dtype=float)
    if len(diffs) < 2:
        return 0.0
    diff_std = diffs.std(ddof=1)
    if diff_std < 1e-12:
        return 0.0
    return float(diffs.mean() / diff_std)


def analyze_payload(payload, metric_name):
    split_metrics = payload["repeated"]["splits"]
    comparisons = {}
    for challenger, baseline in PLANNED_COMPARISONS:
        challenger_values = [split["metrics"][challenger][metric_name] for split in split_metrics]
        baseline_values = [split["metrics"][baseline][metric_name] for split in split_metrics]
        challenger_mean = float(np.mean(challenger_values))
        baseline_mean = float(np.mean(baseline_values))
        mean_delta = challenger_mean - baseline_mean
        relative_gain = 0.0 if abs(baseline_mean) < 1e-12 else 100.0 * mean_delta / baseline_mean
        comparisons[f"{challenger}_vs_{baseline}"] = {
            "metric": metric_name,
            "challenger_mean": challenger_mean,
            "baseline_mean": baseline_mean,
            "mean_delta": mean_delta,
            "relative_gain_percent": relative_gain,
            "paired_cohens_dz": paired_cohens_d(challenger_values, baseline_values),
            "split_values": {
                "challenger": challenger_values,
                "baseline": baseline_values,
            },
        }
    return comparisons


def print_report(title, comparisons):
    print("=" * 88)
    print(title)
    print("=" * 88)
    print(f"{'Comparison':<34} {'Mean Delta':>12} {'Rel Gain %':>12} {'Cohen dz':>12}")
    print("-" * 88)
    for comparison_name, result in comparisons.items():
        print(
            f"{comparison_name:<34} "
            f"{result['mean_delta']:>12.4f} "
            f"{result['relative_gain_percent']:>12.2f} "
            f"{result['paired_cohens_dz']:>12.3f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Compute paired standardized effect sizes from repeated-split results.")
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--metric", default="NDCG@K")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    comparisons = analyze_payload(payload, args.metric)
    report = {
        "dataset_name": payload["dataset_metadata"]["dataset_name"],
        "source_name": payload["dataset_metadata"]["source_name"],
        "metric": args.metric,
        "comparisons": comparisons,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print_report(payload["dataset_metadata"]["source_name"], comparisons)
    print(f"\nWrote effect size report to {output_path.resolve()}")


if __name__ == "__main__":
    main()
