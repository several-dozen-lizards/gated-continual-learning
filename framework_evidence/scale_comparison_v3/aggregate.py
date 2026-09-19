"""Rebuild compact v3 summaries from immutable per-run receipts."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sample_sd(values):
    return statistics.stdev(values) if len(values) >= 2 else None


def aggregate_results(output_dir: Path):
    output_dir = Path(output_dir)
    rows = []
    frozen = []
    invalidated = []
    for report_path in sorted(output_dir.glob("**/report.json")):
        invalidation_path = report_path.parent / "INVALIDATED.md"
        if invalidation_path.exists():
            invalidated.append({
                "receipt": report_path.relative_to(output_dir).as_posix(),
                "reason_file": invalidation_path.relative_to(output_dir).as_posix(),
            })
            continue
        report = _read(report_path)
        relative_path = report_path.relative_to(output_dir).as_posix()
        if report.get("arm_type") is None:
            frozen.append({
                "model": report["model_key"],
                "accuracy": report["evaluation"]["accuracy"],
                "canary_score": report["canary"]["score"],
                "receipt": relative_path,
            })
            continue
        comparison = report["canary"]["final"]["baseline_comparison"]
        trajectory = report["canary"]["trajectory"]
        rows.append({
            "model": report["model_key"],
            "model_name": report["model"]["name"],
            "arm": report["arm_type"],
            "seed": report["seed"],
            "accuracy": report["evaluation"]["accuracy"],
            "transfer_accuracy": report["transfer"]["accuracy"],
            "canary_final_delta": comparison["score_delta"],
            "canary_minimum_delta": trajectory["minimum_score_delta"],
            "canary_baseline_score": report["canary"]["baseline"]["score"],
            "canary_baseline_outcome_sha256": report["canary"]["baseline"].get("outcome_sha256"),
            "canary_final_regressions": comparison["regressions"],
            "canary_final_recoveries": comparison["recoveries"],
            "stream_input_tokens": report["stream_training"]["input_tokens"],
            "stream_steps": report["stream_training"]["steps"],
            "events_trained": report["events_trained"],
            "receipt": relative_path,
        })

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["arm"])].append(row)
    aggregates = []
    for (model, arm), group in sorted(grouped.items()):
        accuracies = [row["accuracy"] for row in group]
        minimum_deltas = [row["canary_minimum_delta"] for row in group]
        aggregates.append({
            "model": model,
            "arm": arm,
            "n": len(group),
            "accuracy_mean": statistics.mean(accuracies),
            "accuracy_median": statistics.median(accuracies),
            "accuracy_sample_sd": _sample_sd(accuracies),
            "accuracy_min": min(accuracies),
            "accuracy_max": max(accuracies),
            "canary_minimum_delta_mean": statistics.mean(minimum_deltas),
            "canary_worst_delta": min(minimum_deltas),
        })

    paired = []
    row_index = {(row["model"], row["arm"], row["seed"]): row for row in rows}
    for model in sorted({row["model"] for row in rows}):
        for comparator in ("ungated", "matched_random"):
            seeds = sorted({
                row["seed"] for row in rows
                if row["model"] == model and row["arm"] == "gated"
                and (model, comparator, row["seed"]) in row_index
            })
            differences = [
                row_index[(model, "gated", seed)]["accuracy"]
                - row_index[(model, comparator, seed)]["accuracy"]
                for seed in seeds
            ]
            if differences:
                baseline_pairs = [(
                    row_index[(model, "gated", seed)]["canary_baseline_outcome_sha256"],
                    row_index[(model, comparator, seed)]["canary_baseline_outcome_sha256"],
                ) for seed in seeds]
                paired.append({
                    "model": model,
                    "contrast": f"gated_minus_{comparator}",
                    "n": len(differences),
                    "mean_accuracy_difference": statistics.mean(differences),
                    "sample_sd": _sample_sd(differences),
                    "paired_baseline_outcomes_aligned": all(
                        left is not None and left == right
                        for left, right in baseline_pairs
                    ),
                    "differences_by_seed": dict(zip(map(str, seeds), differences)),
                })

    summary = {
        "schema": "gated-scale-comparison-v3/summary-1",
        "complete_run_count": len(rows),
        "invalidated_run_count": len(invalidated),
        "invalidated_runs": invalidated,
        "frozen": frozen,
        "runs": rows,
        "aggregates": aggregates,
        "paired_contrasts": paired,
        "notes": [
            "Ranges and sample SDs are descriptive; inferential claims require the prespecified seed set.",
            "Canary deltas are relative to each run's own pre-training baseline.",
            "matched_random is exact on event count and per-event token-shape strata.",
        ],
    }
    path = output_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs")
    args = parser.parse_args()
    summary = aggregate_results(args.output)
    print(f"Aggregated {summary['complete_run_count']} completed runs")


if __name__ == "__main__":
    main()
