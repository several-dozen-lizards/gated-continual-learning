"""CPU-only contract tests for the v3 study design."""

from __future__ import annotations

import sys
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
V1 = HERE.parent / "scale_comparison"
for path in (HERE, V1):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from canary import compare_with_baseline, one_sided_mcnemar_p, wilson_interval
from canary_bank import CANARY_ITEMS
from aggregate import aggregate_results
from world import make_world, matched_random, route


class CanaryBankTests(unittest.TestCase):
    def test_bank_is_large_unique_and_balanced(self):
        self.assertEqual(120, len(CANARY_ITEMS))
        self.assertEqual(120, len({item["id"] for item in CANARY_ITEMS}))
        self.assertEqual(
            {"A": 30, "B": 30, "C": 30, "D": 30},
            dict(Counter(item["correct_label"] for item in CANARY_ITEMS)),
        )
        self.assertEqual(
            {
                "addition": 20,
                "common_knowledge": 20,
                "comparison": 20,
                "multiplication": 20,
                "subtraction": 20,
                "transitive_reasoning": 20,
            },
            dict(Counter(item["family"] for item in CANARY_ITEMS)),
        )

    def test_items_have_four_unique_choices(self):
        for item in CANARY_ITEMS:
            self.assertEqual(4, len(item["choices"]), item["id"])
            self.assertEqual(4, len(set(item["choices"])), item["id"])

    def test_bank_is_disjoint_from_beacon_fixture_names(self):
        world = make_world("test", 17)
        bank_text = " ".join(
            item["question"] + " " + " ".join(item["choices"])
            for item in CANARY_ITEMS
        ).lower()
        for name in world["final"]:
            self.assertNotIn(name.lower(), bank_text)


class BaselineComparisonTests(unittest.TestCase):
    @staticmethod
    def _result(values, score=None, margin=0.0):
        if score is None:
            score = sum(values) / len(values)
        return {
            "score": score,
            "mean_correct_margin": margin,
            "rows": [
                {"id": f"q{index}", "correct": value}
                for index, value in enumerate(values)
            ],
        }

    def test_comparison_counts_regressions_and_recoveries(self):
        baseline = self._result([True, True, False, False], margin=0.4)
        current = self._result([False, True, True, False], margin=0.1)
        comparison = compare_with_baseline(baseline, current)
        self.assertEqual(1, comparison["regressions"])
        self.assertEqual(1, comparison["recoveries"])
        self.assertEqual(2, comparison["unchanged"])
        self.assertEqual(0.0, comparison["score_delta"])
        self.assertAlmostEqual(-0.3, comparison["margin_delta"])

    def test_mcnemar_is_directional_without_a_frying_label(self):
        self.assertEqual(1.0, one_sided_mcnemar_p(1, 1))
        self.assertLess(one_sided_mcnemar_p(10, 0), 0.01)

    def test_wilson_interval_contains_observed_rate(self):
        lower, upper = wilson_interval(60, 120)
        self.assertLess(lower, 0.5)
        self.assertGreater(upper, 0.5)


class MatchingTests(unittest.TestCase):
    def test_matched_random_preserves_exact_cost_multiset(self):
        world = make_world("test", 17)
        selected = [
            index for index, event in enumerate(world["stream"])
            if route(event)["route"] == "formative"
        ]

        def synthetic_token_shape(event):
            return (len(event["name"]), len(event["color"]))

        matched = matched_random(
            world["stream"], selected, synthetic_token_shape, seed=1_000_020,
        )
        self.assertEqual(len(selected), len(matched))
        self.assertEqual(
            Counter(synthetic_token_shape(world["stream"][index]) for index in selected),
            Counter(synthetic_token_shape(world["stream"][index]) for index in matched),
        )


class AggregationTests(unittest.TestCase):
    @staticmethod
    def _report(seed):
        return {
            "model_key": "0.5b",
            "model": {"name": "fixture"},
            "arm_type": "gated",
            "seed": seed,
            "evaluation": {"accuracy": 0.5},
            "transfer": {"accuracy": 0.4},
            "events_trained": 2,
            "stream_training": {"input_tokens": 20, "steps": 3},
            "canary": {
                "baseline": {"score": 0.5, "outcome_sha256": f"baseline-{seed}"},
                "final": {"baseline_comparison": {
                    "score_delta": 0.0, "regressions": 0, "recoveries": 0,
                }},
                "trajectory": {"minimum_score_delta": 0.0},
            },
        }

    def test_aggregation_preserves_but_excludes_invalidated_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid = root / "gated_0.5b" / "seed_17"
            invalid = root / "gated_0.5b" / "seed_29"
            valid.mkdir(parents=True)
            invalid.mkdir(parents=True)
            (valid / "report.json").write_text(
                json.dumps(self._report(17)), encoding="utf-8",
            )
            (invalid / "report.json").write_text(
                json.dumps(self._report(29)), encoding="utf-8",
            )
            (invalid / "INVALIDATED.md").write_text("reason", encoding="utf-8")
            summary = aggregate_results(root)
            self.assertEqual(1, summary["complete_run_count"])
            self.assertEqual(1, summary["invalidated_run_count"])
            self.assertEqual(17, summary["runs"][0]["seed"])


if __name__ == "__main__":
    unittest.main()
