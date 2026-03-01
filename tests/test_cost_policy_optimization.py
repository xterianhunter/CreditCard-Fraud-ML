from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from cost_policy_optimization import (
    annotate_guardrails,
    build_policy_config_payload,
    enforce_promotion_gate,
    evaluate_policy_cost,
    generate_threshold_grid,
    parse_methods,
    select_recommended_policy,
    search_best_policy,
    validate_guardrails,
)


class CostPolicyOptimizationTests(unittest.TestCase):
    def test_parse_methods(self) -> None:
        self.assertEqual(parse_methods("none,platt,isotonic"), ["none", "platt", "isotonic"])
        self.assertEqual(parse_methods("  platt "), ["platt"])
        with self.assertRaises(ValueError):
            parse_methods("")
        with self.assertRaises(ValueError):
            parse_methods("none,beta")

    def test_evaluate_policy_cost(self) -> None:
        y_true = np.array([0, 0, 1, 1], dtype=np.int8)
        y_score = np.array([0.01, 0.30, 0.20, 0.90], dtype=np.float64)
        out = evaluate_policy_cost(
            y_true=y_true,
            y_score=y_score,
            approve_threshold=0.10,
            decline_threshold=0.50,
            cost_false_negative=10.0,
            cost_false_decline=3.0,
            cost_review=0.5,
        )
        self.assertAlmostEqual(out["approve_rate"], 0.25)
        self.assertAlmostEqual(out["review_rate"], 0.5)
        self.assertAlmostEqual(out["decline_rate"], 0.25)
        self.assertAlmostEqual(out["flagged_fraud_capture"], 1.0)
        self.assertAlmostEqual(out["decline_fpr"], 0.0)
        # fraud_approve=0, nonfraud_decline=0, review_count=2 => total cost=1.0
        self.assertAlmostEqual(out["total_cost"], 1.0)

    def test_search_best_policy_prefers_lower_cost(self) -> None:
        y_true = np.array([0, 0, 1, 1], dtype=np.int8)
        y_score = np.array([0.05, 0.20, 0.30, 0.90], dtype=np.float64)
        pairs = [(0.10, 0.50), (0.15, 0.60), (0.02, 0.50)]
        best, ranked = search_best_policy(
            y_true=y_true,
            y_score=y_score,
            threshold_pairs=pairs,
            cost_false_negative=20.0,
            cost_false_decline=5.0,
            cost_review=0.1,
        )
        self.assertEqual(len(ranked), 3)
        self.assertAlmostEqual(float(best["total_cost"]), float(ranked.iloc[0]["total_cost"]))

    def test_generate_threshold_grid(self) -> None:
        grid = generate_threshold_grid(
            t1_min=0.1,
            t1_max=0.2,
            t2_min=0.2,
            t2_max=0.3,
            step=0.1,
        )
        self.assertIn((0.1, 0.2), grid)
        self.assertIn((0.1, 0.3), grid)
        self.assertIn((0.2, 0.3), grid)
        with self.assertRaises(ValueError):
            generate_threshold_grid(
                t1_min=0.2,
                t1_max=0.1,
                t2_min=0.2,
                t2_max=0.3,
                step=0.1,
            )

    def test_validate_guardrails(self) -> None:
        validate_guardrails(
            max_review_rate=0.1,
            min_flagged_fraud_capture=0.9,
            max_decline_fpr=0.02,
        )
        with self.assertRaises(ValueError):
            validate_guardrails(
                max_review_rate=1.1,
                min_flagged_fraud_capture=0.9,
                max_decline_fpr=0.02,
            )

    def test_enforce_promotion_gate(self) -> None:
        enforce_promotion_gate(require_feasible=False, recommended_feasible=False)
        enforce_promotion_gate(require_feasible=True, recommended_feasible=True)
        with self.assertRaises(SystemExit):
            enforce_promotion_gate(require_feasible=True, recommended_feasible=False)

    def test_build_policy_config_payload(self) -> None:
        row = pd.Series(
            {
                "method": "none",
                "approve_threshold": 0.2,
                "decline_threshold": 0.8,
                "review_rate": 0.0516,
                "decline_fpr": 0.0033,
                "flagged_fraud_capture": 0.8636,
                "total_cost": 1138.0,
                "cost_per_row": 0.039956,
            }
        )
        payload = build_policy_config_payload(
            best_method_row=row,
            recommended_feasible=True,
            model_path=Path("models/week2_best_logreg.joblib"),
            methods=["none", "platt"],
            max_review_rate=0.1,
            min_flagged_fraud_capture=0.85,
            max_decline_fpr=0.02,
            grid_step=0.01,
        )
        profiles = payload["profiles"]
        self.assertIsInstance(profiles, dict)
        self.assertIn("phase2_guarded", profiles)
        phase2_profile = profiles["phase2_guarded"]
        self.assertAlmostEqual(float(phase2_profile["approve_threshold"]), 0.2)
        self.assertAlmostEqual(float(phase2_profile["decline_threshold"]), 0.8)

    def test_annotate_guardrails_adds_flags(self) -> None:
        ranked = pd.DataFrame(
            [
                {
                    "review_rate": 0.04,
                    "flagged_fraud_capture": 0.92,
                    "decline_fpr": 0.01,
                    "total_cost": 100.0,
                },
                {
                    "review_rate": 0.12,
                    "flagged_fraud_capture": 0.80,
                    "decline_fpr": 0.03,
                    "total_cost": 90.0,
                },
            ]
        )
        out = annotate_guardrails(
            ranked,
            max_review_rate=0.1,
            min_flagged_fraud_capture=0.9,
            max_decline_fpr=0.02,
        )
        self.assertTrue(bool(out.iloc[0]["meets_guardrails"]))
        self.assertFalse(bool(out.iloc[1]["meets_guardrails"]))

    def test_select_recommended_policy_prefers_feasible(self) -> None:
        ranked = pd.DataFrame(
            [
                {
                    "rank": 1,
                    "approve_threshold": 0.10,
                    "decline_threshold": 0.30,
                    "total_cost": 50.0,
                    "cost_per_row": 0.1,
                    "review_rate": 0.12,
                    "flagged_fraud_capture": 0.93,
                    "decline_fpr": 0.01,
                },
                {
                    "rank": 2,
                    "approve_threshold": 0.12,
                    "decline_threshold": 0.35,
                    "total_cost": 70.0,
                    "cost_per_row": 0.14,
                    "review_rate": 0.08,
                    "flagged_fraud_capture": 0.91,
                    "decline_fpr": 0.01,
                },
            ]
        )
        selected, feasible, _ = select_recommended_policy(
            ranked,
            max_review_rate=0.10,
            min_flagged_fraud_capture=0.90,
            max_decline_fpr=0.02,
        )
        self.assertTrue(feasible)
        self.assertAlmostEqual(float(selected["approve_threshold"]), 0.12)

    def test_select_recommended_policy_soft_fail_uses_lowest_violation(self) -> None:
        ranked = pd.DataFrame(
            [
                {
                    "rank": 1,
                    "approve_threshold": 0.10,
                    "decline_threshold": 0.30,
                    "total_cost": 55.0,
                    "cost_per_row": 0.11,
                    "review_rate": 0.11,
                    "flagged_fraud_capture": 0.84,
                    "decline_fpr": 0.01,
                },
                {
                    "rank": 2,
                    "approve_threshold": 0.08,
                    "decline_threshold": 0.28,
                    "total_cost": 40.0,
                    "cost_per_row": 0.08,
                    "review_rate": 0.13,
                    "flagged_fraud_capture": 0.70,
                    "decline_fpr": 0.04,
                },
            ]
        )
        selected, feasible, _ = select_recommended_policy(
            ranked,
            max_review_rate=0.10,
            min_flagged_fraud_capture=0.90,
            max_decline_fpr=0.02,
        )
        self.assertFalse(feasible)
        self.assertAlmostEqual(float(selected["approve_threshold"]), 0.10)


if __name__ == "__main__":
    unittest.main()
