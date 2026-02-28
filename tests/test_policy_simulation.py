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

from policy_simulation import (
    build_policy_report_markdown,
    evaluate_policy,
    evaluate_threshold_pairs,
    parse_threshold_pairs,
    select_fallback_policy,
    select_recommended_policy,
)


class PolicySimulationTests(unittest.TestCase):
    def test_parse_threshold_pairs(self) -> None:
        pairs = parse_threshold_pairs("0.05:0.45, 0.08:0.50")
        self.assertEqual(pairs, [(0.05, 0.45), (0.08, 0.5)])

        with self.assertRaises(ValueError):
            parse_threshold_pairs("")
        with self.assertRaises(ValueError):
            parse_threshold_pairs("0.05-0.45")
        with self.assertRaises(ValueError):
            parse_threshold_pairs("0.50:0.20")

    def test_evaluate_policy_metrics(self) -> None:
        y_true = np.array([0, 0, 1, 1, 0], dtype=np.int8)
        y_score = np.array([0.01, 0.20, 0.30, 0.90, 0.60], dtype=np.float64)
        metrics = evaluate_policy(
            y_true=y_true,
            y_score=y_score,
            approve_threshold=0.10,
            decline_threshold=0.50,
        )

        self.assertAlmostEqual(metrics["approve_rate"], 1 / 5)
        self.assertAlmostEqual(metrics["review_rate"], 2 / 5)
        self.assertAlmostEqual(metrics["decline_rate"], 2 / 5)
        self.assertAlmostEqual(metrics["fraud_capture_decline"], 0.5)
        self.assertAlmostEqual(metrics["fraud_capture_flagged"], 1.0)
        self.assertAlmostEqual(metrics["decline_fpr"], 1 / 3)

    def test_select_recommended_policy_prefers_feasible(self) -> None:
        ranked = pd.DataFrame(
            [
                {
                    "rank": 1,
                    "policy_id": "p1",
                    "approve_threshold": 0.05,
                    "decline_threshold": 0.45,
                    "review_rate": 0.20,
                    "fraud_capture_flagged": 1.00,
                    "decline_fpr": 0.03,
                },
                {
                    "rank": 2,
                    "policy_id": "p2",
                    "approve_threshold": 0.08,
                    "decline_threshold": 0.50,
                    "review_rate": 0.08,
                    "fraud_capture_flagged": 0.96,
                    "decline_fpr": 0.01,
                },
            ]
        )
        recommended, feasible = select_recommended_policy(
            ranked,
            max_review_rate=0.10,
            min_fraud_capture_flagged=0.95,
            max_decline_fpr=0.02,
        )
        self.assertTrue(feasible)
        self.assertEqual(recommended["policy_id"], "p2")

    def test_select_recommended_policy_uses_lowest_violation_when_infeasible(self) -> None:
        ranked = pd.DataFrame(
            [
                {
                    "rank": 1,
                    "policy_id": "p1",
                    "approve_threshold": 0.03,
                    "decline_threshold": 0.40,
                    "review_rate": 0.30,
                    "fraud_capture_flagged": 1.00,
                    "decline_fpr": 0.03,
                },
                {
                    "rank": 2,
                    "policy_id": "p2",
                    "approve_threshold": 0.10,
                    "decline_threshold": 0.50,
                    "review_rate": 0.11,
                    "fraud_capture_flagged": 0.96,
                    "decline_fpr": 0.015,
                },
                {
                    "rank": 3,
                    "policy_id": "p3",
                    "approve_threshold": 0.12,
                    "decline_threshold": 0.53,
                    "review_rate": 0.095,
                    "fraud_capture_flagged": 0.92,
                    "decline_fpr": 0.012,
                },
            ]
        )
        recommended, feasible = select_recommended_policy(
            ranked,
            max_review_rate=0.08,
            min_fraud_capture_flagged=0.95,
            max_decline_fpr=0.01,
        )
        self.assertFalse(feasible)
        self.assertEqual(recommended["policy_id"], "p2")

    def test_build_policy_report_markdown(self) -> None:
        y_true = np.array([0, 0, 1, 1], dtype=np.int8)
        y_score = np.array([0.02, 0.20, 0.30, 0.90], dtype=np.float64)
        ranked = evaluate_threshold_pairs(y_true, y_score, [(0.05, 0.40), (0.10, 0.50)])
        recommended = ranked.iloc[0]
        fallback = ranked.iloc[1]

        report = build_policy_report_markdown(
            recommended=recommended,
            recommended_feasible=True,
            fallback=fallback,
            max_review_rate=0.10,
            min_fraud_capture_flagged=0.95,
            max_decline_fpr=0.02,
        )
        self.assertIn("# Policy Simulation Report (Week 3)", report)
        self.assertIn("## Expected Operational Impact", report)
        self.assertIn("## Fallback Policy", report)

    def test_select_fallback_policy_prefers_low_review_load(self) -> None:
        ranked = pd.DataFrame(
            [
                {
                    "rank": 1,
                    "policy_id": "p1",
                    "review_rate": 0.18,
                    "decline_fpr": 0.02,
                    "fraud_capture_flagged": 0.99,
                },
                {
                    "rank": 2,
                    "policy_id": "p2",
                    "review_rate": 0.08,
                    "decline_fpr": 0.015,
                    "fraud_capture_flagged": 0.95,
                },
                {
                    "rank": 3,
                    "policy_id": "p3",
                    "review_rate": 0.10,
                    "decline_fpr": 0.010,
                    "fraud_capture_flagged": 0.96,
                },
            ]
        )
        fallback = select_fallback_policy(ranked, "p1")
        assert fallback is not None
        self.assertEqual(fallback["policy_id"], "p2")


if __name__ == "__main__":
    unittest.main()
