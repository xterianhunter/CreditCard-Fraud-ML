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
    evaluate_policy_cost,
    generate_threshold_grid,
    parse_methods,
    search_best_policy,
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


if __name__ == "__main__":
    unittest.main()
