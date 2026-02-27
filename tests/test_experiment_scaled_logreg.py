from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from experiment_scaled_logreg import build_markdown_report, parse_c_values, rank_results


class ExperimentScaledLogRegTests(unittest.TestCase):
    def test_parse_c_values(self) -> None:
        self.assertEqual(parse_c_values("0.1, 1.0,3"), [0.1, 1.0, 3.0])
        with self.assertRaises(ValueError):
            parse_c_values("")
        with self.assertRaises(ValueError):
            parse_c_values("0.1,-1")

    def test_rank_results_orders_by_recall_then_pr_auc(self) -> None:
        df = pd.DataFrame(
            [
                {"name": "a", "recall_at_target_fpr": 0.90, "pr_auc": 0.70, "roc_auc": 0.98},
                {"name": "b", "recall_at_target_fpr": 0.91, "pr_auc": 0.60, "roc_auc": 0.97},
                {"name": "c", "recall_at_target_fpr": 0.90, "pr_auc": 0.80, "roc_auc": 0.96},
            ]
        )
        ranked = rank_results(df)
        self.assertListEqual(ranked["name"].tolist(), ["b", "c", "a"])

    def test_build_markdown_report_contains_recommended_model(self) -> None:
        ranked = pd.DataFrame(
            [
                {
                    "name": "scaled_logreg_c0.3",
                    "is_scaled": True,
                    "c_value": 0.3,
                    "pr_auc": 0.75,
                    "roc_auc": 0.98,
                    "recall_at_target_fpr": 0.91,
                    "threshold_at_target_fpr": 0.33,
                }
            ]
        )
        report = build_markdown_report(
            ranked=ranked,
            train_rows=100,
            test_rows=20,
            target_fpr=0.02,
            best_model_out_path=Path("models/week2_best_logreg.joblib"),
        )
        self.assertIn("## Recommended Config", report)
        self.assertIn("scaled_logreg_c0.3", report)


if __name__ == "__main__":
    unittest.main()
