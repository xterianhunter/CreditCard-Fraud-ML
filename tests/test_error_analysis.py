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

from error_analysis import build_report_markdown, compute_confusion_counts, select_top_errors


class ErrorAnalysisTests(unittest.TestCase):
    def test_compute_confusion_counts(self) -> None:
        y_true = np.array([0, 0, 1, 1], dtype=np.int8)
        y_pred = np.array([0, 1, 0, 1], dtype=np.int8)
        counts = compute_confusion_counts(y_true, y_pred)
        self.assertDictEqual(counts, {"tn": 1, "fp": 1, "fn": 1, "tp": 1})

    def test_select_top_errors_returns_fp_and_fn(self) -> None:
        df = pd.DataFrame(
            {
                "Time": [10.0, 20.0, 30.0, 40.0],
                "Amount": [1.0, 2.0, 3.0, 4.0],
                "Class": [0, 0, 1, 1],
            }
        )
        y_true = df["Class"].to_numpy(dtype=np.int8)
        y_score = np.array([0.9, 0.2, 0.1, 0.8], dtype=np.float64)
        result = select_top_errors(test_df=df, y_true=y_true, y_score=y_score, threshold=0.5, top_n=2)

        self.assertEqual(len(result), 2)
        self.assertSetEqual(set(result["error_type"].tolist()), {"false_positive", "false_negative"})

    def test_build_report_markdown_contains_key_sections(self) -> None:
        rows = pd.DataFrame(
            {
                "row_id": [123],
                "Time": [100.0],
                "Amount": [20.0],
                "Class": [1],
                "predicted_class": [0],
                "score": [0.02],
                "error_type": ["false_negative"],
            }
        )
        report = build_report_markdown(
            train_rows=100,
            test_rows=20,
            target_fpr=0.02,
            threshold=0.3,
            counts={"tn": 15, "fp": 2, "fn": 1, "tp": 2},
            exported_errors=1,
            top_error_rows=rows,
        )
        self.assertIn("# Error Analysis Report (Week 2)", report)
        self.assertIn("Confusion Matrix", report)
        self.assertIn("row_id=123", report)


if __name__ == "__main__":
    unittest.main()
