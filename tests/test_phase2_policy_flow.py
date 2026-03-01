from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from cost_policy_optimization import build_policy_config_payload, write_policy_config
from data_contract import LABEL_COLUMN, REQUIRED_COLUMNS
from inference import run_inference


class StubModel:
    def __init__(self, scores: list[float]) -> None:
        self._scores = np.array(scores, dtype=np.float64)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if len(x) != len(self._scores):
            raise ValueError("StubModel score length does not match input rows.")
        return np.column_stack([1.0 - self._scores, self._scores])


def make_valid_dataframe(rows: int = 3) -> pd.DataFrame:
    data: dict[str, list[float]] = {}
    for col in sorted(REQUIRED_COLUMNS):
        if col == LABEL_COLUMN:
            continue
        data[col] = [0.0 for _ in range(rows)]
    return pd.DataFrame(data)


class Phase2PolicyFlowTests(unittest.TestCase):
    def test_optimizer_policy_artifact_drives_inference_profile(self) -> None:
        input_df = make_valid_dataframe(rows=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = tmp / "input.csv"
            model_path = tmp / "model.joblib"
            policy_path = tmp / "policy_config.json"
            input_df.to_csv(input_path, index=False)

            artifact = {
                "model": StubModel([0.05, 0.30, 0.90]),
                "feature_columns": sorted(input_df.columns.tolist()),
                "threshold_for_target_fpr": 0.4,
                "target_fpr": 0.02,
            }
            joblib.dump(artifact, model_path)

            best_row = pd.Series(
                {
                    "method": "none",
                    "approve_threshold": 0.20,
                    "decline_threshold": 0.80,
                    "review_rate": 0.0516,
                    "decline_fpr": 0.0033,
                    "flagged_fraud_capture": 0.8636,
                    "total_cost": 1138.0,
                    "cost_per_row": 0.039956,
                }
            )
            payload = build_policy_config_payload(
                best_method_row=best_row,
                recommended_feasible=True,
                model_path=Path("models/week2_best_logreg.joblib"),
                methods=["none", "platt", "isotonic"],
                max_review_rate=0.10,
                min_flagged_fraud_capture=0.85,
                max_decline_fpr=0.02,
                grid_step=0.01,
            )
            write_policy_config(policy_path, payload)

            scored_df, approve_t, decline_t = run_inference(
                input_path=input_path,
                model_path=model_path,
                output_path=None,
                policy_profile="phase2_guarded",
                approve_threshold=None,
                decline_threshold=None,
                policy_config_path=policy_path,
            )

            self.assertTrue(policy_path.exists())
            self.assertAlmostEqual(approve_t, 0.20)
            self.assertAlmostEqual(decline_t, 0.80)
            self.assertListEqual(scored_df["decision"].tolist(), ["approve", "review", "decline"])


if __name__ == "__main__":
    unittest.main()
