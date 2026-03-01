from __future__ import annotations

import sys
import tempfile
import unittest
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from data_contract import LABEL_COLUMN, REQUIRED_COLUMNS
from inference import (
    build_policy_status_text,
    load_model_bundle,
    policy_status_snapshot,
    resolve_feature_columns,
    run_inference,
    score_with_policy,
)


class StubModel:
    def __init__(self, scores: list[float]) -> None:
        self._scores = np.array(scores, dtype=np.float64)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if len(x) != len(self._scores):
            raise ValueError("StubModel score length does not match input rows.")
        return np.column_stack([1.0 - self._scores, self._scores])


def make_valid_dataframe(rows: int = 4, include_label: bool = False) -> pd.DataFrame:
    data: dict[str, list[float] | list[int]] = {}
    for col in sorted(REQUIRED_COLUMNS):
        if col == LABEL_COLUMN:
            if include_label:
                data[col] = [0 for _ in range(rows)]
        else:
            data[col] = [0.0 for _ in range(rows)]
    return pd.DataFrame(data)


def write_policy_config(path: Path) -> None:
    payload = {
        "schema_version": 1,
        "generated_at": "2026-03-01",
        "profiles": {
            "primary": {"approve_threshold": 0.11, "decline_threshold": 0.45},
            "fallback": {"approve_threshold": 0.19, "decline_threshold": 0.80},
            "phase2_guarded": {"approve_threshold": 0.20, "decline_threshold": 0.80},
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


class InferenceTests(unittest.TestCase):
    def test_score_with_policy_assigns_expected_decisions(self) -> None:
        df = make_valid_dataframe(rows=4, include_label=False)
        model = StubModel([0.01, 0.08, 0.2, 0.4])
        features = sorted([c for c in df.columns if c != LABEL_COLUMN])

        result = score_with_policy(
            df=df,
            model=model,
            model_features=features,
            approve_threshold=0.08,
            decline_threshold=0.30,
        )

        self.assertListEqual(result["decision"].tolist(), ["approve", "review", "review", "decline"])
        self.assertListEqual(result["predicted_class"].tolist(), [0, 0, 0, 1])

    def test_resolve_feature_columns_raises_if_artifact_feature_missing(self) -> None:
        df = make_valid_dataframe(rows=2, include_label=False)
        df = df.drop(columns=["V3"])
        with self.assertRaises(ValueError):
            resolve_feature_columns(df, ["Time", "Amount", "V3"])

    def test_run_inference_end_to_end_writes_scored_csv(self) -> None:
        input_df = make_valid_dataframe(rows=3, include_label=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = tmp / "input.csv"
            model_path = tmp / "model.joblib"
            output_path = tmp / "output.csv"
            policy_config_path = tmp / "policy_config.json"
            input_df.to_csv(input_path, index=False)
            write_policy_config(policy_config_path)

            artifact = {
                "model": StubModel([0.02, 0.15, 0.85]),
                "feature_columns": sorted(input_df.columns.tolist()),
                "threshold_for_target_fpr": 0.4,
                "target_fpr": 0.02,
            }
            joblib.dump(artifact, model_path)

            scored_df, approve_t, decline_t = run_inference(
                input_path=input_path,
                model_path=model_path,
                output_path=output_path,
                policy_profile="artifact",
                approve_threshold=0.1,
                decline_threshold=None,
                policy_config_path=policy_config_path,
            )

            self.assertTrue(output_path.exists())
            self.assertAlmostEqual(approve_t, 0.1)
            self.assertAlmostEqual(decline_t, 0.4)
            self.assertIn("score", scored_df.columns)
            self.assertIn("decision", scored_df.columns)
            self.assertListEqual(scored_df["decision"].tolist(), ["approve", "review", "decline"])

    def test_run_inference_primary_profile_defaults(self) -> None:
        input_df = make_valid_dataframe(rows=3, include_label=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = tmp / "input.csv"
            model_path = tmp / "model.joblib"
            policy_config_path = tmp / "policy_config.json"
            input_df.to_csv(input_path, index=False)
            write_policy_config(policy_config_path)
            artifact = {
                "model": StubModel([0.05, 0.15, 0.85]),
                "feature_columns": sorted(input_df.columns.tolist()),
                "threshold_for_target_fpr": 0.4,
                "target_fpr": 0.02,
            }
            joblib.dump(artifact, model_path)

            scored_df, approve_t, decline_t = run_inference(
                input_path=input_path,
                model_path=model_path,
                output_path=None,
                policy_profile="primary",
                approve_threshold=None,
                decline_threshold=None,
                policy_config_path=policy_config_path,
            )

            self.assertAlmostEqual(approve_t, 0.11)
            self.assertAlmostEqual(decline_t, 0.45)
            self.assertListEqual(scored_df["decision"].tolist(), ["approve", "review", "decline"])

    def test_run_inference_fallback_profile_defaults(self) -> None:
        input_df = make_valid_dataframe(rows=3, include_label=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = tmp / "input.csv"
            model_path = tmp / "model.joblib"
            policy_config_path = tmp / "policy_config.json"
            input_df.to_csv(input_path, index=False)
            write_policy_config(policy_config_path)
            artifact = {
                "model": StubModel([0.05, 0.15, 0.85]),
                "feature_columns": sorted(input_df.columns.tolist()),
                "threshold_for_target_fpr": 0.4,
                "target_fpr": 0.02,
            }
            joblib.dump(artifact, model_path)

            scored_df, approve_t, decline_t = run_inference(
                input_path=input_path,
                model_path=model_path,
                output_path=None,
                policy_profile="fallback",
                approve_threshold=None,
                decline_threshold=None,
                policy_config_path=policy_config_path,
            )

            self.assertAlmostEqual(approve_t, 0.19)
            self.assertAlmostEqual(decline_t, 0.80)
            self.assertListEqual(scored_df["decision"].tolist(), ["approve", "approve", "decline"])

    def test_run_inference_phase2_guarded_profile_defaults(self) -> None:
        input_df = make_valid_dataframe(rows=3, include_label=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = tmp / "input.csv"
            model_path = tmp / "model.joblib"
            policy_config_path = tmp / "policy_config.json"
            input_df.to_csv(input_path, index=False)
            write_policy_config(policy_config_path)
            artifact = {
                "model": StubModel([0.05, 0.25, 0.85]),
                "feature_columns": sorted(input_df.columns.tolist()),
                "threshold_for_target_fpr": 0.4,
                "target_fpr": 0.02,
            }
            joblib.dump(artifact, model_path)

            scored_df, approve_t, decline_t = run_inference(
                input_path=input_path,
                model_path=model_path,
                output_path=None,
                policy_profile="phase2_guarded",
                approve_threshold=None,
                decline_threshold=None,
                policy_config_path=policy_config_path,
            )

            self.assertAlmostEqual(approve_t, 0.20)
            self.assertAlmostEqual(decline_t, 0.80)
            self.assertListEqual(scored_df["decision"].tolist(), ["approve", "review", "decline"])

    def test_run_inference_default_profile_is_phase2_guarded(self) -> None:
        input_df = make_valid_dataframe(rows=3, include_label=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = tmp / "input.csv"
            model_path = tmp / "model.joblib"
            policy_config_path = tmp / "policy_config.json"
            input_df.to_csv(input_path, index=False)
            write_policy_config(policy_config_path)
            artifact = {
                "model": StubModel([0.05, 0.25, 0.85]),
                "feature_columns": sorted(input_df.columns.tolist()),
                "threshold_for_target_fpr": 0.4,
                "target_fpr": 0.02,
            }
            joblib.dump(artifact, model_path)

            scored_df, approve_t, decline_t = run_inference(
                input_path=input_path,
                model_path=model_path,
                output_path=None,
                approve_threshold=None,
                decline_threshold=None,
                policy_config_path=policy_config_path,
            )

            self.assertAlmostEqual(approve_t, 0.20)
            self.assertAlmostEqual(decline_t, 0.80)
            self.assertListEqual(scored_df["decision"].tolist(), ["approve", "review", "decline"])

    def test_run_inference_raises_for_invalid_threshold_order(self) -> None:
        input_df = make_valid_dataframe(rows=2, include_label=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = tmp / "input.csv"
            model_path = tmp / "model.joblib"
            policy_config_path = tmp / "policy_config.json"
            input_df.to_csv(input_path, index=False)
            write_policy_config(policy_config_path)
            artifact = {
                "model": StubModel([0.2, 0.8]),
                "feature_columns": sorted(input_df.columns.tolist()),
                "threshold_for_target_fpr": 0.4,
                "target_fpr": 0.02,
            }
            joblib.dump(artifact, model_path)

            with self.assertRaises(ValueError):
                run_inference(
                    input_path=input_path,
                    model_path=model_path,
                    output_path=None,
                    policy_profile="primary",
                    approve_threshold=0.7,
                    decline_threshold=0.6,
                    policy_config_path=policy_config_path,
                )

    def test_run_inference_raises_for_missing_required_input_columns(self) -> None:
        input_df = make_valid_dataframe(rows=2, include_label=False).drop(columns=["V7"])
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = tmp / "input_missing.csv"
            model_path = tmp / "model.joblib"
            policy_config_path = tmp / "policy_config.json"
            input_df.to_csv(input_path, index=False)
            write_policy_config(policy_config_path)
            artifact = {
                "model": StubModel([0.1, 0.2]),
                "feature_columns": sorted([c for c in input_df.columns]),
                "threshold_for_target_fpr": 0.4,
                "target_fpr": 0.02,
            }
            joblib.dump(artifact, model_path)

            with self.assertRaises(ValueError):
                run_inference(
                    input_path=input_path,
                    model_path=model_path,
                    output_path=None,
                    policy_profile="primary",
                    approve_threshold=None,
                    decline_threshold=None,
                    policy_config_path=policy_config_path,
                )

    def test_load_model_bundle_from_dict_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "bundle.joblib"
            artifact = {
                "model": StubModel([0.2]),
                "feature_columns": ["Amount", "Time"],
                "threshold_for_target_fpr": 0.35,
                "target_fpr": 0.02,
            }
            joblib.dump(artifact, model_path)

            bundle = load_model_bundle(model_path)
            self.assertEqual(bundle.artifact_feature_columns, ["Amount", "Time"])
            self.assertAlmostEqual(bundle.artifact_decline_threshold or 0.0, 0.35)
            self.assertAlmostEqual(bundle.target_fpr or 0.0, 0.02)

    def test_run_inference_artifact_profile_does_not_require_policy_config(self) -> None:
        input_df = make_valid_dataframe(rows=2, include_label=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = tmp / "input.csv"
            model_path = tmp / "model.joblib"
            missing_policy_path = tmp / "missing_policy_config.json"
            input_df.to_csv(input_path, index=False)
            artifact = {
                "model": StubModel([0.02, 0.90]),
                "feature_columns": sorted(input_df.columns.tolist()),
                "threshold_for_target_fpr": 0.4,
                "target_fpr": 0.02,
            }
            joblib.dump(artifact, model_path)

            scored_df, approve_t, decline_t = run_inference(
                input_path=input_path,
                model_path=model_path,
                output_path=None,
                policy_profile="artifact",
                approve_threshold=None,
                decline_threshold=None,
                policy_config_path=missing_policy_path,
            )

            self.assertAlmostEqual(approve_t, 0.11)
            self.assertAlmostEqual(decline_t, 0.4)
            self.assertListEqual(scored_df["decision"].tolist(), ["approve", "decline"])

    def test_run_inference_raises_for_unsupported_policy_schema_version(self) -> None:
        input_df = make_valid_dataframe(rows=2, include_label=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = tmp / "input.csv"
            model_path = tmp / "model.joblib"
            policy_config_path = tmp / "policy_config.json"
            input_df.to_csv(input_path, index=False)

            bad_payload = {
                "schema_version": 2,
                "profiles": {
                    "primary": {"approve_threshold": 0.11, "decline_threshold": 0.45},
                },
            }
            policy_config_path.write_text(json.dumps(bad_payload), encoding="utf-8")

            artifact = {
                "model": StubModel([0.2, 0.8]),
                "feature_columns": sorted(input_df.columns.tolist()),
                "threshold_for_target_fpr": 0.4,
                "target_fpr": 0.02,
            }
            joblib.dump(artifact, model_path)

            with self.assertRaises(ValueError):
                run_inference(
                    input_path=input_path,
                    model_path=model_path,
                    output_path=None,
                    policy_profile="primary",
                    approve_threshold=None,
                    decline_threshold=None,
                    policy_config_path=policy_config_path,
                )

    def test_policy_status_snapshot_for_phase2_guarded(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            policy_config_path = Path(tmpdir) / "policy_config.json"
            write_policy_config(policy_config_path)
            snapshot = policy_status_snapshot(
                policy_config_path=policy_config_path,
                policy_profile="phase2_guarded",
            )
            self.assertEqual(snapshot["schema_version"], 1)
            self.assertEqual(snapshot["default_policy_profile"], "phase2_guarded")
            selected = snapshot["selected_profile_thresholds"]
            self.assertAlmostEqual(float(selected["approve_threshold"]), 0.20)
            self.assertAlmostEqual(float(selected["decline_threshold"]), 0.80)

    def test_build_policy_status_text_contains_expected_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            policy_config_path = Path(tmpdir) / "policy_config.json"
            write_policy_config(policy_config_path)
            snapshot = policy_status_snapshot(
                policy_config_path=policy_config_path,
                policy_profile="primary",
            )
            text = build_policy_status_text(snapshot)
            self.assertIn("=== Policy Status ===", text)
            self.assertIn("Default profile: phase2_guarded", text)
            self.assertIn("Selected profile: primary", text)


if __name__ == "__main__":
    unittest.main()
