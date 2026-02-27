"""Minimal inference CLI with schema validation and threshold policy decisions.

Usage:
    python src/inference.py --input-path data/creditcard.csv --output-path reports/inference_output.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from data_contract import feature_columns, validate_dataframe


DEFAULT_APPROVE_THRESHOLD = 0.08
DEFAULT_DECLINE_THRESHOLD = 0.50


@dataclass
class ModelBundle:
    model: Any
    artifact_feature_columns: list[str] | None
    artifact_decline_threshold: float | None
    target_fpr: float | None


def load_model_bundle(model_path: Path) -> ModelBundle:
    """Load model artifact from joblib (dict bundle or plain estimator)."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    artifact = joblib.load(model_path)

    if isinstance(artifact, dict):
        model = artifact.get("model")
        if model is None:
            raise ValueError("Invalid model artifact: missing `model` in dict bundle.")

        feature_cols = artifact.get("feature_columns")
        if feature_cols is not None and not isinstance(feature_cols, list):
            raise ValueError("Invalid model artifact: `feature_columns` must be a list if provided.")

        decline_threshold = artifact.get("threshold_for_target_fpr")
        if decline_threshold is not None:
            decline_threshold = float(decline_threshold)

        target_fpr = artifact.get("target_fpr")
        if target_fpr is not None:
            target_fpr = float(target_fpr)

        return ModelBundle(
            model=model,
            artifact_feature_columns=feature_cols,
            artifact_decline_threshold=decline_threshold,
            target_fpr=target_fpr,
        )

    if not hasattr(artifact, "predict_proba"):
        raise ValueError("Model artifact must support `predict_proba`.")

    return ModelBundle(
        model=artifact,
        artifact_feature_columns=None,
        artifact_decline_threshold=None,
        target_fpr=None,
    )


def resolve_feature_columns(df: pd.DataFrame, artifact_feature_columns: list[str] | None) -> list[str]:
    """Resolve feature set, preferring artifact feature order when available."""
    if artifact_feature_columns:
        missing = sorted(set(artifact_feature_columns) - set(df.columns))
        if missing:
            raise ValueError(f"Input data missing model feature columns: {missing}")
        return artifact_feature_columns
    return feature_columns(df.columns)


def validate_thresholds(approve_threshold: float, decline_threshold: float) -> None:
    if not (0.0 <= approve_threshold < decline_threshold <= 1.0):
        raise ValueError(
            "Thresholds must satisfy 0 <= approve_threshold < decline_threshold <= 1."
        )


def score_with_policy(
    df: pd.DataFrame,
    model: Any,
    model_features: list[str],
    approve_threshold: float,
    decline_threshold: float,
) -> pd.DataFrame:
    """Return scored dataframe with policy actions."""
    x = df[model_features].to_numpy(dtype=np.float32)
    score = model.predict_proba(x)[:, 1]

    decision = np.where(
        score < approve_threshold,
        "approve",
        np.where(score < decline_threshold, "review", "decline"),
    )
    predicted_class = (score >= decline_threshold).astype(np.int8)

    return pd.DataFrame(
        {
            "score": score.astype(np.float64),
            "decision": decision,
            "predicted_class": predicted_class,
        }
    )


def run_inference(
    *,
    input_path: Path,
    model_path: Path,
    output_path: Path | None,
    approve_threshold: float | None,
    decline_threshold: float | None,
) -> tuple[pd.DataFrame, float, float]:
    """Load data + model and return scored output with resolved thresholds."""
    df = pd.read_csv(input_path)
    validate_dataframe(df, require_label=False)

    bundle = load_model_bundle(model_path)
    final_decline_threshold = (
        decline_threshold
        if decline_threshold is not None
        else (
            bundle.artifact_decline_threshold
            if bundle.artifact_decline_threshold is not None
            else DEFAULT_DECLINE_THRESHOLD
        )
    )
    final_approve_threshold = (
        approve_threshold
        if approve_threshold is not None
        else min(DEFAULT_APPROVE_THRESHOLD, final_decline_threshold * 0.5)
    )
    validate_thresholds(final_approve_threshold, final_decline_threshold)

    model_features = resolve_feature_columns(df, bundle.artifact_feature_columns)
    scored = score_with_policy(
        df=df,
        model=bundle.model,
        model_features=model_features,
        approve_threshold=final_approve_threshold,
        decline_threshold=final_decline_threshold,
    )
    out = pd.concat([df.reset_index(drop=True), scored], axis=1)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_path, index=False)

    return out, final_approve_threshold, final_decline_threshold


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=Path, required=True, help="CSV input to score.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/baseline_logreg.joblib"),
        help="Path to model artifact.",
    )
    parser.add_argument("--output-path", type=Path, default=None, help="Where to write scored CSV.")
    parser.add_argument("--approve-threshold", type=float, default=None, help="Approve upper bound.")
    parser.add_argument("--decline-threshold", type=float, default=None, help="Decline lower bound.")
    args = parser.parse_args()

    scored_df, approve_threshold, decline_threshold = run_inference(
        input_path=args.input_path,
        model_path=args.model_path,
        output_path=args.output_path,
        approve_threshold=args.approve_threshold,
        decline_threshold=args.decline_threshold,
    )

    print("=== Inference Summary ===")
    print(f"Rows scored: {len(scored_df)}")
    print(f"Approve threshold: {approve_threshold:.6f}")
    print(f"Decline threshold: {decline_threshold:.6f}")
    counts = scored_df["decision"].value_counts().to_dict()
    print(f"Decision counts: {counts}")
    if args.output_path is not None:
        print(f"Wrote scored output to: {args.output_path}")


if __name__ == "__main__":
    main()
