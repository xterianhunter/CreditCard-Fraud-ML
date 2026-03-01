"""Train a low-memory baseline fraud model and print threshold-oriented metrics.

Usage:
    python src/train_baseline.py --data-path data/creditcard.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from data_contract import LABEL_COLUMN, TIME_COLUMN, feature_columns, validate_dataframe
from mlflow_utils import log_artifact, log_metrics, log_params, parse_tags, start_mlflow_run


def recall_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float = 0.02) -> tuple[float, float]:
    """Return (recall, threshold) at highest threshold whose FPR <= target."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    valid = np.where(fpr <= target_fpr)[0]
    if len(valid) == 0:
        return 0.0, 1.0
    idx = valid[-1]
    return float(tpr[idx]), float(thresholds[idx])


def load_data(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    validate_dataframe(df, require_label=True)

    # Keep memory lower for laptop-scale training.
    float_cols = [c for c in df.columns if c != LABEL_COLUMN]
    df[float_cols] = df[float_cols].astype(np.float32)
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(np.int8)
    return df


def split_train_test(df: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Temporal split when available, deterministic stratified split otherwise."""
    if TIME_COLUMN in df.columns:
        df = df.sort_values(TIME_COLUMN)
        split_idx = int(len(df) * (1 - test_size))
        return df.iloc[:split_idx], df.iloc[split_idx:]

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=42,
        stratify=df[LABEL_COLUMN],
    )
    return train_df, test_df


def save_model_bundle(
    model: Any,
    model_feature_columns: list[str],
    target_fpr: float,
    threshold_for_target_fpr: float,
    model_out_path: Path,
) -> None:
    """Persist model plus metadata needed for downstream inference/policy."""
    model_out_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "feature_columns": model_feature_columns,
        "target_fpr": float(target_fpr),
        "threshold_for_target_fpr": float(threshold_for_target_fpr),
    }
    joblib.dump(artifact, model_out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=Path("data/creditcard.csv"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--target-fpr", type=float, default=0.02)
    parser.add_argument(
        "--model-out-path",
        type=Path,
        default=Path("models/baseline_logreg.joblib"),
        help="Path to save trained model bundle.",
    )
    parser.add_argument(
        "--skip-model-save",
        action="store_true",
        help="If set, do not write model artifact to disk.",
    )
    parser.add_argument("--mlflow", action="store_true", help="Enable MLflow tracking.")
    parser.add_argument("--mlflow-experiment", type=str, default=None)
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None)
    parser.add_argument("--mlflow-run-name", type=str, default=None)
    parser.add_argument(
        "--mlflow-tags",
        type=str,
        default=None,
        help="Comma-separated key=value tags for MLflow.",
    )
    args = parser.parse_args()

    tags = parse_tags(args.mlflow_tags)
    with start_mlflow_run(
        enabled=args.mlflow,
        experiment_name=args.mlflow_experiment,
        tracking_uri=args.mlflow_tracking_uri,
        run_name=args.mlflow_run_name or "train_baseline",
        tags=tags,
    ):
        log_params(
            args.mlflow,
            {
                "data_path": str(args.data_path),
                "test_size": float(args.test_size),
                "target_fpr": float(args.target_fpr),
                "model_out_path": str(args.model_out_path),
                "skip_model_save": bool(args.skip_model_save),
            },
        )

        df = load_data(args.data_path)
        train_df, test_df = split_train_test(df, test_size=args.test_size)

        cols = feature_columns(train_df.columns)
        x_train = train_df[cols].to_numpy(dtype=np.float32)
        y_train = train_df[LABEL_COLUMN].to_numpy(dtype=np.int8)
        x_test = test_df[cols].to_numpy(dtype=np.float32)
        y_test = test_df[LABEL_COLUMN].to_numpy(dtype=np.int8)

        model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        model.fit(x_train, y_train)

        y_score = model.predict_proba(x_test)[:, 1]

        pr_auc = average_precision_score(y_test, y_score)
        roc_auc = roc_auc_score(y_test, y_score)
        recall, threshold = recall_at_fpr(y_test, y_score, target_fpr=args.target_fpr)

        log_metrics(
            args.mlflow,
            {
                "pr_auc": float(pr_auc),
                "roc_auc": float(roc_auc),
                "recall_at_target_fpr": float(recall),
                "threshold_at_target_fpr": float(threshold),
            },
        )

        print("=== Baseline Metrics ===")
        print(f"Rows train/test: {len(train_df)}/{len(test_df)}")
        print(f"PR-AUC: {pr_auc:.6f}")
        print(f"ROC-AUC: {roc_auc:.6f}")
        print(f"Recall @ FPR<={args.target_fpr:.2%}: {recall:.6f}")
        print(f"Threshold for target FPR: {threshold:.6f}")
        if not args.skip_model_save:
            save_model_bundle(
                model=model,
                model_feature_columns=cols,
                target_fpr=args.target_fpr,
                threshold_for_target_fpr=threshold,
                model_out_path=args.model_out_path,
            )
            log_artifact(args.mlflow, str(args.model_out_path))
            print(f"Saved model artifact: {args.model_out_path}")


if __name__ == "__main__":
    main()
