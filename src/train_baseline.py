"""Train a low-memory baseline fraud model and print threshold-oriented metrics.

Usage:
    python src/train_baseline.py --data-path data/creditcard.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from data_contract import LABEL_COLUMN, TIME_COLUMN, feature_columns, validate_dataframe


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=Path("data/creditcard.csv"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--target-fpr", type=float, default=0.02)
    args = parser.parse_args()

    df = load_data(args.data_path)

    # Temporal split if Time exists; fallback to deterministic split.
    if TIME_COLUMN in df.columns:
        df = df.sort_values(TIME_COLUMN)
        split_idx = int(len(df) * (1 - args.test_size))
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
    else:
        train_df, test_df = train_test_split(df, test_size=args.test_size, random_state=42, stratify=df[LABEL_COLUMN])

    cols = feature_columns(train_df.columns)
    x_train = train_df[cols].to_numpy(dtype=np.float32)
    y_train = train_df[LABEL_COLUMN].to_numpy(dtype=np.int8)
    x_test = test_df[cols].to_numpy(dtype=np.float32)
    y_test = test_df[LABEL_COLUMN].to_numpy(dtype=np.int8)

    model = LogisticRegression(class_weight="balanced", max_iter=500, n_jobs=-1)
    model.fit(x_train, y_train)

    y_score = model.predict_proba(x_test)[:, 1]

    pr_auc = average_precision_score(y_test, y_score)
    roc_auc = roc_auc_score(y_test, y_score)
    recall, threshold = recall_at_fpr(y_test, y_score, target_fpr=args.target_fpr)

    print("=== Baseline Metrics ===")
    print(f"Rows train/test: {len(train_df)}/{len(test_df)}")
    print(f"PR-AUC: {pr_auc:.6f}")
    print(f"ROC-AUC: {roc_auc:.6f}")
    print(f"Recall @ FPR<={args.target_fpr:.2%}: {recall:.6f}")
    print(f"Threshold for target FPR: {threshold:.6f}")


if __name__ == "__main__":
    main()
