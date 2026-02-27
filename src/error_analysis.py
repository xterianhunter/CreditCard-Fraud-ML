"""Generate Week 2 error analysis outputs for the fraud baseline model.

Usage:
    python src/error_analysis.py --data-path data/creditcard.csv
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from data_contract import LABEL_COLUMN
from inference import load_model_bundle, resolve_feature_columns
from train_baseline import load_data, recall_at_fpr, split_train_test


def compute_confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    """Return standard binary confusion counts."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def select_top_errors(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    top_n: int,
) -> pd.DataFrame:
    """Return top false positives and false negatives at a selected threshold."""
    y_pred = (y_score >= threshold).astype(np.int8)
    work = test_df.reset_index().rename(columns={"index": "row_id"}).copy()
    work["score"] = y_score
    work["predicted_class"] = y_pred

    false_positive = work[(y_true == 0) & (y_pred == 1)].sort_values("score", ascending=False).head(top_n)
    false_negative = work[(y_true == 1) & (y_pred == 0)].sort_values("score", ascending=True).head(top_n)

    false_positive = false_positive.assign(error_type="false_positive")
    false_negative = false_negative.assign(error_type="false_negative")
    combined = pd.concat([false_positive, false_negative], ignore_index=True)
    return combined


def build_report_markdown(
    *,
    train_rows: int,
    test_rows: int,
    target_fpr: float,
    threshold: float,
    counts: dict[str, int],
    exported_errors: int,
    top_error_rows: pd.DataFrame,
) -> str:
    """Build markdown report for week 2 error analysis."""
    tn, fp, fn, tp = counts["tn"], counts["fp"], counts["fn"], counts["tp"]
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    realized_fpr = fp / (fp + tn) if (fp + tn) else 0.0

    lines = [
        "# Error Analysis Report (Week 2)",
        "",
        "## Run Details",
        f"- Date: {date.today().isoformat()}",
        f"- Train rows: {train_rows}",
        f"- Test rows: {test_rows}",
        f"- Threshold source target FPR: {target_fpr:.2%}",
        f"- Selected threshold: {threshold:.6f}",
        "",
        "## Confusion Matrix (at selected threshold)",
        f"- TN: {tn:,}",
        f"- FP: {fp:,}",
        f"- FN: {fn:,}",
        f"- TP: {tp:,}",
        "",
        "## Derived Metrics",
        f"- Recall: {recall:.6f}",
        f"- Precision: {precision:.6f}",
        f"- Realized FPR: {realized_fpr:.6f}",
        "",
        "## Error Sample Export",
        f"- Exported rows (FP + FN): {exported_errors}",
        "- Columns include `row_id`, `Time`, `Amount`, `Class`, `predicted_class`, `score`, `error_type`.",
        "",
        "## Top Error Samples (first 5)",
    ]

    sample_cols = [c for c in ["row_id", "Time", "Amount", "Class", "predicted_class", "score", "error_type"] if c in top_error_rows.columns]
    preview = top_error_rows[sample_cols].head(5)
    if preview.empty:
        lines.append("- No errors at this threshold.")
    else:
        for _, row in preview.iterrows():
            lines.append(
                "- row_id={row_id}, Time={time_val}, Amount={amount}, actual={actual}, pred={pred}, score={score:.6f}, type={etype}".format(
                    row_id=int(row["row_id"]) if not pd.isna(row["row_id"]) else -1,
                    time_val=float(row["Time"]) if "Time" in preview.columns else float("nan"),
                    amount=float(row["Amount"]) if "Amount" in preview.columns else float("nan"),
                    actual=int(row["Class"]) if "Class" in preview.columns else -1,
                    pred=int(row["predicted_class"]) if "predicted_class" in preview.columns else -1,
                    score=float(row["score"]) if "score" in preview.columns else 0.0,
                    etype=str(row["error_type"]) if "error_type" in preview.columns else "unknown",
                )
            )

    lines.append("")
    lines.append("## Interpretation Notes")
    lines.append("- Review high-score false positives to identify segments causing customer friction.")
    lines.append("- Review low-score false negatives to identify missed fraud patterns.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=Path("data/creditcard.csv"))
    parser.add_argument("--model-path", type=Path, default=Path("models/baseline_logreg.joblib"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--target-fpr", type=float, default=0.02)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument(
        "--errors-out-path",
        type=Path,
        default=Path("reports/error_samples_week2.csv"),
        help="CSV path for top FP/FN samples.",
    )
    parser.add_argument(
        "--report-out-path",
        type=Path,
        default=Path("reports/error_analysis_week2.md"),
        help="Markdown path for summary report.",
    )
    args = parser.parse_args()

    if args.top_n <= 0:
        raise ValueError("--top-n must be >= 1")

    df = load_data(args.data_path)
    train_df, test_df = split_train_test(df, test_size=args.test_size)
    y_true = test_df[LABEL_COLUMN].to_numpy(dtype=np.int8)

    bundle = load_model_bundle(args.model_path)
    model_features = resolve_feature_columns(test_df, bundle.artifact_feature_columns)
    x_test = test_df[model_features].to_numpy(dtype=np.float32)
    y_score = bundle.model.predict_proba(x_test)[:, 1]

    if args.threshold is not None:
        threshold = float(args.threshold)
    elif bundle.artifact_decline_threshold is not None:
        threshold = float(bundle.artifact_decline_threshold)
    else:
        _, threshold = recall_at_fpr(y_true, y_score, target_fpr=args.target_fpr)

    y_pred = (y_score >= threshold).astype(np.int8)
    counts = compute_confusion_counts(y_true, y_pred)
    errors = select_top_errors(test_df=test_df, y_true=y_true, y_score=y_score, threshold=threshold, top_n=args.top_n)

    export_cols = [c for c in ["row_id", "Time", "Amount", "Class", "predicted_class", "score", "error_type"] if c in errors.columns]
    args.errors_out_path.parent.mkdir(parents=True, exist_ok=True)
    errors[export_cols].to_csv(args.errors_out_path, index=False)

    report_text = build_report_markdown(
        train_rows=len(train_df),
        test_rows=len(test_df),
        target_fpr=args.target_fpr,
        threshold=threshold,
        counts=counts,
        exported_errors=len(errors),
        top_error_rows=errors,
    )
    args.report_out_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_out_path.write_text(report_text, encoding="utf-8")

    print("=== Error Analysis Summary ===")
    print(f"Rows train/test: {len(train_df)}/{len(test_df)}")
    print(f"Threshold used: {threshold:.6f}")
    print(f"Confusion matrix: TN={counts['tn']} FP={counts['fp']} FN={counts['fn']} TP={counts['tp']}")
    print(f"Wrote error samples: {args.errors_out_path}")
    print(f"Wrote report: {args.report_out_path}")


if __name__ == "__main__":
    main()
