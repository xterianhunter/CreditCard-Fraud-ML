"""Phase 2: calibration + cost-based policy optimization for T1/T2.

Usage:
    python src/cost_policy_optimization.py --data-path data/creditcard.csv --model-path models/week2_best_logreg.joblib
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from data_contract import LABEL_COLUMN
from inference import load_model_bundle, resolve_feature_columns, validate_thresholds
from train_baseline import load_data, split_train_test


def parse_methods(raw: str) -> list[str]:
    allowed = {"none", "platt", "isotonic"}
    methods = [m.strip().lower() for m in raw.split(",") if m.strip()]
    if not methods:
        raise ValueError("At least one calibration method must be provided.")
    invalid = [m for m in methods if m not in allowed]
    if invalid:
        raise ValueError(f"Unsupported calibration methods: {invalid}. Allowed: {sorted(allowed)}")
    return methods


def split_calibration_eval(
    test_df: pd.DataFrame,
    calibration_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not (0.1 <= calibration_fraction <= 0.9):
        raise ValueError("--calibration-fraction must be in [0.1, 0.9].")
    split_idx = int(len(test_df) * calibration_fraction)
    if split_idx <= 0 or split_idx >= len(test_df):
        raise ValueError("Calibration split produced empty partition.")
    return test_df.iloc[:split_idx], test_df.iloc[split_idx:]


def calibrate_scores(
    *,
    method: str,
    calibration_scores: np.ndarray,
    calibration_labels: np.ndarray,
    evaluation_scores: np.ndarray,
) -> np.ndarray:
    """Return calibrated evaluation scores for chosen method."""
    if method == "none":
        return evaluation_scores.astype(np.float64)

    unique = np.unique(calibration_labels)
    if len(unique) < 2:
        raise ValueError(
            f"Calibration requires both classes in calibration labels, got classes={unique.tolist()}."
        )

    if method == "platt":
        calibrator = LogisticRegression(random_state=42, max_iter=2000)
        calibrator.fit(calibration_scores.reshape(-1, 1), calibration_labels)
        return calibrator.predict_proba(evaluation_scores.reshape(-1, 1))[:, 1].astype(np.float64)

    if method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(calibration_scores, calibration_labels)
        return calibrator.predict(evaluation_scores).astype(np.float64)

    raise ValueError(f"Unsupported method: {method}")


def evaluate_policy_cost(
    *,
    y_true: np.ndarray,
    y_score: np.ndarray,
    approve_threshold: float,
    decline_threshold: float,
    cost_false_negative: float,
    cost_false_decline: float,
    cost_review: float,
) -> dict[str, float]:
    """Compute policy metrics and cost for one threshold pair."""
    validate_thresholds(approve_threshold, decline_threshold)
    n = len(y_true)
    if n == 0:
        raise ValueError("Cannot evaluate policy on empty arrays.")
    if len(y_score) != n:
        raise ValueError("y_true and y_score length mismatch.")

    fraud_mask = y_true == 1
    nonfraud_mask = y_true == 0

    approve_mask = y_score < approve_threshold
    review_mask = (y_score >= approve_threshold) & (y_score < decline_threshold)
    decline_mask = y_score >= decline_threshold

    fraud_approve = int(np.sum(fraud_mask & approve_mask))
    fraud_review = int(np.sum(fraud_mask & review_mask))
    fraud_decline = int(np.sum(fraud_mask & decline_mask))
    nonfraud_decline = int(np.sum(nonfraud_mask & decline_mask))

    approve_count = int(np.sum(approve_mask))
    review_count = int(np.sum(review_mask))
    decline_count = int(np.sum(decline_mask))
    fraud_total = int(np.sum(fraud_mask))
    nonfraud_total = int(np.sum(nonfraud_mask))

    total_cost = (
        fraud_approve * cost_false_negative
        + nonfraud_decline * cost_false_decline
        + review_count * cost_review
    )

    flagged_capture = (fraud_review + fraud_decline) / fraud_total if fraud_total else 0.0
    decline_fpr = nonfraud_decline / nonfraud_total if nonfraud_total else 0.0

    return {
        "approve_threshold": float(approve_threshold),
        "decline_threshold": float(decline_threshold),
        "approve_count": float(approve_count),
        "review_count": float(review_count),
        "decline_count": float(decline_count),
        "approve_rate": float(approve_count / n),
        "review_rate": float(review_count / n),
        "decline_rate": float(decline_count / n),
        "fraud_total": float(fraud_total),
        "fraud_approve": float(fraud_approve),
        "fraud_review": float(fraud_review),
        "fraud_decline": float(fraud_decline),
        "flagged_fraud_capture": float(flagged_capture),
        "decline_fpr": float(decline_fpr),
        "total_cost": float(total_cost),
        "cost_per_row": float(total_cost / n),
    }


def generate_threshold_grid(
    *,
    t1_min: float,
    t1_max: float,
    t2_min: float,
    t2_max: float,
    step: float,
) -> list[tuple[float, float]]:
    if step <= 0:
        raise ValueError("--grid-step must be > 0.")
    t1_values = np.round(np.arange(t1_min, t1_max + 1e-9, step), 4)
    t2_values = np.round(np.arange(t2_min, t2_max + 1e-9, step), 4)
    pairs: list[tuple[float, float]] = []
    for t1 in t1_values:
        for t2 in t2_values:
            if t1 >= t2:
                continue
            pairs.append((float(t1), float(t2)))
    if not pairs:
        raise ValueError("Threshold grid generated zero valid pairs.")
    return pairs


def search_best_policy(
    *,
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold_pairs: list[tuple[float, float]],
    cost_false_negative: float,
    cost_false_decline: float,
    cost_review: float,
) -> tuple[pd.Series, pd.DataFrame]:
    rows: list[dict[str, float]] = []
    for t1, t2 in threshold_pairs:
        metrics = evaluate_policy_cost(
            y_true=y_true,
            y_score=y_score,
            approve_threshold=t1,
            decline_threshold=t2,
            cost_false_negative=cost_false_negative,
            cost_false_decline=cost_false_decline,
            cost_review=cost_review,
        )
        rows.append(metrics)

    result = pd.DataFrame(rows)
    ranked = result.sort_values(
        by=["total_cost", "flagged_fraud_capture", "review_rate", "decline_fpr"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked.iloc[0], ranked


def build_markdown_report(
    *,
    train_rows: int,
    calibration_rows: int,
    evaluation_rows: int,
    model_path: Path,
    methods_summary: pd.DataFrame,
    best_method_row: pd.Series,
    grid_step: float,
    cost_false_negative: float,
    cost_false_decline: float,
    cost_review: float,
) -> str:
    lines = [
        "# Phase 2: Calibration + Cost-Based Policy Optimization",
        "",
        "## Run Details",
        f"- Date: {date.today().isoformat()}",
        f"- Model path: `{model_path}`",
        f"- Train rows: {train_rows}",
        f"- Calibration rows: {calibration_rows}",
        f"- Evaluation rows: {evaluation_rows}",
        f"- Grid step: {grid_step:.4f}",
        "",
        "## Cost Function",
        f"- Cost(false negative approve): {cost_false_negative:.3f}",
        f"- Cost(false decline non-fraud): {cost_false_decline:.3f}",
        f"- Cost(per review): {cost_review:.3f}",
        "",
        "## Best Policy By Calibration Method",
        "",
        "| Method | T1 | T2 | Total Cost | Cost/Row | Review% | Decline FPR | Flagged Fraud Capture |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in methods_summary.iterrows():
        lines.append(
            "| {method} | {t1:.4f} | {t2:.4f} | {total:.4f} | {cpr:.6f} | {review:.2%} | {fpr:.2%} | {capture:.2%} |".format(
                method=row["method"],
                t1=float(row["approve_threshold"]),
                t2=float(row["decline_threshold"]),
                total=float(row["total_cost"]),
                cpr=float(row["cost_per_row"]),
                review=float(row["review_rate"]),
                fpr=float(row["decline_fpr"]),
                capture=float(row["flagged_fraud_capture"]),
            )
        )

    lines.extend(
        [
            "",
            "## Recommended Policy",
            f"- Calibration method: `{best_method_row['method']}`",
            f"- `approve`: score < {float(best_method_row['approve_threshold']):.4f}",
            f"- `review`: {float(best_method_row['approve_threshold']):.4f} <= score < {float(best_method_row['decline_threshold']):.4f}",
            f"- `decline`: score >= {float(best_method_row['decline_threshold']):.4f}",
            f"- Expected review rate: {float(best_method_row['review_rate']):.2%}",
            f"- Expected decline FPR: {float(best_method_row['decline_fpr']):.2%}",
            f"- Expected flagged fraud capture: {float(best_method_row['flagged_fraud_capture']):.2%}",
            f"- Total cost: {float(best_method_row['total_cost']):.4f}",
            f"- Cost per row: {float(best_method_row['cost_per_row']):.6f}",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=Path("data/creditcard.csv"))
    parser.add_argument("--model-path", type=Path, default=Path("models/week2_best_logreg.joblib"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--methods", type=str, default="none,platt,isotonic")
    parser.add_argument("--grid-step", type=float, default=0.01)
    parser.add_argument("--t1-min", type=float, default=0.02)
    parser.add_argument("--t1-max", type=float, default=0.20)
    parser.add_argument("--t2-min", type=float, default=0.20)
    parser.add_argument("--t2-max", type=float, default=0.80)
    parser.add_argument("--cost-false-negative", type=float, default=25.0)
    parser.add_argument("--cost-false-decline", type=float, default=5.0)
    parser.add_argument("--cost-review", type=float, default=0.4)
    parser.add_argument(
        "--results-out-path",
        type=Path,
        default=Path("reports/phase2_cost_policy_results.csv"),
    )
    parser.add_argument(
        "--report-out-path",
        type=Path,
        default=Path("reports/phase2_cost_policy_report.md"),
    )
    args = parser.parse_args()

    methods = parse_methods(args.methods)
    threshold_pairs = generate_threshold_grid(
        t1_min=args.t1_min,
        t1_max=args.t1_max,
        t2_min=args.t2_min,
        t2_max=args.t2_max,
        step=args.grid_step,
    )

    df = load_data(args.data_path)
    train_df, test_df = split_train_test(df, test_size=args.test_size)
    calibration_df, evaluation_df = split_calibration_eval(test_df, args.calibration_fraction)

    bundle = load_model_bundle(args.model_path)
    model_features = resolve_feature_columns(test_df, bundle.artifact_feature_columns)
    x_cal = calibration_df[model_features].to_numpy(dtype=np.float32)
    y_cal = calibration_df[LABEL_COLUMN].to_numpy(dtype=np.int8)
    x_eval = evaluation_df[model_features].to_numpy(dtype=np.float32)
    y_eval = evaluation_df[LABEL_COLUMN].to_numpy(dtype=np.int8)

    calibration_scores = bundle.model.predict_proba(x_cal)[:, 1]
    evaluation_scores_raw = bundle.model.predict_proba(x_eval)[:, 1]

    method_rows: list[pd.Series] = []
    detailed_rows: list[pd.DataFrame] = []

    for method in methods:
        eval_scores = calibrate_scores(
            method=method,
            calibration_scores=calibration_scores,
            calibration_labels=y_cal,
            evaluation_scores=evaluation_scores_raw,
        )
        best, ranked = search_best_policy(
            y_true=y_eval,
            y_score=eval_scores,
            threshold_pairs=threshold_pairs,
            cost_false_negative=args.cost_false_negative,
            cost_false_decline=args.cost_false_decline,
            cost_review=args.cost_review,
        )
        best = best.copy()
        best["method"] = method
        method_rows.append(best)

        ranked = ranked.copy()
        ranked["method"] = method
        detailed_rows.append(ranked)

    methods_summary = pd.DataFrame(method_rows).sort_values(
        by=["total_cost", "flagged_fraud_capture", "review_rate", "decline_fpr"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)
    best_method_row = methods_summary.iloc[0]

    full_results = pd.concat(detailed_rows, ignore_index=True)
    args.results_out_path.parent.mkdir(parents=True, exist_ok=True)
    full_results.to_csv(args.results_out_path, index=False)

    report = build_markdown_report(
        train_rows=len(train_df),
        calibration_rows=len(calibration_df),
        evaluation_rows=len(evaluation_df),
        model_path=args.model_path,
        methods_summary=methods_summary,
        best_method_row=best_method_row,
        grid_step=args.grid_step,
        cost_false_negative=args.cost_false_negative,
        cost_false_decline=args.cost_false_decline,
        cost_review=args.cost_review,
    )
    args.report_out_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_out_path.write_text(report, encoding="utf-8")

    print("=== Phase 2 Cost Policy Optimization Summary ===")
    print(f"Methods evaluated: {methods}")
    print(f"Threshold pairs per method: {len(threshold_pairs)}")
    print(f"Best method: {best_method_row['method']}")
    print(
        "Best T1/T2: "
        f"{float(best_method_row['approve_threshold']):.4f}/{float(best_method_row['decline_threshold']):.4f}"
    )
    print(f"Best total cost: {float(best_method_row['total_cost']):.4f}")
    print(f"Best cost/row: {float(best_method_row['cost_per_row']):.6f}")
    print(f"Wrote detailed results: {args.results_out_path}")
    print(f"Wrote report: {args.report_out_path}")


if __name__ == "__main__":
    main()
