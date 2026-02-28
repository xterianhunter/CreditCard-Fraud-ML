"""Week 3 policy simulation: evaluate two-threshold decision policies.

Usage:
    python src/policy_simulation.py --data-path data/creditcard.csv --model-path models/week2_best_logreg.joblib
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from data_contract import LABEL_COLUMN
from inference import load_model_bundle, resolve_feature_columns, validate_thresholds
from train_baseline import load_data, split_train_test


def parse_threshold_pairs(raw: str) -> list[tuple[float, float]]:
    """Parse threshold-pair string in format: '0.05:0.45,0.08:0.50'."""
    pairs: list[tuple[float, float]] = []
    for token in raw.split(","):
        part = token.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid threshold pair '{part}'. Expected format approve:decline.")
        approve_raw, decline_raw = part.split(":", 1)
        approve = float(approve_raw.strip())
        decline = float(decline_raw.strip())
        validate_thresholds(approve, decline)
        pairs.append((approve, decline))

    if not pairs:
        raise ValueError("At least one threshold pair must be provided.")
    return pairs


def default_threshold_pairs(base_decline_threshold: float) -> list[tuple[float, float]]:
    """Build a practical default threshold set around a baseline decline threshold."""
    base = float(np.clip(base_decline_threshold, 0.25, 0.90))
    raw_pairs = [
        (0.03, base - 0.06),
        (0.05, base - 0.04),
        (0.08, base),
        (0.10, base + 0.03),
        (0.12, base + 0.06),
    ]

    cleaned: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    for approve, decline in raw_pairs:
        approve_val = float(np.clip(approve, 0.0, 0.95))
        decline_val = float(np.clip(decline, 0.05, 0.98))
        if approve_val >= decline_val:
            continue
        pair = (round(approve_val, 4), round(decline_val, 4))
        if pair in seen:
            continue
        validate_thresholds(pair[0], pair[1])
        cleaned.append(pair)
        seen.add(pair)

    if len(cleaned) < 4:
        fallback = [(0.03, 0.40), (0.05, 0.45), (0.08, 0.50), (0.12, 0.60)]
        for pair in fallback:
            if pair not in seen:
                cleaned.append(pair)
                seen.add(pair)
            if len(cleaned) >= 4:
                break
    return cleaned


def evaluate_policy(
    y_true: np.ndarray,
    y_score: np.ndarray,
    approve_threshold: float,
    decline_threshold: float,
) -> dict[str, float]:
    """Evaluate one approve/review/decline threshold policy."""
    validate_thresholds(approve_threshold, decline_threshold)

    total = int(len(y_true))
    if total == 0:
        raise ValueError("Cannot evaluate policy on empty arrays.")

    if len(y_score) != total:
        raise ValueError("y_true and y_score must have the same length.")

    fraud_mask = y_true == 1
    nonfraud_mask = y_true == 0
    fraud_total = int(np.sum(fraud_mask))
    nonfraud_total = int(np.sum(nonfraud_mask))

    approve_mask = y_score < approve_threshold
    review_mask = (y_score >= approve_threshold) & (y_score < decline_threshold)
    decline_mask = y_score >= decline_threshold

    approve_count = int(np.sum(approve_mask))
    review_count = int(np.sum(review_mask))
    decline_count = int(np.sum(decline_mask))

    fraud_approve = int(np.sum(fraud_mask & approve_mask))
    fraud_review = int(np.sum(fraud_mask & review_mask))
    fraud_decline = int(np.sum(fraud_mask & decline_mask))

    nonfraud_decline = int(np.sum(nonfraud_mask & decline_mask))

    fraud_capture_decline = fraud_decline / fraud_total if fraud_total else 0.0
    fraud_capture_flagged = (fraud_review + fraud_decline) / fraud_total if fraud_total else 0.0
    decline_fpr = nonfraud_decline / nonfraud_total if nonfraud_total else 0.0
    review_precision = fraud_review / review_count if review_count else 0.0
    decline_precision = fraud_decline / decline_count if decline_count else 0.0

    return {
        "approve_threshold": float(approve_threshold),
        "decline_threshold": float(decline_threshold),
        "approve_count": float(approve_count),
        "review_count": float(review_count),
        "decline_count": float(decline_count),
        "approve_rate": approve_count / total,
        "review_rate": review_count / total,
        "decline_rate": decline_count / total,
        "flagged_rate": (review_count + decline_count) / total,
        "fraud_total": float(fraud_total),
        "fraud_approve": float(fraud_approve),
        "fraud_review": float(fraud_review),
        "fraud_decline": float(fraud_decline),
        "fraud_capture_decline": float(fraud_capture_decline),
        "fraud_capture_flagged": float(fraud_capture_flagged),
        "decline_fpr": float(decline_fpr),
        "review_precision": float(review_precision),
        "decline_precision": float(decline_precision),
    }


def evaluate_threshold_pairs(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold_pairs: list[tuple[float, float]],
) -> pd.DataFrame:
    """Evaluate a list of threshold pairs and return ranked results."""
    rows: list[dict[str, float | str]] = []
    for idx, (approve, decline) in enumerate(threshold_pairs, start=1):
        metrics = evaluate_policy(
            y_true=y_true,
            y_score=y_score,
            approve_threshold=approve,
            decline_threshold=decline,
        )
        policy_id = f"policy_{idx:02d}_t1_{approve:.4f}_t2_{decline:.4f}"
        rows.append({"policy_id": policy_id, **metrics})

    result = pd.DataFrame(rows)
    ranked = result.sort_values(
        by=["fraud_capture_flagged", "decline_precision", "review_rate", "decline_fpr"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked


def select_recommended_policy(
    ranked: pd.DataFrame,
    *,
    max_review_rate: float,
    min_fraud_capture_flagged: float,
    max_decline_fpr: float,
) -> tuple[pd.Series, bool]:
    """Pick recommended row and whether it satisfies guardrails."""
    feasible = ranked[
        (ranked["review_rate"] <= max_review_rate)
        & (ranked["fraud_capture_flagged"] >= min_fraud_capture_flagged)
        & (ranked["decline_fpr"] <= max_decline_fpr)
    ]
    if not feasible.empty:
        return feasible.iloc[0], True

    violation_df = ranked.copy()
    violation_df["review_violation"] = np.maximum(0.0, violation_df["review_rate"] - max_review_rate)
    violation_df["capture_violation"] = np.maximum(0.0, min_fraud_capture_flagged - violation_df["fraud_capture_flagged"])
    violation_df["fpr_violation"] = np.maximum(0.0, violation_df["decline_fpr"] - max_decline_fpr)
    violation_df["violation_score"] = (
        violation_df["review_violation"] * 1.0
        + violation_df["capture_violation"] * 2.0
        + violation_df["fpr_violation"] * 1.5
    )
    best_soft = violation_df.sort_values(
        by=["violation_score", "fraud_capture_flagged", "review_rate", "decline_fpr"],
        ascending=[True, False, True, True],
    ).iloc[0]
    return best_soft, False


def select_fallback_policy(ranked: pd.DataFrame, recommended_policy_id: str) -> pd.Series | None:
    """Select a fallback policy biased toward lower review load."""
    alternatives = ranked[ranked["policy_id"] != recommended_policy_id]
    if alternatives.empty:
        return None
    selected = alternatives.sort_values(
        by=["review_rate", "decline_fpr", "fraud_capture_flagged"],
        ascending=[True, True, False],
    ).iloc[0]
    return selected


def build_experiments_markdown(
    *,
    ranked: pd.DataFrame,
    train_rows: int,
    test_rows: int,
    model_path: Path,
    max_review_rate: float,
    min_fraud_capture_flagged: float,
    max_decline_fpr: float,
    recommended: pd.Series,
    recommended_feasible: bool,
) -> str:
    """Build Week 3 experiments log markdown."""
    lines = [
        "# Week 3 Policy Experiments",
        "",
        "## Run Details",
        f"- Date: {date.today().isoformat()}",
        f"- Train rows: {train_rows}",
        f"- Test rows: {test_rows}",
        f"- Model path: `{model_path}`",
        f"- Guardrail max review rate: {max_review_rate:.2%}",
        f"- Guardrail min fraud capture (review+decline): {min_fraud_capture_flagged:.2%}",
        f"- Guardrail max decline FPR: {max_decline_fpr:.2%}",
        "",
        "## Evaluated Policies",
        "",
        "| Rank | Policy | T1 approve | T2 decline | Approve% | Review% | Decline% | Fraud capture (decline) | Fraud capture (review+decline) | Decline FPR |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in ranked.iterrows():
        lines.append(
            "| {rank} | {policy} | {t1:.4f} | {t2:.4f} | {approve:.2%} | {review:.2%} | {decline:.2%} | {cap_decline:.2%} | {cap_flagged:.2%} | {decline_fpr:.2%} |".format(
                rank=int(row["rank"]),
                policy=row["policy_id"],
                t1=float(row["approve_threshold"]),
                t2=float(row["decline_threshold"]),
                approve=float(row["approve_rate"]),
                review=float(row["review_rate"]),
                decline=float(row["decline_rate"]),
                cap_decline=float(row["fraud_capture_decline"]),
                cap_flagged=float(row["fraud_capture_flagged"]),
                decline_fpr=float(row["decline_fpr"]),
            )
        )

    lines.extend(
        [
            "",
            "## Recommended Policy",
            f"- Policy id: `{recommended['policy_id']}`",
            f"- T1/T2: {float(recommended['approve_threshold']):.4f} / {float(recommended['decline_threshold']):.4f}",
            f"- Meets guardrails: {'yes' if recommended_feasible else 'no'}",
        ]
    )
    return "\n".join(lines)


def build_policy_report_markdown(
    *,
    recommended: pd.Series,
    recommended_feasible: bool,
    fallback: pd.Series | None,
    max_review_rate: float,
    min_fraud_capture_flagged: float,
    max_decline_fpr: float,
) -> str:
    """Build Week 3 policy simulation report markdown."""
    lines = [
        "# Policy Simulation Report (Week 3)",
        "",
        "## Recommended Threshold Policy",
        f"- Date: {date.today().isoformat()}",
        f"- Policy id: `{recommended['policy_id']}`",
        f"- `approve`: score < {float(recommended['approve_threshold']):.4f}",
        f"- `review`: {float(recommended['approve_threshold']):.4f} <= score < {float(recommended['decline_threshold']):.4f}",
        f"- `decline`: score >= {float(recommended['decline_threshold']):.4f}",
        "",
        "## Expected Operational Impact",
        f"- Approve rate: {float(recommended['approve_rate']):.2%}",
        f"- Review rate: {float(recommended['review_rate']):.2%}",
        f"- Decline rate: {float(recommended['decline_rate']):.2%}",
        f"- Fraud capture (decline only): {float(recommended['fraud_capture_decline']):.2%}",
        f"- Fraud capture (review + decline): {float(recommended['fraud_capture_flagged']):.2%}",
        f"- Decline FPR: {float(recommended['decline_fpr']):.2%}",
        "",
        "## Guardrail Check",
        f"- Max review rate target: {max_review_rate:.2%}",
        f"- Min flagged fraud capture target: {min_fraud_capture_flagged:.2%}",
        f"- Max decline FPR target: {max_decline_fpr:.2%}",
        f"- Status: {'PASS' if recommended_feasible else 'SOFT FAIL (best available tradeoff)'}",
    ]

    if fallback is not None:
        lines.extend(
            [
                "",
                "## Fallback Policy",
                f"- Policy id: `{fallback['policy_id']}`",
                f"- T1/T2: {float(fallback['approve_threshold']):.4f} / {float(fallback['decline_threshold']):.4f}",
                "- Use fallback if review queue temporarily exceeds capacity or fraud pressure shifts.",
            ]
        )

    lines.extend(
        [
            "",
            "## Risks and Notes",
            "- Thresholds were simulated on a historical test split; behavior may drift in production.",
            "- Track review volume and decline false positives after deployment.",
            "- Recalibrate thresholds when score distributions shift materially.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=Path("data/creditcard.csv"))
    parser.add_argument("--model-path", type=Path, default=Path("models/week2_best_logreg.joblib"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--threshold-pairs",
        type=str,
        default=None,
        help="Comma-separated approve:decline pairs, e.g. '0.05:0.45,0.08:0.50'.",
    )
    parser.add_argument("--max-review-rate", type=float, default=0.08)
    parser.add_argument("--min-fraud-capture-flagged", type=float, default=0.95)
    parser.add_argument("--max-decline-fpr", type=float, default=0.02)
    parser.add_argument(
        "--results-out-path",
        type=Path,
        default=Path("reports/policy_experiments_week3.csv"),
    )
    parser.add_argument(
        "--experiments-out-path",
        type=Path,
        default=Path("reports/experiments_week3.md"),
    )
    parser.add_argument(
        "--report-out-path",
        type=Path,
        default=Path("reports/policy_simulation_week3.md"),
    )
    args = parser.parse_args()

    if not (0.0 <= args.max_review_rate <= 1.0):
        raise ValueError("--max-review-rate must be in [0, 1].")
    if not (0.0 <= args.min_fraud_capture_flagged <= 1.0):
        raise ValueError("--min-fraud-capture-flagged must be in [0, 1].")
    if not (0.0 <= args.max_decline_fpr <= 1.0):
        raise ValueError("--max-decline-fpr must be in [0, 1].")

    df = load_data(args.data_path)
    train_df, test_df = split_train_test(df, test_size=args.test_size)
    y_true = test_df[LABEL_COLUMN].to_numpy(dtype=np.int8)

    bundle = load_model_bundle(args.model_path)
    model_features = resolve_feature_columns(test_df, bundle.artifact_feature_columns)
    x_test = test_df[model_features].to_numpy(dtype=np.float32)
    y_score = bundle.model.predict_proba(x_test)[:, 1]

    if args.threshold_pairs:
        threshold_pairs = parse_threshold_pairs(args.threshold_pairs)
    else:
        base_decline = bundle.artifact_decline_threshold if bundle.artifact_decline_threshold is not None else 0.50
        threshold_pairs = default_threshold_pairs(base_decline)

    ranked = evaluate_threshold_pairs(y_true=y_true, y_score=y_score, threshold_pairs=threshold_pairs)
    recommended, recommended_feasible = select_recommended_policy(
        ranked,
        max_review_rate=args.max_review_rate,
        min_fraud_capture_flagged=args.min_fraud_capture_flagged,
        max_decline_fpr=args.max_decline_fpr,
    )

    fallback = select_fallback_policy(ranked, str(recommended["policy_id"]))

    args.results_out_path.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(args.results_out_path, index=False)

    experiments_md = build_experiments_markdown(
        ranked=ranked,
        train_rows=len(train_df),
        test_rows=len(test_df),
        model_path=args.model_path,
        max_review_rate=args.max_review_rate,
        min_fraud_capture_flagged=args.min_fraud_capture_flagged,
        max_decline_fpr=args.max_decline_fpr,
        recommended=recommended,
        recommended_feasible=recommended_feasible,
    )
    args.experiments_out_path.parent.mkdir(parents=True, exist_ok=True)
    args.experiments_out_path.write_text(experiments_md, encoding="utf-8")

    report_md = build_policy_report_markdown(
        recommended=recommended,
        recommended_feasible=recommended_feasible,
        fallback=fallback,
        max_review_rate=args.max_review_rate,
        min_fraud_capture_flagged=args.min_fraud_capture_flagged,
        max_decline_fpr=args.max_decline_fpr,
    )
    args.report_out_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_out_path.write_text(report_md, encoding="utf-8")

    print("=== Week 3 Policy Simulation Summary ===")
    print(f"Evaluated policies: {len(ranked)}")
    print(f"Recommended policy: {recommended['policy_id']}")
    print(
        "T1/T2: "
        f"{float(recommended['approve_threshold']):.4f}/{float(recommended['decline_threshold']):.4f}"
    )
    print(f"Review rate: {float(recommended['review_rate']):.2%}")
    print(f"Fraud capture (review+decline): {float(recommended['fraud_capture_flagged']):.2%}")
    print(f"Decline FPR: {float(recommended['decline_fpr']):.2%}")
    print(f"Meets guardrails: {'yes' if recommended_feasible else 'no'}")
    print(f"Wrote policy table: {args.results_out_path}")
    print(f"Wrote experiments log: {args.experiments_out_path}")
    print(f"Wrote policy report: {args.report_out_path}")


if __name__ == "__main__":
    main()
