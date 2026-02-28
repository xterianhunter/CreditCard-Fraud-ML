"""Generate a monitoring snapshot markdown from scored inference output.

Usage:
    python src/monitoring_snapshot.py --scored-path reports/inference_output.csv
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


def compute_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Compute population stability index using quantile bins from reference."""
    eps = 1e-8
    qs = np.linspace(0.0, 1.0, bins + 1)
    edges = np.quantile(reference, qs)
    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_hist, _ = np.histogram(reference, bins=edges)
    cur_hist, _ = np.histogram(current, bins=edges)
    ref_pct = ref_hist / max(np.sum(ref_hist), 1)
    cur_pct = cur_hist / max(np.sum(cur_hist), 1)

    ref_pct = np.clip(ref_pct, eps, 1.0)
    cur_pct = np.clip(cur_pct, eps, 1.0)
    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def build_markdown(
    *,
    scored_path: Path,
    total_rows: int,
    approve_rate: float,
    review_rate: float,
    decline_rate: float,
    mean_score: float,
    p95_score: float,
    p99_score: float,
    label_metrics: list[str],
    drift_metrics: list[str],
) -> str:
    lines = [
        "# Monitoring Snapshot",
        "",
        "## Snapshot Details",
        f"- Date: {date.today().isoformat()}",
        f"- Source file: `{scored_path}`",
        f"- Rows: {total_rows:,}",
        "",
        "## Policy Volume Mix",
        f"- Approve rate: {approve_rate:.2%}",
        f"- Review rate: {review_rate:.2%}",
        f"- Decline rate: {decline_rate:.2%}",
        "",
        "## Score Distribution",
        f"- Mean score: {mean_score:.6f}",
        f"- P95 score: {p95_score:.6f}",
        f"- P99 score: {p99_score:.6f}",
    ]

    if label_metrics:
        lines.extend(["", "## Label-Aware Quality Metrics", *label_metrics])

    if drift_metrics:
        lines.extend(["", "## Drift Signals", *drift_metrics])

    lines.extend(
        [
            "",
            "## Notes",
            "- Treat this as an operational snapshot, not a formal model-evaluation report.",
            "- Recompute frequently and trend metrics over time.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored-path", type=Path, default=Path("reports/inference_output.csv"))
    parser.add_argument(
        "--reference-scored-path",
        type=Path,
        default=None,
        help="Optional previous scored file to compute score PSI.",
    )
    parser.add_argument("--score-column", type=str, default="score")
    parser.add_argument("--decision-column", type=str, default="decision")
    parser.add_argument("--label-column", type=str, default="Class")
    parser.add_argument("--output-path", type=Path, default=Path("reports/monitoring_snapshot.md"))
    args = parser.parse_args()

    if not args.scored_path.exists():
        raise FileNotFoundError(f"Scored file not found: {args.scored_path}")

    df = pd.read_csv(args.scored_path)
    if args.score_column not in df.columns:
        raise ValueError(f"Missing score column: {args.score_column}")
    if args.decision_column not in df.columns:
        raise ValueError(f"Missing decision column: {args.decision_column}")

    total_rows = len(df)
    if total_rows == 0:
        raise ValueError("Scored file is empty.")

    decision_counts = df[args.decision_column].value_counts().to_dict()
    approve_rate = float(decision_counts.get("approve", 0) / total_rows)
    review_rate = float(decision_counts.get("review", 0) / total_rows)
    decline_rate = float(decision_counts.get("decline", 0) / total_rows)

    score_vals = df[args.score_column].to_numpy(dtype=np.float64)
    mean_score = float(np.mean(score_vals))
    p95_score = float(np.percentile(score_vals, 95))
    p99_score = float(np.percentile(score_vals, 99))

    label_metrics: list[str] = []
    if args.label_column in df.columns:
        y_true = df[args.label_column].to_numpy(dtype=np.int8)
        fraud_mask = y_true == 1
        nonfraud_mask = y_true == 0
        fraud_total = int(np.sum(fraud_mask))
        nonfraud_total = int(np.sum(nonfraud_mask))

        review_or_decline = df[args.decision_column].isin(["review", "decline"]).to_numpy()
        decline_mask = (df[args.decision_column] == "decline").to_numpy()
        review_mask = (df[args.decision_column] == "review").to_numpy()

        fraud_flagged = int(np.sum(fraud_mask & review_or_decline))
        fraud_declined = int(np.sum(fraud_mask & decline_mask))
        nonfraud_declined = int(np.sum(nonfraud_mask & decline_mask))

        flagged_capture = fraud_flagged / fraud_total if fraud_total else 0.0
        decline_capture = fraud_declined / fraud_total if fraud_total else 0.0
        decline_fpr = nonfraud_declined / nonfraud_total if nonfraud_total else 0.0
        review_count = int(np.sum(review_mask))
        decline_count = int(np.sum(decline_mask))
        review_precision = fraud_flagged - fraud_declined
        review_precision = review_precision / review_count if review_count else 0.0
        decline_precision = fraud_declined / decline_count if decline_count else 0.0

        label_metrics.extend(
            [
                f"- Fraud rows in snapshot: {fraud_total}",
                f"- Flagged fraud capture (review + decline): {flagged_capture:.2%}",
                f"- Decline-only fraud capture: {decline_capture:.2%}",
                f"- Decline FPR (non-fraud auto-declined): {decline_fpr:.2%}",
                f"- Review precision: {review_precision:.2%}",
                f"- Decline precision: {decline_precision:.2%}",
            ]
        )

    drift_metrics: list[str] = []
    if args.reference_scored_path is not None and args.reference_scored_path.exists():
        ref_df = pd.read_csv(args.reference_scored_path)
        if args.score_column in ref_df.columns:
            ref_score = ref_df[args.score_column].to_numpy(dtype=np.float64)
            psi = compute_psi(ref_score, score_vals, bins=10)
            drift_metrics.append(f"- Score PSI vs reference: {psi:.6f}")
            if psi < 0.1:
                drift_metrics.append("- Drift interpretation: low")
            elif psi < 0.25:
                drift_metrics.append("- Drift interpretation: moderate")
            else:
                drift_metrics.append("- Drift interpretation: high")

    report = build_markdown(
        scored_path=args.scored_path,
        total_rows=total_rows,
        approve_rate=approve_rate,
        review_rate=review_rate,
        decline_rate=decline_rate,
        mean_score=mean_score,
        p95_score=p95_score,
        p99_score=p99_score,
        label_metrics=label_metrics,
        drift_metrics=drift_metrics,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(report, encoding="utf-8")

    print("=== Monitoring Snapshot Summary ===")
    print(f"Rows: {total_rows}")
    print(f"Approve/review/decline rates: {approve_rate:.2%}/{review_rate:.2%}/{decline_rate:.2%}")
    if label_metrics:
        print("Included label-aware quality metrics.")
    if drift_metrics:
        print("Included drift metrics against reference file.")
    print(f"Wrote report: {args.output_path}")


if __name__ == "__main__":
    main()
