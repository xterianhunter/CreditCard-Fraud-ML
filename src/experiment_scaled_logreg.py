"""Week 2 experiment runner: compare raw vs scaled logistic regression models.

Usage:
    python src/experiment_scaled_logreg.py --data-path data/creditcard.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_contract import LABEL_COLUMN, feature_columns
from mlflow_utils import log_artifact, log_metrics, log_params, parse_tags, start_mlflow_run
from train_baseline import load_data, recall_at_fpr, save_model_bundle, split_train_test


@dataclass
class Candidate:
    name: str
    model: object
    is_scaled: bool
    c_value: float
    max_iter: int


def parse_c_values(raw: str) -> list[float]:
    values: list[float] = []
    for part in raw.split(","):
        p = part.strip()
        if not p:
            continue
        val = float(p)
        if val <= 0:
            raise ValueError("All C values must be > 0.")
        values.append(val)
    if not values:
        raise ValueError("At least one C value must be provided.")
    return values


def build_candidates(c_values: list[float], max_iter: int) -> list[Candidate]:
    candidates: list[Candidate] = [
        Candidate(
            name="raw_logreg_c1.0",
            model=LogisticRegression(
                class_weight="balanced",
                max_iter=max_iter,
                C=1.0,
                random_state=42,
            ),
            is_scaled=False,
            c_value=1.0,
            max_iter=max_iter,
        )
    ]
    for c in c_values:
        candidates.append(
            Candidate(
                name=f"scaled_logreg_c{c}",
                model=Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "logreg",
                            LogisticRegression(
                                class_weight="balanced",
                                max_iter=max_iter,
                                C=c,
                                random_state=42,
                            ),
                        ),
                    ]
                ),
                is_scaled=True,
                c_value=c,
                max_iter=max_iter,
            )
        )
    return candidates


def rank_results(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(
        by=["recall_at_target_fpr", "pr_auc", "roc_auc"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def build_markdown_report(
    *,
    ranked: pd.DataFrame,
    train_rows: int,
    test_rows: int,
    target_fpr: float,
    best_model_out_path: Path,
) -> str:
    best = ranked.iloc[0]
    lines = [
        "# Week 2 Experiments: Scaled Logistic Regression",
        "",
        "## Run Details",
        f"- Date: {date.today().isoformat()}",
        f"- Train rows: {train_rows}",
        f"- Test rows: {test_rows}",
        f"- Target FPR constraint: {target_fpr:.2%}",
        "",
        "## Ranked Results",
        "",
        "| Rank | Model | Scaled | C | PR-AUC | ROC-AUC | Recall@FPR<=target | Threshold |",
        "|---:|---|:---:|---:|---:|---:|---:|---:|",
    ]

    for idx, row in ranked.iterrows():
        lines.append(
            "| {rank} | {name} | {scaled} | {cval:.3f} | {pr:.6f} | {roc:.6f} | {recall:.6f} | {thr:.6f} |".format(
                rank=idx + 1,
                name=row["name"],
                scaled="yes" if bool(row["is_scaled"]) else "no",
                cval=float(row["c_value"]),
                pr=float(row["pr_auc"]),
                roc=float(row["roc_auc"]),
                recall=float(row["recall_at_target_fpr"]),
                thr=float(row["threshold_at_target_fpr"]),
            )
        )

    lines.extend(
        [
            "",
            "## Recommended Config",
            f"- Model: `{best['name']}`",
            f"- Scaled: {'yes' if bool(best['is_scaled']) else 'no'}",
            f"- C: {float(best['c_value']):.3f}",
            f"- PR-AUC: {float(best['pr_auc']):.6f}",
            f"- ROC-AUC: {float(best['roc_auc']):.6f}",
            f"- Recall@FPR<=target: {float(best['recall_at_target_fpr']):.6f}",
            f"- Threshold: {float(best['threshold_at_target_fpr']):.6f}",
            "",
            "## Artifact",
            f"- Best-model output path: `{best_model_out_path}`",
            "",
            "## Notes",
            "- Ranking prioritizes recall at constrained FPR first, then PR-AUC, then ROC-AUC.",
            "- Keep threshold from selected model for policy simulation inputs.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=Path("data/creditcard.csv"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--target-fpr", type=float, default=0.02)
    parser.add_argument("--c-values", type=str, default="0.1,0.3,1.0,3.0")
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument(
        "--results-out-path",
        type=Path,
        default=Path("reports/experiments_week2.csv"),
    )
    parser.add_argument(
        "--report-out-path",
        type=Path,
        default=Path("reports/experiments_week2.md"),
    )
    parser.add_argument(
        "--best-model-out-path",
        type=Path,
        default=Path("models/week2_best_logreg.joblib"),
    )
    parser.add_argument(
        "--skip-save-best-model",
        action="store_true",
        help="If set, do not persist best model artifact.",
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
        run_name=args.mlflow_run_name or "experiment_scaled_logreg",
        tags=tags,
    ):
        log_params(
            args.mlflow,
            {
                "data_path": str(args.data_path),
                "test_size": float(args.test_size),
                "target_fpr": float(args.target_fpr),
                "c_values": args.c_values,
                "max_iter": int(args.max_iter),
                "skip_save_best_model": bool(args.skip_save_best_model),
            },
        )

        c_values = parse_c_values(args.c_values)
        candidates = build_candidates(c_values=c_values, max_iter=args.max_iter)

        df = load_data(args.data_path)
        train_df, test_df = split_train_test(df, test_size=args.test_size)
        model_features = feature_columns(train_df.columns)
        x_train = train_df[model_features].to_numpy()
        y_train = train_df[LABEL_COLUMN].to_numpy()
        x_test = test_df[model_features].to_numpy()
        y_test = test_df[LABEL_COLUMN].to_numpy()

        rows: list[dict[str, object]] = []
        best_index = 0
        best_key: tuple[float, float, float] = (-1.0, -1.0, -1.0)

        for idx, cand in enumerate(candidates):
            cand.model.fit(x_train, y_train)
            y_score = cand.model.predict_proba(x_test)[:, 1]
            pr_auc = average_precision_score(y_test, y_score)
            roc_auc = roc_auc_score(y_test, y_score)
            recall, threshold = recall_at_fpr(y_test, y_score, target_fpr=args.target_fpr)

            key = (float(recall), float(pr_auc), float(roc_auc))
            if key > best_key:
                best_key = key
                best_index = idx

            rows.append(
                {
                    "name": cand.name,
                    "is_scaled": cand.is_scaled,
                    "c_value": cand.c_value,
                    "max_iter": cand.max_iter,
                    "pr_auc": float(pr_auc),
                    "roc_auc": float(roc_auc),
                    "recall_at_target_fpr": float(recall),
                    "threshold_at_target_fpr": float(threshold),
                }
            )

        result_df = rank_results(pd.DataFrame(rows))
        args.results_out_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(args.results_out_path, index=False)

        report_text = build_markdown_report(
            ranked=result_df,
            train_rows=len(train_df),
            test_rows=len(test_df),
            target_fpr=args.target_fpr,
            best_model_out_path=args.best_model_out_path,
        )
        args.report_out_path.parent.mkdir(parents=True, exist_ok=True)
        args.report_out_path.write_text(report_text, encoding="utf-8")

        best_candidate = candidates[best_index]
        best_row = result_df.iloc[0]
        if not args.skip_save_best_model:
            save_model_bundle(
                model=best_candidate.model,
                model_feature_columns=model_features,
                target_fpr=args.target_fpr,
                threshold_for_target_fpr=float(best_row["threshold_at_target_fpr"]),
                model_out_path=args.best_model_out_path,
            )

        log_metrics(
            args.mlflow,
            {
                "best_pr_auc": float(best_row["pr_auc"]),
                "best_roc_auc": float(best_row["roc_auc"]),
                "best_recall_at_target_fpr": float(best_row["recall_at_target_fpr"]),
                "best_threshold_at_target_fpr": float(best_row["threshold_at_target_fpr"]),
            },
        )
        log_artifact(args.mlflow, str(args.results_out_path))
        log_artifact(args.mlflow, str(args.report_out_path))
        if not args.skip_save_best_model:
            log_artifact(args.mlflow, str(args.best_model_out_path))

        print("=== Week 2 Experiment Summary ===")
        print(f"Candidates evaluated: {len(candidates)}")
        print(f"Best model: {best_row['name']}")
        print(f"Best recall @ FPR<={args.target_fpr:.2%}: {float(best_row['recall_at_target_fpr']):.6f}")
        print(f"Best PR-AUC: {float(best_row['pr_auc']):.6f}")
        print(f"Best ROC-AUC: {float(best_row['roc_auc']):.6f}")
        print(f"Best threshold: {float(best_row['threshold_at_target_fpr']):.6f}")
        print(f"Wrote results: {args.results_out_path}")
        print(f"Wrote report: {args.report_out_path}")
        if not args.skip_save_best_model:
            print(f"Saved best model artifact: {args.best_model_out_path}")


if __name__ == "__main__":
    main()
