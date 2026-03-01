"""Policy drift check: compare policy_config.json against a freshly recomputed policy."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from datetime import date

from policy_config_validator import (
    DEFAULT_REQUIRED_PROFILES,
    load_policy_config,
    validate_policy_config,
)


def compare_policy_configs(
    current: dict[str, Any],
    fresh: dict[str, Any],
    *,
    required_profiles: tuple[str, ...],
    max_delta: float,
) -> list[str]:
    diffs: list[str] = []

    current_guardrails = current.get("guardrails", {})
    fresh_guardrails = fresh.get("guardrails", {})
    for key in ("max_review_rate", "min_flagged_fraud_capture", "max_decline_fpr"):
        cur_val = float(current_guardrails.get(key, 0.0))
        fresh_val = float(fresh_guardrails.get(key, 0.0))
        if abs(cur_val - fresh_val) > max_delta:
            diffs.append(f"guardrail {key} differs: current={cur_val} fresh={fresh_val}")

    current_profiles = current.get("profiles", {})
    fresh_profiles = fresh.get("profiles", {})

    for profile in required_profiles:
        cur_profile = current_profiles.get(profile, {})
        fresh_profile = fresh_profiles.get(profile, {})
        for key in ("approve_threshold", "decline_threshold"):
            cur_val = float(cur_profile.get(key, 0.0))
            fresh_val = float(fresh_profile.get(key, 0.0))
            if abs(cur_val - fresh_val) > max_delta:
                diffs.append(
                    f"profile {profile} {key} differs: current={cur_val} fresh={fresh_val}"
                )

    return diffs


def build_drift_report(
    *,
    policy_config_path: Path,
    data_path: Path | None,
    diffs: list[str],
) -> str:
    lines = [
        "# Policy Drift Check Report",
        "",
        f"- Date: {date.today().isoformat()}",
        f"- Policy config: `{policy_config_path}`",
        f"- Data path: `{data_path}`" if data_path is not None else "- Data path: `None`",
        f"- Status: {'FAIL' if diffs else 'PASS'}",
        "",
    ]
    if diffs:
        lines.append("## Drift Details")
        for diff in diffs:
            lines.append(f"- {diff}")
    else:
        lines.append("## Drift Details")
        lines.append("- No drift detected.")
    return "\n".join(lines)


def write_report(report_path: Path, report: str) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report + "\n", encoding="utf-8")


def recompute_policy_config(
    *,
    data_path: Path,
    model_path: Path,
    policy_out_path: Path,
) -> None:
    cmd = [
        sys.executable,
        "src/cost_policy_optimization.py",
        "--data-path",
        str(data_path),
        "--model-path",
        str(model_path),
        "--policy-out-path",
        str(policy_out_path),
        "--results-out-path",
        str(policy_out_path.with_suffix(".results.csv")),
        "--report-out-path",
        str(policy_out_path.with_suffix(".report.md")),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy-config-path",
        type=Path,
        default=Path("models/policy_config.json"),
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="If provided, recompute policy config and compare for drift.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/week2_best_logreg.joblib"),
    )
    parser.add_argument(
        "--max-delta",
        type=float,
        default=1e-6,
        help="Maximum allowed numeric delta before failing drift check.",
    )
    parser.add_argument(
        "--report-out-path",
        type=Path,
        default=Path("reports/policy_drift_report.md"),
        help="Optional path to write drift report.",
    )
    args = parser.parse_args()

    payload = load_policy_config(args.policy_config_path)
    validate_policy_config(
        payload,
        require_guardrails=True,
        required_profiles=DEFAULT_REQUIRED_PROFILES,
    )

    diffs: list[str] = []
    if args.data_path is None:
        print("Policy drift check: SKIPPED (no data path provided)")
        report = build_drift_report(
            policy_config_path=args.policy_config_path,
            data_path=None,
            diffs=diffs,
        )
        write_report(args.report_out_path, report)
        return

    if not args.data_path.exists():
        raise FileNotFoundError(f"Data path not found: {args.data_path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "policy_config.json"
        recompute_policy_config(
            data_path=args.data_path,
            model_path=args.model_path,
            policy_out_path=tmp_path,
        )
        fresh = load_policy_config(tmp_path)

    diffs = compare_policy_configs(
        payload,
        fresh,
        required_profiles=DEFAULT_REQUIRED_PROFILES,
        max_delta=args.max_delta,
    )
    report = build_drift_report(
        policy_config_path=args.policy_config_path,
        data_path=args.data_path,
        diffs=diffs,
    )
    write_report(args.report_out_path, report)
    if diffs:
        print("Policy drift check: FAILED")
        for diff in diffs:
            print(f"- {diff}")
        raise SystemExit(1)

    print("Policy drift check: OK")


if __name__ == "__main__":
    main()
