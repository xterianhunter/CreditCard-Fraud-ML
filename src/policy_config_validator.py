"""Policy config validator for CI and ops checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SUPPORTED_SCHEMA_VERSION = 1
DEFAULT_REQUIRED_PROFILES = ("primary", "fallback", "phase2_guarded")


def validate_thresholds(approve: float, decline: float) -> None:
    if not (0.0 <= approve < decline <= 1.0):
        raise ValueError(
            "Thresholds must satisfy 0 <= approve_threshold < decline_threshold <= 1."
        )


def load_policy_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Policy config not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Policy config root must be an object.")
    return payload


def validate_policy_config(
    payload: dict[str, Any],
    *,
    require_guardrails: bool,
    required_profiles: tuple[str, ...],
) -> None:
    schema_version = payload.get("schema_version")
    if not isinstance(schema_version, int):
        raise ValueError("`schema_version` must be an integer.")
    if schema_version != SUPPORTED_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema_version={schema_version}; supported={SUPPORTED_SCHEMA_VERSION}."
        )

    profiles = payload.get("profiles")
    if not isinstance(profiles, dict):
        raise ValueError("`profiles` must be an object.")

    for profile in required_profiles:
        if profile not in profiles:
            raise ValueError(f"Missing required profile: {profile}")

    for name, values in profiles.items():
        if not isinstance(values, dict):
            raise ValueError(f"Profile `{name}` must be an object.")
        if "approve_threshold" not in values or "decline_threshold" not in values:
            raise ValueError(f"Profile `{name}` missing thresholds.")
        approve = float(values["approve_threshold"])
        decline = float(values["decline_threshold"])
        validate_thresholds(approve, decline)
        meets_guardrails = values.get("meets_guardrails")
        if meets_guardrails is not None and not isinstance(meets_guardrails, bool):
            raise ValueError(f"Profile `{name}` has non-boolean meets_guardrails.")

    if require_guardrails:
        guardrails = payload.get("guardrails")
        if not isinstance(guardrails, dict):
            raise ValueError("`guardrails` must be an object when required.")
        for key in ("max_review_rate", "min_flagged_fraud_capture", "max_decline_fpr"):
            if key not in guardrails:
                raise ValueError(f"Missing guardrail: {key}")
            value = guardrails[key]
            if not isinstance(value, (int, float)):
                raise ValueError(f"Guardrail `{key}` must be numeric.")
            if not (0.0 <= float(value) <= 1.0):
                raise ValueError(f"Guardrail `{key}` must be in [0, 1].")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy-config-path",
        type=Path,
        default=Path("models/policy_config.json"),
    )
    parser.add_argument(
        "--require-guardrails",
        action="store_true",
        help="Fail if guardrails are missing or invalid.",
    )
    parser.add_argument(
        "--required-profiles",
        type=str,
        default=",".join(DEFAULT_REQUIRED_PROFILES),
        help="Comma-separated list of required profiles.",
    )
    args = parser.parse_args()

    required_profiles = tuple(p.strip() for p in args.required_profiles.split(",") if p.strip())
    if not required_profiles:
        raise ValueError("At least one required profile must be provided.")

    payload = load_policy_config(args.policy_config_path)
    validate_policy_config(
        payload,
        require_guardrails=args.require_guardrails,
        required_profiles=required_profiles,
    )
    print("Policy config validation: OK")


if __name__ == "__main__":
    main()
