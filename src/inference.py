"""Minimal inference CLI with schema validation and threshold policy decisions.

Usage:
    python src/inference.py --input-path data/creditcard.csv --output-path reports/inference_output.csv
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd

from data_contract import feature_columns, validate_dataframe


PolicyProfile = Literal["primary", "fallback", "phase2_guarded", "artifact"]

DEFAULT_POLICY_CONFIG_PATH = Path("models/policy_config.json")
DEFAULT_ARTIFACT_DECLINE_FALLBACK = 0.45
SUPPORTED_POLICY_SCHEMA_VERSION = 1
DEFAULT_POLICY_PROFILE: PolicyProfile = "phase2_guarded"


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


def load_policy_config_payload(policy_config_path: Path) -> dict[str, Any]:
    """Load and validate policy config payload."""
    if not policy_config_path.exists():
        raise FileNotFoundError(
            f"Policy config not found: {policy_config_path}. "
            "Run src/cost_policy_optimization.py to generate it."
        )

    payload = json.loads(policy_config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Invalid policy config: root must be a JSON object.")

    schema_version = payload.get("schema_version")
    if not isinstance(schema_version, int):
        raise ValueError("Invalid policy config: `schema_version` must be an integer.")
    if schema_version != SUPPORTED_POLICY_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported policy config schema_version={schema_version}; "
            f"supported={SUPPORTED_POLICY_SCHEMA_VERSION}."
        )
    return payload


def load_policy_profiles(policy_config_path: Path) -> dict[str, tuple[float, float]]:
    """Load policy profile thresholds from json config."""
    payload = load_policy_config_payload(policy_config_path)
    profiles_obj = payload.get("profiles")
    if not isinstance(profiles_obj, dict):
        raise ValueError("Invalid policy config: missing object field `profiles`.")

    profiles: dict[str, tuple[float, float]] = {}
    for profile_name, thresholds in profiles_obj.items():
        if not isinstance(thresholds, dict):
            raise ValueError(f"Invalid policy config: profile `{profile_name}` must be an object.")
        if "approve_threshold" not in thresholds or "decline_threshold" not in thresholds:
            raise ValueError(
                f"Invalid policy config: profile `{profile_name}` missing thresholds."
            )
        approve = float(thresholds["approve_threshold"])
        decline = float(thresholds["decline_threshold"])
        validate_thresholds(approve, decline)
        profiles[str(profile_name)] = (approve, decline)
    return profiles


def policy_status_snapshot(
    *,
    policy_config_path: Path,
    policy_profile: PolicyProfile,
) -> dict[str, Any]:
    """Build policy status snapshot for operational diagnostics."""
    payload = load_policy_config_payload(policy_config_path)
    profiles = load_policy_profiles(policy_config_path)

    snapshot: dict[str, Any] = {
        "policy_config_path": str(policy_config_path),
        "schema_version": payload["schema_version"],
        "generated_at": payload.get("generated_at"),
        "default_policy_profile": DEFAULT_POLICY_PROFILE,
        "selected_policy_profile": policy_profile,
        "available_profiles": sorted(profiles.keys()),
    }

    guardrails = payload.get("guardrails")
    if isinstance(guardrails, dict):
        snapshot["guardrails"] = guardrails

    if policy_profile == "artifact":
        snapshot["selected_profile_thresholds"] = "artifact_model_metadata"
        snapshot["note"] = (
            "artifact profile resolves thresholds from model artifact "
            "(`threshold_for_target_fpr`) at inference time."
        )
        return snapshot

    if policy_profile not in profiles:
        raise ValueError(
            f"Selected policy profile `{policy_profile}` not found in policy config."
        )

    approve, decline = profiles[policy_profile]
    snapshot["selected_profile_thresholds"] = {
        "approve_threshold": float(approve),
        "decline_threshold": float(decline),
    }

    profiles_obj = payload.get("profiles")
    profile_raw = profiles_obj.get(policy_profile) if isinstance(profiles_obj, dict) else None
    if isinstance(profile_raw, dict):
        metadata_keys = [
            "source",
            "calibration_method",
            "meets_guardrails",
            "review_rate",
            "decline_fpr",
            "flagged_fraud_capture",
            "total_cost",
            "cost_per_row",
        ]
        metadata = {k: profile_raw[k] for k in metadata_keys if k in profile_raw}
        if metadata:
            snapshot["selected_profile_metadata"] = metadata
    return snapshot


def build_policy_status_text(snapshot: dict[str, Any]) -> str:
    """Render policy snapshot as readable text."""
    lines = [
        "=== Policy Status ===",
        f"Policy config: {snapshot['policy_config_path']}",
        f"Schema version: {snapshot['schema_version']}",
        f"Generated at: {snapshot.get('generated_at')}",
        f"Default profile: {snapshot['default_policy_profile']}",
        f"Selected profile: {snapshot['selected_policy_profile']}",
        f"Available profiles: {snapshot['available_profiles']}",
    ]

    guardrails = snapshot.get("guardrails")
    if isinstance(guardrails, dict):
        lines.append(f"Guardrails: {guardrails}")

    selected_thresholds = snapshot.get("selected_profile_thresholds")
    lines.append(f"Selected thresholds: {selected_thresholds}")

    selected_metadata = snapshot.get("selected_profile_metadata")
    if isinstance(selected_metadata, dict):
        lines.append(f"Selected metadata: {selected_metadata}")

    note = snapshot.get("note")
    if note is not None:
        lines.append(f"Note: {note}")

    return "\n".join(lines)


def resolve_profile_thresholds(
    *,
    policy_profile: PolicyProfile,
    artifact_decline_threshold: float | None,
    policy_profiles: dict[str, tuple[float, float]] | None,
) -> tuple[float, float]:
    """Return default threshold pair for a policy profile."""
    if policy_profile == "artifact":
        decline = (
            float(artifact_decline_threshold)
            if artifact_decline_threshold is not None
            else DEFAULT_ARTIFACT_DECLINE_FALLBACK
        )
        approve = min(0.11, decline * 0.5)
        return float(approve), float(decline)
    if policy_profiles is None:
        raise ValueError("Policy profiles are required for non-artifact policy resolution.")
    if policy_profile not in policy_profiles:
        available = sorted(policy_profiles.keys())
        raise ValueError(
            f"Unsupported policy profile: {policy_profile}. Available profiles: {available}"
        )
    return policy_profiles[policy_profile]


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
    policy_profile: PolicyProfile = DEFAULT_POLICY_PROFILE,
    policy_config_path: Path = DEFAULT_POLICY_CONFIG_PATH,
) -> tuple[pd.DataFrame, float, float]:
    """Load data + model and return scored output with resolved thresholds."""
    df = pd.read_csv(input_path)
    validate_dataframe(df, require_label=False)

    bundle = load_model_bundle(model_path)
    policy_profiles = (
        None
        if policy_profile == "artifact"
        else load_policy_profiles(policy_config_path)
    )
    profile_approve_threshold, profile_decline_threshold = resolve_profile_thresholds(
        policy_profile=policy_profile,
        artifact_decline_threshold=bundle.artifact_decline_threshold,
        policy_profiles=policy_profiles,
    )
    final_decline_threshold = (
        decline_threshold
        if decline_threshold is not None
        else profile_decline_threshold
    )
    final_approve_threshold = (
        approve_threshold
        if approve_threshold is not None
        else profile_approve_threshold
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
    parser.add_argument("--input-path", type=Path, required=False, help="CSV input to score.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/week2_best_logreg.joblib"),
        help="Path to model artifact.",
    )
    parser.add_argument("--output-path", type=Path, default=None, help="Where to write scored CSV.")
    parser.add_argument(
        "--policy-profile",
        choices=["primary", "fallback", "phase2_guarded", "artifact"],
        default=DEFAULT_POLICY_PROFILE,
        help="Default threshold profile when explicit thresholds are not provided.",
    )
    parser.add_argument("--approve-threshold", type=float, default=None, help="Approve upper bound.")
    parser.add_argument("--decline-threshold", type=float, default=None, help="Decline lower bound.")
    parser.add_argument(
        "--policy-config-path",
        type=Path,
        default=DEFAULT_POLICY_CONFIG_PATH,
        help="Path to policy configuration json.",
    )
    parser.add_argument(
        "--print-policy-status",
        action="store_true",
        help="Print loaded policy config details and selected profile thresholds.",
    )
    args = parser.parse_args()

    if args.print_policy_status:
        snapshot = policy_status_snapshot(
            policy_config_path=args.policy_config_path,
            policy_profile=args.policy_profile,
        )
        print(build_policy_status_text(snapshot))
        if args.input_path is None:
            return

    if args.input_path is None:
        parser.error("--input-path is required unless --print-policy-status is used.")

    scored_df, approve_threshold, decline_threshold = run_inference(
        input_path=args.input_path,
        model_path=args.model_path,
        output_path=args.output_path,
        policy_profile=args.policy_profile,
        approve_threshold=args.approve_threshold,
        decline_threshold=args.decline_threshold,
        policy_config_path=args.policy_config_path,
    )

    print("=== Inference Summary ===")
    print(f"Rows scored: {len(scored_df)}")
    print(f"Policy profile: {args.policy_profile}")
    print(f"Approve threshold: {approve_threshold:.6f}")
    print(f"Decline threshold: {decline_threshold:.6f}")
    counts = scored_df["decision"].value_counts().to_dict()
    print(f"Decision counts: {counts}")
    if args.output_path is not None:
        print(f"Wrote scored output to: {args.output_path}")


if __name__ == "__main__":
    main()
