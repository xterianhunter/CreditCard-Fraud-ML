from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from policy_config_validator import (
    DEFAULT_REQUIRED_PROFILES,
    load_policy_config,
    validate_policy_config,
)


def write_policy_config(path: Path, *, schema_version: int = 1) -> None:
    payload = {
        "schema_version": schema_version,
        "guardrails": {
            "max_review_rate": 0.1,
            "min_flagged_fraud_capture": 0.85,
            "max_decline_fpr": 0.02,
        },
        "profiles": {
            "primary": {"approve_threshold": 0.11, "decline_threshold": 0.45},
            "fallback": {"approve_threshold": 0.19, "decline_threshold": 0.80},
            "phase2_guarded": {"approve_threshold": 0.20, "decline_threshold": 0.80},
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


class PolicyConfigValidatorTests(unittest.TestCase):
    def test_validate_policy_config_happy_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "policy_config.json"
            write_policy_config(path)
            payload = load_policy_config(path)
            validate_policy_config(
                payload,
                require_guardrails=True,
                required_profiles=DEFAULT_REQUIRED_PROFILES,
            )

    def test_validate_policy_config_raises_on_schema_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "policy_config.json"
            write_policy_config(path, schema_version=2)
            payload = load_policy_config(path)
            with self.assertRaises(ValueError):
                validate_policy_config(
                    payload,
                    require_guardrails=True,
                    required_profiles=DEFAULT_REQUIRED_PROFILES,
                )

    def test_validate_policy_config_raises_on_missing_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "policy_config.json"
            write_policy_config(path)
            payload = load_policy_config(path)
            payload["profiles"].pop("fallback")
            with self.assertRaises(ValueError):
                validate_policy_config(
                    payload,
                    require_guardrails=True,
                    required_profiles=DEFAULT_REQUIRED_PROFILES,
                )

    def test_validate_policy_config_raises_on_bad_guardrail(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "policy_config.json"
            write_policy_config(path)
            payload = load_policy_config(path)
            payload["guardrails"]["max_review_rate"] = 1.5
            with self.assertRaises(ValueError):
                validate_policy_config(
                    payload,
                    require_guardrails=True,
                    required_profiles=DEFAULT_REQUIRED_PROFILES,
                )


if __name__ == "__main__":
    unittest.main()
