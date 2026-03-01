from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from policy_drift_check import build_drift_report, compare_policy_configs


def make_policy_payload(
    approve: float,
    decline: float,
    *,
    guardrails: dict[str, float] | None = None,
) -> dict[str, object]:
    if guardrails is None:
        guardrails = {
            "max_review_rate": 0.1,
            "min_flagged_fraud_capture": 0.85,
            "max_decline_fpr": 0.02,
        }
    return {
        "schema_version": 1,
        "guardrails": guardrails,
        "profiles": {
            "primary": {"approve_threshold": 0.11, "decline_threshold": 0.45},
            "fallback": {"approve_threshold": 0.19, "decline_threshold": 0.80},
            "phase2_guarded": {"approve_threshold": approve, "decline_threshold": decline},
        },
    }


class PolicyDriftCheckTests(unittest.TestCase):
    def test_compare_policy_configs_detects_no_drift(self) -> None:
        current = make_policy_payload(0.20, 0.80)
        fresh = make_policy_payload(0.20, 0.80)
        diffs = compare_policy_configs(
            current,
            fresh,
            required_profiles=("primary", "fallback", "phase2_guarded"),
            max_delta=1e-6,
        )
        self.assertEqual(diffs, [])

    def test_compare_policy_configs_detects_threshold_drift(self) -> None:
        current = make_policy_payload(0.20, 0.80)
        fresh = make_policy_payload(0.21, 0.80)
        diffs = compare_policy_configs(
            current,
            fresh,
            required_profiles=("primary", "fallback", "phase2_guarded"),
            max_delta=1e-6,
        )
        self.assertTrue(any("approve_threshold" in d for d in diffs))

    def test_compare_policy_configs_detects_guardrail_drift(self) -> None:
        current = make_policy_payload(0.20, 0.80)
        fresh = make_policy_payload(
            0.20,
            0.80,
            guardrails={
                "max_review_rate": 0.12,
                "min_flagged_fraud_capture": 0.85,
                "max_decline_fpr": 0.02,
            },
        )
        diffs = compare_policy_configs(
            current,
            fresh,
            required_profiles=("primary", "fallback", "phase2_guarded"),
            max_delta=1e-6,
        )
        self.assertTrue(any("guardrail max_review_rate" in d for d in diffs))

    def test_build_drift_report_formats_output(self) -> None:
        report = build_drift_report(
            policy_config_path=Path("models/policy_config.json"),
            data_path=Path("data/creditcard.csv"),
            diffs=["profile phase2_guarded approve_threshold differs: current=0.2 fresh=0.21"],
        )
        self.assertIn("Policy Drift Check Report", report)
        self.assertIn("Status: FAIL", report)
        self.assertIn("phase2_guarded approve_threshold", report)


if __name__ == "__main__":
    unittest.main()
