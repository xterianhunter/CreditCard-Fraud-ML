from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from monitoring_snapshot import build_markdown, compute_psi


class MonitoringSnapshotTests(unittest.TestCase):
    def test_compute_psi_is_near_zero_for_same_distribution(self) -> None:
        ref = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)
        cur = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)
        psi = compute_psi(ref, cur, bins=5)
        self.assertLess(abs(psi), 1e-6)

    def test_compute_psi_increases_for_shifted_distribution(self) -> None:
        ref = np.array([0.05, 0.10, 0.15, 0.20, 0.25], dtype=np.float64)
        cur = np.array([0.50, 0.60, 0.70, 0.80, 0.90], dtype=np.float64)
        psi = compute_psi(ref, cur, bins=5)
        self.assertGreater(psi, 0.5)

    def test_build_markdown_contains_sections(self) -> None:
        report = build_markdown(
            scored_path=Path("reports/inference_output.csv"),
            total_rows=100,
            approve_rate=0.8,
            review_rate=0.15,
            decline_rate=0.05,
            mean_score=0.1,
            p95_score=0.4,
            p99_score=0.8,
            label_metrics=["- Decline FPR: 1.00%"],
            drift_metrics=["- Score PSI vs reference: 0.050000"],
        )
        self.assertIn("# Monitoring Snapshot", report)
        self.assertIn("## Label-Aware Quality Metrics", report)
        self.assertIn("## Drift Signals", report)


if __name__ == "__main__":
    unittest.main()
