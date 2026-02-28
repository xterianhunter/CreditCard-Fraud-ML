from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from latency_benchmark import build_markdown, percentile


class LatencyBenchmarkTests(unittest.TestCase):
    def test_percentile(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0]
        self.assertAlmostEqual(percentile(values, 50), 2.5)
        self.assertAlmostEqual(percentile(values, 95), 3.85)

    def test_build_markdown_includes_target_status(self) -> None:
        report = build_markdown(
            model_path=Path("models/week2_best_logreg.joblib"),
            sample_size=1000,
            batch_size=256,
            iterations=10,
            warmup_iterations=2,
            rows_per_second=10000.0,
            mean_ms=1.0,
            p50_ms=0.9,
            p95_ms=1.2,
            p99_ms=1.5,
            target_ms=200.0,
        )
        self.assertIn("# Latency Benchmark Report (Week 4)", report)
        self.assertIn("Status: PASS", report)


if __name__ == "__main__":
    unittest.main()
