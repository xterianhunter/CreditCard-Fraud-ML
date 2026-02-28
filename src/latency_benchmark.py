"""Week 4 latency benchmark for batch fraud inference.

Usage:
    python src/latency_benchmark.py --data-path data/creditcard.csv --model-path models/week2_best_logreg.joblib
"""

from __future__ import annotations

import argparse
import time
from datetime import date
from pathlib import Path

import numpy as np

from data_contract import feature_columns, validate_dataframe
from inference import load_model_bundle, resolve_feature_columns


def score_in_batches(model: object, x: np.ndarray, batch_size: int) -> list[float]:
    """Return per-batch latency (milliseconds) for predict_proba."""
    latencies_ms: list[float] = []
    n = len(x)
    for start in range(0, n, batch_size):
        batch = x[start : start + batch_size]
        t0 = time.perf_counter()
        _ = model.predict_proba(batch)[:, 1]
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)
    return latencies_ms


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.array(values, dtype=np.float64), q))


def build_markdown(
    *,
    model_path: Path,
    sample_size: int,
    batch_size: int,
    iterations: int,
    warmup_iterations: int,
    rows_per_second: float,
    mean_ms: float,
    p50_ms: float,
    p95_ms: float,
    p99_ms: float,
    target_ms: float,
) -> str:
    status = "PASS" if p95_ms <= target_ms else "FAIL"
    return "\n".join(
        [
            "# Latency Benchmark Report (Week 4)",
            "",
            "## Run Details",
            f"- Date: {date.today().isoformat()}",
            f"- Model path: `{model_path}`",
            f"- Sample size: {sample_size:,}",
            f"- Batch size: {batch_size:,}",
            f"- Iterations (measured): {iterations}",
            f"- Warmup iterations: {warmup_iterations}",
            "",
            "## Latency Summary (per batch predict_proba call)",
            f"- Mean latency: {mean_ms:.3f} ms",
            f"- P50 latency: {p50_ms:.3f} ms",
            f"- P95 latency: {p95_ms:.3f} ms",
            f"- P99 latency: {p99_ms:.3f} ms",
            f"- Throughput: {rows_per_second:,.1f} rows/sec",
            "",
            "## Target Check",
            f"- Target p95 latency: < {target_ms:.1f} ms",
            f"- Status: {status}",
            "",
            "## Notes",
            "- Latency reflects local batch scoring conditions and is not a networked service SLA.",
            "- Re-run after major model or feature-pipeline changes.",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=Path("data/creditcard.csv"))
    parser.add_argument("--model-path", type=Path, default=Path("models/week2_best_logreg.joblib"))
    parser.add_argument("--sample-size", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--warmup-iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-p95-ms", type=float, default=200.0)
    parser.add_argument(
        "--report-out-path",
        type=Path,
        default=Path("reports/latency_benchmark_week4.md"),
    )
    args = parser.parse_args()

    if args.sample_size <= 0:
        raise ValueError("--sample-size must be > 0.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0.")
    if args.iterations <= 0:
        raise ValueError("--iterations must be > 0.")
    if args.warmup_iterations < 0:
        raise ValueError("--warmup-iterations must be >= 0.")

    import pandas as pd

    df = pd.read_csv(args.data_path)
    validate_dataframe(df, require_label=True)

    bundle = load_model_bundle(args.model_path)
    cols = resolve_feature_columns(df, bundle.artifact_feature_columns)
    if not cols:
        cols = feature_columns(df.columns)
    x = df[cols].to_numpy(dtype=np.float32)

    n = len(x)
    actual_sample_size = min(args.sample_size, n)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(n, size=actual_sample_size, replace=False)
    x_sample = x[idx]

    for _ in range(args.warmup_iterations):
        _ = score_in_batches(bundle.model, x_sample, args.batch_size)

    all_latencies_ms: list[float] = []
    total_scored_rows = 0
    total_elapsed_s = 0.0

    for _ in range(args.iterations):
        t0 = time.perf_counter()
        latencies_ms = score_in_batches(bundle.model, x_sample, args.batch_size)
        t1 = time.perf_counter()
        all_latencies_ms.extend(latencies_ms)
        total_scored_rows += actual_sample_size
        total_elapsed_s += t1 - t0

    mean_ms = float(np.mean(np.array(all_latencies_ms, dtype=np.float64)))
    p50_ms = percentile(all_latencies_ms, 50)
    p95_ms = percentile(all_latencies_ms, 95)
    p99_ms = percentile(all_latencies_ms, 99)
    rows_per_second = total_scored_rows / max(total_elapsed_s, 1e-9)

    report = build_markdown(
        model_path=args.model_path,
        sample_size=actual_sample_size,
        batch_size=args.batch_size,
        iterations=args.iterations,
        warmup_iterations=args.warmup_iterations,
        rows_per_second=rows_per_second,
        mean_ms=mean_ms,
        p50_ms=p50_ms,
        p95_ms=p95_ms,
        p99_ms=p99_ms,
        target_ms=args.target_p95_ms,
    )
    args.report_out_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_out_path.write_text(report, encoding="utf-8")

    print("=== Latency Benchmark Summary ===")
    print(f"Rows sampled per iteration: {actual_sample_size}")
    print(f"Iterations: {args.iterations}")
    print(f"Batches measured: {len(all_latencies_ms)}")
    print(f"Mean batch latency (ms): {mean_ms:.3f}")
    print(f"P95 batch latency (ms): {p95_ms:.3f}")
    print(f"Throughput (rows/sec): {rows_per_second:,.1f}")
    print(f"Target p95 < {args.target_p95_ms:.1f} ms: {'PASS' if p95_ms <= args.target_p95_ms else 'FAIL'}")
    print(f"Wrote report: {args.report_out_path}")


if __name__ == "__main__":
    main()
