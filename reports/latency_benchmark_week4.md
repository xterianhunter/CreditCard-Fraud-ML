# Latency Benchmark Report (Week 4)

## Run Details
- Date: 2026-02-28
- Model path: `models/week2_best_logreg.joblib`
- Sample size: 20,000
- Batch size: 1,024
- Iterations (measured): 30
- Warmup iterations: 5

## Latency Summary (per batch predict_proba call)
- Mean latency: 0.348 ms
- P50 latency: 0.341 ms
- P95 latency: 0.383 ms
- P99 latency: 0.438 ms
- Throughput: 2,865,625.0 rows/sec

## Target Check
- Target p95 latency: < 200.0 ms
- Status: PASS

## Notes
- Latency reflects local batch scoring conditions and is not a networked service SLA.
- Re-run after major model or feature-pipeline changes.