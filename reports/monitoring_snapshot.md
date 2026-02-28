# Monitoring Snapshot

## Snapshot Details
- Date: 2026-02-28
- Source file: `reports/inference_output.csv`
- Rows: 284,807

## Policy Volume Mix
- Approve rate: 82.86%
- Review rate: 14.58%
- Decline rate: 2.56%

## Score Distribution
- Mean score: 0.072756
- P95 score: 0.295922
- P99 score: 0.729382

## Label-Aware Quality Metrics
- Fraud rows in snapshot: 492
- Flagged fraud capture (review + decline): 97.15%
- Decline-only fraud capture: 92.07%
- Decline FPR (non-fraud auto-declined): 2.40%
- Review precision: 0.06%
- Decline precision: 6.22%

## Notes
- Treat this as an operational snapshot, not a formal model-evaluation report.
- Recompute frequently and trend metrics over time.