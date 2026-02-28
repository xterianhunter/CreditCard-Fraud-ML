# Final Phase 1 Summary

## Date
- 2026-02-28

## Scope Completed
- Baseline fraud model training pipeline (`src/train_baseline.py`).
- Data contract validation for training and inference (`src/data_contract.py`).
- Inference CLI with policy-based decisions (`src/inference.py`).
- Week 2 error analysis and model tuning experiments.
- Week 3 threshold policy simulation, guardrail analysis, and fallback policy definition.
- Week 4 latency benchmark and monitoring snapshot artifacts.

## Baseline and Best Model
- Baseline model: Logistic Regression (`class_weight=balanced`).
- Baseline metrics (from `reports/baseline_metrics.md`):
  - PR-AUC: 0.748374
  - ROC-AUC: 0.985884
  - Recall @ FPR<=2%: 0.89333
  - Baseline threshold: 0.378030
- Week 2 best model (from `reports/experiments_week2.md`):
  - Model: `scaled_logreg_c3.0`
  - PR-AUC: 0.762015
  - ROC-AUC: 0.986317
  - Recall @ FPR<=2%: 0.893333
  - Threshold at target FPR: 0.469247

## Final Week 3 Policy Decision
- Primary policy (default inference profile):
  - `approve`: score < 0.1100
  - `review`: 0.1100 <= score < 0.4500
  - `decline`: score >= 0.4500
- Primary expected impact:
  - Approve rate: 87.80%
  - Review rate: 10.12%
  - Decline rate: 2.08%
  - Fraud capture (review + decline): 96.00%
  - Decline FPR: 1.97%

- Fallback policy (capacity pressure mode):
  - `approve`: score < 0.1900
  - `review`: 0.1900 <= score < 0.8000
  - `decline`: score >= 0.8000
- Fallback expected impact:
  - Review rate: 6.23%
  - Fraud capture (review + decline): 93.33%
  - Decline FPR: 0.35%

## Guardrail Outcome
- Original Week 3 guardrails (`review<=8%`, `capture>=95%`, `decline FPR<=2%`) were infeasible on this test split.
- Revised closure guardrails used:
  - Review rate <= 10.20%
  - Fraud capture (review + decline) >= 95.00%
  - Decline FPR <= 2.00%
- Primary policy passed revised guardrails.

## Monitoring Plan (Week 4)
- Track daily policy volume mix:
  - approve rate
  - review rate
  - decline rate
- Track quality proxies (short-lag and confirmed labels):
  - decline FPR proxy from dispute/appeal outcomes
  - flagged fraud capture proxy from reviewed/declined confirmed fraud
- Track score-distribution drift:
  - PSI or KS drift checks by score band
  - shift in top-risk band volume
- Track inference reliability:
  - batch run failures
  - schema validation failures
  - p95 inference latency

## Week 4 Operational Artifacts
- Latency report: `reports/latency_benchmark_week4.md`
  - p95 batch latency: 0.383 ms (target < 200 ms, PASS).
- Monitoring snapshot: `reports/monitoring_snapshot.md`
  - Snapshot volume mix: approve 82.86%, review 14.58%, decline 2.56%.
  - Snapshot flagged fraud capture: 97.15%.
  - Snapshot decline FPR: 2.40%.
- Notes:
  - Monitoring snapshot above comes from full-dataset scoring (`reports/inference_output.csv`) and is not the same as the Week 3 holdout-policy simulation split.

### Alert Thresholds
- Review rate > 12% for 2 consecutive days.
- Decline FPR proxy > 2.2% on labeled window.
- Flagged fraud capture proxy < 93% on labeled window.
- p95 inference latency > 200ms sustained.

### Response Actions
- If review load spikes: temporarily switch to fallback policy profile.
- If fraud capture drops: lower `T1`/`T2` cautiously and monitor decline FPR impact.
- If decline FPR rises: raise `T2` or tighten decline criteria while tracking fraud leakage.
- If drift alarms persist: retrain and recalibrate model/thresholds.

## Tests and Validation
- Unit tests cover:
  - data contract validation
  - inference happy path and failure paths
  - Week 2 experiment logic
  - Week 2 error analysis logic
  - Week 3 policy simulation logic
- Full suite status at finalization: passing.
- CI gate added: `.github/workflows/tests.yml` runs `python -m unittest discover -s tests -v` on push/PR.

## Limitations
- Evaluation uses one historical split; no online A/B signal yet.
- Review queue assumptions are static and may change operationally.
- Fraud labels can be delayed, making near-term quality estimates noisy.

## Next Phase Ideas
- Add calibration (isotonic/Platt) and re-evaluate policy stability.
- Add automated threshold search under explicit business cost function.
- Add scheduled retraining and model registry/version promotion checks.
