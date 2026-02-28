# Policy Simulation Report (Week 3)

## Recommended Threshold Policy
- Date: 2026-02-28
- Policy id: `primary_t1_0.1100_t2_0.4500`
- `approve`: score < 0.1100
- `review`: 0.1100 <= score < 0.4500
- `decline`: score >= 0.4500

## Expected Operational Impact
- Approve rate: 87.80%
- Review rate: 10.12%
- Decline rate: 2.08%
- Fraud capture (decline only): 89.33%
- Fraud capture (review + decline): 96.00%
- Decline FPR: 1.97%

## Guardrail Check
- Max review rate target: 10.20%
- Min flagged fraud capture target: 95.00%
- Max decline FPR target: 2.00%
- Status: PASS

## Fallback Policy
- Policy id: `fallback_t1_0.1900_t2_0.8000`
- `approve`: score < 0.1900
- `review`: 0.1900 <= score < 0.8000
- `decline`: score >= 0.8000
- Fallback impact:
  - Review rate: 6.23%
  - Fraud capture (review + decline): 93.33%
  - Decline FPR: 0.35%
- Use fallback only when review queue capacity is the dominant constraint.

## Sensitivity Summary
- Local perturbation test: `T1/T2` shifts of `±0.01` and `±0.02`.
- 25 combinations were evaluated around the primary pair.
- Observed ranges:
  - Review rate: 8.05% to 12.68%
  - Fraud capture (review + decline): 93.33% to 96.00%
  - Decline FPR: 1.78% to 2.15%
- Only 1 of 25 local pairs passed the revised guardrails, so this threshold should be monitored closely after rollout.

## Risks and Notes
- Thresholds were simulated on a historical test split; behavior may drift in production.
- Track review volume and decline false positives after deployment.
- Recalibrate thresholds when score distributions shift materially.
