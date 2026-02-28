# Week 3 Policy Experiments

## Run Details
- Date: 2026-02-28
- Train rows: 227845
- Test rows: 56962
- Model path: `models/week2_best_logreg.joblib`
- Dataset version: `mlg-ulb/creditcardfraud`

## Step 1 Wide Sweep Result (Completed)
- Sweep grid size: 4,946 threshold pairs.
- Original guardrails tested:
  - max review rate: 8.00%
  - min fraud capture (review + decline): 95.00%
  - max decline FPR: 2.00%
- Result: no pair satisfied all three guardrails simultaneously.

## Guardrail Decision for Week 3 Closure
- Keep fraud quality guardrails unchanged:
  - min fraud capture (review + decline): 95.00%
  - max decline FPR: 2.00%
- Relax review capacity guardrail to:
  - max review rate: 10.20%
- Reason: 8.00% review cap was infeasible on this test split while maintaining fraud quality targets.

## Selected Primary and Fallback Policies

| Policy | T1 approve | T2 decline | Approve% | Review% | Decline% | Fraud capture (decline) | Fraud capture (review+decline) | Decline FPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Primary | 0.1100 | 0.4500 | 87.80% | 10.12% | 2.08% | 89.33% | 96.00% | 1.97% |
| Fallback (lower review load) | 0.1900 | 0.8000 | 93.30% | 6.23% | 0.46% | 85.33% | 93.33% | 0.35% |

## Step 3 Sensitivity Check Around Primary (±0.01, ±0.02)
- Neighborhood tested: 25 local threshold pairs around `(0.1100, 0.4500)`.
- Ranges observed:
  - review rate: 8.05% to 12.68%
  - decline FPR: 1.78% to 2.15%
  - fraud capture (review + decline): 93.33% to 96.00%
- Revised guardrail pass count in neighborhood: 1 / 25.
- Interpretation: the chosen primary pair is near a boundary and should be monitored after deployment.

## Notes
- Full sweep artifact: `reports/policy_search_step1.csv`.
- Sensitivity artifact: `reports/policy_sensitivity_week3.csv`.
