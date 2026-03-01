# Phase 2: Calibration + Cost-Based Policy Optimization

## Run Details
- Date: 2026-03-01
- Model path: `models/week2_best_logreg.joblib`
- Train rows: 227845
- Calibration rows: 28481
- Evaluation rows: 28481
- Grid step: 0.0100

## Cost Function
- Cost(false negative approve): 25.000
- Cost(false decline non-fraud): 5.000
- Cost(per review): 0.400

## Guardrails
- Max review rate: 10.00%
- Min flagged fraud capture: 85.00%
- Max decline FPR: 2.00%

## Best Policy By Calibration Method

| Method | Guardrails | T1 | T2 | Total Cost | Cost/Row | Review% | Decline FPR | Flagged Fraud Capture |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| none | PASS | 0.2000 | 0.8000 | 1138.0000 | 0.039956 | 5.16% | 0.33% | 86.36% |
| platt | SOFT FAIL | 0.1900 | 0.3000 | 148.6000 | 0.005218 | 0.21% | 0.00% | 77.27% |
| isotonic | SOFT FAIL | 0.0400 | 0.4500 | 160.8000 | 0.005646 | 0.27% | 0.00% | 77.27% |

## Recommended Policy
- Calibration method: `none`
- Guardrail status: PASS
- `approve`: score < 0.2000
- `review`: 0.2000 <= score < 0.8000
- `decline`: score >= 0.8000
- Expected review rate: 5.16%
- Expected decline FPR: 0.33%
- Expected flagged fraud capture: 86.36%
- Total cost: 1138.0000
- Cost per row: 0.039956