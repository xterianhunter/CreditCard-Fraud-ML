# Phase 2: Calibration + Cost-Based Policy Optimization

## Run Details
- Date: 2026-02-28
- Model path: `models/week2_best_logreg.joblib`
- Train rows: 227845
- Calibration rows: 28481
- Evaluation rows: 28481
- Grid step: 0.0100

## Cost Function
- Cost(false negative approve): 25.000
- Cost(false decline non-fraud): 5.000
- Cost(per review): 0.400

## Best Policy By Calibration Method

| Method | T1 | T2 | Total Cost | Cost/Row | Review% | Decline FPR | Flagged Fraud Capture |
|---|---:|---:|---:|---:|---:|---:|---:|
| platt | 0.1900 | 0.3000 | 148.6000 | 0.005218 | 0.21% | 0.00% | 77.27% |
| isotonic | 0.0400 | 0.4500 | 160.8000 | 0.005646 | 0.27% | 0.00% | 77.27% |
| none | 0.2000 | 0.8000 | 1138.0000 | 0.039956 | 5.16% | 0.33% | 86.36% |

## Recommended Policy
- Calibration method: `platt`
- `approve`: score < 0.1900
- `review`: 0.1900 <= score < 0.3000
- `decline`: score >= 0.3000
- Expected review rate: 0.21%
- Expected decline FPR: 0.00%
- Expected flagged fraud capture: 77.27%
- Total cost: 148.6000
- Cost per row: 0.005218