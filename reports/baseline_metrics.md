# Baseline Metrics Report

## Run Details
- Date: 2026-02-27
- Dataset version: mlg-ulb/creditcardfraud
- Train rows: 227845
- Test rows: 56962
- Model: LogisticRegression (`class_weight=balanced`)

## Metrics
- PR-AUC: 0.748374
- ROC-AUC: 0.985884
- Recall @ FPR<=2%: 0.89333
- Chosen threshold: 0.378030

## Threshold Policy (Draft)
- `approve`: score < 0.080000
- `review`: 0.080000 <= score < 0.378030
- `decline`: score >= 0.378030

## Notes
- Confusion matrix at threshold `0.378030`:
  - TN: 55,753
  - FP: 1,134
  - FN: 8
  - TP: 67
- Week 2 error samples and analysis are in `reports/error_analysis_week2.md`.
