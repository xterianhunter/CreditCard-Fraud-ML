# Baseline Metrics Report

## Run Details
- Date: 26/02/2026
- Dataset version: mlg-ulb/creditcardfraud
- Train rows: 227845
- Test rows: 56962
- Model: LogisticRegression (`class_weight=balanced`)

## Metrics
- PR-AUC: 0.745961
- ROC-AUC: 0.984218
- Recall @ FPR<=2%: 0.90667
- Chosen threshold: 0.343726

## Threshold Policy (Draft)
- `approve`: score < 0.078720
- `review`: 0.078720 <= score < 0.343726
- `decline`: score >= 0.343726

## Notes
- Confusion matrix at threshold `0.343726`:
  - TN: 55,984
  - FP: 903
  - FN: 8
  - TP: 67
- Include sample false positives / false negatives for error analysis.
