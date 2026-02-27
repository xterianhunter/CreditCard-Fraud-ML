# Error Analysis Report (Week 2)

## Run Details
- Date: 2026-02-27
- Train rows: 227845
- Test rows: 56962
- Threshold source target FPR: 2.00%
- Selected threshold: 0.378030

## Confusion Matrix (at selected threshold)
- TN: 55,753
- FP: 1,134
- FN: 8
- TP: 67

## Derived Metrics
- Recall: 0.893333
- Precision: 0.055787
- Realized FPR: 0.019934

## Error Sample Export
- Exported rows (FP + FN): 28
- Columns include `row_id`, `Time`, `Amount`, `Class`, `predicted_class`, `score`, `error_type`.

## Top Error Samples (first 5)
- row_id=274771, Time=166198.0, Amount=25691.16015625, actual=0, pred=1, score=1.000000, type=false_positive
- row_id=237192, Time=149133.0, Amount=1059.280029296875, actual=0, pred=1, score=0.999956, type=false_positive
- row_id=229953, Time=146124.0, Amount=302.6499938964844, actual=0, pred=1, score=0.999902, type=false_positive
- row_id=227921, Time=145283.0, Amount=10000.0, actual=0, pred=1, score=0.999873, type=false_positive
- row_id=274468, Time=166023.0, Amount=2.0, actual=0, pred=1, score=0.999836, type=false_positive

## Interpretation Notes
- Review high-score false positives to identify segments causing customer friction.
- Review low-score false negatives to identify missed fraud patterns.