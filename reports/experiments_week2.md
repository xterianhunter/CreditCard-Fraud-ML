# Week 2 Experiments: Scaled Logistic Regression

## Run Details
- Date: 2026-02-27
- Train rows: 227845
- Test rows: 56962
- Target FPR constraint: 2.00%

## Ranked Results

| Rank | Model | Scaled | C | PR-AUC | ROC-AUC | Recall@FPR<=target | Threshold |
|---:|---|:---:|---:|---:|---:|---:|---:|
| 1 | scaled_logreg_c3.0 | yes | 3.000 | 0.762015 | 0.986317 | 0.893333 | 0.469247 |
| 2 | scaled_logreg_c1.0 | yes | 1.000 | 0.762013 | 0.986322 | 0.893333 | 0.469373 |
| 3 | scaled_logreg_c0.3 | yes | 0.300 | 0.760124 | 0.986304 | 0.893333 | 0.469070 |
| 4 | scaled_logreg_c0.1 | yes | 0.100 | 0.758006 | 0.986251 | 0.893333 | 0.469025 |
| 5 | raw_logreg_c1.0 | no | 1.000 | 0.753610 | 0.986122 | 0.893333 | 0.448401 |

## Recommended Config
- Model: `scaled_logreg_c3.0`
- Scaled: yes
- C: 3.000
- PR-AUC: 0.762015
- ROC-AUC: 0.986317
- Recall@FPR<=target: 0.893333
- Threshold: 0.469247

## Artifact
- Best-model output path: `models/week2_best_logreg.joblib`

## Notes
- Ranking prioritizes recall at constrained FPR first, then PR-AUC, then ROC-AUC.
- Keep threshold from selected model for policy simulation inputs.