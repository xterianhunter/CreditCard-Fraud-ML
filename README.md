# Fraud-ML (Learning Project)

## Purpose
This project is **only for learning and hands-on implementation of ML model deployment tech stacks**.

It is a practice repository to understand:
- end-to-end fraud model workflow
- model packaging and threshold-based decisioning
- deployment-oriented engineering basics (data contracts, reproducibility, evaluation, reporting)

## Important Disclaimer
- This is **not** a production fraud system.
- Metrics and policies here are for educational experimentation.
- Do not use this repository directly for real financial-risk decisions.

## Current Scope (Phase 1)
- Train a baseline `LogisticRegression` fraud model.
- Use temporal split to reduce leakage risk.
- Evaluate with:
  - PR-AUC
  - ROC-AUC
  - Recall at constrained FPR (target: `<= 2%`)
- Draft 3-way policy:
  - `approve`
  - `review`
  - `decline`

## Project Structure
- `src/` training and data contract code
- `data/` dataset files (Kaggle credit card fraud data)
- `models/` saved model artifacts (for example: `baseline_logreg.joblib`)
- `reports/` metric reports and run notes
- `docs/` problem definition and learning path/checklists
- `tests/` test placeholders for validation and inference flow

## Dataset
- Kaggle: `mlg-ulb/creditcardfraud`
- Expected schema: `Time`, `V1..V28`, `Amount`, `Class`

## Run Baseline
```bash
cd /home/xterianhunter/Projects/Fraud-ML
python3 src/train_baseline.py --data-path data/creditcard.csv
```
This run also saves a model bundle to `models/baseline_logreg.joblib` by default.

## Run Week 2 Error Analysis
```bash
cd /home/xterianhunter/Projects/Fraud-ML
python3 src/error_analysis.py \
  --data-path data/creditcard.csv \
  --model-path models/baseline_logreg.joblib
```
This writes:
- `reports/error_analysis_week2.md`
- `reports/error_samples_week2.csv`

## Run Inference
```bash
cd /home/xterianhunter/Projects/Fraud-ML
python3 src/inference.py \
  --input-path data/creditcard.csv \
  --model-path models/baseline_logreg.joblib \
  --output-path reports/inference_output.csv
```

## Run Tests
```bash
cd /home/xterianhunter/Projects/Fraud-ML
python3 -m unittest discover -s tests -v
```

## Learning + Deployment Focus
This repo is intended to practice practical deployment-oriented skills, including:
- model serialization/versioning
- threshold policy simulation
- schema validation/data contracts
- basic test coverage for inference safety
- reporting and model monitoring planning

## References
- Problem statement: `docs/problem.md`
- Learning plan: `docs/learning_path_4_weeks.md`
- Interactive checklist: `docs/learning_path_4_weeks_checklist.html`
- Baseline report: `reports/baseline_metrics.md`
