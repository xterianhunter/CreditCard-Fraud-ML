# Fraud-ML (Learning Project)
<img src="profile.jpg" width="200" />

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

## Environment Setup
```bash
cd /home/xterianhunter/Projects/Fraud-ML
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

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

## Run Week 2 Model Sweep (Scaled Logistic Regression)
```bash
cd /home/xterianhunter/Projects/Fraud-ML
python3 src/experiment_scaled_logreg.py \
  --data-path data/creditcard.csv \
  --c-values "0.1,0.3,1.0,3.0"
```
This writes:
- `reports/experiments_week2.csv`
- `reports/experiments_week2.md`
- `models/week2_best_logreg.joblib`

## Run Week 3 Policy Simulation
```bash
cd /home/xterianhunter/Projects/Fraud-ML
python3 src/policy_simulation.py \
  --data-path data/creditcard.csv \
  --model-path models/week2_best_logreg.joblib
```
This writes:
- `reports/policy_experiments_week3.csv`
- `reports/experiments_week3.md`
- `reports/policy_simulation_week3.md`

## Run Inference
```bash
cd /home/xterianhunter/Projects/Fraud-ML
python3 src/inference.py \
  --input-path data/creditcard.csv \
  --model-path models/week2_best_logreg.joblib \
  --policy-config-path models/policy_config.json \
  --output-path reports/inference_output.csv
```
Default policy profile is `primary` (Week 3 selected thresholds: `T1=0.1100`, `T2=0.4500`).

Optional profile switch:
```bash
python3 src/inference.py \
  --input-path data/creditcard.csv \
  --model-path models/week2_best_logreg.joblib \
  --policy-profile fallback
```
Supported profiles:
- `primary`: `T1=0.1100`, `T2=0.4500`
- `fallback`: `T1=0.1900`, `T2=0.8000`
- `phase2_guarded`: `T1=0.2000`, `T2=0.8000`
- `artifact`: uses threshold from model artifact metadata when available

## Run Week 4 Latency Benchmark
```bash
cd /home/xterianhunter/Projects/Fraud-ML
python3 src/latency_benchmark.py \
  --data-path data/creditcard.csv \
  --model-path models/week2_best_logreg.joblib
```
This writes:
- `reports/latency_benchmark_week4.md`

## Build Monitoring Snapshot
```bash
cd /home/xterianhunter/Projects/Fraud-ML
python3 src/monitoring_snapshot.py \
  --scored-path reports/inference_output.csv
```
This writes:
- `reports/monitoring_snapshot.md`

## Phase 2: Cost-Based Policy Optimization
```bash
cd /home/xterianhunter/Projects/Fraud-ML
python3 src/cost_policy_optimization.py \
  --data-path data/creditcard.csv
```
This writes:
- `reports/phase2_cost_policy_results.csv`
- `reports/phase2_cost_policy_report.md`
- `models/policy_config.json`

Default guardrails:
- `max_review_rate=10%`
- `min_flagged_fraud_capture=85%`
- `max_decline_fpr=2%`

Optional override example:
```bash
python3 src/cost_policy_optimization.py \
  --data-path data/creditcard.csv \
  --max-review-rate 0.08 \
  --min-flagged-fraud-capture 0.90 \
  --max-decline-fpr 0.015 \
  --policy-out-path models/policy_config.json
```

Promotion gate example (CI/CD):
```bash
python3 src/cost_policy_optimization.py \
  --data-path data/creditcard.csv \
  --require-feasible
```
This command exits non-zero when no guardrail-feasible policy is found.

## Run Tests
```bash
cd /home/xterianhunter/Projects/Fraud-ML
python3 -m unittest discover -s tests -v
```

## CI Test Gate
- GitHub Actions workflow: `.github/workflows/tests.yml`
- Trigger: push to `main` and pull requests
- Command executed in CI:
  - `python -m unittest discover -s tests -v`

## 5-Minute Demo Walkthrough
```bash
cd /home/xterianhunter/Projects/Fraud-ML

# 1) Train + artifact
python3 src/train_baseline.py --data-path data/creditcard.csv

# 2) Week 2 best-model sweep
python3 src/experiment_scaled_logreg.py --data-path data/creditcard.csv

# 3) Week 3 policy simulation
python3 src/policy_simulation.py --data-path data/creditcard.csv --model-path models/week2_best_logreg.joblib

# 4) Inference with final primary policy
python3 src/inference.py --input-path data/creditcard.csv --model-path models/week2_best_logreg.joblib --policy-profile primary --output-path reports/inference_output.csv

# 5) Week 4 operational checks
python3 src/latency_benchmark.py --data-path data/creditcard.csv --model-path models/week2_best_logreg.joblib
python3 src/monitoring_snapshot.py --scored-path reports/inference_output.csv
```
Expected artifacts to highlight:
- `reports/experiments_week2.md`
- `reports/experiments_week3.md`
- `reports/policy_simulation_week3.md`
- `reports/latency_benchmark_week4.md`
- `reports/monitoring_snapshot.md`
- `reports/final_phase1_summary.md`

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
- Week 3 experiments: `reports/experiments_week3.md`
- Week 3 policy report: `reports/policy_simulation_week3.md`
- Week 4 latency report: `reports/latency_benchmark_week4.md`
- Week 4 monitoring snapshot: `reports/monitoring_snapshot.md`
- Final summary: `reports/final_phase1_summary.md`
