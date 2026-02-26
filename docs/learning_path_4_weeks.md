# 4-Week Learning Path (Fraud-ML Project)

## Goal
Ship a strong Phase 1 fraud baseline with measurable business-ready metrics, then harden it with evaluation discipline and deployment-readiness fundamentals.

## Week 1: Foundations + Reproducible Baseline

### Learn
- Fraud detection framing: precision/recall tradeoff, PR-AUC, ROC-AUC, recall at constrained FPR.
- Why temporal split matters for leakage prevention.
- Project schema contract and safe feature selection.

### Build
- Run and understand current baseline pipeline:
  - `python src/train_baseline.py --data-path data/creditcard.csv`
- Fill `reports/baseline_metrics.md` with real run values.
- Add a short `reports/run_notes_week1.md` with assumptions and observed data issues.

### Deliverable
- Baseline metrics report completed with:
  - PR-AUC
  - ROC-AUC
  - Recall @ FPR<=2%
  - Threshold used

### Checkpoint
- You can explain exactly why your chosen threshold is acceptable for the 3-way policy (`approve/review/decline`).

## Week 2: Error Analysis + Better Features

### Learn
- False positives vs false negatives in fraud operations.
- Feature diagnostics: distribution shifts, outliers, scaling impacts.
- Calibration basics (how scores map to real risk).

### Build
- Add evaluation outputs:
  - Confusion matrix at selected threshold.
  - Top false positive and false negative examples.
- Create `notebooks/error_analysis.ipynb` for segment analysis (`Amount`, time buckets, score bands).
- Try 2-3 baseline improvements (example: scaling strategy, class weights, regularization tuning).

### Deliverable
- `reports/error_analysis_week2.md` with:
  - Error patterns found
  - What changed in model settings
  - Metric movement vs Week 1 baseline

### Checkpoint
- You can point to one model change that improved recall at the same FPR constraint.

## Week 3: Decision Policy + Reliability

### Learn
- Turning model scores into business actions with threshold bands.
- Stability and drift concepts for production ML.
- Basic model governance (versioning, reproducibility, experiment logging).

### Build
- Define and test two-threshold policy:
  - `approve`: score < `T1`
  - `review`: `T1 <= score < T2`
  - `decline`: score >= `T2`
- Simulate policy outcomes on test data (approval rate, review rate, decline rate, fraud capture).
- Add a simple experiment log file: `reports/experiments_week3.md`.

### Deliverable
- `reports/policy_simulation_week3.md` with recommended `T1/T2` and expected operational impact.

### Checkpoint
- You can justify thresholds in both ML terms (metrics) and ops terms (manual review load).

## Week 4: Production Readiness + Portfolio Quality

### Learn
- Inference pipeline design: validation, latency, failure handling.
- Monitoring plan: what to track pre/post deployment.
- How to communicate ML work to stakeholders.

### Build
- Add a minimal inference entrypoint (if missing) using `data_contract.py` checks.
- Add tests in `tests/` for:
  - schema validation failures
  - happy-path inference/prediction flow
- Write a final report consolidating:
  - baseline metrics
  - best model config
  - threshold policy
  - monitoring plan

### Deliverable
- `reports/final_phase1_summary.md`
- Clean repo narrative showing end-to-end fraud baseline workflow.

### Checkpoint
- A recruiter/manager can read your docs and understand the full ML lifecycle without opening notebooks.

## Weekly Time Split (Suggested)
- 40% implementation
- 30% evaluation and analysis
- 20% reading/learning
- 10% documentation

## Definition of Done (End of Week 4)
- Reproducible training run from CLI.
- Metrics and threshold policy documented with rationale.
- Error analysis completed and reflected in model decisions.
- Basic tests covering data contract and core pipeline behavior.
- Final report ready for portfolio/interview walkthrough.
