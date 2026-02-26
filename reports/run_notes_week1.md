# Week 1 Run Notes (Phase 1)

## Date
- 2026-02-26

## Project Objective (Restated)
- Build a card-not-present fraud model to predict `Class=1` before approval.
- Optimize for high fraud capture while constraining false positives.
- Support a 3-way policy:
  - `approve`: low risk
  - `review`: medium risk
  - `decline`: high risk

## Current Status
- Baseline training code exists: `src/train_baseline.py`.
- Data contract exists: `src/data_contract.py`.
- Baseline report template exists: `reports/baseline_metrics.md`.
- Blocking issue: `data/creditcard.csv` is not present yet.

## Assumptions
- Dataset schema should include: `Time`, `Amount`, `V1..V28`, `Class`.
- Temporal split strategy will be used when `Time` is available.
- Primary target metric for Phase 1 is recall at constrained FPR (`<=2%`).

## Open Questions
- What review queue capacity should guide `T1/T2` policy thresholds?
- Is `FPR<=2%` fixed for all experiments or can it vary by policy scenario?
- Which dataset version/date should be tracked in baseline report metadata?

## Day 1-2 Completed Checklist
- [x] Read and restated objective from `docs/problem.md`.
- [x] Confirmed baseline training + data contract code exists.
- [x] Identified missing dataset as current blocker for metric generation.

## Day 3 Command (Run once dataset is available)
```bash
cd /home/xterianhunter/Projects/Fraud-ML
python3 src/train_baseline.py --data-path data/creditcard.csv
```

## Next Immediate Step
- Place dataset at `data/creditcard.csv`.
- Run baseline command and populate `reports/baseline_metrics.md` with actual values.
