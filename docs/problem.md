# Fraud Detection Problem Definition (Phase 1)

## Objective
Build a card-not-present transaction fraud detection model that predicts whether an incoming transaction is fraudulent (`Class=1`) before approval.

## Business Context
- False negatives (missed fraud) are expensive and cause direct financial loss.
- False positives (blocking legitimate users) create customer friction and support overhead.
- The model should support a 3-way decision policy:
  - `approve`: low-risk transactions
  - `review`: medium-risk transactions sent for manual review/rules checks
  - `decline`: high-risk transactions

## Success Metrics
Primary metrics:
- `Recall >= 85%` at `FPR <= 2%`
- `PR-AUC` improvement over baseline

Secondary metrics:
- p95 inference latency target: `< 200ms`
- model output stability (score distribution monitored in production)

## Scope (Phase 1)
- Train and evaluate a baseline `LogisticRegression` model.
- Use temporal split strategy to avoid leakage from future into past.
- Produce threshold analysis for decision policy.

## Out of Scope (Phase 1)
- Deep learning models
- Online learning
- Full retraining automation
