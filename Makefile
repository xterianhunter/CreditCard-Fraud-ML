PYTHON ?= python3
DATA_PATH ?= data/creditcard.csv
MODEL_PATH ?= models/week2_best_logreg.joblib
POLICY_CONFIG ?= models/policy_config.json
SMOKE_INPUT ?= reports/ci_smoke_input.csv
SMOKE_OUTPUT ?= reports/ci_smoke_output.csv

.PHONY: test policy-status ci-smoke promotion-gate

test:
	$(PYTHON) -m unittest discover -s tests -v

policy-status:
	$(PYTHON) src/inference.py \
		--policy-config-path $(POLICY_CONFIG) \
		--policy-profile phase2_guarded \
		--print-policy-status

ci-smoke: policy-status
	$(PYTHON) -c 'import pandas as pd; required=["Time","Amount"]+[f"V{i}" for i in range(1,29)]; df=pd.DataFrame([{c:0.0 for c in required},{c:1.0 for c in required}]); df.to_csv("$(SMOKE_INPUT)", index=False)'
	$(PYTHON) src/inference.py \
		--input-path $(SMOKE_INPUT) \
		--model-path $(MODEL_PATH) \
		--policy-profile phase2_guarded \
		--policy-config-path $(POLICY_CONFIG) \
		--output-path $(SMOKE_OUTPUT)

promotion-gate:
	@if [ -f "$(DATA_PATH)" ]; then \
		$(PYTHON) src/cost_policy_optimization.py --data-path $(DATA_PATH) --require-feasible; \
	else \
		echo "Skipping policy promotion gate: $(DATA_PATH) not found."; \
	fi
