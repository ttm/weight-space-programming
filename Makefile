PYTHON ?= python3
export PYTHONPATH=src
# Committed results use the public seeds, whatever the caller's environment holds.
unexport WSP_SEED_SALT

.PHONY: test benchmark stress repair results check research-results provenance-results model-replay balanced-results

test:
	$(PYTHON) -m unittest discover -s tests -v

benchmark:
	$(PYTHON) -m wsp benchmark --directory examples --random-trials 30 --output results/initial_benchmark.json

stress:
	$(PYTHON) -m wsp stress examples/xor2.json --output results/xor2_stress.json

repair:
	$(PYTHON) -m wsp repair fixtures/xor2_corrupted.json repairs/xor2_bias_fix.json --output results/xor2_repair.json

# Extra no-cost development results.
research-results:
	$(PYTHON) scripts/run_development.py

provenance-results:
	$(PYTHON) scripts/index_provenance.py

balanced-results:
	$(PYTHON) scripts/run_balanced_controls.py

# Regenerate every committed result.
results: benchmark stress repair research-results provenance-results balanced-results

# Replay archived model results without new inference or API credentials.
model-replay:
	$(PYTHON) scripts/verify_model_reports.py
	@for report in experiments/reports/*-balanced-reading-*.json; do \
	  if test -f "$$report"; then $(PYTHON) scripts/verify_balanced_reading.py "$$report"; fi; \
	done
	@if test -f experiments/reports/2026-11-05-balanced-reading-direct.json && \
	    test -f experiments/reports/2026-11-05-balanced-reading-deliberate.json; then \
	  $(PYTHON) scripts/compare_balanced_reading.py \
	    experiments/reports/2026-11-05-balanced-reading-direct.json \
	    experiments/reports/2026-11-05-balanced-reading-deliberate.json >/dev/null; fi
	@for report in experiments/reports/*-constrained-reading.json; do \
	  if test -f "$$report"; then $(PYTHON) scripts/verify_constrained_reading.py "$$report"; fi; \
	done

check: test results model-replay
	@git diff --exit-code -- results/ || { echo "results/ no longer matches committed files; commit intended changes." >&2; exit 1; }
