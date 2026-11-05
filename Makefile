PYTHON ?= python3
export PYTHONPATH=src
# Committed results use the public seeds, whatever the caller's environment holds.
unexport WSP_SEED_SALT

.PHONY: test benchmark stress repair results check research-results provenance-results

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

# Validate and index the exact byte-level origins of development artifacts.
provenance-results:
	$(PYTHON) scripts/index_provenance.py

# Regenerate every committed result.
results: benchmark stress repair research-results provenance-results

# Unit tests, then fail unless regenerating results/ reproduces the committed files exactly.
check: test results
	@git diff --exit-code -- results/ || { echo "results/ no longer matches the committed files; commit them if the change is intended." >&2; exit 1; }
