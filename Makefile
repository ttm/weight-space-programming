PYTHON ?= python3
export PYTHONPATH=src
# Committed results use the public seeds, whatever the caller's environment holds.
unexport WSP_SEED_SALT

.PHONY: test benchmark stress repair

test:
	$(PYTHON) -m unittest discover -s tests -v

benchmark:
	$(PYTHON) -m wsp benchmark --directory examples --random-trials 30 --output results/initial_benchmark.json

stress:
	$(PYTHON) -m wsp stress examples/xor2.json --output results/xor2_stress.json

repair:
	$(PYTHON) -m wsp repair fixtures/xor2_corrupted.json repairs/xor2_bias_fix.json --output results/xor2_repair.json
