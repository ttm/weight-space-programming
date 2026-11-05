# One-time public open-model exploratory weight-synthesis smoke test

This optional integration workflow makes actual language-model inference possible **without a private API key or ChatGPT subscription**, using an open-weight checkpoint on a GitHub Actions CPU runner. Its four tasks are a software/protocol smoke test and **not** a confirmatory evaluation or statistically meaningful measure of model intelligence.

## Pinned inference contract

- Model: `Qwen/Qwen3-0.6B`, Hugging Face revision `c1899de289a04d12100db370d81485cdf75e47ca` (public checkpoint, Apache 2.0).
- Code: `scripts/run_local_hf_pilot.py`, with CPU-only PyTorch 2.6.0 and Transformers 4.57.1 installed **only in this workflow**. Normal package installation and standard tests have no machine-learning runtime dependencies.
- Decode: greedy; `max_new_tokens=640`, non-thinking chat-template mode, zero external tools, zero network evaluation probes.
- Inputs: one seeded generated instance from each of `dnf4`, `hinge1`, `l1ball2`, `threshold4`, seed 20261105; public behavioral specification and exact architecture, no reference weights or oracle labels in model prompt.
- Response contract: JSON candidate with literal numerical arrays, without an `explanation` field. Invalid JSON and mismatched architectures are recorded as failures, not auto-repaired or reprompted.
- Tests: only after the response is immutable, evaluate on the generated task's final test/OOD/edge splits.
- Records: prompt digest, raw model text, raw SHA-256, model revision, token count, elapsed seconds, error classification and held-out scores.

## Run

The workflow `.github/workflows/one-shot-open-model.yml` is triggered automatically exactly when `experiments/launch-2026-11-05.txt` first changes on `main`, or may be rerun manually through GitHub Actions. Ordinary CI never downloads the checkpoint and never performs this model inference. Results are uploaded as a workflow artifact for 30 days; the pipeline does not commit generated model content automatically.

The runner performs a zero-dependency dry run first:

```bash
PYTHONPATH=src python scripts/run_local_hf_pilot.py --dry-run --output .runs/dry-run.json
```

To reproduce inference locally, install a compatible CPU PyTorch and Transformers build, then run:

```bash
python scripts/run_local_hf_pilot.py --max-new-tokens 640 --output .runs/model-smoke.json
```

## Inference limits

- One sample per task, no repetitions, no variance estimates.
- The participant is a small open-weight model, not GPT-6 and not directly comparable to larger model reports.
- Four task *instances* from public *families* are not a held-out family generalization test.
- On GitHub-hosted Actions, network availability, checkpoint download, time limits, model framework versions and CPU constraints may prevent the run.
- A true confirmatory study would separate task generation and evaluation from model inference, preregister model selection and stopping rules, and use multiple independent tasks/seeds and baselines under documented budgets.
