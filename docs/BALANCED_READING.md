# Exploratory balanced weight-reading experiment (H2/H5)

Protocol prepared and committed **before** launching any model inference on this new bank. This is a transparent, predeclared exploratory study, **not external preregistration or sealed confirmatory evidence**. Code, task generator and seeds are public.

## Motivation and falsifier

The November 5 Qwen3-0.6B reading pilot predicted all zeros for all six networks. It scored 66.7% average bit accuracy and agreed on 3/3 equivalent-network pairs without correctly interpreting a single one. Both statistics were driven by the label distribution and constant-output collapse.

Here every target network's 16-point truth table contains **exactly eight zeros and eight ones**. Constant-zero and constant-one predictors therefore receive 50% bit accuracy, 0/16 exact functions and can agree across all pairs without achieving any *both-exact* scores. The **primary outcome is the proportion of underlying functions for which the original AND the gauge-transformed versions are both exactly interpreted**. The sample unit is the *function*, not each correlated presentation.

## Construction

- Four independently programmed function families: 4-bit parity, pairwise XOR, 3-input selector/multiplexer, and nontrivial signed threshold classifiers. Each is produced by a deterministic constructive network, not trained from samples.
- Identical architecture for every task: four binary input bits, four ReLU neurons, one affine output. Classification is `int(z>0)`.
- For each function, create an original and a mathematically equivalent version by hidden-neuron permutation and positive ReLU rescaling. The evaluator checks all 16 outputs and the restricted canonical fingerprint.
- Randomize presentation order. **Only** matrices, fixed input/output semantics and an anonymous ID reach a model prompt. Function family, pairing, construction seed, target outputs and source code are absent from prompts. (The full public repo and generator are still accessible outside those prompts.)
- One-shot inference: no validation tools, no code execution, no forward passes supplied, independent stateless prompts. Strictly parse only a JSON object with exactly one key `truth_table` containing 16 binary digits.

## Frozen first-pilot plan

- Development seed: `20261105`. Pilot sampling: one network/function per family, presented twice: **4 independent functions, 8 network presentations**. Larger `per_family` used for deterministic engineering controls, not claimed as model samples.
- Fixed model: `Qwen/Qwen2.5-1.5B-Instruct`, Hugging Face revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`; greedy decoding, max 128 new tokens, pinned PyTorch/Transformers runtime on GitHub Actions CPU. No paid API calls.
- Two exploratory *prompt* conditions on the **same** eight presentations: `direct` (read the matrices) and `deliberate` (instruct the model to silently compute hidden activations before returning JSON). Prompt, raw output, token counts and time are stored per presentation. Each prompt is an independent call with no prior trials in context.
- Record invalid JSON as invalid and incorrect. Do not repair, rerun selectively, inspect target labels during generation or select best candidate after inference.
- For each condition report: `valid/8`, `exact/8`, `both-exact/4`, `pairs-agree/4`, `trivial-agreement/4`, `constant-prediction/8`, and mean bit accuracy; compare explicitly with constant-zero and deterministic oracle-executor controls. A paired 4-function pilot has **no meaningful power for general capability estimates**.
- If the workflow fails, preserve logs/artifacts and report the infrastructure failure without inventing model outputs. Runtime performance/memory are not guaranteed on a free CI runner.

## Reproduce controls and verify archived outputs

```bash
python scripts/run_balanced_controls.py
python -m unittest discover -s tests -p test_balanced_reading.py -v
python scripts/run_balanced_reading.py --participant oracle --condition direct --output .runs/balanced-oracle.json
python scripts/run_balanced_reading.py --participant zero --condition direct --output .runs/balanced-zero.json
python scripts/verify_balanced_reading.py .runs/balanced-oracle.json .runs/balanced-zero.json
```

A successful reference executor provides a functionality ceiling, **not** a model reasoning achievement. After the model run, its exact report can be independently replayed using `scripts/verify_balanced_reading.py` without downloading the language model.

## Limits and next gate

This benchmark is intentionally small and elementary. Source-open deterministic tasks can be memorized or reconstructed; model capability estimates require separate held-out problems and multiple models with independently sampled generations. Pair agreement is neither sufficient nor necessary for understanding without exact function accuracy. The current benchmark does not establish interpretability of arbitrary trained large-model weights.

## Observed first-pilot outcome

The pinned model run completed successfully: **0/8 strictly valid JSON truth tables for each condition, 0/16 overall**. The model copied the input enumeration as the alleged truth table, identically for all presentations. Consequently, exact function and pair-level success are 0, but these are instruction/serialization failures rather than numerical forecasts. Unmodified model outputs are [direct](../experiments/reports/2026-11-05-balanced-reading-direct.json) and [deliberate](../experiments/reports/2026-11-05-balanced-reading-deliberate.json). The responsive [constrained-format study](CONSTRAINED_READING.md) is explicitly post-hoc exploratory.
