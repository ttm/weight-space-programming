# Weight-Space Programming: a reproducible laboratory

A small, transparent experimental testbed for **reading, writing, repairing, and stress-testing neural-network weights as programs**, without gradients or training loops. This repository is an independent exploration inspired by the [November 2, 2026 article by Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab](https://rikiyat.github.io/blog/weight-space-programming/). It is **not** a reproduction of their larger experiments and makes no claim about independent LLM performance.

## What has actually been implemented

- **Ten direct-weight network constructions** across Boolean logic, nonlinearly separable tasks, real-valued functions, geometric regions, and a three-action controller. Weights are literal matrices in `examples/*.json`; there is no fitting or hidden optimizer.
- **One intentional negative control**, `examples/xor2_linear_failure.json`, which reaches only 3/4 on XOR.
- **A sparse weight-repair experiment**, fixing an intentionally corrupted XOR network by changing a single scalar (`fixtures/` and `repairs/`).
- **A deterministic pure-Python feedforward interpreter**, supporting linear, ReLU, tanh and sigmoid activations; binary, multiclass and regression outputs.
- **Evaluation on complete truth tables, seeded continuous tests, out-of-distribution inputs and exact boundary points** (where applicable).
- **Session-based controls**: no feedback (`blind`), fixed feedback budget (`budgeted`), or unrestricted validation (`open`); explanations `required`, `optional`, or `forbidden`; one recorded final evaluation.
- **Controls and analyses**: random weight baseline with the **same architecture**, quantization, Gaussian perturbations, hidden-neuron permutation and ReLU rescaling invariances.
- Unit tests and machine-readable baseline results in `results/`.

The constructions are **analytic circuit synthesis expressed as neural weights**. That is a useful first benchmark, but **not evidence that a general-purpose LLM can synthesize high-dimensional networks for unfamiliar problems**, outperform training, or infer network function from unlabeled weights. The research agenda separates those questions.

## Run in a fresh checkout

Requires Python 3.10+; **no external dependencies** for the core experiments.

```bash
python3 -m pip install -e .
python3 -m wsp list
python3 -m wsp describe xor2
python3 -m wsp evaluate examples/xor2.json
python3 -m wsp evaluate examples/xor2_linear_failure.json
python3 -m wsp benchmark --directory examples --output results/initial_benchmark.json
python3 -m wsp stress examples/xor2.json --output results/xor2_stress.json
python3 -m wsp repair fixtures/xor2_corrupted.json repairs/xor2_bias_fix.json --output results/xor2_repair.json
python3 -m unittest discover -s tests -v
```

Without installation, prefix commands with `PYTHONPATH=src` (on Unix-based shells).

Reports round floats to 12 significant digits, which absorbs last-bit differences between platforms' math libraries: regenerating `results/` reproduces the committed files byte for byte, and CI checks this with `git diff --exit-code results/`.

## Run controlled generation conditions

An external participant, human or LLM, must **write the candidate JSON artifact directly**. The harness never generates or optimizes candidate weights.

```bash
# Zero validation feedback; explanation required.
python3 -m wsp start xor2 --session .runs/xor-blind.json --mode blind --explanation required
python3 -m wsp finalize examples/xor2.json --session .runs/xor-blind.json

# At most two validation calls; no explanation field permitted.
python3 -m wsp start xor2 --session .runs/xor-budget.json --mode budgeted --budget 2 --explanation forbidden
python3 -m wsp probe fixtures/xor2_weights_only.json --session .runs/xor-budget.json
python3 -m wsp finalize fixtures/xor2_weights_only.json --session .runs/xor-budget.json
```

A `blind` session rejects `probe` altogether. `finalize` records a digest of the candidate, the score, and a timestamp, and rejects repeated finalization. **The `evaluate` command is an unrestricted developer tool**; preventing participants from bypassing the blind setup requires a separate execution service, inaccessible test-set seeds, and permissions or an audit mechanism. See [protocol](docs/PROTOCOL.md).

## Candidate format

`weights[out_neuron][in_neuron]` and `bias[out_neuron]`; hidden activations operate elementwise. Binary outputs are classified as `1` when the final scalar is strictly positive, `0` otherwise. Multiclass output uses index of the greatest logit, breaking ties toward the first index. Regression uses the scalar value.

```json
{
  "schema_version": 1,
  "task": "xor2",
  "network": {
    "layers": [
      {"weights": [[1, -1], [-1, 1]], "bias": [0, 0], "activation": "relu"},
      {"weights": [[1, 1]], "bias": [-0.5], "activation": "linear"}
    ]
  }
}
```

This compact candidate contains only required metadata and weights, without an explanation. Optional `author`, `method`, `explanation` and external prompt/provenance records are useful for experiments. JSON is parsed as data: no evaluation of arbitrary user-supplied Python expressions.

## Current results (local, reproducible)

| Direct-weight tasks | Primary held-out evaluation | OOD / boundary evaluation |
|---|---|---|
| AND2, XOR2, parity3, majority5 | Complete truth tables, 100% accuracy | Not applicable for binary domains |
| Compare2, interval1, diamond2 | 512 sampled cases each, 100% accuracy | 512 OOD and edge tests each, 100% |
| Thermostat1 | 512 sampled cases, 100% accuracy | 512 OOD and edge tests, 100% |
| Absolute difference, maximum | MSE approximately zero | MSE approximately zero |
| XOR incorrect control | 75% accuracy | Intentionally fails |
| Corrupted XOR then one-parameter repair | 50% then 100% accuracy | 1 of 9 parameters edited |

The tests are deterministic, and every result is inspectable. **They are not a secret benchmark, nor do they estimate generalization to unseen problem families.** In particular, binary tasks use the complete finite input space; their "test" split is exhaustive, not statistically sampled. Random-network baselines are weak sanity checks rather than competitive trained models.

## Repository map

- `src/wsp/`: network interpreter, task oracles, evaluator, perturbations, feedback-controlled session bookkeeping, sparse weight patches, CLI.
- `examples/`: ten successful directly authored networks and one negative control.
- `fixtures/`, `repairs/`: corrupted network and explicit repair patch.
- `tests/`: reproducibility, malformed input, feedback-budget, invariance, repair and score checks.
- `results/`: actual initial scores and diagnostic reports from running this code.
- [`docs/PROTOCOL.md`](docs/PROTOCOL.md): rigor, experimental conditions and limitations.
- [`docs/RESEARCH_AGENDA.md`](docs/RESEARCH_AGENDA.md): further investigations and falsifiable hypotheses.
- [`docs/RESULTS.md`](docs/RESULTS.md): interpretation, baseline caveats, and follow-up designs.

## Research question

**Can a model translate a behavioral specification into *numerical network parameters*, or diagnose and edit existing parameters, with a measurable advantage over known symbolic construction, random search, gradient methods, or conventional program synthesis?**

To answer this scientifically, we will need hidden tasks, sealed evaluation, architecture-matched baselines, repeated independent samples, and recorded model/tool access. A local example solved exactly is only the beginning.
