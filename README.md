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
make check   # unit tests, then regenerate results/ and fail unless it matches the committed files exactly
```

Reports round floats to 12 significant digits, which absorbs last-bit differences between platforms' math libraries, so regenerating `results/` reproduces the committed files byte for byte. CI runs `make check` on Linux and macOS. `make results` only regenerates the files; the Makefile holds the exact commands.

To use the command-line tool directly:

```bash
python3 -m pip install -e .
python3 -m wsp list
python3 -m wsp describe xor2
python3 -m wsp evaluate examples/xor2.json
python3 -m wsp evaluate examples/xor2_linear_failure.json
python3 -m wsp benchmark --directory examples
python3 -m wsp stress examples/xor2.json
python3 -m wsp repair fixtures/xor2_corrupted.json repairs/xor2_bias_fix.json
```

Without installation, prefix commands with `PYTHONPATH=src` (on Unix-based shells).

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

A `blind` session rejects `probe` altogether. `finalize` records a digest of the candidate, the score, and a timestamp, and rejects repeated finalization. A rejected candidate (invalid JSON, wrong shapes, a broken explanation rule) still counts: it uses up a probe, or becomes the failed final submission. `start` can also record who writes the candidates, with `--participant`, `--model` and `--prompt-file` (stored as a SHA-256 hash). To hold back the test and OOD inputs, the evaluator sets `WSP_SEED_SALT` to a long random string, e.g. from `python3 -c 'import secrets; print(secrets.token_hex(32))'`; validation inputs stay public, and the final record stores the salt's SHA-256, never the salt. **The `evaluate` command is an unrestricted developer tool**; preventing participants from bypassing the blind setup requires a separate execution service, inaccessible test-set seeds, and permissions or an audit mechanism. See [protocol](docs/PROTOCOL.md).

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
- `examples/`: ten successful directly authored networks and one negative control, written by ChatGPT 6 with high reasoning ([provenance](docs/ROADMAP.md#provenance-of-the-existing-artifacts)).
- `fixtures/`, `repairs/`: corrupted network and explicit repair patch.
- `tests/`: reproducibility, malformed input, feedback-budget, invariance, repair and score checks.
- `results/`: actual initial scores and diagnostic reports from running this code.
- [`docs/PROTOCOL.md`](docs/PROTOCOL.md): rigor, experimental conditions and limitations.
- [`docs/RESEARCH_AGENDA.md`](docs/RESEARCH_AGENDA.md): further investigations and falsifiable hypotheses.
- [`docs/ROADMAP.md`](docs/ROADMAP.md): order of upcoming work, what counts as done, and open decisions.
- [`docs/RESULTS.md`](docs/RESULTS.md): interpretation, baseline caveats, and follow-up designs.

## Generated-task, model-pilot and weight-reading extensions

Four seeded, constructively realizable task families, exact architecture matching, random/hill-climbing and supervised Adam training controls, a factorial participant runner and ReLU gauge-invariance reading challenges are now available. **The controls are not independent LLM evaluations.**

```bash
python3 -m wsp.generated --family dnf4 --seed 202
python3 -m wsp.generated --family dnf4 --seed 202 --reference  # development only
python3 -m wsp.baselines --seeds 4 --budgets 1 3 10
python3 -m wsp.gradient --seeds 2 --steps 100 500 3000
python3 -m wsp.pilot --participant random_stub --seeds 1 --repeats 1 --output .runs/random.json
python3 -m wsp.pilot --participant oracle_stub --seeds 1 --repeats 1 --output .runs/oracle.json
python3 -m wsp.weight_reading --count 8 --seed 2 --bank .runs/reading.json --key .runs/reading-key.json
python3 scripts/run_development.py
python3 scripts/index_provenance.py
```

Ordinary `wsp start` accepts `--architecture-file architecture.json` to lock depth, widths and activations. Without that option, the original permissive development behavior remains. Generated pilot trials always lock architecture.

For actual model calls, explicitly opt into the manual [exploratory GitHub Action](.github/workflows/research-pilot.yml), configure an API key in the repository's `OPENAI_API_KEY` secret, and supply a model ID. Normal CI makes no paid calls; a ChatGPT subscription is separate from API billing.

The [completed November 5 model-trial report](experiments/reports/README.md) archives exact prompts, raw model outputs and a reproducible validation script. The tiny model produced **0/4 valid weight submissions** and **0/6 exact network interpretations**. The apparent 3/3 gauge consistency was a trivial all-zero-prediction collapse, not functional understanding. These are exploratory negative results only. Run `python scripts/verify_model_reports.py` for a no-inference integrity and statistics replay.

See [paired weight-reading model pilot](docs/WEIGHT_READING_PILOT.md), [one-time open-weight-model inference](docs/OPEN_MODEL_SMOKE.md), [artifact provenance](docs/PROVENANCE.md), [gradient baseline methodology](docs/GRADIENT.md), [exploratory methods](docs/EXPLORATORY.md), [weight-space equivalence](docs/THEORY.md), and the [roadmap](docs/ROADMAP.md). `make check` regenerates the development control summary in `results/development_research.json`.

## Research question

**Can a model translate a behavioral specification into *numerical network parameters*, or diagnose and edit existing parameters, with a measurable advantage over known symbolic construction, random search, gradient methods, or conventional program synthesis?**

To answer this scientifically, we will need hidden tasks, sealed evaluation, architecture-matched baselines, repeated independent samples, and recorded model/tool access. A local example solved exactly is only the beginning.

## License

The code and data (everything except the documentation) are released under the [MIT License](LICENSE). The documentation, `README.md` and `docs/`, is © 2026 Renato Fabbri and licensed under [CC BY 4.0](docs/LICENSE).

## Balanced function-reading study (November 2026)

The next exploratory H2/H5 study uses four **balanced** four-bit function families (parity, pairwise XOR, multiplexer, nontrivial threshold), each represented by an original and a mathematically equivalent ReLU network. Every truth table has 8 positive and 8 negative outputs, so a constant guess yields **50%** per-bit accuracy and **zero exact functions**, even when the guesses agree across every equivalent pair. The primary metric is **both presentations exactly correct**, counted per independent function, not per network.

Run the free, deterministic reference and failure controls:

```bash
python scripts/run_balanced_controls.py
python scripts/run_balanced_reading.py --participant oracle --output .runs/oracle.json
python scripts/run_balanced_reading.py --participant zero --output .runs/zero.json
python scripts/verify_balanced_reading.py .runs/oracle.json .runs/zero.json
```

The explicitly launched [one-shot GitHub Action](.github/workflows/one-shot-balanced-reading.yml) uses pinned Qwen2.5-1.5B-Instruct for eight genuine model predictions under two frozen prompt instructions, with no paid API key. Model inferences, if completed, are **exploratory** and cannot be conflated with constructive reference checks or statistical evidence of general skill. See [protocol](docs/BALANCED_READING.md), [implementation tasks](docs/TASKS.md) and [evidence-led roadmap](docs/ROADMAP.md).

### First balanced pilot: model results and follow-up

The November 5, 2026, pinned Qwen2.5-1.5B-Instruct run returned **0/8 valid truth-table JSON responses in each of two prompt conditions (0/16 overall)**. Every completion copied the input enumeration instead of computing model outputs. The strict zero scores are preserved in [direct](experiments/reports/2026-11-05-balanced-reading-direct.json) and [deliberate](experiments/reports/2026-11-05-balanced-reading-deliberate.json) raw-response archives. This is an observed **format/copying failure**, not a diagnostic of numerical reasoning ability.

To separate response-format compliance from function inference, the [post-hoc constrained-output follow-up](docs/CONSTRAINED_READING.md) allows only the two output-token choices `0` and `1` for exactly 16 positions. It records the model's choices and conditional binary-token probabilities with the *same* balanced tasks, preserving the original failed runs. This is explicitly exploratory, not evidence from preregistered tests.

### Constrained-output follow-up: verified negative result

The pinned Qwen2.5-1.5B-Instruct model was evaluated on the **same four balanced functions / eight network presentations** with a post-hoc restricted decoding method that mechanically forced each of 16 output tokens to be either `0` or `1`. The workflow successfully completed and archived [all eight raw predictions](experiments/reports/2026-11-05-constrained-reading.json). All eight predictions were `0000000000000000`, hence **8/8 format-valid, 0/8 exact, 0/4 both-exact equivalent pairs, 50% bit accuracy**. Its mean conditional-binary-token Brier score was **0.39443402**, compared with **0.25** for an uninformative probability of 0.5 on each bit. Despite 4/4 matched-pair agreement, **every agreement was trivial constant-output collapse**. These observations do not generalize to more capable models, unseen families or actual trained-network interpretation. The constrained approach is explicitly responsive to the first invalid-JSON results, not a preregistered confirmation.


## GPT-6 ChatGPT as first direct-weight synthesis subject (exploratory)

A public [direct-weight benchmark](docs/DIRECT_SYNTHESIS.md) now specifies ten exact-topology tasks across Boolean logic, nonlinear regression, geometric classification, multiclass control and affine-XOR impossibility. Tasks independently assign one of four conditions: zero versus up to three validation probes, crossed with explanation required versus weights-only. The local session runner enforces probe quotas and one final submission, **but is not a sealed evaluator**.

The first ChatGPT GPT-6 submissions are archived byte for byte in [experiments/submissions/chatgpt-gpt6-2026-11-05](experiments/submissions/chatgpt-gpt6-2026-11-05/manifest.json). Their [replayed report](experiments/reports/2026-11-05-gpt6-direct-synthesis.json) records **9/10 strict passes**. The remaining affine-XOR impossibility response contains a correct proof but missed the mandated separate explanation field, so its first attempt is a **failure**, not a retroactively corrected success. No validation probes were used, including for budgeted tasks.

The authoring model also designed and inspected these public task definitions and had tool access. The score **does not** measure blinded or independent model competence. See [model identity and unknown build/settings](docs/MODEL_PROVENANCE_GPT6.md). For the no-inference integrity replay run:

```bash
python scripts/verify_direct_synthesis.py
python -m wsp.direct_synthesis list
python -m wsp.direct_synthesis prompt two_hats1
```


## Second GPT-6 direct-weight trial and other-model handoff (2026-11-05)

A new public, self-authored development benchmark includes **14 additional tasks**, from Boolean parity, a four-way selector and multioutput full adder to exact max/median functions, multiclass geometry, floating-point-sensitive classification and an impossibility certificate. **13/14 original submissions pass**; the one failed classifier is preserved and its numerical cancellation mechanism is analyzed. A separately labeled post-hoc fix is not counted in the first-attempt score. Architecture-matched Gaussian random controls: **0/390** exact networks. Neither number estimates GPT-6's generalization to sealed, unseen tasks.

- [Method, results, limitations, and reproduction](docs/DIRECT_SYNTHESIS_ROUND2.md)
- [Original GPT-6 submission files](experiments/submissions/chatgpt-gpt6-2026-11-05-round2/)
- [Model-neutral JSONL prompts for other AIs](experiments/round2_prompts.jsonl)
- [Replayable score report](experiments/reports/2026-11-05-gpt6-direct-synthesis-round2.json)

Run:

```bash
python -m wsp.direct_challenges_v2 list
python -m wsp.direct_challenges_v2 prompt median3
python scripts/export_round2_prompts.py --output .runs/round2-prompts.jsonl
python scripts/verify_gpt6_round2.py
python scripts/run_round2_controls.py
```

See `python -m wsp.direct_sessions_v2 --help` for local probe/finalize bookkeeping. **This is not a sealed evaluator.** The existing `make check` and dedicated CI workflow replay the original artifacts and independent controls.
