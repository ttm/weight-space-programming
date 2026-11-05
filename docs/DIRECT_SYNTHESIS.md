# Direct-weight synthesis benchmark v1 and first ChatGPT GPT-6 participant

**2026-11-05. Exploratory, publicly inspectable, and not an independent or blinded evaluation.**

This track asks a model to provide literal numerical matrices and biases that implement a functional specification in an **exactly prescribed architecture**. No gradients, parameter optimization, training scripts, or executable submission code can substitute for the weights. The evaluator computes the function of the submitted network rather than trusting an explanation. One deliberately impossible XOR architecture tests correct abstention.

## Test design

Ten distinct tasks span Boolean functions, continuous classifiers, exact piecewise-linear regression, multiclass control, and an impossible affine XOR. Their distribution and architecture are fixed in `src/wsp/direct_synthesis.py`. The task oracle is an independent direct mathematical function, not derived by executing the participant's candidate.

| Task | Function | Architecture | Assigned condition |
|---|---|---|---|
| `parity3` | 3-bit odd parity | 3→ReLU(3)→linear(1) | Blind, explanation required |
| `exactly2of4` | Exactly two of four bits | 4→ReLU(3)→linear(1) | Blind, weights only |
| `select3` | Binary data selector | 3→ReLU(2)→linear(1) | Up to 3 probes, weights only |
| `slanted_diamond2` | Rotated Manhattan region | 2→ReLU(4)→linear(1) | Blind, explanation required |
| `hat1` | Triangular bump | 1→ReLU(3)→linear(1) | Up to 3 probes, explanation required |
| `clip2` | Clipped affine function | 2→ReLU(2)→linear(1) | Blind, weights only |
| `shifted_absdiff2` | Shifted absolute difference | 2→ReLU(2)→linear(1) | Up to 3 probes, explanation required |
| `threeway_controller1` | Asymmetric three-class controller | 1→linear(3) | Up to 3 probes, weights only |
| `two_hats1` | Sum of two separated bumps | 1→ReLU(6)→linear(1) | Blind, explanation required |
| `xor2_impossible` | XOR with affine threshold only | 2→linear(1) | Blind, explanation required; submit impossibility proof |

For each realizable Boolean task, all binary inputs are tested. Continuous networks are tested on 512 deterministic in-domain points, 512 out-of-domain points, and explicit boundary points. Continuous regression passes if the largest absolute deviation is at most 1e-6. Every architecture and activation must match, with no extra hidden units.

Conditions `B-E`, `B-W`, `P-E`, `P-W` vary allowable validation feedback (0 versus 3 probes) and whether a textual explanation is required. These are *assigned conditions*, not evidence of causal ablation: **the first GPT-6 subject used zero probes for all ten submissions, including budgeted conditions**. Hence this initial run does not estimate effects of feedback or explanation requirements.

## First participant: what can actually be specified

A machine-readable description is committed as `experiments/submissions/chatgpt-gpt6-2026-11-05/manifest.json`.

| Field | Known status |
|---|---|
| Model family | **GPT-6**, as identified by this ChatGPT session's model context |
| Product surface | ChatGPT, standard text conversation with assistant tool access |
| Exact API model ID or model checkpoint | **Unknown / not exposed** |
| Backend build hash or release identifier | **Unknown / not exposed** |
| Reasoning | Reasoning-capable GPT-6. The actual effort tier (e.g., Medium, High, Extra High) **is not exposed** and must not be invented |
| Generation date | 2026-11-05; full backend inference start/end timestamp unavailable |
| Temperature, seed, model sampling details | **Unknown** |
| Tools available | GitHub connector, container execution, Python, web search |
| Tools used during benchmark development | GitHub repository inspection, local files and Python/terminal evaluation |
| Original system/developer prompts and hidden reasoning | Not fully exportable as an experiment artifact |
| Model output isolation | **None**; the same assistant designed and could inspect both task definitions and evaluation code |
| Method | Hand-derived analytic weight matrices transcribed into static JSON before first score; no weight optimizer was used to choose them |

The user requested a high-reasoning profile and internal model build identifier if available. Those claims cannot be authenticated within this interface. **Do not rename unknown settings to "high" or manufacture a build ID.**

## First-submission results

The initial matrices were written together in `scripts/write_first_submissions.py` and materialized before the evaluator was first invoked in this development workflow. The creation order is reflected in this session's tool trace, but is **not an independently certified timestamp or guarantee of blinded generation**. The resulting candidate files and a SHA-256 digest per exact submitted byte string are committed, and the report replays their evaluation unchanged.

**9/10 first submissions passed all required checks.** The ten mathematical intended behaviors were correctly constructed or addressed, but the `xor2_impossible` JSON included the logically valid proof in a `reason` field and **omitted the separately required `explanation` field**. Its strict result is a failure (9/10), not a corrected success (10/10). A separately documented amended example can pass; this is not retroactively counted in the initial metric.

These tasks were chosen and viewed by the same assistant that solved them. Some relate closely to the public development examples. The fraction 9/10 is therefore **not** an unbiased measure of GPT-6 competence on novel tasks, and has no credible confidence interval for that claim. The result does establish that this participant supplied nine syntactically valid, directly specified, numerically successful networks without a training loop in the resulting code.

The naive N(0,1) random-weight controls provide only weak architecture-matched sanity comparisons. The repository's existing symbolic constructions and supervised Adam are different baselines with different information/compute budgets.

## How to run the same benchmark on another AI

List the tasks and ask the model for a literal schema-valid candidate:

```bash
python -m wsp.direct_synthesis list
python -m wsp.direct_synthesis prompt slanted_diamond2
```

To enforce a **local** trial with a validation budget and a locked finalization:

```bash
python -m wsp.direct_sessions start slanted_diamond2 --session .runs/trial.json --participant YOUR_MODEL_NAME
# The participant outputs .runs/candidate.json containing schema_version=1, task, and network.layers.
# Blind sessions reject probe; budgeted sessions count rejected probes as well.
python -m wsp.direct_sessions finalize --session .runs/trial.json .runs/candidate.json
```

For a `P-E` or `P-W` task, up to three `probe` calls can be made before `finalize`, while blind conditions permit none. Use separate sessions and immutable artifact records for independent attempts. Unrestricted development scoring is also available:

```bash
python -m wsp.direct_synthesis evaluate experiments/submissions/chatgpt-gpt6-2026-11-05/hat1.json
python -m wsp.direct_synthesis report experiments/submissions/chatgpt-gpt6-2026-11-05
python scripts/verify_direct_synthesis.py
python scripts/run_direct_synthesis_controls.py
```

**The local session file is not an access-control boundary.** A truly sealed study requires private evaluation inputs, a separate evaluator process or service, inaccessible oracle code and tamper-evident logging. This public bank is designed for development and reproducibility, not a secret leaderboard.

## Next stage and criteria

1. Execute the same task/condition contracts with different models, with provider-reported model revision, per-request response IDs, prompt and candidate hashes, inference tokens, timestamps, and complete raw responses. Distinguish tools-enabled and zero-tool model instances.
2. Draw at least 20 tasks per independent generated family **after** freezing prompts, without allowing the subject to author or inspect solution code and hidden test seeds.
3. Introduce genuinely difficult multi-layer compositional networks and parameter budgets (100, 1,000 and beyond). Include mathematical impossibility cases and structured abstention.
4. Compare strict correctness and format compliance, near-miss performance, genuine query budgets, and separately accounted inference versus optimizer compute.
5. Preregister the confirmatory study and isolate the scoring service before claiming independent model ability or superiority to conventional synthesis.
