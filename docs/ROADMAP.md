# Roadmap: from reproducible circuits to independently measured weight-space reasoning

**Updated 2026-11-05. This is the plan of record.** [TASKS.md](TASKS.md) tracks concrete acceptance tests, [PROTOCOL.md](PROTOCOL.md) defines experiment rules, [RESEARCH_AGENDA.md](RESEARCH_AGENDA.md) catalogs H1–H10, and [RESULTS.md](RESULTS.md) distinguishes observations from interpretation.

## What the evidence currently supports

| Evidence level | Present status | Claim permitted |
|---|---|---|
| L0: engineering controls | Implemented: exact tensor interpreter; 10 directly specified constructive examples; random/hill-climbing/Adam baselines; 128 generated witnesses; architecture-locked sessions; CI and provenance index | The apparatus executes and checks small networks; analytic synthesis is possible |
| L1: exploratory model calls | Two archived Qwen3-0.6B probes (2026-11-05): synthesis **0/4** valid weight artifacts; reading **0/6** exact functions, trivial all-zero agreement across 3 equivalent pairs | This specific model/prompt setup failed; output invariance alone is not understanding |
| L2: discriminating replicated studies | **In progress:** balanced matched-topology reading; model and prompt-format ablations | Requires multiple independent functions, properly reported uncertainty, and nondegenerate success |
| L3: confirmatory comparisons | Not started: external preregistration, genuinely sealed tasks, audit of tool access, repeated strong-model runs, information- and compute-aware baselines | No positive general WSP claim yet |
| L4: practical large-network repair | Not started: trained defects and collateral-damage testing | No claim of general trained-model editability yet |

The completed L1 experiments are archived under `experiments/reports/` with prompts, raw responses, model revision, validation and replay. Development witnesses, stubs, and generated references **are not LLM trials**. Historical examples have structured provenance (13/14 files, one deliberately metadata-free fixture), but some original generation details cannot be recovered; see [PROVENANCE.md](PROVENANCE.md).

## Research priority, in dependency order

### P0. Experimental validity and reporting (ongoing)

Keep exact architecture constraints, candidate byte hashes, attempt accounting, independent oracle verification, complete raw outputs, model IDs and immutable revisions. Each study must explicitly report whether evaluation is sealed, what is visible to participants, and whether its model inference was actually executed. Keep development oracles in separate records from model inputs.

**Exit:** `make check` and report verification pass; controls deliberately fail when topology, score, or parser contracts are violated. No claims based on unavailable model runs.

### P1. Balanced weight-reading and reparameterization (H2/H5) **NEXT**

The previous 3/3 pair agreement was a constant-output collapse. Construct a new fixed-topology bank with *exactly eight positive and eight negative outputs per truth table* across multiple functions (parity, pairwise XOR, selector/multiplexer, nontrivial linear threshold). For each function present an original and a hidden-unit-permuted, positively rescaled ReLU network. Both have identical input/output semantics and architecture. Do not expose the family, answers, source seed or pair identity in model prompts.

**Primary metric:** fraction of independent function pairs for which **both** networks are interpreted exactly. **Secondary:** individual exact truth tables, strict JSON validity, bitwise accuracy, agreement, rate of constant guesses, and comparison to all-zero/all-one controls (50% bit accuracy; 0 exact on this balanced bank). Evaluate oracle-reader and constant-reader controls before calling models.

**First model pilot:** one deterministic greedy run on each network with a pinned, publicly accessible 1–2B instruction model, recorded prompt, checkpoint SHA, number of output tokens, latency, exact raw text and parser errors. No external API secrets or paid model calls. An explicit one-shot GitHub Actions launch can perform inference. This is exploratory, small, public, and **not** a preregistered or sealed capability estimate.

**Exit:** model outputs (including negative results), checksums, and replayable scores are archived; interpretation distinguishes genuine pair success from trivial agreement. Then extend to repeated seeds and more capable models, without selecting tasks based on test outcomes.

### P2. Weight synthesis on genuinely novel specifications (H1)

Use constructively realizable generated families, with a fixed architecture known to contain a witness; compare weights-only, schema-constrained exemplars, and explanation-required prompts with matched inference settings. Count every invalid JSON response as failure, but report serialization compliance separately from mathematical accuracy. Start exploratory with one-shot/zero-validation, then compare feedback budgets of 0 and 3. Include **insufficient-architecture and contradiction** negatives where successful abstention is separately scored.

**Exit:** at least 20 independent tasks per stratum, with a family held out entirely, recorded per-task scores, and proper conditioning on an explicitly declared data/feedback/tool budget. The symbolic witness must remain a separate baseline, not be shown to the participant.

### P3. Stronger baselines, resources and preregistered analysis (H1/H2)

Random N(0,1), small-integer random, hill climbing, trained gradient systems, symbolic compilation, and an execution-enabled reader are distinct baselines. **Do not** call a comparison compute-matched just because the counts of validation probes are equal. Report the separately measurable resources: labeled oracle examples, forward evaluations, gradient steps, inference tokens, wall-clock time and approximate cost. Hold task, topology, prompt condition and random seeds fixed when making paired comparisons.

Before a confirmatory trial, freeze the model revisions/prompts/code and register `docs/PREREGISTRATION.md` or an external registration specifying population, task families, primary outcome, sample sizes, stopping/exclusions, multiple comparisons and confidence intervals clustered by independent task. Draw held-out tasks **only after** freeze. Put the evaluator behind access restrictions; an environment variable alone is not a security boundary.

**Exit:** the preregistration predates held-out task generation and inference, with independently checked reports and explicit negative results.

### P4. Sparse repair in trained networks (H3)

Train small models on genuinely different task instances, introduce blind parameter-level and functional defects with documented fault generators, and compare restricted direct edits to random-coordinate repair, gradient fine-tuning and unchanged controls. Measure repaired failures, regressions on previously correct cases, parameter count changed and validation calls. Distinguish independent sparse repairs from trivial restoration of a known original weight.

**Exit:** a repeatable trained-checkpoint repair study that reports collateral damage and failure cases, not only one illustrative bias fix.

### P5. Theory and dissemination (H4–H10)

Establish invariance criteria over equivalence classes `W ~ W'` iff their functions agree on the specified domain; study restricted canonicalization, identifiability, complexity and robustness, followed by compositional construction and control systems. Report *what remains non-identifiable* (e.g. redundant units, cancellations, distribution-dependent equivalence). Prepare a methods/benchmark article first, an empirical results paper only if evidence warrants it.

## Run strategy and safety rails

- GitHub Actions on pushes performs **dependency-free validation only**. Dedicated one-shot inference workflows run only when an explicit launch marker changes or the workflow is manually dispatched. No recurring/automatic paid API calls.
- Record each inference workflow's revision, runtime, result path and status. Never treat a successful CI job as proof of model success.
- `main` receives direct fast-forward commits, not PRs. Workflows should preserve artifacts even when inference fails. Avoid auto-deploying secrets or storing hidden keys in the public repository.
- Prefer a single decisive experiment over large matrices of weak tests. Keep exploratory and confirmatory results in different locations.

## Decisions before L3, not blockers for P1

A budget and credentials for genuinely stronger hosted models, access to a sealed evaluator, choice of external preregistration, and an intended paper venue. These are not necessary to validate the current open-weight pilot. Do not retroactively count previous development conversations or prompt-tuned runs as preregistered blind trials.
