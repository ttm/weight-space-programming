# Tasks and verification ledger

This ledger records measurable acceptance criteria, not promised scientific outcomes. Update each checkbox only when the implementation and corresponding evidence exist. Related plans: [ROADMAP.md](ROADMAP.md), [PROTOCOL.md](PROTOCOL.md).

## 0. Experimental hygiene

- [x] Exact input/output dimensions and topology checked in generated-model trials.
- [x] Optional strict topology for ordinary sessions.
- [x] Candidate hashes and validation-budget accounting.
- [x] Structured provenance for historical candidate files, with unknowns marked as unknown.
- [x] Source-controlled open-model raw responses and independent integrity replay.
- [ ] Separate evaluator credentials and permissions for sealed tests (required before confirmatory claims).
- [ ] Architecture hash + candidate hash + signed, immutable record in each sealed attempt.

## 1. Balanced weight-reading study (H2/H5)

- [x] Construct 4 families of 4-bit Boolean classifiers with common `4-ReLU(4)-linear(1)` topology.
- [x] Verify an exact 8/8 class balance for every generated truth table across many seeds.
- [x] Generate and conceal original vs equivalent weight permutations and positive rescalings.
- [x] Prove equality of all 16 outputs and equality of restricted canonical fingerprints on all pairs.
- [x] Build parser and scorer: exact truth table, format compliance, per-bit accuracy, paired *both-exact*, trivial agreement.
- [x] Test oracle control at 100% and all-zero control at 0% exact / 50% bit accuracy.
- [x] Execute an explicit, no-paid-API, one-shot pinned model pilot through CI.
- [x] Archive raw responses and verify their hashes without re-running inference.
- [x] Write balanced-study interpretation, including the independent-function sample size and negative findings.

## 1b. Post-hoc constrained-output diagnostic

- [x] Preserve the original 16/16 invalid JSON outputs as failed strict-format submissions.
- [x] Predeclare a separate exploratory follow-up that forbids copying an input enumeration.
- [x] Implement an exact 16-token binary-alphabet decoder, plus separate restricted-token confidence/Brier scoring.
- [x] Verify oracle and all-zero controls without downloading a model.
- [x] Execute constrained output decoding on the pinned open model and archive the immutable full report.
- [x] Distinguish numerical function signal from the previous instruction-following failure: the forced-format model still collapsed to all-zero predictions.

## 1c. Evidence escalation after small-model negative results

- [ ] Compare at least one substantially stronger language model against the same balanced bank, with full raw provenance.
- [ ] Repeat on independent seeds and newly generated families without changing prompts after score inspection.
- [ ] Report separate rates for format compliance, exact function decoding, trivial agreement, and conditional confidence calibration.
- [ ] Freeze externally sealed H2 protocol before any confirmatory inference.

## 2. Novel weight synthesis (H1)

- [x] Four seeded constructive families, each with reference network and exact fixed architecture.
- [x] Deterministic random, hill-climb and supervised-Adam **development** baselines.
- [ ] Independent tasks of varied difficulty, at least 20 per difficulty stratum.
- [ ] JSON schema prompt ablations with exact numerical candidate requirement, including unformatted output controls.
- [ ] Unseen-family holdout and mathematically infeasible architecture controls.
- [ ] Compare one-shot and fixed-feedback protocols; keep failed JSON in denominators.
- [ ] Publish a non-LLM symbolic compiler baseline for all generated families.

## 3. Confirmatory inference (H1/H2)

- [ ] Freeze model versions, task families, prompts, sampler settings and analysis before tasks are drawn.
- [ ] Preregister primary metric, repeated independent task counts, confidence intervals and stopping rules.
- [ ] Prevent participants from accessing the oracle, hidden test seeds, previous trials and reference solutions.
- [ ] Compare models and classical methods at explicitly **separate** information and compute budgets.
- [ ] Include calibration, class-imbalance, chance and constant-prediction controls.

## 4. Sparse repairs and theory (H3–H10)

- [x] Sparse patch application and a deliberately simple bias restoration example.
- [ ] Trained checkpoints with reproducible generated defects and unchanged-network control.
- [ ] Sparse direct edits vs search and gradient repair under identical observed traces.
- [ ] Collateral-error and OOD regression scores for every repaired checkpoint.
- [ ] Identify gauge equivalence limitations and connect formal claims to explicit counterexamples.

## Result vocabulary

- **Engineering verified** means deterministic tests and mathematical reference witnesses pass.
- **Exploratory model observation** means an actual model generated logged outputs, without a preregistered population-level inference.
- **Confirmatory result** requires a frozen design, independent model sampling, sealed evaluation and preregistered statistical analysis.


## 2a. GPT-6 first-subject direct synthesis development benchmark (completed engineering)

- [x] Ten heterogeneous public fixed-topology tasks: logic, regression, geometry, control, and impossible affine XOR.
- [x] Four assigned conditions crossing blind/budgeted and explanation-required/forbidden.
- [x] Standalone CLI for public task specifications and exact numerical evaluation.
- [x] Local session bookkeeping for probe quotas, malformed submissions, frozen architectures, and one final decision.
- [x] First GPT-6 numerical candidate per task archived exactly, including the failed formatting case.
- [x] Strict results replay: **9/10** first candidates pass; invalid format remains in the denominator.
- [x] First-subject manifest with model identity, interface, observed file timestamp, and unknown build/reasoning settings explicitly null.
- [x] Naive same-topology N(0,1) random controls (0/100 exact for each realizable task), labeled weak and unmatched.
- [x] CI tests covering format, task oracles, topology, budgets, repeatability and finalization.
- [ ] Validate independent participants on externally supplied fresh tasks they did not design.
- [ ] Seal reference implementations and numeric test inputs from participants; benchmark is currently public.
- [ ] Freeze and preregister prompts, repeats, models, analysis and resources for confirmatory claims.


## 5. Expanded direct-synthesis round 2 (GPT-6 first participant)

- [x] Publish 14 additional fixed-topology challenges with explicit task/formal output semantics and four feedback/explanation conditions.
- [x] Submit literal GPT-6 weights for 13 realizable networks plus one affine impossibility proof, without training/optimization.
- [x] Preserve the **13/14** first-attempt score; keep the numerically unstable annulus original unchanged.
- [x] Document and evaluate a *separate*, post-hoc annulus repair with negative exterior decision margin; verify 20 additional deterministic seeds.
- [x] Run 390 Gaussian same-architecture random-control trials (weak sanity baseline only).
- [x] Export provider-neutral prompts, build v2 trial sessions with attempt/byte hashing and exact topology checks, add independent replay and CI tests.
- [ ] Collect independent model responses under frozen prompt/schema conditions, recording authentic provider-supplied version and tool availability.
- [ ] Run an independently administered, sealed, preregistered challenge suite with unseen task families and comparable information/compute budgets.
