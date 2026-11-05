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

- [ ] Construct 4 families of 4-bit Boolean classifiers with common `4-ReLU(4)-linear(1)` topology.
- [ ] Verify an exact 8/8 class balance for every generated truth table across many seeds.
- [ ] Generate and conceal original vs equivalent weight permutations and positive rescalings.
- [ ] Prove equality of all 16 outputs and equality of restricted canonical fingerprints on all pairs.
- [ ] Build parser and scorer: exact truth table, format compliance, per-bit accuracy, paired *both-exact*, trivial agreement.
- [ ] Test oracle control at 100% and all-zero control at 0% exact / 50% bit accuracy.
- [ ] Execute an explicit, no-paid-API, one-shot pinned model pilot through CI.
- [ ] Archive raw responses and verify their hashes without re-running inference.
- [ ] Write balanced-study interpretation, including the independent-function sample size and negative findings.

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
