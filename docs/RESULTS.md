# Baseline experimental results

All results in this document were obtained locally with the code and candidate artifacts in this repository, using the deterministic seeds in `src/wsp/tasks.py`. Commands and raw, machine-readable reports appear below. This is an **implementation smoke test and analytic construction demonstration**, not an independent evaluation of LLM weight generation.

## 1. Directly specified weights

The ten intended-positive candidates pass their full available test, OOD and edge evaluations. XOR, parity and majority networks compute exact Boolean functions; absdiff and max compute exact ReLU identities up to floating-point rounding. The continuous classifiers use precise affine/ReLU decision regions. Their success is expected from the algebra, not a surprising statistical observation.

| Candidate | Test | OOD | Edge | Type |
|---|---:|---:|---:|---|
| `and2` | 4/4 | N/A | N/A | Binary |
| `xor2` | 4/4 | N/A | N/A | Binary |
| `parity3` | 8/8 | N/A | N/A | Binary |
| `majority5` | 32/32 | N/A | N/A | Binary |
| `compare2` | 512/512 | 512/512 | 4/4 | Binary |
| `interval1` | 512/512 | 512/512 | 5/5 | Binary |
| `diamond2` | 512/512 | 512/512 | 5/5 | Binary |
| `thermostat1` | 512/512 | 512/512 | 5/5 | 3-class |
| `absdiff2` | MSE=0 | MSE=0 | MSE=0 | Regression |
| `max2` | MSE=0 | MSE≈5.24e-32 | MSE=0 | Regression |
| `xor2_linear_failure` | 3/4 | N/A | N/A | Negative control |

Note: the text "MSE=0" above is exact for the tested floating-point samples when reported as 0.0. It is *not* a claim that floating-point arithmetic is exact under all machine inputs. Regression outputs are compared using the tolerance stated in the protocol.

## 2. Sparse repair

`fixtures/xor2_corrupted.json` has the same 2→2→1 ReLU/linear architecture as `examples/xor2.json`, but its output bias is -1.5 instead of -0.5. This suppresses both positive XOR inputs, leaving **50% accuracy** on the full four-case truth table. `repairs/xor2_bias_fix.json` explicitly changes the one output bias back to -0.5. Accuracy becomes **100%**, editing **1 of 9 scalar parameters (11.11%)**. The patch mechanism verifies the expected old value, index and tensor shape.

This is an intentionally trivial, transparently constructed fault. It is a functionality test for sparse-edit plumbing, **not** evidence of discovering a hidden defect in a learned neural network.

## 3. Perturbations and equivalences

For `xor2`, permuting the first hidden layer's two neurons and compensating in the next layer preserves the function. Scaling one ReLU unit's incoming weights/bias by 3 and dividing its outgoing weights by 3 likewise preserves the function. The evaluator confirms this on all sample splits.

Quantization of this integer/half-integer XOR construction to 3, 1, or 0 decimal digits retains 100% exhaustive accuracy (rounding -0.5 to -0.0 does not flip the strict binary threshold on the four cases). With independent Gaussian noise added to every scalar weight over 20 seeds, test accuracy averages **100% at sigma 0.01**, **98.75% at sigma 0.1**, and **58.75% at sigma 0.5**. Noise sensitivity is architecture and scale dependent, and these observations are specific to this circuit.

## 4. Random-parameter sanity baseline

`wsp benchmark` also assigns independent Gaussian N(0,1) weights to identical network shapes, with 30 fixed seeds per candidate. This is *not* a training baseline. For example, XOR random-weight variants average 45.83% truth-table accuracy, with 0/30 trials getting all four cases correct. The analytically specified network scores 100%.

For `and2`, **2/30** random networks happened to be perfect on the four-case truth table, illustrating why the sample size and difficulty distribution matter: simple Boolean tasks are an especially weak demonstration of synthesis superiority. A serious experiment needs training and symbolic baselines, task family holdouts, repeated model generations, and statistical comparisons over independent new tasks.

## 5. Exact commands

`make check` runs the unit tests, regenerates `results/` and fails unless every file matches the committed one byte for byte; CI runs it on Linux and macOS. The exact commands behind `make results` are in the [Makefile](../Makefile).

The `results/*.json` files retain the detailed task scores, baselines and invariance diagnostics. These files were generated from committed numeric artifacts; rerunning the commands does not involve model calls, stochastic training or web access.

## 6. What these results do and do not establish

- **Established:** The harness executes literal weight tensors, catches failures, evaluates distributions and boundaries, counts changed parameters, limits validation feedback within a session, and reproduces function-preserving transformations.
- **Established:** Explicit small neural circuits can encode some logical, piecewise-affine and piecewise-linear functions exactly.
- **Not established:** An LLM discovers these solutions reliably in one attempt across new tasks or can interpret arbitrary hidden weights.
- **Not established:** Direct WSP beats optimized baselines, gradient descent, reinforcement learning, or a symbolic compiler.
- **Not established:** Session feedback restrictions prevent access to local raw evaluation commands or publicly committed test seeds.

The next meaningful evidence would be a sealed evaluation in which agents receive specifications and fixed topologies and submit only immutable numeric artifacts. See `PROTOCOL.md` and `RESEARCH_AGENDA.md`.

## 7. Generated-task development controls (November 5, 2026)

Reproducible machine-readable summary: `results/development_research.json`, regenerated by `python scripts/run_development.py` and verified by CI. These are **not independent LLM trials**.

- Constructively specified networks: **128/128** successful tests across four elementary task families and 32 seeds per family.
- Gaussian random and hill-climbing baselines: **0/16** exact final successes for each method at budgets of 1, 3 and 10 validation probes. These are weak optimization controls, not a comprehensive training comparison.
- Oracle stub: **16/16** successes across no-feedback/three-probe and explanation-required/forbidden conditions; random stub: **0/16**. Stubs test runner bookkeeping only.
- 16 same-topology DNF weight-reading tasks pass the exact finite truth-table equivalence and limited canonicalization fingerprint tests after positive ReLU rescaling/permutation.

No scientific claim about general LLM weight-space reasoning follows from these development controls. See [EXPLORATORY](EXPLORATORY.md) and [THEORY](THEORY.md).

## 8. Pure-Python gradient baseline (development)

A conventional supervised Adam baseline, implemented with reverse-mode derivatives and validated against numerical finite differences, uses the same generated-task architectures. On eight tasks (two seeds per family), it passes all held-out splits **0/8** after 100 update pairs, **0/8** after 500, and **2/8** after 3,000. Partial test accuracy for Boolean families improves in several cases. The symbolic reference compiler remains 8/8 by construction.

This comparison is not compute- or information-matched: Adam receives labeled training examples and the LLM participant sees a symbolic behavioral specification. Its zero exact successes at lower budgets cannot establish superiority of direct weights. See [GRADIENT.md](GRADIENT.md) and the generated `results/development_research.json` summary.

## 9. First genuine model trials (November 5, 2026)

A pinned, open-weight, small instruction model (Qwen3-0.6B) ran in public GitHub Actions with **no model-accessible tools and zero validation probes**. These exploratory experiments have an immutable raw-output archive and a no-inference replay script at [experiments/reports](../experiments/reports/README.md).

- Synthesis: **0/4** valid weight submissions. All raw answers were Markdown-fenced JSON that reproduced task/architecture descriptions instead of numerical network weights; a secondary post-hoc extraction did not recover any valid weight artifact.
- Weight reading under equivalent ReLU reparameterization: **6/6** parseable JSON outputs, **0/6** correct truth tables, **66.67%** mean bitwise accuracy. All six predicted the same all-zero 16-bit label sequence, so **3/3 pair consistency** is a trivial degenerate baseline.
- The two Action workflows completed successfully. Those workflow statuses do not imply successful model performance.

**Interpretation:** negative small-model pilot, useful for exposing serialization compliance and trivial baseline artifacts. No conclusion can be extrapolated to general language-model competence, scaling, or causally meaningful invariance from 4 and 3 independent task instances. The original raw prompts, model replies, hashes, decoding metadata, topology contracts and final scores are archived.

## Balanced open-model weight-reading results (November 5, 2026)

A second actual inference study ran Qwen2.5-1.5B-Instruct (pinned revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`) on eight balanced original/gauge network views, across four underlying functions. There were two separate prompts per view: direct and deliberate. **All 16 completions were invalid for the strict JSON contract.** Both conditions had 0/8 exact interpretations and 0/4 both-exact pairs. The model mechanically repeated the list of 16 *input* patterns inside a fenced JSON object rather than reporting output bits, independently of weight values. This does not establish any mathematical inability to interpret weights, because no correct-format predictions were observed.

The before-inference protocol is [BALANCED_READING.md](BALANCED_READING.md); the exact raw reports are [direct](../experiments/reports/2026-11-05-balanced-reading-direct.json) and [deliberate](../experiments/reports/2026-11-05-balanced-reading-deliberate.json). The comparator and verifier replay each result without querying the model. A separate explicitly post-hoc [constrained-output protocol](CONSTRAINED_READING.md) is designed to measure whether conditional binary-token predictions recover any actual signal; it must not be retroactively treated as a preregistered arm of the first study.

## Constrained binary reading follow-up: format repaired, no reasoning signal

An **explicitly post-hoc** follow-up was run after the invalid-JSON failures. Model: `Qwen/Qwen2.5-1.5B-Instruct` revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, deterministic greedy decoding constrained to the single tokens for 0 or 1. The prompt omitted the original 16 input-enumeration strings and the decoder forced exactly 16 binary choices. The same four function instances and their eight equivalent parameter presentations were evaluated.

| Outcome | Result |
|---|---:|
| Valid 16-bit answers | 8/8 |
| Exact truth tables | 0/8 |
| Independent functions with both parameter views correct | 0/4 |
| Agreement of equivalent views | 4/4, all trivial constant-zero agreement |
| Mean bitwise accuracy | 50% (identical to constant-zero baseline) |
| Mean conditional binary-token Brier | 0.39443402 (unconditioned-calibration caveat) |
| Brier for uninformative fixed probability 0.5 | 0.25 |

Every model response was `0000000000000000`, despite balanced true functions. The mean `P(1)` from the binary-restricted token logits was about 0.135, showing a marked bias toward zero. Raw per-bit restricted-token probabilities and prompts are archived in [the exact report](../experiments/reports/2026-11-05-constrained-reading.json) and validated independently by `scripts/verify_constrained_reading.py`. The result is a clearer negative observation for this small model, not a general impossibility result for weight-space interpretation; the design changed after the initial failed experiment and includes just four underlying functions. See [CONSTRAINED_READING.md](CONSTRAINED_READING.md).
