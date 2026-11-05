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
