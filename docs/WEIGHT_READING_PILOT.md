# One-time paired weight-reading experiment (development)

This smoke test exercises the original research article's **function identification from weights** question, rather than generating weights.

Model: pinned public `Qwen/Qwen3-0.6B` revision `c1899de289a04d12100db370d81485cdf75e47ca`, deterministic greedy CPU inference. The runner uses PyTorch 2.6.0 and Transformers 4.57.1 only in its dedicated GitHub Action, not ordinary CI.

For three deterministic DNF functions generated with `wsp.weight_reading.generate_bank(count=3,seed=77)`, the participant receives two anonymous parameterizations:

1. The original two-layer network with 4 binary inputs, 3 ReLU hidden neurons and 1 linear output.
2. A functionally equivalent network obtained through hidden neuron permutation and strictly positive incoming/outgoing ReLU scaling.

The prompt discloses architecture, input interpretation and binary output rule (`logit > 0`) but hides the symbolic specification, task generator seed, answer table and reference program. No execution or probe tools are exposed. The model must return a strict JSON object with a 16-bit truth table corresponding to inputs `0000..1111`, with no explanation.

The evaluator independently checks syntax, exact truth-table accuracy, bit-wise accuracy, and whether the two responses for the same function are identical under legal reparameterization. Prediction consistency alone is insufficient: two incorrect predictions may agree. All invalid responses count as failures; there is no reprompting or post-hoc repair.

The one-time workflow `.github/workflows/one-shot-weight-reading.yml` triggers only on the push adding `experiments/launch-reading-2026-11-05.txt`, or a subsequent manual dispatch. Its artifact holds all raw responses, hashes, exact scores and pairwise invariance statistics. The experiment is **exploratory**, with three paired tasks, no replication, a small language model, public task generation, and no generalizable capability estimate.

The theoretical rationale and limitations of the restricted canonical fingerprint are developed in [THEORY.md](THEORY.md). This pilot evaluates model behavior, not just the deterministic algebraic invariances checked by CI.
