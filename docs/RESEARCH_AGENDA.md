# Research agenda: beyond small hand-written weights

The core article, [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/), describes three phenomena: identifying an unknown network's learned function from its weights, repairing a failed policy by weight edits, and constructing networks from natural-language goals. This project separates those into testable capabilities and extends them with robustness, equivalence, and compositionality.

## Priority 1: genuinely blind, novel weight synthesis

**Hypothesis H1:** Under matched compute budgets, direct numeric weight synthesis by an LLM solves previously unseen, moderately complex tasks more often than architecture-matched random search and is competitive with program compilation. Falsify with a locked suite that includes unseen task families.

- Start with n-bit adders, multiplexers, 2D piecewise-affine controls, finite-state classifiers with explicit history inputs, gridworld policies, and integer arithmetic.
- Hold architecture constant within comparison groups; do not let output shape identify the task.
- Compare zero feedback vs 1/3/10 probes; explanation vs no-explanation; reasoning vs tool access; one-shot vs revision.
- Require the agent to supply **only parameters**, not executable code that secretly computes the target outside the network.
- Track valid-output rate, exact function success, OOD score, weight count, wall time, token count and number of observed forward passes.

**Important negative controls:** linear perceptron on XOR; deliberately insufficient width/depth; incompatible parameter count; contradictory specifications. Correctly declining impossible cases is a capability, not a failure.

## Priority 2: read networks with identical topology

**H2:** An LLM can identify a network's function from its numeric parameters when architecture, input encoding, labels, normalization, and superficial metadata are held constant, without forward passes.

- Build a matched-shape bank of classifiers for parity, majority, OR, AND, threshold and geometric-region tasks; remove filenames and task IDs.
- Randomize hidden-unit permutation and positively rescale ReLU circuits to destroy arbitrary numbering or magnitude conventions without changing the computed function.
- Ask for predictions with calibrated probabilities, a falsifiable internal circuit description and layer/neuron-level attributions.
- Evaluate diagnostic correctness on locked test networks and on randomly parameterized distractors.
- Also hide input/output layers in a deeper model, or show only a specified layer, to test whether structural circuit recognition generalizes.

**Falsifier:** performance collapses to chance under legal function-preserving reparameterizations or unseen architectures.

## Priority 3: sparse repair of real defects

**H3:** A reasoning-based edit can repair failed behaviors with fewer changed parameters or test queries than naive local search.

- Move beyond the included one-bias toy repair to faulty classifiers, buggy controllers, corrupted normalization, delayed transitions, and actual trained policies.
- Construct defects blindly and report an untouched baseline, exact set of changed scalar weights, number of edits, changed-parameter fraction, side-effect regressions and OOD behavior.
- Vary information access: failure trace only; full weights; weights plus one trace; weights plus K queries; activations and gradients allowed/forbidden.
- Include placebo patches, random edits, oracle sparse repairs and gradient-based fine-tuning as baselines.
- Check whether improvements survive unseen tasks instead of patching one failing example.

**Falsifier:** apparent improvement disappears after controlling for validation calls, or improvements are offset by other failures.

## Priority 4: compositional weight-space programming

**H4:** Known subnetworks can be composed by parameter-level constructions without retraining while preserving their individual specifications.

- Combine abs, max, threshold and parity modules into a multi-head policy.
- Test circuits that implement Boolean expression ASTs, finite automata transitions, monotone piecewise-linear functions, and exact arithmetic on bounded integer domains.
- Compare direct composition, symbolic compiler to weights, and numerical program synthesis.
- Measure failure propagation, width growth, numerical conditioning and sparsity under composition.

A neural compiler is **not** necessarily emergent WSP. It is a crucial baseline and an engineering tool.

## Priority 5: gauge freedom, identifiability and canonical weights

**H5:** Agent-generated explanations identify functions and features that are invariant to reparameterization, not arbitrary neuron indices.

- Permute hidden neurons with corresponding outgoing columns; rescale ReLU incoming and outgoing connections by reciprocal positive factors.
- Probe sign symmetries where architecture allows them, activation saturation, and redundant/zeroed neurons.
- Compare explanations before/after equivalent changes, including per-layer causal interventions.
- Explore canonicalization of weight-space equivalence classes and when it fails (nonlinearities, layer norm, tied weights).

Implemented now: permutation and positive ReLU scale tests in `wsp stress`, checked against identical numerical outputs.

## Priority 6: quantization, noise and minimum description length

**H6:** A weight-space designer can trade accuracy, robustness, and parameter precision consciously.

- Quantize to 32/16/8/4 bits or to fixed decimal digits, record task loss.
- Add Gaussian and adversarial parameter perturbations and measure robustness curves.
- Search for integer and sparse exact representations; penalize large coefficient magnitude and condition number.
- Compare weight complexity to a short conventional program or explicit circuit; many toy examples are more compact in symbolic form.

Implemented now: decimal rounding and Gaussian weight noise over 20 fixed seeds per noise level. Future work should estimate uncertainty across independent circuits, not just perturbations of one.

## Priority 7: architecture discovery vs direct numeric assignment

**H7:** Synthesis quality has a sharp dependency on available width, depth, activation function and representation.

- Present provably or plausibly insufficient architectures; expect abstentions.
- Fix network shape and evaluate numeric ability separately from architecture-search ability.
- Let agents propose architecture only in a separate tracked arm.
- Evaluate sensitivity to input encodings and normalization.

For example, XOR is not linearly separable in its raw 2D inputs; its failure with a single affine logit is a sanity check, not an intrinsic neural limit.

## Priority 8: physical control and dynamical systems

**H8:** Directly generated controllers can satisfy stability and performance constraints for low-dimensional physics.

- Discrete double integrator, thermostats, PID approximations, energy balancing and conservative force fields.
- Closed-loop tests with different initial states, disturbances, delayed observations and actuator limits.
- Require stability arguments or Lyapunov certificates where available, and test robustness to model mismatch.
- Compare MPC, analytic controllers, RL and direct-weight policies under matched information.

A controller that passes selected trajectories is not automatically stable; unstable or unsafe outputs should remain in simulation.

## Priority 9: explanations as a causal intervention

**H9:** Requiring a step-by-step weight-level explanation may improve verification and transfer, but may also anchor an agent to fragile symbolic approximations.

- Randomize explanation requirement independently of feedback quota and model identity.
- Use blinded scorers and locked prompts; separate *plausibility* from actual functionality.
- Test an explanation-after-submission arm to distinguish a design aid from post-hoc rationalization.
- Have independent agents predict behavior from a model and its explanation, then selectively ablate supposedly functional neurons.

## Priority 10: real large-model interventions

**H10:** Sparse editing of transformer weights can produce predictable behavioral changes without disproportionate collateral damage. This is the most ambitious and highest-risk track.

- Begin with small open-weight character transformers and transparent mechanisms (copy/induction, simple algorithmic tasks).
- Evaluate multiple held-out datasets and side effects; use parameter-efficient fine-tuning, activation steering, LoRA and causal tracing as baselines.
- Pin checkpoint hashes, tokenizer versions, decoding settings, and the exact edited tensor indices.
- Separate one-off intervention success from general weight-space programmability. Avoid extrapolating a report about one checkpoint to all LLMs.

## Practical sequence

| Stage | Deliverable | Gate |
|---|---|---|
| 0: implemented | 10 exact constructions, one failure, one bias repair, session harness, stress tests | All unit tests and examples pass |
| 1 | Matched-architecture, anonymous weight-reading bank | Above-chance blind identification across permutations |
| 2 | New tasks generated after prompts and baselines are frozen | WSP beats matched controls on preregistered primary score |
| 3 | Sparse repairs on trained small networks | Gains survive held-out cases and collateral-damage tests |
| 4 | Compositional compilation and constrained dynamical policies | Provable/exhaustive or stress-tested function preservation |
| 5 | Independent multi-model benchmark, sealed evaluator and public paper | Replication by an external team |

The most important transition is **from inspectable showcase examples to pre-registered, externally scored tasks that the agent has not seen and cannot tune against**.

[ROADMAP.md](ROADMAP.md) holds the current order of work. It runs stage 2 before stage 1, because the baselines, task generator and participant runner that stage 2 needs are also what stage 1 needs.

## Balanced null-prediction guardrail for H2/H5

Matched-topology, equivalent-parameter agreement must not be credited as understanding unless the underlying truth tables are also predicted correctly. The November 2026 exploratory pilots showed that a small model can agree on all pairs by repeating a constant-zero answer. All future reading studies should include label-balanced function banks or explicitly account for prevalence, and report per-function both-exact success, constant-prediction rates, chance baselines, and calibration separately from raw pair agreement. Under forced binary-token decoding, conditional next-token confidence is not the same as unrestricted model probability. See [RESULTS.md](RESULTS.md) and [CONSTRAINED_READING.md](CONSTRAINED_READING.md).
