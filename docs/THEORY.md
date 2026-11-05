# Weight-space programmability: functions, symmetries and identifiability

This note formalizes a tractable question that is often obscured by the phrase *programming in weight space*: **does an agent understand the represented function, or does it recognize arbitrary conventions of one parameterization?**

## 1. Functional equivalence

Fix a network architecture, activation functions, numeric input/output semantics and a domain \(\mathcal D\). Let \(\mathcal W\) be its parameter space and \(f_w:\mathcal D\to\mathcal Y\) its input/output function. Define

\[
w\sim_{\mathcal D}w' \quad\Longleftrightarrow\quad \forall x\in\mathcal D,\;f_w(x)=f_{w'}(x).
\]

This is an equivalence relation, but the quotient \(\mathcal W/\sim_{\mathcal D}\) can have complicated structure. Equality of class labels is weaker than equality of logits, and equality on a sampled test set is weaker than functional equivalence on all inputs. Every experiment must state which notion it tests.

For a first-layer ReLU hidden neuron \(h_j(x)=\max(0,w_j^\top x+b_j)\) and outgoing coefficient \(v_j\), any **positive** scalar \(s_j\) produces the same contribution after \((w_j,b_j)\mapsto s_j(w_j,b_j)\), \(v_j\mapsto v_j/s_j\):

\[
(v_j/s_j)\operatorname{ReLU}(s_j(w_j^\top x+b_j))=v_j\operatorname{ReLU}(w_j^\top x+b_j).
\]

Permuting hidden neurons while permuting the corresponding columns of the next layer also preserves output. These are *exact analytic identities* over real arithmetic (numerical implementations may differ by rounding).

The resulting permutation/positive-scaling operations form a restricted family of gauge transformations. They are **not** all function-preserving transformations. Duplicating neurons, canceling contributions, zero-output neurons and other degeneracies generate further equivalences. General deep network identifiability remains a harder problem.

## 2. A limited canonical representation

For a nonzero first-layer ReLU neuron, normalize its incoming weights and bias by \(s_j=\max(|w_{j1}|,\ldots,|w_{jd}|,|b_j|)\), multiply its outgoing coefficients by \(s_j\), then sort the resulting neuron tuples by their normalized incoming parameters. This produces the same canonical fingerprint under first-layer hidden permutations and positive scaling in the two-layer architecture used by the development bank.

Implemented in `wsp.weight_reading.canonicalize` with 12-decimal rounding. This is a **restricted numerical canonicalization**, not a unique canonical form for the represented function. It rejects zero incoming neurons and does not resolve redundant units, cancellations, nonlinear equivalences or arbitrary deep structures. Equality of its fingerprints is a useful invariance check; unequal fingerprints do **not** prove unequal functions.

## 3. What a reading experiment should measure

A sealed task bank should fix topology, activation functions, input encoding and output semantics, then show only weight tensors and ask participants to predict the function. For the initial four-bit bank, a function can be represented as an exact 16-bit truth table. Every challenge can be transformed by a secret permutation and positive scaling, without changing its answer.

Record: exact function identification, per-input prediction accuracy, confidence/calibration, invalid responses, explanations, latency and query access. Compare each model to (i) random or majority-class guesses, (ii) an analytical reading algorithm, and (iii) unrestricted network execution as a *privileged upper bound*, not as a competing no-execution model.

**Falsifiable criterion:** if an apparent ability depends on hidden neuron order or magnitudes, it should degrade after function-preserving transformations; if it reasons about function classes, performance should remain stable. A null result is informative. Do not overinterpret a strong result on a publicly solved bank.

## 4. Distinguishing three programmability regimes

1. **Symbolic compilation:** an explicit formula is translated into a network (the present constructive witnesses). It proves expressibility for that architecture and supports reproducible baselines.
2. **Task-directed numerical synthesis:** a general agent selects numerical parameters from a new behavioral specification under a locked compute/query budget. The pilot runner makes this testable, but does not yet establish it.
3. **Opaque-weight interpretation and modification:** an agent extracts meaning from trained or generated matrices and proposes predictable edits, potentially in large networks. Stronger evidence requires functional equivalence controls, real trained models, hidden tasks and serious baselines.

A surprising result in regime 1 is not automatically evidence for regime 3.

## 5. Threats and next theory problems

- Input/output behavior on finite samples may miss knife-edge boundaries, unreachable states and rare counterexamples.
- Functions can be equivalent under one domain but not another; a closed-loop controller requires trajectories, not only single-step mapping checks.
- Neuron-level attributions are not generally invariant under legal parameter changes. Ask for functional circuit explanations or equivalence-class descriptions, not privileged neuron indices.
- Canonicalization of general neural functions, invariance under normalization layers, and the computational complexity of weight-to-function inference are open research directions.
- Comparing token budgets to gradient passes via a single conversion factor is not scientifically justified. Report separate information budgets, forward-pass counts, model tokens, GPU/CPU seconds and monetary cost.
