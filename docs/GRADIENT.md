# Supervised gradient-training baseline

A minimal pure-Python reverse-mode differentiation and Adam implementation lives in `src/wsp/gradient.py`. This gives the repository a nontrivial conventional training comparison, without external numerical libraries. It is a **development baseline**, not an optimized or compute-matched comparison with an LLM.

## Run

```bash
python -m wsp.gradient --seeds 2 --steps 100 500 3000
python -m unittest tests.test_gradient -v
python scripts/run_development.py  # rewrites the compact, CI-tracked summary
```

## Training objective

The same architecture used by the symbolic witness is initialized with small Gaussian weights. For binary classification, the network's final scalar output \(z\) is a logit with label \(y\in\{0,1\}\) and objective

\[
\ell(z,y)=\log(1+e^z)-yz, \qquad \partial\ell/\partial z=\sigma(z)-y.
\]

The loss is computed with a numerically stable softplus formulation. For regression the objective is \(\frac12(z-y)^2\). Hidden layer gradients use exact reverse-mode differentiation of linear, ReLU, tanh and sigmoid units, with ReLU derivative zero at the kink. Every parameter is updated by Adam, with first- and second-moment bias correction and a default learning rate of 0.02. Unit tests compare the analytical gradient with central finite differences across binary and regression networks.

**Information regime:** supervised training uses oracle-labeled inputs from a separate deterministic `train` split. Boolean tasks supply the complete 16-point truth table to the optimizer, so test points in that finite domain are not independent out-of-sample points; real-valued tasks use 256 seeded training points. The held-out real-valued test/OOD and edge inputs are not used for training. LLM participants instead receive the exact symbolic function specification, not the full list of labeled examples. These are different information regimes and must be reported as such.

## Budget semantics

A training budget of \(B\) is exactly \(B\) stochastic single-example forward/backward pairs. This is **not equivalent** to \(B\) validation calls by a direct-weight participant, nor to \(B\) LLM forward passes. Report labeled examples, optimization steps, estimated compute, tokens and human/model engineering effort separately. Comparing under matched wall-clock/financial budgets will need machine details and model inference telemetry.

## Development observations

For eight fixed generated task instances (two per family), the number of trials passing all required test splits is:

| Adam updates | Exact successes / 8 |
|---:|---:|
| 100 | 0 |
| 500 | 0 |
| 3000 | 2 |

These are deterministic software checks, not estimates of optimizer superiority or LLM deficiency. Task families are elementary, initialization is not heavily tuned, and the exact regression tolerance (maximum absolute error <= 1e-6) is much stricter than an ordinary practical approximation criterion. The symbolic compiler has an exact network for **every** generated task by construction, providing a stronger reference baseline than both this small Adam implementation and Gaussian random search.

Next: tune and report strong optimization baselines on larger preregistered task distributions, compare multiple initializations, include continuous loss/accuracy, inspect near-boundary failures and quantify test leakage/compute separately.
