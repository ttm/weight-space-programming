# Exploratory phase: implementation and evidence boundaries

**Status:** the code builds a reproducible experimental laboratory, not a registered scientific result. The current CI uses only local, deterministic development controls. No paid LLM API experiment is implied by successful CI.

## Construction task families

`python -m wsp.generated --family dnf4 --seed 101` prints a novel task specification **without** its compiled witness. Add `--reference` to inspect the symbolic compiler's weights in development only.

| Family | Input | Function | Fixed architecture |
|---|---|---|---|
| `dnf4` | 4 bits | seeded OR of 3 conjunctions of signed literals | 4-3(ReLU)-1(linear) |
| `hinge1` | real scalar | piecewise linear sum of three hinges | 1-3(ReLU)-1(linear) |
| `l1ball2` | 2 real coordinates | strict membership in centered L1 ball | 2-4(ReLU)-1(linear) |
| `threshold4` | 4 bits | seeded affine half-space on Boolean cube | 4-1(linear) |

The generator provides a *constructive realizability witness* for every instance, evaluates 32 seeds per family in the CI development script, and imposes exact depth/width/activations. The task family itself is public and basic; no claim of contamination resistance or sophisticated model discovery is warranted.

## Test and feedback budgets

A candidate must have `schema_version`, task ID, exact architecture, and finite literal weights. Binary evaluation uses `>0`, regression the largest absolute error `<=1e-6`. Boolean final tests enumerate all 16 points; real-valued tasks use salted, deterministic held-out samples and edge checks. `WSP_SEED_SALT` is read only at evaluation time and must remain outside an untrusted participant's workspace.

`python -m wsp.baselines --seeds 4 --budgets 1 3 10` compares Gaussian random search with hill climbing. A `budget=B` baseline **evaluates B proposals on validation** and selects the best (or hill-climbed incumbent) before one final held-out evaluation. This is not compute-equivalent to an LLM: record tokens, latency and number of generated proposals separately. The random weight scale is configurable. Hill climbing uses a simple full-tensor Gaussian perturbation, not an optimized zeroth-order method.

`python -m wsp.pilot --participant oracle_stub --seeds 1 --repeats 1 --output .runs/oracle.json` exercises the complete factorial `blind`/`budgeted (3)` by explanation `required`/`forbidden` across all task families. `random_stub` is a negative integration control. **Neither represents a genuine LLM experiment.** The runner stores the raw candidate, SHA-256, validation feedback, final results, prompt and condition. Invalid JSON consumes an attempt. The runner currently enforces the trial format but does not provide operating-system isolation for a model that can independently call tools.

## Optional model calls

`OPENAI_API_KEY=... python -m wsp.pilot --participant openai --model YOUR_MODEL --seeds 1 --repeats 1 --output .runs/model-pilot.json`

This calls the OpenAI Responses API (`store=false`, with no tool interface supplied) and incurs charges on the **separately configured API account**. ChatGPT subscriptions do not include API credits. This experiment must be explicitly opted into through a manual command or `workflow_dispatch` action. No API key is present in the repository and no CI run triggers paid calls automatically. Restrict the `--families` and trial counts to contain spend, review model pricing, and store the output artifact privately when the trial is meant to be sealed.

## What counts as a scientific trial

A genuine LLM trial requires a specified model version, frozen prompt and task distributions, audit of tool access, independent task seeds generated after protocol freeze, task-level statistical comparison with matched resources, and full transcripts. A public repository cannot enforce secrecy merely with an environment-variable salt if the participant has filesystem access to the runner. Place the evaluator behind an independent service or separate trusted process boundary before confirmatory tests.

For a first exploratory pilot, use one model, 5-10 independently generated tasks, crossed no-feedback / 3-probe conditions and explanation required / forbidden. Run repeated attempts. Label the report exploratory regardless of success; use it to estimate difficulty, error modes, and sample size before preregistration.

## Development result provenance

`python scripts/run_development.py` regenerates `results/development_research.json`. The result reports exact constructive witness counts, random/hill-climbing controls, two stub participants, and 16 gauge-transformed reading tasks. It is intentionally integer/Boolean-only for stable cross-platform CI and does not report an LLM success rate.

## Planned confirmatory design

Preregister primary outcomes and stopping rules in `docs/PREREGISTRATION.md` **before** drawing confirmatory tasks. Preserve both full oracle test scores and resource vectors (prompt/completion tokens, wall-clock, feedback calls, training FLOPs where estimated, cost). A single scalar normalizing LLM compute to optimizer passes is unjustified. Include symbolic-program compiler, explicit gradient training, random search, hill climbing and negative-controls for insufficient architectures.
