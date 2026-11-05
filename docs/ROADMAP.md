# Roadmap

This file is the plan of record: what exists, what gets built next and in what order, and what counts as done. [RESEARCH_AGENDA.md](RESEARCH_AGENDA.md) holds the hypotheses, and [PROTOCOL.md](PROTOCOL.md) the rules every experiment follows. Update this file whenever a step lands or the plan changes.

## Where things stand

As of version 0.2.0 (November 2026) the laboratory works, but no independent LLM participant experiment has been run.

- **Harness.** A pure-Python interpreter for small feedforward networks, ten fixed tasks with deterministic splits, scoring, random-weight baselines, perturbation and invariance checks, sparse weight patches, and feedback-limited sessions (`wsp start`, `probe`, `finalize`).
- **Session integrity.** Rejected submissions count as attempts, the recorded hash is of the exact bytes scored, and `wsp start` records the participant, model, prompt hash and harness version. Setting `WSP_SEED_SALT` holds back the test and OOD inputs.
- **Scoring.** Regression outputs pass when the largest absolute error is at most 1e-6.
- **Reproducibility.** `make check` runs the tests, regenerates `results/` and fails unless the files match the committed ones byte for byte. CI runs it on Linux and macOS with Python 3.10 to 3.14.
- **Licenses.** MIT for code and data, CC BY 4.0 for documentation.

Still missing: tasks a participant hasn't seen, baselines with a matched budget, a way to run real models as participants, and an analysis plan. The ten example networks were written during development and are not trials (PROTOCOL §9), so the research question remains open (RESULTS.md §6).

## Provenance of the existing artifacts

Renato Fabbri, the repository owner, reported on 2026-11-05 that the existing networks were produced by **ChatGPT 6 with high reasoning, used through a ChatGPT Plus plan**. That covers the eleven networks in `examples/` (the ten constructions and the linear XOR control) and the two in `fixtures/`, all added in commit `b298e6e` on 2026-11-05. The repair patch in `repairs/` came in the same commit; its origin is to be confirmed in step 0.

The files themselves record this only loosely. Their `author` field says "GPT-6 (analytically constructed, not trained)", and `fixtures/xor2_weights_only.json` has no author field by design. Nothing records the exact model version, when each network was generated, the prompts and transcript, or whether code-execution tools were used. If the ChatGPT conversation can still be exported, archiving it would close most of that gap.

## Plan

| Step | Work | Needs | Status |
|---|---|---|---|
| 0 | Record which model made each network | – | next |
| 1 | Random-search baseline | – | implemented for development tasks; final confirmatory comparisons pending |
| 2 | Hill-climbing and gradient-training baselines | 1 | hill climbing implemented; gradient training pending |
| 3 | Task generator | 1 and 2, for difficulty levels | four realizable development families implemented; sealed difficulty strata pending |
| 4 | Participant runner | 0; 3 for real runs | stub runner and opt-in OpenAI API adapter implemented; sealed execution pending |
| 5 | Preregistration and analysis | 1–4 | planned |
| 6 | First pilot (H1) | 5 | deterministic stub controls passed; independent model study pending |

This order runs the agenda's stage 2 (synthesis on new tasks, H1) before its stage 1 (the weight-reading bank, H2), because the baselines, generator and runner built here are also what stage 1 needs.

### Step 0: record which model made each network

Every network and patch in the repository should say who made it, in structured fields rather than free text.

- Add a `provenance` object to candidate and patch files: model name, reasoning setting, access route (for example "ChatGPT Plus plan, chat interface"), generation date, tools used, and a reference to the prompt or transcript. Unknown values are `null`, never guessed.
- Backfill the existing files with the facts above, and confirm the origin of `repairs/xor2_bias_fix.json`. `fixtures/xor2_weights_only.json` stays minimal on purpose; its provenance is recorded here.
- `wsp benchmark` copies `provenance` into `results/`, as it already does with `author` and `method`.
- Give `wsp start` the same fields. It has `--participant`, `--model` and `--prompt-file`; it also needs the reasoning setting, access route and tools, so sessions and repository files describe a model the same way.
- In trials, provenance comes from the session, which the runner writes. What a participant writes about itself inside its submission is self-reported (PROTOCOL §9) and isn't trusted.

Done when every file in `examples/`, `fixtures/` and `repairs/` states its origin or says it is unknown, `results/` carries it, and a test fails for a new file in those folders without it.

### Step 1: random-search baseline

How well does blind search do with the same number of probes as a participant? A participant in a budgeted session sees validation scores for up to B candidates and then submits one. Random search gets the same deal.

- For each task and each budget B of 1, 3, 10 and 100: draw B random networks of the task's shape, score each on the validation split, keep the best, and score that one once on the final splits. Repeat with 30 seeds. Report the success rate per budget and the median number of probes before a network first passes validation. With B = 1 there is nothing to choose between, so that case equals the existing random baseline.
- Draw weights from N(0, 1), as the existing baseline does. Make the distribution a parameter, so scale-matched or small-integer variants can be added (see the decisions below).
- Keep the committed run small enough that `make check` stays under a minute; run larger budgets on demand.

Done when the results are committed in `results/` and reproduced by `make check`, tests check determinism and that the budget is respected, and RESULTS.md reports which tasks random search solves cheaply. Those tasks are weak tests of direct weight writing, and the numbers tell step 3 what "hard enough" means.

### Step 2: hill-climbing and gradient-training baselines

- **Hill climbing.** Start from a random network. Each probe perturbs the current best, and the change is kept if the validation score doesn't get worse. Same budgets as step 1.
- **Gradient training.** Pure-Python backpropagation for the four supported activations, keeping the zero-dependency rule. Train on a seeded training split labeled by the oracle; the participant has the specification, so labeled data is fair. Measure the budget in forward and backward passes.
- The hand-written constructions already serve as the symbolic baseline (PROTOCOL §5 (b)). A compiler from formulas to circuits (§5 (e)) comes later.

Done when both report success rate against budget on the step 1 tasks, committed and reproduced by `make check`.

### Step 3: task generator

The ten fixed tasks are textbook constructions whose solutions are in this repository, so a model may simply remember them. PROTOCOL §6 asks for at least 20 new tasks per difficulty level and at least one problem family held out entirely.

- Seeded families with a size parameter. Boolean: random functions of n bits given as truth tables, bits of an adder's output, multiplexers, threshold functions. Continuous: random piecewise-linear functions of one or two inputs, random convex regions (generalizing `diamond2`), and controllers with more actions (generalizing `thermostat1`).
- Each generated task has a natural-language and a formal specification, input and output semantics, an oracle, splits, and a network shape fixed per family, so the shape doesn't give away the function (PROTOCOL §8.2).
- Difficulty levels come from size and from the step 1 and 2 baselines, for example "random search with 10 probes succeeds on fewer than 5% of seeds".
- Development uses public seeds. Experiment tasks are drawn only after prompts and baselines are frozen (step 5), from secret seeds combined with `WSP_SEED_SALT`.
- In code, the fixed `TASKS` table and the `target()` chain become a registry of families, with the ten current tasks as fixed instances.

Done when at least 20 tasks per level and family can be generated deterministically from a seed, the evaluator, sessions and baselines all run on them, and tests pin a few generated instances.

### Step 4: participant runner

- A script outside the evaluator, so scoring never depends on who produced a candidate (PROTOCOL §7).
- For each trial: assign the condition at random with a trial ID (PROTOCOL §3); call `wsp start` with full provenance; send the frozen prompt (templates in PROTOCOL §7) with the task specification and network shape; keep the returned JSON exactly as received; probe within the budget; finalize once; and save the full transcript next to the session file.
- Pilot conditions: blind against budgeted with 3 probes, crossed with explanation required against forbidden.
- At least 5 independent attempts per model, condition and task (PROTOCOL §6), with a written cost estimate before anything runs.
- Test it first with a stub participant that returns stored or random networks, which costs nothing.

Done when a stub run completes end to end and every session file passes a provenance completeness check.

### Step 5: preregistration and analysis

- Before any experiment task is drawn or any model runs, commit `docs/PREREGISTRATION.md`. It states the hypothesis (H1); the primary metric (exact success on the final test, OOD and edge splits); the comparisons (each model and condition against random search, hill climbing and gradient training at the same probe budget); the statistics (bootstrap confidence intervals across tasks, paired comparisons on the same tasks); sample sizes; the stopping rule; and the exclusion rule, which is none: every failure counts, rejected JSON included. The commit date is the registration date.
- Freeze prompts and baselines with a tagged release before drawing the experiment tasks.
- An analysis script reads the session files and baseline results and writes its tables into `results/`, reproducible with `make check`.

Done when the preregistration is committed before the first pilot session and the analysis runs on the stub sessions from step 4.

### Step 6: first pilot

Draw the experiment tasks from secret seeds, run the participants, analyze exactly as preregistered, and report in RESULTS.md, including failure and rejected-JSON rates. A null or negative result is reported the same way.

## Implementation checkpoint: November 5, 2026

- **Architecture:** a supplied contract is now enforced by the ordinary session evaluator. Generated-model trials always lock exact depth, width and activations.
- **Task construction:** four deterministic families have constructively realizable witness networks. A development CI job checks 128 witnesses; the families remain public and elementary.
- **Baselines:** budgeted random search and hill climbing are implemented for generated tasks. Gradient training remains open.
- **Participants:** deterministic oracle/random stubs exercise all feedback/explanation conditions. An opt-in OpenAI Responses API participant and manual GitHub Action are available, but no paid model run has been made.
- **Interpretation:** 16 matched-topology development challenges preserve behavior and a *restricted* canonical fingerprint under positive ReLU rescaling and hidden permutations. This is not a complete canonicalization of neural functions.
- **Documents:** see [EXPLORATORY.md](EXPLORATORY.md) and [THEORY.md](THEORY.md).

Remaining prerequisites for confirmatory evidence include securely separated evaluator/participant, registered model trials, freeze of prompts and model versions, serious matched training/compiler baselines, gradient training and preregistration. Passing stubs do not demonstrate LLM competence.

## After the first pilot

The other hypotheses in RESEARCH_AGENDA.md reuse these pieces. The weight-reading bank (H2, the agenda's stage 1) needs same-shape networks for different functions, randomized with `permute_hidden` and `rescale_relu`. Repairing real defects (H3) needs the networks trained in step 2. Composition (H4) and the rest follow.

## Decisions for the repository owner

- Which models to test, and the API budget for the pilot.
- Whether chat-interface sessions, like the one that produced the existing networks, count as trials, and how their transcripts are archived.
- The weight distribution for random search: N(0, 1) only, or also scale-matched and small-integer variants.
- Where to preregister: a dated commit here, or an external registry as well.

## Housekeeping

- Add a `CITATION.cff`.
- Set the GitHub repository description, which is empty.
