# Exploratory language-model trials, 5 November 2026

**Status:** completed development experiments, not preregistered; no external paid API calls. Both independent model inference workflows completed successfully, and the main CI matrix passed. These experiments are not a statistically powered evaluation.

## Provenance

- Model: `Qwen/Qwen3-0.6B`, revision `c1899de289a04d12100db370d81485cdf75e47ca`.
- Decoding: deterministic greedy; CPU-based inference via public Hugging Face model checkpoint in GitHub Actions. No model access to evaluator or code tools; 0 validation probes.
- Synthesis original run: [workflow 42019675404](https://github.com/ttm/weight-space-programming/actions/runs/42019675404), strict success 0/4, but its upload skipped hidden `.runs` files; raw responses were observed in logs.
- Synthesis rerun with artifact retention fixed: [workflow 42020407036](https://github.com/ttm/weight-space-programming/actions/runs/42020407036), the data archived in `2026-11-05-synthesis.json`.
- Weight-reading initial run failed on a Python import before inference: [workflow 42020100201](https://github.com/ttm/weight-space-programming/actions/runs/42020100201). It contributes no measurements.
- Weight-reading successful run: [workflow 42020406965](https://github.com/ttm/weight-space-programming/actions/runs/42020406965), the data archived in `2026-11-05-reading.json`.
- Dry-run files include development-task reference results and reading-pair construction checks. They are not model-generated outputs.

## Weight synthesis: 0/4

Four seeded generated task instances from separate **public** families: `dnf4`, `hinge1`, `l1ball2`, and `threshold4`. The frozen architecture for each candidate and the formal specification were in the prompt. A successful submission had to contain **literal numerical matrices and biases**, conforming to the exact topology and meeting all task tests.

- Strictly valid candidate submissions: **0/4**.
- Solved tasks: **0/4**, since none reached valid numeric inference.
- All outputs were fenced Markdown JSON; additionally, after removing fences they contained task descriptions or architecture metadata, not correctly shaped numeric neural parameters.
- The secondary *post-hoc* JSON extraction diagnostic also recovered **0/4** candidate networks. This does not change the primary metric.

This establishes poor **instruction / representation compliance** for this specific small model and prompt. It does **not** show that appropriately specified numerical networks would fail the mathematical tasks.

## Weight interpretation under reparameterization: 0/6 exact

Three fixed-topology Boolean-function networks were each shown in original and function-preserving, neuron-permuted/positively rescaled form. Model outputs were 16-bit truth-table predictions.

| Statistic | Outcome |
|---|---:|
| Valid JSON responses | 6/6 |
| Exactly correct truth tables | 0/6 |
| Mean bitwise accuracy | 66.67% |
| Same prediction within equivalent pairs | 3/3 |
| Distinct model predictions across all six prompts | 1 |

Every response predicted **`0000000000000000`**. Thus pairwise consistency is *trivial* and cannot be interpreted as function-level invariance or recognition. The 66.67% marginal bit accuracy reflects the prevalence of zero labels, not genuine network interpretation. For this experiment, the all-zero predictor is the appropriate baseline, and exact function success is decisive.

## Scientific significance and limitations

These two negative exploratory findings are useful as **pipeline diagnostics**, not general evidence that language models cannot program or read weight tensors. The samples are tiny, model size is only 0.6B parameters, tasks and format requirements are narrow, all task families are publicly documented, and no stronger language model was sampled. Both inference runs are deterministic single samples, and the reading test has only three underlying independent functions.

The next properly interpretable study should include a stronger model, repeated independent tasks and model samples, instruction-format ablations, explicit baselines for all-zero reading, and registered criteria separating JSON conformance from mathematical correctness. An independent, permission-restricted evaluator is still needed for a confirmatory trial.

## Reproduction without inference

```
python scripts/verify_model_reports.py
```

This rechecks immutable recorded model outputs, hashes, summary counts and pair agreement. It **does not regenerate model completions**. Full prompt texts and exact raw completions are preserved in the two archived JSON files. Model checkpoint and code are pinned in the workflow scripts; to rerun inference, use the original workflow configuration and explicitly allow the associated resource use.
