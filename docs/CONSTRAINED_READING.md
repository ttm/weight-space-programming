# Constrained-output weight-reading follow-up (explicitly post-hoc exploratory)

**Research question:** Does a small language model show any nontrivial information about four neural functions when a strict 16-bit output format is mechanically enforced? This follow-up was **designed after observing** Qwen2.5-1.5B copy the *input enumeration* instead of the outputs in all 16 previous calls. Those earlier invalid-format results are immutable and remain in `experiments/reports/`. No confirmation, model-selection inference or preregistration claim is made.

## Fixed exploratory method before this follow-up inference

- Model: `Qwen/Qwen2.5-1.5B-Instruct` at revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, greedy with the exact two tokens encoding `0` and `1` allowed by the generation logits processor. Exactly 16 output tokens. No model tools, gradients, validation feedback or external paid calls.
- Same eight original/gauge-transformed network presentations as the previous study (`seed=20261105`, `per_family=1`): 4 independent functions, exactly 8 zeros and 8 ones for each. The model receives the same W1,b1,W2,b2 tensors and fixed input/output semantics, but the prompt does **not** enumerate the 16 input patterns to avoid a literal-copy affordance.
- All 16 emitted tokens come from a binary-constrained vocabulary. The model still chooses each bit by its logits conditional on preceding choices. This **changes the generation process**, and the result must not be represented as unconstrained instruction-following ability.
- The probability of a one is calculated from the **two restricted next-token logits** after filtering, not from full-vocabulary probabilities. Use it only as a binary decision confidence diagnostic; the mean Brier score provides a calibrated auxiliary metric.
- Primary metric: independent function pairs with **both** predictions exactly equal to the 16-case ground truth (`both_exact/4`). Secondary: individual exact tables (`exact/8`), mean bit accuracy, trivial or nontrivial pair agreement, number of all-constant predictions, restricted-token Brier score.
- Positive control: exact executor `8/8` and `4/4`, Brier 0. Negative controls: all-zero `0/8`, `0/4`, 50% bit accuracy, Brier 0.5 (confidently wrong on half the bits). An uninformative *probability* baseline of 0.5 per bit yields Brier **0.25**.
- Every original and transformed network is checked to be the same Boolean function; a model prediction is counted correct only on all 16 inputs. Answer keys are never included in prompts and become visible only in the archived output after generation.
- Report raw bit decisions, prompt text and SHA-256, binary token IDs, conditional per-bit probabilities, exact checkpoint revision, no-tool and no-probe metadata, and independently replayable scores.

## Interpretation safeguards

A successful constrained run cannot repair or reverse the earlier strict JSON failures. Because this experiment is a responsive prompt/output intervention, positive results would be hypothesis-generating and require independent replication on new task families, richer networks and stronger models. The four functions are too few for uncertainty or comparative-effect estimates. The source generator is public, not a sealed benchmark. A binary-token decoder creates the format automatically, not the underlying functional knowledge.

## Reproduce without model inference

```bash
python scripts/run_constrained_reading.py --participant oracle --output .runs/constrained-oracle.json
python scripts/run_constrained_reading.py --participant zero --output .runs/constrained-zero.json
python scripts/verify_constrained_reading.py .runs/constrained-oracle.json .runs/constrained-zero.json
```

The one-shot open-model workflow runs the `model` participant without an API key only when the explicit launch marker is pushed or manually dispatched. It archives raw model choices to `experiments/reports/2026-11-05-constrained-reading.json` upon successful completion.
