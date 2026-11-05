# Direct weight synthesis: GPT-6 exploratory round 2

**Date:** 5 November 2026. **Research status:** transparent, self-authored development experiment; neither blinded nor preregistered. The subject is GPT-6 in ChatGPT, not a pinned API checkpoint. This round complements rather than replaces [the first direct-synthesis study](DIRECT_SYNTHESIS.md).

## Question and protocol

Can an LLM write literal numerical weights and biases for a *specified neural-network topology* that implement nontrivial functions, including multi-layer computation and multiple outputs, without training or numerical search?

The test suite was prepared by the same participant that supplied the weights. The subject had access to task and oracle code, and to an execution environment. The first 14 weights/proof artifacts were written before the first scoring call in this conversation, with no optimization or gradient training; this tool ordering is not independently authenticated. **Do not interpret the results as performance on unseen, independently administered tasks.**

The fixed study cases have 14 distinct specifications, four feedback/explanation conditions (blind or at most three probes, explanation required or forbidden), and explicit fixed architectures. The subject used **zero validation probes** to generate all original candidate tensors, including those assigned a budget of three. Task answers come from independent mathematical oracles, not from evaluating the candidate as a reference. The evaluator checks exact truth tables for finite Boolean tasks; 512 seeded test samples, 512 OOD samples and targeted edges for continuous tasks. Regression passes only for maximum absolute error <= 1e-6. All results include exact candidate byte hashes. An accepted impossibility certificate is labeled as *text submitted*, not machine-proven mathematics.

| Task | Architecture | Parameters | Condition | First pass |
|---|---|---:|---|---|
| `parity5` | 5->ReLU(5)->1 | 36 | B-E | yes |
| `exactly3of6` | 6->ReLU(3)->1 | 25 | B-W | yes |
| `xor_pairs_and` | 4->ReLU(4)->ReLU(1)->1 | 27 | P-W | yes |
| `mux6` | 6->ReLU(4)->1 | 33 | P-E | yes |
| `full_adder3` | 3->ReLU(3)->2 | 20 | B-W | yes |
| `compare_two_2bit` | 4->ReLU(2)->3 | 19 | P-E | yes |
| `trapezoid1` | 1->ReLU(4)->1 | 13 | B-E | yes |
| `max3` | 3->ReLU(5)->ReLU(3)->1 | 42 | P-W | yes |
| `median3` | 3->ReLU(8)->ReLU(5)->1 | 83 | B-E | yes |
| `annulus_l1_2` | 2->ReLU(4)->ReLU(3)->1 | 31 | B-W | **no** |
| `rotated_quadrants4` | 2->4 linear | 12 | P-W | yes |
| `fourlevel1` | 1->4 linear | 8 | B-W | yes |
| `hinge2` | 2->ReLU(2)->1 | 9 | P-E | yes |
| `parity3_affine_impossible` | 3->1 linear | N/A | B-E | proof text accepted, not machine verified |

**First-submission result: 13/14 protocol passes** comprising 12/13 numerically correct realizable networks and 1 accepted negative-control certificate. Original submissions are immutable under `experiments/submissions/chatgpt-gpt6-2026-11-05-round2/`; the complete report is `experiments/reports/2026-11-05-gpt6-direct-synthesis-round2.json`.

## Important failure: numerically unstable annulus classifier

The first network implements, mathematically, the strict L1 annulus `0.5 < |x|+|y| < 2`. It computes `s=|x|+|y|` and the logit

```text
f0(s) = ReLU(s-0.5) - 2 ReLU(s-1.25) + ReLU(s-2)
```

Over real arithmetic this is positive precisely in the annulus and *zero outside*. In IEEE-754 binary64, cancellation leaves tiny **positive** residual logits on some outside points, causing false positives. Original results: **490/512** test points correct, **505/512** OOD correct, **8/8** targeted edges correct. For example, a false positive may have logit `2.220446049250313e-16`. This is a useful reminder that exact symbolic functional equivalence does not guarantee strict-threshold behavior under floating-point arithmetic.

**Post-hoc revised artifact:** `experiments/revisions/gpt6_round2_annulus_posthoc.json` uses the *same* layer dimensions but a different logit:

```text
f1(s) = s - 0.5 - 2 ReLU(s-1.25)
```

which is negative both below 0.5 and above 2. It passed the regular test, OOD, and edge splits and **20 additional seeded evaluation sets**. This is a **separate diagnostic revision**, not an allowed probe/revision under the original task's blind B-W condition. It must **never** be substituted into the first-attempt total.

The numeric performance is bounded to this interpreter, its inputs and IEEE-754 operations; it is not a formal proof of robust classification for every floating-point or mathematical real input.

## Controls and limitations

A deterministic architecture-matched Gaussian N(0,1) control ran **30 trials for each of 13 realizable tasks**: **0/390** exact solutions. This merely checks that chance weights rarely solve these exact tasks under this crude proposal distribution. It is not a competitive training or symbolic-program-synthesis baseline, nor does it match inference computation. Exact analytic ReLU constructions are known and remain the meaningful symbolic comparator.

The subjective success rate cannot establish that GPT-6 (or other frontier models) reliably synthesizes weights on unknown task families, because the same agent constructed and inspected the problem specifications, wrote the source, and ran evaluation. Prompt templates and model access conditions can further confound comparisons. The correct next study must be externally administered, with task and evaluator hidden from the subject, the candidate locked before scoring, and repeated independent attempts.

## Reproduce exactly

```bash
python -m wsp.direct_challenges_v2 list
python -m wsp.direct_challenges_v2 prompt median3
python scripts/verify_gpt6_round2.py
python scripts/run_round2_controls.py
python -m unittest discover -s tests -p 'test_direct_*v2.py' -v
```

For provider-neutral prompts:

```bash
python scripts/export_round2_prompts.py --output .runs/round2-prompts.jsonl
```

The committed `experiments/round2_prompts.jsonl` file already contains these 14 tasks without answers. Do not send the reference network files, session results, keys or oracle code as prompt context.

For another participant (example), using a budgeted task:

```bash
python -m wsp.direct_sessions_v2 start mux6 \
  --session .runs/model-a-mux.json --participant model-a --model-id PROVIDER_REPORTED_ID
python -m wsp.direct_sessions_v2 probe --session .runs/model-a-mux.json model-a-try1.json
python -m wsp.direct_sessions_v2 finalize --session .runs/model-a-mux.json model-a-final.json
```

For a blind task, skip `probe`; it is prohibited. The session archives submitted bytes and hashes in a sibling `*-artifacts` folder and rejects any attempt to resubmit after finalization. **The local CLI is not a sealed or security-isolated evaluator.** Use it only for transparent, auditable model comparisons. Run another model with the same prompt/schema, declared tool access, and all invalid submissions included in denominators.

## Provenance

`manifest.json` identifies the participant as **GPT-6 via ChatGPT**, dates this round to 2026-11-05, identifies that tool execution was available, and explicitly records unknown backend build, exact checkpoint, effort tier, sampling temperature, and random seed as `null`. No synthetic `ChatGPT GPT-6 high` or fabricated build hash is asserted.
