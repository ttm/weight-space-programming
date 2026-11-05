# Provenance and claims of model authorship

The baseline weight artifacts and repair patch have separate provenance requirements. A filename, `author` field or self-reported text inside a model-generated JSON candidate cannot authenticate how the artifact was created. Structured provenance is an auditable *claim*, not a signature or an independent reproduction.

## Existing artifacts

- The repository owner reported on November 5, 2026 that the ten working networks plus the deliberate XOR negative control and the corrupted fixture were authored using **GPT-6 with high reasoning in a ChatGPT Plus chat interface**. The exact model checkpoint, model identifier in API nomenclature, token usage, prompt transcript, generation time and tools accessed are unverified, so those fields are `null` where unknown.
- `fixtures/xor2_weights_only.json` intentionally contains no metadata. Its origin is recorded in the repository's roadmap instead, and the provenance index identifies it as the sole documented minimal-artifact exception.
- The provenance of `repairs/xor2_bias_fix.json` has not been separately confirmed. Its model, reasoning setting and other details are deliberately `null`, not inferred from file proximity.

The same eight keys are required in every non-exempt artifact's `provenance` object: `creator_type`, `model`, `reasoning_effort`, `access_route`, `generated_at`, `tools_used`, `prompt_reference`, and `evidence`. Unknown values remain `null`. See `wsp.provenance.validate`.

## Audit index

`python scripts/index_provenance.py` rebuilds `results/provenance_index.json` from **the exact bytes** of every JSON artifact under `examples/`, `fixtures/`, and `repairs/`. It includes SHA-256, pathname, task, status and claimed origin for each artifact. CI runs this as part of `make check` and rejects any uncommitted provenance drift.

This index checks that artifacts are unchanged and metadata is complete. It does not attest that a named LLM actually authored a file, that a candidate was written without tools, or that a participant was unable to inspect held-out tests. For new experiments, the runner/evaluator must independently record the model version, response ID, tool permissions, prompt hash, submitted byte hash, and final score; participant-written `provenance` is untrusted input.

## Follow-up

If a complete original chat transcript can be exported, record a stable pointer or a checksum in `prompt_reference` and preserve it outside the public repository if it contains sensitive material. Independently identify model snapshots and permitted tool access before using historical examples in claims of model capability. Never backfill unsupported facts.
