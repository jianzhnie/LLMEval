# LLMEval Improvement Roadmap

## Completed

- Stable `doc_id` and explicit `sample_index` resume state.
- Redacted inference configuration logging and compact/debug score output.
- MC `first_token` default with explicit continuation mode.
- Structured scorer results with sample-level and problem-level math metrics.

## Next

- Run code evaluation in a container or OS sandbox with network isolation and
  CPU, memory, process, file-size, and filesystem limits.
- Carry inference failures in a run manifest so scoring can report them beside
  extraction, verification, and wrong-answer outcomes.
- Keep registry persistence as the single writer and remove legacy side effects
  from scorer implementations after downstream callers migrate.
- Add distributed resume locking and atomic output writes for multi-process
  runners.

## Schema Contract

Prepared inference input requires a unique non-empty `doc_id`. Output records
use `sample_index` for one sample and `sample_indices` for grouped responses.
Private `_llmeval_*` fields are request metadata only and must not be emitted in
final result JSONL.
