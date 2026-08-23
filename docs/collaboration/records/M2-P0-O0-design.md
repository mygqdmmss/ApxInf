# M2 Protocol, Loader Manifest, and Oracle Design

Date: 2026-08-23
Owner: member2 / protocol-oracle
Prerequisite commit: `58d9fec` (`RuntimeCapabilities`, `AdmissionDecision`, and bounded runtime worker)
Delivery branches: `feat/protocol-stub`, then `feat/oracle-loader`

## Scope

This design covers the member2 CPU-only first batch:

- the evaluator-facing HTTP, JSON, and SSE protocol surface;
- pure protocol admission and error recovery around a model-neutral runtime adapter;
- an independent protocol stub and raw protocol-gate evidence;
- an immutable loader manifest and synthetic W4 directionality fixture;
- a portable oracle job generator, schemas, hashes, and the M2-O0 server handoff.

It does not change the evaluator, scorer, `src/main.rs`, Cargo production entry,
model forward path, CUDA kernels, public answers, or hidden data. It does not
download the checkpoint, expand a full BF16 model, or execute real 8K/16K
oracle workloads. Member1 owns production entry integration, real runtime
capacity checks, GPU1 oracle execution, and GPU0 formal replay.

## Delivery Structure

The work is split into two independently reviewable branches.

1. `feat/protocol-stub` delivers the server protocol core, standalone stub,
   protocol probes, tests, and raw gate evidence.
2. `feat/oracle-loader` delivers loader manifest validation, synthetic W4
   fixture tests, the oracle generator/schema, and the completed handoff
   record. It may be stacked on the protocol branch while both are under
   review, but each commit remains separable.

Both branches declare `58d9fec` as their prerequisite until member1 pushes or
integrates that runtime baseline.

## Protocol Architecture

`src/server/` contains framework-neutral protocol types, validation, response
serialization, SSE event construction, EOS handling, usage accounting, and a
small runtime adapter boundary. HTTP code receives model-neutral capabilities
and an incremental token stream; it never sees CUDA allocators or
device-specific types. The stream exposes `next_token()` and `cancel()`, so a
client disconnect or EOS stop can halt generation before the full budget is
materialized.

The standalone `src/bin/apxinf_protocol_stub.rs` uses existing workspace
dependencies and a minimal standard-library HTTP/1.1 listener, so this branch
does not modify `Cargo.toml`, `Cargo.lock`, or `src/main.rs`. It accepts
`--bind HOST:PORT` (default `127.0.0.1:8001`) and runs until SIGINT/connection
shutdown; the test harness starts it as a child process. Member1 can later
mount the same protocol core under the production HTTP framework and connect it
to the existing bounded runtime worker.

The stub health response is explicitly marked as a fixture while preserving
the frozen contract identity, revision, vocabulary size, maximum model length,
parallel request count, fallback state, and capabilities.

## Admission and Error Semantics

The request parser validates JSON shape and scalar types before conversion to
runtime request types. The accepted request keys are exactly `input_ids`,
`max_new_tokens`, `temperature`, `ignore_eos`, and `stream`; unknown keys are
rejected. It rejects unsupported fields, including the `images` probe, with
HTTP 400 and a JSON `error` object. Raw malformed JSON is also returned as
HTTP 400; JSON error formatting for this one case is useful but is not recorded
as an evaluator hard condition.

Pure protocol admission enforces:

- non-empty integer `input_ids`;
- model vocabulary `[0, 248320)`, accepting `image_token_id=248056` and
  rejecting `4294967295`;
- a positive integer `max_new_tokens`;
- exactly greedy `temperature=0`;
- boolean `ignore_eos` and `stream`;
- `prompt_tokens + max_new_tokens <= max_model_len` using checked arithmetic.

Capacity errors from the runtime adapter remain distinct from malformed
requests. All request-level errors use the shape
`{"error":{"type":"invalid_request","message":"..."}}` for protocol
admission failures and
`{"error":{"type":"capacity","message":"..."}}` for runtime capacity
rejections. They must leave the stub healthy for the next request.

## Generation, SSE, EOS, and Cancellation

Non-streaming success returns HTTP 200 with
`{"type":"result","request_id":"...","output_ids":[...],"usage":{...}}`,
where usage has `prompt_tokens`, `completion_tokens`, and `total_tokens`
computed from the actual returned tokens. Streaming success emits consecutive
token indexes under one request ID, then one done event containing the same
usage object, followed by `data: [DONE]`. The fixture executor returns a
deterministic token (`7`) for a one-token request and never represents this as
model correctness evidence.

The frozen EOS IDs are `248046` and `248044`. With `ignore_eos=false`, the first
EOS token is included once in the generated token sequence and generation then
stops. With `ignore_eos=true`, EOS does not shorten the requested budget.

Every request owns an incremental token stream and cancellation handle. A
socket write failure or client disconnect triggers cancellation and drops the
request guard. Non-streaming requests consume the same token stream into a JSON
result, so EOS and usage behavior cannot diverge between response modes.
Runtime errors are request-scoped; a subsequent health probe and short request
must still succeed. The stub exposes deterministic fake outputs only for
protocol testing and does not claim model correctness.

## Protocol Evidence

A standard-library probe records timestamp, commit SHA, request body, HTTP
status, parsed JSON or raw response, and pass/fail for:

- `malformed_json`;
- `empty_input_ids`;
- `negative_token_id`;
- `out_of_vocabulary_token_id`;
- `unsupported_temperature`;
- `over_budget`;
- `unsupported_modality_field`;
- `valid_short_nostream_request`;
- `health_after_invalid_requests`;
- `health_contract_identity`.

All six structured negative controls explicitly send `stream=false`. The short
request uses eight input tokens, requests one output token, and requires usage
8/1/9. Evidence distinguishes protocol eligibility from the five global
reliability booleans; it never edits or bypasses scorer logic.

## Loader Manifest and Synthetic W4 Fixture

`apxinf-loader` gains immutable manifest types for checkpoint revision, model
configuration, tensor dtype/shape, quantization role, pack axis, and group
size. Manifest parsing reads metadata and headers without materializing a full
BF16 checkpoint.

The required inventory is validated exactly:

| Tensor | Shape | Dtype | Direction |
| --- | --- | --- | --- |
| `k_proj.weight_packed` | `[1024,640]` | I32 | K packed |
| `k_proj.weight_scale` | `[1024,160]` | BF16 | K group-32 |
| `k_proj.weight_zero_point` | `[128,160]` | I32 | N packed |
| `down_proj.weight_packed` | `[5120,2176]` | I32 | K packed |
| `down_proj.weight_scale` | `[5120,544]` | BF16 | K group-32 |
| `down_proj.weight_zero_point` | `[640,544]` | I32 | N packed |

Tests declare these production shapes without allocating their full buffers.
Compact synthetic vectors exercise nibble values 0 and 15, K group boundaries,
non-aligned tail blocks, pack/unpack round trips, and directed failures when N
and K axes are swapped. No real checkpoint slice is copied into Git.

## Oracle Generator and Artifact Identity

`tools/oracle/generate_golden.py` uses the Python standard library for local
manifest/schema generation. It accepts explicit `--model-dir`, `--output-dir`,
`--revision`, `--layers`, and `--stages`, with optional input-manifest and
runner arguments. Local tests use a tiny synthetic model directory and token
manifest. Without `--runner`, the command writes only a job manifest and
schema; it never invents hidden/state/logit values. With `--runner`, the
runner receives the canonical job manifest path and must write declared output
files; missing or extra files fail the command before hashes are recorded.

The generator writes canonical JSON for:

- checkpoint and input identity;
- selected layers and stages;
- reference input/output token IDs and decoded-text fields;
- hidden, recurrent state, KV, and logit artifact schemas;
- generation parameters, EOS policy, dtype/shape metadata, and tolerances;
- expected artifact filenames and SHA256 values.

Manifest-only mode prepares a reproducible job bundle without claiming real
golden outputs. Server execution mode delegates checkpoint-specific inference
to an explicit runner command, validates its output against the same schema,
and hashes every produced artifact. The artifact identity is the generator
SHA, model revision, input-manifest SHA, layer/stage selection, and schema
version. Unchanged identities reuse existing server artifacts. The generator
records its own source SHA when invoked from a Git checkout and fails closed if
the requested revision is empty or the model directory does not exist.

Member1 executes the real job under the global lock on GPU1 and records GPU
UUID, peak VRAM, command, raw artifact path, exported file list, and hashes
using `docs/collaboration/templates/oracle-handoff.md`.

## Testing and Rollback

Implementation follows red-green-refactor cycles. Local acceptance includes
focused Rust tests, Python generator/probe tests, formatting, workspace check,
contract self-check, the live stub protocol gate, and a review that the frozen
evaluation directory is unchanged. The known default-feature failure in the
unrelated `pi05_integrity_probe` example is recorded separately.

Rollback points are the parent of each feature branch and the last green commit
within each branch. Protocol integration is additive: member1 can omit the
server module or disconnect the runtime adapter without altering model code.
Loader/oracle changes are isolated to their owned crate and tools directory.
