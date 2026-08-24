# Qwen3.5 Bounded Prefill Report

Status: hardened implementation, not release-ready. The final branch tip is
authoritative via `git rev-parse HEAD`.

## Identity

- Model: `/mnt/chuangxin/team2/models/Qwen3.8-27B-AWQ-INT4`
- Checkpoint revision: `63768c10df38c0395e12ef49edac1bd539eaeeea`
- Development/replay GPU: `GPU-343bc895-b011-22fa-4449-97207aa2bdec`
- Service command: `target/debug/apxinf serve --model /mnt/chuangxin/team2/models/Qwen3.8-27B-AWQ-INT4 --revision 63768c10df38c0395e12ef49edac1bd539eaeeea --gpu-uuid GPU-343bc895-b011-22fa-4449-97207aa2bdec --bind 127.0.0.1:18080 --max-model-len 32768 --queue-capacity 1`
- CUDA toolkit: 12.8.93; driver: 580.82.07; Rust: 1.98.0
- Contract SHA256: `520349b1279c3bf999a6848b296c23d20cdaeab7420934e9196c90018bac7433`
- Model `config.json` SHA256: `fece2915d4c8ad4c10877622f04ea5e01cd3ae38768ce5c1edb700dd1de290f6`
- Model safetensors index SHA256: `82b1bf79f5b61333e83da17ec3bf89c9f178e29395a14c6b3ce3bbc474e1ead8`
- Approved safetensor shard SHA256s:
  `54d83c1d36631de231876217a8e0c2483eccee8746369a482b79442bdfc5d958`,
  `64be5fc2f66a3e5679ba229261a7a0d8112b06f6f560c750a62ca9457f90006c`,
  `7b90d6c7059d615a560cd4d2e766d328210605041061681550d80f380a8b529b`,
  `03b2624ec788780a2915003cd2871c29c87dfb6f2a8d189ef3918662d6a1ed56`,
  `eb5ea1fbef28b13ac89158924ee7cfe7c9f111c79ae177b290c0abd45c38925c`.

## Implementation

The Qwen3.5 runtime executes prompt prefill in contiguous blocks of at most 64
tokens. Every block runs all 64 layers before the next block, carries GDN
convolution/recurrent state and full-attention KV state across boundaries, uses
absolute positions, allocates request KV capacity as `prompt_len + max_new_tokens`,
and retains only the final block's last row for logits. Readiness now performs a
prefill-plus-decode warmup before binding, fails closed while unhealthy, and
serially attempts recovery from `/health`.

## Verification

- `cargo test --bin apxinf -- --nocapture --test-threads=1`: 52 passed, 0 failed.
- `cargo test -p apxinf-model --locked qwen35 -- --nocapture`: 54 passed, 0 failed, 2 ignored.
- `CUDA_VISIBLE_DEVICES=GPU-343bc895-b011-22fa-4449-97207aa2bdec APXINF_TEST_GPU_UUID=GPU-343bc895-b011-22fa-4449-97207aa2bdec cargo test -p apxinf-cuda context::tests::attested_context_accepts_expected_uuid_and_rejects_mismatch`: 1 passed, 0 failed.
- Designated-GPU GDN suite: 17 passed, 0 failed.
- Python protocol/oracle tests: 29 passed, 0 failed.
- `cargo check --workspace --locked`: passed.
- `cargo build --features cuda-no-nvtx --locked --bin apxinf`: passed.
- `cargo fmt --all -- --check` and `git diff --check`: passed.

The pinned checkpoint test used a 65-token prompt and passed in 916.56 seconds.
It verified two bounded prefill blocks, final position/KV length 65, request KV
capacity 67, and one prefill plus one decode token. Artifact SHA256:
`9a2c325f2254fd681b6de4c8e2b4d2b755aebfca0aba584ce2cb63bb1bb683b0`.

## Fresh Service Evidence

Artifact directory:
`/mnt/chuangxin/team2/artifacts/apxinf/midterm/20260825T164057Z-readiness-final`

- `/health`: HTTP 200, `stub=false`, frozen revision and contract identity,
  `max_model_len=32768`, `parallel_requests=1`, `fallback_active=false`.
- Frozen protocol gates: 10/10 passed. `protocol.json` SHA256:
  `562dec14609fa7508ea6194361c45cfc4d9ef81258ff96b48e605218da52ce2c`.
- SSE request: HTTP 200, token prefix `[2037, 9]`, valid done usage and `[DONE]`.
- Capacity rejection: prompt 1 plus `max_new_tokens=32768` returned structured HTTP
  400; `/health` remained healthy and the next short request returned HTTP 200.
- 65-token non-stream request: HTTP 200, output `[1]`, prompt usage 65 and total 66.
- Client disconnect during the optional evaluator attempt was followed by healthy
  `/health` and a successful short recovery request.
- No Xid lines were observed in the captured kernel log window.
- On shutdown, PID 1317001 exited, port 18080 was free, all GPUs returned to 1 MiB,
  and `/tmp/apxinf-gpu-job.lock` was available.
- Artifact manifest SHA256: `a4bbba80042bb282285b4d627c4b7372b194425fc9c8440177aba3533c66758a`.

An additional hardening startup gate used the same strict service command. The
port stayed unbound during full-checkpoint digest verification and model warmup;
startup then independently reported CUDA UUID
`GPU-343bc895-b011-22fa-4449-97207aa2bdec`. `/health` returned HTTP 200 and an
8-token non-stream request returned HTTP 200 with output `[2037]` and usage 8/1/9.
After interrupting PID 1353165, the port was free, the target GPU returned to 1 MiB,
and the global GPU-job lock was available.

## Evaluator Scope

The approved public dataset was available at
`/mnt/chuangxin/team2/ApxInf/benchmarks/qwen38_4090/evaluation/.cache/public`.
Its manifest SHA256 is
`1ec4f360e8dce8cb366251d9b92f8f91a393e5534bb93277a955f8b9e3e5e1e4`.
The official evaluator was started against the six public functional cases, but
the first 8K-class request produced no completed row after more than 16 minutes
and was interrupted for cleanup. No functional score is claimed. Hidden evaluation
was unavailable. The required approved `--trajectory-reference` was unavailable;
the candidate was not used as its own oracle, and no trajectory score is reported.
No evaluator or scorer files were modified.

## Known Release Blockers

1. `request_state_bytes` is a conservative estimate, not allocator instrumentation
   with a measured peak-memory margin.
2. KV append rollback on a failed request is achieved by dropping the failed
   session; an in-place transaction rollback is not implemented.
3. No successful 1024/8192/32768-token service request was completed in this run;
   the 65-token cross-block request is the long-prompt evidence. The evaluator's
   interrupted 8K attempt is not a pass.

The strict production path now independently attests the selected CUDA device UUID,
requires the frozen 64-layer Qwen3.5 contract, and streams SHA-256 over the approved
config, index, and five safetensor payloads before model admission. Runtime errors
map `WorkerStopped` to HTTP 503/unavailable, and recovery continues to serialize
through a poisoned mutex. These are no longer release blockers.

Rollback point: `47ec280d2f88e8daf87750c0957e596e3a5390c1` (pre-integration HEAD).
