# M2-O0 Oracle Handoff Record

```text
TASK_ID: M2-O0
GENERATOR_SHA: commit a52f7ef79546f460b17c932cc658dc01215703cb; source SHA256 41254c3e9c1b284ff8e8b1bfdf9e2b0f25733c5bf95c716995a141f887cad1bb
GENERATOR_ENTRYPOINT: tools/oracle/generate_golden.py
MODEL_REPO: cyankiwi/Qwen3.8-27B-AWQ-INT4
MODEL_REVISION: 63768c10df38c0395e12ef49edac1bd539eaeeea
CONTRACT_SHA256: 520349b1279c3bf999a6848b296c23d20cdaeab7420934e9196c90018bac7433
INPUT_MANIFEST_SHA256: d9a6eb390e7d4da20d6296a27e4048071e0b48296c84d49c9dce81c1452fbcc4
LAYER_OR_STAGE_SELECTION: layers [0,3,31,32,60,63]; stages [embedding,gdn_state,kv_state,layer_hidden,logits,tokens]; max_new_tokens 128
SYNTHETIC_FIXTURE_SHA256: crates/apxinf-loader/src/w4.rs source SHA256 ff07de2815400baa9c3e6ba267de807bffb979b2ec6da608a8a6ae1a6a94b5f6
EXPECTED_ARTIFACT_SCHEMA: apxinf.oracle-golden-schema.v1 plus artifact-report generation {completion_tokens,stop_reason}; local metadata-only schema SHA256 4225aa76119262c87a75af6ac5626a21e56a572a26ed52ec4645c9e79fcae35f
APPROVED_EXPORT_METHOD: server-only; any minimal golden export requires member1 approval and project artifact channel registration
APPROVED_EXPORT_FILES: control manifests/schema/hash; real golden export list pending member1 server execution and approval
APPROVED_EXPORT_SHA256: input-manifest d9a6eb390e7d4da20d6296a27e4048071e0b48296c84d49c9dce81c1452fbcc4; selection a6ee71934a7881e8168322e496c9cb3456bb14e92032db220fc89ad61ff87dff; generation 89bae0d244a3bf4a5206ca6ba6553096f5371b96353233d43196a2834baef543; golden-schema 4225aa76119262c87a75af6ac5626a21e56a572a26ed52ec4645c9e79fcae35f; artifact-manifest 775afab258c4b77d6ee5b4e3070384b8fb8d6046e8d7eb114221fae547fcc4e7; job-manifest 6a6284230a42a299b1a4a7f9cb37aa71b0ef0425b7ee120b8e709cfc633ae9b4; artifact-identity 24dfbc8e0d741906b67c19055fd82d4e2d6229b55434bb542d18622730638018; pending real artifact hashes
SERVER_QUEUE_PRIORITY: P0-oracle
TARGET_GPU_UUID: GPU-343bc895-b011-22fa-4449-97207aa2bdec
REPLAY_COMMAND: exact command block below; checkpoint runner executable is a member1 integration prerequisite and the command fails closed if absent
EXPECTED_ARTIFACT_DIR: /mnt/chuangxin/team2/artifacts/apxinf/oracle/63768c10df38c0395e12ef49edac1bd539eaeeea/a52f7ef79546f460b17c932cc658dc01215703cb
EXPECTED_MAX_VRAM_MIB: pending member1 GPU1 measurement; no local estimate is asserted as evidence
STOP_CONDITION: any lock/UUID/revision/config/EOS mismatch, runner absence/nonzero exit, missing/extra/symlink artifact, control-manifest mutation, schema/dtype/shape/hash/token failure, OOM, Xid, NaN, or fallback
ROLLBACK_OR_DISABLE_COMMAND: omit --runner to remain manifest-only; discard the incomplete artifact directory; code rollback order is a52f7ef, ae2fb4e, a5cf465, 7e26b94, then 8e5c1cc
```

## Exact GPU1 replay command

Member1 first installs or integrates the checkpoint-specific executable at the
path below. M2-O0 intentionally does not provide that real model runner.

```bash
cd /mnt/chuangxin/team2/ApxInf
exec 9>/tmp/apxinf-gpu-job.lock
flock -n 9 || { echo "another ApxInf GPU job is running" >&2; exit 2; }
export CUDA_VISIBLE_DEVICES=GPU-343bc895-b011-22fa-4449-97207aa2bdec
export APXINF_GPU_LABEL=GPU1
test -x /mnt/chuangxin/team2/ApxInf/tools/oracle/qwen35_checkpoint_runner || {
  echo "member1 checkpoint runner is not installed" >&2
  exit 3
}
python3 tools/oracle/generate_golden.py \
  --model-dir /mnt/chuangxin/team2/models/Qwen3.8-27B-AWQ-INT4 \
  --output-dir /mnt/chuangxin/team2/artifacts/apxinf/oracle/63768c10df38c0395e12ef49edac1bd539eaeeea/a52f7ef79546f460b17c932cc658dc01215703cb \
  --revision 63768c10df38c0395e12ef49edac1bd539eaeeea \
  --input-manifest tools/oracle/manifests/m2-o0-short.json \
  --layers 0,3,31,32,60,63 \
  --stages embedding,layer_hidden,gdn_state,kv_state,logits,tokens \
  --max-new-tokens 128 \
  --runner /mnt/chuangxin/team2/ApxInf/tools/oracle/qwen35_checkpoint_runner
```

Layer roles in this selection are early/middle/late representatives:

- GDN / linear attention: 0, 32, 60;
- full attention: 3, 31, 63.

The generator uses the checkpoint's 64-entry `layer_types` map and will emit
GDN state only for the first set and KV state only for the second set.

## Expected declared artifact files

```text
embedding.f32.bin
layer-000-gdn-state.f32.bin
layer-000-hidden.f32.bin
layer-003-hidden.f32.bin
layer-003-kv-key.f32.bin
layer-003-kv-value.f32.bin
layer-031-hidden.f32.bin
layer-031-kv-key.f32.bin
layer-031-kv-value.f32.bin
layer-032-gdn-state.f32.bin
layer-032-hidden.f32.bin
layer-060-gdn-state.f32.bin
layer-060-hidden.f32.bin
layer-063-hidden.f32.bin
layer-063-kv-key.f32.bin
layer-063-kv-value.f32.bin
logits.f32.bin
output-tokens.json
artifact-report.json
```

The metadata-only local bundle had artifact identity
`24dfbc8e0d741906b67c19055fd82d4e2d6229b55434bb542d18622730638018`,
config SHA256
`fece2915d4c8ad4c10877622f04ea5e01cd3ae38768ce5c1edb700dd1de290f6`,
and generation-config SHA256
`e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e`.
The control-file SHA256 values are recorded in `APPROVED_EXPORT_SHA256` above;
they are metadata-only identities, not real checkpoint artifact results.
Member1 must compare these identities before running. A mismatch stops the job;
it is not silently accepted as a new oracle.

## Member1 append-only server results

The following fields are intentionally pending real execution:

```text
QUEUE_ID: pending member1
STARTED_AT_UTC: pending member1
ENDED_AT_UTC: pending member1
ACTUAL_GPU_UUID: pending member1
PEAK_VRAM_MIB: pending member1
COMMAND_OUTPUT_PATH: pending member1
RAW_ARTIFACT_FILE_LIST: pending member1
RAW_ARTIFACT_SHA256: pending member1
APPROVED_MINIMAL_EXPORT: pending member1 approval
INCIDENT_OR_SUCCESS_STATUS: pending member1
```

No local result in this record is a real checkpoint correctness, GPU memory,
hidden-case, reliability, or eligibility claim. The five reliability booleans
remain global scorer gates and are not modified by this handoff.
