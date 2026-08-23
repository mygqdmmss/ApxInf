# M2-O0 Oracle Handoff Record

```text
TASK_ID: M2-O0
GENERATOR_SHA: commit 7e26b944c06e4953480b445b3746e854aa32d527; source SHA256 74b9f75aee8b88009aa928dedf6632ef02d963f50de1397809e61b6dd017060e
GENERATOR_ENTRYPOINT: tools/oracle/generate_golden.py
MODEL_REPO: cyankiwi/Qwen3.8-27B-AWQ-INT4
MODEL_REVISION: 63768c10df38c0395e12ef49edac1bd539eaeeea
CONTRACT_SHA256: 520349b1279c3bf999a6848b296c23d20cdaeab7420934e9196c90018bac7433
INPUT_MANIFEST_SHA256: d9a6eb390e7d4da20d6296a27e4048071e0b48296c84d49c9dce81c1452fbcc4
LAYER_OR_STAGE_SELECTION: layers [0,3,31,32,60,63]; stages [embedding,gdn_state,kv_state,layer_hidden,logits,tokens]; max_new_tokens 128
SYNTHETIC_FIXTURE_SHA256: 6a390cb8604c90b9daee0da6e3b81d67c134ea5d40e3b8b96d50bcfdf6c27992
EXPECTED_ARTIFACT_SCHEMA: apxinf.oracle-golden-schema.v1; local metadata-only schema SHA256 3dde4369d3820ffe6fe1905d4d3c69cc11610c43ad2a2b5a4f7e8418c09f8548
APPROVED_EXPORT_METHOD: server-only; any minimal golden export requires member1 approval and project artifact channel registration
APPROVED_EXPORT_FILES: control manifests/schema/hash; real golden export list pending member1 server execution and approval
APPROVED_EXPORT_SHA256: input d9a6eb390e7d4da20d6296a27e4048071e0b48296c84d49c9dce81c1452fbcc4; selection a6ee71934a7881e8168322e496c9cb3456bb14e92032db220fc89ad61ff87dff; generation 89bae0d244a3bf4a5206ca6ba6553096f5371b96353233d43196a2834baef543; golden-schema 3dde4369d3820ffe6fe1905d4d3c69cc11610c43ad2a2b5a4f7e8418c09f8548; pending real artifact hashes
SERVER_QUEUE_PRIORITY: P0-oracle
TARGET_GPU_UUID: GPU-343bc895-b011-22fa-4449-97207aa2bdec
REPLAY_COMMAND: exact command block below; checkpoint runner executable is a member1 integration prerequisite and the command fails closed if absent
EXPECTED_ARTIFACT_DIR: /mnt/chuangxin/team2/artifacts/apxinf/oracle/63768c10df38c0395e12ef49edac1bd539eaeeea/7e26b944c06e4953480b445b3746e854aa32d527
EXPECTED_MAX_VRAM_MIB: pending member1 GPU1 measurement; no local estimate is asserted as evidence
STOP_CONDITION: any lock/UUID/revision/config/EOS mismatch, runner absence/nonzero exit, missing/extra/symlink artifact, control-manifest mutation, schema/dtype/shape/hash/token failure, OOM, Xid, NaN, or fallback
ROLLBACK_OR_DISABLE_COMMAND: omit --runner to remain manifest-only; discard the incomplete artifact directory; code rollback is git revert a5cf465 7e26b944c06e4953480b445b3746e854aa32d527 8e5c1cc496c96a2423caa82a99128bb278e8aa65
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
  --output-dir /mnt/chuangxin/team2/artifacts/apxinf/oracle/63768c10df38c0395e12ef49edac1bd539eaeeea/7e26b944c06e4953480b445b3746e854aa32d527 \
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
`50d97941d3c9700be06b3189d85e900734d8f21f5d893c17995c7b7e35c2d939`,
config SHA256
`fece2915d4c8ad4c10877622f04ea5e01cd3ae38768ce5c1edb700dd1de290f6`,
and generation-config SHA256
`e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e`.
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
