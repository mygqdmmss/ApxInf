# 协作进度

更新时间：2026-08-22
集成分支：APXinf-Contest-2026
集成 owner：成员1

状态定义：planned / active / blocked / review / integrated / rejected / done

| Task ID | Owner | Scope | Status | Latest SHA/PR | Evidence / next action |
| --- | --- | --- | --- | --- | --- |
| COLLAB-001 | 成员1 | 创建三人协作 spec、角色手册、Git/GPU 流程和模板 | done | local docs commit (HEAD) | 结构/链接/合同校验通过；`cargo check --workspace --locked` 通过；完整 cargo test 受既有 pi05_integrity_probe 默认 feature 问题阻塞，未修改源码 |
| M1-R0 | 成员1 | 固定环境、模型、合同 hash，clean build baseline | planned | - | 记录 GPU0 UUID、driver/CUDA、test.py check |
| M2-P0 | 成员2 | protocol stub、schema、/health、七项负控和恢复 gate | planned | - | 先在本地 fake runtime 完成逐项原始证据 |
| M2-O0 | 成员2 | checkpoint manifest、oracle、hidden 代理集 | planned | - | 锁定 revision/tokenizer/generation config，输出 manifest |
| M1-R1 | 成员1 | runtime adapter、bounded GPU worker、device-budget admission | planned | - | 等 protocol adapter contract |
| M1-C0 | 成员1 | Qwen35 loader/model/state/GDN/full-attention vertical slice | planned | - | 逐算子/逐层对拍后再接服务 |
| M3-E0 | 成员3 | W4/GEMV/Graph baseline 与 paired benchmark harness | planned | - | 本地静态准备，服务器由成员1在 GPU2 replay |
| M3-B0 | 成员3 | 显存账本、context/C4/C8/MTP/vision bonus evidence | planned | - | 文本 BASE_GOOD 后逐项开启 |
| REL-001 | 成员1+2 | protocol/reliability eligibility campaign | planned | - | 七项 gate、五项 boolean、失败后恢复 |
| REL-002 | 成员1+3 | GPU0 clean checkout final replay | planned | - | 固定 GPU0，warmup 1 + measured 5，CV <= 10% |

## 里程碑

| Milestone | Exit condition | Owner |
| --- | --- | --- |
| R0 reproducible | clean checkout、合同/model hash、GPU identity、test.py check | 成员1 |
| P0 protocol ready | stub 与真实服务逐项通过 protocol gate | 成员2 + 成员1 |
| BASE_CORRECT | public 6/6，hidden 代理集通过，EOS/usage/recovery 正常 | 成员1 + 成员2 |
| BASE_GOOD | request success >= 99%，五项 reliability 全 true，GPU0 可重放 | 成员1 |
| OPT_CANDIDATE | 单变量 paired A/B，端到端净收益，CV 和显存通过 | 成员3 + 成员1 |
| FINAL | clean checkout、最终 report、合同/环境/artifact hash 完整 | 成员1 |

## 更新规则

成员2/3在自己的 PR 或 progress log 写日常细节；成员1合并后把源 SHA、集成 SHA、服务器 replay 和状态回填到本表。阻塞超过一个工作周期必须新增 incident 或 decision 记录。
