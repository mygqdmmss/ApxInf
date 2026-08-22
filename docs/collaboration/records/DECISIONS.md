# 协作决策记录

| ID | Date | Decision | Reason | Owner | Status |
| --- | --- | --- | --- | --- | --- |
| D-001 | 2026-08-22 | 成员1是唯一集成 owner，维护 APXinf-Contest-2026 并负责 GPU0 正式验证 | 服务器只有一个账号，避免主工作树和 GPU 任务冲突 | 成员1 | accepted |
| D-002 | 2026-08-22 | 成员2、3在自己的电脑开发，通过 GitHub feature/experiment 分支和 PR 交付 | 不要求三人同时登录服务器；本地可完成协议、oracle、脚本和静态验证 | 成员1 | accepted |
| D-003 | 2026-08-22 | 成熟算子混合是唯一主线；SM89 GEMV/Graph、paged KV、C4/C8、MTP、vision 为隔离候选 | 先拿文本 eligibility，再按证据逐项吸收优化 | 成员1 | accepted |
| D-004 | 2026-08-22 | GPU0 是正式成绩来源，GPU1-3只做 replay/development evidence，服务器任务串行 | 满足固定单卡合同并降低单账号并发风险 | 成员1 | accepted |
| D-005 | 2026-08-22 | max_model_len 是 prompt + output 的总 admission 上限 | evaluator 的 over_budget probe 将 max_new_tokens 设为 health.max_model_len | 成员1 + 成员2 | accepted |
| D-006 | 2026-08-22 | 七项 protocol probe 是 eligibility gate；五项 reliability boolean 任一失败即 eligible=false | evaluator/scorer 的实际逻辑，不是 PR review 可选加分项 | 成员2 + 成员1 | accepted |
| D-007 | 2026-08-22 | evaluator、scorer、contract、公开数据生成器只读 | 保持比赛合同和提交身份完整 | 全员 | accepted |
| D-008 | 2026-08-22 | prefix cache、mega-kernel、262K INT4 KV 默认不进入交付 | 风险高且不能替代基础资格；只有新鲜证据才能重新决策 | 成员1 | accepted |
| D-009 | 2026-08-22 | `integrated` 是分层 PR 合入状态，`done/release` 才表示最终 eligibility/release gate 通过 | 协议 stub、loader 和实验 PR 必须能在真实 runtime 完成前独立合入 | 成员1 | accepted |
| D-010 | 2026-08-22 | loader 解析/API 由成员2拥有，生产接线和 device-budget admission 由成员1拥有；协议 admission 与 runtime capacity admission 通过稳定接口交接 | 避免同一 server 入口和 loader 文件发生 owner 冲突 | 成员1 + 成员2 | accepted |
| D-011 | 2026-08-22 | C4/C8 同时记录合同 1.5x TTFT guard 和当前 scorer 的 concurrency-scaled guard，候选采用更严格的 1.5x team policy | 冻结 contract 与 scorer 实现存在差异，不能修改 scorer 或混淆实验接收标准 | 成员1 + 成员3 | accepted |
| D-012 | 2026-08-22 | 三名 agent 的启动 prompt 通过聊天线下发送，不提交仓库；本地开发环境和协作边界入库；成员2/3不并发写聚合 PROGRESS | prompt 含启动上下文但不应成为生产源码/合同的一部分；共享账号下并发修改聚合记录会制造冲突 | 成员1 | accepted |

## 新决策模板

新增决策时追加一行，必须包含：ID、日期、具体行为、证据/理由、owner、accepted/rejected/pending。
