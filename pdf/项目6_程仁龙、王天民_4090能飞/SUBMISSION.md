# 项目6 · Agent for Kernel — 提交说明

- **作品名称**：4090 能飞 —— Qwen3.8-27B W4A16 单卡 RTX 4090 推理引擎
- **团队**：程仁龙（负责人，约 95%）、王天民（约 5%，离线 benchmark 脚手架）
- **仓库**：https://github.com/mygqdmmss/ApxInf （fork 自 infinigence/ApxInf）
- **分支**：`integrate/member2`
- **验收 commit SHA（实现提交，验收以此为准）**：
  `06993a2d2642c6f7177b57493b797d5d537e4d64`
  （"feat(qwen35): add C4 concurrent batched serving behind a single switch"）
- 分支 HEAD：`d9b6626`（实现提交之上的报告提交，仅改 REPORT.md，不改任何代码）

## 成果摘要

- 文本 eligibility 全门禁通过：协议 gate 12/12、公开功能 6/6 精确、200 请求混合
  soak 100%、proxy hidden 11/12（自建代理集，已标注非官方）、7/7 性能 cell 有效。
- 性能：TTFT 最高约 48 倍（16K：1635 s → 34.6 s），TPOT 约 2 倍（133.7 → 66.5 ms）；
  17 次单变量配对实验，9 接受 / 8 拒绝，全部有数据与根因记录。
- 多模态 bonus 已交付：vision 塔 CUDA 移植 + `/v1/chat/completions`，预处理与 HF
  逐位一致，自建探针 4/4，文本数值零变化，单开关启用/回滚。
- C4 多请求 bonus 已交付：批式 protocol runtime，官方 evaluator 校准 run 全部
  效度门通过（成功率/正确率 1.0、Jain 0.9993、p95 TTFT/TPOT 门槛内、goodput
  23.53 tok/s），单请求路径零回归，单开关启用/回滚。

## 评审要求六项材料对照

| 评审要求 | 所在位置 |
|---|---|
| 1. 设计变化及影响的执行阶段 | `REPORT.md` → "Design Changes and Affected Execution Stages"（11 项变更表）；两份计划文档（`APXINF_FINAL_EXECUTION_PLAN_2026-08-22.md`、`APXINF_QWEN38_TECHNICAL_PLANS.md`）保留原始规划并附「实际执行结果」订正，完整记录规划与执行的偏差 |
| 2. `test.py check` 与 `test.py run` 的命令和结果 | `REPORT.md` → "`test.py check` and `test.py run`: Commands and Results"（check 通过；run 因平台参照件缺失无法产出评分件，等价负载已直测并完成一次披露的管道完整性校准 run） |
| 3. 至少一个负控制或回归测试 | `REPORT.md` → "Negative Controls and Regression Tests"（7 项协议负控、3 项断连故障注入回归、9 组数值/位相等回归，均在 `cargo test` 套件内） |
| 4. correctness / 性能 / 稳定性 / 显存之间的取舍 | `REPORT.md` → "Trade-offs: Correctness, Performance, Stability and VRAM" |
| 5. 已知限制、失败实验和回滚方法 | `REPORT.md` → "Known Limitations, Failed Experiments and Rollback"（5 项限制、8 项被拒实验索引、逐项开关与整树回滚点） |
| 6. REPORT.md：baseline、假设、实现、测量、结果和复现步骤 | `REPORT.md` 全文（Required Submission Materials → Engineering Record → Appendix 三部分结构），复现命令见 "Reproduction Steps" |

## 材料清单

| 文件 | 说明 |
|---|---|
| `REPORT.md` | 技术报告（提交件核心，来自验收分支 HEAD） |
| `APXINF_FINAL_EXECUTION_PLAN_2026-08-22.md` | 执行计划（原始规划 + 实际执行订正） |
| `APXINF_QWEN38_TECHNICAL_PLANS.md` | 三套技术方案比较（原始规划 + 落地对照） |
| `项目6_程仁龙、王天民_4090能飞_v1.pptx` | 评审答辩 PPT（15 页） |
| `evidence/` | 测量证据档案（REPORT 中以 `apxinf-evidence/...` 引用的原始产物：协议 gate、A/B、soak、性能 cell、C4 回归等，约 1.9 MB） |
| `SUBMISSION.md` | 本说明 |

## 合规声明

- 未修改 `benchmarks/qwen38_4090/evaluation/` 下任何合同、生成器、scorer 文件；
  `submission.json` 图片字段未手工填写。
- 无按 case ID、公开 token 序列或已知答案的硬编码输出。
- 提交材料不含模型权重、凭据、机器地址或未公开评测数据
  （evidence 已扫描：仅含回环地址 127.0.0.1，无凭据特征）。
- 无平台批准的 scorer 参照件，因此不声称 eligible、轨迹得分或任何评分结论；
  官方 hidden 集与图片套件不在开发机，相关自测均明确标注非官方。

## 快速复现

```
cargo build --release --features cuda-no-nvtx --locked --bin apxinf
target/release/apxinf serve --model <model-dir> \
  --revision 63768c10df38c0395e12ef49edac1bd539eaeeea \
  --gpu-uuid <gpu-uuid> --bind 127.0.0.1:18080 \
  --max-model-len 32768 --queue-capacity 1
# 多模态：前缀 APXINF_ENABLE_MULTIMODAL=1
# C4 并发：前缀 APXINF_Q35_MAX_CONCURRENCY=4 且 --queue-capacity 4
python3 benchmarks/qwen38_4090/evaluation/test.py check   # → assignment checks passed
```

详细步骤与门禁预期见 `REPORT.md` → "Reproduction Steps"。
