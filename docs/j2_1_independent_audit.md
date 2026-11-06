# J2.1 独立审计报告（综合）

日期：2026-11-05 | 触发：外部任务书指认 J2 轮两个协议缺陷
（身份负样本误标 + 候选时间共享泄漏）
自测：J2.1 回归 **12/12** + legacy 套件（35+20）全绿
原始产物：`j2_ablation.json` / `j2_association.json` /
checkpoints **只读保留**；新结果全部在 `runs/j2_1_reeval/`

## 1. 审计发现与处置

| # | 指认 | 核实 | 处置 | 影响 |
|---|---|---|---|---|
| D1 | 非对角线当负样本（同 actor 多序列误标） | **属实**（val 42/3540 对 = 1.2%） | (source,actor) mask + hard/easy 分解 + 回归 | 数值 <0.002；**负结果维持** |
| D2 | 竞选集共享真主 delta（身份泄漏） | **属实**（泄漏路径存在） | 每候选独立 delta + 无效→unknown + 回归 | 语义修复完成；本床数值不变（候选时间天然聚集——测试床局限记录） |
| D3 | 数据规模与独立性数字与报告不符 | **属实**（我口头报 73/18/53 源错误） | 机器复核真实数字（53/15/8/2）+ 勘误 | final_test 2 源不具统计力——声明强化 |
| D4 | 模块耦合（J2 证据需可开关） | 架构缺口 | `latent_world_state` 六文件模块 + off-by-default + 一致性回归 | 零侵入接入就绪 |

## 2. 修正后的最终证据图景（统一协议，旧 checkpoint 只读）

| 层 | J1 静态 | J2-C GRU | J2-D |
|---|---|---|---|
| 区分间距（val，身份 mask 修正） | **0.256** | 0.111 | 0.108 |
| hard 负样本 cos（同源竞争者） | 0.557 | **0.686**（更弱） | 0.687 |
| 竞争集融合（A3 vs A1 基线） | — | 无增量（0.119→0.167） | 未运行 |

**核心结论**：J2 v1 负结果**在两项协议修复后维持成立**；
首要嫌疑为训练量（160 独立 actor）与目标参数化（直接预测→
共性化解），v2 残差设计针对后者（见 j2_v2_training_design）。

## 3. 新增资产

- `src/wmeval/latent_world_state/`（六文件模块：encoder adapter /
  actor tokenizer / temporal predictor / association evidence /
  event-state 接口 / interfaces 开关）——12/12 单元测试；
- `src/run_j2_1_reevaluation.py`（统一协议重评器）；
- `runs/j2_1_reeval/{pair_reeval, competition_reeval}.json`；
- `src/selftest_j2_1.py`（Gate 0/1 回归：重复 actor / 跨视频同 ID /
  hard-easy / 每候选时间 / 无效间隔 / tokenizer 契约）。

## 4. 决策

**CONDITIONAL-GO**（详见 j2_1_go_no_go.md §3）——GO：J2 v2-B
残差设计在全量数据上训练（条件：监控三件套+过拟合梯）；
NO-GO：C/D 消融、V-JEPA 2.1、事件级实验（各自前置未满足）。
