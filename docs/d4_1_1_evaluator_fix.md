# D4.1.1 评价器修复报告（Evaluator Fix）

日期：2026-11-06 | 范围：仅评价器与统计代码（不训练、不修改 Checkpoint、不重新提取视觉特征）

## 1. 上一会话的失效模式（根因）

| # | 失效模式 | 证据 |
|---|---|---|
| 1 | **统计单位错误**：内联脚本按 `source` 聚合整个视频的候选行并在视频级排序，一个 Anchor 不再对应一个决策 | video_0310 在 val 中恰有 **325 条候选行 / 23 个 Anchor**，旧会话输出"325 个正确关联"——325 是候选行数，不是决策数 |
| 2 | **无统计不变量**：正确数可以超过 Anchor 数并照常发布 | 同上（325 > 23）未被任何检查拦截 |
| 3 | **Coverage 用候选行统计**，LOSO 只算 D4 自身 margin 波动 | 旧 `d4_1_fixed_coverage_risk.md`、`d4_1_source_level_loso.md`（M_hard 系列） |
| 4 | **文档携带过期数值**：Kin+D4 wrong_50 记为 8/30，而独立复核与旧会话后期自产的 `seven_methods.json` 均为 **9/30** | 11:55–12:01 的 docs 未随 12:38 的重算更新 |
| 5 | **发布先于校验**：结果文件在参考数值检查判定之前写盘 | 旧内联流程无发布门 |
| 6 | **运行方式**：反复复制的 heredoc 脚本执行约 25 次，无版本管理，最终触发 API 输入超限 | 任务书 P0/P1 所述 |

## 2. 修复实现

新增 `src/run_d4_1_1_fixed_evaluator.py`（纯标准库，只读冻结的
`runs/d4_1_reevaluation/audit_candidates.csv`，不加载任何模型）：

### 2.1 Anchor 分组与 Top-1

- 分组键 `(split, source, anchor_id)`；**每个 Anchor 对每个方法恰好产生一个 Top-1 决策**，候选绝不跨 Anchor 混合。
- 七方法统一评分（同一候选集合，分数越大越可信）：

| 方法 | 公式 |
|---|---|
| kin | `-kin` |
| static | `static_cos` |
| kin_static | `static_cos - kin` |
| b4 | `b4_cos` |
| kin_b4 | `b4_cos - kin` |
| d4 | `d4_cos` |
| kin_d4 | `d4_cos - kin` |

- 平局用**内容键**（数值感知的 ped 升序）打破：与 CSV 行顺序无关。
- 非有限分数**不参与排序**（有效分记为 -inf，永不赢过有限分），但候选保留在所有方法的集合中（不静默跳过）。
- 重复候选键 `(split, source, anchor_id, ped)`：值冲突 → 硬错误拒绝评价；完全相同 → 折叠为一行并记录（不虚增正确数）。

### 2.2 统计不变量（I1–I8，发布前强制）

| 编号 | 不变量 |
|---|---|
| I1 | 一个 Anchor 一个 Top-1 决策（决策键 == 分组键，7 方法逐一核对） |
| I2 | 每源每方法 correct ≤ n_anchors（违反抛 InvariantViolation） |
| I3 | 每划分每方法 correct ≤ 该划分 Anchor 数（val ≤ 60） |
| I4 | **行顺序不变性**：倒序重跑全流程，7 方法全部决策逐位一致 |
| I5 | 无冲突重复候选键 |
| I6 | 非有限分数从未在有有限分可选时胜出 |
| I7 | 每个决策用独立 max 校验复核为该方法自己评分的 argmax（防止七方法共用一份结果） |
| I8 | 各方法每个 Anchor 的候选集合一致（n_candidates 相同） |

失败时：只保存失败原因与诊断（`invariant_checks.json`、运行日志、manifest 记
`invariant_violation`），退出码 2，**不发布任何正式结果**。

### 2.3 固定覆盖率（Coverage–Risk）

- 单位是 **Anchor 决策**：先 60 个 Anchor → 60 个 Top-1，再跨 Anchor 按该方法自己的 Top-1 分数（置信度）降序排序。
- `accepted = round-half-up(coverage × n_anchors)`；**50% × 60 = 30**，分母永远是 Anchor 数。
- 必需字段齐全：`accepted_correct`、`accepted_wrong`、`wrong_among_accepted`、
  `coverage`、`top1_accuracy`（另含分数、边界平局标记、0.1–1.0 曲线）。

### 2.4 源级统计与 LOSO

- 先算每源 `n_anchors` 与七方法 correct（`per_source_results.csv`）。
- LOSO：**排除被留出源的全部 Anchor**，对剩余 Anchor 重新汇总每方法正确率；
  报告每方法 `delta_vs_kin / delta_vs_kin_static / delta_vs_kin_b4` 及汇总
  （均值/最小/最大/正负折数）。文件中显式注明：LOSO 是同划分内的稳定性
  检查，**不是独立跨域测试**。

### 2.5 参考数值回归断言（只断言，不输出）

外部独立复核值（val 588 行 / 60 Anchor / 15 源；video_0310 23 Anchor、
kin 12 / kin_static 15 / kin_b4 15 / kin_d4 14；Top-1 kin 39、kin_static 39、
kin_b4 38、kin_d4 42；Coverage@50% accepted=30、wrong：kin 8、kin_static 7、
kin_d4 9）**只用于运行后断言**（`REFERENCE_CHECKS`），从不写入评价输出。
不一致 → 退出码 3，写出 `reference_mismatch_details.json`（具体 Anchor、
候选、分数），不发布正式结果。

### 2.6 发布门与运行身份（run_id）

- 流程固定为：计算 → 不变量 → 参考断言 → **全部通过才发布**。
- 正式文件先写入 `.staging/` 临时目录，逐个 `os.replace` 原子落位；运行日志与
  `run_manifest.json` 最后写——失败或部分发布的运行永远不会被标记
  `completed`。
- `run_id = sha256(code_sha256, input_sha256, config_sha256)`（代码版本 +
  冻结 CSV 哈希 + 方法与参数配置）。
- manifest 完整记录：代码版本、输入哈希、配置、参考检查状态、开始/结束
  时间、最终状态、退出码。
- 单次执行保护：同 `run_id` 在同输出目录重跑 → 拒绝（退出码 4）；换输入或
  换配置 → 明确识别为**不同实验**并拒绝混入同一输出目录；针对冻结输入另设
  跨目录注册表 `.formal_run_registry.json`，防止换 `--out-dir` 绕过。
- `--force` 只允许重试**未完成**的运行；覆盖已完成的正式结果必须显式确认
  `--force-run-id <run_id>`（且旧正式文件自动归档到 `superseded_*/`，不与
  新结果混淆）。

## 3. 自测（`src/selftest_d4_1_1_evaluator.py`，先测试后评价）

| 测试 | 断言 |
|---|---|
| T1 | 同源两 Anchor → 两个独立 Top-1（混合排序会失败） |
| T2 | 不同源相同 anchor_id → 分组不混合 |
| T3 | 评分方向：kin 小者优、cos 大者优、cos−kin 大者优（构造数据上联合与纯 cos 翻转相反） |
| T4 | 冲突重复键抛错；相同重复折叠、不虚增正确数 |
| T5 | 源级 correct ≤ n_anchors；伪造违反抛 InvariantViolation |
| T6 | 60 Anchor × 4 候选（240 行）@0.5 → **恰好 30 个不同 Anchor 被接受**（行级实现会接受 120），wrong=12/30 |
| T7 | 随机打乱行序 → 全部结果结构逐位一致 |
| T8 | 构造 7 候选使七方法各有不同 argmax → 七结果确证来自各自评分 |
| T9 | CLI 端到端 + 同实验重跑被拒（退出码 4）+ 输出契约 + coverage accepted=2/4 |
| T10 | 人工制造参考不匹配 → 退出码 3、manifest 记 `reference_mismatch`、保留诊断、**无正式结果文件** |
| T11 | run_id 身份：同三元组重跑拒绝；输入变化识别为不同实验（同目录拒绝、新目录可跑）；manifest 字段齐全；跨目录注册表读写语义 |

结果：**11/11 PASS**（详见 `runs/d4_1_1_integrity/selftest_results.txt`）。

## 4. 对抗代码评审（发布前）

4 个评审视角（分组/Top-1、评分与覆盖率、LOSO 与源级、测试与输出契约）+
每条 critical/major 发现 2 名独立反驳者（双方一致才确认）：

- **确认的 P0/P1：0 条**。分组、评分方向、覆盖率单位、LOSO 语义均通过。
- P2（已处理）：
  1. T6 原构造每 Anchor 单候选，无法区分"Anchor 分母 vs 候选行分母"——已按
     建议重建为 60×4 候选并在 T9 增补 accepted 断言（两反驳者均确认评价器
     本身实现正确，此为测试强度问题）；
  2. 运行守卫仅按输出目录生效——已为冻结输入增加跨目录注册表。
- 评审全程只读：未运行评价器、未修改任何文件。

## 5. 输出契约

`runs/d4_1_1_integrity/`（发布门通过后才存在正式文件）：

```
seven_method_results.json   per_anchor_predictions.csv
per_source_results.csv      loso_relative_gains.json
risk_coverage.json          invariant_checks.json
reference_check.json        run_manifest.json
evaluation_run_log.txt      selftest_results.txt
.formal_run_registry.json   （跨目录单次执行注册表）
previous_unverified_run_20261106/   （旧会话未复核输出，隔离保留，不覆盖不删除）
```

重跑方式（仓库根目录）：

```bash
python3 -I src/selftest_d4_1_1_evaluator.py      # 先测试
python3 -I src/run_d4_1_1_fixed_evaluator.py     # 后评价（同一 run_id 只允许一次）
```

## 6. 范围与遗留

- 本轮**不复算** M_hard / G1（身份区分 margin）类指标——它们需要视觉特征，
  超出"只修评价器与统计"的范围；旧文档中相关结论维持"未复核"状态。
- 旧文档与本轮重算冲突处（Kin+D4 wrong_50 8/30 vs 9/30 等）以
  `docs/d4_1_1_final_assessment.md` 的复核数值为准。
