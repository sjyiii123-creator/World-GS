# E2 实验计划：数据划分、消融设计与进入条件

日期：2026-11-05 | 前置：v3.1 冻结 + 人工标注（e1_manual_labels.csv）+ 协议复核

---

## 1. 数据资产现状（全部本地，零新下载）

| 资产 | 规模 | 状态 |
|---|---|---|
| SynPeDAS 四元组 | 211 组（166 个源视频） | ✅ 本地 |
| event_protocol.csv | 211 行，全 unknown | ✅ 待人工复核 |
| e2_split_proposal.csv | dev 36 / val 71 / test 104 | ✅ 提案待确认 |
| e1_manual_labels.csv | 20 行空白 | ⏳ 人工标注中 |
| v3.1 输出 | 20/20 冻结 | ✅ 不再改动 |

## 2. 数据划分（泄漏安全）

### 2.1 划分不变量（脚本已验证）

- **同源视频整组同侧**：19 个源视频贡献 2-6 样本（video_0313 最多 6 个），
  脚本断言"跨侧源组数 = 0"；
- **E1 污染隔离**：v1-v3.1 规则开发用过的 10 个样本固定在 dev
  （其源视频在 manifest 中无其他样本，天然不扩散）；
- **确定性**：sha256(source_group + salt) 哈希分配，无随机数；
- 比例 dev/val/test ≈ 15/35/50（组级）。

### 2.2 分层（机械代理 + 人工维度）

机械 stratum（event_protocol.csv auto 列）：crossing-phrase（164）/
person-only（40）/ no-person-prompt（7）。当前 val 缺 no-person
stratum（6/7 落在 test）——**人工确认划分时须把 1-2 个 no-person 样本
调入 val**（协议歧义样本恰是 Val 要回答的问题）。

人工 stratum（标注后回填，用于 val 内配额）：
- 条件明确 × 事件成立/失败；
- 可疑运动失真（E2 错误类型分析的产出：消失/未横穿/位移不足/身份断裂）；
- 高遮挡难度（occlusion_present=yes 的样本）。

### 2.3 各 split 的用途纪律

| split | 用途 | 禁止 |
|---|---|---|
| dev (36) | 规则/阈值开发（含 E1 10 样本的历史经验） | — |
| val (71) | 协议确认 + 人工真值 + 阈值校准 + 消融主战场 | 禁止用 test 反馈 |
| test (104) | 一次性最终评测（论文数字的唯一来源） | 标注完成前不跑；跑后不回调阈值 |

## 3. 样本筛选（E2 错误类型分析的取材，从 211 组中选）

四类目标样本（前两类机械可排，后两类需检测管线跑 val 子集后人工定）：
1. **条件明确**：auto crossing-phrase 且 prompt 描述无歧义（人工抽查 20 个）；
2. **条件不明确**：no-person-prompt 7 个 + person-only 中的边缘表述；
3. **可疑运动失真**：v4 管线在 val 上输出的 L3=weak/L4=in_progress/
   ambiguous 链样本（预计 20-40 个）；
4. **高遮挡难度**：real 侧 weak-evidence 归因密集（occlusion/
   detection_quality）的源组。

## 4. 消融设计：v3.1 vs v4-experimental（B0–B3）

| 基线 | 内容 | 实现 |
|---|---|---|
| B0 | Presence + Region Presence（v3.1 的检查子集，无归因） | 现有 event_checks 裁剪 |
| B1 | + Event-Aware Constraint Checking（L1-L4 分层证据 → 状态） | crossing_evidence.py |
| B2 | + Actor Identity Association（链绑定，证据不跨个体） | actor_binding.py |
| B3 | + Uncertainty（三态判定 + 归因链） | v4 adjudicate（待设计，见 event_protocol_design §6） |

对照轴同时保留 v3.1 整体作为参照点（非基线，历史版本）。

### 4.1 指标（全部在 val 人工真值上计算；无真值样本一律剔除）

- Event Precision / Event Recall（事件级，按 event_required=yes 的样本）；
- **IAR**（Invalid Acceptance Rate：真违约被判 Valid 的比例）；
- **VRR**（Valid Retention Rate：真合格被判 Valid 的比例）；
- **Uncertain Rate**；
- **Selective Risk**（放行集合内的违约密度：R(f) = Q(f)/C(f)，f 为
  自动放行分数阈值扫掠）；
- **自动放行覆盖率**（auto-release coverage：判 Valid 的样本占比）；
- 单视频运行时间。

### 4.2 防"Uncertain 刷指标"协议（硬约束）

1. **IAR 必须在固定自动放行率下比较**：报告
   "auto-release coverage @ IAR" 联合曲线，不允许只报 IAR 单点；
   若 B3 的 IAR 低于 B0 但 coverage 也低 30pp，判为"未证明价值"；
2. Uncertain Rate 上升必须在选择性风险曲线下解释（放行更少→
   剩余风险应更低，否则 Uncertain 只是把难题藏起来）；
3. 每个基线报告同一张全量混淆表（Valid/Invalid/Uncertain × 真值），
   不许只报比率；
4. test 集只跑最终选定配置一次。

### 4.3 真值门控

- event_required=unknown 的样本：不计入 Precision/Recall/IAR/VRR
  （可计入 Uncertain Rate 分布统计，单独列）；
- human_verdict=uncertain 的真值样本：计入"不可判"层，验证器的
  Uncertain 判对它算什么另行定义（预注册：验证器对真 uncertain 样本
  输出 uncertain 记为命中，输出 valid/invalid 记为过决）。

## 5. 时间对齐与注入（沿用/升级）

- 对照图与人工材料：t_norm 相位对齐（crossing_event_semantics.md §5，
  含禁止用途清单）；
- 人为注入异常（与真实结果分开报告）：本地注入协议沿用
  experiment_plan.md E3（帧删除/box 抹除/延迟出现），CARLA 升级列为
  可选（未下载，不阻塞）。

## 6. 执行顺序与进入条件

前置条件全部满足（e1_v3_1_independent_review.md §5）后：

| 步骤 | 产出 | GPU |
|---|---|---|
| S1 协议复核（人工，val 30-50 个样本的 event_required） | event_protocol.csv 更新 | 0 |
| S2 标注 20 行 E1 真值 | e1_manual_labels.csv | 0 |
| S3 v4 管线跑 val 子集（检测+链+L1-L4 证据） | runs/e2_val/（新目录，不动 v3.1） | ~20 分钟 |
| S4 错误类型统计 + L3 阈值校准（val 真值驱动） | 失真类型×频率表 | 0 |
| S5 消融 B0-B3 @ val | 基线对照表 + 全量混淆表 | ~1 小时 |
| S6 test 一次性运行 | 论文最终数字 | ~40 分钟 |

**纪律重申**：S4-S5 期间任何阈值修改必须登记 failure_log 并在 val 上
重跑受影响基线；test 结果出来后冻结一切参数。
