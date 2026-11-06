# E2.1 下一轮实验计划

日期：2026-11-05 | 前置：三个 P0 代码修复（30/30 回归）+ 条件审计 +
DrivingGen 对齐分析。**P0 已全部解决，满足进入正式实验的代码侧条件。**

---

## 1. 进入 B0–B3 消融的剩余条件（全部为人工侧）

| # | 条件 | 状态 | 阻塞什么 |
|---|---|---|---|
| 1 | `e1_manual_labels.csv` 20 行真值标注 | ⏳ 待人工 | 全部需要真值的指标（P/R/IAR/VRR/Selective Risk） |
| 2 | `event_protocol.csv` val 子集复核（event_required 三值） | ⏳ | event_applicability 的样本级确定 |
| 3 | v2 方向审计表终审（0044 confirmed / actor_match / 0035 归因） | ⏳ | 8 样本的条件层真值 |
| 4 | `e2_split_proposal.csv` 确认（含 no-person 调入 val） | ⏳ | 划分冻结 |

## 2. 下一轮实验序列（人工条件满足后）

| 步 | 实验 | 数据/口径 | 产出 |
|---|---|---|---|
| S1 | **B0–B3 消融 @ Val** | val 71 样本（协议确认后）；B0=presence+region、B1=+L1-L4、B2=+actor 链、B3=+三态 | coverage-risk 曲线（固定放行率比较，e2 计划 §4.2 硬约束）+ 全量混淆表 |
| S2 | Agent Binding 真值校准 | 0016 等人工确认的碎片链案例 | 链合并正确率；OC-SORT 观察缓冲升级的触发判定 |
| S3 | 条件冲突样本的协议化处理 | 0044（+人工终审结果） | 论文的数据质量发现小节（SynPeDAS prompt–source mismatch 审计方法） |
| S4 | DrivingGen B1 对齐（px→世界系） | ego npy 轨迹（HF 数据已备）+ UniDepth 可选 | 双口径秩相关（E4 判据） |
| S5 | **Test 集一次性运行** | test 104 样本 | 论文最终数字（跑后冻结参数） |

## 3. 本轮新增可立即消费的资产

- `prompt_source_generation_audit_v2.csv`（13 字段协议表，机器列已填、
  人工列 pending）——**S3 的直接输入**；
- `runs/e2_1_conditioning/`（帧相似度证据）——contact sheet/时间对齐
  说明的修正依据（生成段与真实后段无共同时间轴）；
- `docs/theory_to_code_matrix.md` 的四个对照实验设计（均预注册判据）；
- `docs/drivinggen_baseline_analysis.md` 的对齐口径表（论文方法节素材）。

## 4. 风险登记

| 风险 | 缓解 |
|---|---|
| 20 行标注完成前 v4 阈值 temptation | 纪律已三处固化（review/plan/code_audit）；任何阈值改动登记 failure_log |
| 0044 人工终审推翻 conflict | candidate/confirmed 分离后可无损回滚（v1/v2 表并存） |
| conditioning 16 帧假说被生成代码推翻 | 审计文档三假说并记录证据等级；核心推论（无源后段复制合同）依赖"仅前段为条件"而非确切帧数 |
| B1 世界系复算显存/时间超预算 | ego npy 已消除 SLAM 依赖；UniDepth 为可选路径（px 口径为主） |

## 5. 明确不做（本轮已确认边界）

下载大型生成模型权重（Cosmos 32.5GB VRAM 不可行）、重下已有数据、
训练任何生成模型、系统级 CUDA 改动、覆盖 v3.1/v1 审计表、
用测试集调参、向文本模型发送图像。
