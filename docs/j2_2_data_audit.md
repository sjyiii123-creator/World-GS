# J2.2 数据审计报告（§三）

日期：2026-11-05 | 构建：`src/build_j2_2_data.py` →
`runs/j2_2_experimental/{j2_sequences_expanded.npz,
j2_manifest_expanded.csv, data_expansion_stats.json}`

## 1. 协议执行（任务 §三逐项）

| 要求 | 执行 |
|---|---|
| 原 Val/Expl/Final 源不进训练 | **排除 25 个源**（v2 val 15 + expl 8 + final 2——未重新划分） |
| 不重随机划分已有测试源 | 零重划分（split 沿用 v2 源级桶） |
| 去除完全重复记录 | pair_id 唯一 ✓；**anchor 去重**：每 (source,actor,anchor_end) 留 1 个代表 gap（30f）——v2 保留 3 个 gap 变体（45 重复键），J2.2 去除 |
| 重复 Anchor/时间重叠统计 | 去重后同 actor 重叠窗 171（报告非删除——独立 anchor 间时间重叠是数据现实） |
| 按源/Actor 报告数量 | 见 §2 |
| 共享真实视频帧标记 | overlap 统计 + manifest 逐行 anchor_frames/target_frame（可机械复核） |
| 目标与时间间隔明确 | 每行 target=首个有效观测帧 + delta_s（v2 协议） |
| 不用 ByteTrack ID 作真值 | 全部 JAAD 官方 ped id ✓ |

## 2. 扩充规模（独立源/Actor 报告——任务 §三.9）

| 口径 | 数量 |
|---|---|
| 全量去重池 | **1,245 独立 anchor / 82 源 / 444 actor** |
| v2 train 基线 | 53 源 / 160 actor |
| **新增（池 vs v2）** | **+29 独立源 / +284 独立 actor** |
| 实际抽样训练（特征预算） | **800 序列 / 81 源 / 402 actor**（0 提取错误） |

**预算声明（诚实）**：800 抽样受特征提取时间约束（~695 新序列
× 13 帧本地提取）；**全量协议与 manifest 已固化**——后续扩展为
全量 1,245 是增量运行而非重定义。

## 3. 源泄漏核验

扩充源池 = JAAD-train ∩ ¬(v2 val/expl/final 源)；验证集沿用 v2-val
（15 源）——扩充源与验证源**零交集**（构造保证 + 训练脚本加载时
分离核验）。Final Test（2 源）保持零接触。
