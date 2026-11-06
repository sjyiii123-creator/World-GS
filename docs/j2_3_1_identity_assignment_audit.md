# J2.3.1 身份归属审计（P0-1）

日期：2026-11-05 | 指认核实 + 修复 + 回归

## 1. 错误确认（数值）

`build_tiered_negatives.py`（J2.3 版）把每条样本的
**anchor_frames 与 cand_frames 全部登记到 (src, anchor_ped_id)**：

| 事实 | 数值 |
|---|---|
| samples.csv 负身份对 | **7,603**（100% 存在 anchor_ped ≠ cand_ped） |
| 被错误归属到 anchor 身份的 cand 帧观测 | **90,122** |
| 后果 | 其他行人的框进入 anchor 身份的观测图 → Tier-1 的"竞争者"时空证据被污染；anchor 自身观测也被混入他人轨迹 |

## 2. 修复（build_tiered_negatives_v2.py）

- anchor_frames → **(src, anchor_ped_id)**；cand_frames →
  **(src, cand_ped_id)**（正对两者相同、负对分离）；
- 回归测试（selftest_j2_1 追加）：双 actor 负样本的候选框只进入
  其真实身份的观测记录；
- 身份真值始终 JAAD 官方 ID（ByteTrack 零使用）。

## 3. 影响判定

**J2.3 的 391/412 Tier-1 统计作废**（观测图污染 + 未来位置门控
泄漏）；G1 表示诊断（身份键来自 v2 manifest 的正对 anchor——
归属正确）**不受影响**；G3 可见性对照（同样基于正对）不受影响。
