# J2.3 困难负样本审计（G2）

日期：2026-11-05 | 构建：`src/build_tiered_negatives.py` →
`runs/j2_3_diagnostics/{tiered_negatives.csv, tier_stats.json}`
身份真值：JAAD 官方 ID（从未使用 ByteTrack/生成侧自动 ID）

## 1. 分层定义（时空证据）

- **Tier-1**：同源、在 gap 目标 ±5 帧窗口内观测、且与 anchor
  轨迹空间门控（中心距 ≤0.30 norm）——**真实共现竞争者**；
- **Tier-2**：同源不同时段（>60 帧外）；
- **Tier-3**：跨源（easy）。

每行证据字段：source_video_id / anchor_actor_id /
candidate_actor_id / anchor_last_timestamp /
candidate_first_timestamp / temporal_overlap / spatial_gate /
identity_label / hard_negative_tier / evidence_provenance。

## 2. 统计（412 anchor 序列）

| 项 | 值 |
|---|---|
| 有 Tier-1（空间门控后）竞争者的序列 | **391/412（95%）** |
| 帧级 Tier-1 负样本对计数 | 119,081（去 actor 后见 CSV 逐行） |
| Tier-2 同源不同时段 actor 总数 | 1,741 |
| gap 三档 Tier-1 覆盖 | 0.5s:128 / 1.0s:142 / 1.5s:130（均衡） |

## 3. 结论（任务 §十三.5/6 的答案）

- **真正的 Tier-1 时空竞争负样本充足**（95% 序列可用）；
- **gap 覆盖均衡**（短/中/长间隙都有竞争者）；遮挡与外观相似性
  维度由 occ 标签（V2 manifest 逐行）与特征空间（G1 hard 统计）
  分别携带；
- **旧"同源即 hard"的定义已被修正**：G1/G3 使用的 hard 近似
  （同源不同 actor）在 Tier-1 审计后确认**方向正确但纯度不足**
  （未含空间门控）——判别训练（D4）必须用 Tier-1 门控集。
