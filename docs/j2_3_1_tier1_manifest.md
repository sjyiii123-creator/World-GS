# J2.3.1 Tier-1 候选级清单

日期：2026-11-05 | 数据：`runs/j2_3_1_alignment/tier1_candidates.csv`
（修复后协议：正确归属 + 去重 + 因果外推门控）

## 1. 候选级表结构（任务 §六全字段）

anchor_id / candidate_id / candidate_timestamp /
is_true_match / candidate_feature_key / kinematic_distance /
spatial_gate_result / hard_negative_tier / source_video_id / split /
evidence_provenance（+anchor_last_frame / candidate_delta_frames）。
**每 anchor 可恢复完整竞争集合**（不再只存数量与前 3 示例）。

## 2. 真实可训练竞争候选（去重后）

| 类别 | 数量 |
|---|---|
| 真主候选（t0，监督标签） | **412** |
| **Tier-1 门控竞争者**（同源/异身份/时间相容/外推空间门控） | **2,043** |
| Tier-2（同源未过门控） | 1,128 |
| 有 Tier-1 竞争的 anchor | **390/412（94.7%）** |
| gap 分层（15/30/45 帧 ±8）Tier-1 覆盖 | 均衡三档 |

## 3. 与 J2.3 旧结果对照（作废声明）

| 口径 | J2.3（污染） | J2.3.1（修复） |
|---|---|---|
| Tier-1 "数量" | 119,081（帧级重复） | 2,043 独立候选 |
| 有竞争 anchor | 391/412 | 390/412 |
| 竞争者证据 | 归属污染 + 未来位置门控 | 可审计（逐行溯源键） |

数量相近属巧合（旧统计的 anchor 覆盖被两种错误部分抵消）——
**证据质量根本不同**：新表每行可追溯到 (src, actor, frame) 的
去重观测。
