# J2 数据集 v2 审计报告（阶段 2）

日期：2026-11-05 | 构建：`src/build_j2_v2.py` →
`runs/j2_experimental/{j2_sequences_v2.npz, j2_manifest.csv,
dataset_v2_stats.json}` | 验收：外部 `preflight_j2.py --manifest`
**exit 0，formal_train_ready = true**

## 1. 修复的问题（任务 §二）

| # | v1 缺陷（GATES 缺失证据 #1/#2） | v2 修复 |
|---|---|---|
| 1 | NPZ 无溯源（源视频/actor/pair） | manifest 全字段 + NPZ 内嵌 `source_video_id/actor_id/pair_id` 数组 |
| 2 | 目标=段均值特征，Δ 对应段首——**语义错位** | 目标=候选段**首个有效观测帧**的单帧特征（prefer_visible k=1）；`target_time_s` = 该帧真实时间戳 |
| 3 | 无独立 val | 源视频级四划分（见 §2） |
| 4 | 测试集已被反复查看 | **exploratory_test（53 已查看源）与 final_test（2 个从未查看源）显式分离**；final 仅 12 序列，标注"方向性 sanity，非有统计力的测试" |

严格检查（preflight 逐行验证）：`target_time_s > anchor_end_time_s` ✓；
`delta == target_time_s − anchor_end_time_s`（<0.06s 容差）✓；
源视频不跨 split ✓（构造保证 + preflight 复核）。

## 2. 数据构成（412 序列）

| split | 源视频 | 序列数 | 说明 |
|---|---|---|---|
| train | 73（J1 查看过，可作校准） | 240 | 阈值/训练 |
| val | 18（从 train 源 carve，seed 51） | 60 | 早停/验证 |
| exploratory_test | 53（**J 系列已查看**） | 100 | 显式标记探索性——不冒充独立测试 |
| final_test | **2（从未查看）** | 12 | 方向性 sanity ONLY |

## 3. 重复与重叠统计（防虚增）

- **重复 anchor 键**（同源+同 actor+同 anchor 末帧）：45——来自
  同一 actor 多 gap 档（Δ∈{15,30,45}）共享 anchor；
- **同 actor 重叠时间窗**：101——anchor 窗口与该 actor 其他样本
  的目标/窗口重叠；
- 处置：如实记录（未删除——多 gap 档是**设计维度**而非重复），
  但**有效独立训练单元 = 240 序列 / ~73 源 / ~几十个独立 actor**
  ——这个量级是后续"训练数据不足"结论的直接依据。

## 4. 溯源与缓存标识

manifest 每行含 `feature_cache_revision: j1_baseline+M1_mapping_v1`
（帧映射 M1 已核验）+ `split_note`（viewed/never_viewed）+
`target_type: first_valid_candidate_observation`。

## 5. 数据准入结论

**通过**（preflight 0 blockers；时间对齐/源隔离/目标语义全部
机器验证）——正式训练的前置条件满足；但 final_test 的统计力
限制与训练量级限制在结论解读中显式携带。
