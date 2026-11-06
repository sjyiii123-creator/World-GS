# Actor 时序样本数据集（J 系列训练/评估基础）

日期：2026-11-05 | 生成：`src/build_actor_sequences.py`（seed=11，可复现）
位置：`datasets/processed/actor_sequences/`

---

## 1. 身份真值来源（非 ByteTrack）

**JAAD 2.0 官方行人 ID**（XML `<box><attribute name="id">`，形如
`0_16_72b`）——与 SynPeDAS sample_id 中段同源（此前已建立映射）。
ByteTrack ID 从未被用作身份真值（任务纪律 §二.6）。

## 2. 关键数据事实（决定协议设计）

实测（211 窗口、1,815 个行人）：
- **全遮挡帧 = 0、可见性间隙 = 0**——JAAD 采用 CVAT *插值*标注，
  bbox 逐帧连续存在（即使行人被遮挡/出画边界外也有插值框）；
- 部分遮挡（occ==1）帧 54,109（163 clips）——**可见性质量信号存在，
  但不产生间隙**；
- 204/211 clips 有 ≥2 行人（负样本/竞争候选来源充足）。

**推论**：真实"遮挡间隙-重关联"样本在 JAAD 标注中不存在。协议采用
**合成挖除 + 真实身份标签**：从连续轨迹中移除 Δ 帧观测模拟跟踪丢失
（与 E3 注入实验同一哲学——合成的是**条件**，标签是**真实**的）。
`synthetic_gap=yes` 字段显式标记，任何下游使用不得冒充真实遮挡数据。

## 3. 样本结构

| 类型 | 构造 | 数量 |
|---|---|---|
| A 正对（same_identity） | 同一 JAAD 行人：anchor=[t−11..t]（12 帧）vs candidate=[t+Δ..t+Δ+11]，Δ∈{15,30,45} | 7,920 |
| B 负对（different_identity） | anchor 同上 + 同 clip 另一行人在 [t+Δ−3, t+Δ+15] 窗口内的首个观测段（时间竞争**真实存在**） | 7,603 |
| C 竞争集（competitions.jsonl） | anchor + 窗口内**全部**行人候选（含正主）→ ranking 任务 | 7,849 |

每行字段：`pair_id / synpedas_sample / jaad_video / split / fps /
anchor_ped_id / cand_ped_id / label / gap_frames / gap_seconds /
synthetic_gap / anchor_frames / cand_frames`
（frames = [[xml_frame, bbox_xyxy_px, occluded], ...] JSON）。

## 4. 时间与坐标约定

- **真实时间戳**：`gap_seconds = (t+Δ − anchor末帧) / clip_fps`——
  211 clips 的 fps 混合 29.97/59.94（来自 manifest 实测），**从不按
  帧索引假设固定 FPS**，跨视频比较一律走真实秒；
- **像素坐标**：JAAD XML 的 1920×1080 源坐标 = SynPeDAS real 片段
  同坐标系（clip 帧 i ↔ XML frame − window_start）——特征阶段直接
  裁剪 `repos/SynPeDAS/JAAD_crossing_moments/<sample>.mp4`（本地，
  无需下载 JAAD 全量视频）。

## 5. 划分（防泄漏）

按 **JAAD 官方 split_ids**（default）：train 9,394 / val 1,036 /
test 5,093 对。同一源视频的全部样本在同一侧；test 为官方 held-out
（从未参与行人意图社区训练的 clips，叠加我们自己的"非 E1 开发源"
约束已在 e2 split 处理，此处独立于 E2 划分）。**J1 的特征阈值只在
train 校准，test 只评估。**

## 6. 已知局限（如实）

1. 合成 gap 均匀挖除——真实跟踪丢失的间隙分布（检测置信度渐降、
   部分遮挡拖尾）未建模；J2 若训练，需在协议里补"置信度加权挖除"；
2. 负样本为窗口内真实共现行人的**首个观测**——并非专门构造的
   "外观最相似"hard negative（同衣着/同行群组的 hard case 占比未控）；
3. JAAD 插值 bbox 在遮挡期间的位置是线性插值——用其做运动学外推的
   anchor 时包含插值伪迹（对 J0/J1 对称成立，公平但非真实检测噪声）；
4. 出画（outside）轨迹未用作 gap 事件（出画重入是另一种关联场景，
   留待 v2 扩展）。
