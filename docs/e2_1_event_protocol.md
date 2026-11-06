# E2.1 事件协议：分层字段定义与人工复核材料

日期：2026-11-05 | 实现载体：`prompt_source_generation_audit_v2.csv`（8 行）
+ `wmeval/direction_audit.py`（字段语义的代码化）

---

## 1. 十三分层字段（P0-5）

| 字段 | 层 | 取值 | 填写方 |
|---|---|---|---|
| `prompt_requirement` | ① 文本条件 | `direction;semantic_type`（left_to_right/right_to_left/from_right/from_left/toward_camera/away_camera/unspecified/none_required/unknown_prompt_unavailable） | 机器解析（人工复核） |
| `source_observation` | ② 源观察 | `image_left/image_right/approx_static;scale=…;dx=…` | 机器（轨迹数值） |
| `generated_observation` | ③ 生成观察 | 同上 | 机器 |
| `condition_source` | ①③桥 | `prompt_text+real_video_first~16f(video2world)`（P0-4 结论） | 机器（条件审计） |
| `conditioning_frames_scope` | ①③桥 | `likely_first_~16_frames_replayed_then_77_generated` | P0-4 审计 |
| `actor_match_status` | ④身份 | confirmed/possible/unknown | **人工**（单/多轨迹只是证据） |
| `camera_motion_status` | ⑤相机 | unknown（默认）/excluded（人工确认后） | **人工** |
| `event_applicability` | ⑥适用性 | applicable/not_applicable/unknown（无行人条件→not_applicable；prompt 缺失→unknown） | 机器+人工 |
| `event_satisfaction` | ⑦满足 | not_assessed（默认）/yes/no/partial | **人工**（真值列） |
| `prompt_source_conflict_candidate` | ⑧冲突候选 | yes/no | 机器 |
| `prompt_source_conflict_confirmed` | ⑧冲突确认 | yes/no/**unknown**（永不自动 yes） | **人工**终审 |
| `evidence_level` | ⑨证据 | none/medium/strong | 机器 |
| `review_required` | ⑩复核 | yes/no | 机器 |
| `failure_attribution` | ⑪归因 | conflicted_conditions_excluded / unresolved_pending_evidence / not_applicable / unresolved_prompt_unavailable / descriptive_only | 机器（协议决定） |

**层间禁压缩规则**（代码化于 assess_direction，回归覆盖）：
- ①≠②：prompt 方向与源观察相反 → 冲突候选（非模型违约）；
- ②≠③：像素方向差 → 观察记录（非语义反转；相机未排除时永不升级）；
- ③像素一致 ≠ ④actor 一致：`actor_level=unknown` 直到身份确认；
- ⑥not_applicable → 永不输出 Invalid；
- ①unknown（prompt 缺失）→ 永不视为 none_required。

## 2. 人工复核材料包（P0-6，四重点样本）

### 0044（冲突候选确认——最高优先）

| 材料 | 路径 |
|---|---|
| 原始/生成视频 | `repos/SynPeDAS/{JAAD_crossing_moments,generated_video}/video_0044_0_44_202b_f60-210.mp4` |
| Prompt 原文 | `repos/SynPeDAS/video_generation_prompts/video_0044_0_44_202b_f60-210.txt`（sha256=719e6f45…） |
| 数值证据 | v2 审计表行 8；`runs/e2_1_conditioning/frame_similarity_full.csv`（前 16 帧差 7.7、末 20 帧 13.1） |
| 关键帧 | `runs/manual_review/` 无 0044 专帧——用 v3.1 keyframes：`runs/synpedas_smoke_v3_1/video_0044_0_44_202b_f60-210__{real,gen}/keyframes/` |
| JAAD GT | `jaad_gt_evidence.csv`：crossing=1、crossing_point=74（窗口 t≈0.09，**条件帧范围内**）、无遮挡无出画 |
| Actor track | real t1（150 帧全轨迹）、gen t1（93 帧）——双侧单轨迹（身份待确认） |
| 待答 | ①源/生成起点同一行人？②条件帧（前 ~16 帧）内行人可见？③sidecar 建议文本 approve？④生成器条件范围证据 |

### 0035（像素差语义归因）

| 材料 | 路径 |
|---|---|
| 数值 | real dx=−0.523 image_left、gen dx=+0.126 image_right；双侧高度比 3.58/3.35（growing）；JAAD GT：**crossing=0、motion_direction=LONG（非横穿）** |
| 待答 | ①接近语义（尺寸/人行道参照）双侧是否成立；②自车运动是否解释像素差；③（GT 已示非横穿）prompt"towards the camera"在生成中是否满足 |

### 0003（actor 对应）

| 材料 | 数值：real 4 轨（主 t29 +0.345 L→R）、gen 6 轨（主 t25 +0.073）；prompt 提及第一名行人 from the right + 后续行人+狗 |
|---|---|
| 待答 | 原生分辨率辨认三名目标并对应 track（材料：v3.1 双侧 keyframes + annotated.mp4） |

### 0016（碎片链身份）

| 材料 | t30/t64 三段链（dx≈0.002/帧连续）；10 条 gen person track；材料 `runs/manual_review/0016_*` |
|---|---|
| 待答 | 链外观连续性（衣着/步态）——决定 redetection 归因去留 |

**纪律**：全部 PENDING_HUMAN_REVIEW；本模型未查看任何图像；
结论回填 v2 审计表 `human_review_status` / `prompt_source_conflict_confirmed`
/ `actor_match_status` / `event_satisfaction` 列。

## 3. 与 E2 标注任务的合并口径

- `e1_manual_labels.csv` 的 20 行真值标注**不受本轮影响**（事件级真值
  独立于方向审计）；
- v2 审计表是**方向/条件层**的补充真值采集，两表以 sample_id 关联；
- conditioning 结论（P0-4）修正了 `event_protocol.csv` 复核的语义
  基础：`event_required` 的判定对象是 **prompt 文本事件在生成
  roll-out 中的发生**（非源视频后段复制）。
