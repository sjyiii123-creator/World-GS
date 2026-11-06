# E1 v3 人工视觉复核清单（manual review）

> 状态：**PENDING_HUMAN_VISUAL_REVIEW**
> 本文档由纯数值分析生成。撰写本文档的模型**不支持视觉输入**，
> 未经人工查看图片/视频，所有视觉语义问题均保持待确认状态。
> 所有图片文件仅在本地由 OpenCV 生成，路径如下，请在本地查看。

生成日期：2026-11-04
对应实验：`runs/synpedas_smoke_v3/`（E1 v3，20/20 处理完成）
标注表：`datasets/annotations/e1_manual_labels.csv`（真值字段全部留空，待人工填写）

---

## 0. 为什么需要人工复核

v3 的两个 `invalid` 判定（0016 gen、0024 gen）经数值证据复核后发现：

- **0016 gen 的"行人在事件区域内消失"高度疑似 ByteTrack ID 碎片化（假阳性）**；
- **0024 gen 的"无行人"与生成 prompt 不一致**：该样本的 prompt（夜景雪天场景）
  通篇未要求行人出现，"行人缺失"是否构成事件失败取决于测试协议的解释，
  不能由系统单方面判定。

因此这两个样本的最终结论需要人工目视确认。此外 v3 判 `valid` 的样本中，
部分生成视频的行人位移量远小于原始视频（见 §3），也需要抽查。

---

## 1. 样本 0016（video_0016_0_16_72b_f0-150）

### 1.1 系统判定

| 项 | 值 |
|---|---|
| v3 判定（gen） | **invalid** — actor (person) vanished inside event region [(0.25, 0.55, 0.75, 0.95)] — in-progress event interrupted (tracks [30]) |
| v3 判定（real） | uncertain — weak evidence endings: t36 occlusion, t54 clip_end_cut, t75 clip_end_cut |
| v2 判定（gen/real） | invalid / invalid |

### 1.2 数值证据（来自 detections.jsonl / tracks.json，v3）

**核心事实——同一行人的轨迹被切成 3 段 track：**

| 片段 | 帧 | x_norm 范围 | y_norm | 备注 |
|---|---|---|---|---|
| track 64 | f45–f54 | 0.519 → 0.536 | ≈0.635 | 匀速右移 ~0.002/帧 |
| track 30 | f55–f62 | 0.538 → 0.552 | ≈0.633 | 匀速右移 ~0.002/帧 |
| track 64 | f63–f92 | 0.553 → 0.599+ | ≈0.630 | 匀速右移 ~0.002/帧 |

- 三段在时间上互补、空间上连续、速度一致（x 增速 ~0.002/帧贯穿），
  是**同一个人向右横穿**的完整轨迹被 ByteTrack 交替赋了两个 ID。
- track 30 最后检测：f62，bbox (697,424,715,468)，conf 0.63，
  中心 (0.552, 0.634)——事件区域内部、远离画面边缘（x 余量 0.45）。
- f63 起 track 64 的 bbox (696,425,720,469) 出现在**同一位置**（中心距 0.001）。
- track 30 消失后，视频剩余部分仍有 128 个 person 检测
  （检测器全程工作：coverage 93/93 = 1.0，person 检测 342 个）。
- 无遮挡证据：f62 无任何 bbox 与 track 30 的 bbox IoU ≥ 0.10。
- v3 归因为 anomaly 的原因：redetection 逻辑只识别
  "消失帧之后**新建**的 track"（first_frame > last_frame），
  而 track 64 首帧 f45 早于 f62，属于**已有 track 的重新接管**，被漏检。
  （该逻辑盲区已在 v3.1 修复，见 error_analysis.md §2。）

### 1.3 需要人工查看的文件

| 文件 | 内容 |
|---|---|
| `runs/manual_review/0016_contact_sheet.jpg` | real（左）与 gen（右）按 t_norm 对齐的 9 个时间点对照，含检测框与 track ID |
| `runs/manual_review/0016_gen_f045_boxes.jpg` | gen f45：track 64 段开始（橙框 t64，蓝框 t30） |
| `runs/manual_review/0016_gen_f054_boxes.jpg` | gen f54：track 64 段结束前 |
| `runs/manual_review/0016_gen_f055_boxes.jpg` | gen f55：track 30 接管 |
| `runs/manual_review/0016_gen_f062_boxes.jpg` | gen f62：track 30 最后帧（判定"消失"的帧） |
| `runs/manual_review/0016_gen_f063_boxes.jpg` | gen f63：track 64 重新接管（关键帧） |
| `runs/manual_review/0016_real_f100_boxes.jpg` | real 对齐帧（t≈0.67） |
| `runs/inspect_gen0016_f20.jpg` / `runs/inspect_gen0016_f62.jpg` | 上一轮生成的无框抽查帧 |
| 原始视频 | `repos/SynPeDAS/JAAD_crossing_moments/video_0016_0_16_72b_f0-150.mp4`（150f@29.97fps，1920x1080，5.005s） |
| 生成视频 | `repos/SynPeDAS/generated_video/video_0016_0_16_72b_f0-150.mp4`（93f@16fps，1280x704，5.8125s） |

**时间对齐方式**：real 与 gen 帧数/FPS/时长均不同
（150f@29.97fps=5.005s vs 93f@16fps=5.8125s），对照图按
**归一化时间 t_norm = frame/(num_frames−1)** 对齐，不假设帧号一一对应。
对齐记录见 `runs/manual_review/0016_alignment.json`。

### 1.4 需要人工回答的问题（0016）

1. **行人是否真的在 f62（t≈0.67，约 3.9s）从画面中消失？**
   （数值证据显示 f63 起同一位置有 track 64 的检测——即行人很可能持续存在，
   只是换了 track ID。请目视确认 f62 与 f63 中行人是否为同一人且都在画面中。）
2. 消失判定帧（f62）前后是否存在车辆、树木、建筑物或其他目标的**真实遮挡**？
   （IoU 遮挡检查为阴性，但 2D bbox IoU 对"从行人前方掠过的细长目标"不敏感。）
3. f45–f54 与 f55–f62 与 f63+ 的三段，目视上是**同一个人**吗
   （衣着、体型、步态）？还是两个前后跟随的行人？
4. track 30 ↔ track 64 两次"换手"（f54→f55、f62→f63）画面上是否发生了
   行人间交叉、遮挡或检测框竞争？
5. 该行人是否就是 prompt 中描述的 "others begin to step across the access
   road" 的横穿 actor？（prompt 原文见
   `repos/SynPeDAS/video_generation_prompts/video_0016_0_16_72b_f0-150.txt`，
   其中提到 "one figure walking away from the camera's view"。）
6. real 侧 t75（v2 曾判 invalid 的轨迹）在 t=0.926 于 (0.71, 0.62) 处结束：
   目视确认是视频截断时行人仍在画面中（clip_end_cut 成立），
   还是行人真的在末尾消失？

---

## 2. 样本 0024（video_0024_0_24_99b_f60-210）

### 2.1 系统判定

| 项 | 值 |
|---|---|
| v3 判定（gen） | **invalid** — presence check failed while detector was active — no person found anywhere in the video |
| v3 判定（real） | uncertain — failed checks ['ped_present>ped_in_road']（v3.1 修复精度 bug 后应为通过） |
| v2 判定（gen/real） | invalid / uncertain |

### 2.2 数值证据

**gen 侧（93 帧，16fps，1280x704）：**
- 检测器活跃：82/93 帧有检测（coverage 0.882）；
- 但全部 82 个检测均为 **car**，且只属于 1 条 track（track 1，
  x∈[0.28,0.38]，y∈[0.62,0.65]，均值置信 0.60）——画面中只稳定检出一辆车；
- person 检测数为 **0**；无消失轨迹、无归因记录。
- 11 个无检测帧：f18–21, 23, 28–29, 48–50, 56（检测闪烁，夜景噪声符合预期）。

**real 侧（150 帧，29.97fps，1920x1080）：**
- 行人 track 21 **仅在 frame 107 出现 1 帧**（t=0.7181，norm 位置未记录于
  matches 之外；这是整段视频中唯一一次行人检测）；
- 事件链：ped_present(t=0.7181) → ped_in_road(t=0.7181)，同一帧触发；
- v3 中 order check 因精度 bug 误判 failed（0.71812 > 0.7181），已修复。

**生成条件（prompt 原文要点，
`repos/SynPeDAS/video_generation_prompts/video_0024_0_24_99b_f60-210.txt`）：**
- 夜间、冬季街景、雨雪、低能见度、湿滑路面反光、远处车辆大灯；
- **通篇没有任何 pedestrian/行人/横穿的描述**；
- 场景元素：雪堆、路灯、双黄线、多层建筑、车灯。

### 2.3 需要人工查看的文件

| 文件 | 内容 |
|---|---|
| `runs/manual_review/0024_contact_sheet.jpg` | real（左）/ gen（右）8 个时间点 t_norm 对齐对照 |
| `runs/manual_review/0024_real_f106/107/108_boxes.jpg` | real 行人唯一检出帧（f107）及前后帧 |
| `runs/manual_review/0024_gen_f000/030/066/092_boxes.jpg` | gen 侧抽查帧（显示仅有 car 检测） |
| 原始视频 | `repos/SynPeDAS/JAAD_crossing_moments/video_0024_0_24_99b_f60-210.mp4` |
| 生成视频 | `repos/SynPeDAS/generated_video/video_0024_0_24_99b_f60-210.mp4` |
| 对齐记录 | `runs/manual_review/0024_alignment.json`（t_norm 对齐） |

### 2.4 需要人工回答的问题（0024）

1. gen 视频中是否**确实**不存在行人？（夜景+雪+低对比度下 YOLO 可能漏检
   小目标/模糊目标；请目视扫全片，尤其是中景和路灯照亮区域。）
2. 是否存在很小、模糊、部分被雪/车灯/雨丝遮挡的行人？
3. real 侧行人（f107 唯一检出）目视是什么状态：远处小目标？半遮挡？
   还是清晰可辨的横穿行人？该行人是否构成"安全关键事件"？
4. **生成条件是否明确要求保留这个行人？**（prompt 未提及行人；
   若 SynPeDAS 协议只在部分样本中要求事件复现，则 0024 的
   "行人缺失"不应记为生成失败——请对照 SynPeDAS README/论文确认协议。）
5. 若协议确实要求行人存在：该样本是"生成失败"还是"检测失败"？
   （gen 侧 conf≥0.3 阈值下 0 个 person 检出，可在本地以更低阈值重跑检测验证。）

---

## 3. v3 判 valid 样本的抽查建议（region_presence 语义问题）

数值分析显示：v3 的 `ped_in_road` 使用 `region_presence`（曾在区域内）
语义，**不构成"横穿完成"的证据**。以下 gen 样本判 valid 但行人位移量
远小于 real 侧对应片段，建议抽查（优先级从高到低）：

| 样本 | gen 行人最大 dx（norm） | real 对应 dx | gen 区域两侧证据 | 建议 |
|---|---|---|---|---|
| 0003 gen | 0.073（t25, 61 帧全在区域内） | 0.345（t29，走出区域右侧） | 无 | 高优先抽查：疑似"区域内小幅移动"被当成交付了横穿事件 |
| 0044 gen | 0.108（t1, 93 帧全在区域内） | 0.518（t1，从区域左外进入） | 无 | 同上，行人基本原地/小幅移动 |
| 0044 real | —（对照） | — | — | 判断 real 行人是否确在横穿中途被剪辑 |
| 0030 gen | 0.332（t3，走出区域右侧） | 0.472（t9） | 有 | 部分横穿（进入→走出区域），抽查运动方向是否合理 |
| 0002 gen | 0.415（t3，走出区域右侧） | 0.768（t2，两侧都有） | 有 | 同上 |

抽查用文件：各样本目录 `runs/synpedas_smoke_v3/<sid>__gen/keyframes/`
（每个 person track 的首/中/尾帧）与 `annotated.mp4`（带框视频）。

---

## 4. 人工标注表填写指南

文件：`datasets/annotations/e1_manual_labels.csv`（20 行：10 样本 × real/gen）

字段含义与建议取值：

| 字段 | 含义 | 建议取值 |
|---|---|---|
| `event_required` | 生成条件（prompt+协议）是否**要求**该事件出现 | yes / no / unclear |
| `event_type` | 事件类型 | pedestrian_crossing（本批默认，请确认） |
| `actor_visible` | 横穿 actor 在视频中目视可见 | yes / no / partial |
| `actor_identity_confirmed` | 目视能否确认 actor 身份（与 prompt/原视频对应） | yes / no |
| `event_satisfied` | 事件目视是否发生（进入并穿越道路区域） | yes / no / partial / unclear |
| `occlusion_present` | 关键帧处是否存在真实遮挡 | yes / no / unclear |
| `tracking_failure` | 是否目视确认 track 断裂/ID 切换（非真实消失） | yes / no |
| `generation_failure` | 是否目视确认生成内容本身缺失/畸变 | yes / no / unclear |
| `human_verdict` | 人工最终判定 | valid / invalid / uncertain |
| `reviewer_notes` | 自由备注 | 文本 |

**填写纪律**：只填人工目视确认的结果；不要把 v3 自动判定抄进真值字段。
自动判定仅供交叉对照（见 `docs/e1_v3_error_analysis.md`）。

在人工标注完成之前，**任何"误放行率/误拦截率下降"的结论都不成立**。

---

## 5. 复核优先级汇总

1. **0016 gen f62/f63**（invalid 假阳性嫌疑，已获数值证据支持）—— 最高
2. **0024 gen 全片**（invalid 的协议依据问题）—— 高
3. **0003 gen / 0044 gen**（valid 的语义过宽嫌疑）—— 高
4. 0016 real t75 末尾截断、0024 real f107 行人状态 —— 中
5. 其余 uncertain 样本的 weak-evidence 归因抽查 —— 低（可抽 0002 real t5/t22）
