# E2 方向审计修正报告（prompt–source–generation 三层分离）

日期：2026-11-05（E2 第二轮，响应外部独立审计修正）
性质：**修正报告**——修正第一轮方向审计中的概念混淆；本轮全部结论由
`src/wmeval/direction_audit.py`（防混淆规则）+ 8 项场景评估产出，
回归测试 `src/selftest_direction_audit.py` **14/14 通过**。
原始数据零改动（8 份 prompt SHA256 与外部溯源表逐一比对一致，回归测试 7）。

---

## 0. 修正的概念框架（五层分离）

| 层 | 内容 | 本轮纪律 |
|---|---|---|
| L1 图像坐标 | `sign(dx_norm)`、bbox 高度比 | 只作**观察**记录，永不直接等于语义 |
| L2 运动语义 | 世界系左/右/接近/远离 | 需 actor 身份 + prompt 行为 + 相机运动排除；**证据不足一律 unknown** |
| L3 Prompt 要求 | 文本字面语义 | `from the right`=出发侧≠行进方向；`towards the camera`=深度运动无横向主张 |
| L4 源视频行为 | 原始视频实际运动 | 与 L3 冲突时 → 数据集构建层缺陷（PROMPT_SOURCE_CONFLICT） |
| L5 模型违约 | 生成是否违反条件 | 只有 L3 明确 + actor 可靠 + 事件可验证时才允许归因 |

## 1. 五类结论（8 样本）

| 类别 | 样本 | 结论 |
|---|---|---|
| **A. 原 Prompt 与源视频冲突** | **0044** | prompt "from left to right"（travel direction）vs 源视频 image_right→image_left（dx=−0.518，150f@29.97）；外部抽样帧目视同向佐证。**隔离出干净 prompt-adherence 统计，单独报告；不改写原 prompt；不归因于模型运动方向失败**。生成侧同为 image_left、位移更小（dx=−0.108）——与源同向，方向层面反而一致 |
| **B. 图像运动存在差异（无语义结论）** | **0035** | real image_left（dx=−0.523）vs gen image_right（dx=+0.126），**但** prompt=toward_camera（无横向主张）、双侧 bbox 高度比 3.58/3.35（均显著增大=接近证据）、相机运动未排除 → **`semantic_direction_reversal=unknown`**。第一轮"real↔gen 方向反转"的表述**降级为像素层观察** |
| **C. 行为语义明确不一致** | ——（无） | 当前 8 样本无一满足 C 类证据门槛（actor 确认 + prompt 方向 + 相机运动排除） |
| **D. 证据不足** | 0002、0003、0016、0024、0030、0037（自动分类） | 原因分层见 §2；**actor 绑定是多前的前置门槛**——即使 real/gen 像素同向（0002/0003/0030/0037 px=no），在 actor 未确认前不授予"无方向问题"结论（比外部审计更严格） |
| **E. 未发现目标方向问题** | ——（候选：0002/0030/0037） | E 类授予条件 = actor 绑定 + 同向确认；当前全部因 actor gate 归入 D，人工绑定后可升级 |

## 2. D 类样本的分层原因

| 样本 | 瓶颈 | 关键事实 |
|---|---|---|
| 0003 | actor 绑定 + "from the right" 语义 | `from the right` 解析为**出发侧**（starting_side）；real/gen 各 4/6 条 person track；prompt 中还有第二名行人与狗——最长轨（t29, +0.345）不必然是 prompt actor |
| 0016 | 多 actor + ID 碎片 | 12/10 条 person track；t30/t64 碎片链已知；主轨（t6）为静止站位者 |
| 0024 | 无行人条件 | prompt **none_required**（无行人提及）→ 横穿缺失**不可计为违约**（event_adjudication=not_applicable）；gen 0 person track ≠ 画面无人（夜景低检出可靠性） |
| 0002 | actor 绑定 | 车库场景 2 行人 R→L 同向（real dx=−0.768 / gen dx=−0.394）；且**非道路横穿事件**（车库开口）——不可按横穿完成度评分 |
| 0030 | actor 绑定 | prompt 为"construction area 接近"无横向横穿要求；real/gen 同向 image_right |
| 0037 | actor 绑定 + 采样差异 | real 300f@59.94 vs gen 93f@16——**帧索引不可比**（frame_index_comparable=False，回归测试 5），同向结论仅像素层 |

## 3. 0044 专项报告（条件冲突样本）

- **证据链**：prompt 短语 "from left to right"（travel_direction，机器解析+回归 1a）；
  real 主轨 x 0.663→0.145（image_left，dx=−0.518）；gen 主轨 x 0.663→0.555
  （image_left，dx=−0.108）；双侧 bbox 高度比 2.33/3.01（均接近）；
  external visual review（WorldGS_E2_Review，抽样帧 0–5.5s）确认双侧同向。
- **结论**：`PROMPT_SOURCE_CONFLICT`——**数据集构建层缺陷候选**（SynPeDAS 的
  prompt 生成与源视频行为不符），非已证实的模型方向失败。生成视频与源
  **同向**（位移幅度差异 0.41 norm 属观察，未标定世界距离）。
- **处置**（已固化）：
  1. 隔离出干净 prompt-adherence 分母（`failure_attribution=conflicted_conditions_excluded`）；
  2. 原 prompt 保持不可变（sha256=719e6f45… 记入审计表）；
  3. sidecar 提供**衍生**源描述建议（`use_for_original_prompt_compliance=NO`，
     `change_type=SOURCE_DESCRIPTION_ONLY_DERIVED_NOT_ORIGINAL_PROMPT`），
     仅供人工复核参考，永不替代原始条件评分；
  4. 人工复核点：确认源/生成起点是否同一行人；查证生成器实际条件输入
     （video2world 用整段还是首段——`conditioning_frames_scope` 目前 unknown）。
- **论文表述建议**：作为 SynPeDAS 数据质量发现单独报告（prompt–source
  mismatch 审计方法本身是贡献素材），与世界模型失败案例严格分开。

## 4. 第一轮结论的勘误记录

| 第一轮表述 | 修正后 | 依据 |
|---|---|---|
| "0035 real↔gen 方向反转"（暗示语义反转） | **像素位移符号不同；语义反转未证实**（prompt 朝镜头 + 双侧接近证据） | 外部审计 + 回归 2b |
| "0044 prompt 描述错误（数据集构建缺陷）" | 保留但强化：**PROMPT_SOURCE_CONFLICT 正式类别 + 隔离协议**，且注明"生成与源同向" | 外部审计 + 回归 1b/1c |
| numeric 审计以"主轨"为代表 | **selection_basis=max_detections 永不等于 actor 身份**（回归 4） | 任务书 §四 |
| （未言明）帧比较 | **frame_index_comparable=False 必须显式**（150@29.97 vs 93@16，回归 5） | 任务书 §四 |

受影响文档已同步修正：`e1_v3_1_independent_review.md`（U5/U6 措辞）、
`e2_prompt_direction_review` 包（已发出的包不可改，本报告即勘误）。

## 5. 交付物清单

| 文件 | 内容 |
|---|---|
| `datasets/annotations/prompt_source_generation_audit.csv` | 8 行主审计表：任务书 §三全字段 + 数值证据（dx/高度比/人数/fps/帧数）+ 五类分类 + 外部复核引用 + `human_review_status=pending` |
| `datasets/annotations/prompt_correction_sidecar.csv` | 0044 专用：衍生源描述建议（NO 合规使用、never_overwrite=YES、approval=pending） |
| `src/wmeval/direction_audit.py` | 防混淆语义判定模块（prompt 解析/像素分类/五类评估/帧可比性守卫/SHA256） |
| `src/selftest_direction_audit.py` | 14 项回归（任务书 §五 7 项全覆盖） |
| `src/build_direction_audit.py` | 审计表生成器（只读输入，可重跑复现） |
| 本报告 | 五类结论 + 0044 专项 + 勘误 + 复核清单 |

## 6. 待人工复核清单（PENDING_HUMAN_REVIEW）

1. **0044**（最高优先）：源/生成起点同一行人确认（全尺寸看 0/2/4/5s，
   材料 `WorldGS_E2_Review/assets/video_0044_*_raw.jpg` 与本地视频）；
   生成器条件输入范围查证（conditioning_frames_scope）；
   sidecar 建议文本的 approve/reject。
2. **0035**：接近语义确认（尺寸变化/人行道参照物相对位置，非横向位移）；
   自车运动检查——决定 px_mismatch 是否为相机运动伪影。
3. **0003**：native 分辨率下辨认 prompt 的第一/第二行人与狗，
   建立源侧 actor 对应（全部 4 条 real track 的起点帧/位置已可从
   v3.1 tracks.json 提供）。
4. **0016**：t30/t64 碎片链的外观连续性确认（材料沿用
   `runs/manual_review/0016_*`）。
5. **event_protocol.csv 复核**时同步消费本表：0044 的
   `event_required` 决策必须引用冲突标记（若按原 prompt 评 = 隔离；
   若按衍生源描述评 = 标记 derived，不进未修改合规统计）。

## 7. 纪律重申（本轮固化进代码的守卫）

- `sign(dx)` 不同 → 只产生 `pixel_horizontal_direction_mismatch=yes`，
  语义列**永远单独评估**且证据不足时 `unknown`；
- `from the right` ≠ `moving rightward`（解析器语义类型分离）；
- 最长轨选择永远标记 `selection_basis`，不隐含 actor 身份；
- 跨视频帧比较必须先过 `frame_index_comparable`（fps+帧数都相等才可），
  SynPeDAS 全部配对**不可**直接比帧号；
- 无行人条件 prompt → `not_applicable`，禁止进入正/负样本分母
  （违约率统计自动排除 conflicted 与 unknown 样本——与
  e2_experiment_plan.md §4.3 真值门控一致）。
