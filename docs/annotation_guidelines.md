# 人工标注指南（E2 前置真值采集）

日期：2026-11-05 | 目标读者：论文作者本人（唯一标注者）
对象：`datasets/annotations/e1_manual_labels.csv`（20 行 = 10 样本 × real/gen）
配套材料：`runs/manual_review/`（24 个可视化文件）、
`docs/e1_v3_manual_review.md`（样本级复核问题）、原始与生成视频本地路径。

**标注纪律（先读）**：
1. 只填人工目视确认的结果；**禁止**把 v3.1 自动判定抄进任何字段；
2. `unknown` 是合法且鼓励的答案——不确定就填 unknown，不要猜；
3. **真实视频不自动标 Valid**：real 侧只回答"发生了什么"，它是 gen 侧
   评价的参照，不是被评对象；
4. 看不懂/看不清时先填 unknown，把疑问写进 reviewer_notes。

---

## 1. 七个核心问题 → 字段映射

按顺序回答（每行 sample+side 独立回答一次）：

| # | 问题 | 字段 | 允许值 |
|---|---|---|---|
| Q1 | 当前**生成条件**是否明确要求该事件？ | event_required | yes / no / unknown |
| Q2 | 指定目标（行人）是否存在？ | actor_visible | yes / no / partial / unknown |
| Q3 | 指定目标能否可靠关联（全程同一人）？ | actor_identity_confirmed | yes / no / unknown |
| Q4 | 是否存在预期行为（横穿）？ | required_behavior | crossing / standing_walking / none_required / unknown |
| Q5 | 是否完成指定事件？ | event_satisfied | yes / no / partial / not_applicable / unknown |
| Q6 | 观察证据是否足以判断？ | evidence_sufficient | yes / no / partial |
| Q7 | 错误（若有）主要来自哪层？ | primary_error_source | generation / detection / tracking / protocol / none / unknown |

辅助字段：event_type（pedestrian_crossing 或空）、occlusion_present
（yes/no/unclear）、tracking_failure（yes/no）、generation_failure
（yes/no/unclear）、human_verdict（valid/invalid/uncertain）、reviewer_notes。

**Q1 与 Q5 的联动规则（防误标）**：
- Q1=unknown 或 no → Q5 只能填 not_applicable 或 unknown，
  human_verdict 不得为 invalid（协议未确认不算违约）；
- Q1=yes → Q5 按目视如实填 yes/partial/no。

## 2. 优先样本与使用材料（按优先级）

### 样本 0016（目标身份切换 — Q3 的关键案例）

材料：`runs/manual_review/0016_contact_sheet.jpg`（左 real 右 gen，
t_norm 对齐 9 个时间点）、`0016_gen_f055/062/063_boxes.jpg`（t64→t30→t64
换手帧，蓝框 t30、橙框 t64）、`0016_real_f100_boxes.jpg`。

回答重点：
- Q3（gen）：f62 与 f63 的行人是否**同一人**（衣着/体型/步态连续）？
  - 是 → tracking_failure=yes，primary_error_source=tracking；
  - 否（两人前后跟随）→ generation_failure 视情况、
    v3.1 的 uncertain 归因错误需回滚；
- Q1：prompt 原文 "others begin to step across the access road, one
  figure walking away" —— 横穿是否为明确的生成条件？
- real 侧 t75：t=0.926 截断时行人是否仍在画面（clip_end_cut 成立性）。

### 样本 0024（生成协议是否要求行人 — Q1 的关键案例）

材料：`runs/manual_review/0024_contact_sheet.jpg`、
`0024_real_f106/107/108_boxes.jpg`（行人唯一检出帧）、gen 全片。

回答重点：
- Q1（gen）：prompt（夜景雪天）通篇无行人描述——video2world 条件下
  行人是否被要求存在？建议对照 SynPeDAS README 的任务定义；
- Q2（gen）：夜景低对比度下是否目视存在小/模糊行人
  （有 → primary_error_source=detection；无 → generation 或 protocol）；
- real 侧：f107 唯一检出的行人目视状态（远小目标/半遮挡/清晰）。

### 样本 0003 / 0044（Region Presence ≠ Crossing — Q5 语义的关键案例）

材料：各样本 `runs/synpedas_smoke_v3_1/<sid>__gen/keyframes/` 与
`annotated.mp4`；对照 real 侧 keyframes。

回答重点（两个样本同构）：
- Q5（gen）：**区分三档**——行人 (a) 完整横穿道路区域、(b) 进入区域但
  停留/小幅移动（real 侧是横穿的）、(c) 从未进入区域；
- 数值参考（自动证据，供对照不作答案）：0003 gen 最大 dx=0.073、
  0044 gen dx=0.108，均无区域两侧证据；real 侧 dx 0.345/0.518 有贯通；
- 若 (b) → 这是 v3.1 region_presence 语义过宽的实锤人工确认，
  event_suggested=no 或 partial，v4 的 L3/L4 分层将以此校准阈值。

### 其余 6 样本（0002/0020/0027/0030/0035/0037）

每样本先用 `annotated.mp4` 快速过一遍，keyframes 定位关键帧；
重点核对 v3.1 的 weak-evidence 归因是否目视成立（occlusion/
redetection/clip_end_cut 标签与画面是否相符）。

## 3. 填表流程建议（每行 ≤8 分钟，总计约 2.5h）

1. 读 `event_protocol.csv` 该样本的 auto_* 证据列（**只有 Q1 用**，
   作为参考不作答案）；
2. 看 contact sheet / keyframes / annotated.mp4（gen 侧必须看，
   real 侧至少看事件发生前后）；
3. 按 Q1→Q7 顺序填，unknown 兜底；
4. reviewer_notes 记录：关键帧号、看到的具体现象、与自动判定的出入。

## 4. 完成后进入的下游（预注册，标注期间不回头改）

- 20 行全部非空（unknown 也算已填）→ 触发 v4-experimental 消融
  （e2_experiment_plan.md §4）；
- 0003/0044 的 Q5 答案 → 校准 L3 报告阈值；
- 0016 Q3 答案 → 决定 redetection 归因保留/回滚；
- 0024 Q1 答案 → 决定 invalid 判定的协议前置条件实现。
