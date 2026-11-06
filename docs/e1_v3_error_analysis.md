# E1 v3 错误分析（纯数值，无视觉输入）

日期：2026-11-04
数据：`runs/synpedas_smoke_v3/`（E1 v3）与 `runs/synpedas_smoke_v3_1/`（v3.1 = v3 + 两处已确认 bug 修复）
分析模型：**文本模型，未查看任何图像内容**；所有结论均来自
detections.jsonl / tracks.json / events.json / summary.json 的数值统计。
视觉语义判断全部标记为 `PENDING_HUMAN_VISUAL_REVIEW`
（配套人工清单：`docs/e1_v3_manual_review.md`）。

---

## 1. 实验状态核验

v3 输出与此前日志一致（已逐项核对，未重跑）：

| 侧 | valid | uncertain | invalid | 合计 |
|---|---|---|---|---|
| real | 2 | 8 | 0 | 10 |
| gen | 3 | 5 | 2 | 10 |

20/20 视频处理完成；自测 23 passed / 0 failed（本轮修复前基线）。

三张上轮生成的抽查图（`runs/inspect_real_0002_f40.jpg` 等）经
Python/OpenCV 校验：均可解码、尺寸正常（960 宽缩放）、对应视频帧号有效
（f40 → t=0.2685；f20 → t=0.2174；f62 → t=0.6739，与 events.json 中
track 30 的 last_t_norm=0.6739 一致）。

---

## 2. 已证实的代码问题（本轮修复，v3 → v3.1）

### 2.1 Bug A：order check 的时间精度不一致 → 假 failed

**修改前行为**：`check_order` 直接比较两个事件的归一化时间。
presence 检查存全精度浮点（如 0.7181208053691275），region_presence 存
round(t,4)（0.7181）。**同一帧触发的事件**因 0.71812 > 0.7181 被判
顺序失败。实锤案例：0024 real，ped_present 与 ped_in_road 均由 f107
同一帧触发，order check 误判 failed → 整体 uncertain。

**错误原因**：`run_rule_battery` 收集 event_times 时混合了两种精度来源；
`check_order` 未做精度归一。

**最小修复**（`src/wmeval/event_checks.py` check_order）：
比较前统一 `round(t, 4)`（与 CheckResult.evidence 输出精度一致）。
帧间距 ≥ 1/(num_frames−1) ≈ 0.0067 ≫ 1e-4，相邻帧不会被误合并。

**回归测试**：`src/selftest_metrics.py` 新增
"order: same-frame events -> True" 与 "order: reversed events still -> False"。

**对判定语义的影响**：仅消除假 failed；真正的时间倒序仍判 False。
v3.1 实测：0024 real uncertain → **valid**（注意：该 valid 的事件证据
仅 1 帧 person 检测，见 §4.2）。

### 2.2 Bug B：redetection 归因盲区 → 假 INVALID（安全关键）

**修改前行为**：`attribute_endings` 的 redetection 分支只接受
"消失帧之后**新建**的同类别 track"（`0 < first_frame − last_frame ≤ 15`）。
若接管者是一条**已存在的 track**（首帧早于消失帧、检测在消失点附近
中断后恢复），归因落到 anomaly → `adjudicate` 判 INVALID。

**实锤案例**（0016 gen）：同一行人的轨迹被 ByteTrack 切成三段——

| 片段 | 帧 | x_norm | 说明 |
|---|---|---|---|
| track 64 | f45–f54 | 0.519→0.536 | 匀速右移 |
| track 30 | f55–f62 | 0.538→0.552 | 匀速右移 ~0.002/帧 |
| track 64 | f63–f92 | 0.553→0.599+ | 匀速右移 |

三段时间互补、空间连续（段间中心距 ≤0.002）、速度一致（~0.002/帧）。
track 30 于 f62 在 (0.552, 0.634)（事件区域中心、远离边缘）"结束"，
f63 track 64 出现在距 0.001 处接管。**行人并未消失**——
v3 却判 invalid（"actor vanished inside event region"），属于假阳性。
旁证：f62 之后全视频另有 128 个 person 检测（检测器持续工作，
coverage 93/93=1.0）；f62 无任何 bbox 与 track 30 的 IoU≥0.10（无遮挡证据）。

**错误原因**：redetection 只建模"tracker drop & re-acquire（新 ID）"，
未建模"id fragmentation / 交错检测（已有 ID 重新接管）"。

**最小修复**（`src/wmeval/uncertainty.py` attribute_endings）：
redetection 分支增加情形 (b)——同类别 track 在
`(last_frame, last_frame+reappear_frame_gap]` 内有检测、首检测位置距
消失点 ≤ reappear_dist_norm，且该 track **在 last_frame 当帧无检测**
（排除"一直在场的另一 actor"）。阈值未改动。

**回归测试**：新增 "attribution: id-fragmentation handover -> redetection"
（模拟 0016 的交错模式）与 "attribution: coexisting actor at vanish
point -> not redetection"（共存保护）。原有 23 项全部通过
（含"真消失 + 高置信 → INVALID"用例不受影响），现为 **27 passed / 0 failed**。

**对判定语义的影响**：把一类确定性假阳性 INVALID 降级为
UNCERTAIN（redetection 是弱证据，绝不会升 valid）。安全方向：
宁可 uncertain，不误伤 invalid。
v3.1 实测：0016 gen invalid → **uncertain**
（actor endings on weak evidence: [(30, 'redetection')])。

### 2.3 v3.1 复现性验证

| 指标 | 结果 |
|---|---|
| 检测层一致性 | 20/20 视频的 num_detections 与 num_tracks 与 v3 完全相同 |
| 判定变化 | 恰好 2/20，且均为上述两个 bug 的直接因果 |
| v3.1 分布 | real: 3 valid / 7 uncertain / 0 invalid；gen: 3 valid / 6 uncertain / 1 invalid |

检测/跟踪层完全复现，判定差异纯来自归因逻辑——v3.1 可作为后续
人工标注对照的基线版本。

---

## 3. 检测与跟踪层的问题（证据充分，非代码 bug）

### 3.1 ID 碎片化在 gen 侧系统性偏重

person track 内部 gap（相邻检测帧差 >1）统计：

| 侧 | 平均 person track 数 | 平均轨迹连续性* | 平均 gap 数/视频 |
|---|---|---|---|
| real (1920x1080) | 5.7 | 0.96 | 7.5 |
| gen (1280x704) | 4.0 | 0.89 | 5.0 |

\* 连续性 = len(frames)/span(frames)。

最碎样本：0016 gen（10 条 person track / 342 检测 / 15 个 gap / 三段式
碎片链）。差异与生成视频分辨率更低（1280x704）导致的检测不稳定一致。
**含义**：对 gen 侧的"消失类"证据必须先排除 ID 碎片化（v3.1 已做），
否则 invalid 率会被跟踪噪声抬高。

### 3.2 0024 gen 的检测闪烁

82/93 帧有检测但 11 帧空（f18–21, 23, 28–29, 48–50, 56），且全部检测为
单一 car track——夜景雪天低对比度场景下检测器工作状态本身不稳定。
`detection_quality_proxy` 的 coverage=0.882 仍 ≥ 0.5，"detector was
active" 的判定成立，但置信度证据一般（mean 0.60）。
若人工复核发现 0024 gen 中确有小/模糊行人，则该样本应记为
**检测失败**而非生成失败。

### 3.3 real 侧的背景车消失（agent_flags）

0037 real 有 6 条 car 轨迹 unexplained 消失（v3 全部归 anomaly → agent_flags，
不影响事件判定）。v2 时代这类证据曾把 real 判成 invalid；
v3 的 actor-aware 设计（非事件 actor 异常只作 flag）是有数据依据的——
真实视频上几何法无法区分"车开走了"与"车消失了"。

---

## 4. 事件约束设计问题（语义，不是 bug；改动需人工标注支持）

### 4.1 region_presence ≠ crossing event（语义过宽，有数值实锤）

当前 `DEFAULT_CROSSING_SPEC` 的 `ped_in_road` 用 `region_presence`
（"任一 person 检测中心曾落在 [0.25,0.55,0.75,0.95] 内"）。
代码中同时存在更严格的 `check_region_entry`（require_from_outside，
进入语义），但默认 spec 未启用（理由记录为 JAAD 片段常从横穿中途开始）。

**后果实测**（v3 判 valid 的 gen 样本 vs real 对应片段的行人位移）：

| 样本 | gen 行人最大 dx（norm） | real 对应 dx | gen 是否有区域两侧证据 |
|---|---|---|---|
| 0003 gen | **0.073**（t25，61 帧全在区域内） | 0.345（t29，走出右侧） | 否 |
| 0044 gen | **0.108**（t1，93 帧全在区域内） | 0.518（t1，从左外进入） | 否 |
| 0030 gen | 0.332（t3，走出右侧） | 0.472 | 是 |
| 0002 gen | 0.415（t3，走出右侧） | 0.768（两侧都有） | 是 |

0003/0044 的 gen 行人**没有横穿**（区域内小幅移动、无两侧证据），
却因 region_presence=True + 无 weak ending 被判 **valid**。
即当前系统把"出现在道路区域"当作"完成横穿事件"——
正是"行人站在道路边缘/从横穿中途开始的片段被记为满足横穿"的语义混淆。

**建议**（暂不实施，等人工标注后定阈值）：
1. 事件 ID 改名为 `ped_seen_in_road`，如实反映"曾在区域内"语义；
2. 新增 crossing-completion 组合检查：region_presence ∧ 位移
   （如 dx≥0.2）∧ 在区域内持续 ≥N 帧，可选 exit-region 证据；
3. 进入但未离开且非 clip_end_cut 的"半横穿"单独归类 partial_crossing
   → 默认 uncertain；
4. 报告两级证据（seen_in_road / entered_road），region_entry 保留为
   严格模式，便于消融。

### 4.2 presence 的"1 帧即满足"弱证据

0024 real 的 v3.1 valid 建立在**单帧** person 检测（f107，1 match）之上。
presence 检查无最小持续时间/最小匹配数要求。建议（同样待标注后定）：
presence 满足但匹配数 < k（如 3）时降为 weak-evidence uncertain，
或输出 `presence_weak` 标志供 adjudicate 使用。

### 4.3 Invalid 的判定依据必须锚定测试协议

0024 gen 的 invalid 依据是"presence failed while detector was active"。
但该样本生成 prompt（夜景雪天）**通篇未要求行人**——若 SynPeDAS 协议
对 0024 不要求行人事件，则此 invalid 是**验证器施加了协议外约束**。
在人工确认 prompt/协议映射（`event_required` 字段）之前，
0024 gen 的 invalid 只能算"待协议确认"，不能计入模型失败率。
（对应上轮提醒：不能把未在 prompt/协议中规定的条件事后强加给生成结果。）

---

## 5. v2 → v3 判定迁移分析（回应"是否用 Uncertain 换 Invalid"）

v2 → v3 共 6 处变化：

| 样本[侧] | v2 | v3 | 驱动机制 |
|---|---|---|---|
| 0003 real | uncertain | valid | region_entry→region_presence 放宽 |
| 0044 real | uncertain | valid | 同上 |
| 0044 gen | uncertain | valid | 同上 |
| 0030 gen | uncertain | valid | 同上 |
| 0016 real | invalid | uncertain | 弱证据归因（t75 → clip_end_cut，t=0.926 截断） |
| 0030 real | invalid | uncertain | 弱证据归因（t37 → detection_quality，len=6 conf=0.497） |

侧计：real 2 invalid→0 invalid（0→2 valid，uncertain 8→8）；
gen 1→3 valid（2 invalid 不变）。

**对"0 Invalid 是否等于误报消除"的回答：不能。** 机制分解显示：
- real 侧 Invalid 清零**完全**来自两处 ending 重分类（异常→弱证据），
  不是检测或跟踪变好（检测层与 v2 相同）；
- 其中 0016 real t75 的 clip_end_cut（视频末 10% 截断）有明确数值依据
  （last_frame=138/150, t=0.926），0030 real t37 的 detection_quality
  （6 帧轨迹、conf 0.497）同样成立——这两处重分类本身有理；
- 但 v3 的 valid 增加来自 region_presence 放宽（§4.1），其中 0003/0044
  gen 的 valid 在"横穿完成"语义下站不住；
- **结论**：v3 用更合理的弱证据归因消除了 2 个真实误拦截，同时用
  过宽的 region_presence 放进了 2 个疑似误放行。净效果不能在没有
  人工真值的情况下宣称——这正是需要 `e1_manual_labels.csv` 的原因。

---

## 6. v3.1 最终判定与原因分布

v3.1（20/20，检测层与 v3 完全一致）：

| 侧 | valid | uncertain | invalid |
|---|---|---|---|
| real | 3（0003/0024/0044） | 7 | 0 |
| gen | 3（0002/0030/0044） | 6（0003/0016/0020/0027/0035/0037） | 1（0024） |

uncertain 的原因分布（v3.1，按 reason 类别）：
- gen 侧 6 个全部为 "actor endings on weak evidence"
  （redetection / clip_end_cut / id_switch / detection_quality / occlusion）；
- real 侧 7 个：weak evidence 6、anomaly-ending-outside-region 1（0020）、
  detection_quality 若干（0030/0035）。
- **invalid 只剩 0024 gen 一例**，且其协议依据待人工确认（§4.3）。

0016 gen 在 v3.1 的归因链：
track 30 → redetection（接管者 track 64，reappear_frame=63）→
weak evidence → uncertain。行人是否真实存在全程 **PENDING_HUMAN_VISUAL_REVIEW**
（若人工确认三段为同一人 → 0016 是跟踪问题，非生成失败；
若人工确认 f63 起的是另一个人 → 0016 是真实生成异常，v3 判定应恢复）。

---

## 7. 分层的结论清单

**已证实的代码问题（修复并回归测试）：**
1. order check 精度不一致（假 failed）→ v3.1 修复；
2. redetection 盲区（ID 碎片化 → 假 INVALID）→ v3.1 修复。

**检测/跟踪层问题（证据充分）：**
3. gen 侧 ID 碎片化系统性偏重（连续性 0.89 vs real 0.96，0016 最碎）；
4. 0024 gen 检测闪烁（11 空帧）+ 夜景低对比度可能是漏检温床；
5. real 侧背景车消失无法几何判定（agent_flags 处理正确）。

**事件约束设计问题：**
6. region_presence 语义过宽（0003/0044 gen 假 valid 嫌疑，数值实锤）；
7. presence 1 帧即满足（0024 real valid 证据薄弱）；
8. invalid 判定未锚定 per-sample 生成协议（0024 gen）。

**需人工确认的生成错误（PENDING_HUMAN_VISUAL_REVIEW）：**
9. 0024 gen 行人是否真的不存在（vs 漏检）+ 协议是否要求行人；
10. 0016 gen f62→f63 的"换手"是否同一人（vs 两人跟随）；
11. 0003/0044 gen 行人位移不足是否目视可辨（vs 检测框压缩）；
12. 0016 real t75、0030 real t37 的弱证据归因是否目视成立。

**数据边界声明**：在 `datasets/annotations/e1_manual_labels.csv`
填写完成之前，以下结论**不成立**：误放行率下降、Event Precision/Recall、
任何"验证器优于基线"的量化宣称。当前所有 valid/invalid/uncertain
均为自动判定，仅用于暴露问题，不构成性能指标。

---

## 8. 下一轮实验建议（按优先级）

1. **人工标注 20 条样本**（用 §3 复核清单 + 标注表）——产出第一份
   真值，使 v3.1 的 Precision/Recall/Uncertain-Rate 可计算；
2. **协议映射**：逐样本判定 event_required（读 prompt + SynPeDAS 协议），
   修正 adjudicate 的 invalid 前置条件——0024 gen 类判定需锚定协议；
3. **crossing-completion 检查**（§4.1 建议 2/3/4）：以人工标注为阈值
   依据，做 v3.1 vs v4 消融（region_presence vs 组合检查的
   Event-P/R 变化）；
4. **扩样本**：30–50 组（manifest 已支持），重点加入 prompt 明确要求
   行人横穿的样本，降低 0024 类协议歧义占比；
5. B0/B1/B2/B3 基线对照（原实验计划 §七）在上述 1–3 完成后启动，
   人为注入异常集与真实生成集分开报告。

资源估计：1（人工，~2h）；2/3（代码，各 <半天）；4（GPU ~15min/50 样本）。
