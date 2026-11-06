# 横穿事件语义：Presence → Region Occupancy → Progress → Completion

日期：2026-11-05（E2 第五、六阶段）
实现：`src/wmeval/crossing_evidence.py`（独立证据输出，**未接入 v3.1 判定**）；
自测 `src/selftest_e2.py` 覆盖全部层级语义。

---

## 1. 动机

v3.1 的 `region_presence=True` 只证明"某个 person 检测中心至少一帧落在
矩形内"。E1 v3.1 错误分析 §4.1 的实测：0003/0044 gen 行人 dx=0.073/0.108、
无区域两侧证据（未横穿）却被判 valid——**"曾在区域内"被当成了"完成横穿"**。

## 2. 四级证据层次（每条身份链独立评估）

| 级 | 名称 | 判定内容 | 输出状态 |
|---|---|---|---|
| L1 | Presence | 该（绑定的）actor 是否出现 | bool + n_detections + 帧跨度 |
| L2 | Region Occupancy | 其中心是否曾进入规定区域 | bool + 首末入区帧 + 入区帧占比 |
| L3 | Crossing Progress | 在区域内是否有**持续横向运动** | moving / stationary / weak_or_ambiguous |
| L4 | Crossing Completion | 是否满足预定义完成条件 | 见 §3 状态枚举 |

L3 证据量（全部记录）：`net_dx_norm`（首末中心 x 差）、
`direction_consistency`（同号位移步比例）、`moving_frame_frac`、
`mean_speed_norm_per_frame`、`in_region_frames`。

**报告阈值（非判定阈值）**：`progress_dx_min=0.15`、
`stationary_dx_max=0.03`、`direction_consistency_min=0.6`。
依据：E1 v3.1 实测 real 侧横穿 dx 0.35-0.77 vs 原地/漂移 0.005-0.11。
**v4 判定阈值必须等人工标注后校准**（E2 纪律），当前值只影响证据
措辞不影响 verdict。

## 3. L4 完成状态枚举（刻意非布尔）

| 状态 | 语义 | 判定条件 |
|---|---|---|
| `completed` | 完整横穿 | L3=moving ∧ 从区域一侧进入、另一侧离开（entered_from ≠ exited_to，均为 left/right） |
| `in_progress` | 横穿进行中、链在区域内且**早于片尾**结束 | 无出区证据且 last_t < 0.90 |
| `partial_clip_cut` | 视频在横穿中途截断，**不可判** | 链持续到片尾 10% 且仍在区域内 |
| `not_progressing` | 在区域内但无持续横移（站着/小幅移动） | L3 ∈ {stationary, weak} |
| `no_occupancy` | 从未进入区域 | L2=False |

**为什么 partial 是一等公民**：JAAD 衍生片段常从横穿中途开始或结束于
横穿中途（E1 实测：0016 real t75 于 t=0.926 被截断）。对这类片段强行
判 pass/fail 是协议错误；`partial_clip_cut` 显式表达"不可判定"。

**in_progress 的歧义保留**：链在区域内、早于片尾消失——可能是
(a) 跟踪丢失（链关联证据会显示接管/歧义）或 (b) 真实消失（E1 v3 的
"vanished mid-event" 签名）。本模块**不裁决**，把链关联证据与该状态
一起上报，由 v4 adjudicate 结合 uncertainty 归因处理。

## 4. 图像坐标 ≠ 世界坐标（永久性限制声明）

L3/L4 的全部位移在**归一化图像空间**度量。透视投影下：
远处的行人真实横穿 3 m 可能只产生 dx=0.05；近处行人横穿 1 m 可能
产生 dx=0.4。因此：
- `net_dx_norm` **不是**世界系横穿距离，不可跨深度比较；
- L4 的 completed 证明的是"图像区域两侧贯通"，不是"通过了 N 米道路"；
- 证据 dict 携带 `caveats` 字段永久声明此限制；
- 世界系升级路径 = E4（UniDepth 深度反投影 + 估计内参），明确列为
  可选后续，不在本轮。

## 5. 时间对齐协议（E2 第六阶段）

### 5.1 t_norm 的适用范围与禁止用途

`t_norm = frame_index / (num_frames − 1)`：

| 用途 | 合法性 | 说明 |
|---|---|---|
| 跨长度视频的**事件相位**粗对齐（real↔gen 对照图、contact sheet） | ✅ 当前用法 | 唯一可用的通用轴 |
| clip-cut 判定（last_t ≥ 0.90） | ✅ | 片内相对位置，与对齐无关 |
| **声称两视频同一 t_norm = 同一物理时刻** | ❌ | real 5.005s vs gen 5.8125s：t=0.5 处物理时间相差 0.4s |
| **事件时刻偏移指标（Δt 物理秒）** | ❌ | 需先建立物理时间锚点（见 5.2） |
| 动作条件下的"未来时间" | ❌ | video2world 的未来段长度≠源片段剩余长度 |

**约束来源必须显式**：任何跨视频比较必须报告其使用的对齐轴
（当前全部为 t_norm 相位对齐，记录于各 alignment.json）。
若可获取真实视频时间戳（VideoMeta.timestamp 保留了
frame/native_fps 物理秒），**应优先保留物理时间**用于片内分析；
跨视频比较在建立物理锚点前只允许相位级结论。

### 5.2 物理时间锚点（待研究，列为 v4 议题）

候选方案（均未实施）：(a) 以生成条件首帧为 t=0 锚（video2world 的
自然语义）；(b) 以事件起始（行人首次入区）为相位锚，报 Δt_norm 而非
Δt_s；(c) prompt 时序描述解析（弱证据）。在锚点确立并验证前，
**不计算任何"事件时刻偏移"指标**（research_gap 缺口 3 的机会
暂缓兑现，避免在错误轴上发明指标）。

### 5.3 target_fps 一致性（已审计并修复，F-08）

`VideoReader` 采样时保持 **native 帧索引与 native 时间戳**（回归测试
验证）；三处跟随分析帧率的地方已统一为 `analysis_fps`/`n_analyzed`：
轨迹 dt、annotated.mp4 回放帧率、检测覆盖率分母。
E1 全程 target_fps=None，v3.1 输出不受影响。

## 6. 自测语义用例（selftest_e2.py 已通过）

| 用例 | 期望 |
|---|---|
| 全程贯通（左入右出，dx≈0.49） | L4=completed |
| 区域内站立（net dx≈0，0003-gen 模式） | L2=True, L3=stationary, **L4=not_progressing** |
| 片尾 10% 仍在区域内 | L4=partial_clip_cut |
| 早于片尾结束于区域内 | L4=in_progress |
| **反混淆**：A 链（远端静止）+ B 链（区域内移动） | A: L1∧¬L2；B: L2∧progress；**整体 completed=False**（无单链满足全部层级） |
| 碎片链贯通（两段 track） | 合并后 completed 归属该单链 |
