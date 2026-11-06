# E2.1 代码审计报告：三个确定性缺陷的修复

日期：2026-11-05 | 回归：`selftest_direction_audit.py` **30/30 通过**
（v1 语义保留 + E2.1 新增 16 项）；v1 审计表保留未覆盖
（`prompt_source_generation_audit.csv`），修复后输出
`prompt_source_generation_audit_v2.csv`。

---

## P0-1：bbox 高度比按值排序（已修复）

**修改前行为**：`sorted(hs)[:n]` / `sorted(hs)[-n:]` 按高度**值**排序取
首尾段——测量的是高度分布极值，非时间变化。实锤：三角轨迹（先接近
后远离）被误判 ratio=8.18（"growing"），时间排序正确值为 0.60。

**根因**：与外部审计工具（`vals.sort()` 排 `(frame, height)` 元组）不同，
本地实现把 frame 丢弃后排序。

**修复**：`wmeval/direction_audit.py::bbox_height_ratio(frame_heights)`
——按 `frame_index` 排序，取时间上前/后 15% 段中位数；<2 检测返回
None。`build_direction_audit.py` 与 `selftest_direction_audit.py` 的
`side_ev` 统一调用该函数（单一实现）。

**对 8 样本实际数值证据的影响（重算对比，`frame_similarity` 之外独立）**：

| 侧 | 数值变化 | 趋势分类变化 |
|---|---|---|
| 16/16（8 样本 × real/gen） | 3 处数值微差（0016 real 1.259→1.234、gen 1.674→1.584；0002 gen 1.234→1.224） | **0 处**（growing/stable/shrinking 全部不变） |

解释：当前 8 样本主轨高度近似单调（值序≈时序）。**风险面向未来**：
非单调轨迹（先接近后远离/往返）在旧实现下必然误判——P0-1c 测试
固化该场景。0035 的"growing"结论（接近证据）经修复后仍成立。

**测试**（P0-1a–g，7 项）：单调增/单调减/三角/中间异常框/单框/
缺帧/乱序输入。

## P0-2：冲突未分级（candidate/confirmed 已分离）

**修改前行为**：`prompt_source_conflict=yes` 可在
`source_actor_match` 未确认时直接输出（0044 拿到确定性的 yes）。

**修复**（`DirectionAssessment` v2）：
- `prompt_source_conflict_candidate`：机器可判（prompt 行进方向解析 +
  源像素方向相反）；
- `prompt_source_conflict_confirmed`：**永不自动置 yes**——需要 actor
  身份确认 + 方向参考 + 可追溯证据，且终审在人工；
- `conflict_evidence_level`：none/medium（数值+文本）/strong
  （actor 已确认后）；
- `review_required`：candidate 或任何 unknown 证据 → yes。

0044 现状：**candidate=yes、confirmed=unknown、evidence=medium、
review_required=yes**（与任务要求一致）。

**测试**（P0-2a–c）：actor=possible → 不 Confirmed；相机运动 unknown +
像素差 → 不得推导世界运动冲突（0035 类留在 B 类）；actor=confirmed +
方向明确 → 证据升 strong 但 confirmed 仍待人工。

## P0-3：单轨迹身份推断（已分离两级一致性）

**修改前行为**：单条 person track + 像素同向可输出
`no_direction_issue_observed`（把"恰好一条轨迹"当成身份证据）。

**修复**：
- `pixel_direction_agreement`（观察层）与
  `actor_level_direction_agreement`（语义层）分离；
- E 类结论（no_direction_issue）**要求 actor 确认**；单轨迹未确认 →
  D 类（insufficient_evidence），理由显式注明"track 数量不是身份证据"；
- actor 确认 + 像素一致 → actor_level=yes（唯一路径）。

**测试**（P0-3a–c）：单轨迹身份未知 → D 类 + actor_level=unknown；
actor 确认 + 同向 → E 类；碎片链落在相邻行人路径上 → 关联证据
呈现（ambiguous/evidence 记录），actor 仍 unknown 不自动消解。

## 附带修复（P0-5 联动）

- `prompt_direction_semantics(text, prompt_available)`：缺失/空 prompt →
  `unknown_prompt_unavailable`（**绝不**自动视为 none_required）；
  `assess_direction` 对应分支 `unresolved_prompt_unavailable` +
  review_required=yes（测试 15/15b）；
- 评估分支字段数 bug（4 处 DirectionAssessment 构造缺
  `event_applicability`）修复——首次运行即暴露，测试全绿。

## 修复纪律核验

- 原始 prompt SHA256：8/8 与外部溯源一致（回归测试 7，每次运行重验）；
- v3.1 输出冻结未动；v1 审计表/sidecar 保留；
- 新输出：`prompt_source_generation_audit_v2.csv`（P0-5 13 字段 +
  P0-2/3 新字段 + conditioning 结论列）、
  `prompt_correction_sidecar_v2.csv`；
- 数值实验独立目录：`runs/e2_1_conditioning/`（P0-4 帧相似度证据）。
