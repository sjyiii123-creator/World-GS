# J2.3 Visibility-Aware Memory 结果（G3）

日期：2026-11-05 | 代码：
`src/wmeval/latent_world_state/visibility_memory.py`
（契约：不伪造观测）+ `src/run_j2_3_visibility.py` →
`runs/j2_3_diagnostics/visibility_v0_v3.json`
首轮可见性信号：**JAAD 官方遮挡标签**（生成侧不可用其作真值——
边界已声明）。

## 1. 契约实现（任务 §六）

- ReferenceMemory：reference_feature / reference_timestamp /
  current_timestamp / **reference_age_seconds** /
  visibility_quality / memory_update_status / skip_reason /
  identity_status（默认 uncertain，永不自动确认）；
- **跳过更新时**：参考特征/时间戳保持旧值（旧观测=旧证据），
  当前时间与年龄照常推进——**观测状态按真实视频填写，不复制
  旧观测冒充新观测**（示例记录落盘于 json.memory_records_sample）。

## 2. V0–V3 对照（v2 val 60 序列，同集内对照）

| 策略 | 参考 | positive cos | hard cos | **M_hard** |
|---|---|---|---|---|
| V0 末帧（现状） | 最后一帧 | 0.734 | 0.557 | 0.074 |
| V1 最近可见帧 | last reliable | 0.738 | 0.558 | 0.070 |
| **V2 可见帧均值** | multi-reliable | 0.765 | 0.578 | **0.085** |
| V3 质量加权 | weighted | 0.772 | 0.579 | **0.085** |

## 3. 结论（H3 裁决）

1. **V1≈V0**：val anchor 的末帧大多可见（JAAD 段构造使然）——
   **"最后一帧非最佳参考"的遮挡污染假设在此数据上证据弱**；
2. **V2/V3 轻量正效应**（M_hard +0.011、positive +0.03）：
   多帧聚合 > 末帧——与 J1.1 遮挡策略 B 的结论一致（聚合优于
   单帧），但幅度小；
3. **优先级裁决**：H3 是次要因素——Visibility Memory 保留为
   可选轻量模块（不训练、可审计），**不进入下一轮主训练**。
