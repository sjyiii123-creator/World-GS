# JEPA 论文原文核验与迁移判定（AD-E2E-JEPA / WA-JEPA）

日期：2026-11-06 | 核验版本：AD-E2E-JEPA（arXiv 2609.34085v1，2026-09-28）、
WA-JEPA（arXiv 2608.20974v2，2026-09-05）| 本文档为 E3 阶段的论文依据基准

## 0. 论文定位原则（所有迁移建议的前提）

> **World-GS 要证明的不是 JEPA 能生成更好的未来，而是 JEPA 提供的事件时序
> 表示能否改善生成视频的条件满足性验证。**

## 1. 三条不可直接作为论文依据的修正

1. **AD-E2E-JEPA 的潜在规划能力 ≠ 已学会判断安全事件条件是否满足。**
   它用真实未来图像作为规划目标、通过潜在距离选择自车轨迹；可启发
   World-GS 的条件对照设计，但并未验证行人停止/恢复横穿等事件。
   其训练目标仍是潜在未来预测（+SIGReg 约束），规划阶段才用潜在距离
   选轨迹——"预测空间可用于目标选择"是事实，"已接受安全事件判别监督"
   不是事实。
2. **WA-JEPA 的时间变化诊断值得优先借鉴，但 D4 的 Participation Ratio
   下降不能直接推出时间过度平滑**——还需要逐时间步预测 vs 真实目标的
   动态变化证据（E3-D0 的任务）。
3. **"两篇论文都用 Teacher Forcing + AR Rollout，因此必须加 Rollout
   Loss"不准确。** AD-E2E-JEPA 的 Rollout 是**可选训练变体**；WA-JEPA
   用未来掩码 + 条件 Flow Matching，没有相同组合。Rollout 是待检验的
   实验假设（可做消融），不是已证明的必选组件。另：自由 Rollout 评价
   （测误差积累）不需要先实现 Rollout Loss 训练。

## 2. 数值与配置修正表（引用论文时以此为准）

| 项 | 曾用（错误） | 原文核验（正确） |
|---|---|---|
| WA-JEPA Full-mask / Patch-mask / 混合 | 89.0 / 91.0 / 91.7 | **91.3 / 91.0 / 91.7**（Table 4(b)） |
| WA-JEPA 优化器 | 统一 lr=1e-4, wd=0.01 | **分模块不同 lr, wd=0.04** |
| WA-JEPA 训练资源 | "完整 32×A800" | **Stage 1: 64×A800；Stage 2: 32×A800** |
| WA-JEPA HUGSIM HD-Score | 0.462/0.466 | **0.4462**（v2） |
| WA-JEPA 十次随机推理 std | ~0.04 | **0.0531 EPDMS** |
| WA-JEPA 编码器消融 | — | V-JEPA 2 EPDMS 89.5 > DINOv3 83.8（**无通用最优骨干**） |
| AD-E2E-JEPA navtrain | 4×A100、8 小时 | **1×A100、20 小时** |
| AD-E2E-JEPA navtrain+Rollout | 1×A100 | **1×A100、约 46 小时** |
| AD-E2E-JEPA trainval | 4×A100、2–4 天 | 正确（约 53–98 小时） |
| WA-JEPA 官方仓库 | AFAIR-Lab/OpenWA-JEPA | **AFARI-Research/WA-JEPA** |
| AD-E2E-JEPA 官方仓库 | — | HaoranZhuExplorer/AD-E2E-JEPA |

注意：WA-JEPA 的 `EPDMS*` 与 `EPDMS` 涉及评价协议修正，论文要求区分，
不能混用——与本项目 D4.1.1 的教训同构：**模型能力变化与评价协议变化
必须分别记录**。

## 3. 两个可移植的时间诊断（E3-D0 的依据）

$$\Delta_{\mathrm{dir}} = C(\hat z) - C(z) \qquad
R_\Delta = \frac{D(\hat z)}{\max(D(z),\epsilon)}$$

- $C$：序列内不同时间步之间的平均余弦相似度（方向集中度）；
- $D$：相邻时间步的平均特征变化幅度；
- $\hat z$：预测未来序列；$z$：真实未来序列。
- $\Delta_{\mathrm{dir}}$ 显著为正 → 预测的未来各时间步比真实更相似
  （时间方向坍缩）；$R_\Delta \ll 1$ → 预测序列变化幅度不足。

**区分两个此前混淆的现象**：身份表征集中（不同 Actor 的预测特征趋同，
J2.3 已观察到）vs 时间动态衰减（同一 Actor 不同时间的预测变化趋弱，
**尚未由本项目数据证明**——E3-D0 回答）。

**协议限制**：WA-JEPA 的原始诊断在多帧、空间 Token 上进行，Flow
Matching 用训练流时间下的单步 clean-latent 预测。World-GS 的 384 维
池化特征 + 确定性预测器不能照搬其协议数值，只能做同协议内的
预测 vs 真实对照。

## 4. 迁移判定总表

| 技术思想 | 原文依据 | 判定 |
|---|---|---|
| 面向未来的因果预测（Full-mask） | WA-JEPA | **GO：优先采用** |
| 事件相关潜在状态 | 两篇论文任务对齐思想（事件定义是我们的扩展） | **GO：优先设计** |
| 时间变化诊断（Δ_dir、R_Δ） | WA-JEPA | **GO：先做诊断（E3-D0）** |
| 密集 ROI Patch 表示 | AD-E2E-JEPA Patch 预测与投影 | **CONDITIONAL-GO：先轻量消融** |
| Teacher-Forcing + Rollout | AD-E2E-JEPA 可选 Rollout | **CONDITIONAL-GO：必要时消融** |
| SIGReg | AD-E2E-JEPA | **暂缓：先证明投影器退化** |
| Hybrid Future Masking | WA-JEPA | **暂缓完整复现：先确保 Full-mask 有效** |
| Conditional Flow Matching | WA-JEPA | **暂缓：先证明单未来预测有系统性局限** |
| Joint World–Action Predictor | WA-JEPA | **暂缓：自车动作与行人事件不是同一监督对象** |
| 完整预训练大模型 | 两篇论文原始配置 | **NO-GO：当前无必要性** |

## 5. 目标分离原则（E3 方法设计约束）

World-GS 需要两类目标，**不能合并为单一预测残差**：

- **目标 A（学习）**：$\hat h^{event}_{t+\Delta} = P_\theta(H_{\le t})$ —— 由观察
  历史能预测怎样的事件状态表示；
- **目标 B（验证）**：$v = V(C, O_{1:T}, U) \in \{PASS, FAIL, UNKNOWN\}$ ——
  生成视频是否真实出现了条件要求的事件（$C$ 条件、$O$ 观测证据、
  $U$ 不确定性）。

反例：模型预测"继续行走"最可能，但生成条件要求"突然停止"且视频中行人
真的停止了 → 条件验证应 PASS，不能因潜在预测不一致而拒绝。

**JEPA 提供时序状态证据；事件验证器依据独立的实际观测与生成条件作
最终判断。** 观测侧状态提取尽量独立于生成条件文本（防条件确认偏差），
条件与观测只在最终验证层比较。

## 6. 下一步（详见 docs/e3_event_state_jepa_plan.md）

借鉴顺序：

> WA-JEPA 未来预测目标 → 时间变化诊断（E3-D0）→ AD-E2E-JEPA 局部
> 表示 → Event-State JEPA（E3-B，门控）→ 事件级风险验证

**不先复制 Flow Matching、SIGReg 或完整规划网络。先证明事件状态的
预测表示比当前外观特征更有用，才是本阶段的破局点。**
