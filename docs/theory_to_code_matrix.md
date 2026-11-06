# 理论—代码对应矩阵（E2.1 文献学习）

日期：2026-11-05 | 纪律：**只为已有实验需求引入理论**——每篇论文
必须回答"借鉴什么、不适用什么、对应哪段代码、设计什么对照"，
禁止为深度而复杂化。

---

## 1. DrivingGen（ICLR 2026，arXiv:2601.01528）

| 项 | 内容 |
|---|---|
| 研究问题 | 生成式驾驶世界模型的综合评价（14 模型 × 质量/一致性/轨迹维度） |
| 核心假设 | 世界模型质量可分解为感知质量 + 轨迹物理合理性 + agent 一致性，且可由自动指标度量 |
| 关键公式 | vel/accel 一致性 `exp(-σ_v/μ_v)` 型；FTD（MTR 特征 Fréchet 距离）；R/A DINOv3 余弦 |
| 可借鉴算法 | ① 轨迹一致性公式（已移植）；② Agent Missing 的"消失可解释性"任务定义（VLM 裁决换几何归因）；③ 全场平均的反面教材（事件级评价的动机） |
| 对应代码位置 | `wmeval/traj_metrics.py`（公式）、`wmeval/uncertainty.py`（Agent Missing 的几何化重实现）、`docs/drivinggen_baseline_analysis.md`（对齐口径） |
| 不适用假设 | ① 32GB+ VLM 裁决可复现（本机不可行）；② 世界系轨迹可从 nuplan GT 内参获得（SynPeDAS 无内参）；③ 全场平均足以评价安全关键事件（事件被淹没——研究缺口 2） |
| 对照实验 | B1-vs-B3：同一批视频，官方公式口径（px）vs 事件协议口径的 missing/异常率差异（e2_experiment_plan §4） |

## 2. OC-SORT（CVPR 2023，观察中心 SORT）

| 项 | 内容 |
|---|---|
| 研究问题 | 遮挡期间卡尔曼状态估计漂移导致 SORT 类跟踪 ID 错误 |
| 核心假设 | 长时隙关联应依赖**累积的观察**（observation-centric）而非漂移的隐状态（state-centric） |
| 关键公式 | 虚拟轨迹（unassociated observations 前向外推）；OCR（observation-centric re-update）：关联成功后用虚拟轨迹重置协方差；consistency-aware anti-drift |
| 可借鉴算法 | ① 观察缓冲：把遮挡期的检测存入虚拟轨迹用于长隙关联——**actor_binding 的下一步**：当前链构造用尾部速度外推（state-centric），可升级为"遮挡期全部 person 检测的前向累积"（OC-SORT 核心）；② 关联一致性度量可作 ambiguity 判定的第三证据 |
| 对应代码位置 | `wmeval/actor_binding.py::build_track_chains`（外推关联，即其 state-centric 对立面——升级路径已注释）；`uncertainty.py::attribute_endings`（case-b 候选=简化观察缓冲） |
| 不适用假设 | ① 竞技 MOT 场景的高帧率稠密检测（SynPeDAS gen 侧检测稀疏且碎）；② 卡尔曼滤波状态（我们无滤波器，直接用检测几何） |
| 对照实验 | E2 消融增补：actor_binding v0（外推）vs v1（+观察缓冲）——碎片链合并正确率（以 0016 类人工确认为真值）；判据预注册：合并正确率 +0 且歧义误降不升 |

## 3. Robust Online Monitoring of STL（2017）

| 项 | 内容 |
|---|---|
| 研究问题 | 信号时序逻辑（STL）公式的**在线**监控：前缀信息下判定当前满足/违反/未定 |
| 核心假设 | 鲁棒度 ρ(φ,σ,t)（语义距离，正=满足裕度）可在线增量计算；输出三态：已满足/已违反/**未知**（前缀不足以判定） |
| 关键公式 | ρ 的 min/max 组合语义（定量语义）；监控器的在线前缀计算 |
| 可借鉴算法 | ① **前缀三态语义正是 Valid/Invalid/Uncertain 的形式化基础**——"证据不足时未定"不是工程妥协而是监控理论的标准输出；② 定量鲁棒度→事件裕度：region_presence 的裕度（离边界距离）可量化"多接近满足"；③ 分段监控（clip_end_cut = 前缀截断不可判的形式化） |
| 对应代码位置 | `wmeval/crossing_evidence.py`（L4 状态枚举=STL 三态的事件化：completed/in_progress/partial_clip_cut）；`uncertainty.py::adjudicate`（三态判定）；`docs/crossing_event_semantics.md`（状态语义表） |
| 不适用假设 | ① 连续信号域（我们的是离散检测事件流）；② 形式化时序逻辑语法（事件协议用语义分层而非 STL 公式——引入完整 STL 语法无实验必要） |
| 对照实验 | 合成轨迹上事件状态 vs STL 鲁棒度的一致性测试：构造 20 条合成轨迹（完成/中途/截断/静止），验证 L1-L4 输出与手工 STL 判定一致——**E2.1 已部分实现**（selftest_e2 的 crossing 用例即其子集，可扩展为正式一致性表） |

## 4. Selective Classification（Geifman & El-Yaniv，NeurIPS 2017）

| 项 | 内容 |
|---|---|
| 研究问题 | 分类器配备拒识（reject option）时的理论保证：覆盖率与风险的关系 |
| 核心假设 | 拒识函数 g 与分类器 f 联合，selective risk R(f,g)=E[loss·g]/E[g] 随覆盖率单调可 tradeoff |
| 关键公式 | R(f,g) 定义；Softmax Response 拒识（SR：max softmax 为置信度）；coverage-risk 曲线（AUC 比较拒识策略） |
| 可借鉴算法 | ① **三态判定的理论框架**：Uncertain=拒识；IAR/VRR=selective risk/coverage 的实例化——"固定放行率下比较风险"（e2_experiment_plan §4.2 硬约束）正是 coverage-matched risk 比较；② SR 的启示：几何归因置信度（如 pred_err_norm、检测 mean_conf）可作拒识分数，扫掠得风险-放行曲线 |
| 对应代码位置 | `wmeval/uncertainty.py`（弱证据归因=拒识信号源）；`docs/e2_experiment_plan.md` §4.1-4.2（Selective Risk/自动放行覆盖率指标与防刷指标协议——理论的直接实例化） |
| 不适用假设 | ① 拒识与错误代价独立（我们的 Uncertain 本身有复核成本）；② i.i.d. 测试集（20 样本开发集有泄漏历史——split 协议已处理）；③ 类平衡（invalid 极稀少——召回评估需外部异常源） |
| 对照实验 | B0（无拒识）vs B3（拒识）的 **coverage-risk 曲线**而非单点比较（Val 真值上）；AUC 与固定覆盖率 0.5/0.7 处的风险差双报告——已预注册于 e2_experiment_plan |

## 5. 矩阵总结（引入/不引入决策）

| 论文 | 引入状态 | 理由 |
|---|---|---|
| DrivingGen | **已实现**（公式+对齐口径） | B1 基线必需 |
| OC-SORT | **设计就绪，实现留 v4 可选** | actor_binding 升级路径明确，但当前碎片链测试全绿——等真实标注暴露合并错误再引入（判据预注册） |
| STL 监控 | **语义已实例化**（三态+截断语义），完整语法不引入 | L1-L4 即其事件化子集；完整 STL 无实验必要 |
| Selective Classification | **指标协议已实例化**（coverage-risk 硬约束） | B0-B3 消融的理论框架；无需实现新算法 |

**四篇均不新增代码模块**——理论与现有实现的对应关系已全部落位，
进一步形式化（STL 语法/OC-SORT 完整算法）均有预注册的触发条件
（真实证据暴露对应缺陷时）。
