# 公开基线数据集调研与扩展计划

日期：2026-11-04（2026-11-05 E2 修订：补齐三个已确认基准的实测分析，
重定位 JAAD/DoTA 的用途命名；下载计划暂缓，优先 E2）
目的：为 E2–E3 之后的**独立泛化验证**与**真实异常召回测试**选定学术界
公认的公开数据集。本文件为调研结论，**未下载任何新数据**（E2 决定：
先完成 E2，再议 A/B 下载）。

事实来源：官方页面/GitHub/论文（2026-11-04/05 联网核实，来源链接见 §7）；
未能联网核实的项均显式标注"待核实"。引用数为检索结果所示，仅供参考。

---

## 0. 选择标准与角色映射

| 角色 | 需求 | 对应实验 |
|---|---|---|
| R1 真实侧事件素材 + 官方 GT | 行人横穿行为标注、车载第一视角 | E2 扩样、GT 弱监督、泛化测试 |
| R2 真实安全关键/异常事件 | 事故/异常片段 + 时间/空间/类别标注 | **Invalid 侧召回验证**（v3.1 只有 1 个真 invalid，无法评估召回） |
| R3 生成侧评测对 | 真实↔生成配对 | 已有 SynPeDAS 211 组（本地） |
| R4 基线/文献口径对齐 | 官方评测集 | B1（DrivingGen benchmark）、B0（视频质量） |
| R5 受控注入 | 可控场景合成 | E3 人为注入异常协议 |

硬约束：RTX 5060 Ti 16GB / 磁盘可用 638G / 公开可下载 / 学术界认可
（顶会发表 + 后续工作广泛用作基准）。

**当前数据短板**（v3.1 后的实证结论）：
1. v3.1 仅 1 个 invalid（0024）且协议依据待确认 → **无法评估 Invalid 召回率**；
2. 211 组 SynPeDAS 是唯一评测对 → 规则阈值存在**过拟合风险**（用户第七节已预警）；
3. 人工标注 20 条样本真值即将完成，但**缺独立测试集**验证泛化。

---

## 1. 推荐总表（按优先级排序）

| # | 数据集 | 角色 | 规模（核实状态） | 磁盘 | 优先级理由 |
|---|---|---|---|---|---|
| 1 | **JAAD**（完整版） | R1 | 346 段 1080p 车载视频（5–10s）✅ | 数 GB（待核实） | SynPeDAS 真实侧的**同源母数据集**（crossing_moments 即其子集）；官方自带横穿行为标注 → 直接充当弱 GT，补齐 211 组之外的行为真值。**用途命名（E2 修正）**：held-out 片段实验是**同源未见场景泛化**（same-source unseen-scene generalization），不是跨域泛化 |
| 2 | **DoTA** | R2 辅助 | 4,677 段行车记录视频，18 类异常 ✅ | 待核实（约 10GB 级） | **用途重定位（E2 修正）**：DoTA 标注的是真实交通异常，**不是**"生成视频违反给定生成条件"的真值——不能直接当 Invalid 样本。仅作为候选辅助实验：真实异常检测、时空事件定位、跨域鲁棒性。暂不下载完整集 |
| 3 | **CARLA** + ScenarioRunner | R5 | 开源仿真器（MIT） | ~20GB | 受控注入协议（行人消失/延迟/ID 断裂可精确合成）+ 独立于任何数据分布 |
| 4 | **PIE** | R1 | 6h 视频，1,842 名行人，OBD 自车运动 ✅ | ~74GB | JAAD 同团队后继（ICCV 2019，检索示 736+ 引用）；扩样到 50+ 组后的升级路径；自车速度可改善"出画 vs 消失"判别 |
| 5 | **DADA-2000** | R2 | 2,000 段事故视频，54 类 ✅ | ~53GB（压缩包） | 事故类别+时间窗+空间标注；与 DoTA 二选一或互补 |
| 6 | **DrivingGen benchmark** | R4 | nuPlan 来源+14 模型生成视频（ICLR 2026）✅ | 需 HF token | B1 官方口径对齐的正式途径（此前 P-8 阻塞项） |
| 7 | nuScenes（mini） | R1/R4 备选 | 1000 场景（全量）；mini 子集 | mini ~4GB（经验值，待核实） | 3D 框+传感器套件，E4 世界系轨迹扩展的可选素材；非必需 |

不建议本轮引入：Waymo（许可注册+非商用条款较严）、BDD100K（体量大但无事件级 GT）、
KITTI（视角/任务不匹配，仅检测器标定时考虑）。

---

## 2. 数据集详情卡

### 2.1 JAAD（Joint Attention in Autonomous Driving）— 首选 R1

- **出处**：Rasouli, Kotseruba, Tsotsos, ICCVW 2017；York University
  （[官方页面](https://data.nvision2.eecs.yorku.ca/JAAD_dataset/)，
  [GitHub ykotseruba/JAAD](https://github.com/ykotseruba/JAAD)）。
- **内容**：346 段高清车载视频（5–10s，1080p@30，城市环境），行人包围框
  + 横穿行为（crossing/not-crossing）+ 动作 + 遮挡 + 属性标注。
- **学术认可**：行人意图/横穿预测领域的标准基准之一；PIE 的前作、同一团队。
- **与本研究关系（E2 修正定位）**：SynPeDAS 的 `JAAD_crossing_moments/`
  （211 组）即从 JAAD 抽取的横穿时刻片段——**完整 JAAD 与现有管线零格式
  适配成本**（同为 1080p30 ego-view）；官方行为标注可直接用作
  `event_satisfied` 的人工标注交叉校验（弱 GT）。
  **命名纪律**：held-out JAAD 片段的实验 = **同源未见场景泛化**
  （same-source unseen-scene generalization）。称为"跨域泛化"是错的
  （同一采集协议/同一城市分布/同一传感器族）。
  **泄漏防护**：划分必须按源视频分组（已在
  `src/build_e2_split.py` 实现：166 个源组整组同侧，E1 污染的 10 个
  源视频锁死在 dev）。若未来引入 JAAD 完整版，还需按 JAAD 官方场景/
  采集日分组做二次校验。
- **获取**：官方页面直接下载；标注在 GitHub（JAAD_2.0 分支）。
  视频总体积待下载前核实（346×~7s 1080p 估计数 GB 级）。
  小体积标注文件（GitHub 上）可先检查获取方式，**实际下载前确认
  许可、规模与必要性**（E2 纪律：先 E2 后下载）。
- **风险**：低。许可为研究用途。

### 2.2 DoTA（Detection of Traffic Anomaly）— R2 辅助（E2 重定位）

- **出处**："When, Where, and What? A New Dataset for Anomaly Detection
  in Driving Videos"（arXiv:2004.03044）；期刊版 TPAMI 2023（检索示
  ~224 引用）；University of Michigan。
  [GitHub MoonBlvd/Detection-of-Traffic-Anomaly](https://github.com/MoonBlvd/Detection-of-Traffic-Anomaly)。
- **内容**：4,677 段行车记录视频；时间边界（异常起止）、空间标注、
  类别（18 类）、自车卷入标志。
- **与本研究关系（E2 修正定位）**：DoTA 的标注是**真实交通异常**，
  不是"生成视频违反其生成条件"的真值——**不能直接作为本研究的
  Invalid 样本**（把真实事故视频塞给验证器判"生成失败"是类别错误）。
  合法用途仅限三个候选辅助实验：
  ① 真实异常检测（验证器对真实世界异常事件的一般检出能力）；
  ② 时空事件定位（其时间/空间边界标注可校准 interrupt_region 与
  时间窗设计）；③ 跨域鲁棒性（YouTube 行车记录 vs JAAD 分布的
  迁移测试，结果单独报告为跨域实验）。
  **当前不下载完整 DoTA**（E2 决定）；若未来需要，先抽小样本试点。
- **获取**：GitHub（通常经 Google Drive）。体积待核实。

### 2.3 CARLA + ScenarioRunner — R5

- MIT 许可开源仿真器；ScenarioRunner 提供标准场景 DSL。
- **与本研究关系**：E3"人为注入异常协议"的受控实现——
  行人中途消失、遮挡、延迟出现、ID 断裂均可精确合成并**自带真值**，
  解决"注入实验必须与真实失败分开报告"的可信度问题（计划 §E3）。
  同时天然构成第三条独立分布（非 JAAD、非 YouTube）。
- **资源**：单卡 16GB 足够跑 CARLA 服务器+采集脚本；~20GB 安装。
- **替代方案**（更轻）：在现有 JAAD 片段上做像素级/检测级注入
  （删帧+inpaint、box 抹除）——计划中已有该协议；CARLA 是其升级版。

### 2.4 PIE — R1 扩样升级（74GB）

- **出处**：Rasouli et al., ICCV 2019（检索示 736+ 引用）。
  [官方页面](https://data.nvision2.eecs.yorku.ca/PIE_dataset/)，
  [GitHub aras62/PIE](https://github.com/aras62/PIE)。
- **内容**：多伦多 6 小时车载视频（10 分钟一段、6 组路线），
  1,842 名行人 2D 框 + 横穿动作 + 意图（人类被试标注）；
  **自车 OBD 运动**（速度/加速度/航向）。
- **与本研究关系**：① 扩到 50+ 组行人事件样本的天然池；
  ② 自车速度使"行人出画（自车前进）vs 行人消失（生成异常）"
  的判别有物理依据——直接强化 uncertainty 归因链；
  ③ 与 JAAD 同团队同分布族，管线零适配。
- **资源**：视频全量 ~74GB（638G 磁盘无压力）；YUV/统一分辨率需确认。

### 2.5 DADA-2000 — R2 互补

- **出处**：Fang et al., ITSC 2019（arXiv:1904.12634，检索示 ~156 引用）。
  [GitHub JWFangit/LOTVS-DADA](https://github.com/JWFangit/LOTVS-DADA)。
- **内容**：2,000 段事故视频（~65.8 万帧），54 类事故；
  事故时间窗 + 碰撞物空间位置 + 驾驶员注意力 GT。
- **与本研究关系**：与 DoTA 同为 R2，但偏中国路况行车记录、
  事故密度更高；若 DoTA 覆盖不足可作为补充。
- **风险**：官方链路为百度网盘（提取码 9pab，~53GB）——下载速度不可控；
  OpenDataLab 有镜像。**列为 DoTA 之后的备选**。

### 2.6 DrivingGen benchmark — R4（B1 对齐）

- **更新**：DrivingGen 已发表于 **ICLR 2026**（arXiv:2601.01528，
  [GitHub](https://github.com/youngzhou1999/DrivingGen)）——
  本地仓库 HEAD `48ed356` 的复现目标因此升级为"已录用论文的官方基准"，
  复现价值上升。
- **HF 数据实测预检（2026-11-05）**：
  [yangzhou99/DrivingGen](https://huggingface.co/datasets/yangzhou99/DrivingGen)
  **非 gated、Apache-2.0、2.24GB、imagefolder 格式 ~22.9k 行**
  （image + label 字段，224 个 label 类）。注意：**此前 P-8 "需 token"
  的记录已过时**；且 imagefolder/224-class 的形态与"评价输入+条件"
  的描述存在出入——**下载后需实际核查文件树再定性**（可能是按类别
  组织的输入帧集合）。
- **与本研究关系**：B1 基线的官方口径；拿到后可直接把 wmeval 的
  px 空间指标与世界系官方数值对齐（E4 的 UniDepth 路线仍为备选）。

### 2.6b SafeMVDrive / SMD（E2 方案新增，R3' 生成侧第二数据源）

- **出处**：ICML 2026（[PMLR v306/zhou26aj](https://proceedings.mlr.press/v306/zhou26aj.html)）；
  代码 [github.com/WAVE-AT-IDEA/SMD](https://github.com/WAVE-AT-IDEA/SMD)。
- **HF 数据实测预检（2026-11-05）**：
  [JiaweiZhou/SafeMVDrive](https://huggingface.co/datasets/JiaweiZhou/SafeMVDrive)
  **非 gated、CC BY-SA 4.0、7.97GB、nuScenes 格式**；41 场景 × 9s ×
  6 相机视角 + 车辆 3D Bounding Box；构建自 250 个 nuScenes 验证集
  样本，经 VLM 引导的对抗 crash-then-evasion 轨迹合成 + UniMLVG
  视频生成器。
- **与本研究关系**：SynPeDAS（行人横穿）之外的**第二个生成侧数据源**
  ——危险车辆交互场景，**自带明确的生成条件**（预设危险轨迹）：
  预设危险参与者是否保留、关键车辆是否消失、运动是否与目标轨迹
  一致、危险交互是否被弱化——正是安全关键事件保持性研究的直接
  测试材料。**无现成 Valid/Invalid/Uncertain 真值**（评价协议仍需
  自建，与 SynPeDAS 同理）。
- **注意**：基于 nuScenes 构建，使用需同时核对 nuScenes 来源协议；
  下载后先核查文件树（HF viewer 报 schema mismatch，不影响文件下载）。

### 2.6c DriveJudgeBench 数据（R4 对照，暂缓下载）

- HF：[SeagullofLeman/DriveJudgeBench](https://huggingface.co/datasets/SeagullofLeman/DriveJudgeBench)
  （1,597 段 / 7,371 QA）。按用户方案：先克隆代码研究，数据待确认
  所需子集后再下。**没有**"生成结果是否违反 prompt 指定事件"的完整
  真值——仅作对照数据。

### 2.7 文献口径类基准（R4 讨论用，不必下载）

- **WorldSimBench**（ICML 2025，arXiv:2410.18072）：生成视频的
  显式感知（物理一致性）+ 隐式操作一致性双路评测。
- **WorldModelBench**（NeurIPS 2025，arXiv:2502.20694）：350 组
  图像/文本条件对，检验"视频生成模型是否是世界模型"。
- **PhyWorldBench**（NVIDIA）/ **PhyGenBench**（OpenGVLab, ICML 2025）：
  物理常识正确性评测。
- **VBench**：通用视频生成多维评测（B0 的候选工具链）。
- **用途**：论文 Related Work 与评测维度的定位参照；证明"事件级时空
  约束 + 不确定性归因"在现有基准谱系中的空位（现有基准偏整段
  质量/物理，无一提供事件级 Valid/Invalid/Uncertain 三态协议）。

### 2.8 已确认存在的三个基准（E2 实测分析，2026-11-05）

此前"未能核实"的三项已由用户确认并联网核实。**逐项对比评价对象、
指标定义与实验协议**（不以"没用三态标签"论创新）：

#### WorldLens（[github.com/worldbench/WorldLens](https://github.com/worldbench/WorldLens)，CVPR 2026 Oral，arXiv:2512.10958）

- **评价对象**：多视角驾驶世界模型（MagicDrive/Panacea/DreamForge/
  DriveDreamer-2/DrivingSphere 等，nuScenes 生态），5 大任务
  （Generation/Reconstruction/**Action-Following**/Downstream Task/
  Human Preference）、24 个维度。
- **与本研究六问的交集**：驾驶场景 ✓；时空约束 ✓（temporal depth
  consistency、cross-view consistency、4D 重建）；**安全性已涉及**——
  Action-Following 闭环测试报告 collision/off-road drift 失败，
  Human Preference 维度含行为安全；**生成条件遵循**部分（动作条件，
  非文本事件条件）。
- **未覆盖**：事件级验证无命名指标；**不确定性评价无**。
- **结论**：WorldLens 是**最近的强相关工作**。差异点必须落在：
  (a) 单目 ego 视频（SynPeDAS/Cosmos 类）而非多视角 nuScenes 生态；
  (b) per-event/per-actor 的事件级协议（L1-L4 状态分层）；
  (c) 三态不确定性及其放行-风险权衡曲线。**安全评价本身不能宣称
  为空白**——WorldLens 的闭环安全失败检测是必须引用和对照的先例。

#### WorldReasonBench（[github.com/UniX-AI-Lab/WorldReasonBench](https://github.com/UniX-AI-Lab/WorldReasonBench)，NeurIPS 2026，arXiv:2605.10434）

- **评价对象**：通用视频生成的"世界状态预测"能力，436 个结构化 QA
  用例（World-Knowledge/Human-Centric/Logic-Reasoning/Information 四维），
  VLM-as-judge 两段协议（VLM 答题 + LLM 判卷，AccQA/ScorePR/ΔRG）。
- **与六问交集**：状态演化一致性部分（phase-aware QA）；
  条件遵循部分（expected reasoning elements 检查）。
- **未覆盖**：**无驾驶类目**；不确定性无；安全无；时空约束无显式任务。
- **结论**：方法论参照（把评价重构为"世界状态一致性"的思路与本工作
  的事件状态分层同源），但域和目标不重叠。

#### DriveJudge / DriveJudgeBench（[github.com/vita-epfl/DriveJudge](https://github.com/vita-epfl/DriveJudge)，暂未见正式发表）

- **评价对象**：**VLM 充当生成驾驶视频裁判的能力**（不是世界模型
  本身）：1,597 个合成驾驶片段 + 7,371 个视频-问答对，六类
  （reality detection / artifact recognition / **safety assessment** /
  **traffic-law compliance** / **spatio-temporal reasoning** / visual
  understanding）；被评的是 Gemini/GPT/Qwen3-VL/Cosmos-Reason 等
  裁判模型。
- **与六问交集**：安全评估 ✓（VQA 形式）；时空推理 ✓；驾驶专属 ✓。
- **未覆盖**：**prompt/条件遵循明确不做**（README 明示只做
  real-vs-generated）；事件级一致性无专用检测（temporal behavior 以
  VQA 形式）；**不确定性无**。
- **结论**：它评"裁判"，本研究建"确定性验证器"——互补而非竞争；
  其六类目可作为错误类型分类学的对照参考；其结论（最强 VLM 裁判
  准确率 ~64%）恰好支持"纯 VLM 裁决不可靠/需证据链"的立论。

#### 修正后的研究空位表述（供 research_gap.md 引用）

已被覆盖：驾驶生成视频的整体质量（WorldLens）、闭环行为安全
（WorldLens Action-Following）、安全/法规 VQA（DriveJudge）、世界状态
一致性 QA（WorldReasonBench）。
**仍空缺（据三者公开协议）**：① per-event/per-actor 的事件级一致性
验证协议（L1-L4 分层 + 单链归因）；② 三态不确定性判定及其
自动放行率-风险权衡曲线（三者均无不确定性输出）；③ 文本事件条件的
协议化三值映射（event_required: true/false/unknown）。
上述空位**是候选贡献，不是已证实的优势**——价值取决于 E2 消融能否
在固定放行率下降低 IAR（e2_experiment_plan.md §4.2）。

---

## 3. 论文实验的数据组成建议（映射到 E 系列）

| 实验 | 数据 | 目的 | 磁盘 | 状态 |
|---|---|---|---|---|
| E2 协议确认+真值+消融 | **现有 SynPeDAS 211 组**（dev/val/test 划分已建） | 条件层/状态层/判定层证据链 | 0 | **进行中（当前优先）** |
| E8' 同源未见场景泛化 | JAAD 完整版 held-out 片段（211 组之外的源视频） | 阈值/规则在**同源未见场景**的泛化（E2 修正命名：非跨域） | 数 GB | 待批准（E2 后） |
| E7' 真实异常辅助实验 | DoTA 小样本试点 | 真实异常检测/时空定位/跨域鲁棒（**非 Invalid 真值**） | 小样本 | 候选（E2 后） |
| E3 注入升级(可选) | CARLA + ScenarioRunner | 受控合成注入异常 | ~20GB | 待批准 |
| 扩样 50+(如需) | PIE(74GB)或 DADA-2000(53GB) | 扩样/R2 互补 | 53–74GB | 待批准 |

E2 用现有 211 组 SynPeDAS 即可，**不依赖任何新下载**。
DrivingGen benchmark 已确认发表于 ICLR 2026（arXiv:2601.01528），
B1 官方对齐仍需用户提供 HF token（P-8）。

---

## 4. 分阶段下载计划（**整体暂缓：优先完成 E2**，2026-11-05 用户决定）

| 阶段 | 数据 | 预估磁盘 | 依据 | 状态 |
|---|---|---|---|---|
| A | JAAD 完整版 + 标注 | ~数 GB（下载前核实） | E8' 同源未见泛化 + 弱 GT | 暂缓 |
| B | DoTA | ~10GB 级（待核实） | E7' 真实异常辅助 | 暂缓 |
| C | CARLA 0.9.x | ~20GB | E3 注入升级 | 暂缓 |
| D（择一） | PIE（74GB）或 DADA-2000（53GB） | 53–74GB | 扩样 / R2 互补 | 暂缓 |
| E | DrivingGen benchmark | 未知（HF token 后核实） | B1 对齐 | 需 token |

磁盘基线：638G 可用，A–C 合计 <40G，全部引入仍有充足余量。
**JAAD 小体积标注文件**（GitHub JAAD_2.0 分支）可先行检查获取方式
（许可/规模），实际下载前确认必要性。

## 5. 学术认可度说明

- 上表引用数（PIE 736+、DoTA ~224、DADA ~156、WorldSimBench ~149）
  为检索结果所示，非精确统计；
- "认可"判定依据：顶会/顶刊发表 + 后续模型论文普遍用作训练/评测基准
  （PIE/JAAD 为行人意图预测两大标准库；DoTA/DADA 为驾驶异常检测标准库；
  DrivingGen 为 ICLR 2026 世界模型评测基准）；
- SynPeDAS 本身作为生成对数据集是本论文的核心评测素材（211 组，
  本地已具备）。

## 6. 风险与许可速记

- JAAD/PIE：研究用途许可（York 大学条款）；
- DoTA：公开研究许可（GitHub）；
- DADA-2000：研究用途（百度盘分发，速度风险）；
- CARLA：MIT；
- DrivingGen HF：需 token，许可随 HF repo；
- 所有数据仅用于本论文离线评测，无再分发计划。

## 7. 来源链接（2026-11-04 核实）

- JAAD 官方: https://data.nvision2.eecs.yorku.ca/JAAD_dataset/ ;
  GitHub: https://github.com/ykotseruba/JAAD ;
  论文: https://arxiv.org/html/1609.04741v6
- PIE 官方: https://data.nvision2.eecs.yorku.ca/PIE_dataset/ ;
  GitHub: https://github.com/aras62/PIE ;
  论文(ICCV 2019): https://openaccess.thecvf.com/content_ICCV_2019/papers/Rasouli_PIE_A_Large-Scale_Dataset_and_Models_for_Pedestrian_Intention_Estimation_ICCV_2019_paper.pdf
- DoTA GitHub: https://github.com/MoonBlvd/Detection-of-Traffic-Anomaly ;
  论文: https://arxiv.org/abs/2004.03044 ;
  TPAMI 2023 版: https://www.computer.org/csdl/journal/tp/2023/01/09712446/1AZL0aV7fpu
- DADA-2000 GitHub: https://github.com/JWFangit/LOTVS-DADA ;
  论文: https://arxiv.org/html/1904.12634v1 ; OpenDataLab 镜像:
  https://opendatalab.com/OpenDataLab/DADA2000/download
- DrivingGen: https://github.com/youngzhou1999/DrivingGen ;
  arXiv: https://arxiv.org/html/2601.01528v1 (ICLR 2026)
- WorldSimBench: https://arxiv.org/abs/2410.18072 (ICML 2025)
- WorldModelBench: https://www.alphaxiv.org/abs/2502.20694 (NeurIPS 2025)
- PhyWorldBench: https://research.nvidia.com/labs/cosmos-lab/phyworldbench/
- PhyGenBench: https://github.com/OpenGVLab/PhyGenBench (ICML 2025)
- VBench: https://vchitect.github.io/VBench-project/
