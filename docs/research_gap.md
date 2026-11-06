# 研究缺口分析:什么已被解决,什么值得做

**日期**:2026-11-04(2026-11-05 E2 修订:补入 WorldLens/WorldReasonBench/
DriveJudge 实测对比,区分"已证实/待验证";JAAD/DoTA 用途重命名)
**依据**:DrivingGen/SynPeDAS 源码级分析(见 `repository_analysis.md`)+
三个已确认基准的官方 README/协议核查(见 `dataset_survey.md` §2.8)+
本机环境实测。

---

## 1. 现有方法已经解决了什么(以源码为证)

| 能力 | 由谁解决 | 证据(源码位置/官方协议) |
|---|---|---|
| 视频质量/保真度评价 | DrivingGen | FVD(StyleGAN-V I3D)、P2020 式客观质量、CLIP-IQA+ 主观质量、DINO 场景一致性 |
| 物理合理性(轨迹层) | DrivingGen | vel/accel consistency(`traj_consistency.py`,纯公式)、comfort/curvature/speed(`traj_quality.py`)、FTD(MTR 特征 Fréchet)、ADE/DTW |
| Agent 外观漂移 | DrivingGen | DINOv3 R/A 余弦一致性(`video_a_consist.py`) |
| "目标突然消失"检测 | DrivingGen(部分) | `video_a_missing.py`:非末尾结束的 track → VLM 判断 Unnatural → missing 率 |
| 危险场景生成→下游杀伤力 | SMD/SafeMVDrive(外部) | GRPO 对抗选择 + UniAD 碰撞率评估(本机不可复现,仅引用) |
| 反应式参与者闭环评价 | ReactSim-Bench(外部) | nuPlan 轨迹级,2,636 场景(未克隆,调研报告评级 A) |
| 真实-生成行人横穿数据对 | SynPeDAS | 211 组四元组已公开,评价套件为作者明示 TODO |
| **多视角驾驶世界模型全景评测** | **WorldLens**(CVPR 2026 Oral) | nuScenes 生态 14+ 模型、5 任务 24 维度;Action-Following 闭环报告 collision/off-road 失败;行为安全入 Human Preference 维度 |
| **生成驾驶视频的 VLM 裁判能力** | **DriveJudge/DriveJudgeBench** | 1,597 片段/7,371 问答对,六类含 safety assessment 与 traffic-law compliance;评的是裁判模型(最强 ~64% 准确率),非世界模型 |
| **世界状态一致性 QA 评价** | **WorldReasonBench**(NeurIPS 2026) | 436 结构化用例,物理/社会/逻辑/信息四维,VLM+LLM 两段判卷(AccQA/ScorePR) |

## 2. 仍然空缺的问题(= 论文空间;**已证实 vs 待验证**分级)

> E2 修订说明:下列缺口在对照 WorldLens/WorldReasonBench/DriveJudge 的
> 实际协议后重新分级。**不以"对方没用 Valid/Invalid/Uncertain 标签"
> 论创新**——差异必须落在评价对象、指标定义、实验协议三层。

### 缺口 1:Agent Missing 的判定既不可复现也不含不确定性 【协议差异,已证实】
- **源码证据**:遮挡/出画几何豁免逻辑被**注释禁用**,裁决完全交给
  Cosmos-Reason1-7B VLM;`min_track_len` 调用处传 2;多处代码缺陷。
- **对照**:DriveJudge 证明最强 VLM 裁判在驾驶视频上仅 ~64% 准确率,
  且其六类中无"目标消失可解释性"专用检查;WorldLens 无不确定性输出;
  WorldReasonBench 无不确定性输出。
- **机会**:几何+跟踪证据链的三态判定(替代 VLM 裁决),
  DrivingGen 官方实现作对照基线。**差异化是"证据链+可复现+不确定性",
  不是"安全评价本身"**(WorldLens 闭环安全已存在)。

### 缺口 2:所有指标都是"全场平均",没有事件级判定 【已证实】
- **源码证据**:DrivingGen 各指标 `np.nanmean` 跨场景单标量;
  WorldLens 24 维度全部是**整段视频级**打分(无 per-event 检查);
  DriveJudge 以 VQA 形式问"temporal behavior"但不做事件状态机;
  WorldReasonBench 检查世界状态一致性但无驾驶事件类目。
- **机会**:事件级结构化验证(条件→状态→判定三层协议,
  L1-L4 横穿语义分层)——**目前无先例**;但价值需
  E2 消融证明(B0 vs B1 的 Event P/R 差异)。

### 缺口 3:真实-生成对的时间对齐协议缺失 【已证实,E2 已补文档】
- v3.1 起实现 t_norm 相位对齐并明确其禁止用途(见
  crossing_event_semantics.md §5);**物理时间锚点仍未建立**,
  "事件时刻偏移"指标在锚点确立前不发明。

### 缺口 4:检测/跟踪的不确定性没有进入评价 【已证实,v3.1 部分解决】
- **v3.1 已做**:弱证据归因链(遮挡/重检测/ID 碎片化/质量/截断);
  **未做**:三态的实际价值证明(消融 B0-B3,待人工真值)。
- **对照**:三基准均无不确定性输出——空位真实存在,
  但"存在空位"≠"三态有价值":必须报告固定放行率下的 IAR 权衡
  (e2_experiment_plan.md §4.2 硬约束),防止用高 Uncertain 刷指标。

### 缺口 5:生成条件的协议化映射 【新识别,已证实无先例】
- **对照**:DriveJudge 明示不做 prompt/条件遵循(只做 real-vs-generated);
  WorldLens 的条件遵循是**动作**条件(action-following),非文本事件条件;
  WorldReasonBench 检查 expected reasoning elements(部分覆盖,非驾驶)。
- **机会**:`event_required: true/false/unknown` 三值协议
  (event_protocol.csv)——"生成模型被要求保留什么事件"这一问题
  在三个基准中均未出现。unknown 样本不计入违约率的纪律是
  方法论贡献的一部分。

### 缺口 6:SynPeDAS 评价套件仍是作者明示 TODO 【已证实】
- 仓库无评价代码;**第一个把事件级评价做在这份数据上的工作有先发
  优势**(MDPI Q2 定位,引用时按"实验起点"处理)。

**诚实的总结**:六个缺口中,1/2/4/5 是"空位真实存在但价值未证明"
(待 E2 消融);3 已部分解决;6 是数据机会。**没有任何一个可以在
人工真值缺失前写成"已证实的优势"**。

## 3. 是否需要引入 JEPA / Flow Matching(任务书第六阶段的两个问题)

**问题 1:JEPA/FlowWM 的抽象特征预测思想是否值得用于安全关键事件评价?**

- **支持理由**:EB-JEPA 的能量函数天然提供一个"状态转移可信度"标量(能量高=转移在训练分布外),可作为 Uncertain 判据的第三个证据源(与几何遮挡、检测置信度并列);FlowWM 证明 DINOv3 特征空间的 flow matching 预测在随机环境里优于确定性回归——特征空间残差可能捕获"物理不合理运动"。
- **反对理由**(源码与证据侧):
  1. 两者都**没有预训练权重发布**,Bouncing Shapes/Moving MNIST 级别的训练与"驾驶安全事件"域差距巨大;在小论文周期内(6–8 周)自训一个可用模型风险高;
  2. 它们回答的是"表征预测好坏",不直接回答"行人是否异常消失/事件顺序是否正确"——后者的直接证据(检测+跟踪+几何)更便宜、更可解释、更可对齐 DrivingGen 官方基线;
  3. DINOv3 特征一致性(DrivingGen 已用)在"外观漂移"维度已覆盖特征空间证据。
- **结论(暂定)**:**不作为主线引入**。作为 B3→B4 的可选消融实验:若时间允许,用 DINOv3 特征(已有 HF 权重)在生成视频上计算"特征空间预测残差"作为异常信号,与事件级判定做 Late Fusion,检验是否降低 Uncertain Rate 而不升假阳性。**不为引入而引入;引入必须有可测的增量价值(ΔIAR/ΔVRR)。**

**问题 2:相对目标轨迹/事件规则/普通视觉特征是否有明确增量?**
- 目前**证据不足,倾向否定**——缺一个对照实验;已列入 experiment_plan.md 的可选实验 E6,判据预设:在同等 IAR 下 Uncertain Rate 相对纯几何方案下降 ≥5 个百分点才算有增量,否则论文中只作讨论不作贡献。

## 4. 推荐的小论文切入点(基于真实代码与数据状态)

**"面向条件生成驾驶视频的事件级一致性验证,带不确定性感知三态判定"**
- 数据:SynPeDAS 211 组真实/生成对(已下载,零生成成本);
- 基线:DrivingGen 官方指标公式的忠实移植(轨迹一致性组已 100% 可移植,DINOv3 组可移植,VLM 组降级为对照讨论);
- 方法贡献:① 归一化时间轴事件约束验证器;② 证据链式 Uncertain 判定(替代 VLM);③ 人为注入异常的受控测试协议(帧删除/遮挡/身份替换)区分"生成异常"与"检测/跟踪噪声";
- 与 3DGS 背景的衔接(调研报告 §六):后续可把事件级 2D 验证升级为深度/几何一致性验证(UniDepth 已在依赖清单里)。

**不强行支持既定选题的证据**:若"不确定性感知"实验显示 Uncertain 判定不能稳定区分检测噪声与生成异常(如 E3/E4 阶段发现),则退守"事件级一致性指标 + 失真类型统计"仍有独立价值(缺口 2/3 独立于不确定性)。判据在 experiment_plan.md 预注册。
