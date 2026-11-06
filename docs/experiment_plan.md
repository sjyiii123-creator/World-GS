# 第一轮实验计划

**日期**:2026-11-04 | **硬件预算**:RTX 5060 Ti 16GB 单卡,系统内存 16GB,磁盘新增 <5GB
**总原则**(延续用户 RouteSurf 纪律):每阶段先定"完成判据"再动手;单变量推进;全部产出落 `runs/`,全部命令落 `commands.md`(区分已验证/未验证)。

---

## E0 环境与数据冒烟(本 session 目标)

**步骤**:
1. `wm_eval` conda 环境(克隆自本机已验证的 gs5060_clean:torch 2.7.1+cu128)→ 装 ultralytics(YOLO 检测 + ByteTrack 跟踪);
2. 下载 yolov10x.pt(THU-MIG 官方 release,~25MB);
3. `build_manifest`:SynPeDAS 211 组四元组 → `datasets/annotations/manifest.csv`,OpenCV 实测 fps/分辨率/帧数(补 JSON 里缺失的 `"resolution":"none"`、无 fps 字段)。

**完成判据**:
- manifest.csv 覆盖 211 组,四元组完整率 100%;
- 实测得到生成视频的 fps/分辨率(目前未知,可能 24fps);
- YOLOv10x 在单帧上推理成功(GPU 识别正常)。

**预计资源**:<10 分钟 CPU,<30MB 下载。

## E1 最小评价闭环(本 session 目标,任务书第四阶段 A–D)

**输入**:manifest 前 10 组样本,每组的真实片段 + 生成视频(共 20 个视频)。

**流程**(全部代码已写好,`src/wmeval/`):
video_io 读取(可配 fps)→ YOLOv10x + ByteTrack → track_id/category/bbox/conf/frame/timestamp →
(a) DrivingGen 官方公式移植:速度/加速度一致性、comfort、curvature、speed(px 空间,dt=1/fps,阈值自适应并记录,结果显式标注"公式复现,非论文口径");
(b) 事件规则组:行人出现 / 进入道路区(归一化矩形)/ 时窗 / 顺序 / 无法解释消失;
(c) 不确定性归因:遮挡(同帧 IoU)→ 重检测(同类新轨迹就近出现)→ ID 切换(末段速度跳变)→ 检测质量(mean conf)→ 异常;
(d) 三态判定:Valid / Invalid / Uncertain,附证据链。

**输出**:`runs/synpedas_smoke/<sample>__{real,gen}/` 下 detections.jsonl、tracks.json、metrics.json、events.json、annotated.mp4、keyframes/、summary.csv。

**完成判据**(预注册):
1. 20 个视频全部跑完无崩溃(允许个别视频报错但必须记录);
2. 每个视频产出 ≥1 个官方公式的数值结果(速度/加速度一致性)+ ≥1 个三态判定;
3. 至少 1 个可视化视频人工抽查:框/ID 与画面一致;
4. 真实片段上行人检出率 ≥80%(10 组中 ≥8 组有 person 轨迹)——低于此值说明检测/跟踪配置有问题,先修再谈生成侧结论。

**预计资源**:20 视频 × ~150 帧 × YOLOv10x@640 ≈ 5–10 分钟 GPU,输出 <1GB。

## E2 错误类型分析(30–50 组,小论文实验一)

- 扩到 manifest 前 50 组(100 视频);对"生成侧行人缺失/消失/身份断裂"逐例人工核对(keyframes + annotated.mp4);
- 产出"失真类型 × 频率"表:行人不存在 / 出现但未横穿 / 半途消失(可解释 vs 不可解释)/ 外观突变 / 事件时序偏移;
- **注意协议**:真实片段有横穿 ≠ 生成必须复制横穿;以 JSON prompt 描述 + video2world 条件为"应发生事件"的依据,逐例记录判定理由。
- 完成判据:≥30 组有效标注;每类失真 ≥3 个实例或声明该类不存在。
- 资源:GPU ~20 分钟 + 人工 4–8 小时。

## E3 基线对照(任务书第七阶段,B0–B3)

| 基线 | 实现 | 指标 |
|---|---|---|
| B0 视频质量 | pyiqa(PSNR/SSIM/LPIPS 真实 vs 生成对) | 质量分 |
| B1 DrivingGen 相关 | 本项目移植(公式层一致,px 空间 + 世界系升级后重跑) | vel/accel 一致性、comfort、Agent 一致性(若 DINOv3 跑通) |
| B2 规则式事件检查 | event_checks 去 uncertainty(只输出 pass/fail) | 事件 P/R/F1 |
| B3 不确定性感知 | 完整流程 | IAR / VRR / Event P/R/F1 / Uncertain Rate / 单视频耗时 |

**人为注入异常协议**(必须与真实结果分开报告):
- 行人帧删除模拟消失(从某帧起抹除检测框 + 像素 inpaint 简化为遮挡块);
- 轨迹断裂模拟 ID 切换;
- 延迟出现模拟事件时序偏移。
每种注入 ≥10 例,计算各基线的检出率;**论文中明确标注注入实验 ≠ 真实模型失败案例**。

**完成判据**:四基线 × 同一批视频的对比表,单视频耗时列,注入实验与真实实验分开两张表。

## E4(可选,按优先级)UniDepth 深度反投影

- UniDepthV2 vitl14(HF ~1.3GB)→ Agent 轨迹世界系化 → 官方公式默认阈值(m/v_static=0.1/length_eps=1.0)重跑 → 得到**可与论文口径对齐的数值**;
- 前置:解决 P-3(内参来源:先用 UniDepth 预测内参,记录为偏离官方)。
- 完成判据:同一批视频产出 m 空间指标,与 px 空间指标的秩相关(说明排序是否稳定)。

## E5(可选)DINOv3 Agent 一致性移植

- `facebook/dinov3-vith16plus-pretrain-lvd1689m`(ModelScope/HF);16GB 需独占推理;
- 复现 R/A 余弦一致性,并修官方返回值 bug(记录两种口径);
- 完成判据:真实片段自一致性 ≈1 量级 sanity check + 生成侧分布图。

## E6(可选,研究问题)特征空间预测残差是否降低 Uncertain Rate

- 见 research_gap.md §3;判据预注册:同等 IAR 下 Uncertain Rate 降 ≥5pp 才算 JEPA/特征思想有增量,否则只作讨论。
- 资源:DINOv3 特征提取 ~1 GPU 小时 + 简单预测头训练(几小时 CPU/GPU 混合)。**仅在 E2–E3 完成且时间富余时启动。**

## 里程碑与总资源估计

| 阶段 | 累计 GPU 时间 | 人工时间 | 产出 |
|---|---|---|---|
| E0–E1(本 session) | ~15 分钟 | 1 小时 | 冒烟闭环 + 7 份文档 |
| E2 | +20 分钟 | 4–8 小时 | 失真类型统计表(小论文实验一) |
| E3 | +1–2 小时 | 4 小时 | 基线对照表(实验二)+ 注入协议 |
| E4–E6(可选) | +3–6 小时 | 6–10 小时 | 口径对齐数值 / 消融 |

**回退判据**(诚实红线):若 E1 完成判据 4 不满足(真实侧行人检出率 <80%),先调检测配置(imgsz/权重/置信度)再进入 E2;若调后仍 <60%,报告阻塞并请示是否引入更大检测模型。

---

## E7/E8(2026-11-04 新增,数据扩展计划)

E1 v3.1 复盘暴露两个方法学缺口:**Invalid 召回未测**(仅 1 个真 invalid)、
**规则阈值泛化未测**(211 组 SynPeDAS 是唯一分布)。经公开数据集调研
(见 `docs/dataset_survey.md`,联网核实),新增:

| 实验 | 数据 | 目的 | 磁盘 | 状态 |
|---|---|---|---|---|
| E7 真实异常召回 | DoTA 行人类别抽样 50–100 段 | 真实 Invalid 事件上的召回率 | ~10GB 级 | **待批准下载** |
| E8 跨域泛化 | JAAD 完整版 held-out 片段 | 阈值/规则的域外泛化 | 数 GB | **待批准下载** |
| E3 注入升级(可选) | CARLA + ScenarioRunner | 受控合成注入异常 | ~20GB | 待批准 |
| 扩样 50+(如需) | PIE(74GB)或 DADA-2000(53GB) | 扩样/R2 互补 | 53–74GB | 待批准 |

E2(30–50 组)用现有 SynPeDAS 211 组即可,**不依赖新下载**。
DrivingGen benchmark 已确认发表于 ICLR 2026(arXiv:2601.01528),
B1 官方对齐仍需用户提供 HF token(P-8)。
