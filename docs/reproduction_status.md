# 复现状态报告

**日期**:2026-11-04(20:40 更新)
**状态分级定义**(严格区分,不得混淆):
- L0 仓库已下载
- L1 环境已配置
- L2 模型能加载
- L3 推理/处理能执行
- L4 评价能执行(产出结构化指标)
- L5 结果与论文数值对齐

---

## 1. 仓库状态总表

| 仓库 | 本地状态 | 级别 | 说明 |
|---|---|---|---|
| **DrivingGen** | ✅ 已克隆(HEAD `48ed356`,origin `youngzhou1999/DrivingGen`,working tree clean,143MB,depth=50) | **L0** | 源码级分析完成(scripts 0–5 全链 + 全部指标模块);环境/数据/权重未配 |
| **SynPeDAS** | ✅ 已克隆(HEAD `f34227f`,origin `levilovearch/SynPeDAS`,2.6GB,7 commits) | **L0**(数据即内容) | 211 组完整四元组实测;manifest 已建(见下) |
| Flow-World-Models | 未克隆 | L-1 | README 级预研完成(无官方权重、CC BY-NC、Bouncing Shapes 可行) |
| eb_jepa | 未克隆 | L-1 | README 级预研完成(无官方权重、Apache-2.0、单卡可训) |
| vjepa2 / Cosmos / Vista / SMD / DriveWAM / WorldLens / ReactSim-Bench | 本机不存在 | L-1 | 暂无计划(见 experiment_plan.md) |

## 2. 分级进度(按模块)

### 研究代码框架 `src/wmeval/`(自研,非复现)

| 模块 | 状态 | 证据 |
|---|---|---|
| traj_metrics(DrivingGen 官方公式移植) | ✅ **27 项自检全过** | `PYTHONPATH=src python src/selftest_metrics.py`;匀速→1.0、静止→NaN、v=6 解析一致、事件/归因/判定逻辑行为正确;v3.1 新增 4 项回归(order 精度 ×2、ID 碎片化 redetection、共存保护) |
| event_checks / uncertainty | ✅ 同上(合成数据) | 见同一次自检输出 |
| video_io / build_manifest | ✅ manifest 已产出 | `datasets/annotations/manifest.csv` 212 行;211/211 组四元组完整 |
| detect_track(YOLO+ByteTrack) | ✅ **可用** | ultralytics 已恢复安装;E1 v1–v3.1 四轮全量运行成功 |

### E1 实验进度(2026-11-04)

| 轮次 | 状态 | 输出 | 判定分布(real/gen) |
|---|---|---|---|
| E1 v1/v2 | ✅ 20/20 | `runs/synpedas_smoke{,_v2}/` | 见各 summary.csv |
| E1 v3 | ✅ 20/20 | `runs/synpedas_smoke_v3/` | 2/8/0 与 3/5/2 |
| E1 v3.1(bug 修复) | ✅ 20/20 | `runs/synpedas_smoke_v3_1/` | 3/7/0 与 3/6/1;检测层与 v3 完全一致(20/20),判定变化恰 2 处且均为已证实 bug 的因果 |

配套:人工复核体系(`docs/e1_v3_manual_review.md`、`runs/manual_review/`、
`datasets/annotations/e1_manual_labels.csv` 真值表待人工填写)、
错误分析(`docs/e1_v3_error_analysis.md`)、数据集扩展调研
(`docs/dataset_survey.md`)。
**当前级别:wmeval 管线整体 L4**(结构化指标+三态判定可复现执行);
L5(与论文数值对齐)待 DrivingGen 官方数据(HF token, P-8)。

### DrivingGen 官方管线(复现侧)

| 步骤 | 状态 | 缺口 |
|---|---|---|
| 0 数据下载 | ❌ | HF token(用户提供)+ 整仓体积未知 |
| 1 生成(Wan2.2-14B) | — 跳过 | 16GB 不可行(既定策略) |
| 2/3 轨迹提取 | 未执行 | ultralytics(F-02)+ UniDepth + SAMURAI + P-1/P-2/P-3 修补 |
| 4 视频指标 | 未执行 | 依赖 3 的 pkl;DINOv3/StyleGAN-V/VLM 各自依赖 |
| 5 轨迹指标 | **公式已 100% 移植**(traj_metrics.py) | 世界系轨迹(等 UniDepth);纯公式部分已可用 |

## 3. 数据实测事实(SynPeDAS,manifest 全量)

| 项 | 值 |
|---|---|
| 样本数 | 211(全部四元组完整) |
| 真实片段 | 150 帧;29.97 / 30.0 / 59.94 fps 混合;1920×1080 与 1280×720 混合 |
| 生成视频 | **统一 93 帧、16.00 fps、1280×704** |
| 生成参数 | 全部 video2world、seed=0、guidance=7、声明 77 帧(实测 93) |
| 时长 | 真实 2.5–5.0s vs 生成 5.8s → **时序对比必须归一化**(已实现) |

## 4. 已完成(真实命令与输出,全部可复核)

| 事项 | 命令/产物 |
|---|---|
| 环境 | `conda create -n wm_eval --clone gs5060_clean` → `torch 2.7.1+cu128` GPU 识别 ✅ |
| 仓库 | `git clone` ×2(见 commands.md §2) |
| 权重 | `curl -o ckpt/yolov10x.pt`(128,288,859 B;torch.load 校验待 F-02 恢复后补) |
| manifest | `python -I src/build_manifest_stdout.py` → `datasets/annotations/manifest.csv`(212 行) |
| 公式自检 | `python -I src/selftest_metrics.py` → 22/22 PASS |
| 文档 | environment_report / repository_analysis / research_gap / experiment_plan / commands / failure_log / reproduction_status(本文件) |

## 5. 未完成 / 被阻塞(如实)

| 事项 | 状态 | 原因 |
|---|---|---|
| ultralytics 安装 | **被阻塞**(4 次尝试) | Bash 安全分类器间歇不可用(failure_log F-01/F-02);非 pip/网络本身问题 |
| YOLOv10x 完整性 torch.load 校验 | 待执行 | 同上 |
| 最小实验(检测+跟踪+指标+事件判定,10 组) | 待执行 | 依赖 ultralytics;代码与 manifest 已就绪,恢复后一条命令可跑 |
| DrivingGen HF 数据集 | 未执行 | 需用户 HF token;体积未知,按安全要求先报告 |
| UniDepth / SAMURAI / DINOv3 / MTR | 未执行 | 属 E4–E6 可选阶段,优先级低于最小闭环 |

## 6. 恢复后的执行顺序

1. `pip install ultralytics`(F-02 预案命令);
2. YOLO 单视频冒烟(真实片段 1 个)→ 验证权重加载与 GPU 推理;
3. `run_minimal --manifest ... --num-samples 10` → E1 最小闭环(完成判据见 experiment_plan.md);
4. 视结果更新本文件与 commands.md 的 ✅/⚠️ 标记。
