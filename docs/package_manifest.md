# 交付包清单：e1_v3_deliverables_20261104

日期：2026-11-04。本包固化 E1 v3/v3.1 阶段的全部可复核交付物
（代码、文档、标注骨架、数值证据、人工复核可视化）。
不含大体积媒体与外部资源（见 §3，均可按 §4 重建/再获取）。

完整性校验（打包后执行，值见包外 `dist/SHA256SUMS.txt`）：

```bash
sha256sum e1_v3_deliverables_20261104.tar.gz
tar tzf e1_v3_deliverables_20261104.tar.gz | wc -l   # 应与打包记录一致
```

---

## 1. 包内结构（解包后根目录 `e1_v3_deliverables_20261104/`）

```
docs/                          # 12 份文档
  environment_report.md        # 硬件/驱动/CUDA/环境
  repository_analysis.md       # DrivingGen/SynPeDAS/EB-JEPA/FlowWM 代码结构
  research_gap.md              # 研究空位分析
  experiment_plan.md           # E0–E8 实验计划(含 2026-11-04 数据扩展)
  commands.md                  # 已验证/未验证命令手册
  failure_log.md               # F-01~F-06 失败与修复记录
  reproduction_status.md       # L0–L5 分级复现状态
  e1_v3_manual_review.md       # 人工复核清单(PENDING_HUMAN_VISUAL_REVIEW)
  e1_v3_error_analysis.md      # 分层错误分析(代码/检测/约束设计/待人工)
  dataset_survey.md            # 公开基线数据集调研(联网核实)
  package_manifest.md          # 本文件
src/                           # 研究代码
  wmeval/                      # 库:video_io, detect_track, event_checks,
                               #     traj_metrics, uncertainty, run_minimal,
                               #     build_manifest
  selftest_metrics.py          # 27 项自测(含 v3.1 回归测试)
  run_e1_smoke.py              # E1 十组样本入口
  make_manual_review.py        # 人工复核 contact sheet 生成器
  build_manifest_stdout.py, run_minimal_entry.py, smoke_run.py
datasets/annotations/
  manifest.csv                 # SynPeDAS 211 组四元组清单
  e1_manual_labels.csv         # 人工真值标注表(骨架,待人工填写)
runs/
  manual_review/               # 24 个文件:2 张 contact sheet + 20 张带框
                               # 关键帧 + 2 份 t_norm 对齐记录
  inspect_*.jpg                # 3 张早期抽查帧(无框)
  synpedas_smoke_v3_1/         # 当前基线:每视频 meta/events/metrics/tracks/
                               # detections JSON(20 视频;mp4/keyframes 已剥离)
  synpedas_smoke{,_v2,_v3}/    # 历史版本仅保留 summary.csv + summary.json
```

## 2. 关键数值（快速核验锚点）

- 自测：`27 passed, 0 failed`（`PYTHONPATH=src python src/selftest_metrics.py`）
- v3.1：20/20 处理；real 3 valid/7 uncertain/0 invalid；gen 3/6/1；
  检测层与 v3 完全一致（dets+tracks 20/20 相同），判定变化恰 2 处
  （0024 real→valid：order 精度修复；0016 gen→uncertain：redetection 盲区修复）
- 0016 gen 证据链：track64(f45–54)→track30(f55–62)→track64(f63–92)，
  x 单调右移 ~0.002/帧——ID 碎片化假阳性（详见 error_analysis §2.2）

## 3. 未包含内容与原因

| 内容 | 原因 | 重建/获取方式 |
|---|---|---|
| `runs/**/annotated.mp4`、`keyframes/` | 体积大且可由管线再生成 | `run_e1_smoke.py` 或 `run_minimal --outdir runs/synpedas_smoke_v3_1`(~5 分钟 GPU) |
| `ckpt/yolov10x.pt`(128MB) | 权重可再下载 | `curl -sL -o ckpt/yolov10x.pt https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10x.pt` |
| `repos/`(SynPeDAS 2.6GB, DrivingGen 143MB) | git 可再克隆 | `git clone https://github.com/levilovearch/SynPeDAS.git` 等(见 commands.md §2) |
| `runs/synpedas_smoke{,_v2,_v3}/` 逐视频 JSON | 与 v3.1 检测层一致(v3.1 已含) | 历史版本 summary 保留在包内 |

## 4. 复跑前置条件

- conda env `wm_eval`（torch 2.7.1+cu128 + ultralytics；
  创建命令见 `commands.md` §1）
- GPU：RTX 5060 Ti 16GB（任何 ≥8GB CUDA 卡亦可，改 imgsz）
- 工作目录约定：解包后把 `repos/SynPeDAS`（需另克隆）放同级，
  `ckpt/yolov10x.pt` 放 `ckpt/`

## 5. 本包之后的下一步（按优先级）

1. 人工填写 `datasets/annotations/e1_manual_labels.csv`（20 行真值，
   指南见 `e1_v3_manual_review.md` §4）；
2. 批准并下载 `docs/dataset_survey.md` §4 的阶段 A/B 数据
   （JAAD、DoTA → E8 泛化 / E7 召回实验）；
3. E2 扩到 50 组（现有数据即可）；
4. B1 对齐需用户提供 HF token（DrivingGen, P-8）。
