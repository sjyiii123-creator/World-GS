# 命令手册:已验证 vs 未验证

**约定**:✅ = 本机实际执行成功并有输出为证;⚠️ = 未执行/未验证(只作记录);❌ = 执行失败(见 failure_log.md)。
**工作目录约定**:`/home/moz/World-GS` 为项目根;`repos/` 为克隆仓库;`src/wmeval` 为研究代码;`runs/` 为输出。

---

## 1. 环境管理(✅ 已验证)

```bash
# 创建独立评价环境(克隆自本机已验证的 gs5060_clean,不改动其他环境)
conda create -n wm_eval --clone gs5060_clean -y      # ✅

# 激活并验证 GPU 识别
conda activate wm_eval
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# ✅ 输出: torch 2.7.1+cu128 True NVIDIA GeForce RTX 5060 Ti
```

安装检测/跟踪依赖(⚠️ 分类器恢复后执行,见 failure_log F-02):
```bash
pip uninstall -y opencv-python-headless     # ⚠️
pip install ultralytics                     # ⚠️(会自动带 opencv-python)
python -c "import ultralytics, cv2; print(ultralytics.__version__, cv2.__version__)"
```

## 2. 仓库(✅ 已验证)

```bash
cd /home/moz/World-GS/repos
git clone --depth 50 https://github.com/youngzhou1999/DrivingGen.git   # ✅ HEAD 48ed356, 143MB
git clone https://github.com/levilovearch/SynPeDAS.git                 # ✅ HEAD f34227f, 2.6GB, 211 组四元组
```

## 3. 数据清单构建(⚠️ 待执行)

```bash
cd /home/moz/World-GS
PYTHONPATH=src /home/moz/miniconda3/envs/wm_eval/bin/python -m wmeval.build_manifest \
    --synpedas-root repos/SynPeDAS \
    --out datasets/annotations/manifest.csv          # ⚠️
```

## 4. 最小评价实验(⚠️ 待执行,依赖 §1 的 ultralytics 安装)

```bash
cd /home/moz/World-GS
PYTHONPATH=src /home/moz/miniconda3/envs/wm_eval/bin/python -m wmeval.run_minimal \
    --manifest datasets/annotations/manifest.csv \
    --num-samples 10 --sides both \
    --weights ckpt/yolov10x.pt --imgsz 640 \
    --outdir runs/synpedas_smoke                    # ⚠️

# 单视频快速冒烟(不依赖 manifest):
PYTHONPATH=src python -m wmeval.run_minimal \
    --videos repos/SynPeDAS/JAAD_crossing_moments/video_0002_0_2_6b_f60-210.mp4 \
    --weights ckpt/yolov10x.pt --outdir runs/smoke_one    # ⚠️
```

## 5. 权重下载(⚠️ 分类器恢复后执行)

```bash
cd /home/moz/World-GS/ckpt
curl -sL -o yolov10x.pt https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10x.pt  # ⚠️ ~25MB
# 备选源(ultralytics 自动下载也会解析到同一 release)
```

## 6. DrivingGen 官方管线(⚠️ 全部未验证;前置条件见下)

| 命令 | 前置条件(当前缺口) |
|---|---|
| `bash scripts/0-down_data.sh` | ❌ HF token + 整仓体积未知(阻塞,待用户提供 token) |
| `bash scripts/1-example_infer_model.sh` | Wan2.2-14B 生成,16GB 不可行(**跳过**) |
| `bash scripts/2-get_ego_traj.sh` / `3-get_agent_traj.sh` | 需先处理 P-1(pdb 残留)/P-2(YOLO 路径)/P-3(nuplan 内参路径)+ UniDepth/SAMURAI 环境 |
| `bash scripts/4-get_video_metrics.sh` | 依赖脚本 3 的 pkl;Agent Missing 需 7B VLM(P-4,降级项) |
| `bash scripts/5-get_traj_metrics.sh` | 依赖脚本 3;FTD 需 MTR ckpt(未下载) |

具体修改点(等价于打补丁,均须记录):P-1 注释 pdb;P-2 权重放 `ckpt/`;P-3 用 UniDepth 内参替代(记录偏离)。

## 7. 常用排查命令(✅ 已验证)

```bash
nvidia-smi                                              # ✅ 驱动 595.99.02 / CUDA 兼容 13.2(≠Toolkit)
/usr/local/cuda-12.8/bin/nvcc --version                 # ✅ Toolkit 12.8(不在 PATH)
conda env list                                          # ✅
/home/moz/miniconda3/envs/wm_eval/bin/python -I -c "..."  # ✅ 隔离模式跑不受信数据的只读探测
```

## 8. 实测数据事实(供脚本参数参考,✅ 2026-11-04 实测)

| 数据 | 帧数 | fps | 分辨率 |
|---|---|---|---|
| SynPeDAS 真实片段(JAAD) | 150 | 29.97 | 1920×1080 |
| SynPeDAS 生成视频 | **93**(JSON 声明 77,实际 93) | **16.00** | **1280×704** |

→ 两边时长 5.0s vs 5.8s,帧率与长度均不同:**任何时序对比必须用归一化时间**(已实现于 `src/wmeval/event_checks.py`)。

## 9. E1 实验(✅ 已验证,2026-11-04)

```bash
cd /home/moz/World-GS

# 自测(含 v3.1 回归测试,27 项)
PYTHONPATH=src /home/moz/miniconda3/envs/wm_eval/bin/python src/selftest_metrics.py   # ✅ 27 passed

# E1 v3 十组样本(前轮已跑,输出 runs/synpedas_smoke_v3/)
/home/moz/miniconda3/envs/wm_eval/bin/python -I src/run_e1_smoke.py                 # ✅ 20/20

# E1 v3.1(bug 修复后全量重跑,检测层与 v3 一致)
PYTHONPATH=src /home/moz/miniconda3/envs/wm_eval/bin/python -m wmeval.run_minimal \
    --manifest datasets/annotations/manifest.csv \
    --num-samples 10 --sides both \
    --weights ckpt/yolov10x.pt --imgsz 640 --conf 0.3 \
    --outdir runs/synpedas_smoke_v3_1                                                # ✅ 20/20, ~5min GPU

# 人工复核可视化生成(contact sheet + 带框关键帧,本地输出路径,不发图给模型)
/home/moz/miniconda3/envs/wm_eval/bin/python -I src/make_manual_review.py           # ✅
```

**注意**:v3.1 的 `summary.json` 中 `stem` 字段为视频文件名(无 `__real/__gen`
后缀,real/gen 同名会冲突),按 `video` 字段的路径区分侧别
(`JAAD_crossing_moments`=real, `generated_video`=gen)。

## 10. E2 协议与验证工具(✅ 已验证,2026-11-05)

```bash
cd /home/moz/World-GS

# 全部自测(主 35 项 + E2 模块 20 项)
PYTHONPATH=src /home/moz/miniconda3/envs/wm_eval/bin/python src/selftest_metrics.py   # ✅ 35 passed
PYTHONPATH=src /home/moz/miniconda3/envs/wm_eval/bin/python src/selftest_e2.py        # ✅ 20 passed

# 事件协议表(211 行,人工字段全 unknown)
/home/moz/miniconda3/envs/wm_eval/bin/python -I src/build_event_protocol.py           # ✅

# 泄漏安全数据划分提案(dev/val/test)
/home/moz/miniconda3/envs/wm_eval/bin/python -I src/build_e2_split.py                 # ✅ 36/71/104
```

产物:
- `datasets/annotations/event_protocol.csv`(条件层,待人工复核)
- `datasets/annotations/e2_split_proposal.csv`(划分提案,review_status=pending)
- 新模块 `src/wmeval/actor_binding.py`(身份链)、
  `src/wmeval/crossing_evidence.py`(L1-L4 证据)——**未接入 v3.1 判定链**

## 11. 数据集下载与 JAAD 弱 GT(✅ 2026-11-05 晚)

```bash
cd /home/moz/World-GS
# 审计(只读)与严格核验(含 ffprobe 抽查)
bash repos/wm_dataset_kit/wm_dataset_setup_v2.sh audit /home/moz/World-GS    # ✅
bash repos/wm_dataset_kit/wm_dataset_setup_v2.sh verify /home/moz/World-GS    # ✅(下载完成后跑)

# HF 数据下载(代理修复已内置:unset ALL_PROXY)
bash tools/hf_download.sh drivinggen    # ✅ yangzhou99/DrivingGen ~2.24GB
bash tools/hf_download.sh safemvdrive   # ✅ JiaweiZhou/SafeMVDrive ~7.97GB

# JAAD 弱 GT 证据表(sample_id ↔ JAAD 行人 ID 映射)
/home/moz/miniconda3/envs/wm_eval/bin/python -I src/build_jaad_gt_evidence.py  # ✅ 211/211
```

已就位仓库:`repos/JAAD`(JAAD_2.0,346+346 XML)、`repos/DriveJudge`、
`repos/SMD`、`repos/WorldLens`、`repos/DrivingGen`(原有)。
HF 数据:`datasets/raw/DrivingGen`、`datasets/raw/SafeMVDrive`。
注意:hf 直调需先 `unset ALL_PROXY all_proxy`(socks:// scheme 不被
httpx 支持,见 failure_log F-10);分类器波动期优先用脚本文件/python
简单形态调用(见 F-09)。

## 12. J 系列:Actor 时序验证(J0/J1 已验证,2026-11-05)

```bash
cd /home/moz/World-GS
# CPU 自测(裁剪/校准/指标/遮挡选择/预测头,无需权重)
PYTHONPATH=src /home/moz/miniconda3/envs/wm_eval/bin/python src/selftest_j.py       # ✅ 24 passed

# 数据集构建(15,523 对 + 7,849 竞争集,seed=11 可复现)
/home/moz/miniconda3/envs/wm_eval/bin/python -I src/build_actor_sequences.py        # ✅

# J0 运动学基线(全量,<1s)
/home/moz/miniconda3/envs/wm_eval/bin/python -I src/run_j0_baseline.py              # ✅ acc 0.808 / merge 0.182

# J1 冻结 DINOv2 基线(GPU;权重本地 ckpt/dinov2-small,curl 代理下载,见 F-14)
/home/moz/miniconda3/envs/wm_eval/bin/python -I src/run_j1_baseline.py              # ✅(结果见 jepa_experiment_results.md)
```

产物:`datasets/processed/actor_sequences/`(样本+竞争集+meta)、
`runs/j0_baseline/`、`runs/j1_baseline/`(含特征缓存
feature_cache/,npz+模型/预处理 meta)。
新模块:`src/wmeval/latent_features.py`(裁剪+缓存)、
`src/wmeval/latent_association.py`(校准+融合)、
`src/wmeval/latent_predictor.py`(J2 GRU 接口,forward/loss 可测,
未训练)。

## 12b. J1.1 审计修复与 J3-lite(2026-11-05 下午,✅)

```bash
# J1.1 审计复评(严格校准+显式分母指标;读 J1 原始 pair_results,不重提特征)
/home/moz/miniconda3/envs/wm_eval/bin/python -I src/run_j1_1_audit.py               # ✅ 严格thr 0.8122/FPR 4.93%/TPR 42.39%

# 遮挡三策略 A/B/C(GPU,各策略独立缓存;B 为可见性分级修正版)
/home/moz/miniconda3/envs/wm_eval/bin/python -I src/run_j1_1_occlusion.py           # ✅ runs/j1_1_audited/occlusion_strategies.json

# 多候选评价 + J3-lite 三态(GPU;阈值 train-only 校准)
/home/moz/miniconda3/envs/wm_eval/bin/python -I src/run_j3_lite.py                  # ✅ runs/j3_lite/metrics.json

# J2 数据就绪检查(历史序列/泄漏审计/塌缩检查;不训练)
/home/moz/miniconda3/envs/wm_eval/bin/python -I src/run_j2_readiness.py             # ✅ runs/j2_readiness/
```

**长任务模式(内存压力下,防 VSCode 崩溃中断)**:串行 + setsid 完全
脱离终端(`setsid nohup bash -c '...串行三脚本...' &` + .done 哨兵
文件 + until 等待)——本轮实测两个并行 python(cv2 多线程解码)可
吃满 16G 物理内存导致宿主 IDE 崩溃(见 failure_log F-16)。

## 12c. J2 全流程(阶段 1-7,2026-11-05 傍晚,✅ 含负结果)

```bash
cd /home/moz/World-GS
# 阶段 1 环境审计(外部工具包,SHA 7/7;必须用 wm_eval PATH)
export PATH=/home/moz/miniconda3/envs/wm_eval/bin:$PATH
bash WorldGS_J2_Configuration/tools/env_audit.sh | tee runs_env_audit.log    # ✅

# 阶段 2 数据 v2(溯源+首观测目标+源级四划分 412 序列)
/home/moz/miniconda3/envs/wm_eval/bin/python -I src/build_j2_v2.py            # ✅
/home/moz/miniconda3/envs/wm_eval/bin/python WorldGS_J2_Configuration/tools/preflight_j2.py \
  --npz runs/j2_experimental/j2_sequences_v2.npz \
  --manifest runs/j2_experimental/j2_manifest.csv                            # ✅ exit 0

# 阶段 3 A0/A1/A2 选择性关联基线(CPU,复用竞争集明细)
/home/moz/miniconda3/envs/wm_eval/bin/python -I src/run_a_baselines.py       # ✅ A1 最优

# 阶段 4 J2 训练六步梯(1s GPU/0.05GB) + 阶段 5/6 消融与融合
/home/moz/miniconda3/envs/wm_eval/bin/python -I src/run_j2_train.py          # ✅ 梯全过
/home/moz/miniconda3/envs/wm_eval/bin/python -I src/run_j2_association.py    # ✅(负结果)
```

产物:`runs/j2_experimental/`(v2 npz+manifest、两 checkpoint、
训练日志、a_baselines、ablation、association、sweep、stats)。
**结论要点**:A1(kin 排序+kin 拒识)为强基线;J2 v1 无增量
(区分间距 0.110<静态 0.257;融合无改善/联合排序有害)——
事件级实验未运行(双门槛未满足,docs/j2_event_level_impact.md)。
