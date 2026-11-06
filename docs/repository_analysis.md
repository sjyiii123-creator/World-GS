# 仓库代码结构分析:DrivingGen 与 SynPeDAS(+ FlowWM / EB-JEPA 预研)

**日期**:2026-11-04
**分析方法**:DrivingGen 与 SynPeDAS 的分析基于逐文件源码级阅读(scripts/*、drivinggen/ 下全部评价模块、SynPeDAS 数据目录与元数据样本)。分析时本地克隆尚未完成(工具阻塞,见 `failure_log.md`),结论来自 GitHub 原始文件(raw.githubusercontent.com)逐个抓取,与本地克隆内容同源同 commit(main 分支 HEAD)。**文中所有"实现细节"均来自源码原文,非 README 转述。**

---

## 一、DrivingGen(ICLR 2026,arXiv 2601.01528)

### 1.1 仓库概况

- URL:https://github.com/youngzhou1999/DrivingGen(Apache-2.0,9 commits,48 stars)
- 评测 14 个生成式驾驶视频世界模型(通用/物理/驾驶专用三类),视觉 + 轨迹两大类指标
- 官方环境:`environment.yml` → Python 3.10.0 + torch 2.7.1(pip 版,随附 nvidia-*-cu12==12.6.x)+ vllm 0.10.1.1 + transformers 4.56.1 + pyiqa 0.1.14 + timm + decord/av + filterpy
  - **对 5060 Ti 的含义**:torch 2.7.1 与本机已验证的 2.7.1+**cu128** 兼容(官方用的 cu126 装不上 sm_120,需换 cu128 轮子,其余依赖不变);vllm 用于 Agent Missing 的 VLM 推理
- 顶层结构:`drivinggen/`(主包)、`scripts/`(0–5 号入口)、`third_parties/`(7 个 vendored 依赖,**非 git submodule**,直接进库)、`assets/`、`environment.yml`

### 1.2 管线全流程(scripts 0–5 的真实调用链)

| 脚本 | 实际调用 | 作用 |
|---|---|---|
| `0-down_data.sh` | `drivinggen/down_dataset.py`(内嵌 HF token 占位符) | `snapshot_download(repo_id="yangzhou99/DrivingGen", repo_type="dataset", local_dir="./data")` 整仓下载,需 HF token |
| `1-example_infer_model.sh` | `drivinggen/infer_example_wan.py` | 生成侧:Wan2.2-14B 图生视频,**16GB 不可行,跳过**(按既定策略) |
| `2-get_ego_traj.sh` | `drivinggen/func/extract_traj_ego_unidepth.py` | 自车轨迹:UniDepth 深度 + 自带 visual SLAM 特征匹配 |
| `3-get_agent_traj.sh` | `drivinggen/func/extract_traj_agent_unidepth.py` | **Agent 轨迹提取(核心)**,详见 1.3 |
| `4-get_video_metrics.sh` | `drivinggen/z-sample_fvd.py --metric all` | 视频指标:FVD / 客观质量 / 主观质量 / 场景一致性 / Agent 一致性 / Agent Missing |
| `5-get_traj_metrics.sh` | `drivinggen/z-sample_ftd.py --metric all` | 轨迹指标:FTD / 质量组 / 一致性组 / ADE / DTW |

### 1.3 Agent 轨迹提取(脚本 3 → `extract_traj_agent_unidepth.py`)

**依赖模型**:UniDepthV2(`lpiccinelli/unidepth-v2-vitl14`,HF 下载)+ YOLOv10x(`ckpt/yolov10x.pt`,源码里硬编码作者路径 `/shared_disk/users/yang.zhou/...`,需自行下载放 `ckpt/`)+ SAMURAI(SAM2 视觉跟踪,`third_parties/samurai`)。

**真实算法**(与 README 描述有出入,以下为源码事实):
1. **只在第 0 帧跑检测**:YOLOv10x,类别 `[0,1,2,3,5,7]`(person/bicycle/car/motorcycle/bus/truck),conf ≥ 0.3;
2. 按 box 中心 x、y 排序,**逐个用 SAMURAI 全程跟踪**(首帧框初始化,后续帧纯视觉跟踪,**不做逐帧重检测**);
3. 每帧 UniDepthV2 推理深度;
4. 每帧生成"动态物体掩膜"(类别 0–8、14–23 的 box 置 0)供 VO 排除动态特征点;
5. 自车轨迹:特征提取 + 匹配(Lowe ratio 0.7)→ 位姿估计,`locs = [x, z]`;**GT 相机内参从硬编码 nuplan 路径读取**(`/mnt/cache/zhouyang/dg-bench/nuplan_1.1/val_sensor_data_10hz_0530/.../intrinsic.npy`)——**本地复现的最大移植障碍之一**;
6. Agent 轨迹:box 中心像素 + 掩膜内深度(2–98 百分位截断均值)→ `K_inv @ pixel * Z` 反投影 → `R,T` 变换到世界系 → 取 `(x, z)` 平面坐标。

**输出 pkl**(后续所有轨迹/Agent 指标的输入):
- `{outdir}/{s_name}/{model}/{exp_id}/unidepth-estimate_ego_traj.pkl`:`{'locs': (N,2), 'poses_3x3': ...}`
- `unidepth-estimate_agents_traj.pkl`:`[(frame_ids, global_coords, label), ...]`
- `unidepth-estimate_agents_bbox.pkl`:每 agent `[(frame_id, (x1,y1,x2,y2)), ...]`
- `unidepth-estimate_agents_bbox_label.pkl`:`"x1-y1-x2-y2" → 类别名`

**输入视频格式**:`root_path/<run>/<model_name>/<exp_id>/images/` 帧目录(`sorted()` 字典序,**文件名必须零填充**);GT 走 `CAM_F0/` 目录。z-sample_fvd.py 中评价时**丢弃第 0 帧**(GT 条件帧),期望 101 帧(0–100)。

**源码缺陷(复现时必须处理)**:
- 一处裸 `import pdb; pdb.set_trace()` 遗留在内参加载前——直接运行会卡死在调试器;
- 硬编码绝对路径 2 处(YOLO 权重、nuplan 内参);
- 帧排序依赖文件名字典序。

### 1.4 视频指标(脚本 4 → `z-sample_fvd.py` + `drivinggen/videos/`)

| 指标 | 实现文件 | 依赖模型 | 16GB 可行性 |
|---|---|---|---|
| FVD | `videos/video_distribution.py` | vendored StyleGAN-V 的 `fvd2048_100f`(I3D 2048 维特征,256 分辨率,100 帧) | 可行(256 分辨率 I3D) |
| 客观质量 | `videos/video_obj_q.py`(VBench 式技术指标) | pyiqa | 可行 |
| 主观质量 | `videos/video_sub_q.py` | CLIP-IQA+(pyiqa) | 可行 |
| 场景一致性 | `videos/video_v_consist.py` | DINO 特征 | 可行 |
| **Agent 一致性** | `videos/video_a_consist.py` | **DINOv3 ViT-H/16+**(`facebook/dinov3-vith16plus-pretrain-lvd1689m`,经 ModelScope 加载) | 边缘可行(需单独跑,fp16) |
| **Agent Missing** | `videos/video_a_missing.py` | **Cosmos-Reason1-7B(vLLM, gpu_mem_util=0.6, enforce_eager)** | **不可行/边缘**(7B VLM bf16 ~16GB,fp8 或量化后勉强) |

**Agent 一致性实现**(源码事实):
- 输入:脚本 3 产出的 bbox pkl + label pkl + 帧目录;
- 每个 agent 轨迹取首段(允许 ≤10 帧间隙的连续段),逐帧按 bbox 裁剪(crop 安全扩展,最小 32px);
- DINOv3 嵌入(`pooler_output`),计算:
  - **R(reference)**:各帧 vs 首帧余弦相似度均值,`clamp_min(0)`;
  - **A(adjacent)**:相邻帧余弦相似度均值;
  - S(语义 CLIP 分支)**已注释禁用**,权重 (0.5, 0.5, 0);
- 最终 `score = (R + A) / 2`,越高越一致;
- **注意源码 bug**:返回值用的是最后一个 scene 的 `scene_stability` 而非全 scene 均值 `scenes_stability`,且 return 后有不可达死代码、embedding 缓存路径逻辑失效(每次必 miss)。**复现时若对齐论文数值,需先确认作者口径,若自己重实现则按注释意图修。**

**Agent Missing 实现**(源码事实):
- 机制:对"非因到达视频末尾(`last_fid != 100`)而结束、且长度 ≥ min_track_len(调用处传 2)"的轨迹,取 6 帧(首次出现 2 帧 + 消失前 2 帧 + 消失后 2 帧),画绿框,送 **VLM(Cosmos-Reason1-7B)**判断消失是否自然,`<answer>` 含 `Unnatural` → 判为 missing;
- 边缘/遮挡豁免逻辑(出画边缘 margin、IoU 遮挡)在源码中**已被注释禁用**——纯 VLM 裁决;
- 场景级 score = 1 − 场景内 missing agent 占比,跨场景取均值,越高越好;
- **源码缺陷**:`types` 未 import 就被使用(NameError 风险)、`if __name__ == '__main__' or True:` 无条件 spawn、`sorted(...)` 结果未赋值等。**该指标是复现风险最高的一项(VLM 依赖 + 代码质量问题)。**

### 1.5 轨迹指标(脚本 5 → `z-sample_ftd.py` + `drivinggen/trajs/`)

| 指标 | 实现文件 | 是否需要 GT 轨迹 | 依赖模型 |
|---|---|---|---|
| FTD | `trajs/traj_distribution.py` | **是**(`data/ego_condition/ego_motion/{s_name}.npy`) | **MTR**(`ckpt/mtr-epoch=28-step=176552.ckpt`,Waymo 100% 官方权重,仅用 agent_polyline_encoder 做 11 帧滑窗特征,Fréchet 距离) |
| ADE / DTW | `trajs/traj_alignment.py` | **是** | 无(纯数值) |
| 质量组(舒适度/曲率/速度) | `trajs/traj_quality.py` | 否 | 无(纯 NumPy) |
| **一致性组(速度/加速度)** | `trajs/traj_consistency.py` | 否 | 无(纯 NumPy) |

**速度/加速度一致性——官方公式(源码原文,可直接移植)**:
```
v = ||diff(xy)/dt||,  dt=0.1 (10Hz)
静态过滤: max(v) < 0.1 → NaN(不参与统计)
速度一致性   = exp( -std(v)  / (mean(v)  + 1e-9) )
加速度: a = diff(v)/dt
加速度一致性 = exp( -std(a)  / (mean(|a|) + 1e-9) )    # 注意 μ 用 |a|,σ 用带符号 a
总一致性分   = 0.5 * (速度项 + 加速度项)                 # 源码是算术平均(文档字符串写的乘积是错的)
```
**质量组公式**:中心差分求 v/a/jerk + 航向角差分求 yaw rate → 时间聚合(均值或 95 分位)→ **按轨迹长度(米)归一** → `1/(1+x/τ)` 软映射(jerk/acc/yaw 三项)→ 几何平均;曲率 RMS 用叉积形式 `|x'y'' - y'x''| / (x'^2+y'^2)^1.5`;速度分 = `log1p(v)/log1p(2.5*6.0)` 截断 [0,1],静止轨迹记 0。

**预处理链**(所有轨迹指标共用):`smooth_traj_sg(dt=0.1, win_sec=0.4, poly=3)` Savitzky-Golay 平滑 + `slam_align_to_gt_fix_origin`(SVD 刚体对齐,无尺度,首帧钉在原点)。

**GT 轨迹格式**:`data/ego_condition/ego_motion/{s_name}.npy`,`gt[:101, :2]` 取世界系 x,y。

### 1.6 16GB 复现可行性总结(DrivingGen)

| 模块 | 可行性 | 缺什么 |
|---|---|---|
| 轨迹一致性 + 质量组 | ✅ **完全可行**(纯 NumPy,零模型) | 无 |
| 检测(YOLOv10x) | ✅ 可行 | yolov10x.pt 权重(~25MB,GitHub 可下) |
| SAMURAI 跟踪 | ✅ 可行 | SAM2.1 权重 + samurai 环境适配 |
| UniDepthV2 深度 | ✅ 可行 | HF 权重(~1.3GB,ViT-L) |
| Agent 一致性(DINOv3 ViT-H+) | ⚠ 边缘 | 大模型,单进程独占跑 |
| FVD(I3D 256px) | ✅ 可行 | stylegan-v 管线移植 |
| FTD(MTR) | ✅ 可行 | MTR Waymo ckpt + vendored MTR 代码 |
| Agent Missing(7B VLM) | ❌ 边缘不可行 | 量化 Cosmos-Reason1-7B 或换小 VLM(将偏离官方口径) |
| 数据集 | ⚠ 需 HF token | `yangzhou99/DrivingGen` 整仓大小未知,先问 token 与体积 |
| 生成侧(Wan2.2-14B 等) | ❌ 跳过 | 按既定策略 |

---

## 二、SynPeDAS(MDPI Electronics 15(14):3033)

### 2.1 仓库概况

- URL:https://github.com/levilovearch/SynPeDAS(WIP,7 commits,**无 License 文件,无代码**)
- README 自述结构 + TODO:"Generation pipeline"、"Evaluation suite" **均为待做**——评价能力为空,恰好是本研究的切入点
- 三个数据目录 + `repo_media/`(截图)

### 2.2 数据结构与对应关系(源码事实)

三类文件以**同一 base name** 关联:

```
video_{JAAD视频ID:04d}_0_{ID}_{N}b_f{start}-{end}
```

- `JAAD_crossing_moments/`:`{base}.mp4`,JAAD 真实行人横穿片段,每段 **150 帧**(JAAD 原始 30fps → 5 秒),ID 从 0002 到 0222 带空洞,GitHub 页面显示 ≥100 段(截断,总数待本地 clone 确认);
- `generated_video/`:`{base}.mp4` + **`{base}.json` 成对**;生成视频为 **77 帧**;
- `video_generation_prompts/`:`{base}.txt`,与 JSON 里的 prompt 字段内容一致。

### 2.3 JSON 元数据格式(实测样本 `video_0002_0_2_6b_f60-210.json`)

```json
{
  "name": "video_0002_0_2_6b_f60-210",
  "prompt_path": "/cosmos-predict2.5/assets/prompts/....txt",
  "prompt": "(场景全文描述……两名深色衣服行人从右侧走向左侧……)",
  "negative_prompt": "(标准负向提示词)",
  "seed": 0,
  "guidance": 7,
  "inference_type": "video2world",
  "input_path": "/cosmos-predict2.5/assets/JAAD_crossing_moments/video_0002_0_2_6b_f60-210.mp4",
  "resolution": "none",
  "num_output_frames": 77,
  "num_steps": 35,
  "enable_autoregressive": false,
  "chunk_size": 77,
  "chunk_overlap": 1
}
```

### 2.4 对事件一致性验证的三个决定性事实

1. **生成器是 Cosmos-Predict2.5 的 `video2world` 模式,条件输入就是对应 JAAD 真实片段**(`input_path` 指向 JAAD mp4)——所以真实/生成对是"同源条件生成",事件级对比有依据;
2. **帧数不匹配:真实 150 帧 vs 生成 77 帧**;`enable_autoregressive=false` + `chunk_size=77` 说明是单块生成。**逐帧索引对齐不可能,时序类约束必须归一化(时间比例/事件相对时刻)后再比**——这正是用户任务书中"不得事后强加精确时序条件"的落地处;
3. `resolution: "none"`、`seed: 0`、`guidance: 7`——分辨率未显式记录,需用 ffprobe/opencv 逐个实测;seed 全 0 意味着**无 seed 多样性可研究**;生成视频的 FPS 未见记录(Cosmos 常为 24fps,JAAD 为 30fps,需实测确认)。

### 2.5 评价侧缺口(= 研究机会)

仓库无任何:行人检测、跟踪、身份关联、轨迹提取、事件判定、一致性统计代码。**用户计划的"事件级时空约束验证器"落在作者自己声明的 TODO 空位上**;且数据(真实片段+生成视频+prompt+生成参数)四元组齐全,零生成成本即可启动。**注意无 License——仅作研究分析用, Redistribution 前需联系作者。**

---

## 三、Flow-World-Models(预研,未克隆)

- URL:https://github.com/facebookresearch/Flow-World-Models(FAIR+Inria;**CC BY-NC 4.0**;arXiv 2606.29059)
- 定位:在 **DINOv3 预训练特征空间**做 flow matching 的随机世界模型(DiT 预测器 + 1024 维投影头 + `time_dist_shift` 关键超参),解决"VAE 随机但表征弱 / 特征模型强但确定"的矛盾
- **仅有两个 benchmark**:FuturePerception(Waymo 真实数据,4 帧上下文预测 12 帧,以 COCO AP 检测/深度做下游评价)+ **Bouncing Shapes**(合成,32 帧 256×256,131k 训练/512 评测片段)——**无 Atari/MazeBall/DMC**
- 工程:**仅 uv(`uv sync` + `uv run torchrun -m main --cfg_path ...`),无预训练权重发布**;Waymo 转换需单独 TF 环境;示例命令 `--nproc_per_node=8`
- 5060 Ti 判断:Bouncing Shapes 任务**可行**(合成数据本地生成、256×256、单卡 torchrun 改 `--nproc_per_node=1` + 减 batch);Waymo 任务**不优先**(转换链重 + 数据许可申请)
- **注意**:README 未声明 python/torch 版本(在 pyproject/uv.lock 里),克隆后需实测;Waymo 许可禁止商业化用途,与用户"商业化仿真"的开题语境有张力,引用时留意

## 四、eb_jepa(预研,未克隆)

- URL:https://github.com/facebookresearch/eb_jepa(**Apache-2.0**;arXiv 2602.03604;806 stars)
- 定位:能量式 JEPA 的轻量参考实现库(I-JEPA 图像 / V-JEPA 视频 Moving MNIST / AC-JEPA "Two Rooms" 动作条件世界模型+规划)
- 工程:Python 3.12 + uv;**无预训练权重**;默认配置按 H100 调优,A100/V100 建议减 batch;单卡数小时级
- 5060 Ti 判断:CIFAR-10 / Moving MNIST / Two Rooms 三个例子**均可本地训练**(小模型);
- 对本研究的意义:EB-JEPA 的**能量函数可作为"预测不确定性"的信号源**(能量高 = 状态转移不可信),与"检测置信度、遮挡状态"并列成为 Uncertain 判据之一——这是"是否引入 JEPA"问题的技术连接点(详见 research_gap.md)

---

## 五、两个仓库尚未克隆的说明

审计时 Bash 写通道被安全分类器临时阻塞(见 `failure_log.md`),本地克隆延后执行。上述分析全部来自 GitHub 同 commit 的原始文件抓取,克隆完成后仅需做抽查校验(文件数、目录结构、视频元数据实测),预计与本文一致。**以下事项必须等本地克隆后才能定论**:
1. SynPeDAS 三个目录的**确切文件总数**(GitHub 页面列表在 100 项截断);
2. 生成视频的**实际分辨率/FPS**(JSON 记录为 "none");
3. DrivingGen vendored third_parties 的完整性与权重清单;
4. `data/` HF 数据集的真实目录结构与体积。
