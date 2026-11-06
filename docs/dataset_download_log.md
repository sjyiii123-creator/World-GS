# 数据集下载执行记录（E2 后续，2026-11-05 晚）

工具包：`repos/wm_dataset_download_kit_20261104.zip`（ChatGPT 侧准备）
解包至：`repos/wm_dataset_kit/`（隔离目录，脚本先审查后执行）

## 1. 脚本安全审查（已通过）

`wm_dataset_setup_v2.sh` 逐行审查结论：
- `set -euo pipefail`；**无** `rm -rf` / `sudo` / `curl|sh` / 覆盖已有目录；
- `clone_if_missing`：目标存在即跳过（EXISTS, not overwriting）；
- 大下载门控 `CONFIRM_DOWNLOAD=YES`（synpedas_full / jaad_videos /
  drivinggen_data / drivejudge_data / smd_data）；
- 内嵌 Python 经 heredoc、路径走 argv（符合不受信数据处理规范）；
- 下载走官方源（GitHub / data.nvision2.eecs.yorku.ca / HF 官方 repo），
  无第三方重打包。
- 逻辑与用户描述、`wm_dataset_plan_20261104.md` 一致。

## 2. 本地审计（已用独立命令完成，与脚本 audit 等价）

| 项 | 结果 |
|---|---|
| SynPeDAS real/gen/prompt/json | **211 / 211 / 211 / 211（完整，2.6G，无需重下）** |
| DrivingGen 代码 | PRESENT（HEAD 48ed356，origin 官方） |
| 磁盘 | 637G 可用 |
| huggingface_hub | wm_eval 环境未装 → 需 `pip install huggingface_hub` |

## 3. HF 数据集访问预检（2026-11-05 WebFetch，无需下载即确认）

| 数据集 | 访问 | 体积 | 许可 | 内容预检 |
|---|---|---|---|---|
| yangzhou99/DrivingGen | **非 gated，无需 token**（P-8 记录已过时） | 2.24GB | Apache-2.0 | imagefolder，~22.9k 行（image+label，224 类）——形态与"评价输入+条件"描述有出入，**下载后须核查文件树再定性** |
| JiaweiZhou/SafeMVDrive | **非 gated** | 7.97GB | CC BY-SA 4.0 | nuScenes 格式，41 场景×9s×6 视角+3D 车辆框；构建自 nuScenes 验证集 + VLM 引导对抗轨迹 + UniMLVG 生成 |

## 3b. JAAD→SynPeDAS 身份映射发现（2026-11-05）

SynPeDAS sample_id 中段即 JAAD 行人 ID：
`video_0016_0_16_72b_f0-150` → JAAD 视频 0016、行人 `0_16_72b`
（'b'=行为标注行人）、帧窗 f0-150。JAAD_2.0 的
`annotations_attributes/`（crossing 特征）与 `annotations/`
（逐帧 bbox + occlusion 0/1/2）可按 ID+帧窗直接映射到 211 个样本，
产出**真实侧弱 GT**（人工标注交叉校验 + 遮挡归因审计）。
映射脚本已备：`src/build_jaad_gt_evidence.py`（克隆完成后运行）。

## 4. 执行结果（2026-11-05 晚，逐项回填）

| 任务 | 状态 | 产出 |
|---|---|---|
| 脚本安全审查 | ✅ | 见 §1（无危险操作、门控完整） |
| 本地审计 | ✅ | SynPeDAS 211×4 完整（2.6G），无需重下 |
| jaad_annotations | ✅ | `repos/JAAD`（JAAD_2.0 分支，346+346 XML 全量） |
| drivinggen_code | ✅ 已存在 | `repos/DrivingGen`（HEAD 48ed356） |
| drivejudge_code | ✅ | `repos/DriveJudge`（914a78f，88M） |
| smd_code | ✅ | `repos/SMD`（fbb7d55，232M） |
| WorldLens 代码 | ✅ 追加 | `repos/WorldLens`（fa6c5ff，142M） |
| huggingface_hub | ✅ | wm_eval 环境安装 1.8.0 |
| drivinggen_data | ✅ 完成(重试 1 次成功,F-11) | `datasets/raw/DrivingGen` 2.3G:**open_domain 222 + ego_condition 222 条件全配对**(caption/首帧图/ego_motion) |
| safemvdrive | ✅ 下载+解压完成 | 7.5G data.tar.gz → 8.8G `datasets/raw/SafeMVDrive/4_30_adversarial_trajectory_gen_41_itmes/` |
| verify | ✅ 已跑（OpenCV 等价实现） | 严格配对 **211/211**；抽查 3 样本 × real/gen 共 6 视频全部可解码（150f@30 1920x1080 / 93f@16 1280x704） |

**SafeMVDrive 解压验证**：nuScenes 格式全表在位——scene.json **41 场景** ✓、
sample 779、sample_data 31,283、sample_annotation **21,995 条 3D 框** ✓、
6 相机通道全在（CAM_FRONT/BACK/LEFT/RIGHT ×2）+ LIDAR_TOP sweeps；
场景名含 collision/adv 元数据（对抗轨迹即明确的生成事件条件）。
注意：本机无 ffprobe 二进制，视频核验用 OpenCV 等价实现。

**DrivingGen 数据定性（下载后实测）**：444 个评测输入条件——
`ego_condition/`（caption 文本 + imgs 首帧图 + ego_motion **jpg+npy 数值
轨迹** 各 222）与 `open_domain/`（caption+imgs 各 222）；caption 为
nuPlan 域文本描述（BE/DE 等前缀）。**注意：这是生成任务的输入条件集，
非已生成结果**——B1 官方对齐可直接消费 ego npy 轨迹（无需 UniDepth
估计 ego 运动），与 wmeval 的轨迹指标对照。
ego_motion 444 文件 = 222 jpg + 222 npy。

**代理问题与修复**：hf CLI 首次失败——环境 `ALL_PROXY=socks://127.0.0.1:7891`
的 `socks://` scheme 不被 httpx 支持；修复 = 清除 ALL_PROXY 保留
`http://127.0.0.1:7890`（git 同路径成功）。包装脚本
`tools/hf_download.sh` + python snapshot_download 形式均内置该修复。

## 5. JAAD 弱 GT 产出（本轮新增资产）

`datasets/annotations/jaad_gt_evidence.csv`（211 行，211/211 匹配）：
- 字段：crossing（195=1/10=0/6=-1）、motion_direction（198 LAT/11 LONG）、
  crossing_point、decision_point、signalized、num_lanes 等 per-ped 属性
  + 窗口内逐帧统计（gt_frames、occ_partial/full、outside）；
- **遮拦审计素材**：158/211 样本窗口内有官方遮挡标注
  （occ_partial 帧 1-14+）——可校验 wmeval 的 occlusion 归因；
- **片段截断证据**：123 个 crossing_point 落在窗口内，均值 t=0.20，
  70 个 t<0.2（多数片段从横穿中途开始——partial_clip_cut 语义的
  数据侧印证）；
- E1 十样本：8 个 crossing=1；**0035/0037 crossing=0 且 dir=LONG**
  （非横穿样本——用户此前对 prompt 的判断获 GT 证实；
  0035 正是 real↔gen 方向反转的样本）；
- 语义约束：REAL-side weak GT，不得填入人工真值列
  （`build_jaad_gt_evidence.py` 自动附 semantics 声明）。

## 6. E8' 泛化子集（JAAD 官方划分交叉）

SynPeDAS 166 源 ∩ JAAD 官方 split：train 90 源（121 样本）、
val 14 源（15）、**test 55 源（68）**。扣除 E1 开发的 10 源后
**E8' 候选 = 54 源 / 67 样本**（JAAD test ∩ SynPeDAS ∩ 非 E1）。
建议：e2_split 的 test 侧与该子集取并集报告（两套"未见"定义：
我们的组划分 + JAAD 官方 test）。

```bash
ROOT=/home/moz/World-GS
S=$ROOT/repos/wm_dataset_kit/wm_dataset_setup_v2.sh

# 第一批（无大下载，无需门控）
bash $S jaad_annotations $ROOT
bash $S drivinggen_code  $ROOT      # 已存在会自动跳过
bash $S drivejudge_code  $ROOT
bash $S smd_code         $ROOT

# hf CLI（独立环境安装）
/home/moz/miniconda3/envs/wm_eval/bin/pip install -U huggingface_hub

# 第二批（门控大下载，PATH 注入 hf）
PATH=/home/moz/miniconda3/envs/wm_eval/bin:$PATH \
  CONFIRM_DOWNLOAD=YES bash $S drivinggen_data $ROOT   # ~2.24GB
PATH=/home/moz/miniconda3/envs/wm_eval/bin:$PATH \
  CONFIRM_DOWNLOAD=YES bash $S smd_data $ROOT          # ~7.97GB

# 核验
bash $S verify $ROOT
```

## 4. 阻塞记录

Bash 安全分类器间歇不可用（同 failure_log F-01 模式），执行窗口随机；
只读操作不受影响。恢复后按 §3 顺序执行，逐项在本文件回填状态。

## 5. 许可速记（承接 dataset_survey.md §6）

- JAAD 2.0：代码 MIT；视频 CC BY 4.0（以官方页面为准）
- DrivingGen：代码 Apache 2.0；HF 数据页声明 Apache 2.0（需核对图像来源）
- SafeMVDrive：CC BY-SA 4.0（基于 nuScenes，需核对来源协议）
- SynPeDAS：公开访问≠可再分发，引用按"实验起点"处理
