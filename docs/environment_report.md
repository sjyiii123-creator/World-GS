# 环境审计报告

**生成时间**:2026-11-04 15:58–16:10(本地时间)
**审计方式**:全部为实际执行命令的结果,未经任何推测。审计时 GPU 处于轻度占用状态(桌面环境 + Firefox + Zotero,共约 590MiB 显存)。

---

## 1. 操作系统与内核

| 项 | 值 | 来源命令 |
|---|---|---|
| 发行版 | Ubuntu 24.04.4 LTS (noble) | `lsb_release -a` |
| 内核 | 7.0.0-38-generic x86_64 | `uname -a` |
| 是否 WSL2 | **否**(原生 Linux;`/proc/version` 无 microsoft 字样,`WSL_DISTRO_NAME` 未设置) | `/proc/version`, `env` |
| 主机名 | moz-B760M-GAMING-WIFI6-PLUS-GEN5 | `uname -a` |

## 2. GPU 与驱动

| 项 | 值 | 说明 |
|---|---|---|
| GPU | NVIDIA GeForce RTX 5060 Ti | Blackwell 架构,**sm_120**(`torch.cuda.get_device_capability() == (12,0)`) |
| 显存 | 16311 MiB(约 16GB) | nvidia-smi 报告 |
| 驱动版本 | 595.99.02 | `nvidia-smi` |
| 驱动支持的 CUDA 上限 | 13.2 | ⚠ **这是驱动兼容上限,不是本机安装的 CUDA Toolkit 版本** |
| 审计时显存占用 | 589MiB / 16311MiB | Xorg、Firefox、Zotero 等桌面进程 |
| CUDA Toolkit(实际安装) | **12.8**,位于 `/usr/local/cuda-12.8` | `nvcc` 不在 PATH,需 `/usr/local/cuda-12.8/bin/nvcc` 或自行 export PATH |
| 驱动 595 + Toolkit 12.8 | 兼容(驱动向后兼容旧 Toolkit) | 正常组合 |

**RTX 5060 Ti 关键约束**:sm_120 架构要求 PyTorch ≥ 2.7 且 CUDA ≥ 12.8 构建;所有 CUDA 11.x 官方环境(如 WorldLens)必须移植。

## 3. CPU / 内存 / 磁盘

| 项 | 值 | 说明 |
|---|---|---|
| CPU 线程数 | 20 | `nproc` |
| 系统内存 | 15Gi 总计,审计时 8.3Gi 可用 + 6.3Gi buff/cache | **16GB 物理内存是隐性瓶颈**:CPU offload、大规模并行编译(建议 `ninja -j4`)、nuScenes 全量处理都受它限制 |
| Swap | 31Gi | 充足 |
| 根分区 | 915G 总量,650G 可用(26% 已用) | nvme1n1p2,空间充足 |

## 4. 编译工具链

| 工具 | 版本 |
|---|---|
| gcc / g++ | 11.5.0(Ubuntu 24.04 官方包) |
| cmake | 3.28.3 |
| git | 2.43.0 |
| ffmpeg / ffprobe | **不在 PATH**(如需视频处理,优先用 Python 的 opencv/imageio-ffmpeg,或后续经确认后安装) |

gcc 11.5 + CUDA 12.8 组合在本机已被验证可用于编译 CUDA 扩展(用户既有 3DGS 环境即此配置)。

## 5. Python 与 Conda

- **Miniconda3** 位于 `/home/moz/miniconda3`,conda 26.7.3,**base 环境为 Python 3.14.6,未安装 PyTorch**(base 仅作包管理入口,符合良好实践)。
- **uv 不在 PATH**(但存在 `~/.cache/uv`,说明曾用过 uv)。
- 现有 9 个 conda 环境,全部为既有 3DGS/具身智能项目专用,**不属于本世界模型项目,不得改动**:

| 环境名 | torch 版本 | GPU 识别 | 用途线索 |
|---|---|---|---|
| fastgs5060 | 2.7.1+cu128 | ✅ RTX 5060 Ti (12,0) | FastGS |
| fastpgsr_bw | 2.7.1+cu128 | ✅ | PGSR |
| gs5060_3dgs | 2.7.1+cu128 | ✅ | 3DGS 基线 |
| gs5060_absgs | 2.7.1+cu128 | ✅ | AbsGS |
| gs5060_clean | 2.7.1+cu128 | ✅ | 干净基线(含 numpy 1.26.4 / opencv-headless 4.11 / scipy / pandas / matplotlib) |
| gs5060_pixelgs | 2.7.1+cu128 | ✅ | Pixel-GS |
| gs_geom | 2.7.1+cu128 | ✅ | 几何研究(numpy 2.2.6) |
| partslip5060 | 2.7.1+cu128 | ✅ | PartSLIP(**含源码编译的 mmcv 2.1.0 + mmdet 3.3.0 + transformers**) |
| rtx5060_embodied | **2.10.0+cu128** | ✅ | 具身智能(huggingface_hub 1.28) |

**结论:torch 2.7.1+cu128 与 torch 2.10.0+cu128 均能正确识别 GPU(sm_120)。本世界模型项目应新建独立环境,参考此已验证的版本组合。**

mm 栈(mmcv/mmdet)在 Blackwell 上的源码编译已在 partslip5060 环境完成过一次,该经验可直接迁移到 WorldLens 等需要 mm 栈的仓库。

## 6. 现有模型权重与数据集

- HuggingFace 缓存(`~/.cache/huggingface`,共 572M):
  - `models--laion--CLIP-ViT-B-16-laion2B-s34B-b88K`(CLIP 文本/图像编码器,可用于 CLIPScore 类指标)
  - `datasets--MinghanQin--lerf_ovs`(LERF 数据集,3DGS 用,与世界模型研究无关)
- torch 缓存(`~/.cache/torch`,761M):hub 下载缓存。
- **未发现**:YOLO、SAM/SAM2、UniDepth、任何视频生成模型权重;未发现 nuScenes/JAAD/BDD 等驾驶数据集。
- 用户既有数据(3DGS 相关,勿动):`/home/moz/3DGS`(PGSR、street_gaussians)、`/home/moz/3D_GS_ideal`、`/home/moz/workspace/3dgs_geometry_study`(drivestudio、street_gaussians、day1_se3_lab)。

## 7. 网络连通性

- GitHub 可达:`git ls-remote` 对 DrivingGen(HEAD `48ed3569...`)和 SynPeDAS(HEAD `f34227f...`)均成功。
- HuggingFace 此前有过下载记录(cache 存在)。

## 8. 对本研究的直接结论

1. **硬件满足"评价管线"级实验**:检测/跟踪/深度/CLIP 类评价模型在 16GB 显存内可行;视频扩散生成(720p Cosmos 等)不可行或需大量 offload,按计划跳过。
2. **必须新建独立 conda 环境**(如 `wm_eval`),Python 3.10–3.12 + torch 2.7.1+cu128,不碰既有 9 个环境。
3. CUDA Toolkit 12.8 已装(只是不在 PATH),如需编译扩展:`export PATH=/usr/local/cuda-12.8/bin:$PATH; export CUDA_HOME=/usr/local/cuda-12.8`。
4. 系统内存 16GB 是比显存更隐蔽的限制:避免大并行编译、避免一次性加载大数据集、视频处理逐帧/分片进行。
