# J2 环境报告（阶段 1）

日期：2026-11-05 | 工具：`WorldGS_J2_Configuration/tools/env_audit.sh`
（外部审计包，SHA256 7/7 校验通过；只读、零安装、零下载）

## 审计结果（wm_eval 环境运行）

| 项 | 值 |
|---|---|
| Python | 3.9.23（wm_eval conda 环境） |
| torch | 2.7.1+cu128，CUDA 可用，bf16 支持 |
| GPU | RTX 5060 Ti，16311 MiB 总量 / **15168 MiB 空闲**，驱动 595.99.02 |
| 依赖 | numpy / transformers / yaml / cv2 / safetensors 全部就位 |
| DINOv2 权重 | `ckpt/dinov2-small`（本地，未重复下载） |

**注意**：直接 `bash env_audit.sh` 会落到 base 环境 python 3.14（无
torch）——审计必须 `PATH=wm_eval/bin` 前置运行（已记录于
commands.md；环境快照存 `runs_env_audit.log`）。

## 保护核验

J0/J1/J3-lite 原始结果、人工标注、原始视频与 Prompt 零改动；
新增产物全部位于 `runs/j2_experimental/`（独立目录纪律）。
