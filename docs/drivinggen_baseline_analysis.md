# DrivingGen 基线对齐分析（E2.1 预研，只读源码未运行）

日期：2026-11-05 | 源码：`repos/DrivingGen`（HEAD 48ed356，ICLR 2026）
对照对象：`src/wmeval/`（E1 v3.1）+ HF 数据（`datasets/raw/DrivingGen` 444 条件）
结论等级：逐项标注【可公平对齐 / 域内对照 / 不可比】。

---

## 1. 官方管线结构（源码级事实）

| 阶段 | 官方实现（文件） | 关键细节 |
|---|---|---|
| 轨迹提取 | `func/extract_traj_agent_unidepth.py` | YOLOv10x 检测（**权重路径硬编码**）→ SAMURAI mask 跟踪 → UniDepthV2 深度（mask 内 median_top）→ `reconstruct_global_trajectory`：像素+深度+内参 K+相机位姿 → **世界系 (x,z)**；ego 侧另有 visual_slam（`extract_traj_ego_unidepth.py`） |
| Agent Missing | `videos/video_a_missing.py`（810 行） | vLLM + **Cosmos-Reason1-7B**：取消失时刻周围 6 帧（I1–I6 首末+后继帧），问"Natural（遮挡/出画）vs Unnatural（非物理消失）"；**几何豁免被注释禁用**，裁决全在 VLM |
| Agent Consistency | `videos/video_a_consist.py`（284 行） | **DINOv3**（vith16plus）pooler 特征：R=生成帧 vs 参考帧余弦、A=相邻帧余弦；agent bbox `safe_crop_expand` 裁剪；已知缺陷（P-5）：返回值取最后 scene 而非全 scene 均值 |
| 轨迹指标 | `traj_consistency.py`/`traj_quality.py` | vel/accel 一致性（exp(−σ/μ)）、comfort、curvature、speed——**wmeval 已公式级移植**（selftest 35 项含解析对齐用例） |
| 汇总 | `z-sample_fvd.py` 等 | 全场 `np.nanmean` 单标量（事件级信息被平均掉——研究缺口 2 的源码证据） |

## 2. 逐指标对齐表

| 指标 | 官方口径 | wmeval 口径 | 对齐性 |
|---|---|---|---|
| vel/accel consistency | **世界系米单位**（UniDepth+内参+SLAM 全局轨迹） | **像素系**（diag 缩放阈值，公式一致） | 公式【可公平对齐】（selftest 验证）；**绝对数值不可比**——单位不同。可做秩相关对照（E4 计划：同一批视频双口径秩相关） |
| comfort / curvature / speed | 世界系 | 像素系 | 同上 |
| Agent Missing | VLM（7B，多 GPU 32GB+）Natural/Unnatural | 几何+跟踪归因（遮挡/重检测/碎片/质量/截断）三态 | **不可直接比**（判定主体不同）。【域内对照】：同一批视频各自口径报告 missing/异常率，差异即方法对比——论文的 B1-vs-B3 实验正是此设计。16GB 无法跑官方 VLM（记录为 P-4 阻塞） |
| Agent Consistency R/A | DINOv3 pooler 余弦 | 未实现（E5 可选） | 【可移植】：DINOv3 推理 16GB 可行；语义对齐（外观漂移层）；需修官方"最后 scene"缺陷并双口径报告 |
| FTD | MTR 特征 Fréchet | 未实现 | 【不可比】（需 MTR ckpt 未下载；第三方依赖重） |
| FVD | StyleGAN-V I3D | 未实现（B0 用 pyiqa PSNR/SSIM/LPIPS 替代） | 【不可比】（实现栈不同；B0 自立基线，不冒充官方口径） |
| 事件级判定 | **无**（全场平均） | 事件协议（L1-L4+三态） | 本研究增量（缺口 2），非对齐项 |

## 3. 数据输入对齐

- 官方 HF 数据（已下载 2.3G）：444 评测条件 = open_domain 222
  （caption+首帧）+ ego_condition 222（caption+首帧+**ego_motion
  jpg+npy 数值轨迹**）；
- **ego npy 轨迹是官方对齐捷径**：wmeval 未来世界系复算可直接消费
  该 ego 轨迹（无需 SLAM），agent 轨迹仍需检测+深度（UniDepth 路线）；
- 官方评测域为 nuPlan/nuScenes 派生；wmeval 当前域为 SynPeDAS
  （JAAD 派生）——**所有指标只能在各自域内报告**，跨域只比方法学。

## 4. 公平比较的结论性判据（写论文用）

1. **能公平对齐**：轨迹公式的公式级一致性（已完成）+ 秩相关对照
   （E4：同视频 px/m 双口径）；
2. **域内对照**：Agent Missing 的 VLM-vs-几何两口径同批报告
   （B1-vs-B3 核心实验；官方 VLM 在本机不可行为记录性事实）；
3. **不可比必须声明**：FVD/FTD（实现栈/依赖不同）、绝对速度
   （单位）、跨域任何指标；
4. **无法运行项**：官方 Agent Missing VLM 全量（VRAM）、UniDepth
   世界系全量（E4 可选路径，依赖 UniDepthV2 权重 ~1.3GB 可行）。

## 5. B1 基线实验的最低可行口径（下一轮启动条件之一）

- 输入：SynPeDAS 8 审计样本 + Val 集子集（人工标注后）；
- 官方公式口径：wmeval px 空间（已就绪）+（可选）UniDepth 世界系；
- 官方 VLM 口径：降级为"记录性引用论文数值 + 本机不可行声明"，
   不冒充复现；
- Agent Consistency：若 E5 时间允许，DINOv3 移植双口径。
