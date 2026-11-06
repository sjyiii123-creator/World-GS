# SynPeDAS 生成条件审计（P0-4）

日期：2026-11-05（E2.1）| 证据：本地数值实验（`runs/e2_1_conditioning/`）+
SynPeDAS JSON 参数普查 + Cosmos 官方文档（Predict2 与 Predict2.5）
+ HF 模型卡。**模型未查看任何图像**——帧相似度为 OpenCV 数值计算。

## 结论先行

**`conditioning_frames_scope = likely_first_~16_frames_from_real_video_replayed_then_77_generated`**
（数值强证据 + 结构吻合；确切机制 unknown——见 §3 三假说）。

**最重要的协议推论**：video2world 的生成合同是"（真实视频前段条件 +
prompt 文本）→ **模型自己的未来 roll-out**"，**不包含"复制源视频后续"**。
因此 Source-Behavior-Preservation 评价（E2 方向审计协议的第二问）**没有
默认合同依据**——只有 prompt 文本条件是可评价的显式条件；源视频后段
行为未被协议要求保留。这一条直接修正 event_protocol 的语义基础。

## 1. 输入证据（全部本地可复核）

### 1.1 SynPeDAS JSON 参数普查（8 样本 + manifest 全量一致）

| 字段 | 值 | 含义 |
|---|---|---|
| inference_type | `video2world`（全部 211） | 视频到世界生成 |
| input_path | `/cosmos-predict2.5/assets/JAAD_crossing_moments/<sid>.mp4` | **完整真实片段**作为输入（容器内路径） |
| num_output_frames / chunk_size | 77 / 77 | 单次生成 77 帧 |
| chunk_overlap | 1 | chunk 交叠 1 帧 |
| **enable_autoregressive** | **False** | **单 chunk、无自回归续写** |
| num_steps / guidance / seed | 35 / 7 / 0 | 采样参数 |
| prompt / negative_prompt | 文本条件 | 直接进入模型的条件文本 |
| resolution | `none`（未指定→用模型默认 720P=1280×704，与实测一致） | |

### 1.2 实测帧结构（关键数值证据）

生成视频实测 **93 帧 @16fps 1280×704**（JSON 声明 77）——**93 ≈ 16+77**。

逐帧灰度差实验（真实视频按 16fps 重采样对齐后与生成帧比较，
`runs/e2_1_conditioning/frame_similarity_full.csv`）：

| 样本 | d[0:16] 均值 | d[17] d[18] d[19] | 最大跳变 | 末 20 帧均值 |
|---|---|---|---|---|
| 0002 | 12.7 | 19.0 19.9 21.6 | +2@f9（无硬跳变） | 41.3 |
| 0003 | 10.3 | 12.8 12.9 13.0 | +1@f6（无） | 19.6 |
| 0016 | 10.5 | 13.5 14.0 14.0 | +2@f1（无） | 22.4 |
| **0024** | **9.1** | **21.5 23.5 → 95.2** | **+72@f18（硬跳变）** | 85.0 |
| 0030 | 16.1 | 25.3 26.6 26.8 | +2@f14（无） | 42.1 |
| 0035 | 17.3 | 21.8 21.8 21.9 | +2@f1（无） | 52.9 |
| 0037 | 24.3 | 32.6 32.0 32.8 | +4@f1（无） | 49.5 |
| 0044 | 7.7 | 8.9 9.2 9.2 | +2@f1（无） | 13.1 |

**8/8 样本前 ~16-18 帧差异显著低于后段**；0024 在 f18 出现 +72 的硬跳变
（夜景场景内容分布差异大，切换点不被平滑掩盖）；其余 7 样本为平滑发散
（预测不确定性累积模式——与"条件重绘→自由生成"兼容，但分界被场景
相似性掩盖）。重采样相位误差（29.97→16fps）可解释 16 与 18 的 ±2 偏差。

### 1.3 官方文档交叉验证

- **Cosmos-Predict2**（`documentations/inference_video2world.md`）：
  `--num_conditional_frames` 选项 **1 或 5（默认 1）**；5 帧模式取输入
  视频**最后** 5 帧；单次前向=一个 chunk；**输出是否含重绘条件帧未说明**。
- **Cosmos-Predict2.5 HF 模型卡**（nvidia/Cosmos-Predict2.5-2B）：
  video2world 输入"**5 帧**视频（1280×704）"；输出 ~5s@16fps；
  架构线索：输入 latent 与生成帧**沿时间维拼接**参与去噪（暗示条件帧
  参与输出，但未确认呈现方式）。VRAM 32.5GB（SynPeDAS 作者侧运行）。
- **SynPeDAS 仓库**：README 的 Generation pipeline 与 Evaluation suite
  均为 **TODO**——生成代码未随仓库发布，无法查证其调用参数。

## 2. 三种相容假说（机制层 unknown）

| 假说 | 内容 | 支持/反对证据 |
|---|---|---|
| H1 重绘拼接 | 输出=真实前 16 帧（重采样重编码）+ 77 生成帧 | 支持：93=16+77 精确吻合；前段差 4-24 符合重编码噪声+相位误差；0024 硬跳变。反对：官方条件协议为 5 帧（2.5 模型卡）——16≠5 无法从协议推出 |
| H2 模型重绘 | 前段是模型对条件帧的高保真重建（5 帧条件→输出前段重建+外推） | 支持：模型卡"latent 沿时间拼接"。反对：无法解释 16 帧长度与 5 帧输入的差 |
| H3 长条件协议 | 2.5 的某种 16 帧条件模式（文档未覆盖） | 无法验证（文档缺） |

**判据缺失**：SynPeDAS 生成脚本未公开。需要向作者索取或在
cosmos-predict2.5 复现实验（16GB 显存 <32.5GB 需求，不可行）。

## 3. 对七个必答问题的回答（P0-4 任务清单）

| # | 问题 | 回答 | 证据等级 |
|---|---|---|---|
| 1 | video2world 实际输入包含哪些视频帧？ | **官方协议：5 帧**（2.5 模型卡）；SynPeDAS 实际用法 unknown（代码未公开）；输出前 ~16 帧含真实视频内容（数值） | 5 帧：官方文档；16 帧：本地数值 |
| 2 | 是否使用完整真实视频？ | **否**——末 20 帧与真实对应帧差 13–85（发散）；仅前段进入条件 | 数值（8/8） |
| 3 | 是否只使用历史片段？ | 是前段（上下文式），非全历史、非"最后 5 帧"（输入是完整文件但仅前段生效） | 推断 |
| 4 | 首帧或部分参考帧？ | 条件为前段帧 + 文本；像素级来源（拼接 vs 重绘）unknown | 数值+文档 |
| 5 | 文本 prompt 如何进入模型？ | 作为 video2world 的文本条件（JSON 的 prompt/negative_prompt 字段即条件文本；无其他条件通道证据） | JSON 字段语义 |
| 6 | 生成视频与原始视频时间范围如何对应？ | 生成 t∈[0,~16/93]≈前 1s 对应真实前段（重采样）；**t 之后是模型自身的未来 roll-out——与真实视频后 4s 不是同一物理时间段，且无合同要求对应** | 数值+协议推论（本轮核心发现） |
| 7 | 生成任务要求保留什么信息？ | 可证实的显式条件只有 prompt 文本；**无任何字段要求保留源视频行为**——Source-Behavior-Preservation 无默认合同 | JSON 全字段普查 |

## 4. 对 E2 评价协议的直接修正（联动 P0-5）

1. **event_protocol 的 condition_source 语义更新**：条件 = prompt 文本 +
   真实前段帧（~16 帧）；**不含**源视频后续行为。
2. **0044 类冲突的重新理解**：prompt 描述的是**完整原视频**的行为
   （含后段），但模型的条件帧只有前段——横穿发生在原视频的哪个时段
   决定了模型能否"看到"它（JAAD GT：0044 crossing_point=74 在窗口
   f60-210 的 t≈0.09 早期——**条件帧覆盖范围内**，值得人工比对条件帧
   里行人是否可见）。
3. **生成段行为与源后段的分歧不再自动算违约**——评价只问"prompt
   文本事件是否在生成 roll-out 中发生"。
4. **时间对齐协议修正**：t_norm 相位对齐仅对前 ~17 帧有物理意义
   （条件段）；生成段（t>0.18）与真实后段**不存在可比时间轴**——
   此前 contact sheet 的全程对齐仅是排版约定，报告需注明。

## 5. 证据缺口清单（无法进一步确认的部分）

- SynPeDAS 生成脚本（仓库 TODO）——**决定性缺口**；
- cosmos-predict2.5 中 video2world 输出结构的权威文档（HF 卡/官方 docs
  均未给出 93=16+77 公式）；
- 16 帧前段的像素来源（拼接 vs 重绘）：可由像素哈希/压缩域特征进一步
  区分（本轮未做，列为可选后续）；
- 5 帧条件的帧号选择（首 5？末 5？SynPeDAS 传整段视频时框架如何取）。

## 6. 来源

- 本地：`repos/SynPeDAS/generated_video/*.json`（211 份）、
  `runs/e2_1_conditioning/frame_similarity{,_full}.csv`（OpenCV 数值）
- [Cosmos-Predict2 inference_video2world 文档](https://github.com/nvidia-cosmos/cosmos-predict2)
- [nvidia/Cosmos-Predict2.5-2B 模型卡](https://huggingface.co/nvidia/Cosmos-Predict2.5-2B)
- [cosmos-predict2.5 仓库](https://github.com/nvidia-cosmos/cosmos-predict2.5)
- [NVIDIA Cosmos 官方文档](https://docs.nvidia.com/cosmos/latest/predict2/reference.html)
