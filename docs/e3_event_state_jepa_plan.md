# E3 Event-State JEPA 阶段计划（E3-D0 / E3-A / E3-B）

日期：2026-11-06 | 依据：`docs/jepa_paper_audit.md`（论文核验）+ E2 资产盘点
| 纪律：沿用 D4.1.1 的文件化、先自测后运行、结果落盘、不重复执行

## 0. 定位

> World-GS 要证明的不是 JEPA 能生成更好的未来，而是 JEPA 提供的事件
> 时序表示能否改善生成视频的**条件满足性验证**。

核心研究假设（留给 E3 实验回答，不得提前作为结论）：

$$\text{Event-Aligned JEPA Representations}
\overset{?}{\longrightarrow}
\text{Improved Conditioned Event Verification}$$

## 1. 现有资产盘点（2026-11-06）

| 资产 | 位置 | 状态 |
|---|---|---|
| 事件协议（13 分层字段） | `datasets/annotations/prompt_source_generation_audit_v2.csv` + `docs/e2_1_event_protocol.md` | 机器列已填，人工列 PENDING（8 样本） |
| 事件级真值表（20 行） | `datasets/annotations/e1_manual_labels.csv` | **全空，待人工标注**（E3-B 门控条件） |
| 事件协议全集 | `datasets/annotations/event_protocol.csv`（212 行） | event_required 三值待 val 子集复核 |
| 观测轨迹（图像平面） | `runs/synpedas_smoke_v2/<sample>__{real,gen}/tracks.json`（10 对视频，person 轨迹含 norm_centers） | 可用（E3-A 输入） |
| 行人序列 + 真实未来框 | `datasets/processed/actor_sequences/samples.csv`（pair_id、anchor_frames 12 帧、cand_frames 12 帧、label=same_identity/different_identity） | 可用（E3-D0 输入） |
| 预测器 + 编码器 | `runs/d4_experimental/d4_lam0.5.pt`、`runs/j2_2_experimental/b4_residual_expanded.pt`、`ckpt/dinov2-small` | 已验证可加载 |
| 特征缓存 | `runs/j1_baseline/feature_cache/features.npz` | 已验证 |
| 环境 | conda `wm_eval`：Python 3.9.23、torch 2.7.1+cu128（RTX 5060 Ti）、ultralytics 8.4.202、`ckpt/yolov10x.pt` | **已就绪**（F-02/F-03 均已解决） |

## 2. E3-D0：时间变化与过度平滑诊断（先做）

**问题**：当前预测器（D4/B4）是否丢失事件状态的时间变化？
（D4 的 Participation Ratio 下降**不能**直接证明时间过度平滑——见
`docs/jepa_paper_audit.md` §1.2。）

**实现**：`src/e3_d0_temporal_diagnostics.py`

- 数据：`samples.csv` 中 `label == "same_identity"` 的对——
  `anchor_frames`（同一行人的 12 帧历史，含框）与 `cand_frames`
  （同一行人的 12 帧真实未来，含框）；
- 真实未来特征 $z$：DINOv2（冻结）对 cand_frames 逐帧提取（缓存复用）；
- 预测未来特征 $\hat z$：D4/B4 预测器在逐帧 delta（秒）下从历史预测；
- 参照基线：**persistence**（重复历史最后一帧）——确定性坍缩的下界
  参照（其 $D(\hat z)=0$、$C(\hat z)=1$）；
- 指标（同协议内对照，**不与 WA-JEPA 数值比较**）：
  - $C(\cdot)$：序列内不同时间步平均余弦相似度；
  - $D(\cdot)$：相邻时间步平均 L2 变化幅度；
  - $\Delta_{\mathrm{dir}} = C(\hat z) - C(z)$；$R_\Delta = D(\hat z)/\max(D(z),\epsilon)$；
  - 每对样本计算，按模型（D4/B4/persistence）与 split 汇总
    （均值/中位数/四分位 + 自助法 CI，固定种子）。

**预注册判读规则（运行前固定，不得事后调整）**：

| 结论 | 条件 |
|---|---|
| 存在时间过度平滑证据 | 中位数 $\Delta_{\mathrm{dir}} > 0$ **且** 中位数 $R_\Delta < 0.5$，两者的自助 95% CI 均不含临界值 |
| 边界证据 | 仅其一成立 |
| 无证据 | 两者均不成立 → **不得**据此引入多未来预测或时序损失改动 |

需要 ≥2 个未来时间步（12 帧满足）。协议差异声明：384 维池化特征 +
确定性预测器 vs WA-JEPA 的空间 Token + Flow Matching 单步预测——
结论只在我们自己的协议内有效。

**产出**：`runs/e3_d0_diagnostics/`（per-pair 明细 + 汇总 JSON + 代码
SHA/输入 SHA 溯源）。

## 3. E3-A：无 JEPA 的事件状态规则基线（现在即可）

**任务不是预测未来，而是确定生成视频实际呈现了什么。**

**实现**：`src/e3_a_event_state_baseline.py`

- 输入：`tracks.json` 的 person 轨迹（norm_centers 逐帧观测）；
- 每帧观测：位置、速度 $v$、$|v|$、$\Delta v$、可见性（帧缺口）；
- 状态机（含滞回与防抖，避免检测抖动造成状态翻动）：

| 状态 | 进入条件 |
|---|---|
| UNKNOWN | 预热不足（<3 帧）/ 帧缺口 > 容忍 / 刚出缺口待重稳 |
| WAITING | 从未运动过的静止（平滑速度 < v_stop 持续 ≥ n_stop 帧） |
| CROSSING | 平滑速度 > v_move 持续 ≥ n_move 帧 |
| STOPPED | 先前为 CROSSING 后进入静止 |
| RESUMING | STOPPED/WAITING 后运动恢复（过渡窗内） |

- 滞回带：$v_{stop} < v_{move}$（归一化坐标/帧）；缺口容忍与重稳窗
  显式配置并写入输出；
- **图像平面声明**：图像平面速度 ≠ 地面速度。无自车运动/相机几何
  补偿前，停止/运动判断限于**图像观测层面的事件证据**（输出字段
  `observation_level: "image_plane"`）；
- 输出：`runs/e3_a_baseline/`（逐帧相位 + 逐段摘要 + 每视频汇总，
  real/gen 成对）。

**验收（E3-B 门控之一）**：`e1_manual_labels.csv` 20 行人工真值完成后，
按预注册指标（与人工 event_satisfaction/required_behavior 的一致性、
混淆表）评估；在此之前 E3-A 只产出证据，**不产出研究结论**。

## 4. E3-B：Event-State JEPA（门控启动，本轮不实现）

$$h_t = E_{\mathrm{event}}(O_{\le t}, Z^{ROI}_{\le t}), \qquad
\hat h_{t+\Delta} = P_\theta(h_{\le t}, \Delta t)$$

- 第一轮**只比较**：同一个事件验证器，不加 JEPA vs 加 JEPA；
- 禁止同时改变：视频编码器、数据划分、事件标注、预测损失、拒识
  策略、判定阈值（单变量原则）；
- 若 E3-B ≤ E3-A：再判断是状态表示不足、时间预测退化，还是事件
  标签不可靠；
- 启动条件：① E3-D0 出结果；② E3-A 验收完成；③ 人工事件标签达到
  验收条件。

## 5. 本轮明确边界

- 不训练任何网络（E3-B 属后续轮次）；
- 不下载大模型权重；不改 D4 Checkpoint；
- 不用测试集调参；阈值改动必须登记 `docs/failure_log.md`；
- E3-D0/E3-A 代码：先自测（`src/selftest_e3_stage.py`）后运行，
  结果落盘、带溯源哈希——沿用 D4.1.1 纪律；
- 路线 A（D4.1.1 正式评价）与路线 B 并行：A 被权限分类器间歇故障
  阻塞时，B 不等待。
