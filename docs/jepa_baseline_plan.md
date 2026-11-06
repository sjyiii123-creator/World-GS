# J 系列基线计划：J0–J3 对照实验设计

日期：2026-11-05 | 状态：J0 已运行（真实结果）、J1 已实现待权重运行、
J2/J3 设计完成（训练/运行由 J1 结果决定，不伪造结果）。

---

## 1. 统一评估协议

- **数据**：`datasets/processed/actor_sequences/`（15,523 对 + 7,849
  竞争集；JAAD 官方身份真值；合成 gap 协议见数据集文档 §2）；
- **阈值纪律**：任何可调阈值只在 train split 校准（J1 的 cos 阈值
  以 FPR≤5% 校准于 train），test 只做最终评估——违反即作废重跑；
- **指标**（任务 §六）：
  - 关联准确性（pair accuracy）；
  - **错误轨迹合并率**（different-identity 被接受——J0 的核心缺陷）；
  - 漏关联率（same-identity 被拒；遮挡后恢复能力的下界）；
  - 竞争集 top-1 身份准确率 + 歧义率；
  - 推理时间与 GPU 显存（benchmark 实测）；
  - 事件验证准确性 / 自动接受覆盖率 / 错误放行率：**需人工真值**
    （e1_manual_labels 完成后在 B0–B3 协议下评估，J 系列不单独宣称）。

## 2. 四个基线

| 基线 | 实现 | 状态 |
|---|---|---|
| **J0 运动关联** | `run_j0_baseline.py`（actor_binding 同款外推规则） | ✅ 已运行 |
| **J1 + 冻结视觉特征** | `run_j1_baseline.py`（dinov2-small CLS + 融合 AND 规则 + cos 阈值 train 校准） | ✅ 已实现，权重就绪后运行 |
| **J2 + 时序预测特征** | `latent_predictor.py`（GRU 头接口可运行，**未训练**） | 设计完成 |
| **J3 + 关联不确定性** | J2 预测误差与关联置信度进入三态（accept/reject/**uncertain**） | 设计完成 |

### J2 训练设计（若 J1 触发）

- **输入**：anchor 段逐帧 L2 归一化特征序列（T=12，frames≤t ONLY）；
- **目标**：t+Δ 的真实冻结特征（L2 归一化）——cosine loss；
  Δ 分桶嵌入（log2 间隔，8 桶——`delta_bucket`）；
- **数据**：A 正对样本的 (history, z_future) 对，train split；
- **防泄漏**：forward 不接受未来参数（API 形状测试固化）；
  时序上只用 ≤t 的帧（assemble_history 契约）；
- **评估**（J2 vs J1）：预测特征 ẑ 与真实候选特征 cos 是否优于
  外推位置匹配（大 gap=45 帧时外推退化——J0 实测漏关联 5.3%）；
- **触发判据（预注册）**：J1 的 same/diff 分离 AUC ≥0.8 且
  错误合并率 ≤5% 时启动训练；否则 J2 无增量空间，不训练；
- **红线**：预测误差只是**证据**（J3 的拒识信号），绝不单独判
  Invalid；预测状态不得当作实际观测。

### J3 设计（证据→不确定性）

- 关联三态：cos ≥ 高阈值 → accept；cos ≤ 低阈值 → reject；
  中间带 → **uncertain**（进 adjudicate 的弱证据路径，与 v3.1
  归因哲学一致）；
- 低置信带宽来自 J2 预测残差（预测的未来特征 vs 实际候选特征的
  偏离大 → 该关联更不确定）；
- **评价**：Selective-Risk 框架（theory_to_code_matrix §4）——
  自动接受覆盖率 vs 错误放行率的 tradeoff 曲线，与 B0/B3 同协议。

## 3. 与事件验证器的接入设计（下一步）

```
J1/J3 关联证据(每链: cos, ambiguity, 拒识带)
   → actor_binding.build_track_chains 增加外观分数列
     (链 links 记录 kin_err + cos，共存保护不变)
   → crossing_evidence 按链评估(不变)
   → v4 adjudicate: 链置信度进入 Uncertain 判据
```

- **不改 v3.1 语义**（本轮全部独立评估目录 runs/j*）；
- 接入点是 v4-experimental 的链构造参数化（合并规则从纯 kin 扩展为
  kin∧cos），v3.1 结果保持冻结对照。

## 4. 运行成本登记

| 项 | 实测/预估 |
|---|---|
| J0 全量 15,523 对 | <1s CPU |
| J1 特征提取（抽样 1,800 对 × 6 帧） | 权重就绪后实测（benchmark 先行） |
| 特征缓存 | runs/j1_baseline/feature_cache/（npz + meta：模型版本+预处理配置） |
| J2 训练（若触发） | GRU 0.9M 参数，CPU 可训；数据 A 正对 ~7,900 × 12 步序列，<10 分钟量级 |
