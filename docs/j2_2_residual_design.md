# J2.2 残差预测器设计与验证（§四）

日期：2026-11-05 | 代码：
`src/wmeval/latent_world_state/residual_predictor.py`
（测试 `selftest_j2_2.py` **7/7**）

## 1. 设计

`z_pred = normalize(z_last + GRU_residual(history, delta))`

- 输入：冻结 DINOv2 历史（T≤12，因果）、实际 delta_seconds
  （log1p 逐步入 GRU）；**无** hard-negative 损失/运动状态/多步
  rollout/位置编码（任务 §四纪律——单变量：残差参数化）；
- 损失：Cosine Prediction Loss（唯一）。

**残差假设的机理**：v1 直接预测收敛到"均值脸"（共性化解）；
残差参数化把外观近不变性交给静态末帧，GRU 只学漂移修正——
**退化解在残差空间 = 零输出 = 精确回退静态基线（安全退化）**，
而 v1 的退化解是 batch 均值特征（有害）。

## 2. 六项验证（全部通过，selftest_j2_2）

| # | 验证 | 结果 |
|---|---|---|
| 1 | 零残差 == 静态末帧基线 | **精确成立**（maxdiff < 1e-6，kill_residual 钩子） |
| 2 | 输出有限且归一化 | ✓（恒 L2-normalize） |
| 3 | 因果输入 | forward(history, delta) only（API 形状测试） |
| 4 | 无未来候选特征进入输入 | ✓（目标扰动不变性测试） |
| 5 | CPU + GPU 单步梯度 | ✓（全参数有限梯度） |
| 6 | 小样本过拟合 | loss 1.024 → −0.0000（32 样本 400 步） |

**零残差基线 = 静态特征基线**（非原始 GRU 基线）——B0 评估即
静态末帧（区分间距 0.249），任务 §四.3 的要求满足。

## 3. 训练配置

与 v1 全对齐（seed 51 / batch 32 / AdamW 3e-4 wd 0.01 / ≤40
epochs / patience 6 / clip 1.0）；**模型选择按 val 区分间距**
（非 loss——任务 §五）；监控十项逐 epoch（见训练报告）。
