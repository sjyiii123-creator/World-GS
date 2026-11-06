# J2 训练报告（阶段 4）

日期：2026-11-05 | 运行：`src/run_j2_train.py` →
`runs/j2_experimental/{j2_train_log.json, j2_predictor.pt,
j2_predictor_D.pt}` | 配置：config yaml 全对齐（seed 51/batch 32/
AdamW 3e-4 wd 0.01/max 40 epochs/patience 6/clip 1.0/cos loss）
config_hash 59723fb1608a

## 1. 六步检查梯（全部通过，顺序执行）

| 步 | 结果 |
|---|---|
| 1 张量检查 | shape (412,12,384)/target (412,384)/delta>0/四 split ✓ |
| 2 CPU 单步 | loss 有限 ✓ |
| 3 GPU 单步 | loss 有限 ✓ |
| 4 反向传播 | 全参数梯度有限 ✓ |
| 5 小批过拟合 | 32 样本 400 步：loss **1.046→0.0001** ✓（模型容量足够） |
| 6 正式训练 | 见下 |

## 2. 正式训练（J2-C GRU 384→128→384；J2-D +2D 速度）

| 变体 | epochs | best val loss | val pred-cos | same/diff actor cos | 耗时 | 峰值显存 |
|---|---|---|---|---|---|---|
| J2-C | 40（无早停触发） | **0.2703** | 0.729 | **0.729 / 0.613** | 1s | 0.05GB |
| J2-D | 40 | 0.2690 | 0.731 | 0.731 / 0.618 | 1s | 0.05GB |

训练动态（J2-C）：epoch0 train 0.930/val 0.798 → epoch39 train
0.211/val 0.271——损失健康下降，无发散；区分能力（same vs diff
actor 的 cross-cos）从 0.003 起步升至 0.116 间距。

## 3. 训练结束 ≠ 实验成功（任务 §四纪律）

损失下降与预测 cos 0.73 的表象之下，**判定性指标在消融中检验**
（j2_ablation_report）：预测特征的 actor 区分间距 0.110，**低于
静态末帧的 0.257**——模型学到的解偏"共性化"（预测特征对所有
actor 目标都中等相似）。训练本身合规（六步梯+全日志），但按
预注册判据（区分力与下游改善），J2 v1 未通过价值检验。
