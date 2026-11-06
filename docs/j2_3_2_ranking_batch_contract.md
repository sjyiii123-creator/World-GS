# J2.3.2 排序损失 Batch 契约（变长负样本）

日期：2026-11-05 | 实现：`ranking_loss_masked` + **6/6 测试**

## 1. 契约

- `scores_neg (B,K)` padding + `valid_negative_mask (B,K)`；
- **Padding 永不参与 hardest max**（-inf 掩码）；
- **无负行从均值分母排除**（非零填充进梯度）；
- 同身份候选永不标 True（采样器契约——manifest 按
  (src,actor) 键分组保证）；
- 梯度方向与未掩码版一致（正确符号：正分升/最难负降）。

## 2. 测试（6/6）

全零 mask → 零/无梯度；单负 → 正损失+正分负梯度；多负 →
hardest 独得梯度；**混合 batch（行 A 2 负/行 B 0 负）→ 无负行
排除出均值**；**padding 毒值（99.0）不赢 hardest**；全有效 ==
未掩码语义。

## 3. D4 冒烟（P0 门）

五门全过（梯度方向/溯源键 per-anchor 唯一/时间因果 API/
单批过拟合 1.08→0.00/无负边界）——λ∈{0.1,0.3,0.5,1.0}
仅 val 选择。
