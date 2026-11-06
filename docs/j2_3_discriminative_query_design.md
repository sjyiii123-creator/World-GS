# J2.3 Actor-Conditioned Discriminative Query 设计（不训练）

日期：2026-11-05 | 状态：**设计 + 接口冒烟通过**（QUERY_INTERFACE_OK，
未训练）

## 1. 设计（GOT-JEPA 判别思想的最小迁移）

```
omega_a  = normalize(z_ref + R_phi(H_a, delta))      # 查询向量
s(a, j)  = cos(omega_a, z_j)                          # 候选判别分数
L        = L_pred + lambda_rank * L_rank              # 第一轮仅两项
L_rank   = mean( max(0, m + s(a,a+) - max_{j∈Tier-1} s(a,j)) )
```

- **保留 B4 的残差结构**（z_ref=可见性参考，R=残差 GRU）——只改
  变它服务的**任务目标**（回归→判别）；
- **Tier-1 硬负样本**（空间门控后的真实共现者，G2 已建）作
  L_rank 的负集——随机负样本与判别任务不符（诊断 §2 的 H2 教训）；
- 接口冒烟已验证：omega (B,384)、s (B,K)、rank loss 有限、
  **候选特征永不进入预测器输入路径**（因果保持）。

## 2. 与 GOT-JEPA 的明确边界（不冒充）

- GOT 的 ω 作用于 **2D 特征图**（定位滤波器）；我们的 ω_a 是
  **384 维池化特征上的判别查询向量**——受启发（task-objective
  层面），非复现；
- Teacher-Student / Copy-Paste / Expander 均**不在第一轮**（单
  变量纪律：先只加 L_rank 对照 B4）。

## 3. 训练准入条件（预注册）

| 条件 | 状态 |
|---|---|
| H1/H2 诊断支持判别目标 | ✅（G1：同向漂移 + Tier-1 素材就绪） |
| Tier-1 负样本集 | ✅（391/412 序列，gap 均衡） |
| 协方差正则独立消融位 | ✅ 预留（D5，判据：先 D4 有效） |
| 评估协议 | 沿用 J2.1 修正（身份 mask/每候选时间/固定覆盖） |
| 判定标准 | **M_hard > B0 静态（0.112）**才判方法有效（非 > B4） |

## 4. 未来可选扩展（各自独立消融，不并入第一轮）

- D5：+ Covariance 正则（G1 动机成立但单独不足——须与判别并用）；
- D6：+ 非对称扰动（准入=先验证 DINOv2 patch tokens 提取与空间
  映射，池化向量维度替换不等价于 Copy-Paste）。
