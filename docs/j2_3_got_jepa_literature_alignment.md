# J2.3 GOT-JEPA 文献对齐与方法边界

日期：2026-11-05 | 核查：GOT-JEPA（TCSVT 2026，arXiv:2602.14771，
项目 github.com/chenshihfang/GOT）+ 前两轮官方页面核查

## 1. 组件边界核查（哪些思想可迁移、哪些不可直接搬）

| GOT-JEPA 组件 | 原文事实 | World-GS 迁移判定 |
|---|---|---|
| **Tracking Model Prediction（ω）** | 预测**目标条件化的跟踪模型**（作用于当前特征图生成定位响应）——非预测未来图像特征 | **最值得吸收的思想**：Actor-Conditioned Discriminative Query（ω_a）；但我们的 384 维池化特征上是**判别查询向量**，非 2D 特征图滤波器——明确声明不是 GOT 复现 |
| Expander + Covariance Loss | 存在且有消融 | **只读诊断先行**（G1 已完成）；协方差去相关**数学上不能单独防塌缩**（常量特征 cov=0 反例已演示） |
| Invariance:Covariance=25:1 | 论文实验值 | 不可直接当我们的最优权重（独立超参） |
| ProjNet / 10× 学习率 | 消融支持 | 当前已有输出投影——不重复添加 |
| Copy-Paste 特征扰动 | **Patch-token 空间结构上** | 池化向量替换维度**不等价**——准入=先取 DINOv2 patch tokens + 空间映射（未启动） |
| OccuSolver / FIFO 门控 / 85% 阈值 | 点级可见性 | 轻量可见性门控先行（G3 V0-V3 已完成）；85% 不作 JAAD 普适阈值 |
| Teacher-Student | Teacher 干净观测 / Student 扰动观测 | 后备（非对称预训练）——本轮不引入 |

**根本区别（不可混淆声明）**：GOT-JEPA 的监督对象是"如何在当前
帧定位目标"（判别模型）；我们 J2 历来的对象是"未来外观特征"
（回归）。**任务目标不一致正是 H2 假设的来源**。

## 2. Teacher/Student 与我们的 GRU 的任务差异

- GOT-JEPA：teacher 从干净帧预测跟踪模型 ω；student 从**受扰帧**
  预测同一 ω——学习对遮挡/干扰的鲁棒性；
- World-GS J2：因果历史 → 未来特征回归，无 teacher-student 结构；
  观测到的 bbox 速度**不是**外部控制动作（V-JEPA 2-AC 的 action
  conditioning 不适用——三轮一致边界）。

## 3. 结论

优先吸收**判别目标设计**（ω_a + 排序损失——与既定 C 计划同构，
GOT-JEPA 提供了文献依据）；协方差正则与扰动留作独立消融候选，
准入条件由 G1 诊断决定（见 go/no-go）。
