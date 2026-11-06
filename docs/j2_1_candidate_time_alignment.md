# J2.1 候选时间对齐修复报告（Gate 1）

日期：2026-11-05 | 重评：`runs/j2_1_reeval/competition_reeval.json`
实现：`latent_world_state/association_evidence.py::candidate_delta`
+ `run_j2_1_reevaluation.py`（竞争集段）；回归 12/12

## 1. 确认的缺陷（泄漏路径）

旧 `run_j2_association.py` 对竞争集的**全部候选**用**同一个
delta**——即**正确关联目标的时间间隔**。这构成两重违规：
1. 推理时不应知道哪个候选是正确身份（任务 §三.2）；
2. 其他候选的时间被真主时间替代（§三.3）。

## 2. 修复协议

- 每候选独立：`delta_c = candidate_first_time_s − anchor_last_time_s`
  （从原始 competitions.jsonl 的 first_frame/clip fps 计算）；
- 预测器**逐候选拟合**：z_c = Predictor(history, delta_c)；
  pred_cos_c = cos(z_c, feature_c)；
- 无效间隔（≤0/缺失/非有限）→ 证据 `unknown`（计入 uncertain，
  **拒绝猜测**——本测试床实测 0 例）；
- 回归测试：多候选不同出现时刻各自 delta、无效间隔 None、
  真主时间不被替代（12/12）。

## 3. 重评结果与诚实判读

| 规则 | 旧（真主 delta 泄漏） | **新（每候选独立 delta）** |
|---|---|---|
| A1（kin 基线，不涉时间） | top1 0.805 / cov 0.720 / wrong 0.119 | 完全一致（0.805/0.720/0.119） |
| A3（+pred 证据） | cov 0.146 / wrong 0.167 | **完全一致**（0.146/0.167） |

**数值不变的原因（测试床局限，如实记录）**：本竞争集的候选来自
anchor gap 后 ±3 帧窗口——**所有候选的首观测时间天然聚集**
（delta 差 <0.1s，log1p 嵌入下输入差异微小）→ 逐候选预测与
共享预测输出几乎相同。**泄漏在语义上已消除**（协议正确性义务），
但该测试床无法区分两种协议——"候选时间多样性"列为竞争集
构造的下一改进项（更宽窗口/异时刻候选）。

## 4. 结论

Gate 1 **通过（协议修复完成、零泄漏、回归覆盖）**；数值结论
不变——A3 无增量、A1 为基线的 J2 轮结论维持。
