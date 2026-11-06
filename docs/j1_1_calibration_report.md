# J1.1 校准与指标修复报告

日期：2026-11-05 | 修复对象：`latent_association.py`（校准协议 + 指标定义）
原始结果对照：`runs/j1_baseline/`（未动）；新结果：`runs/j1_1_audited/`

---

## 1. 确认的缺陷（任务书指认 + 本地复核）

### 1.1 阈值校准的静默回退

- **现象复核**：v1 校准器的"分布重叠→midpoint 回退"分支在 J1 train
  负样本上产出 0.6913，**实测 FPR = 28.33%**（声称 ≤5%——差 5.7 倍）；
- **根因**：当负样本 P95（0.813）> 正样本中位（0.800）时回退触发，
  midpoint 阈值**不携带任何 FPR 约束**；
- **附带发现（修复过程中）**：v2 严格化初版引入 1-ULP 爬升 while 循环
  ——`ceil(0.05×406)=21` 允许 21 个但 21/406=5.17%>5% 的协议矛盾使
  循环需 10^12 步跨越 4 位小数并列间隔（**死循环**，已定位并替换为
  唯一值扫描）；round(4) 会把 nextafter 阈值圆回并列值破坏约束
  （已修：round 仅在不重新引入并列负样本时使用）。

### 1.2 指标定义

- v1 的 `error_merge_rate = FP/全部对数` **不是 FPR**（分母错）。

## 2. 修复后的协议（代码化于 `calibrate_threshold`/`pair_metrics`）

**阈值协议**（floor-quantile 唯一值扫描，无回退）：
1. 允许通过负样本数 k = floor(target_fpr × n_neg)（floor 保证 k/n ≤ target）；
2. 阈值 = 满足 count(neg ≥ t) ≤ k 的最小唯一负样本值（TPR 友好的
   合法选择；并列显式处理）；
3. 无解时返回 (None, reason)——**永不静默放宽**；
4. 返回报告携带**实测** FPR/TPR（下游任何"FPR≤x"声明可查证）。

**指标定义（显式分母）**：
| 指标 | 分母 |
|---|---|
| False Merge Count（FP） | —（计数） |
| False Merge Fraction of All Pairs | FP / n_all_pairs |
| **FPR** | FP / n_negatives |
| Missed Relink Rate | FN / n_positives |
| Pair Accuracy | (TP+TN) / n_all |
| Balanced Accuracy | (TPR+TNR)/2 |
| Precision | TP / (TP+FP) |

## 3. 修复后的实际数字（J1 特征，train 校准 / test 评估）

**严格校准**：阈值 **0.8122**（train 实测 FPR 4.93%、TPR 42.39%、
允许负样本 20/406）；val 抽查 FPR/TPR 见 metrics.json。

**不可兼得声明（如实）**：FPR≤5% 与高召回**不可同时满足**——
严格阈值把 TPR 从 0.83（旧 0.6913）压到 0.42。未修改任何指标。

**TEST（n=800，显式分母指标）**：

| 规则 | acc | **FPR** | mergeFrac | miss | prec | balAcc |
|---|---|---|---|---|---|---|
| J0 运动学 | 0.815 | 0.340 | 0.174 | 0.023 | 0.733 | 0.819 |
| J1 cos@旧0.6913（缺陷对照） | 0.790 | 0.262 | 0.134 | 0.156 | 0.755 | 0.791 |
| **J1 cos@严格0.8122** | 0.709 | **0.066** | 0.034 | **0.527** | 0.873 | 0.704 |
| J1 fused@严格 | 0.709 | **0.054** | **0.028** | 0.540 | **0.891** | 0.703 |
| J1 fused@旧（对照） | 0.825 | 0.171 | 0.088 | 0.179 | 0.821 | 0.825 |

**判读**：修复揭示的真实图景——外观特征在**固定低 FPR 下把精度
推到 0.891**（错误合并绝对数 FP 从 139 → 22），代价是过半真配对
落入需人工/更高层证据的区域。这一"precision@coverage"结构正是
J3-lite 三态设计要处理的（Uncertain 带显式接管中间带，见
j3_lite_results.md）。

## 4. 回归测试（selftest_j，19/19）

新增 8 项校准/指标回归：可分离/完全重叠/无负样本/重并列
（FPR 恰好约束）/ target=0 全并列 / 报告含实测 TPR / 指标手算用例
（tp=fp=fn=tn=1）/ 无负样本时 FPR=None 而非 0。
