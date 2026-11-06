# J0/J1 选择性关联强基线（阶段 3）

日期：2026-11-05 | 运行：`src/run_a_baselines.py` →
`runs/j2_experimental/{a_baselines.json, a_baselines_sweep.json}`
数据：J3-lite 竞争集明细（388 test + train，纯 CPU 复用已算特征）；
阈值 train-only（FPR≤5% 于错误候选，严格协议）

## 1. 三基线（TEST 388 竞争集；top-1 排序全部为运动学——J3-lite 教训）

| 规则 | top1 | coverage | wrong-acc | precision | recall | no-match FA | uncertain |
|---|---|---|---|---|---|---|---|
| A0 kin 全接受 | 0.807 | 1.000 | 0.193 | 0.807 | 0.807 | **1.000** | 0 |
| **A1 kin+kin 拒识**（thr 0.0265/hi 0.0648） | 0.807 | **0.719** | **0.147** | 0.853 | 0.613 | 0.312 | 0.165 |
| A2 kin+cos 拒识（thr 0.7584/lo 0.5853） | 0.807 | 0.534 | 0.159 | 0.841 | 0.449 | 0.309 | 0.343 |

## 2. 关键判读

1. **A1 是当前最优强基线**：coverage 0.72 下错误接受率降 24%
   （0.193→0.147），CI95 top1 [0.743, 0.854]、wrong-acc
   [0.105, 0.204]；
2. **A2（cos 可信度拒识）更保守但不更准**——cos 阈值拦掉的多是
   真主（recall 0.449 vs A1 的 0.613）：外观置信在竞争场景下
   对难例不敏感（同 J3-lite 结论）；
3. **no-match 组（真主缺席孪生）**：A1/A2 把假接受从 1.0 压到
   ~0.31——仍有 31% 的缺席场景被错误接受（kin 排序 top1 恒有
   获胜者）——**结构性弱点**：拒识应结合"绝对分数"而非只看
   排序内部相对值；
4. 覆盖率匹配比较（sweep）：A1 的 kin 阈值扫掠 vs A2 的 cos 阈值
   扫掠——同 coverage 下 A1 曲线一致优于 A2（a_baselines_sweep）。

## 3. uncertain 统计修复（任务 §三指认）

- 旧 `uncertain_band_true_share=1.0` 为**恒真式**（每个竞争集必含
  真主）——j3_metrics_audit.py 的诊断确认；
- 替换统计：**带内 top-1 正确率 vs 总体**——A1 带 0.719 < 总体
  0.807（带聚集难例 ✓ 有信息量）；A2 带 0.805 ≈ 总体（cos 带
  不区分难度——无信息）。

## 4. A 基线对 J2 的意义

A1 是 J2 必须击败的强基线（kin 排序 + 运动学拒识，无需任何
视觉特征）；J2 的增量空间 = A1 的两个弱点：no-match 31% 假接受
与长 gap 外推退化（missed relink 5.3% @45 帧，J0 实测）。
