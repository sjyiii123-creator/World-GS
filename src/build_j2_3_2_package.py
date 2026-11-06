"""Build the J2.3.2 alignment package (candidate evidence integrity).

Contents:
  README.md     P0 fix table + old-vs-new baselines + GO decision
  docs/         7 J2.3.2 documents + failure log (F-22)
  code/         fixed builder + aligned baselines + calibrator +
                masked ranking loss + tests
  results/      runs/j2_3_2_alignment (canonical index, fixed-query
                manifest, aligned baselines, audit reproduction)
"""
from __future__ import annotations

import shutil
import tarfile
import tempfile
from pathlib import Path

ROOT = Path("/home/moz/World-GS")
PKG = "j2_3_2_alignment_package_20261105"
OUT = ROOT / "dist" / f"{PKG}.tar.gz"

README = """# J2.3.2 候选证据完整性与评价对齐包

日期：2026-11-05 | 判定：**GO（D4 训练准入达成，本轮不启动）**
验证：Workflow 双阶段（4 agent 修复前独立确认 + 4 skeptic 修复后
反驳尝试 + completeness critic）；回归 6/6+7/7+27/27；零旧结果覆盖

## 四个 P0（全部独立确认后修复）

| P0 | 独立确认 | 修复 |
|---|---|---|
| A 框检索 | 3,011 对/555 缺失/17 错框(3 真主) | 规范索引(94,748 观测/0 冲突)→**3,597/3,597 全覆盖** |
| B 阈值方向 | kin thr 0.6517 使 94.38% 负候选接受 | calibrate_lower_is_better(thr **0.0036**/FPR 0.17%) |
| C Tier 混入 | 959 Tier-2 混入 M_hard(val 46/60) | 纯 tier==1(Tier-2 独立分列 0.19 vs Tier-1 0.059) |
| D oracle 时间 | 412/412 强制注入+答案时刻窗口 | **固定查询 1.0s**(预注册常量,oracle/注入零使用) |
| 变长负样本 | 矩形 batch 限制 | ranking_loss_masked 6/6(padding 不赢 hardest) |

## 新基线（B4 冻结，未重训——同候选集合）

| | val Tier-1 M_hard | expl | top1 (val/expl) |
|---|---|---|---|
| **B0 静态** | **0.0594** | 0.0497 | 0.783 / 0.870 |
| B4 预测 | 0.0172 | 0.0147 | 同 |

旧 0.0333 二次作废。**D4 预注册判定标准：val Tier-1 M_hard > 0.0594**。

## 诚实边界

- no-match=0（插值标注下 1.0s 窗内真主自然存在——数据事实）；
- final_test 12 anchor/2 源/0 Tier-1——无法检验困难竞争；
- kin TPR 22%（FPR≤5% 硬约束——不可兼得，如实声明）。

## 目录

docs/(7 文档+failure_log) code/(7 文件) results/(5 项)
复现：各脚本头部 docstring + docs/commands.md 各轮记录。
"""


def main() -> None:
    stage = Path(tempfile.mkdtemp(prefix="wmeval_j232_pkg_"))
    root = stage / PKG
    root.mkdir()

    (root / "docs").mkdir()
    for f in ("j2_3_2_candidate_feature_integrity.md",
              "j2_3_2_temporal_query_protocol.md",
              "j2_3_2_distance_calibration.md",
              "j2_3_2_tier1_margin_recalibration.md",
              "j2_3_2_baseline_comparison.md",
              "j2_3_2_ranking_batch_contract.md",
              "j2_3_2_go_no_go.md", "failure_log.md"):
        shutil.copy2(ROOT / "docs" / f, root / "docs" / f)

    (root / "code").mkdir()
    for f in ("build_fixed_query_v3.py", "run_j2_3_2_baselines.py",
              "selftest_ranking_mask.py", "selftest_ranking_loss.py"):
        shutil.copy2(ROOT / "src" / f, root / "code" / f)
    shutil.copy2(ROOT / "src/wmeval/latent_world_state/ranking_loss.py",
                 root / "code")
    shutil.copy2(ROOT / "src/wmeval/latent_association.py",
                 root / "code")

    shutil.copytree(ROOT / "runs/j2_3_2_alignment", root / "results",
                    ignore=shutil.ignore_patterns("__pycache__"))

    (root / "README.md").write_text(README, encoding="utf-8")

    OUT.parent.mkdir(exist_ok=True)
    with tarfile.open(OUT, "w:gz") as tar:
        tar.add(root, arcname=PKG)
    n = len(tarfile.open(OUT).getmembers())
    print(f"[pkg] {OUT} ({OUT.stat().st_size/1e6:.2f} MB, {n} members)")


if __name__ == "__main__":
    main()
