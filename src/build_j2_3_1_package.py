"""Build the J2.3.1 alignment package (independent-audit oriented).

Contents:
  README.md     P0 fix table + old-vs-new numbers + audit entry points
  docs/         7 J2.3.1 documents + failure log (F-21)
  code/         fixed builder + ranking loss + baselines + smoke + tests
  results/      runs/j2_3_1_alignment (candidate manifest, stats,
                recalibration, smoke)
"""
from __future__ import annotations

import shutil
import tarfile
import tempfile
from pathlib import Path

ROOT = Path("/home/moz/World-GS")
PKG = "j2_3_1_alignment_package_20261105"
OUT = ROOT / "dist" / f"{PKG}.tar.gz"

README = """# J2.3.1 因果困难负样本对齐包（独立审计用）

日期：2026-11-05 | 判定：**CONDITIONAL-GO（训练就绪，待启动指令）**
测试：ranking 7/7 + 边界 27/27 + 残差 7/7；零训练、零旧结果覆盖

## 三个 P0（独立审计发现，全部先数值确认再修复）

| P0 | 确认事实 | 修复 |
|---|---|---|
| 身份归属 | 100% 负对（7,603）的 90,122 帧观测错误归属 anchor 身份 | anchor/cand 双侧各归其 (src,ped) |
| 重复虚增 | "119,081 负样本"含 276,690 次重复追加 | (src,actor,frame) 规范索引 → **独立候选 2,043** |
| 未来位置泄漏 | 门控中心用 anchor 未来真实观测 | 因果历史+运动学外推（每候选独立时刻）；真值仅存标签列 |
| 排序符号 | 设计公式反写（未进训练） | 正确实现+7/7 梯度测试（含反写守卫） |

## 旧值作废 → 新基线（审计关键对照）

| 口径 | 旧（污染） | 新（修复） |
|---|---|---|
| B0 静态 M_hard | ~~0.112~~ | **0.033**（虚高 3.4 倍） |
| kin top1 | 0.805 | **0.667** |
| B4 M_hard | 0.053 | 0.015（差距仍在） |
| Tier-1 "数量" | 119,081（重复） | **2,043 独立候选**（390/412 anchor） |
| J2.3 391/412 统计 | 作废 | 归属污染+泄漏双重错误 |

**新 D4 预注册判定标准：val M_hard > 0.033**。
D4 冒烟五门全过（梯度/溯源/因果/过拟合 1.08→0.00/无负边界）。

## 独立审计入口

1. P0 复现：`code/build_tiered_negatives_v2.py` 修复 vs
   `src/build_tiered_negatives.py`（旧版仍只读在库中）——归属/
   去重/门控三处 diff；
2. 损失梯度：`code/selftest_ranking_loss.py`（7 项含符号守卫）；
3. 基线复核：`results/baseline_recalibration.json`（阈值
   train-only 校准 + 每候选独立 delta）；
4. 候选清单：`results/tier1_candidates.csv`（3,583 行，逐行
   candidate_feature_key 可溯至 (src,actor,frame) 去重观测）；
5. 影响隔离：G1 表示诊断/G3 可见性对照不受影响（正对身份键），
   J2.3 Tier 统计与 M_hard=0.112 作废（见 docs 各勘误）。

## 目录

docs/(7 文档+failure_log) code/(5 文件) results/(4 项)
复现命令：docs/commands.md 各轮记录 + 各脚本头部 docstring。
"""


def main() -> None:
    stage = Path(tempfile.mkdtemp(prefix="wmeval_j231_pkg_"))
    root = stage / PKG
    root.mkdir()

    (root / "docs").mkdir()
    for f in ("j2_3_1_identity_assignment_audit.md",
              "j2_3_1_observation_dedup.md",
              "j2_3_1_causal_candidate_generation.md",
              "j2_3_1_tier1_manifest.md",
              "j2_3_1_ranking_loss_tests.md",
              "j2_3_1_baseline_recalibration.md",
              "j2_3_1_go_no_go.md", "failure_log.md"):
        shutil.copy2(ROOT / "docs" / f, root / "docs" / f)

    (root / "code").mkdir()
    for f in ("build_tiered_negatives_v2.py", "run_j2_3_1_baselines.py",
              "run_j2_3_1_d4_smoke.py", "selftest_ranking_loss.py"):
        shutil.copy2(ROOT / "src" / f, root / "code" / f)
    shutil.copy2(ROOT / "src/wmeval/latent_world_state/ranking_loss.py",
                 root / "code")

    shutil.copytree(ROOT / "runs/j2_3_1_alignment", root / "results",
                    ignore=shutil.ignore_patterns("__pycache__"))

    (root / "README.md").write_text(README, encoding="utf-8")

    OUT.parent.mkdir(exist_ok=True)
    with tarfile.open(OUT, "w:gz") as tar:
        tar.add(root, arcname=PKG)
    n = len(tarfile.open(OUT).getmembers())
    print(f"[pkg] {OUT} ({OUT.stat().st_size/1e6:.2f} MB, {n} members)")


if __name__ == "__main__":
    main()
