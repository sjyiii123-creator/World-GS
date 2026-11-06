"""Build the J2.3.2a package (final calibration + query-time alignment).

Contents:
  README.md     P0 fix table + old-vs-new baselines + GO decision
  docs/         6 J2.3.2a documents + failure log (F-23)
  code/         fixed calibrator + fixed query builder + baselines
  results/      runs/j2_3_2a_alignment (manifest, stats, baselines)
"""
from __future__ import annotations

import shutil
import tarfile
import tempfile
from pathlib import Path

ROOT = Path("/home/moz/World-GS")
PKG = "j2_3_2a_alignment_package_20261106"
OUT = ROOT / "dist" / f"{PKG}.tar.gz"

README = """# J2.3.2a 最终校准与查询时间对齐包

日期：2026-11-06 | 判定：**GO（D4 准入，本轮不启动）**
验证：Workflow 三镜头 skeptic 全过（校准器/查询/帧映射各
fix_holds=True）——agent 从零独立重建 CSV 逐行一致

## 两个 P0 修复

| P0 | 确认 | 修复 | 实测 |
|---|---|---|---|
| A 校准器 min-first | 39 可行阈值取了最小(0.0036/TPR 22%) | 全部可行→取 **max** | thr **0.0188** / FPR 4.93% / TPR 50.4%（审计预测 0.0183 精确命中） |
| B 查询名实不符 | 412/412 真主 delta=**0.600s** vs 名义 1.0s | 最接近 query 的自身观测+0.2s 容差 | 全部 delta=**1.000s**；error max 0.200 |

**Skeptic 独立复算**：A 逐行代码+数值精确；B 独立重建索引+
新旧 CSV 逐行复现（旧 0.600 是真实 pre-fix 产物）；C 0/3,513
越界+0037 模板跟踪 MAE 9.3px（恒等映射最优）。

## 新基线（B4 冻结，3,513 候选 / 0 缺失）

| | val Tier-1 M_hard | top1 (val/expl) |
|---|---|---|
| **B0 静态** | **0.0339** | 0.650 / 0.790 |
| B4 预测 | 0.0114 | 同 |

top1 降自 0.783/0.870——真实 1.0s 预测比 0.6s 更难（任务难度
回归，非协议退化）。旧 0.0594 属 0.6s 协议，作废。

## D4 新预注册门槛（四条件并发）

1. val Tier-1 M_hard > 0.0339；
2. top1 ≥ 0.650/0.790；
3. 固定覆盖率 ≥0.50 下 wrong-identity 改善（A1 val 0.258）；
4. 多源非单视频驱动。
λ∈{0.1,0.3,0.5,1.0}（仅 val）；Final Test 零接触。

## 目录

docs/(6 文档+failure_log) code/(4 文件) results/(4 项)
"""


def main() -> None:
    stage = Path(tempfile.mkdtemp(prefix="wmeval_j232a_pkg_"))
    root = stage / PKG
    root.mkdir()

    (root / "docs").mkdir()
    for f in ("j2_3_2a_calibration_fix.md",
              "j2_3_2a_temporal_query_fix.md",
              "j2_3_2a_frame_mapping_audit.md",
              "j2_3_2a_candidate_comparison.md",
              "j2_3_2a_b0_b4_recalibration.md",
              "j2_3_2a_d4_go_no_go.md", "failure_log.md"):
        shutil.copy2(ROOT / "docs" / f, root / "docs" / f)

    (root / "code").mkdir()
    shutil.copy2(ROOT / "src/wmeval/latent_association.py", root / "code")
    shutil.copy2(ROOT / "src/build_fixed_query_v3.py", root / "code")
    shutil.copy2(ROOT / "src/run_j2_3_2a_baselines.py", root / "code")
    # run_j2_2_baselines.py not present (2a script replaces it)

    shutil.copytree(ROOT / "runs/j2_3_2a_alignment", root / "results",
                    ignore=shutil.ignore_patterns("__pycache__"))

    (root / "README.md").write_text(README, encoding="utf-8")

    OUT.parent.mkdir(exist_ok=True)
    with tarfile.open(OUT, "w:gz") as tar:
        tar.add(root, arcname=PKG)
    n = len(tarfile.open(OUT).getmembers())
    print(f"[pkg] {OUT} ({OUT.stat().st_size/1e6:.2f} MB, {n} members)")


if __name__ == "__main__":
    main()
