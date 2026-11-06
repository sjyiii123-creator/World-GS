"""Build the J2.3 diagnostics package (read-only diagnosis round).

Contents:
  README.md     gate results + hypothesis verdicts + GO decision
  docs/         6 J2.3 documents + failure log (F-20)
  code/         diagnostics + visibility memory + tier builder + fixed
                modules + regression tests
  results/      runs/j2_3_diagnostics (representation/tier/visibility JSONs)
"""
from __future__ import annotations

import shutil
import tarfile
import tempfile
from pathlib import Path

ROOT = Path("/home/moz/World-GS")
PKG = "j2_3_diagnostics_package_20261105"
OUT = ROOT / "dist" / f"{PKG}.tar.gz"

README = """# J2.3 GOT-JEPA 表示诊断包

日期：2026-11-05 | 判定：**GO（受控 D4 判别训练）**
自测：27/27（G0 契约修复 + 全部边界回归）；零新训练、零旧结果覆盖

## Gate 结果

| Gate | 结果 |
|---|---|
| G0 契约修复 | 非 accept 不提交身份（tendency 分离）；三处 thr-or-default 修复 |
| G1 共性化 | **H1 强支持**：PR 18.25→6.46 / mean-dir 0.692→0.845 / 方差减半；常量反例（塌缩 cov=0） |
| G2 Tier 负样本 | Tier-1 空间门控竞争者 391/412（95%）；gap 均衡 128/142/130 |
| G3 可见性记忆 | V2/V3 M_hard 0.085 vs V0 0.074；V1≈V0 → H3 弱 |

## 三假设裁决

**H1（共性化）+ H2（目标不一致）主导，H3（遮挡污染）次要**——
B4 same 0.780 最高但 hard 0.653 同升（同向漂移）；两轮负结果
（v1 0.110 → v2.2 0.189）首次获得可执行因果链。

## GO 内容（下一轮唯一训练项）

D4 = B4 残差架构 + Tier-1 hard-negative ranking（λ 单变量、
对照 B4/B0）；**判定标准预注册：val M_hard > 0.112（静态基线）**；
未达标保留第三轮负结果。CONDITION：Tier-1 采样器时间重叠过滤 +
M_hard 早停。NO-GO：D5 协方差（待 D4）、D6 扰动（池化特征不满足
Copy-Paste 前提）、Teacher-Student、事件级。

## 目录

docs/(6 文档+failure_log) code/(8 文件) results/(4 项)
"""


def main() -> None:
    stage = Path(tempfile.mkdtemp(prefix="wmeval_j23_pkg_"))
    root = stage / PKG
    root.mkdir()

    (root / "docs").mkdir()
    for f in ("j2_3_got_jepa_literature_alignment.md",
              "j2_3_representation_diagnostics.md",
              "j2_3_hard_negative_audit.md",
              "j2_3_visibility_memory_results.md",
              "j2_3_discriminative_query_design.md",
              "j2_3_training_go_no_go.md", "failure_log.md"):
        shutil.copy2(ROOT / "docs" / f, root / "docs" / f)

    (root / "code").mkdir()
    shutil.copy2(ROOT / "src/jepa_experiments/diagnose_representation.py",
                 root / "code")
    shutil.copy2(ROOT / "src/wmeval/latent_world_state/visibility_memory.py",
                 root / "code")
    shutil.copy2(ROOT / "src/wmeval/latent_world_state/association_evidence.py",
                 root / "code")
    for f in ("build_tiered_negatives.py", "run_j2_3_visibility.py",
              "run_j2_2_association.py", "selftest_j2_1.py"):
        shutil.copy2(ROOT / "src" / f, root / "code" / f)

    shutil.copytree(ROOT / "runs/j2_3_diagnostics", root / "results",
                    ignore=shutil.ignore_patterns("__pycache__"))

    (root / "README.md").write_text(README, encoding="utf-8")

    OUT.parent.mkdir(exist_ok=True)
    with tarfile.open(OUT, "w:gz") as tar:
        tar.add(root, arcname=PKG)
    n = len(tarfile.open(OUT).getmembers())
    print(f"[pkg] {OUT} ({OUT.stat().st_size/1e6:.2f} MB, {n} members)")


if __name__ == "__main__":
    main()
