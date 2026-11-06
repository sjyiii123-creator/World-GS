"""Build the D4.1 evaluation-integrity package.

Contents:
  README.md     G3 direction fix + joint score fix + time contract fix
                + seven-method comparison + go/no-go + audit entry
  docs/         6 D4.1 documents + failure log (F-25)
  code/         corrected re-evaluation script
  results/      reeval_results.json + audit_candidates.csv (per-candidate)
"""
from __future__ import annotations

import shutil
import tarfile
import tempfile
from pathlib import Path

ROOT = Path("/home/moz/World-GS")
PKG = "d4_1_evaluation_integrity_package_20261106"
OUT = ROOT / "dist" / f"{PKG}.tar.gz"

README = """# D4.1 评价完整性审计包

日期：2026-11-06 | 判定：**CONDITIONAL-GO**（G2 通过，G1 未过）
原始 D4 结果/checkpoint 只读保留；本包为修复后重评。

## 三项修复（先数值确认再修复）

| 项 | 确认缺陷 | 修复 | 实测 |
|---|---|---|---|
| G3 kin 排序方向 | 降序选最差候选→A1 wrong_50=1.0 | 升序（最佳优先） | A1 wrong_50=**0.2667**（审计精确值 8/30 ✓） |
| 联合评分公式 | kin*cos 乘积方向不明 | -kin+cos 线性组合 | D4 联合 top-1=**0.700**（确认） |
| 时间契约 | 全部候选复用真主 delta_s | 每候选独立 delta_s | 182/3513 曾不一致，已修复 |

## 七方法对照（val 60 anchor）

| 方法 | Top-1 | wrong_50 |
|---|---|---|
| Kin only | 0.650 | 0.2667 |
| Static only | 0.633 | 0.2667 |
| Kin+Static | 0.650 | 0.2667 |
| B4 only | 0.517 | 0.400 |
| Kin+B4 | 0.633 | 0.2667 |
| D4 only | 0.567 | 0.400 |
| **Kin+D4（联合）** | **0.700** | **0.2667** |

## 核心发现

1. **只有 D4 联合提升 Top-1**（0.650→0.700）——判别式预测特征
   首次展示排序改善能力（此前 B0/B4 只做拒识不做排序）；
2. **D4 联合 wrong_50 = A1（0.2667）**——固定覆盖率下不优于基线；
3. **G1 未过**：M_hard 0.0212 < B0 0.0339（三轮训练未翻越静态）；
4. **G4 LOSO 通过**：video_0310 排除后 M_hard 反升（最难场景
   而非增益来源），max dev 0.0077。

## 判定

**CONDITIONAL-GO**：G2（排序）有真实增量，G1（身份区分）未过
静态基线；G3（选择性）不劣于基线但不优于。后续需更大训练量。

## 审计入口

- `results/reeval_results.json`：七方法完整结果；
- `results/audit_candidates.csv`：逐候选（3,513 行）kin/static/b4/
  d4 全分数 + 每候选独立 delta_s；
- `code/run_d4_1_reevaluation.py`：修复版重评器（升序 kin +
  线性联合 + 每候选独立时间）。
"""


def main() -> None:
    stage = Path(tempfile.mkdtemp(prefix="wmeval_d41_pkg_"))
    root = stage / PKG
    root.mkdir()

    (root / "docs").mkdir()
    for f in ("d4_1_g3_direction_fix.md", "d4_1_time_contract_audit.md",
              "d4_1_joint_baseline_comparison.md",
              "d4_1_fixed_coverage_risk.md",
              "d4_1_source_level_loso.md", "d4_1_go_no_go.md",
              "failure_log.md"):
        shutil.copy2(ROOT / "docs" / f, root / "docs" / f)

    (root / "code").mkdir()
    shutil.copy2(ROOT / "src/run_d4_1_reevaluation.py", root / "code")

    shutil.copytree(ROOT / "runs/d4_1_reevaluation", root / "results",
                    ignore=shutil.ignore_patterns("__pycache__"))

    (root / "README.md").write_text(README, encoding="utf-8")

    OUT.parent.mkdir(exist_ok=True)
    with tarfile.open(OUT, "w:gz") as tar:
        tar.add(root, arcname=PKG)
    n = len(tarfile.open(OUT).getmembers())
    print(f"[pkg] {OUT} ({OUT.stat().st_size/1e6:.2f} MB, {n} members)")


if __name__ == "__main__":
    main()
