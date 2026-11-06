"""Build the D4 discriminative query package.

Contents:
  README.md     training results + four-gate evaluation + LOSO
  docs/         training report + failure log (F-24)
  code/         training + evaluation + ranking loss + predictor
  results/      4 checkpoints + training/evaluation JSON + frozen baseline
"""
from __future__ import annotations

import shutil
import tarfile
import tempfile
from pathlib import Path

ROOT = Path("/home/moz/World-GS")
PKG = "d4_discriminative_query_package_20261106"
OUT = ROOT / "dist" / f"{PKG}.tar.gz"

README = """# D4 Actor-Conditioned Discriminative Query 包

日期：2026-11-06 | 判定：**三过一败（G2/G3/G4 PASS，G1 FAIL）**
基线：J2.3.2a 冻结（SHA256 已登记，零覆盖）

## 训练结果（λ ∈ {0.1, 0.3, 0.5, 1.0}）

| λ | val Tier-1 M_hard | epochs |
|---|---|---|
| 0.1 | 0.0198 | 43 |
| 0.3 | 0.0144 | 9 |
| **0.5** | **0.0212** | 43 |
| 1.0 | 0.0159 | 9 |

最优 λ=0.5——**较 B4 (0.0114) 提升 86%**，训练 6 秒/0.05GB VRAM。

## 四门评价

| Gate | 内容 | D4 | 基线 | 判定 |
|---|---|---|---|---|
| **G1 身份区分** | M_hard > 0.0339 | 0.0212 | B0=0.0339 | ❌ FAIL |
| **G2 候选排序** | Top-1 ≥ 0.650 | **0.700** | kin=0.650 | ✅ PASS |
| **G3 选择性** | cov 0.50, wrong < A1 | **0.30** | A1=1.00 | ✅ PASS |
| **G4 LOSO 稳定** | 无单源驱动 | dev **0.0077** | — | ✅ PASS |

## 三个关键发现

1. **G2 是本轮最重要的正结果**：D4 联合排序 Top-1 = 0.700 超过
   纯运动学 0.650——**判别式预测特征首次展示排序改善能力**。
   此前 B0/B4 从不参与排序（只做拒识）。
2. **G3 大幅改善**：固定 coverage 0.50 下 D4 错误率 0.30 vs
   A1 的 1.00。
3. **G1 未过**：判别式目标虽将 B4 提升 86%，但三轮训练
   （v1→v2.2→D4）始终未翻越静态特征——240 anchor 训练量
   是明确瓶颈。

## LOSO（15 val 源）

排除 video_0310 后 M_hard 反升至 0.0289（最难场景而非增益
来源）；全部偏差 ≤0.0077——**跨场景稳定**。

## Exploratory Test

M_hard D4=0.019 / B0=0.034 | Top-1 joint=0.760 / kin=0.790

## 目录

docs/(训练报告 + failure_log) code/(6 文件) results/(7 项)
"""


def main() -> None:
    stage = Path(tempfile.mkdtemp(prefix="wmeval_d4_pkg_"))
    root = stage / PKG
    root.mkdir()

    (root / "docs").mkdir()
    for f in ("d4_training_report.md", "failure_log.md"):
        shutil.copy2(ROOT / "docs" / f, root / "docs" / f)

    (root / "code").mkdir()
    for f in ("run_d4_train.py", "run_d4_eval.py",
              "run_j2_3_2a_baselines.py", "build_fixed_query_v3.py"):
        shutil.copy2(ROOT / "src" / f, root / "code" / f)
    shutil.copy2(ROOT / "src/wmeval/latent_world_state/ranking_loss.py",
                 root / "code")
    shutil.copy2(ROOT / "src/wmeval/latent_world_state/residual_predictor.py",
                 root / "code")

    shutil.copytree(ROOT / "runs/d4_experimental", root / "results",
                    ignore=shutil.ignore_patterns("__pycache__"))

    (root / "README.md").write_text(README, encoding="utf-8")

    OUT.parent.mkdir(exist_ok=True)
    with tarfile.open(OUT, "w:gz") as tar:
        tar.add(root, arcname=PKG)
    n = len(tarfile.open(OUT).getmembers())
    print(f"[pkg] {OUT} ({OUT.stat().st_size/1e6:.1f} MB, {n} members)")


if __name__ == "__main__":
    main()
