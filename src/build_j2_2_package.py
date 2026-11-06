"""Build the J2.2 package (residual predictor round).

Contents:
  README.md     boundary fixes + B0-B4 results + CONDITIONAL-GO
  docs/         6 J2.2 documents + failure log (F-19)
  code/         residual predictor module + 5 run scripts + regression
                tests + boundary-fixed association/interfaces modules
  results/      runs/j2_2_experimental (data, 3 checkpoints, all JSON)
  logs/         data-build log
"""
from __future__ import annotations

import shutil
import tarfile
import tempfile
from pathlib import Path

ROOT = Path("/home/moz/World-GS")
PKG = "j2_2_residual_package_20261105"
OUT = ROOT / "dist" / f"{PKG}.tar.gz"

README = """# J2.2 Static Residual Predictor 包

日期：2026-11-05 | 判定：**CONDITIONAL-GO**（C 消融有条件允许）
自测：残差六项验证 7/7 + J2.1+J2.2 边界 22/22 + legacy 套件全绿
原始 J2 v1 checkpoint/j2_ablation.json 只读核验未覆盖

## 修复摘要（§一，全部属实）

| 漏洞 | 修复 |
|---|---|
| `err or 9.9` 零误差替换 | 显式 None/NaN/Inf/负数校验 |
| 单候选无证据自动 accept | 无校准阈值最高 uncertain；全无效→no_match |
| margin 写死 + links 原地改 | 参数化 + sidecar 副本化（冻结结果零突变） |

## B0–B4（区分间距，val 60，身份 mask 修正协议）

| 组 | 间距 | 结论 |
|---|---|---|
| B0 静态（零残差） | **0.249** | 天花板 |
| B1 v1 GRU@240 | 0.110 | v1 负结果 |
| B2 残差@240 | 0.154 | 残差 +0.044 |
| B3 GRU@800 | 0.159 | 数据 +0.049 |
| **B4 残差@800** | **0.189** | 叠加有效，未超静态（−0.06） |

零残差==静态基线**精确**成立（1e-6）；监控十项（共性化减轻未消除
batchmean 0.853 vs 0.906）；模型选择按 val 间距非 loss。

## 数据扩充（零泄漏）

池 +29 源/+284 actor（82/444）；训练抽样 800/1,245（预算声明）；
25 排除源未重划；0 提取错误；身份真值=JAAD 官方 ID。

## 关联（固定覆盖率诚实分析）

A3 sweep"最优 0.119@0.72"= **A1 行为退化**（宽松阈值门全开）——
预测证据无真实增量；A2 改善来自静态特征；top1 恒 0.805。

## 十二问速答（详 j2_2_go_no_go.md）

1是 2否 3+29源/+284actor 4是(精确) 5**否**(0.189<0.249) 6**是**
(>原始GRU) 7否 8否(退化归因) 9减轻未消除 10**有条件允许**
(判据:间距>0.249) 11未具备。

## 目录

docs/(6 文档+failure_log) code/(7 文件) results/(8 项) logs/(构建日志)
"""


def main() -> None:
    stage = Path(tempfile.mkdtemp(prefix="wmeval_j22_pkg_"))
    root = stage / PKG
    root.mkdir()

    (root / "docs").mkdir()
    for f in ("j2_2_data_audit.md", "j2_2_residual_design.md",
              "j2_2_training_report.md", "j2_2_ablation.md",
              "j2_2_association_results.md", "j2_2_go_no_go.md",
              "failure_log.md"):
        shutil.copy2(ROOT / "docs" / f, root / "docs" / f)

    (root / "code").mkdir()
    shutil.copy2(ROOT / "src/wmeval/latent_world_state/residual_predictor.py",
                 root / "code")
    # boundary-fixed modules (J2.2 updates of J2.1 modules)
    for f in ("association_evidence.py", "interfaces.py"):
        shutil.copy2(ROOT / "src/wmeval/latent_world_state" / f,
                     root / "code" / f)
    for f in ("build_j2_2_data.py", "run_j2_2_train.py",
              "run_j2_2_association.py", "selftest_j2_2.py",
              "selftest_j2_1.py"):
        shutil.copy2(ROOT / "src" / f, root / "code" / f)

    shutil.copytree(ROOT / "runs/j2_2_experimental", root / "results",
                    ignore=shutil.ignore_patterns("__pycache__"))

    (root / "logs").mkdir()
    p = ROOT / "dist/audit_logs/j2_2_data_build.log"
    if p.exists():
        shutil.copy2(p, root / "logs" / "j2_2_data_build.log")

    (root / "README.md").write_text(README, encoding="utf-8")

    OUT.parent.mkdir(exist_ok=True)
    with tarfile.open(OUT, "w:gz") as tar:
        tar.add(root, arcname=PKG)
    n = len(tarfile.open(OUT).getmembers())
    print(f"[pkg] {OUT} ({OUT.stat().st_size/1e6:.1f} MB, {n} members)")


if __name__ == "__main__":
    main()
