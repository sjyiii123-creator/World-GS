"""Build the COMPLETE J1.1 package (all three experiments finished).

Extends j1_1_audit_package with the finished J3-lite multi-candidate
results and J2-readiness dataset/verdict.  The earlier snapshot package
(j1_1_audit_package_20261105.tar.gz, J3 marked RUNNING) is kept for
provenance; this is the complete deliverable.

Contents:
  README.md        fix summary + ALL experiment results + 9-question answers
  docs/            5 J1.1 documents (fully populated) + failure log
  code/            fixed modules + 5 run scripts + selftest (24 tests)
  audit_results/   runs/j1_1_audited JSON/CSV
  j3_lite/         runs/j3_lite (metrics + competitions detail)
  j2_readiness/    runs/j2_readiness (sequences npz + readiness.json)
  logs/            selftest + all run logs
"""
from __future__ import annotations

import shutil
import tarfile
import tempfile
from pathlib import Path

ROOT = Path("/home/moz/World-GS")
PKG = "j1_1_complete_package_20261105"
OUT = ROOT / "dist" / f"{PKG}.tar.gz"

README = """# J1.1 完整交付包（三个实验全部完成）

日期：2026-11-05 | 串行 driver（setsid）全程存活,内存零中断
自测 24/24;原始 J0/J1/v3.1/人工标注零改动。

## 修复与实验结果总表

| 项 | 结果 |
|---|---|
| D1 校准修复 | 严格阈值 0.8122(train FPR 4.93%/TPR 42.39%);FPR≤5% 与高召回不可兼得(如实) |
| D2 指标修复 | 7 指标显式分母;test: J1 fused@strict FPR 0.054/precision 0.891 |
| D3 帧映射 | M1 核验通过(结构 211+IoU 10 样本);0037=声明失实非映射错误 |
| D4/D8 遮挡采样 | 修正后生效;B(可见性优先) precision 0.933 最优 |
| 多候选评价 | **Top-1: kin 0.807 > cos 0.598**;accept precision 0.691(≠pair 级 0.891) |
| J3-lite 三态 | coverage 0.626/wrong-sel 0.309;相对 J1-无拒识 +9pp precision,**形态需联合排序**(见 j3_lite_results §3) |
| J2 readiness | 300 序列/泄漏 0/有效秩 383-384(无塌缩)/same-cos 0.757±0.102 |

## 九问速答(详文见各文档)

1. 校准已修复(严格协议+回归);
2. 实际值:严格点 FPR 4.93%/TPR(召回) 42.39%(train),test FPR 0.066/召回 0.473;
3. 映射问题:仅 0037 窗口声明失实(映射本身正确,未入 J1 test);
4. 遮挡采样生效(B≠A,precision 0.933);
5. 多候选:cos 排序 0.598 弱于 kin 0.807——pair 级结论不可外推;
6. J3-lite 相对 J1-无拒识改善(+9pp)但绝对值差,J0-kin-三态对照未运行(如实);
7. J2 数据可信:泄漏 0/无塌缩/负可区分;
8. J2 增量假设:预测漂移补偿长 gap 召回(未验证);
9. 不建议立即训练 J2——先补联合排序与 J0-三态对照(触发门见 j2_training_readiness §4)。

## 目录

docs/(5 文档+failure_log) code/(7 文件) audit_results/(6) j3_lite/(4)
j2_readiness/(2) logs/(6)
复现:docs/commands.md §12b(串行+setsid 模式,F-16)。
"""


def main() -> None:
    stage = Path(tempfile.mkdtemp(prefix="wmeval_j11c_"))
    root = stage / PKG
    root.mkdir()

    (root / "docs").mkdir()
    for f in ("j1_1_independent_audit.md", "j1_1_calibration_report.md",
              "j1_1_frame_alignment.md", "j3_lite_results.md",
              "j2_training_readiness.md", "failure_log.md"):
        shutil.copy2(ROOT / "docs" / f, root / "docs" / f)

    (root / "code").mkdir()
    shutil.copy2(ROOT / "src/wmeval/latent_features.py", root / "code")
    shutil.copy2(ROOT / "src/wmeval/latent_association.py", root / "code")
    for f in ("run_j1_1_audit.py", "run_j1_1_occlusion.py",
              "run_j3_lite.py", "run_j2_readiness.py", "selftest_j.py"):
        shutil.copy2(ROOT / "src" / f, root / "code" / f)

    (root / "audit_results").mkdir()
    for f in ("metrics.json", "report.csv", "occlusion_strategies.json",
              "occlusion_pairs_all.csv", "occlusion_pairs_prefer_visible.csv",
              "occlusion_pairs_weighted.csv"):
        p = ROOT / "runs/j1_1_audited" / f
        if p.exists():
            shutil.copy2(p, root / "audit_results" / f)

    shutil.copytree(ROOT / "runs/j3_lite", root / "j3_lite",
                    ignore=shutil.ignore_patterns("*.done"))
    shutil.copytree(ROOT / "runs/j2_readiness", root / "j2_readiness",
                    ignore=shutil.ignore_patterns("*.done"))

    (root / "logs").mkdir()
    for f in ("selftest_j_output.txt", "occlusion_run.log",
              "j3_lite_run.log", "j2_readiness_run.log"):
        p = ROOT / "dist/audit_logs" / f
        if p.exists():
            shutil.copy2(p, root / "logs" / f)

    (root / "README.md").write_text(README, encoding="utf-8")

    OUT.parent.mkdir(exist_ok=True)
    with tarfile.open(OUT, "w:gz") as tar:
        tar.add(root, arcname=PKG)
    n = len(tarfile.open(OUT).getmembers())
    print(f"[pkg] {OUT} ({OUT.stat().st_size/1e6:.1f} MB, {n} members)")


if __name__ == "__main__":
    main()
