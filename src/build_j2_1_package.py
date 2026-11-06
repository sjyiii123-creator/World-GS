"""Build the J2.1 package (protocol fixes + module architecture).

Contents:
  README.md     gate results + CONDITIONAL-GO + ten-question answers
  docs/         8 J2.1 documents + failure log
  code/         re-evaluation script + latent_world_state module (6 files)
                + regression tests
  results/      runs/j2_1_reeval (fixed-protocol re-evaluation JSONs)
"""
from __future__ import annotations

import shutil
import tarfile
import tempfile
from pathlib import Path

ROOT = Path("/home/moz/World-GS")
PKG = "j2_1_protocol_fix_package_20261105"
OUT = ROOT / "dist" / f"{PKG}.tar.gz"

README = """# J2.1 协议修复与模块对齐包

日期：2026-11-05 | Gates 0/1/2 全过 → **CONDITIONAL-GO**
自测：J2.1 回归 12/12 + legacy 套件（35+20）全绿
原始 J2 结果/checkpoint 只读保留；新产物在 runs/j2_1_reeval/

## 修复摘要

| 指认 | 核实 | 修复 | 影响 |
|---|---|---|---|
| 身份负样本误标（非对角线） | 属实（42/3540 对） | (source,actor) mask + hard/easy | <0.002，负结果维持 |
| 候选共享真主 delta（泄漏） | 属实 | 每候选独立 delta + 无效→unknown | 语义修复；本床数值不变（时间聚集局限） |
| 源数口径错误 | 属实 | 机器复核 53/15/8/2 源 + 勘误 | final_test 声明强化 |

## 修正后核心数字（旧 checkpoint 只读重评）

区分间距（val）：J1 静态 **0.256** / J2-C 0.111 / J2-D 0.108；
hard 负样本（同源竞争者）cos：J1 0.557 vs J2-C **0.686**（预测特征
在真实同场景竞争者上更弱）——**J2 v1 负结果在公平复核后成立**。

## 十问速答（详 j2_1_go_no_go.md）

1是(42 对)；2 J1 0.256/J2-C 0.111；3是(独立 delta)；4 53/15/8/2 源；
5成立；6是(off-by-default)；7是(links 原样+legacy 全绿)；8已知路径
全消除；9设计值得/时机未到；10缺人工真值+J2 有效增量。

## CONDITIONAL-GO 内容

GO：J2 v2-B 残差设计全量训练（条件：监控三件套+过拟合梯）。
NO-GO：C/D 消融（待 B 有效）、V-JEPA 2.1、事件级实验（双门槛未满足）。

## 目录

docs/(8 文档+failure_log) code/(重评器+latent_world_state 六文件
+回归测试) results/(pair_reeval+competition_reeval)
"""


def main() -> None:
    stage = Path(tempfile.mkdtemp(prefix="wmeval_j21_pkg_"))
    root = stage / PKG
    root.mkdir()

    (root / "docs").mkdir()
    for f in ("j2_1_independent_audit.md", "j2_1_identity_label_fix.md",
              "j2_1_candidate_time_alignment.md",
              "j2_1_dataset_independence.md",
              "j2_1_old_checkpoint_reevaluation.md",
              "jepa_latent_world_state_architecture.md",
              "j2_v2_training_design.md", "j2_1_go_no_go.md",
              "failure_log.md"):
        shutil.copy2(ROOT / "docs" / f, root / "docs" / f)

    (root / "code").mkdir()
    shutil.copy2(ROOT / "src/run_j2_1_reevaluation.py", root / "code")
    shutil.copy2(ROOT / "src/selftest_j2_1.py", root / "code")
    lws = root / "code/latent_world_state"
    lws.mkdir()
    for f in sorted((ROOT / "src/wmeval/latent_world_state").glob("*.py")):
        shutil.copy2(f, lws / f.name)

    (root / "results").mkdir()
    for f in sorted((ROOT / "runs/j2_1_reeval").glob("*.json")):
        shutil.copy2(f, root / "results" / f.name)

    (root / "README.md").write_text(README, encoding="utf-8")

    OUT.parent.mkdir(exist_ok=True)
    with tarfile.open(OUT, "w:gz") as tar:
        tar.add(root, arcname=PKG)
    n = len(tarfile.open(OUT).getmembers())
    print(f"[pkg] {OUT} ({OUT.stat().st_size/1e6:.2f} MB, {n} members)")


if __name__ == "__main__":
    main()
