"""Build the J1.1 audit package (reliability fixes + occlusion study).

Contents:
  README.md          guide + fix summary + experiment status table
  docs/              5 J1.1 documents + failure log (F-15/F-16)
  code/              fixed modules + 4 run scripts + selftest (24 tests)
  audit_results/     runs/j1_1_audited JSON/CSV (caches excluded)
  logs/              selftest + run logs (incl. in-progress J3 snapshot)

Feature caches (15MB) excluded — rebuildable via commands.md §12b.
J3-lite / J2-readiness were RUNNING at packaging time — marked
RUNNING (not claimed as results) per the no-fabrication rule.
"""
from __future__ import annotations

import shutil
import tarfile
import tempfile
from pathlib import Path

ROOT = Path("/home/moz/World-GS")
PKG = "j1_1_audit_package_20261105"
OUT = ROOT / "dist" / f"{PKG}.tar.gz"

README = """# J1.1 审计修复包（人工检查用）

日期：2026-11-05 下午 | 触发：外部审计任务书指认的可靠性缺陷
自测：**24/24 通过**（logs/selftest_j_output.txt）
原始 J0/J1/v3.1/E2 审计/人工标注/原始视频与 Prompt：零改动（保护核验）

## 修复摘要（检查入口）

| # | 缺陷 | 状态 | 检查文件 |
|---|---|---|---|
| D1 | 阈值校准静默回退（0.6913 实测 FPR 28.33% vs 声称 ≤5%） | ✅ 修复 | docs/j1_1_calibration_report.md §1-3 |
| D2 | error_merge_rate=FP/n 冒充 FPR | ✅ 修复（7 指标显式分母） | 同上 §2 + code/latent_association.py |
| D3 | 帧映射未核验 + 0037 异常 | ✅ 核验通过（M1 成立） | docs/j1_1_frame_alignment.md |
| D4/D8 | 遮挡采样（occ<2 形同虚设，A/B 首跑逐值相同） | ✅ 修正+实测生效 | docs/j3_lite_results.md 附录 A |
| D5/D6 | 缓存版本化 / 多候选语义 | ✅ 实现 | code/latent_features.py + run_j3_lite.py |
| F-15/F-16 | 修复过程 bug（死循环/round/缓存 meta）+ 内存 OOM 处置 | ✅ | docs/failure_log.md |

## 核心数字（test，严格 FPR≤5% 校准于 train）

| 规则 | acc | FPR | prec | miss |
|---|---|---|---|---|
| J0 运动学 | 0.815 | 0.340 | 0.733 | 0.023 |
| J1 cos@旧0.6913（缺陷对照） | 0.790 | 0.262 | 0.755 | 0.156 |
| **J1 cos@严格0.8122** | 0.709 | **0.066** | 0.873 | 0.527 |
| J1 fused@严格 | 0.709 | **0.054** | **0.891** | 0.540 |

遮挡三策略（500 对）：A prec 0.897 / **B(修正) 0.933** / C 0.889。
**FPR≤5% 与召回≥80% 不可兼得**（如实声明）。

## 实验状态表（未运行/运行中如实标注）

| 实验 | 状态 |
|---|---|
| J1.1 审计复评（严格校准+新指标） | ✅ 完成（audit_results/metrics.json） |
| 遮挡三策略 A/B/C | ✅ 完成（audit_results/occlusion_strategies.json） |
| **J3-lite 多候选 + 三态** | 🔄 **打包时仍在运行**（logs/j3_lite_run_partial.log 为快照）——docs/j3_lite_results.md §1-3 数据留空待填，**未声称任何结果** |
| **J2 readiness** | ⏳ 待 J3-lite 后串行启动（driver 存活）——docs/j2_training_readiness.md 为骨架 |
| J2 训练 | ❌ 未运行（触发门未满足） |

## 目录导览

```
docs/      5 份 J1.1 文档 + failure_log
code/      latent_features / latent_association（修复版）
           run_j1_1_audit / run_j1_1_occlusion / run_j3_lite / run_j2_readiness
           selftest_j（24 项回归）
audit_results/  metrics.json / report.csv / occlusion_strategies.json
                occlusion_pairs_{all,prefer_visible,weighted}.csv
logs/      selftest_j（24/24）+ occlusion 运行日志 + J3 进行中快照
```

复现命令见 docs/commands.md §12/§12b（串行 + setsid 模式说明于
failure_log F-16——16G 内存机器长任务纪律）。
"""


def main() -> None:
    stage = Path(tempfile.mkdtemp(prefix="wmeval_j11_pkg_"))
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
    src = ROOT / "runs/j1_1_audited"
    for f in ("metrics.json", "report.csv", "occlusion_strategies.json",
              "occlusion_pairs_all.csv", "occlusion_pairs_prefer_visible.csv",
              "occlusion_pairs_weighted.csv"):
        p = src / f
        if p.exists():
            shutil.copy2(p, root / "audit_results" / f)

    (root / "logs").mkdir()
    for f in ("selftest_j_output.txt", "occlusion_run.log",
              "j3_lite_run_partial.log"):
        p = ROOT / "dist/audit_logs" / f
        if p.exists():
            shutil.copy2(p, root / "logs" / f)

    (root / "README.md").write_text(README, encoding="utf-8")

    OUT.parent.mkdir(exist_ok=True)
    with tarfile.open(OUT, "w:gz") as tar:
        tar.add(root, arcname=PKG)
    n = len(tarfile.open(OUT).getmembers())
    print(f"[pkg] {OUT} ({OUT.stat().st_size/1e6:.2f} MB, {n} members)")


if __name__ == "__main__":
    main()
