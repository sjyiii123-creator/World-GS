"""Build the J2 full-audit package (stages 1-7, incl. negative results).

Contents:
  README.md            stage status table + core conclusions
  docs/                7 J2 stage documents + failure log
  code/                build_j2_v2 / run_a_baselines / run_j2_train /
                       run_j2_association
  results/             runs/j2_experimental (complete: npz, manifest,
                       checkpoints, all json, stats)
  env/                 environment audit log + external kit SHA manifest
"""
from __future__ import annotations

import shutil
import tarfile
import tempfile
from pathlib import Path

ROOT = Path("/home/moz/World-GS")
PKG = "j2_full_audit_package_20261105"
OUT = ROOT / "dist" / f"{PKG}.tar.gz"

README = """# J2 全流程审计包（阶段 1–7 完整，含负结果）

日期：2026-11-05 傍晚 | 外部工具包 WorldGS_J2_Configuration（SHA 7/7）
数据验收 preflight exit 0 | 训练六步梯全过 | 原始 J0/J1/J3-lite 零改动

## 九阶段状态表

| 阶段 | 状态 | 核心结果 |
|---|---|---|
| 1 环境审计 | ✅ | torch 2.7.1+cu128 / 15.2GB 空闲 / DINOv2 本地 |
| 2 数据 v2 | ✅ | 412 序列、源级四划分（train240/val60/expl100/final12-never-viewed）、首观测目标、preflight **exit 0** |
| 3 A 基线 | ✅ | **A1（kin 排序+kin 拒识）最优**：coverage 0.72/wrong-acc 0.147；A2 cos 拒识保守不更准；no-match FA 0.31；带统计修复生效 |
| 4 J2 训练 | ✅ | 六步梯全过（overfit 1.05→0.0001）；J2-C val 0.270；1s/0.05GB |
| 5 消融 | ✅ **负结果** | 区分间距 **J2-C 0.110 < 静态 0.257**——预测特征共性化，无增量 |
| 6 融合 | ✅ **负结果** | A3 无改善（0.119→0.167）；JOINT 联合排序有害（top1 0.805→0.415） |
| 7 事件级 | ❌ **未运行** | 双门槛未满足（人工真值未填 + J2 无增量）——协议已固定 |
| 8 V-JEPA 2.1 | ❌ 未启动 | 前置（DINOv2+J2 有效结果）未成立 |
| 9 论文验证 | 部分完成 | 已运行/训练/验证/独立测试/未验证五级区分见各报告 |

## 核心结论（一句话）

在 240 训练序列量级下，**J2 v1 未显示超越静态 DINOv2 的增量**
（消融区分力减半、融合无改善）——按预注册判据如实报告；
**A1（运动学排序+运动学选择性拒识）是当前最强基线**，J2 证据列
暂不接入 actor_binding。下一轮候选：全量 7,920 正对训练 +
hard-negative 排序损失（触发条件已在 ablation 报告 §3 预注册）。

## 目录

```
docs/     7 份阶段报告 + failure_log
code/     4 个运行脚本
results/  j2_experimental 全部产物（npz/manifest/2 checkpoints/
          train_log/a_baselines{,_sweep}/ablation/association/stats）
env/      runs_env_audit.log + 外部工具包 SHA 清单
```
复现：docs/commands.md §12c（含 PATH 注意事项）。
"""


def main() -> None:
    stage = Path(tempfile.mkdtemp(prefix="wmeval_j2_pkg_"))
    root = stage / PKG
    root.mkdir()

    (root / "docs").mkdir()
    for f in ("j2_environment_report.md", "j2_dataset_v2_audit.md",
              "j0_j1_selective_comparison.md", "j2_training_report.md",
              "j2_ablation_report.md", "j2_actor_association_report.md",
              "j2_event_level_impact.md", "failure_log.md"):
        shutil.copy2(ROOT / "docs" / f, root / "docs" / f)

    (root / "code").mkdir()
    for f in ("build_j2_v2.py", "run_a_baselines.py",
              "run_j2_train.py", "run_j2_association.py"):
        shutil.copy2(ROOT / "src" / f, root / "code" / f)

    shutil.copytree(ROOT / "runs/j2_experimental", root / "results",
                    ignore=shutil.ignore_patterns("*.done", "__pycache__",
                                                  "_tmp*"))

    (root / "env").mkdir()
    p = ROOT / "runs_env_audit.log"
    if p.exists():
        shutil.copy2(p, root / "env" / "runs_env_audit.log")
    shutil.copy2(ROOT / "WorldGS_J2_Configuration/SHA256SUMS.txt",
                 root / "env" / "external_kit_SHA256SUMS.txt")

    (root / "README.md").write_text(README, encoding="utf-8")

    OUT.parent.mkdir(exist_ok=True)
    with tarfile.open(OUT, "w:gz") as tar:
        tar.add(root, arcname=PKG)
    n = len(tarfile.open(OUT).getmembers())
    print(f"[pkg] {OUT} ({OUT.stat().st_size/1e6:.1f} MB, {n} members)")


if __name__ == "__main__":
    main()
