"""Build the E2.1 correction package (P0 fixes + conditioning audit).

Contents (all E2.1 increments; v1 audit table included for old/new diff):

  README.md                package guide + P0-1 old/new diff summary
  docs/                    six E2.1 documents + updated failure log
  code/                    direction_audit v2 + 30-test suite + v2 builder
  audit_tables/            v2 protocol CSVs (+ v1 for comparison)
  conditioning_evidence/   frame-similarity numeric evidence (P0-4)
  logs/                    35 + 20 + 30 selftest logs

Model never inspects images; this script only copies files.
Staging under tempfile; archive to dist/.
"""
from __future__ import annotations

import shutil
import tarfile
import tempfile
from pathlib import Path

ROOT = Path("/home/moz/World-GS")
PKG = "e2_1_correction_package_20261105"
OUT = ROOT / "dist" / f"{PKG}.tar.gz"

README = """# E2.1 修正包（P0 修复 + 条件审计 + 正式实验前置）

日期：2026-11-05 | 触发：外部审计任务书（三个确定性实现问题）
回归状态：**35 + 20 + 30 全部通过**（logs/ 下有完整输出）

## 本包内容导览

| 部分 | 内容 | 用途 |
|---|---|---|
| `docs/e2_1_code_audit.md` | P0-1/2/3 修复记录 + 新旧数值差异 | 修复审计 |
| `docs/synpedas_conditioning_audit.md` | **核心新发现**：video2world 条件审计（前 ~16 帧来自真实视频 + 77 帧生成 roll-out；生成合同不含"复制源后段"） | 协议语义依据 |
| `docs/e2_1_event_protocol.md` | 十三分层字段协议 + 四重点样本复核材料清单 | 人工复核入口 |
| `docs/theory_to_code_matrix.md` | DrivingGen/OC-SORT/STL 监控/Selective Classification 四论文对应矩阵（均不新增代码） | 论文理论定位 |
| `docs/drivinggen_baseline_analysis.md` | 官方指标逐项对齐（可公平对齐/域内对照/不可比） | B1 基线口径 |
| `docs/e2_1_next_experiments.md` | S1–S5 实验序列 + 人工侧进入条件 | 下一轮计划 |
| `code/` | `direction_audit.py`（v2 语义）+ 30 项回归 + v2 表生成器 | 可复现 |
| `audit_tables/` | **v2 协议表**（机器列已填/人工列 pending）+ v1 表（对照） | 人工回填对象 |
| `conditioning_evidence/` | 8 样本逐帧差数值（0024 f18 硬跳变 +72 等） | P0-4 证据 |

## P0-1 新旧差异速览（16 侧趋势分类 0 变化）

修复正确且实锤（三角轨迹 8.18→0.60），但当前 8 样本主轨高度近似
单调——仅 3 处数值微差（0016 real 1.259→1.234、gen 1.674→1.584、
0002 gen 1.234→1.224），**growing/stable/shrinking 分类全部不变**。
风险由 P0-1a–g 七项测试封堵（非单调轨迹未来必现场景）。

## 人工待办（进入 B0–B3 的四个条件）

1. `e1_manual_labels.csv` 20 行真值标注；
2. `event_protocol.csv` val 子集复核；
3. **v2 审计表终审**：0044 conflict_confirmed / actor_match_status /
   0035 归因（材料见 e2_1_event_protocol.md §2，含 0044 的 JAAD GT
   新线索：crossing_point=74 落在条件帧范围内）；
4. `e2_split_proposal.csv` 划分确认。

## 相关历史包（未变更，可按需取用）

- `e2_direction_audit_correction_20261105.tar.gz`（上一轮方向审计修正，
  含外部独立审计包与 16 张对照图）
- `e2_prompt_direction_review_20261105.tar.gz`（视频与带框材料，214MB）
- `e2_annotation_kit_20261105.tar.gz`（标注工作包）

## 不可变性

原始 prompt / 原始视频 / 生成视频 / v3.1 输出 / v1 审计表全部未动
（时间戳冻结核验于 e2_1_code_audit.md；SHA256 由回归测试 7 每次重验）。
"""


def main() -> None:
    stage = Path(tempfile.mkdtemp(prefix="wmeval_e21_pkg_"))
    root = stage / PKG
    root.mkdir()

    (root / "docs").mkdir()
    for f in ("e2_1_code_audit.md", "synpedas_conditioning_audit.md",
              "e2_1_event_protocol.md", "theory_to_code_matrix.md",
              "drivinggen_baseline_analysis.md",
              "e2_1_next_experiments.md", "failure_log.md"):
        shutil.copy2(ROOT / "docs" / f, root / "docs" / f)

    (root / "code").mkdir()
    shutil.copy2(ROOT / "src/wmeval/direction_audit.py", root / "code")
    for f in ("selftest_direction_audit.py", "build_direction_audit.py"):
        shutil.copy2(ROOT / "src" / f, root / "code" / f)

    (root / "audit_tables").mkdir()
    for f in ("prompt_source_generation_audit_v2.csv",
              "prompt_correction_sidecar_v2.csv",
              "prompt_source_generation_audit.csv"):  # v1 for diff
        shutil.copy2(ROOT / "datasets/annotations" / f,
                     root / "audit_tables" / f)

    (root / "conditioning_evidence").mkdir()
    for f in ("frame_similarity.csv", "frame_similarity_full.csv"):
        shutil.copy2(ROOT / "runs/e2_1_conditioning" / f,
                     root / "conditioning_evidence" / f)

    (root / "logs").mkdir()
    for f in ("selftest_metrics_output.txt", "selftest_e2_output.txt",
              "selftest_direction_audit_output.txt"):
        shutil.copy2(ROOT / "dist/audit_logs" / f, root / "logs" / f)

    (root / "README.md").write_text(README, encoding="utf-8")

    OUT.parent.mkdir(exist_ok=True)
    with tarfile.open(OUT, "w:gz") as tar:
        tar.add(root, arcname=PKG)
    n = len(tarfile.open(OUT).getmembers())
    print(f"[pkg] {OUT} ({OUT.stat().st_size/1e6:.2f} MB, {n} members)")


if __name__ == "__main__":
    main()
