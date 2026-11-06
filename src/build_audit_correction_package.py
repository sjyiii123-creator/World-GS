"""Build the E2 direction-audit correction package (for human audit).

Contents (self-contained for audit review; heavy videos excluded —
unchanged since dist/e2_prompt_direction_review_20261105.tar.gz):

  README.md                 audit workflow guide (top level)
  docs/                     correction report + updated review/failure docs
  audit_tables/             the two NEW deliverable CSVs
  external_review/          the external independent review package (verbatim)
  prompts_original/         8 raw prompts + SHA256SUMS (immutability anchor)
  numeric_evidence_v3_1/    8 samples x real/gen tracks/meta/events JSON
  code/                     direction_audit module + tests + builder
  logs/                     all three selftest logs

Model never inspects any image/video; this script only copies files.
Staging under tempfile; archive written to dist/.
"""
from __future__ import annotations

import shutil
import tarfile
import tempfile
from pathlib import Path

ROOT = Path("/home/moz/World-GS")
PKG = "e2_direction_audit_correction_20261105"
OUT = ROOT / "dist" / f"{PKG}.tar.gz"

SIDS = ["video_0002_0_2_6b_f60-210", "video_0003_0_3_7b_f0-150",
        "video_0016_0_16_72b_f0-150", "video_0024_0_24_99b_f60-210",
        "video_0030_0_30_134b_f0-150", "video_0035_0_35_163b_f0-150",
        "video_0037_0_37_168b_f0-150", "video_0044_0_44_202b_f60-210"]

README = """# E2 方向审计修正包（人工审计用）

日期：2026-11-05（第二轮修正，响应外部独立审计）
背景：第一轮方向审计混淆了图像坐标/运动语义/prompt 要求/源视频行为/
模型违约五层概念。本包是修正后的完整审计材料。

## 审计顺序建议

1. **读结论**：`docs/direction_audit_report.md` §1 五类结论表
   （A 冲突=0044；B 像素差异=0035；C 语义不一致=无；D 证据不足=6；
   E 无问题=候选 0002/0030/0037）。
2. **对照外部审计**：`external_review/REVIEW_REPORT.md` 与
   `direction_review_8_samples.csv`（外部独立审计结论——本轮修正的触发源）。
3. **逐样本复核**（材料见下），把结论回填到
   `audit_tables/prompt_source_generation_audit.csv` 的
   `human_review_status` 列（pending → confirmed）与
   `audit_tables/prompt_correction_sidecar.csv` 的 `approval` 列。
4. **数值核对**：`numeric_evidence_v3_1/` 下每样本 real/gen 的
   tracks.json（norm_centers 即审计表 dx 来源）；
   `external_review/numeric_evidence.csv` 为外部独立提取（含 bbox 高度比）。

## 复核重点（按优先级）

| 样本 | 问题 | 材料 |
|---|---|---|
| 0044 | ① 源/生成起点是否同一行人；② 生成器条件输入范围（整段 vs 首段）；③ sidecar 建议文本 approve/reject | `external_review/assets/video_0044_*_raw.jpg` + 本地视频 |
| 0035 | ① 接近语义确认（尺寸/人行道参照物）；② 自车运动——像素符号差是否相机伪影 | `external_review/assets/video_0035_*_raw.jpg` |
| 0003 | 辨认 prompt 的第一/第二行人与狗，建立源侧 actor 对应 | `external_review/assets/video_0003_*_annotated.jpg` |
| 0016 | t30/t64 碎片链外观连续性 | 同上（0016） |

**视频与带框材料**（real.mp4/gen.mp4/annotated.mp4/keyframes）未变更，
沿用 `dist/e2_prompt_direction_review_20261105.tar.gz`（214MB，已分发）。

## 原始数据不可变性

- 8 份原始 prompt 原文在 `prompts_original/`，SHA256 见其
  `SHA256SUMS.txt`（与外部溯源表 `external_review/prompt_provenance_8.csv`
  一致；回归测试 7 每次运行重验）。
- v3.1 实验输出保持冻结（Nov 4 20:14），本包仅含只读副本。

## 回归状态

`logs/` 三份自测日志：主 35 项 + E2 模块 20 项 + 方向审计 14 项，
全部通过（含任务书 §五 的 7 项方向防混淆回归）。
"""


def main() -> None:
    stage = Path(tempfile.mkdtemp(prefix="wmeval_audit_pkg_"))
    root = stage / PKG
    root.mkdir()

    # docs
    (root / "docs").mkdir()
    for f in ("direction_audit_report.md", "e1_v3_1_independent_review.md",
              "failure_log.md"):
        shutil.copy2(ROOT / "docs" / f, root / "docs" / f)

    # audit tables
    (root / "audit_tables").mkdir()
    for f in ("prompt_source_generation_audit.csv",
              "prompt_correction_sidecar.csv"):
        shutil.copy2(ROOT / "datasets/annotations" / f,
                     root / "audit_tables" / f)

    # external review package (verbatim)
    shutil.copytree(ROOT / "WorldGS_E2_Review", root / "external_review",
                    ignore=shutil.ignore_patterns("__pycache__"))

    # original prompts + sha256
    (root / "prompts_original").mkdir()
    sha_lines = []
    for sid in SIDS:
        src = ROOT / "repos/SynPeDAS/video_generation_prompts" / f"{sid}.txt"
        shutil.copy2(src, root / "prompts_original" / f"{sid}.txt")
        import hashlib
        h = hashlib.sha256(src.read_bytes()).hexdigest()
        sha_lines.append(f"{h}  {sid}.txt")
    (root / "prompts_original/SHA256SUMS.txt").write_text(
        "\n".join(sha_lines) + "\n")

    # v3.1 numeric evidence (JSON only — no mp4/keyframes)
    (root / "numeric_evidence_v3_1").mkdir()
    for sid in SIDS:
        for side in ("real", "gen"):
            src = ROOT / "runs/synpedas_smoke_v3_1" / f"{sid}__{side}"
            dst = root / "numeric_evidence_v3_1" / f"{sid}__{side}"
            dst.mkdir()
            for f in ("tracks.json", "meta.json", "events.json",
                      "metrics.json", "detections.jsonl"):
                p = src / f
                if p.exists():
                    shutil.copy2(p, dst / f)

    # code
    (root / "code").mkdir()
    shutil.copy2(ROOT / "src/wmeval/direction_audit.py", root / "code")
    for f in ("selftest_direction_audit.py", "build_direction_audit.py"):
        shutil.copy2(ROOT / "src" / f, root / "code" / f)

    # logs
    (root / "logs").mkdir()
    for f in ("selftest_metrics_output.txt", "selftest_e2_output.txt",
              "selftest_direction_audit_output.txt"):
        shutil.copy2(ROOT / "dist/audit_logs" / f, root / "logs" / f)

    (root / "README.md").write_text(README, encoding="utf-8")

    OUT.parent.mkdir(exist_ok=True)
    with tarfile.open(OUT, "w:gz") as tar:
        tar.add(root, arcname=PKG)
    n = len(tarfile.open(OUT).getmembers())
    print(f"[pkg] {OUT} ({OUT.stat().st_size/1e6:.1f} MB, {n} members)")
    print(f"[pkg] staging: {stage}")


if __name__ == "__main__":
    main()
