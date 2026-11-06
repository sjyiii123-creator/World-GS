"""Build the E2 prompt-direction review package (8 samples).

Purpose: the user personally eyeballs real vs generated videos against the
prompt's direction wording — after numerical evidence showed 0044's prompt
("from left to right") contradicts the real track (x 0.663 -> 0.145, i.e.
right-to-left) and 0035's gen direction is reversed vs real.

NO video content is ever inspected by the model: this script only copies
files and writes numeric evidence (from v3.1 tracks.json) into a README.

Staging under tempfile.mkdtemp (system-managed), archive to dist/.
"""
from __future__ import annotations

import json
import re
import shutil
import tarfile
import tempfile
from pathlib import Path

SIDS = [
    "video_0002_0_2_6b_f60-210", "video_0003_0_3_7b_f0-150",
    "video_0016_0_16_72b_f0-150", "video_0024_0_24_99b_f60-210",
    "video_0030_0_30_134b_f0-150", "video_0035_0_35_163b_f0-150",
    "video_0037_0_37_168b_f0-150", "video_0044_0_44_202b_f60-210",
]
PKG_NAME = "e2_prompt_direction_review_20261105"
OUT = Path("dist") / f"{PKG_NAME}.tar.gz"

REAL_DIR = Path("repos/SynPeDAS/JAAD_crossing_moments")
GEN_DIR = Path("repos/SynPeDAS/generated_video")
PROMPT_DIR = Path("repos/SynPeDAS/video_generation_prompts")
V31 = Path("runs/synpedas_smoke_v3_1")

DIR_SENT = re.compile(
    r"[^.!?]*(?:\bleft\b|\bright\b|toward[s]? the camera|away from the camera|"
    r"approach(?:es|ing)?|cross(?:es|ing)?|walks? across|steps? across|"
    r"travers(?:es|ing))[^.!?]*[.!?]", re.I)


def track_rows(tracks):
    """Report both the most-detected and the largest-|dx| person track."""
    persons = [t for t in tracks if t["category"] == "person"]
    if not persons:
        return []
    def line(t):
        cs = t["norm_centers"]
        x0, x1 = cs[0][0], cs[-1][0]
        y0, y1 = cs[0][1], cs[-1][1]
        dx, dy = x1 - x0, y1 - y0
        d = "R->L" if dx < -0.02 else ("L->R" if dx > 0.02 else "~static")
        return (t, x0, x1, dx, dy, d)
    by_n = max(persons, key=lambda t: t["num_detections"])
    by_dx = max(persons, key=lambda t: abs(
        t["norm_centers"][-1][0] - t["norm_centers"][0][0]))
    out = [line(by_n)]
    if by_dx["track_id"] != by_n["track_id"]:
        out.append(line(by_dx))
    return out


def main() -> None:
    stage = Path(tempfile.mkdtemp(prefix="wmeval_dir_review_"))
    root = stage / PKG_NAME
    (root / "samples").mkdir(parents=True)

    table_lines = []
    prompt_notes = {}
    for sid in SIDS:
        sdir = root / "samples" / sid
        sdir.mkdir()
        shutil.copy2(PROMPT_DIR / f"{sid}.txt", sdir / "prompt.txt")
        meta_src = GEN_DIR / f"{sid}.json"
        if meta_src.exists():
            shutil.copy2(meta_src, sdir / "metadata.json")
        shutil.copy2(REAL_DIR / f"{sid}.mp4", sdir / "real.mp4")
        shutil.copy2(GEN_DIR / f"{sid}.mp4", sdir / "gen.mp4")
        for side in ("real", "gen"):
            src = V31 / f"{sid}__{side}"
            if src.exists():
                shutil.copytree(src, sdir / f"v3_1_{side}",
                                ignore=shutil.ignore_patterns("__pycache__"))
            tk = src / "tracks.json"
            if tk.exists():
                for (t, x0, x1, dx, dy, d) in track_rows(json.loads(tk.read_text())):
                    tag = "max-n" if t["track_id"] == max(
                        (x for x in json.loads(tk.read_text())
                         if x["category"] == "person"),
                        key=lambda z: z["num_detections"])["track_id"] else "max-dx"
                    table_lines.append(
                        f"| {sid[:8]} | {side} | {tag} | {t['track_id']} | "
                        f"{t['num_detections']} | {x0:.3f} | {x1:.3f} | "
                        f"{dx:+.3f} | **{d}** | {dy:+.3f} |")
        txt = (PROMPT_DIR / f"{sid}.txt").read_text(errors="replace")
        sents = [" ".join(s.split()) for s in DIR_SENT.findall(txt)]
        prompt_notes[sid] = sents[:3]

    table = "\n".join(table_lines)
    prompt_section = "\n\n".join(
        f"**{sid}**\n" + ("\n".join(f"- {s[:230]}" for s in prompt_notes[sid])
                          or "- (no directional sentence matched)")
        for sid in SIDS)

    readme = f"""# E2 Prompt-Direction 人工核查包（8 样本）

日期：2026-11-05 | 打包动机：数值审计发现 prompt 方向描述与原始视频轨迹
**可能矛盾**（0044 实锤：prompt "from left to right" vs real 轨迹
x 0.663→0.145 右→左），以及 real↔gen 方向反转（0035）。
**本包由脚本自动生成，模型未查看任何视频内容**；方向结论以你的目视为准。

## 每个样本目录包含

```
samples/<sample_id>/
  prompt.txt          生成 prompt 原文（重点读方向/行为描述句）
  metadata.json       生成参数（video2world / seed / guidance）
  real.mp4            原始 JAAD 片段（目视主对象）
  gen.mp4             生成视频（目视主对象）
  v3_1_real/          v3.1 数值证据：tracks.json（轨迹坐标）、
                      annotated.mp4（带框视频，快速核对轨迹）、
                      keyframes/（每人轨首/中/尾帧）、events.json 等
  v3_1_gen/           同上，生成侧
```

## 数值方向表（来自 v3.1 tracks.json，主 person 轨迹）

口径：每侧列"检测数最多"轨（max-n）；若与"位移最大"轨不同则加列 max-dx。
x 为归一化横坐标（0=左边缘，1=右边缘）；dx<0 = 向左移动（R→L）。
dy>0 = 向画面下方（一般=朝镜头）。

| sample | side | 口径 | trk | n | x_first | x_last | dx | 方向 | dy |
|---|---|---|---|---|---|---|---|---|---|
{table}

## Prompt 中的方向性句子（自动摘录，供对照）

{prompt_section}

## 请逐样本回答（顺序建议：0044 → 0035 → 0003 → 其余）

1. **0044**：prompt 写 "traverses the crosswalk from left to right"，
   real.mp4 中行人实际从哪边走到哪边？数值显示 R→L（0.663→0.145）。
   - 若目视同为 R→L → **prompt 描述错误（数据集构建层问题）**，
     生成视频的"未遵循方向"归因需重新评估（gen 同为 R→L，方向上其实
     与 real 一致，只是位移小）；
2. **0035**：real 数值 R→L（0.562→0.039），gen 数值 L→R（0.562→0.688）
   ——目视确认 gen 是否真的反向行走（还是检测/跟踪错位）；
3. **0003**：prompt "walks briskly from the right" vs real L→R——
   "from the right" 指出发侧还是行进方向？目视后定义该句语义；
4. 其余样本：核对 prompt 方向/行为句与 real、gen 的目视方向是否一致，
   不一致处记入 `datasets/annotations/e1_manual_labels.csv` 的
   reviewer_notes（或后续 protocol 复核表）。

## 纪律提醒

- 结论只写目视确认的事实；数值表仅供定位（x 坐标不含深度/遮挡信息）；
- 0044 若确认为 prompt 错误：该样本的 event 协议字段
  （event_protocol.csv 的 required_behavior/temporal_constraint）应按
  **真实视频行为**填写，而不是按 prompt 文字——并把"prompt 与源视频
  不符"记为 SynPeDAS 数据集本身的缺陷（论文中可作为数据质量发现
  单独报告，注意与"世界模型失败"严格区分）。
"""
    (root / "README.md").write_text(readme, encoding="utf-8")

    OUT.parent.mkdir(exist_ok=True)
    with tarfile.open(OUT, "w:gz") as tar:
        tar.add(root, arcname=PKG_NAME)
    n_files = sum(1 for _ in tarfile.open(OUT).getmembers())
    size_mb = OUT.stat().st_size / 1e6
    print(f"[pkg] {OUT} ({size_mb:.1f} MB, {n_files} members)")
    print(f"[pkg] staging left at {stage} (system temp, self-cleaning)")


if __name__ == "__main__":
    main()
