"""Build the SynPeDAS sample manifest (datasets/annotations/manifest.csv).

Links the four artifacts of every sample by base name:
    real clip (JAAD_crossing_moments/{base}.mp4)
    generated video (generated_video/{base}.mp4)
    prompt (video_generation_prompts/{base}.txt)
    generation metadata (generated_video/{base}.json)

Probing is real (OpenCV) — fps/resolution/duration/num_frames are measured,
not assumed, because the SynPeDAS JSON stores ``resolution: "none"`` and no
fps for the generated side.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from .video_io import probe

BASE_RE = re.compile(r"^(video_\d{4}_0_\d+_\d+b_f\d+-\d+)$")


def _first_sentence(text: str, limit: int = 240) -> str:
    s = text.strip().split("\n")[0]
    if len(s) > limit:
        s = s[:limit - 3] + "..."
    return s


def build_manifest(
    synpedas_root: str | Path,
    out_csv: str | Path,
    probe_videos: bool = True,
) -> list[dict]:
    root = Path(synpedas_root)
    real_dir = root / "JAAD_crossing_moments"
    gen_dir = root / "generated_video"
    prompt_dir = root / "video_generation_prompts"

    real = {p.stem: p for p in sorted(real_dir.glob("*.mp4"))}
    gen = {p.stem: p for p in sorted(gen_dir.glob("*.mp4"))}
    meta = {p.stem: p for p in sorted(gen_dir.glob("*.json"))}
    prompts = {p.stem: p for p in sorted(prompt_dir.glob("*.txt"))}

    all_bases = sorted(set(real) | set(gen) | set(prompts))
    rows: list[dict] = []
    for base in all_bases:
        m = BASE_RE.match(base)
        sid = m.group(1) if m else base
        row = {
            "sample_id": sid,
            "source_video": str(real[base]) if base in real else "",
            "generated_video": str(gen[base]) if base in gen else "",
            "prompt_path": str(prompts[base]) if base in prompts else "",
            "metadata_path": str(meta[base]) if base in meta else "",
            "real_num_frames": "",
            "real_fps": "",
            "real_resolution": "",
            "real_duration_s": "",
            "gen_num_frames": "",
            "gen_fps": "",
            "gen_resolution": "",
            "gen_duration_s": "",
            "generation_seed": "",
            "guidance": "",
            "inference_type": "",
            "num_output_frames_declared": "",
            "event_description": "",
            "annotation_status": "unannotated",
        }
        # generation metadata
        if base in meta:
            try:
                j = json.loads(meta[base].read_text(encoding="utf-8"))
                row["generation_seed"] = j.get("seed", "")
                row["guidance"] = j.get("guidance", "")
                row["inference_type"] = j.get("inference_type", "")
                row["num_output_frames_declared"] = j.get("num_output_frames", "")
                row["event_description"] = _first_sentence(j.get("prompt", ""))
            except Exception as e:  # noqa: BLE001
                row["annotation_status"] = f"metadata_error:{e}"
        # probe real side
        if base in real and probe_videos:
            try:
                v = probe(real[base])
                row["real_num_frames"] = v.num_frames
                row["real_fps"] = round(v.fps, 3)
                row["real_resolution"] = v.resolution
                row["real_duration_s"] = round(v.duration_s, 3)
            except Exception as e:  # noqa: BLE001
                row["annotation_status"] = f"real_probe_error:{e}"
        # probe generated side
        if base in gen and probe_videos:
            try:
                v = probe(gen[base])
                row["gen_num_frames"] = v.num_frames
                row["gen_fps"] = round(v.fps, 3)
                row["gen_resolution"] = v.resolution
                row["gen_duration_s"] = round(v.duration_s, 3)
            except Exception as e:  # noqa: BLE001
                row["annotation_status"] = f"gen_probe_error:{e}"
        completeness = sum(bool(row[k]) for k in
                           ("source_video", "generated_video",
                            "prompt_path", "metadata_path"))
        if completeness < 4 and row["annotation_status"] == "unannotated":
            row["annotation_status"] = f"incomplete({completeness}/4)"
        rows.append(row)

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"[manifest] wrote {len(rows)} rows -> {out_csv}")
    complete = sum(1 for r in rows if all(r[k] for k in
                    ("source_video", "generated_video", "prompt_path", "metadata_path")))
    print(f"[manifest] complete quadruples: {complete}/{len(rows)}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--synpedas-root", required=True,
                    help="path to the SynPeDAS repo clone")
    ap.add_argument("--out", default="datasets/annotations/manifest.csv")
    ap.add_argument("--no-probe", action="store_true",
                    help="skip OpenCV probing (fast, metadata-only)")
    args = ap.parse_args()
    build_manifest(args.synpedas_root, args.out, probe_videos=not args.no_probe)


if __name__ == "__main__":
    main()
