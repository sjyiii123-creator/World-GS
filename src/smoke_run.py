"""No-argument smoke run (matches fast-path command shape).

Hardcoded single-video configuration for the first end-to-end validation:
real JAAD clip -> YOLOv10x + ByteTrack -> metrics/events/visualization.

Run from project root:
    /home/moz/miniconda3/envs/wm_eval/bin/python -I src/smoke_run.py
"""
import sys

sys.path.insert(0, "src")

from wmeval.run_minimal import process_one_video  # noqa: E402

VIDEO = "repos/SynPeDAS/JAAD_crossing_moments/video_0002_0_2_6b_f60-210.mp4"
OUT = "runs/smoke_one"
WEIGHTS = "ckpt/yolov10x.pt"

if __name__ == "__main__":
    summary = process_one_video(
        VIDEO, __import__("pathlib").Path(OUT), WEIGHTS,
        imgsz=640, conf=0.3, target_fps=None, actor="person", device=0,
        keyframes=True)
    for k, v in summary.items():
        print(f"{k}: {v}")
