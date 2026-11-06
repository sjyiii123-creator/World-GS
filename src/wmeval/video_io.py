"""A. Video reading module.

Requirements from the research plan:
- OpenCV based;
- configurable FPS (resampling), configurable resolution;
- keyframe saving;
- correct timestamp handling.

Design notes
------------
* Timestamps are computed from the *container* fps reported by OpenCV.
  SynPeDAS pairs are known to have mismatched fps/duration between the real
  JAAD clip (150 frames) and the generated video (77 frames), so any temporal
  comparison downstream must use **normalized time** t_norm = t / duration,
  never raw frame indices.  `VideoMeta` therefore exposes both.
* Reading is lazy (generator) to keep 16GB RAM safe on 1080p clips.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np


@dataclasses.dataclass
class VideoMeta:
    path: str
    num_frames: int
    fps: float
    width: int
    height: int
    duration_s: float

    @property
    def resolution(self) -> str:
        return f"{self.width}x{self.height}"

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def norm_time(self, frame_index: int) -> float:
        """Normalized time in [0, 1] for cross-video temporal alignment."""
        if self.num_frames <= 1:
            return 0.0
        return frame_index / (self.num_frames - 1)

    def timestamp(self, frame_index: int) -> float:
        return frame_index / self.fps if self.fps > 0 else float(frame_index)


def probe(path: str | os.PathLike) -> VideoMeta:
    """Open a video with OpenCV and return its metadata without decoding frames."""
    path = str(path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"cannot open video: {path}")
    try:
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    if not fps or math.isnan(fps) or fps <= 0:
        fps = 30.0  # JAAD default; flagged by caller via meta if changed
        fps_reported = False
    else:
        fps_reported = True
    meta = VideoMeta(
        path=path,
        num_frames=num_frames,
        fps=float(fps),
        width=width,
        height=height,
        duration_s=num_frames / fps if fps else float("nan"),
    )
    meta.fps_reported = fps_reported  # type: ignore[attr-defined]
    return meta


class VideoReader:
    """Lazy frame reader with fps/resolution control and keyframe dump.

    Parameters
    ----------
    path: video file.
    target_fps: if set, frames are yielded at (approximately) this rate by
        nearest-frame selection; ``frame_index`` stays the *native* index.
    target_size: (w, h) resize applied to yielded frames (detections then run
        on resized frames; boxes are returned in resized pixel coords).
    """

    def __init__(
        self,
        path: str | os.PathLike,
        target_fps: Optional[float] = None,
        target_size: Optional[tuple[int, int]] = None,
    ) -> None:
        self.path = str(path)
        self.meta = probe(self.path)
        self.target_fps = target_fps
        self.target_size = target_size

    def __len__(self) -> int:
        return max(self.meta.num_frames, 0)

    def iter_frames(self) -> Iterator[tuple[int, float, np.ndarray]]:
        """Yield (native_frame_index, timestamp_s, frame_bgr)."""
        step = 1.0
        if self.target_fps and self.meta.fps > 0:
            step = max(1.0, self.meta.fps / self.target_fps)
        cap = cv2.VideoCapture(self.path)
        try:
            idx_f = 0.0  # float cursor in native frame units
            while True:
                idx = int(round(idx_f))
                if self.meta.num_frames and idx >= self.meta.num_frames:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if not ok:
                    break
                if self.target_size is not None:
                    frame = cv2.resize(frame, self.target_size,
                                       interpolation=cv2.INTER_AREA)
                yield idx, self.meta.timestamp(idx), frame
                idx_f += step
        finally:
            cap.release()

    def read_all(self, limit: Optional[int] = None) -> tuple[list[int], list[float], list[np.ndarray]]:
        """Eager read (only for short clips / tests)."""
        idxs, tss, frames = [], [], []
        for i, t, f in self.iter_frames():
            idxs.append(i)
            tss.append(t)
            frames.append(f)
            if limit is not None and len(frames) >= limit:
                break
        return idxs, tss, frames

    def save_keyframes(
        self,
        out_dir: str | os.PathLike,
        frame_indices: list[int],
        prefix: str = "kf",
    ) -> list[str]:
        """Decode and dump selected native frame indices as jpg."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        wanted = set(frame_indices)
        saved: list[str] = []
        cap = cv2.VideoCapture(self.path)
        try:
            idx = 0
            while wanted:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx in wanted:
                    p = out_dir / f"{prefix}_{idx:05d}.jpg"
                    cv2.imwrite(str(p), frame)
                    saved.append(str(p))
                    wanted.discard(idx)
                idx += 1
        finally:
            cap.release()
        return saved


def write_annotated_video(
    out_path: str | os.PathLike,
    frames_with_boxes: list[tuple[np.ndarray, list[dict]]],
    fps: float,
) -> str:
    """Write an annotated visualization video.

    frames_with_boxes: list of (frame_bgr, [ {track_id, bbox(xyxy), label, conf} ])
    """
    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    if not frames_with_boxes:
        raise ValueError("no frames to write")
    h, w = frames_with_boxes[0][0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    try:
        palette = {}
        for frame, boxes in frames_with_boxes:
            canvas = frame.copy()
            for b in boxes:
                tid = int(b["track_id"])
                if tid not in palette:
                    palette[tid] = tuple(int(c) for c in
                                         np.random.RandomState(tid).uniform(60, 255, 3))
                x1, y1, x2, y2 = [int(round(v)) for v in b["bbox"]]
                color = palette[tid]
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
                tag = f'#{tid} {b.get("label","?")} {b.get("conf",0):.2f}'
                cv2.putText(canvas, tag, (x1, max(12, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            writer.write(canvas)
    finally:
        writer.release()
    return out_path


def dump_video_facts(paths: list[str], out_json: str | os.PathLike) -> dict:
    """Probe a batch of videos and dump metadata (for manifest building)."""
    facts = {}
    for p in paths:
        try:
            m = probe(p)
            facts[p] = m.to_dict()
        except Exception as e:  # noqa: BLE001 - report, don't crash batch
            facts[p] = {"error": str(e)}
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(facts, f, indent=2, ensure_ascii=False)
    return facts
