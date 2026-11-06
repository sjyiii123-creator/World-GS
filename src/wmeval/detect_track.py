"""B. Object detection + multi-object tracking.

Per the research plan: YOLO detection + ByteTrack association (stable,
dependency-light).  Note this deviates from DrivingGen's internal choice
(YOLOv10 frame-0 init + SAMURAI mask tracking); the deviation and its
rationale are recorded in docs/repository_analysis.md and experiment_plan.md.

Output contract (per detection):
    track_id, category, bbox (xyxy), detection_confidence, frame_index, timestamp
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

# COCO classes used by DrivingGen's agent pipeline (person/bicycle/car/
# motorcycle/bus/truck == ids 0,1,2,3,5,7).  We keep all of them for context.
DRIVINGGEN_CLASSES = (0, 1, 2, 3, 5, 7)
COCO_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 14: "bench",
}


@dataclasses.dataclass
class Detection:
    frame_index: int          # native frame index in the source video
    timestamp: float          # seconds
    track_id: int             # -1 when the tracker assigned no id
    class_id: int
    category: str
    bbox: tuple[float, float, float, float]      # xyxy, pixel coords of the *analyzed* frame size
    conf: float
    norm_bbox: tuple[float, float, float, float] # xyxy normalized to [0,1] by frame w/h

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class Track:
    track_id: int
    category: str
    class_id: int
    detections: list[Detection]

    @property
    def first_frame(self) -> int:
        return self.detections[0].frame_index

    @property
    def last_frame(self) -> int:
        return self.detections[-1].frame_index

    @property
    def length(self) -> int:
        return len(self.detections)

    @property
    def mean_conf(self) -> float:
        return float(np.mean([d.conf for d in self.detections]))

    @property
    def centers(self) -> np.ndarray:
        """(N,2) bbox centers in analyzed-frame pixel coords."""
        return np.array([[(d.bbox[0] + d.bbox[2]) / 2.0,
                          (d.bbox[1] + d.bbox[3]) / 2.0] for d in self.detections])

    @property
    def norm_centers(self) -> np.ndarray:
        return np.array([[(d.norm_bbox[0] + d.norm_bbox[2]) / 2.0,
                          (d.norm_bbox[1] + d.norm_bbox[3]) / 2.0] for d in self.detections])

    @property
    def frames(self) -> list[int]:
        return [d.frame_index for d in self.detections]

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "category": self.category,
            "class_id": self.class_id,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "num_detections": self.length,
            "mean_conf": self.mean_conf,
            "frames": self.frames,
            "norm_centers": self.norm_centers.tolist(),
        }


class DetectTrackPipeline:
    """YOLO + ByteTrack over a VideoReader's frames.

    Parameters mirror common knobs; defaults follow DrivingGen (conf 0.3,
    agent classes).  ``imgsz`` may be lowered for speed on 16GB GPUs.
    """

    def __init__(
        self,
        weights: str = "yolov10x.pt",
        conf: float = 0.3,
        iou: float = 0.5,
        classes: Sequence[int] = DRIVINGGEN_CLASSES,
        tracker: str = "bytetrack.yaml",
        imgsz: int = 640,
        device: Optional[int | str] = 0,
    ) -> None:
        from ultralytics import YOLO  # deferred import: heavy
        self.model = YOLO(weights)
        self.conf = conf
        self.iou = iou
        self.classes = list(classes)
        self.tracker = tracker
        self.imgsz = imgsz
        self.device = device

    def run(self, reader) -> list[Detection]:
        """Run tracking over a wmeval.video_io.VideoReader; returns detections."""
        detections: list[Detection] = []
        h_ref = reader.meta.height
        w_ref = reader.meta.width
        for idx, ts, frame in reader.iter_frames():
            res = self.model.track(
                source=frame,
                persist=True,
                tracker=self.tracker,
                conf=self.conf,
                iou=self.iou,
                classes=self.classes,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
            )
            r = res[0]
            if r.boxes is None:
                continue
            ids = r.boxes.id.tolist() if r.boxes.id is not None else [None] * len(r.boxes)
            H, W = frame.shape[:2]
            self._analyzed_size = (W, H)
            for tid, xyxy, cf, cl in zip(
                ids, r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                r.boxes.cls.cpu().numpy().astype(int),
            ):
                x1, y1, x2, y2 = [float(v) for v in xyxy]
                detections.append(Detection(
                    frame_index=idx,
                    timestamp=ts,
                    track_id=int(tid) if tid is not None else -1,
                    class_id=int(cl),
                    category=COCO_NAMES.get(int(cl), str(cl)),
                    bbox=(x1, y1, x2, y2),
                    conf=float(cf),
                    norm_bbox=(x1 / W, y1 / H, x2 / W, y2 / H),
                ))
        # reference sizes: analyzed (post-resize) vs native container size
        self._native_size = (w_ref, h_ref)
        return detections


def group_tracks(detections: list[Detection]) -> list[Track]:
    """Group detections by track_id; drops unassigned (-1) ids."""
    by_id: dict[int, list[Detection]] = {}
    for d in detections:
        if d.track_id < 0:
            continue
        by_id.setdefault(d.track_id, []).append(d)
    tracks = []
    for tid in sorted(by_id):
        ds = sorted(by_id[tid], key=lambda d: d.frame_index)
        tracks.append(Track(track_id=tid, category=ds[0].category,
                            class_id=ds[0].class_id, detections=ds))
    return tracks


def save_detections_jsonl(detections: list[Detection], path: str | Path) -> None:
    import json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for d in detections:
            f.write(json.dumps(d.to_dict()) + "\n")


def save_tracks_json(tracks: list[Track], path: str | Path) -> None:
    import json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([t.to_dict() for t in tracks], f, indent=2)
