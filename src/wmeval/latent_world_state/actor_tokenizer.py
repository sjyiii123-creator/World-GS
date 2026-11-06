"""Actor tokenizer: organize one actor's causal observation history.

Holds ONLY past-facing fields (frames/timestamps/bboxes/visibility/
features at frames <= t).  Identity is (source_video_id, actor_id) —
the same numeric actor id in a different video is a different person.
"""
from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np


@dataclasses.dataclass
class ActorObservation:
    source_video_id: str
    actor_id: str
    pair_id: str
    frames: list                    # xml frames, ascending, <= anchor end
    timestamps: list                # real seconds (frame / clip fps)
    bboxes: list                    # xyxy px
    visibilities: list              # JAAD occluded 0/1/2
    features: Optional[np.ndarray]  # (T, D) frozen embeddings, causal
    split: str = ""

    @property
    def identity_key(self) -> tuple:
        return (self.source_video_id, self.actor_id)

    @property
    def anchor_end_time_s(self) -> Optional[float]:
        return self.timestamps[-1] if self.timestamps else None

    def causal_violations(self) -> int:
        """Frames must be ascending (a strictly causal history)."""
        return sum(1 for a, b in zip(self.frames[:-1], self.frames[1:])
                   if b <= a)


@dataclasses.dataclass
class CandidateObservation:
    """One association candidate: its FIRST valid observation."""
    source_video_id: str
    actor_id: str
    first_frame: int
    first_time_s: float             # real seconds of that observation
    bbox: tuple
    visibility: int
    feature: Optional[np.ndarray] = None
    is_true_match: Optional[bool] = None   # evaluation-time only

    @property
    def identity_key(self) -> tuple:
        return (self.source_video_id, self.actor_id)
