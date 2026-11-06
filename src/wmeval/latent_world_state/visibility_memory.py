"""J2.3 G3: Visibility-Aware Memory — reference selection (no training).

Contract (task §六): the memory NEVER fabricates observations.  When
visibility is insufficient the reference feature is NOT updated, but
the record still advances current_timestamp / reference_age /
observation_status — an old reference is evidence of an OLD
observation, never of the actor still being visible.

V0 last-frame | V1 last-RELIABLE-visible | V2 multi-reliable aggregate
| V3 quality-weighted aggregate.
"""
from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np


@dataclasses.dataclass
class ReferenceMemory:
    actor_key: tuple
    reference_feature: Optional[np.ndarray]
    reference_timestamp: Optional[float]
    current_timestamp: Optional[float] = None
    visibility_quality: Optional[str] = None      # visible|partial|unknown
    memory_update_status: str = "initialized"     # updated|skipped|...
    skip_reason: Optional[str] = None
    identity_status: str = "uncertain"            # never auto-confirmed

    @property
    def reference_age_seconds(self) -> Optional[float]:
        if self.reference_timestamp is None or self.current_timestamp is None:
            return None
        return round(self.current_timestamp - self.reference_timestamp, 3)

    def to_record(self) -> dict:
        return {"actor_id": f"{self.actor_key[0]}|{self.actor_key[1]}",
                "reference_timestamp": self.reference_timestamp,
                "current_timestamp": self.current_timestamp,
                "reference_age_seconds": self.reference_age_seconds,
                "reference_update": self.memory_update_status,
                "skip_reason": self.skip_reason,
                "visibility_quality": self.visibility_quality,
                "observation_status": self.visibility_quality or "unknown",
                "identity_status": self.identity_status}


def visibility_class(occ: int) -> str:
    return {0: "visible", 1: "partial", 2: "unknown"}.get(occ, "unknown")


def update_memory(mem: ReferenceMemory, obs: dict,
                  min_quality: str = "visible") -> ReferenceMemory:
    """obs = {timestamp, visibility ('visible'|'partial'|'unknown'),
    feature}.  Update reference ONLY on sufficient visibility; the
    record (current time / age / status) ALWAYS advances."""
    mem.current_timestamp = obs["timestamp"]
    mem.visibility_quality = obs["visibility"]
    rank = {"visible": 2, "partial": 1, "unknown": 0}
    if rank.get(obs["visibility"], 0) >= rank[min_quality]:
        mem.reference_feature = obs["feature"]
        mem.reference_timestamp = obs["timestamp"]
        mem.memory_update_status = "updated"
        mem.skip_reason = None
    else:
        mem.memory_update_status = "skipped"
        mem.skip_reason = f"low_visibility:{obs['visibility']}"
        # reference feature/timestamp stay at the OLD observation —
        # never fabricated as current
    return mem


def select_reference(frames: list, strategy: str,
                      weights=(1.0, 0.7, 0.3)) -> tuple:
    """frames = [(timestamp, occ, feature)] chronological.

    Returns (reference_feature, record_dict).  Strategies:
      V0 last-frame (current behaviour);
      V1 last RELIABLE visible frame (occ==0), fallback last frame;
      V2 mean of visible frames (fallback V1);
      V3 visibility-weighted aggregate of ALL frames.
    The record always states what the reference actually is.
    """
    if not frames:
        raise ValueError("no frames")
    last = frames[-1]
    vis = [f for f in frames if f[1] == 0]
    if strategy == "V0":
        ref, note = last[2], "reference=last_frame"
    elif strategy == "V1":
        base = vis[-1] if vis else last
        ref, note = base[2], ("reference=last_visible" if vis
                              else "fallback=last_frame_no_visible")
    elif strategy == "V2":
        base = vis if vis else [last]
        ref = np.mean([f[2] for f in base], axis=0)
        n = np.linalg.norm(ref)
        ref = ref / n if n > 0 else ref
        note = f"reference=mean_of_{len(base)}_visible"
    elif strategy == "V3":
        w = np.array([weights[min(f[1], 2)] for f in frames], dtype=float)
        w = w / w.sum() if w.sum() > 0 else np.full(len(frames), 1 /
                                                    len(frames))
        ref = np.einsum("i,ij->j", w, np.stack([f[2] for f in frames]))
        n = np.linalg.norm(ref)
        ref = ref / n if n > 0 else ref
        note = "reference=quality_weighted_all_frames"
    else:
        raise ValueError(f"unknown strategy {strategy}")
    return ref, {"strategy": strategy, "note": note,
                 "n_frames": len(frames),
                 "n_visible_frames": len(vis),
                 "last_frame_occ": last[1],
                 "reference_timestamp": (vis[-1][0] if (strategy in
                                                       ("V1", "V2") and vis)
                                         else last[0])}
