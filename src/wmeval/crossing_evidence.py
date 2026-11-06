"""E2 Phase 5: layered crossing-event evidence (design prototype).

v3.1's ``region_presence=True`` only proves "a person detection center was
once inside a rectangle".  It was never evidence of a *completed crossing*,
yet the verdict logic treated it as the event (E1 v3.1 error analysis
§4.1: 0003/0044 gen judged valid with dx=0.07/0.11, no either-side
evidence — pedestrians standing in the road region).

This module outputs an explicit 4-level evidence hierarchy, evaluated
PER IDENTITY CHAIN (see actor_binding.py), never pooled across actors:

  Level 1  Presence            — does the (bound) actor appear at all?
  Level 2  Region Occupancy    — does its center ever lie in the region?
  Level 3  Crossing Progress   — sustained lateral motion while in region?
  Level 4  Crossing Completion — pre-defined completion condition met?

Level 4 statuses are deliberately non-boolean:
  completed | in_progress | partial_clip_cut | not_progressing |
  interrupted_unexplained | no_occupancy
because a clip cut mid-crossing (common in JAAD-derived clips) or a
mid-clip unexplained vanish cannot be adjudicated as pass/fail.

Known limitation (documented, by design): all motion is measured in
IMAGE space (normalized pixel coordinates).  Image dx is NOT world-space
crossing distance — a distant pedestrian crosses 3 m with dx=0.05 while a
near one crosses 1 m with dx=0.4.  Evidence records this caveat; 3D
lifting (UniDepth) is the E4 path, out of scope here.

This module is NOT wired into v3.1 adjudication — independent evidence
output for v4-experimental and for human review support.
"""
from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np

from .actor_binding import TrackChain, chain_detections
from .detect_track import Track
from .event_checks import _in_region, _norm_time

DEFAULT_REGION = (0.25, 0.55, 0.75, 0.95)


@dataclasses.dataclass
class CrossingEvidence:
    chain: dict                     # chain summary (ids, confidence)
    level1_presence: dict
    level2_region_occupancy: dict
    level3_crossing_progress: dict
    level4_completion: dict

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _centers(dets) -> np.ndarray:
    return np.array([[(d.norm_bbox[0] + d.norm_bbox[2]) / 2.0,
                      (d.norm_bbox[1] + d.norm_bbox[3]) / 2.0]
                     for d in dets])


def evaluate_crossing(
    tracks: list[Track],
    num_frames: int,
    region: tuple[float, float, float, float] = DEFAULT_REGION,
    actor: str = "person",
    chains: Optional[list[TrackChain]] = None,
    progress_dx_min: float = 0.15,
    stationary_dx_max: float = 0.03,
    direction_consistency_min: float = 0.6,
    clip_cut_t: float = 0.90,
) -> list[CrossingEvidence]:
    """Evaluate the 4-level crossing hierarchy per identity chain.

    Thresholds (progress_dx_min etc.) are *reporting* thresholds for the
    evidence layer, not verdict thresholds — v4 tuning must wait for human
    labels (E2 discipline).  Defaults chosen from E1 v3.1 measured facts:
    real-side crossings measured dx 0.35-0.77; standing/generated-drift
    cases measured dx 0.005-0.11.
    """
    if chains is None:
        from .actor_binding import build_track_chains
        chains = build_track_chains(tracks, actor=actor)
    out: list[CrossingEvidence] = []
    for ch in chains:
        dets = [d for d in chain_detections(tracks, ch)
                if d.frame_index is not None]
        ev = CrossingEvidence(
            chain={"chain_id": ch.chain_id, "track_ids": ch.track_ids,
                   "association_confidence": ch.confidence,
                   "links": ch.links},
            level1_presence={}, level2_region_occupancy={},
            level3_crossing_progress={}, level4_completion={})
        # ---- Level 1: presence ------------------------------------------
        ev.level1_presence = {
            "status": bool(dets),
            "n_detections": len(dets),
            "frame_span": [dets[0].frame_index, dets[-1].frame_index]
            if dets else None,
        }
        centers = _centers(dets)
        times = np.array([_norm_time(d.frame_index, num_frames)
                          for d in dets]) if dets else np.zeros(0)
        inside_mask = np.array([_in_region(cx, cy, region)
                                for cx, cy in centers]) if dets else np.zeros(0, bool)
        # ---- Level 2: region occupancy -----------------------------------
        if not dets or not inside_mask.any():
            ev.level2_region_occupancy = {"status": False}
            ev.level3_crossing_progress = {"status": "no_occupancy"}
            ev.level4_completion = {"status": "no_occupancy"}
        else:
            first_in = int(np.argmax(inside_mask))
            last_in = int(len(inside_mask) - 1 - np.argmax(inside_mask[::-1]))
            ev.level2_region_occupancy = {
                "status": True,
                "first_inside_frame": dets[first_in].frame_index,
                "last_inside_frame": dets[last_in].frame_index,
                "frames_inside": int(inside_mask.sum()),
                "occupancy_ratio": round(float(inside_mask.mean()), 3),
            }
            # ---- Level 3: crossing progress (in-region motion) ----------
            seg_c = centers[inside_mask]
            dx_steps = np.diff(seg_c[:, 0])
            net_dx = float(seg_c[-1, 0] - seg_c[0, 0])
            if len(dx_steps):
                pos = float((dx_steps > 1e-4).mean())
                neg = float((dx_steps < -1e-4).mean())
                consistency = max(pos, neg)
                moving_frac = float((np.abs(dx_steps) > 1e-4).mean())
                mean_speed = float(np.mean(np.abs(dx_steps)))
            else:
                consistency, moving_frac, mean_speed = 0.0, 0.0, 0.0
            if net_dx >= progress_dx_min and consistency >= direction_consistency_min:
                l3 = "moving"
            elif net_dx <= stationary_dx_max:
                l3 = "stationary"
            else:
                l3 = "weak_or_ambiguous"
            ev.level3_crossing_progress = {
                "status": l3,
                "net_dx_norm": round(net_dx, 4),
                "direction_consistency": round(consistency, 3),
                "moving_frame_frac": round(moving_frac, 3),
                "mean_speed_norm_per_frame": round(mean_speed, 5),
                "in_region_frames": int(inside_mask.sum()),
            }
            # ---- Level 4: completion ------------------------------------
            entry_x = float(seg_c[0, 0])
            exit_x = float(seg_c[-1, 0])
            entered_from = ("left" if entry_x <= region[0] + 0.02 else
                            "right" if entry_x >= region[2] - 0.02 else "inside")
            exited_to = ("right" if exit_x >= region[2] - 0.02 else
                         "left" if exit_x <= region[0] + 0.02 else "inside")
            last_det = dets[-1]
            last_t = _norm_time(last_det.frame_index, num_frames)
            ends_inside = _in_region(*centers[-1], region)
            l4_ev = {
                "entered_from": entered_from, "exited_to": exited_to,
                "last_frame": last_det.frame_index,
                "last_t_norm": round(last_t, 4),
                "ends_inside_region": bool(ends_inside),
            }
            if l3 != "moving":
                ev.level4_completion = {"status": "not_progressing", **l4_ev}
            elif not ends_inside and entered_from in ("left", "right") \
                    and exited_to in ("left", "right") and exited_to != entered_from:
                ev.level4_completion = {"status": "completed", **l4_ev}
            elif last_t >= clip_cut_t:
                ev.level4_completion = {"status": "partial_clip_cut", **l4_ev}
            else:
                # track ends before clip end while still inside region and
                # no exit evidence — the E1 v3 'vanished mid-event' signature;
                # for THIS module it means completion not decidable from this
                # chain alone (association gap is reported by the chain).
                ev.level4_completion = {
                    "status": "in_progress", **l4_ev,
                    "note": "chain ends inside region before clip end; "
                            "see chain association evidence for gap handling"}
        out.append(ev)
    return out


def summarize(evidences: list[CrossingEvidence]) -> dict:
    """Cross-chain summary — completion is attributed to a SINGLE chain.

    This is the anti-conflation guarantee: no mixing Level-1 from chain A
    with Level-2 from chain B.  Overall status mirrors the best single-chain
    status; evidence lists per-chain outcomes.
    """
    per_chain = [e.to_dict() for e in evidences]
    completed = [e for e in evidences
                 if e.level4_completion.get("status") == "completed"]
    order = {"completed": 0, "in_progress": 1, "partial_clip_cut": 2,
             "not_progressing": 3, "no_occupancy": 4}
    best = None
    best_ev = None
    for e in evidences:
        s = e.level4_completion.get("status", "no_occupancy")
        if best is None or order[s] < order[best]:
            best, best_ev = s, e
    return {
        "per_chain": per_chain,
        "completed_by_single_chain": bool(completed),
        "completed_chain": completed[0].chain["chain_id"] if completed else None,
        "best_single_chain_status": best,
        "best_chain_id": best_ev.chain["chain_id"] if best_ev else None,
        "caveats": [
            "image-space-only: normalized-pixel dx is NOT world-space "
            "crossing distance (perspective scaling); 3D lifting = E4",
            "statuses are evidence, not verdicts; v4 adjudication TBD",
        ],
    }
