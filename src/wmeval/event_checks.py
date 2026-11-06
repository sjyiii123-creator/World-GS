"""C. Event constraint verification (rule-based, evidence-first).

Checks implemented (per research plan §六.C):
  1. presence        — does a given actor (e.g. a pedestrian) appear at all?
  2. region_entry    — does the actor enter a target region (normalized rect)?
  3. time_window     — does the event happen inside a prescribed window?
  4. order           — does a sequence of events happen in the given order?
  5. disappearance   — does any actor vanish without a plausible explanation?

Temporal convention
-------------------
All times are **normalized** t∈[0,1] (frame_index / (num_frames-1)) because
SynPeDAS real/generated pairs differ in fps (30 vs ~24) and length
(150 vs 77 frames).  Raw frame comparison would be a protocol error, not a
model error — this is enforced by design here.

A constraint spec is plain JSON (serializable, no code):

{
  "actor": "person",                    # category filter
  "region": [x1, y1, x2, y2],           # normalized rect, optional
  "t_window": [t0, t1],                 # normalized window, optional
  "order": ["ev_a", "ev_b"]             # optional global ordering
}

Every check returns a CheckResult with machine-readable evidence.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np

from .detect_track import Track


@dataclasses.dataclass
class CheckResult:
    check: str
    event_id: str
    satisfied: Optional[bool]          # None = cannot decide (uncertain)
    evidence: dict

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _norm_time(frame_index: int, num_frames: int) -> float:
    if num_frames <= 1:
        return 0.0
    return frame_index / (num_frames - 1)


def _in_region(cx: float, cy: float, region: tuple[float, float, float, float]) -> bool:
    x1, y1, x2, y2 = region
    return (x1 <= cx <= x2) and (y1 <= cy <= y2)


# --- 1. presence -------------------------------------------------------------
def check_presence(
    tracks: list[Track],
    num_frames: int,
    event_id: str = "presence",
    actor: str = "person",
    region: Optional[tuple[float, float, float, float]] = None,
    t_window: Optional[tuple[float, float]] = None,
) -> CheckResult:
    """True if any track of ``actor`` has a detection inside region+window."""
    candidates = []
    for tr in tracks:
        if tr.category != actor:
            continue
        for d in tr.detections:
            t = _norm_time(d.frame_index, num_frames)
            if t_window and not (t_window[0] <= t <= t_window[1]):
                continue
            cx = (d.norm_bbox[0] + d.norm_bbox[2]) / 2.0
            cy = (d.norm_bbox[1] + d.norm_bbox[3]) / 2.0
            if region and not _in_region(cx, cy, region):
                continue
            candidates.append((tr.track_id, d.frame_index, t))
    if not candidates:
        # distinguish "no actor at all" from "actor only outside region/window"
        any_actor = [tr for tr in tracks if tr.category == actor]
        return CheckResult(
            check="presence", event_id=event_id, satisfied=False,
            evidence={"matches": [], "actor_tracks_anywhere": len(any_actor),
                      "reason": "no matching detection" if not any_actor
                      else "actor present but outside region/window"})
    return CheckResult(
        check="presence", event_id=event_id, satisfied=True,
        evidence={"matches": candidates[:20],
                  "num_matches": len(candidates)})


# --- 2. region entry ---------------------------------------------------------
def check_region_entry(
    tracks: list[Track],
    num_frames: int,
    event_id: str,
    actor: str = "person",
    region: tuple[float, float, float, float] = (0.30, 0.55, 0.70, 0.95),
    t_window: Optional[tuple[float, float]] = None,
    require_from_outside: bool = True,
) -> CheckResult:
    """True if an actor's center crosses into ``region``.

    With ``require_from_outside`` the track must start (or previously be)
    outside the region — mere constant presence inside does not count as an
    entry event (crossing semantics).
    """
    for tr in tracks:
        if tr.category != actor:
            continue
        centers = tr.norm_centers
        times = np.array([_norm_time(f, num_frames) for f in tr.frames])
        inside = np.array([_in_region(cx, cy, region) for cx, cy in centers])
        if not inside.any():
            continue
        entry_idx = int(np.argmax(inside))          # first inside index
        entered_from_outside = (not require_from_outside) or (not inside[0])
        t = float(times[entry_idx])
        in_window = (t_window is None) or (t_window[0] <= t <= t_window[1])
        if entered_from_outside and in_window:
            return CheckResult(
                check="region_entry", event_id=event_id, satisfied=True,
                evidence={"track_id": tr.track_id,
                          "entry_frame": tr.frames[entry_idx],
                          "entry_t_norm": round(t, 4),
                          "region": list(region),
                          "frames_inside": int(inside.sum())})
    return CheckResult(
        check="region_entry", event_id=event_id, satisfied=False,
        evidence={"region": list(region),
                  "reason": "no actor track entered the region"})

# --- 2b. region presence (robust to clips cut mid-event) --------------------
def check_region_presence(
    tracks: list[Track],
    num_frames: int,
    event_id: str,
    actor: str = "person",
    region: tuple[float, float, float, float] = (0.25, 0.55, 0.75, 0.95),
    t_window: Optional[tuple[float, float]] = None,
) -> CheckResult:
    """True if any actor center is ever inside ``region`` (in window).

    Unlike region_entry this does not require entering from outside, so it
    stays correct for clips cut mid-crossing (common in JAAD clips).
    """
    best = None
    for tr in tracks:
        if tr.category != actor:
            continue
        centers = tr.norm_centers
        times = [(_norm_time(f, num_frames), cx, cy)
                 for f, (cx, cy) in zip(tr.frames, centers)]
        for t, cx, cy in times:
            if t_window and not (t_window[0] <= t <= t_window[1]):
                continue
            if _in_region(cx, cy, region):
                if best is None or t < best[1]:
                    best = (tr.track_id, t, f_idx_of(tr, cx, cy))
                break
    if best is None:
        return CheckResult(
            check="region_presence", event_id=event_id, satisfied=False,
            evidence={"region": list(region),
                      "reason": "no actor detection inside region"})
    tid, t, fi = best
    return CheckResult(
        check="region_presence", event_id=event_id, satisfied=True,
        evidence={"track_id": tid, "first_inside_frame": fi,
                  "first_inside_t_norm": round(t, 4),
                  "region": list(region)})


def f_idx_of(tr: Track, cx: float, cy: float) -> int:
    for d, (ncx, ncy) in zip(tr.detections, tr.norm_centers):
        if abs(ncx - cx) < 1e-9 and abs(ncy - cy) < 1e-9:
            return d.frame_index
    return tr.detections[0].frame_index


# --- 3. time window is folded into 1/2 via t_window; standalone helper -------


def check_time_window(
    t_norm: float, window: tuple[float, float], event_id: str,
) -> CheckResult:
    ok = window[0] <= t_norm <= window[1]
    return CheckResult(
        check="time_window", event_id=event_id, satisfied=ok,
        evidence={"t_norm": round(t_norm, 4), "window": list(window)})


# --- 4. event order ----------------------------------------------------------
def check_order(
    event_times: dict[str, Optional[float]], order: list[str],
) -> CheckResult:
    """True iff the events' normalized times are non-decreasing in ``order``
    (missing/undecided events make the check uncertain, not failed)."""
    missing = [e for e in order if event_times.get(e) is None]
    known = [(e, event_times[e]) for e in order if event_times.get(e) is not None]
    # None-propagation semantics: an undecided event ANYWHERE in the sequence
    # makes the *complete* sequence order unverifiable -> satisfied=None.
    # Silently skipping the undecided event would let
    # appears(0.1) -> crosses(None) -> exits(0.8) pass as ordered even though
    # the middle event was never verified (confirmed pre-fix behavior,
    # E2 phase-2 bug report).  NOTE: event_times records None both for
    # "check undecided" and "check decided False" — the two are conflated
    # here; the v4 event-protocol redesign separates them (see
    # docs/event_protocol_design.md).
    if missing or len(known) < 2:
        return CheckResult(check="order", event_id=">".join(order),
                           satisfied=None,
                           evidence={"missing": missing,
                                     "known_partial": [(e, round(t, 4))
                                                       for e, t in known],
                                     "reason": "undecided event(s) in sequence "
                                               "— full order not decidable"})
    # Normalize to the 4-decimal precision used by CheckResult evidence:
    # event_times come from different check types that store time at
    # different precision (presence stores the full float, region_* stores
    # round(t, 4)).  Without this, two events triggered by the *same* frame
    # can compare unequal (E1 v3: 0024 real ped_present=0.7181208 vs
    # ped_in_road=0.7181 -> spurious order failure).  Frame spacing
    # (>= 1/(num_frames-1) >= ~0.0067) is far above 1e-4, so distinct
    # frames never merge.
    times = [round(t, 4) for _, t in known]
    ok = all(times[i] <= times[i + 1] for i in range(len(times) - 1))
    return CheckResult(
        check="order", event_id=">".join(order), satisfied=ok,
        evidence={"sequence": [(e, round(t, 4)) for e, t in known],
                  "missing": missing})


# --- 5. disappearance --------------------------------------------------------
@dataclasses.dataclass
class DisappearanceRecord:
    track_id: int
    category: str
    last_frame: int
    last_t_norm: float
    last_center_norm: tuple[float, float]
    near_edge: bool
    occluder_track_ids: list[int]
    unexplained: bool
    note: str

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def analyze_disappearances(
    tracks: list[Track],
    num_frames: int,
    end_frame_tolerance: int = 3,
    edge_center_margin: float = 0.10,
    edge_bbox_margin: float = 0.02,
    occlusion_iou: float = 0.10,
    min_track_len: int = 5,
) -> list[DisappearanceRecord]:
    """Classify every track that ends before the video ends.

    An ending is *explained* when:
      * the track survives to (nearly) the last frame — video simply ends; or
      * the last bbox center lies within ``edge_center_margin`` of the image
        border **or** a bbox edge lies within ``edge_bbox_margin`` — the actor
        plausibly left the field of view (E1 lesson: center-only margins
        misclassify exiting cars/pedestrians as anomalies); or
      * some other track's bbox overlaps the last bbox (IoU ≥
        ``occlusion_iou``) at the same frame — plausible occlusion.
    Tracks shorter than ``min_track_len`` are skipped (tracking noise;
    DrivingGen's call site uses 2 but defers to a VLM — geometry-only
    adjudication needs longer evidence).
    Otherwise the disappearance is flagged ``unexplained`` (candidate
    anomaly; final adjudication happens in uncertainty.py).
    """
    records: list[DisappearanceRecord] = []
    last_frame_idx = num_frames - 1
    for tr in tracks:
        if tr.last_frame >= last_frame_idx - end_frame_tolerance:
            continue                                    # track survives to the end
        if tr.length < min_track_len:
            continue                                    # too short to adjudicate
        d = tr.detections[-1]
        cx = (d.norm_bbox[0] + d.norm_bbox[2]) / 2.0
        cy = (d.norm_bbox[1] + d.norm_bbox[3]) / 2.0
        near_edge = (cx <= edge_center_margin or cx >= 1 - edge_center_margin
                     or cy <= edge_center_margin or cy >= 1 - edge_center_margin
                     or d.norm_bbox[0] <= edge_bbox_margin
                     or d.norm_bbox[2] >= 1 - edge_bbox_margin
                     or d.norm_bbox[1] <= edge_bbox_margin
                     or d.norm_bbox[3] >= 1 - edge_bbox_margin)
        # occluders: other tracks whose bbox at the same frame overlaps
        occluders: list[int] = []
        for other in tracks:
            if other.track_id == tr.track_id:
                continue
            for od in other.detections:
                if od.frame_index == d.frame_index:
                    iou = _bbox_iou(d.bbox, od.bbox)
                    if iou >= occlusion_iou:
                        occluders.append(other.track_id)
                        break
        unexplained = (not near_edge) and (not occluders)
        if near_edge:
            note = "left field of view (near border)"
        elif occluders:
            note = f"possible occlusion by track(s) {occluders}"
        else:
            note = "no edge/occlusion explanation — candidate anomaly"
        records.append(DisappearanceRecord(
            track_id=tr.track_id, category=tr.category,
            last_frame=tr.last_frame,
            last_t_norm=round(_norm_time(tr.last_frame, num_frames), 4),
            last_center_norm=(round(cx, 4), round(cy, 4)),
            near_edge=near_edge, occluder_track_ids=occluders,
            unexplained=unexplained, note=note))
    return records


def _bbox_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-8)


# --- convenience: full rule battery ------------------------------------------
DEFAULT_CROSSING_SPEC = {
    "ped_present": {
        "check": "presence", "actor": "person",
        "region": [0.05, 0.20, 0.95, 0.92], "t_window": [0.0, 0.9],
    },
    "ped_in_road": {
        "check": "region_presence", "actor": "person",
        "region": [0.25, 0.55, 0.75, 0.95],
    },
    "order": ["ped_present", "ped_in_road"],
}


def run_rule_battery(
    tracks: list[Track],
    num_frames: int,
    spec: Optional[dict] = None,
) -> tuple[list[CheckResult], list[DisappearanceRecord]]:
    """Run the standard rule battery; returns (checks, disappearances)."""
    spec = spec or DEFAULT_CROSSING_SPEC
    results: list[CheckResult] = []
    event_times: dict[str, Optional[float]] = {}
    for eid, c in spec.items():
        if eid == "order" or c.get("check") == "order":
            continue
        if c["check"] == "presence":
            r = check_presence(tracks, num_frames, event_id=eid,
                               actor=c.get("actor", "person"),
                               region=tuple(c["region"]) if "region" in c else None,
                               t_window=tuple(c["t_window"]) if "t_window" in c else None)
        elif c["check"] == "region_presence":
            r = check_region_presence(tracks, num_frames, event_id=eid,
                                      actor=c.get("actor", "person"),
                                      region=tuple(c.get("region", (0.25, 0.55, 0.75, 0.95))),
                                      t_window=tuple(c["t_window"]) if "t_window" in c else None)
        elif c["check"] == "region_entry":
            r = check_region_entry(tracks, num_frames, event_id=eid,
                                   actor=c.get("actor", "person"),
                                   region=tuple(c.get("region", (0.30, 0.55, 0.70, 0.95))),
                                   t_window=tuple(c["t_window"]) if "t_window" in c else None)
        else:
            continue
        results.append(r)
        # first-satisfaction time for ordering
        if r.satisfied and r.check == "region_entry":
            event_times[eid] = r.evidence.get("entry_t_norm")
        elif r.satisfied and r.check == "region_presence":
            event_times[eid] = r.evidence.get("first_inside_t_norm")
        elif r.satisfied:
            matches = r.evidence.get("matches")
            event_times[eid] = matches[0][2] if matches else None
        else:
            event_times[eid] = None
    if "order" in spec:
        r = check_order(event_times, spec["order"])
        results.append(r)
    dis = analyze_disappearances(tracks, num_frames)
    return results, dis
