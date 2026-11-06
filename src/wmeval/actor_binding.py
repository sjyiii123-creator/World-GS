"""E2 Phase 4: event-level Actor identity binding (design prototype).

Problem (E1 v3.1 lesson)
------------------------
Event checks currently pool ALL actor-category tracks, so different
pedestrians can satisfy different checks: pedestrian A satisfies
``presence`` (appears far left, static), pedestrian B satisfies
``region_presence`` (stands/moves inside the road region) — and the system
may conclude "an actor completed the crossing".  This module binds event
evidence to *identity chains* instead.

Approach (geometry-only, no ReID model)
---------------------------------------
Tracks are linked into chains when a candidate successor track of the same
category starts within ``max_gap`` frames of the predecessor's end AND near
the *extrapolated* position (last center + median tail velocity x gap).
ByteTrack id fragmentation (E1 0016: track64 -> track30 -> track64) is
exactly this pattern.

Every link carries numeric evidence (gap, extrapolation error, velocity
ratio) and a confidence label.  Ambiguity — two plausible successors — is
*surfaced, not resolved*: the chain gets confidence=ambiguous and downstream
logic must treat it as uncertain.

This module is a design prototype for v4-experimental.  It is NOT wired
into v3.1 adjudication (``run_minimal`` does not call it); it outputs
independent, inspectable evidence.
"""
from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np

from .detect_track import Track


@dataclasses.dataclass
class TrackChain:
    chain_id: int
    track_ids: list[int]
    category: str
    links: list[dict]              # per-link association evidence
    confidence: str                # high | medium | low | ambiguous
    first_frame: int
    last_frame: int

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _tail_velocity(tr: Track, k: int = 5) -> np.ndarray:
    """Median velocity (norm-units/frame) over the last k centers."""
    c = tr.norm_centers
    if len(c) < 2:
        return np.zeros(2)
    steps = np.diff(c[-(k + 1):], axis=0)
    return np.median(steps, axis=0)


def _candidates_after(tr: Track, others: list[Track], max_gap: int
                      ) -> list[tuple[Track, object]]:
    """Successor candidates for a track that ends at ``tr.last_frame``.

    Semantics (v3.1 redetection case-(b), generalized): a candidate is any
    same-category track with a detection in the window
    ``(last_frame, last_frame + max_gap]`` whose position can continue the
    extrapolated motion, AND that has no detection AT ``last_frame``
    (coexistence guard — a track standing next to the actor is a different
    person, not a continuation).  Candidates do NOT need to be brand-new
    tracks: an *existing* track whose detections alternate with ``tr``
    (ByteTrack id fragmentation, E1 0016: track64 f45-54 -> track30
    f55-62 -> track64 f63-92) is exactly the pattern to link.
    """
    out = []
    for o in others:
        if tr.last_frame in o.frames:
            continue                    # coexisting at the end frame
        soon = [(f, d) for f, d in zip(o.frames, o.detections)
                if tr.last_frame < f <= tr.last_frame + max_gap]
        if soon:
            out.append((o, soon[0][1]))
    return out


def _center_of(d) -> np.ndarray:
    return np.array([(d.norm_bbox[0] + d.norm_bbox[2]) / 2.0,
                     (d.norm_bbox[1] + d.norm_bbox[3]) / 2.0])


def build_track_chains(
    tracks: list[Track],
    max_gap: int = 15,
    max_pred_err: float = 0.15,
    ambiguity_ratio: float = 1.5,
    actor: Optional[str] = None,
) -> list[TrackChain]:
    """Link same-category tracks into identity chains.

    Parameters
    ----------
    max_gap : a candidate's continuation detection must appear within this
        many frames after the predecessor's last detection (matches
        uncertainty.py reappear window).  Candidates may be brand-new
        tracks OR existing tracks that re-appear at the vanish point
        (alternating id fragmentation).
    max_pred_err : the continuation detection's center must lie within this
        normalized distance of the extrapolated position.
    ambiguity_ratio : if the 2nd-best candidate's extrapolation error is
        < ratio x best error, the link (and chain) is ``ambiguous`` —
        two plausible continuations exist and identity is not decidable.

    Returns chains ordered by first_frame; isolated tracks become
    single-track chains (confidence high — nothing to associate).

    Known limitation (documented): greedy forward growth is
    seed-order-dependent; a globally optimal association (e.g. DP over
    candidate graph) is deferred to v4 if needed.
    """
    pool = [t for t in tracks if actor is None or t.category == actor]
    by_cat: dict[str, list[Track]] = {}
    for t in pool:
        by_cat.setdefault(t.category, []).append(t)

    chains: list[TrackChain] = []
    for cat, cat_tracks in by_cat.items():
        remaining = {t.track_id: t for t in cat_tracks}
        # deterministic processing order
        ordered = sorted(cat_tracks, key=lambda t: (t.first_frame, t.track_id))
        used_as_successor: set[int] = set()
        for seed in ordered:
            if seed.track_id in used_as_successor and seed.track_id not in remaining:
                continue
            if seed.track_id not in remaining:
                continue            # already consumed as a successor
            # grow a chain forward from `seed`
            member = seed
            chain_ids = [member.track_id]
            links: list[dict] = []
            confidences = []
            while True:
                remaining.pop(member.track_id, None)
                cands = _candidates_after(
                    member, list(remaining.values()), max_gap)
                if not cands:
                    break
                v = _tail_velocity(member)
                last_c = member.norm_centers[-1]
                errs = []
                for cand_track, first_det in cands:
                    gap = first_det.frame_index - member.last_frame
                    pred = last_c + v * gap
                    oc = _center_of(first_det)
                    errs.append((float(np.linalg.norm(oc - pred)),
                                 cand_track, gap, v))
                errs.sort(key=lambda e: e[0])
                best_err, best, gap, v = errs[0]
                if best_err > max_pred_err:
                    break            # no plausible successor
                link = {
                    "from_track": member.track_id,
                    "to_track": best.track_id,
                    "gap_frames": gap,
                    "pred_err_norm": round(best_err, 4),
                    "velocity_norm_per_frame": [round(float(v[0]), 4),
                                                round(float(v[1]), 4)],
                }
                # ambiguity only when BOTH candidates are individually
                # plausible (2nd err within the gate) AND nearly equally
                # good relative to the best (ratio test, no floor: a
                # near-perfect best with a merely-okay 2nd is NOT ambiguous)
                if (len(errs) > 1 and errs[1][0] <= max_pred_err
                        and errs[1][0] < ambiguity_ratio * best_err):
                    link["ambiguous_with"] = errs[1][1].track_id
                    link["second_err_norm"] = round(errs[1][0], 4)
                    confidences.append("ambiguous")
                else:
                    confidences.append(
                        "high" if best_err < 0.5 * max_pred_err else "medium")
                links.append(link)
                chain_ids.append(best.track_id)
                member = best
            if "ambiguous" in confidences:
                conf = "ambiguous"
            elif confidences and "low" in confidences:
                conf = "low"
            elif not confidences:
                conf = "high"        # single track, no association needed
            else:
                conf = "medium" if "medium" in confidences else "high"
            member_tracks = [t for t in cat_tracks if t.track_id in chain_ids]
            for t in member_tracks:
                used_as_successor.add(t.track_id)
                remaining.pop(t.track_id, None)
            chains.append(TrackChain(
                chain_id=len(chains), track_ids=sorted(chain_ids),
                category=cat, links=links, confidence=conf,
                first_frame=min(t.first_frame for t in member_tracks),
                last_frame=max(t.last_frame for t in member_tracks)))
    chains.sort(key=lambda c: (c.first_frame, c.chain_id))
    for i, c in enumerate(chains):
        c.chain_id = i
    return chains


def chain_detections(tracks: list[Track], chain: TrackChain) -> list:
    """All detections of a chain, ordered by frame (across member tracks)."""
    by_id = {t.track_id: t for t in tracks}
    dets = []
    for tid in chain.track_ids:
        dets.extend(by_id[tid].detections)
    dets.sort(key=lambda d: d.frame_index)
    return dets
