"""D. Uncertainty-aware adjudication: Valid / Invalid / Uncertain.

Separates four evidence types (research plan §六.D):
  1. detection failure   — the detector/tracker is the weak link
  2. occlusion           — the actor is plausibly hidden
  3. identity switch     — the track broke, the actor likely persists
  4. genuine generation anomaly — none of the above explains the evidence

Every verdict carries a machine-checkable evidence dict; nothing is reported
without a reason.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np

from .detect_track import Track
from .event_checks import CheckResult, DisappearanceRecord

VALID, INVALID, UNCERTAIN = "valid", "invalid", "uncertain"


@dataclasses.dataclass
class Verdict:
    verdict: str
    reasons: list[str]
    evidence: dict

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# track-ending attribution
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class EndingAttribution:
    track_id: int
    attribution: str          # occlusion | redetection | id_switch |
                              # detection_quality | exited_frame | anomaly
    confidence_note: str
    evidence: dict
    category: str = ""        # track category, for actor-aware adjudication

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def attribute_endings(
    tracks: list[Track],
    disappearances: list[DisappearanceRecord],
    *,
    reappear_frame_gap: int = 15,
    reappear_dist_norm: float = 0.15,
    id_switch_speed_jump: float = 4.0,
    low_conf: float = 0.45,
    noise_len: int = 10,
    noise_conf: float = 0.60,
) -> list[EndingAttribution]:
    """Attribute each disappearance to one of the four evidence types.

    Heuristics (all thresholds explicit, all recorded in evidence):
    * ``clip_end_cut``     — track ends in the last 10 % of the video: the
                             source clip (or generated chunk) simply ends;
                             cannot be adjudicated → weak evidence (E1
                             lesson: JAAD clips are cut mid-crossing);
    * ``occlusion``        — another track overlapped the last bbox;
    * ``redetection``      — another same-category track covers the vanish
                             point within ``reappear_frame_gap`` frames and
                             ``reappear_dist_norm`` (normalized) of the end
                             point: either a brand-new track (tracker drop &
                             re-acquire) or an existing fragmented track re-
                             appearing there (id fragmentation / interleaved
                             detections) — actor likely persisted;
    * ``id_switch``        — speed of the track jumps by ≥
                             ``id_switch_speed_jump``× its median just before
                             the end (classic association break signature);
    * ``detection_quality``— track mean confidence < ``low_conf``, or a
                             short noisy track (``len < noise_len`` and
                             ``mean_conf < noise_conf``) (E1 lesson: 6-frame
                             conf-0.5 tracks are noise, not anomalies);
    * ``exited_frame``     — end near the border (explained, no anomaly);
    * ``anomaly``          — none of the above: genuine candidate anomaly.
    """
    by_id = {t.track_id: t for t in tracks}
    out: list[EndingAttribution] = []
    for rec in disappearances:
        tr = by_id.get(rec.track_id)
        ev: dict = rec.to_dict()
        if tr is None:
            out.append(EndingAttribution(rec.track_id, "anomaly",
                         "track object missing", ev,
                         category=""))
            continue
        ev["mean_conf"] = round(tr.mean_conf, 3)
        ev["length"] = tr.length

        if rec.last_t_norm >= 0.90:
            out.append(EndingAttribution(
                rec.track_id, "clip_end_cut",
                "track ends in the last 10% of the video — clip-cut, "
                "not adjudicable", ev, category=rec.category))
            continue
        if rec.near_edge:
            out.append(EndingAttribution(
                rec.track_id, "exited_frame",
                "last bbox near image border — actor plausibly left view", ev,
                category=rec.category))
            continue
        if rec.occluder_track_ids:
            ev["occluders"] = rec.occluder_track_ids
            out.append(EndingAttribution(
                rec.track_id, "occlusion",
                "overlap with another track at last frame", ev,
                category=rec.category))
            continue

        # redetection: another same-category track covering the vanish point
        # soon after.  Two shapes:
        #   (a) a *brand-new* track starting within the window (classic
        #       tracker drop & re-acquire) — first_frame > last_frame;
        #   (b) an *existing* track whose detections interleave with this
        #       one (id fragmentation): its first_frame may precede
        #       last_frame, it has a detection gap across last_frame, and
        #       it re-appears at the vanish point within the window.
        #       E1 v3 lesson (0016 gen): track30 f12-62 and track64
        #       f45-54/f63-92 are one pedestrian with two alternating ids;
        #       case-(a)-only logic misread the handover as a vanishing
        #       anomaly -> false INVALID.
        # Excluded: tracks that have a detection *at* last_frame itself
        # (a coexisting actor standing near the point — different story).
        last_c = np.array(rec.last_center_norm)
        reapp: Optional[int] = None
        for other in tracks:
            if other.track_id == tr.track_id or other.category != tr.category:
                continue
            gap = other.first_frame - rec.last_frame
            if 0 < gap <= reappear_frame_gap:
                oc = other.norm_centers[0]
                if float(np.linalg.norm(oc - last_c)) <= reappear_dist_norm:
                    reapp = other.track_id
                    break
            soon = [(f, c) for f, c in zip(other.frames, other.norm_centers)
                    if rec.last_frame < f <= rec.last_frame + reappear_frame_gap]
            present_at_end = any(f == rec.last_frame for f in other.frames)
            if soon and not present_at_end:
                f0, c0 = soon[0]
                if float(np.linalg.norm(np.array(c0) - last_c)) \
                        <= reappear_dist_norm:
                    reapp = other.track_id
                    ev["reappear_frame"] = f0
                    break
        if reapp is not None:
            ev["reappearing_track"] = reapp
            out.append(EndingAttribution(
                rec.track_id, "redetection",
                "same-category track re-acquired nearby soon after — "
                "tracker drop likely, actor probably persisted", ev,
                category=rec.category))
            continue

        # id switch: speed spike before the end
        centers = tr.centers
        if len(centers) >= 6:
            steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
            med = float(np.median(steps))
            tail = steps[-3:]
            if med > 1e-6 and float(tail.max()) > id_switch_speed_jump * med:
                ev["median_step_px"] = round(med, 2)
                ev["tail_max_step_px"] = round(float(tail.max()), 2)
                out.append(EndingAttribution(
                    rec.track_id, "id_switch",
                    "discontinuous motion right before track end — "
                    "association break signature", ev,
                    category=rec.category))
                continue

        if tr.mean_conf < low_conf or (tr.length < noise_len
                                       and tr.mean_conf < noise_conf):
            why = (f"mean detection confidence {tr.mean_conf:.2f} < {low_conf}"
                   if tr.mean_conf < low_conf else
                   f"short noisy track (len {tr.length} < {noise_len}, "
                   f"conf {tr.mean_conf:.2f} < {noise_conf})")
            out.append(EndingAttribution(
                rec.track_id, "detection_quality", why, ev,
                category=rec.category))
            continue

        out.append(EndingAttribution(
            rec.track_id, "anomaly",
            "no edge/occlusion/redetection/id-switch/quality explanation — "
            "genuine generation anomaly candidate", ev,
            category=rec.category))
    return out


# ---------------------------------------------------------------------------
# whole-event verdict
# ---------------------------------------------------------------------------
def detection_quality_proxy(detections: list, num_frames: int) -> dict:
    """Cheap proxy for 'was the detector working at all in this video'."""
    per_frame_counts: dict[int, int] = {}
    person_frames = 0
    for d in detections:
        per_frame_counts[d.frame_index] = per_frame_counts.get(d.frame_index, 0) + 1
        if d.category == "person":
            person_frames += 1
    frames_with_any = len(per_frame_counts)
    return {
        "frames_with_detections": frames_with_any,
        "num_frames": num_frames,
        "coverage": round(frames_with_any / max(num_frames, 1), 3),
        "person_detections": person_frames,
        "mean_dets_per_active_frame": round(
            sum(per_frame_counts.values()) / max(frames_with_any, 1), 2),
    }


def _inside_any_region(center: tuple[float, float],
                        regions: list[tuple[float, float, float, float]]) -> bool:
    x, y = center
    return any(r[0] <= x <= r[2] and r[1] <= y <= r[3] for r in regions)


def adjudicate(
    checks: list[CheckResult],
    endings: list[EndingAttribution],
    detections: list,
    num_frames: int,
    min_coverage: float = 0.5,
    actor: str = "person",
    interrupt_regions: Optional[list[tuple[float, float, float, float]]] = None,
) -> Verdict:
    """Event-level verdict, v2 policy (tuned on E1 real-video evidence).

    INVALID (strong evidence only):
      * presence check failed while the detector demonstrably works — the
        event actor is absent from the whole video; or
      * an ``anomaly`` ending of the event **actor** that interrupts an
        in-progress event — the actor vanished inside an ``interrupt_region``
        (e.g. mid-road) rather than leaving the frame/being occluded.
    UNCERTAIN (evidence tainted):
      * actor endings on weak evidence (occlusion/redetection/id-switch/
        low confidence); actor anomaly endings *outside* event regions;
        failed non-presence checks; undecided order; weak detector.
    VALID: every check passed, no actor anomaly/weak ending.

    Non-actor anomalies (background cars vanishing, etc.) do NOT change the
    event verdict — they are returned as ``agent_flags`` (the
    DrivingGen-style agent-missing signal, reported separately) because E1
    showed geometry-only adjudication of those has an unacceptable false
    positive rate on real videos.
    """
    dq = detection_quality_proxy(detections, num_frames)
    reasons: list[str] = []
    failed = [r for r in checks if r.satisfied is False]
    undecided = [r for r in checks if r.satisfied is None]
    interrupt_regions = interrupt_regions or []

    anomalies = [e for e in endings if e.attribution == "anomaly"]
    actor_anoms = [e for e in anomalies if e.category == actor]
    other_anoms = [e for e in anomalies if e.category != actor]
    weak_actor = [e for e in endings
                  if e.attribution in ("occlusion", "redetection",
                                       "id_switch", "detection_quality",
                                       "clip_end_cut")
                  and e.category == actor]

    detector_ok = dq["coverage"] >= min_coverage and dq["mean_dets_per_active_frame"] >= 1.0

    # --- invalid branch -------------------------------------------------
    if not detector_ok:
        return Verdict(UNCERTAIN, ["detector coverage too low to adjudicate"],
                       {"detection_quality": dq, "agent_flags":
                        [e.to_dict() for e in other_anoms]})
    actor_missing = any(
        r.check == "presence" and r.satisfied is False and
        r.evidence.get("actor_tracks_anywhere", 0) == 0 for r in failed)
    if actor_missing:
        reasons.append(f"presence check failed while detector was active — "
                       f"no {actor} found anywhere in the video")
        return Verdict(INVALID, reasons, {"detection_quality": dq,
                                          "failed_checks": [r.event_id for r in failed],
                                          "agent_flags": [e.to_dict() for e in other_anoms]})
    interrupting = [e for e in actor_anoms
                    if _inside_any_region(e.evidence.get("last_center_norm",
                                                         (0.5, 0.5)),
                                          interrupt_regions)]
    if interrupting:
        reasons.append(f"actor ({actor}) vanished inside event region(s) "
                       f"{interrupt_regions} — in-progress event interrupted "
                       f"(tracks {[e.track_id for e in interrupting]})")
        return Verdict(INVALID, reasons,
                       {"interrupting_anomalies": [e.to_dict() for e in interrupting],
                        "detection_quality": dq,
                        "agent_flags": [e.to_dict() for e in other_anoms]})

    # --- uncertain branch ------------------------------------------------
    if failed:
        reasons.append(f"failed checks {[r.event_id for r in failed]}")
    if undecided:
        reasons.append(f"undecided checks {[r.event_id for r in undecided]}")
    if weak_actor:
        reasons.append(f"actor ({actor}) endings on weak evidence: "
                       f"{[(e.track_id, e.attribution) for e in weak_actor]}")
    non_interrupting = [e for e in actor_anoms if e not in interrupting]
    if non_interrupting:
        reasons.append(f"actor anomaly endings outside event regions: "
                       f"{[(e.track_id, e.evidence.get('last_center_norm')) for e in non_interrupting]}")
    if dq["person_detections"] == 0:
        reasons.append("no person detections at all")
    if reasons:
        return Verdict(UNCERTAIN, reasons,
                       {"detection_quality": dq,
                        "ending_attributions": [e.to_dict() for e in endings],
                        "agent_flags": [e.to_dict() for e in other_anoms]})

    # --- valid branch ----------------------------------------------------
    return Verdict(VALID, ["all checks passed; no actor anomalies or "
                           "weak-evidence actor endings"],
                   {"detection_quality": dq,
                    "ending_attributions": [e.to_dict() for e in endings],
                    "agent_flags": [e.to_dict() for e in other_anoms]})
