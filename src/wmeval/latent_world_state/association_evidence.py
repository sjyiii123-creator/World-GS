"""Association evidence: identity-continuation evidence per candidate.

Pure computation + data classes (no thresholds baked in — thresholds
belong to calibrated callers).  Includes the J2.1 fixes:
  * identity relation = (source_video_id, actor_id);
  * per-candidate delta = candidate_first_time_s - anchor_last_time_s;
  * unusable intervals -> evidence 'unknown' (never guessed);
  * ambiguity surfaced when top-2 evidence scores are close.
"""
from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np


def build_identity_mask(source_ids, actor_ids) -> np.ndarray:
    """same[i,j] iff (source,actor) equal — repeated actors across
    sequences are SAME; identical actor ids in different videos are
    DIFFERENT pedestrians."""
    n = len(source_ids)
    keys = [(s, a) for s, a in zip(source_ids, actor_ids)]
    m = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            m[i, j] = keys[i] == keys[j]
    return m


def candidate_delta(candidate: dict, anchor_last_time_s: float):
    """delta = candidate's OWN first-time minus anchor end.

    Returns None when the interval is invalid (<=0 or missing) —
    the caller must treat that candidate's prediction evidence as
    unknown, never substitute another candidate's time."""
    t = candidate.get("first_time_s")
    if t is None or anchor_last_time_s is None:
        return None
    d = float(t) - float(anchor_last_time_s)
    return d if d > 0 and np.isfinite(d) else None


def evidence_quality(delta: Optional[float]) -> str:
    if delta is None:
        return "unknown"
    return "usable" if np.isfinite(delta) and delta > 0 else "unknown"


def valid_kinematic_error(err) -> bool:
    """J2.2 fix: explicit validity — None / NaN / Inf / negative are
    ALL invalid (the old `err or 9.9` truthiness idiom silently
    replaced a legitimate ZERO error with 9.9)."""
    if err is None:
        return False
    try:
        v = float(err)
    except (TypeError, ValueError):
        return False
    return np.isfinite(v) and v >= 0.0


@dataclasses.dataclass
class CandidateEvidence:
    identity_key: tuple
    kinematic_error: Optional[float]      # normalized extrapolation error
    static_similarity: Optional[float]    # anchor last frame vs candidate
    predicted_similarity: Optional[float] # predictor(hist, own delta) vs
                                          # candidate feature — None when
                                          # time evidence unknown
    candidate_delta_s: Optional[float]
    evidence_state: str                   # usable | unknown
    is_true_match: Optional[bool] = None  # evaluation only

    def kinematic_usable(self) -> bool:
        return valid_kinematic_error(self.kinematic_error)


@dataclasses.dataclass
class AssociationDecision:
    chosen_identity: Optional[tuple]       # COMMITTED only on 'accept'
    decision: str                          # accept|reject|uncertain|no_match
    ambiguity: Optional[float]             # top-2 score gap (None=sole cand)
    reasons: list
    tendency_identity: Optional[tuple] = None  # ranking inclination ONLY —
    # J2.3 fix: non-accept decisions must never implicitly commit an actor
    # identity; the ranking tendency is exposed separately for audit.


def rank_candidates(
    evidences: list,
    score=None,
    ambiguity_margin: float = 0.05,        # J2.2: configurable, not baked in
    accept_fn=None,
) -> AssociationDecision:
    """Kinematic ranking stays the MAIN ranker (J3-lite lesson).

    J2.2 boundary contract (regression-tested):
      * kinematic_error validity is checked EXPLICITLY (None/NaN/Inf/
        negative are invalid; zero is valid);
      * candidates with invalid kinematics rank LAST and cannot be
        auto-accepted;
      * without a calibrated ``accept_fn`` the strongest possible
        decision is ``uncertain`` — a sole candidate is NEVER
        auto-accepted on missing evidence;
      * sole candidate with valid evidence and accept_fn -> the fn
        decides; without fn -> uncertain (identity never implicitly
        committed);
      * top-2 score gap below ``ambiguity_margin`` -> uncertain;
      * no valid-evidence candidate at all -> no_match.
    """
    def default_score(e):
        if not e.kinematic_usable():
            return -1e9                     # invalid evidence ranks last
        return -float(e.kinematic_error)    # zero error ranks first (fix)

    score = score or default_score
    if not evidences:
        return AssociationDecision(
            None, "no_match", None, ["no candidates"])
    usable = [e for e in evidences if e.kinematic_usable()]
    if not usable:
        return AssociationDecision(
            None, "no_match", None,
            ["no candidate has valid kinematic evidence"])
    ranked = sorted(evidences, key=score, reverse=True)
    top = ranked[0]
    top_valid = top.kinematic_usable()
    gap = None
    if len(ranked) > 1:
        s1, s2 = score(top), score(ranked[1])
        gap = round(float(s1 - s2), 6)
    reasons = []
    if not top_valid:
        return AssociationDecision(
            None, "uncertain", gap,
            ["top-ranked candidate lacks valid kinematic evidence"])
    if len(ranked) > 1 and gap is not None and gap < ambiguity_margin:
        return AssociationDecision(
            None, "uncertain", gap,
            ["top-2 evidence gap below ambiguity margin "
             f"({gap} < {ambiguity_margin})"],
            tendency_identity=top.identity_key)
    if accept_fn is None:
        return AssociationDecision(
            None, "uncertain", gap,
            ["no calibrated accept threshold supplied — safe default"],
            tendency_identity=top.identity_key)
    if accept_fn(top):
        return AssociationDecision(
            top.identity_key, "accept", gap,
            ["top-1 with valid evidence passed calibrated accept rule"])
    return AssociationDecision(
        None, "reject", gap,
        ["top-1 failed calibrated accept rule"],
        tendency_identity=top.identity_key)
