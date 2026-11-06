"""E2 direction audit: pixel evidence != semantic direction != compliance.

Correction context (external independent review, 2026-11-05)
------------------------------------------------------------
The first E2 direction audit conflated five distinct layers:
  (1) image-coordinate motion (dx sign in normalized pixel space);
  (2) pedestrian world-space motion semantics (which needs 3D/ego info);
  (3) what the PROMPT literally requires;
  (4) what the SOURCE video actually shows;
  (5) whether the GENERATOR violated its conditions.
Confirmed corrections:
  * 0044 is a PROMPT-SOURCE CONFLICT (prompt "left to right" vs source
    image right-to-left) — NOT a clean model-following failure;
  * 0035 has opposite pixel dx signs, but the prompt says "towards the
    camera" (no lateral claim) and BOTH real/gen bbox heights grow
    (approach evidence) — semantic reversal is UNPROVEN -> unknown;
  * "from the right" = starting side, NOT travel direction;
  * the max-detections track is not necessarily the prompted actor;
  * videos with different fps/length must never be compared by raw
    frame index (use t_norm phase alignment only);
  * a prompt with no pedestrian condition cannot yield a
    crossing-absence violation (0024).

This module is EVIDENCE ONLY: it classifies into the five audit
categories and marks everything semantic as unknown unless strong
evidence exists.  It never touches raw prompts / videos / v3.1 outputs
(read-only inputs).
"""
from __future__ import annotations

import dataclasses
import hashlib
import re
from typing import Optional

# ---------------------------------------------------------------------------
# 1. prompt direction semantics (text parsing — the layer most easily confused)
# ---------------------------------------------------------------------------
_TRAVEL_LR = re.compile(
    r"from (?:the )?left to (?:the )?right|left[- ]to[- ]right|"
    r"left to right", re.I)
_TRAVEL_RL = re.compile(
    r"from (?:the )?right to (?:the )?left|right[- ]to[- ]left|"
    r"right to left", re.I)
_FROM_RIGHT = re.compile(
    r"(?:walks?|steps?|comes?|approach(?:es|ing)?|arriv(?:es|ing)?|"
    r"moving|emerg(?:es|ing)?) (?:briskly |steadily |out )?from the right\b|"
    r"from the right side", re.I)
_FROM_LEFT = re.compile(
    r"(?:walks?|steps?|comes?|approach(?:es|ing)?|arriv(?:es|ing)?|"
    r"moving|emerg(?:es|ing)?) (?:briskly |steadily |out )?from the left\b|"
    r"from the left side", re.I)
_TOWARD_CAM = re.compile(
    r"towards? the camera|approach(?:es|ing)? (?:the )?(?:camera|viewer)|"
    r"walks? .* towards? (?:the )?camera|advanc(?:es|ing) toward", re.I)
_AWAY_CAM = re.compile(r"away from the camera|reced(?:es|ing)", re.I)
_PED_MENTION = re.compile(
    r"pedestrian|person|people|walker|figure|individual|commuter|"
    r"shopper|child|children", re.I)


def prompt_direction_semantics(text: str, prompt_available: bool = True) -> dict:
    """Parse the literal direction semantics of a prompt.

    P0-5 rule: a MISSING/unreadable/empty prompt is `unknown_prompt_
    unavailable` — it must NEVER be treated as "no event required"
    (absence of text is not evidence of absence of a condition; only a
    fully-read prompt that mentions no pedestrian yields none_required).

    Returns direction ∈
      left_to_right | right_to_left      # explicit TRAVEL direction
      from_right | from_left             # STARTING side only (NOT travel)
      toward_camera | away_camera        # depth motion, no lateral claim
      unspecified                         # pedestrians present, no direction
      none_required                       # full text read, no pedestrian
      unknown_prompt_unavailable          # prompt missing/empty/unreadable
    """
    if not prompt_available or text is None or not text.strip():
        return {"direction": "unknown_prompt_unavailable",
                "semantic_type": "unavailable",
                "matched_phrases": [],
                "pedestrian_condition": "unknown",
                "note": "prompt unavailable — condition unknown; "
                        "NOT 'no event required'"}
    if not _PED_MENTION.search(text):
        return {"direction": "none_required",
                "semantic_type": "none",
                "matched_phrases": [],
                "pedestrian_condition": "absent_from_prompt",
                "note": "no pedestrian mentioned — crossing absence "
                        "cannot be scored as a violation"}
    phrases = []
    if _TRAVEL_LR.search(text):
        direction, tag = "left_to_right", "travel_direction"
        phrases.append(_TRAVEL_LR.search(text).group(0))
    elif _TRAVEL_RL.search(text):
        direction, tag = "right_to_left", "travel_direction"
        phrases.append(_TRAVEL_RL.search(text).group(0))
    elif _FROM_RIGHT.search(text):
        direction, tag = "from_right", "starting_side"
        phrases.append(_FROM_RIGHT.search(text).group(0))
    elif _FROM_LEFT.search(text):
        direction, tag = "from_left", "starting_side"
        phrases.append(_FROM_LEFT.search(text).group(0))
    elif _TOWARD_CAM.search(text):
        direction, tag = "toward_camera", "depth_motion"
        phrases.append(_TOWARD_CAM.search(text).group(0))
    elif _AWAY_CAM.search(text):
        direction, tag = "away_camera", "depth_motion"
        phrases.append(_AWAY_CAM.search(text).group(0))
    else:
        direction, tag = "unspecified", "none"
    return {"direction": direction, "semantic_type": tag,
            "matched_phrases": phrases,
            "pedestrian_condition": "explicit"}


# ---------------------------------------------------------------------------
# 2. pixel-space evidence (observation only)
# ---------------------------------------------------------------------------
def pixel_direction(dx_norm: float, eps: float = 0.02) -> str:
    if dx_norm > eps:
        return "image_right"
    if dx_norm < -eps:
        return "image_left"
    return "approx_static"


def bbox_height_ratio(frame_heights, frac: float = 0.15):
    """First/last-in-TIME median bbox height and their ratio.

    P0-1 FIX (E2.1): the previous implementation sorted the height VALUES
    (``sorted(hs)[:n]``) — measuring the height distribution's extremes,
    NOT temporal change.  A triangular profile (approach then retreat)
    was misread as 'growing' (ratio ~8) instead of ~stable.

    Correct semantics: order (frame_index, height) pairs by frame_index,
    take the median of the first `frac` (earliest) and last `frac`
    (latest) segments in TIME.  Returns
    (h_first_median, h_last_median, ratio) or (None, None, None) when
    fewer than 2 samples.  Ratio>1 = bbox grew over time (weak approach
    evidence; NEVER direct 3D distance — perspective/camera also scale).
    """
    vals = sorted(((f, h) for f, h in frame_heights), key=lambda x: x[0])
    if len(vals) < 2:
        return None, None, None
    n = max(1, round(len(vals) * frac))
    import statistics
    h_first = statistics.median(h for _, h in vals[:n])
    h_last = statistics.median(h for _, h in vals[-n:])
    ratio = round(h_last / h_first, 3) if h_first > 0 else None
    return round(h_first, 4), round(h_last, 4), ratio


def scale_trend(height_ratio: Optional[float],
                grow: float = 1.5, shrink: float = 0.67) -> str:
    """bbox height ratio (last/first) as APPROACH/RECEDE evidence (weak)."""
    if height_ratio is None:
        return "unknown"
    if height_ratio >= grow:
        return "growing"
    if height_ratio <= shrink:
        return "shrinking"
    return "stable"


def frame_index_comparable(meta_a: dict, meta_b: dict) -> bool:
    """Raw frame indices are comparable ONLY if fps AND length match.

    SynPeDAS pairs (150@29.97 vs 93@16) are NOT comparable — cross-video
    comparison must go through t_norm = frame/(num_frames-1) phase
    alignment, and even that is phase alignment, not physical time.
    """
    return (meta_a.get("fps") == meta_b.get("fps")
            and meta_a.get("num_frames") == meta_b.get("num_frames"))


# ---------------------------------------------------------------------------
# 3. five-category audit assessment (machine-decidable part only)
# ---------------------------------------------------------------------------
CATEGORIES = ("prompt_source_conflict",          # A (candidate or confirmed)
              "image_motion_difference",         # B
              "semantic_inconsistency_established",  # C (needs strong proof)
              "insufficient_evidence",           # D
              "no_direction_issue_observed")     # E (needs actor confirmation)


@dataclasses.dataclass
class SideEvidence:
    dx_norm: Optional[float]
    pixel_dir: str
    height_ratio: Optional[float]
    scale: str
    n_person_tracks: int
    track_selection_basis: str       # e.g. max_detections (NOT actor identity)
    fps: float
    num_frames: int


@dataclasses.dataclass
class DirectionAssessment:
    """E2.1 v2 assessment — candidate/confirmed split + two-level agreement.

    P0-2: an automatic travel-direction contradiction with the source is a
          CONFLICT CANDIDATE only.  Confirmed requires actor identity +
          direction reference + traceable evidence (human review).
    P0-3: pixel_direction_agreement is NOT actor_level_direction_agreement;
          the latter stays unknown unless the actor identity is confirmed,
          even for single-track clips.
    """
    category: str
    prompt_source_conflict_candidate: str     # yes/no
    prompt_source_conflict_confirmed: str     # yes/no/unknown — never auto
    conflict_evidence_level: str              # strong/medium/weak/none
    actor_match_status: str                   # confirmed/possible/unknown
    pixel_direction_agreement: str            # yes/no/unknown (observation)
    actor_level_direction_agreement: str      # unknown unless actor confirmed
    camera_motion_status: str
    review_required: str
    event_applicability: str                  # applicable/not_applicable/unknown
    failure_attribution: str
    reasons: list

    # ---- backward-compatible views used by v1 tests / builders ----------
    @property
    def prompt_source_conflict(self) -> str:
        if self.prompt_source_conflict_confirmed == "yes":
            return "yes"
        if self.prompt_source_conflict_candidate == "yes":
            return "candidate"
        return "no"

    @property
    def pixel_horizontal_mismatch(self) -> str:
        return "yes" if self.pixel_direction_agreement == "no" else "unknown"

    @property
    def semantic_direction_reversal(self) -> str:
        return self.actor_level_direction_agreement  # unknown unless proven

    @property
    def event_adjudication(self) -> str:
        return self.failure_attribution


def assess_direction(prompt_sem: dict, real: SideEvidence,
                     gen: SideEvidence,
                     source_actor_match: str = "unknown",
                     gen_actor_match: str = "unknown",
                     camera_motion_uncertainty: str = "unknown",
                     prompt_available: bool = True) -> DirectionAssessment:
    """Apply the anti-conflation rules and pick an audit category.

    Hard rules (E2.1 regression-tested):
      * P0-2: actor unconfirmed -> conflict stays CANDIDATE, never Confirmed;
              camera motion not excluded -> pixel direction differences
              never escalate to world-motion conflict claims;
      * P0-3: single-track + pixel agreement is NOT actor-level agreement;
              category E requires actor confirmation;
      * no pedestrian condition in prompt -> event not applicable,
        crossing absence cannot be a violation;
      * prompt unavailable/empty -> unknown (NEVER 'none_required').
    """
    reasons: list = []
    direction = prompt_sem.get("direction", "unknown")

    # --- prompt unavailable (P0-5): unknown, not 'no requirement' ---------
    if not prompt_available or direction == "unknown_prompt_unavailable":
        return DirectionAssessment(
            "insufficient_evidence", "no", "unknown", "none", "unknown",
            "unknown", "unknown", "unknown", "yes", "unknown",
            "unresolved_prompt_unavailable",
            ["prompt missing/unreadable — condition unknown; must NOT be "
             "treated as 'no event required'"])

    # --- 0024-type: no pedestrian condition at all -----------------------
    if direction == "none_required":
        return DirectionAssessment(
            "insufficient_evidence", "no", "unknown", "none", "unknown",
            "unknown", "unknown", "unknown", "no",
            "not_applicable", "not_applicable",
            ["prompt prescribes no pedestrian condition — crossing absence "
             "cannot be scored as a violation (needs independent "
             "documented conditioning)"])

    # --- pixel agreement layer (observation) ------------------------------
    px_agreement = "unknown"
    if real.pixel_dir in ("image_left", "image_right") and \
            gen.pixel_dir in ("image_left", "image_right"):
        px_agreement = "yes" if real.pixel_dir == gen.pixel_dir else "no"

    # --- actor level agreement (P0-3) --------------------------------------
    actor_status = source_actor_match if source_actor_match != "unknown" \
        else gen_actor_match
    actor_agreement = "unknown"
    if actor_status == "confirmed" and px_agreement == "yes":
        actor_agreement = "yes"
    elif actor_status == "confirmed" and px_agreement == "no":
        actor_agreement = "no"

    # --- P0-2: conflict candidate / confirmed ------------------------------
    conflict_candidate = "no"
    conflict_confirmed = "unknown"
    evidence_level = "none"
    if prompt_sem.get("semantic_type") == "travel_direction" and \
            real.pixel_dir in ("image_left", "image_right"):
        want = {"left_to_right": "image_right",
                "right_to_left": "image_left"}[direction]
        if real.pixel_dir != want:
            conflict_candidate = "yes"
            evidence_level = "medium"       # numeric + parsed prompt text
            reasons.append(
                f"CANDIDATE: prompt travel direction ({direction}) "
                f"contradicted by source image motion ({real.pixel_dir}); "
                "automatic evidence = parsed prompt + track numbers only")
            # confirmation gate (P0-2): actor identity + direction
            # reference + traceable evidence — none of which the machine
            # can supply alone; external visual review upgrades evidence
            # but human confirmation is still required.
            if source_actor_match == "confirmed":
                evidence_level = "strong"
                reasons.append(
                    "actor identity confirmed + direction reference known; "
                    "confirmation eligible AFTER human review of "
                    "traceable evidence (never automatic)")
            else:
                reasons.append(
                    "actor match unconfirmed -> stays CANDIDATE; "
                    "review_required=yes")

    if conflict_candidate == "yes":
        return DirectionAssessment(
            "prompt_source_conflict", conflict_candidate,
            "unknown", evidence_level, source_actor_match,
            px_agreement, actor_agreement, camera_motion_uncertainty,
            "yes", "unknown", "conflicted_conditions_excluded",
            reasons + ["excluded from clean prompt-adherence statistics; "
                       "report separately; never rewrite the prompt"])

    # --- pixel mismatch without lateral prompt claim (0035-type) ----------
    if px_agreement == "no":
        reasons.append(
            f"pixel dx signs differ (real {real.pixel_dir}, gen "
            f"{gen.pixel_dir}) — OBSERVATION ONLY")
        reasons.append(
            f"prompt has no lateral travel requirement (direction="
            f"{direction}); scale trends real={real.scale}, gen="
            f"{gen.scale}; camera motion status="
            f"{camera_motion_uncertainty}; camera motion NOT excluded -> "
            "no world-motion conflict claim, semantic reversal NOT "
            "established")
        return DirectionAssessment(
            "image_motion_difference", "no", "unknown", "none",
            source_actor_match, px_agreement, actor_agreement,
            camera_motion_uncertainty, "yes", "unknown",
            "unresolved_pending_evidence", reasons)

    # --- actor binding gate (0003/0016-type; P0-3 single-track too) -------
    if source_actor_match != "confirmed" or gen_actor_match != "confirmed":
        gate_reasons = []
        if real.n_person_tracks > 1 or gen.n_person_tracks > 1:
            gate_reasons.append(
                "multiple person tracks; the max-detections track is not "
                "necessarily the prompted actor")
        elif real.n_person_tracks == 1 and gen.n_person_tracks == 1:
            gate_reasons.append(
                "single person track on both sides — pixel agreement "
                "observed, but IDENTITY is unconfirmed (same track count "
                "is not identity evidence)")
        if gen.n_person_tracks == 0:
            gate_reasons.append(
                "no person track in generated side; zero detections is "
                "not proof of absence under low detection reliability")
        if gate_reasons:
            return DirectionAssessment(
                "insufficient_evidence", "no", "unknown", "none",
                source_actor_match, px_agreement, "unknown",
                camera_motion_uncertainty, "yes", "unknown",
                "unresolved_pending_evidence",
                gate_reasons + ["actor binding required before any "
                                "actor-level direction or event judgment"])

    # --- actor confirmed + pixel agreement: category E allowed ------------
    reasons.append(f"pixel directions agree (real {real.pixel_dir}, gen "
                   f"{gen.pixel_dir}) AND actor identity confirmed -> "
                   "actor-level agreement at the image-motion layer; "
                   "world-space semantics remain unassessed")
    return DirectionAssessment(
        "no_direction_issue_observed", "no", "unknown", "none",
        source_actor_match, px_agreement, actor_agreement,
        camera_motion_uncertainty, "no", "applicable",
        "descriptive_only", reasons)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()
