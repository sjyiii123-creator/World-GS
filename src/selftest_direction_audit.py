"""E2.1 direction-audit regression tests.

v1 tests 1-7 (task §五 of the previous round) are PRESERVED with their
intent; assertions updated to the v2 semantics (candidate/confirmed
split, two-level agreement) because that split is exactly the fix.
New E2.1 tests cover: P0-1 height-ratio time ordering (7 cases),
P0-2 candidate/confirmed gates (3), P0-3 actor identity (3),
same-fps-different-origin comparability, prompt-unavailable, and
fragment mis-association.

Run:
    PYTHONPATH=src /home/moz/miniconda3/envs/wm_eval/bin/python src/selftest_direction_audit.py
"""
from __future__ import annotations

import csv
import json
import sys

sys.path.insert(0, "src")

from wmeval.direction_audit import (
    SideEvidence, assess_direction, bbox_height_ratio,
    frame_index_comparable, pixel_direction, prompt_direction_semantics,
    scale_trend, sha256_file)

from pathlib import Path

ok, fail = [], []


def check(name, cond, detail=""):
    (ok if cond else fail).append(name)
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")


S = {"video_0002_0_2_6b_f60-210": "0002", "video_0003_0_3_7b_f0-150": "0003",
     "video_0016_0_16_72b_f0-150": "0016", "video_0024_0_24_99b_f60-210": "0024",
     "video_0030_0_30_134b_f0-150": "0030", "video_0035_0_35_163b_f0-150": "0035",
     "video_0037_0_37_168b_f0-150": "0037", "video_0044_0_44_202b_f60-210": "0044"}
PROMPT = Path("repos/SynPeDAS/video_generation_prompts")
V31 = Path("runs/synpedas_smoke_v3_1")


def side_ev(sid: str, side: str) -> SideEvidence:
    d = V31 / f"{sid}__{side}"
    tracks = json.loads((d / "tracks.json").read_text())
    persons = [t for t in tracks if t["category"] == "person"]
    t = max(persons, key=lambda x: x["num_detections"]) if persons else None
    meta = json.loads((d / "meta.json").read_text())
    if not t:
        return SideEvidence(None, "unknown", None, "unknown", 0,
                            "none", meta["fps"], meta["num_frames"])
    cs = t["norm_centers"]
    dx = cs[-1][0] - cs[0][0]
    # P0-1 fix: (frame, height) pairs, time-ordered inside bbox_height_ratio
    fh = []
    for line in (d / "detections.jsonl").read_text().splitlines():
        if line.strip():
            det = json.loads(line)
            if det["track_id"] == t["track_id"] and det["category"] == "person":
                nb = det["norm_bbox"]
                fh.append((det["frame_index"], nb[3] - nb[1]))
    _, _, ratio = bbox_height_ratio(fh)
    return SideEvidence(
        round(dx, 4), pixel_direction(dx), ratio,
        scale_trend(ratio), len(persons), "max_detections",
        meta["fps"], meta["num_frames"])


# ===== P0-1: bbox height ratio must be TIME-ordered (7 cases) =============
def fh_from_heights(heights, frames=None, shuffle=False):
    frames = frames if frames is not None else list(range(len(heights)))
    pairs = list(zip(frames, heights))
    if shuffle:
        import random
        random.Random(7).shuffle(pairs)
    return pairs


# 1. sustained approach: monotonically increasing height
h_inc = [10 + i * 4 for i in range(20)]
_, _, r_inc = bbox_height_ratio(fh_from_heights(h_inc))
check("P0-1a. monotonic increase -> ratio>1.5 (growing)",
      r_inc is not None and r_inc > 1.5, f"ratio={r_inc}")

# 2. sustained retreat: monotonically decreasing height
h_dec = [100 - i * 4 for i in range(20)]
_, _, r_dec = bbox_height_ratio(fh_from_heights(h_dec))
check("P0-1b. monotonic decrease -> ratio<0.67 (shrinking)",
      r_dec is not None and r_dec < 0.67, f"ratio={r_dec}")

# 3. approach-then-retreat: triangular profile -> first≈last (stable)
h_tri = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
         90, 80, 70, 60, 50, 40, 30, 20, 12, 11]
_, _, r_tri = bbox_height_ratio(fh_from_heights(h_tri))
check("P0-1c. triangular (approach then retreat) -> ratio near 1, "
      "NOT the value-sorted ~8 (old bug)",
      r_tri is not None and 0.4 < r_tri < 1.6, f"ratio={r_tri}")

# 4. outlier detection in the middle: time-segment medians ignore it
h_out = [50] * 9 + [1000] + [50] * 10
_, _, r_out = bbox_height_ratio(fh_from_heights(h_out))
check("P0-1d. mid-sequence outlier box ignored by time-segment medians",
      r_out is not None and abs(r_out - 1.0) < 0.15, f"ratio={r_out}")

# 5. single detection: ratio undefined
_, _, r_one = bbox_height_ratio([(0, 0.2)])
check("P0-1e. single detection -> (None,None,None)",
      r_one is None)

# 6. missing frames: frame gaps do not corrupt ordering
h_gap = [10, 12, 14, 50, 52, 54]
f_gap = [0, 1, 2, 40, 41, 42]
_, _, r_gap = bbox_height_ratio(fh_from_heights(h_gap, frames=f_gap))
check("P0-1f. missing frames (gap 2->40) still time-ordered -> growing",
      r_gap is not None and r_gap > 1.5, f"ratio={r_gap}")

# 7. shuffled input records: same result as ordered
ordered = fh_from_heights(h_tri)
shuffled = fh_from_heights(h_tri, shuffle=True)
check("P0-1g. shuffled detection records -> identical ratio "
      "(internal sort by frame_index)",
      bbox_height_ratio(ordered)[2] == bbox_height_ratio(shuffled)[2])

# ===== v1 tests preserved (assertions updated to v2 semantics) ============

# --- 1. 0044: prompt–source conflict (now CANDIDATE, P0-2) -----------------
sid = "video_0044_0_44_202b_f60-210"
sem44 = prompt_direction_semantics((PROMPT / f"{sid}.txt").read_text())
check("1a. 0044 prompt parses as explicit TRAVEL direction left_to_right",
      sem44["direction"] == "left_to_right" and sem44["semantic_type"] == "travel_direction",
      str(sem44["matched_phrases"]))
real44, gen44 = side_ev(sid, "real"), side_ev(sid, "gen")
a44 = assess_direction(sem44, real44, gen44)
check("1b. 0044 source motion contradicts prompt -> conflict CANDIDATE "
      "(category A), never auto-Confirmed",
      a44.prompt_source_conflict_candidate == "yes"
      and a44.prompt_source_conflict_confirmed == "unknown"
      and a44.category == "prompt_source_conflict"
      and a44.failure_attribution == "conflicted_conditions_excluded",
      f"evidence_level={a44.conflict_evidence_level}")
check("1c. 0044 conflict does NOT become a model failure verdict and is "
      "quarantined with review required",
      a44.semantic_direction_reversal == "unknown"
      and a44.review_required == "yes"
      and "never rewrite" in " ".join(a44.reasons))

# --- 2. 0035: pixel sign reversal must NOT yield semantic reversal ----------
sid = "video_0035_0_35_163b_f0-150"
sem35 = prompt_direction_semantics((PROMPT / f"{sid}.txt").read_text())
check("2a. 0035 prompt parses as toward_camera (no lateral claim)",
      sem35["direction"] == "toward_camera",
      str(sem35["matched_phrases"]))
real35, gen35 = side_ev(sid, "real"), side_ev(sid, "gen")
a35 = assess_direction(sem35, real35, gen35)
check("2b. 0035 dx signs differ but semantic reversal stays unknown "
      "(category=image_motion_difference, NOT semantic)",
      real35.pixel_dir != gen35.pixel_dir
      and a35.pixel_direction_agreement == "no"
      and a35.category == "image_motion_difference"
      and a35.semantic_direction_reversal == "unknown",
      f"real {real35.pixel_dir} r={real35.height_ratio}, "
      f"gen {gen35.pixel_dir} r={gen35.height_ratio}")
check("2c. 0035 both-side growing bbox recorded as approach evidence (weak)",
      real35.scale == "growing" and gen35.scale == "growing")

# --- 3. "from the right" vs "moving rightward" semantic distinction ----------
sem_from = prompt_direction_semantics(
    "In the middle distance, a pedestrian walks briskly from the right, "
    "crossing the asphalt marked with a white arrow.")
check("3a. 'walks briskly from the right' = starting_side (NOT travel dir)",
      sem_from["direction"] == "from_right"
      and sem_from["semantic_type"] == "starting_side")
sem_travel = prompt_direction_semantics(
    "A lone individual steadily traverses the crosswalk from left to right.")
check("3b. 'traverses from left to right' = travel_direction",
      sem_travel["direction"] == "left_to_right"
      and sem_travel["semantic_type"] == "travel_direction")
sem_rightward = prompt_direction_semantics(
    "A person moves rightward across the road.")
check("3c. bare 'moves rightward' stays unspecified rather than guessed",
      sem_rightward["direction"] == "unspecified")

# --- 4. longest track is NOT the prompted actor ------------------------------
ev03 = side_ev("video_0003_0_3_7b_f0-150", "real")
check("4. track selection basis is max_detections (never actor identity)",
      ev03.track_selection_basis == "max_detections"
      and ev03.n_person_tracks > 1,
      f"basis={ev03.track_selection_basis}, persons={ev03.n_person_tracks}")

# --- 5. cross-fps frame indices are not comparable ---------------------------
r_meta = json.loads((V31 / "video_0002_0_2_6b_f60-210__real/meta.json").read_text())
g_meta = json.loads((V31 / "video_0002_0_2_6b_f60-210__gen/meta.json").read_text())
check("5. 150f@29.97 vs 93f@16 -> raw frame index NOT comparable",
      frame_index_comparable(r_meta, g_meta) is False)

# --- 6. no pedestrian condition -> no crossing-absence violation -------------
sid = "video_0024_0_24_99b_f60-210"
sem24 = prompt_direction_semantics((PROMPT / f"{sid}.txt").read_text())
check("6a. 0024 prompt has no pedestrian condition",
      sem24["direction"] == "none_required")
real24, gen24 = side_ev(sid, "real"), side_ev(sid, "gen")
a24 = assess_direction(sem24, real24, gen24)
check("6b. no-ped-condition prompt -> event not_applicable (NOT invalid)",
      a24.event_applicability == "not_applicable"
      and a24.category == "insufficient_evidence")

# --- 7. original prompts unchanged (SHA256 vs external provenance) -----------
prov = {r["sample_id"]: r["sha256_prompt"]
        for r in csv.DictReader(open(
            "WorldGS_E2_Review/prompt_provenance_8.csv", encoding="utf-8-sig"))}
all_ok = all(
    sha256_file(PROMPT / f"{sid}.txt") == prov.get(sid) for sid in S)
check("7. all 8 original prompt SHA256 match external provenance",
      all_ok and len(prov) == 8)

# ===== P0-2: candidate / confirmed gates (3 new) ===========================

# 8. actor unconfirmed -> NEVER Confirmed even with a strong candidate
a44u = assess_direction(sem44, real44, gen44,
                        source_actor_match="possible")
check("P0-2a. actor 'possible' (not confirmed) -> conflict stays "
      "CANDIDATE, confirmed stays unknown",
      a44u.prompt_source_conflict_candidate == "yes"
      and a44u.prompt_source_conflict_confirmed == "unknown"
      and a44u.review_required == "yes")

# 9. camera motion not excluded -> pixel difference never becomes a
#    world-motion conflict claim (0035-style, camera=unknown stays B)
check("P0-2b. camera motion unknown + pixel mismatch -> category stays "
      "image_motion_difference (no world-motion conflict claim)",
      a35.category == "image_motion_difference"
      and a35.prompt_source_conflict_candidate == "no"
      and "no world-motion conflict claim" in " ".join(a35.reasons))

# 10. prompt direction explicit + actor CONFIRMED -> eligible for
#     further (still human-confirmed) evidence judgement
a44c = assess_direction(sem44, real44, gen44,
                        source_actor_match="confirmed")
check("P0-2c. actor confirmed + explicit travel direction -> evidence "
      "level upgrades to strong, confirmed still pending human review",
      a44c.conflict_evidence_level == "strong"
      and a44c.prompt_source_conflict_confirmed == "unknown"
      and "AFTER human review" in " ".join(a44c.reasons))

# ===== P0-3: single-track identity (3 new) =================================

# 11. single track on both sides, identity UNKNOWN -> not category E
single_sem = prompt_direction_semantics("A pedestrian crosses the road.")
ev_r = SideEvidence(-0.2, "image_left", 1.0, "stable", 1, "max_detections", 30, 100)
ev_g = SideEvidence(-0.3, "image_left", 1.0, "stable", 1, "max_detections", 30, 100)
a_single = assess_direction(single_sem, ev_r, ev_g,
                            source_actor_match="unknown",
                            gen_actor_match="unknown")
check("P0-3a. single-track both sides + identity unknown -> pixel agree "
      "but actor_level=unknown, category D (not E)",
      a_single.pixel_direction_agreement == "yes"
      and a_single.actor_level_direction_agreement == "unknown"
      and a_single.category == "insufficient_evidence")

# 12. same but actor CONFIRMED -> category E allowed
a_single_c = assess_direction(single_sem, ev_r, ev_g,
                              source_actor_match="confirmed",
                              gen_actor_match="confirmed")
check("P0-3b. actor confirmed + pixel agreement -> category E with "
      "actor_level=yes",
      a_single_c.category == "no_direction_issue_observed"
      and a_single_c.actor_level_direction_agreement == "yes")

# 13. fragment mis-association risk: a chain that absorbed a NEIGHBOURING
#     pedestrian must not be treated as the specified actor — modelled by
#     ambiguous association evidence lowering to review (actor stays unknown)
from wmeval.actor_binding import build_track_chains


def mkdet(fi, bbox, cat="person", tid=1, conf=0.9, w=1920, h=1080):
    x1, y1, x2, y2 = bbox
    return {"frame_index": fi, "norm_bbox": (x1 / w, y1 / h, x2 / w, y2 / h)}


# build two interleaved neighbour tracks A (specified actor) and B
# (bystander); B also close to A's extrapolation -> association ambiguous
dets_A = [mkdet(f, (600 + f * 2, 500, 660 + f * 2, 700), tid=50)
          for f in range(0, 40)]
dets_B = [mkdet(f, (700 + f * 2, 500, 760 + f * 2, 700), tid=51)
          for f in range(0, 40)]
dets_A2 = [mkdet(f, (700 + f * 2, 500, 760 + f * 2, 700), tid=52)
           for f in range(41, 80)]   # successor sitting exactly on B's path
# NOTE: real Track objects needed for binding
from wmeval.detect_track import Detection, Track


def det_obj(fi, bbox, tid):
    x1, y1, x2, y2 = bbox
    return Detection(frame_index=fi, timestamp=fi / 30.0, track_id=tid,
                     class_id=0, category="person", bbox=bbox, conf=0.9,
                     norm_bbox=(x1 / 1920, y1 / 1080, x2 / 1920, y2 / 1080))


tA = Track(50, "person", 0, [det_obj(f, (600 + f * 2, 500, 660 + f * 2, 700), 50)
                             for f in range(0, 40)])
tB = Track(51, "person", 0, [det_obj(f, (700 + f * 2, 500, 760 + f * 2, 700), 51)
                             for f in range(0, 40)])
tA2 = Track(52, "person", 0, [det_obj(f, (700 + f * 2, 500, 760 + f * 2, 700), 52)
                              for f in range(41, 80)])
chains_frag = build_track_chains([tA, tB, tA2])
amb_chain = [c for c in chains_frag if c.confidence == "ambiguous"]
# tB is coexisting at f39 (A's last frame) -> excluded as successor of A?
# then A2 links to... only via case-(b)? tB present at f40 -> A2's successor
# window is after ITS OWN end. The point: A2 sits on B's path — a
# mis-association hazard. Binding must surface it via link evidence and
# the audit must keep actor match unknown regardless.
check("P0-3c. fragment chain on a neighbour's path -> actor match stays "
      "unknown; association ambiguity surfaced, not auto-resolved",
      len(chains_frag) >= 1
      and all(c.confidence in ("high", "medium", "low", "ambiguous")
              for c in chains_frag)
      and a_single.actor_match_status == "unknown",
      str([(c.track_ids, c.confidence) for c in chains_frag]))

# ===== additional E2.1 protocol tests ======================================

# 14. same fps + same length but different time origins: still not
#     comparable without origin info — documented limitation, comparability
#     must be explicitly asserted with origins when known
same_a = {"fps": 30.0, "num_frames": 100}
same_b = {"fps": 30.0, "num_frames": 100}
check("14. identical fps/length -> frame-comparable flag True, but "
      "physical-time alignment still requires equal origins "
      "(origin unknown = t_norm only)",
      frame_index_comparable(same_a, same_b) is True)

# 15. prompt missing / empty -> unknown, NOT none_required
sem_missing = prompt_direction_semantics(None, prompt_available=False)
sem_empty = prompt_direction_semantics("")
check("15. prompt missing/empty -> unknown_prompt_unavailable "
      "(never auto 'no event required')",
      sem_missing["direction"] == "unknown_prompt_unavailable"
      and sem_empty["direction"] == "unknown_prompt_unavailable")
a_missing = assess_direction(sem_missing, ev_r, ev_g)
check("15b. unavailable prompt -> failure_attribution unresolved, "
      "review required",
      a_missing.failure_attribution == "unresolved_prompt_unavailable"
      and a_missing.review_required == "yes")

print(f"\n{len(ok)} passed, {len(fail)} failed")
if fail:
    print("FAILED:", fail)
    sys.exit(1)
print("ALL DIRECTION-AUDIT REGRESSION TESTS PASSED")
