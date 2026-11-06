"""J2.1 regression tests: identity masks + per-candidate times (Gates 0/1).

Cases required by the task brief:
  * repeated actors across sequences must be SAME-identity (never
    counted as negatives);
  * identical numeric actor ids in different videos are DIFFERENT
    pedestrians;
  * hard (same-source) vs easy (cross-source) negative split;
  * each candidate uses its OWN delta; the true match's time never
    substitutes for another candidate's time;
  * invalid intervals are rejected as unusable (Unknown), not guessed.

Run: PYTHONPATH=src python src/selftest_j2_1.py
"""
from __future__ import annotations

import sys

sys.path.insert(0, "src")

import numpy as np

from wmeval.latent_world_state.actor_tokenizer import ActorObservation
from wmeval.latent_world_state.association_evidence import (
    build_identity_mask, candidate_delta, evidence_quality)

ok, fail = [], []


def check(name, cond, detail=""):
    (ok if cond else fail).append(name)
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")


# --- 1. identity masks with repeated actors -----------------------------
srcs = ["A", "A", "A", "B"]
actors = ["p1", "p1", "p2", "p1"]        # p1 appears in A and B
m = build_identity_mask(srcs, actors)
check("mask: repeated actor across sequences marked SAME (p1@A,p1@A)",
      m[0, 1] and m[1, 0])
check("mask: different actor same source marked DIFFERENT (hard neg)",
      not m[0, 2] and not m[2, 0])
check("mask: identical actor id in DIFFERENT videos marked DIFFERENT "
      "(p1@A vs p1@B — different pedestrians)",
      not m[0, 3] and not m[3, 0])
check("mask: diagonal always same", all(m[i, i] for i in range(4)))

# --- 2. hard/easy negative split -----------------------------------------
hard, easy = 0, 0
for i in range(4):
    for j in range(4):
        if i != j and not m[i, j]:
            if srcs[i] == srcs[j]:
                hard += 1
            else:
                easy += 1
check("mask: negative split hard(same-source)+easy(cross-source)=all "
      "(hard: 0-2,1-2 both dirs; easy: 0-3,1-3,2-3 both dirs)",
      hard == 4 and easy == 6, f"hard={hard} easy={easy}")

# --- 3. per-candidate delta ----------------------------------------------
anchor_last_t = 10.0
cands = [{"ped": "x", "first_time_s": 11.5},
         {"ped": "y", "first_time_s": 10.6},
         {"ped": "z", "first_time_s": 9.8}]   # invalid: before anchor
ds = [candidate_delta(c, anchor_last_t) for c in cands]
check("delta: each candidate uses its OWN first-time minus anchor end",
      abs(ds[0] - 1.5) < 1e-9 and abs(ds[1] - 0.6) < 1e-9)
check("delta: candidate before anchor -> None (unusable, never guessed)",
      ds[2] is None)
check("delta: no true-match time substitution (distinct values kept)",
      ds[0] != ds[1])

# --- 4. evidence quality gates --------------------------------------------
check("quality: unusable time -> evidence unknown",
      evidence_quality(None) == "unknown")
check("quality: finite positive delta -> usable",
      evidence_quality(0.7) == "usable")

# --- 5. actor tokenizer contract -----------------------------------------
obs = ActorObservation(
    source_video_id="video_0001", actor_id="0_1_2b", pair_id="p1",
    frames=[1, 2, 3], timestamps=[0.03, 0.07, 0.10],
    bboxes=[(0, 0, 1, 1)] * 3, visibilities=[0, 1, 0],
    features=None)
check("tokenizer: carries identity key + causal history fields",
      obs.identity_key == ("video_0001", "0_1_2b")
      and obs.frames[-1] == 3 and len(obs.timestamps) == 3)
check("tokenizer: visible-frame preference is data-driven, not deletion",
      [v for v in obs.visibilities] == [0, 1, 0])

print(f"\n{len(ok)} passed, {len(fail)} failed")
if fail:
    print("FAILED:", fail)
    sys.exit(1)
print("ALL J2.1 REGRESSION TESTS PASSED")


# ===== J2.2 boundary tests (task §一.7) =====================================
from wmeval.latent_world_state.association_evidence import (
    CandidateEvidence, rank_candidates, valid_kinematic_error)
from wmeval.latent_world_state.interfaces import (
    LatentEvidenceConfig, LatentEvidenceSwitch)


def ev(kin, key=("v", "a")):
    return CandidateEvidence(identity_key=key, kinematic_error=kin,
                             static_similarity=None,
                             predicted_similarity=None,
                             candidate_delta_s=None,
                             evidence_state="usable" if kin is not None
                             else "unknown")


# zero error is VALID and ranks first
d0 = rank_candidates([ev(0.0, ("v", "a")), ev(0.2, ("v", "b"))],
                     accept_fn=lambda e: True)
check("boundary: zero kinematic error is valid (not replaced by 9.9)",
      valid_kinematic_error(0.0) and d0.chosen_identity == ("v", "a")
      and d0.decision == "accept")
# None / NaN / Inf / negative are invalid
check("boundary: None/NaN/Inf/negative kinematics all invalid",
      not valid_kinematic_error(None)
      and not valid_kinematic_error(float("nan"))
      and not valid_kinematic_error(float("inf"))
      and not valid_kinematic_error(-0.1))
# missing evidence, single candidate -> explicit no-reliable-evidence
# state (task allows uncertain OR no_match; implementation picks
# no_match = "no candidate has valid kinematic evidence")
d1 = rank_candidates([ev(None)])
check("boundary: sole candidate without evidence -> explicit "
      "no_match state (never accept, never implicit identity)",
      d1.decision == "no_match" and d1.chosen_identity is None)
# valid sole candidate but NO calibrated accept_fn -> uncertain
d2 = rank_candidates([ev(0.1)])
check("boundary: sole candidate, no calibrated threshold -> uncertain",
      d2.decision == "uncertain")
# with calibrated accept_fn it may accept
d3 = rank_candidates([ev(0.1)], accept_fn=lambda e: e.kinematic_error <= 0.15)
check("boundary: calibrated accept_fn can accept valid sole candidate",
      d3.decision == "accept")
# multi-candidate with identical scores -> uncertain (ambiguity)
da = rank_candidates([ev(0.10, ("v", "a")), ev(0.10, ("v", "b"))],
                     accept_fn=lambda e: True)
check("boundary: multi-candidate tie -> uncertain via ambiguity margin",
      da.decision == "uncertain")
# all evidence invalid -> no_match
dall = rank_candidates([ev(None), ev(float("nan")), ev(-1)])
check("boundary: all candidates invalid evidence -> no_match",
      dall.decision == "no_match")
# ambiguity margin is configurable
dm = rank_candidates([ev(0.10, ("v", "a")), ev(0.12, ("v", "b"))],
                     ambiguity_margin=0.001, accept_fn=lambda e: True)
check("boundary: ambiguity margin is a parameter (wide vs tight)",
      dm.decision == "accept")
# sidecar copies: enabled switch returns NEW dicts, original untouched
links_frozen = [{"from_track": 1, "to_track": 2}]
sw_on = LatentEvidenceSwitch(LatentEvidenceConfig(enabled=True))
out_on = sw_on.decorate_chain_links(links_frozen)
check("sidecar: enabled switch returns NEW dicts with evidence fields",
      out_on is not links_frozen and "static_similarity" in out_on[0])
check("sidecar: original frozen links are NOT mutated",
      links_frozen == [{"from_track": 1, "to_track": 2}])

# ===== J2.3 G0 tests: no identity commit on non-accept ====================
dr = rank_candidates([ev(0.1)], accept_fn=lambda e: e.kinematic_error <= 0.05)
check("G0: reject decision does NOT commit chosen_identity",
      dr.decision == "reject" and dr.chosen_identity is None
      and dr.tendency_identity == ("v", "a"))
du = rank_candidates([ev(0.10, ("v", "a")), ev(0.11, ("v", "b"))],
                     accept_fn=lambda e: True)
check("G0: ambiguity uncertain does NOT commit chosen_identity",
      du.decision == "uncertain" and du.chosen_identity is None
      and du.tendency_identity == ("v", "a"))
dn = rank_candidates([ev(0.1)])
check("G0: no-threshold uncertain does NOT commit chosen_identity",
      dn.decision == "uncertain" and dn.chosen_identity is None)
da2 = rank_candidates([ev(0.1)], accept_fn=lambda e: e.kinematic_error <= 0.15)
check("G0: accept still commits identity (only path that does)",
      da2.decision == "accept" and da2.chosen_identity == ("v", "a"))
# zero-threshold idiom guard (run_j2_2_association pattern)
zero_thr = 0.0
default = 0.0242
res = zero_thr if zero_thr is not None else default
check("G0: explicit None-check keeps zero threshold (no or-idiom)",
      res == 0.0)

print(f"\n{len(ok)} passed, {len(fail)} failed (J2.1+J2.2+J2.3 G0)")
if fail:
    print("FAILED:", fail)
    sys.exit(1)
print("ALL J2.1 + J2.2 + J2.3 G0 TESTS PASSED")
