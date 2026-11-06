"""E2 self-test: actor binding + layered crossing evidence.

Run (read-only, prints only):
    PYTHONPATH=src /home/moz/miniconda3/envs/wm_eval/bin/python src/selftest_e2.py

Covers the anti-conflation requirements from the E2 task brief:
  * event evidence must be attributed to identity CHAINS, never pooled
    across actors (pedestrian A must not satisfy presence while
    pedestrian B satisfies region occupancy for the same "event");
  * fragmented tracks (E1 0016 pattern) must merge into one chain;
  * ambiguous association must surface as uncertain, not be silently
    resolved;
  * Level 4 completion must distinguish completed / in_progress /
    partial_clip_cut / not_progressing.
"""
from __future__ import annotations

import sys

sys.path.insert(0, "src")

from wmeval.detect_track import Detection, Track
from wmeval.actor_binding import build_track_chains, chain_detections
from wmeval.crossing_evidence import evaluate_crossing, summarize

ok, fail = [], []


def check(name, cond, detail=""):
    (ok if cond else fail).append(name)
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")


def mkdet(fi, bbox, cat="person", tid=1, conf=0.9, w=1920, h=1080):
    x1, y1, x2, y2 = bbox
    return Detection(frame_index=fi, timestamp=fi / 30.0, track_id=tid,
                     class_id=0 if cat == "person" else 2, category=cat,
                     bbox=bbox, conf=conf, norm_bbox=(x1 / w, y1 / h, x2 / w, y2 / h))


def mktrack(tid, frames, xs, cat="person", y=650, w=1920):
    dets = [mkdet(f, (x, y, x + 55, y + 160), cat=cat, tid=tid) for f, x in zip(frames, xs)]
    return Track(tid, cat, 0, dets)


# region mirrors DEFAULT (0.25, 0.55, 0.75, 0.95) in 1920x1080 norm space:
# x in [480, 1440] px, y in [594, 1026] px  (y=650+80=730 inside)

# --- 1. chain building -------------------------------------------------------
# E1 0016 pattern: interleaved fragments of ONE pedestrian moving right.
# track 10: f45-54 x 0.52->0.536 ; track 11: f55-62 x 0.538->0.552 ;
# track 12: f63-92 x 0.553->0.599
t10 = mktrack(10, range(45, 55), [998 + i * 4 for i in range(10)])
t11 = mktrack(11, range(55, 63), [1038 + i * 4 for i in range(8)])
t12 = mktrack(12, range(63, 93), [1070 + i * 4 for i in range(30)])
car = mktrack(2, range(0, 93), [300] * 93, cat="car")
chains = build_track_chains([t10, t11, t12, car])
merged = [c for c in chains if set(c.track_ids) == {10, 11, 12}]
check("binding: 0016-style fragments merge into ONE chain",
      len(merged) == 1 and len(chains) == 2,
      f"chains={[(c.track_ids, c.confidence) for c in chains]}")
check("binding: merged chain confidence >= medium (not ambiguous)",
      bool(merged) and merged[0].confidence in ("medium", "high"),
      merged[0].confidence if merged else "none")
if merged:
    check("binding: link evidence recorded (gap + extrapolation error)",
          all(("gap_frames" in l and "pred_err_norm" in l) for l in merged[0].links),
          str(merged[0].links))
    dets = chain_detections([t10, t11, t12], merged[0])
    check("binding: chain detections frame-ordered, no duplicates",
          [d.frame_index for d in dets] == sorted(set(d.frame_index for d in dets)))

# --- 1b. EXACT 0016 topology: alternating handover, NO co-located frames ------
# track 70 mirrors t30: f12-37 then f55,56,59,61,62 (internal gap);
# track 71 mirrors t64: f45-54 then f63-92.  They alternate with NO
# shared frame (the real E1 0016 pattern).  Forward growth from the
# earliest track (70) must absorb 71 via case-(b) semantics (71 has a
# detection inside (37, 37+max_gap] and none AT f37).
d70 = [mkdet(f, (830 + (f - 12) * 4, 650, 885 + (f - 12) * 4, 810), tid=70)
       for f in list(range(12, 38)) + [55, 56, 59, 61, 62]]
d71 = ([mkdet(f, (1000 + (f - 45) * 4, 650, 1055 + (f - 45) * 4, 810), tid=71)
        for f in range(45, 55)]
       + [mkdet(f, (1065 + (f - 63) * 4, 650, 1120 + (f - 63) * 4, 810), tid=71)
          for f in range(63, 93)])
t70, t71 = Track(70, "person", 0, d70), Track(71, "person", 0, d71)
n_orig = len(d70) + len(d71)
chains1b = build_track_chains([t70, t71])
check("binding: exact 0016 alternating handover -> ONE chain {70,71}",
      len(chains1b) == 1 and set(chains1b[0].track_ids) == {70, 71},
      str([(c.track_ids, c.confidence) for c in chains1b]))
d1b = chain_detections([t70, t71], chains1b[0])
check("binding: alternating chain keeps ALL original frames, ordered, "
      "no duplicates",
      len(d1b) == n_orig
      and [d.frame_index for d in d1b] == sorted(d.frame_index for d in d1b),
      f"n={len(d1b)} expect={n_orig} span "
      f"{d1b[0].frame_index}-{d1b[-1].frame_index}")

# --- 2. separate pedestrians stay separate -----------------------------------
far_a = mktrack(20, range(0, 60), [100 + i * 1 for i in range(60)])   # x~0.05-0.08
far_b = mktrack(21, range(0, 60), [900 + i * 5 for i in range(60)])   # x~0.47-0.62
chains2 = build_track_chains([far_a, far_b])
check("binding: spatially distant tracks do NOT merge",
      sorted(len(c.track_ids) for c in chains2) == [1, 1],
      str([c.track_ids for c in chains2]))

# --- 3. ambiguous successor surfaces as uncertain ----------------------------
# ped A ends at f40 near x~0.49 moving right; TWO successors start at f42
# at nearly-equal extrapolation error -> ambiguous link, chain confidence
# ambiguous, and the rival candidate is NAMED but not silently resolved.
a = mktrack(30, range(0, 41), [600 + i * 8 for i in range(41)])
b1 = mktrack(31, range(42, 70), [932 + i * 8 for i in range(28)])
b2 = mktrack(32, range(42, 70), [940 + i * 8 for i in range(28)])
chains3 = build_track_chains([a, b1, b2])
amb = [c for c in chains3 if c.confidence == "ambiguous"]
check("binding: two plausible successors -> ambiguous chain containing A",
      len(amb) == 1 and 30 in amb[0].track_ids and len(amb[0].track_ids) == 2,
      str([(c.track_ids, c.confidence) for c in chains3]))
if amb:
    check("binding: ambiguous link names the rival successor",
          any(l.get("ambiguous_with") in (31, 32) for l in amb[0].links),
          str(amb[0].links))
    rival = ({31, 32} - set(amb[0].track_ids)).pop()
    check("binding: rival stays OUT of the ambiguous chain",
          rival not in amb[0].track_ids
          and any(c.track_ids == [rival] for c in chains3))

# --- 4. crossing evidence: the four levels -----------------------------------
R = (0.25, 0.55, 0.75, 0.95)

# (a) full crossing: enters from left of region, exits right (past x2), dx
xs = [100 + i * 23 for i in range(60)]          # x 0.05 -> 0.77 (out of region)
full = mktrack(40, range(0, 60), xs)
evs = evaluate_crossing([full], 60, region=R)
e = evs[0]
check("crossing: full traversal -> L4 completed",
      e.level4_completion["status"] == "completed",
      f"{e.level3_crossing_progress['status']} "
      f"dx={e.level3_crossing_progress['net_dx_norm']}")

# (b) standing inside region (0003-gen pattern): ~zero net dx, all inside
xs = [620 + (i % 2) for i in range(61)]         # x ~0.32, net dx ~ 0
stand = mktrack(41, range(0, 61), xs)
e = evaluate_crossing([stand], 61, region=R)[0]
check("crossing: standing-in-region -> L2 True, L3 stationary",
      e.level2_region_occupancy["status"] is True
      and e.level3_crossing_progress["status"] == "stationary",
      f"L3={e.level3_crossing_progress['status']} "
      f"dx={e.level3_crossing_progress['net_dx_norm']}")
check("crossing: standing-in-region -> L4 not_progressing (NOT completed)",
      e.level4_completion["status"] == "not_progressing",
      e.level4_completion["status"])

# (c) mid-crossing, clip ends inside region in last 10% -> partial_clip_cut
xs = [200 + i * 16 for i in range(100)]         # x 0.10 -> 0.93 by f99
mid = mktrack(42, range(0, 100), xs)
# but force end inside: stop movement at frame 70 (x~0.68), stay to f99
xs = [200 + min(i, 70) * 16 for i in range(100)]
mid = mktrack(42, range(0, 100), xs)
e = evaluate_crossing([mid], 100, region=R)[0]
check("crossing: cut mid-crossing at clip end -> partial_clip_cut",
      e.level4_completion["status"] == "partial_clip_cut",
      f"status={e.level4_completion['status']} "
      f"t={e.level4_completion.get('last_t_norm')}")

# (d) in-progress: ends inside region BEFORE the last 10%
xs = [200 + min(i, 55) * 16 for i in range(80)]  # ends f79 t=0.9875? -> use 0..79, last t=79/79=1.0
# need last_t < 0.90 and ends inside: make track end at f60 of 80-frame video
xs = [200 + min(i, 55) * 16 for i in range(61)]
prog = mktrack(43, range(0, 61), xs)
e = evaluate_crossing([prog], 80, region=R)[0]
check("crossing: ends inside region before clip end -> in_progress",
      e.level4_completion["status"] == "in_progress",
      f"status={e.level4_completion['status']}")

# --- 5. ANTI-CONFLATION: two actors, checks must not pool --------------------
# Actor A: appears far left, static (would satisfy presence only)
# Actor B: inside region, moving right (would satisfy occupancy+progress)
#          but ends mid-region before clip end
a2 = mktrack(50, range(0, 60), [120] * 60)                       # x~0.06 static
b2 = mktrack(51, range(0, 60), [500 + i * 18 for i in range(60)])  # x 0.26->0.81... need end inside
b2 = mktrack(51, range(0, 60), [500 + min(i, 40) * 18 for i in range(60)])  # ends x~0.63 inside
evs = evaluate_crossing([a2, b2], 60, region=R)
s = summarize(evs)
per_a, per_b = evs[0].to_dict(), evs[1].to_dict()
check("anti-conflation: chain A presence yes but occupancy no",
      per_a["level1_presence"]["status"] is True
      and per_a["level2_region_occupancy"]["status"] is False)
check("anti-conflation: chain B occupancy+progress but overall NOT completed",
      per_b["level2_region_occupancy"]["status"] is True
      and s["completed_by_single_chain"] is False,
      f"best={s['best_single_chain_status']}")
check("anti-conflation: no chain satisfies all four levels",
      all(evd["level4_completion"]["status"] != "completed"
          for evd in s["per_chain"]),
      str([(evd['chain']['track_ids'], evd['level4_completion']['status'])
           for evd in s["per_chain"]]))

# --- 6. chain + crossing integrated: fragmented full crossing ----------------
# fragments of ONE crossing pedestrian (0016 pattern) -> chain -> completed
f1 = mktrack(60, range(0, 30), [200 + i * 20 for i in range(30)])    # x 0.10->0.41
f2 = mktrack(61, range(33, 70), [800 + i * 20 for i in range(37)])   # x 0.42->0.80
chains6 = build_track_chains([f1, f2])
evs6 = evaluate_crossing([f1, f2], 70, region=R, chains=chains6)
s6 = summarize(evs6)
check("integrated: fragmented crossing merges and completes as ONE actor",
      s6["completed_by_single_chain"] is True
      and s6["completed_chain"] is not None,
      f"chains={[c.track_ids for c in chains6]} "
      f"status={s6['best_single_chain_status']}")
check("integrated: completion attributed to the merged chain only",
      len([e for e in evs6
           if e.level4_completion.get("status") == "completed"]) == 1)

print(f"\n{len(ok)} passed, {len(fail)} failed")
if fail:
    print("FAILED:", fail)
    sys.exit(1)
print("ALL E2 SELF-TESTS PASSED")
