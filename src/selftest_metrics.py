"""Self-test for wmeval modules runnable without ultralytics.

Validates:
1. traj_metrics ports against hand-computed expectations
   (constant-velocity -> consistency 1.0; static -> NaN; jerkiness lowers score)
2. event_checks + uncertainty on synthetic tracks
   (person entering region; unexplained mid-air disappearance -> anomaly)

Run (read-only, prints only):
    /home/moz/miniconda3/envs/wm_eval/bin/python -I src/selftest_metrics.py
"""
import sys

sys.path.insert(0, "src")

import numpy as np

from wmeval.traj_metrics import (comfort_score_norm, curvature_rms,
                                 speed_score, smooth_traj_sg,
                                 trajectory_consistency, evaluate_trajectories)
from wmeval.detect_track import Detection, Track
from wmeval.event_checks import (analyze_disappearances, check_presence,
                                 check_region_entry, run_rule_battery)
from wmeval.uncertainty import adjudicate, attribute_endings

ok = []
fail = []


def check(name, cond, detail=""):
    (ok if cond else fail).append(name)
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")


# --- 1. trajectory_consistency ------------------------------------------------
t = np.arange(100) * 0.1
straight = np.stack([3.0 * t, np.zeros_like(t)], axis=1)          # v=3 m/s const
c = trajectory_consistency(straight[None], reduce="none")
# NOTE: for a perfectly constant velocity the official formula returns
# exp(-sigma/(mu+eps)); finite-difference noise makes sigma ~1e-4 and mu_a ~
# noise, so ~0.9998 is the numerical floor — not a porting error.
check("consistency: constant velocity -> ~1 (>0.9995)",
      c[0] > 0.9995, f"got {c[0]:.6f}")

zigzag = np.stack([t, 0.8 * np.sin(6 * t)], axis=1)               # varying speed
cz = trajectory_consistency(zigzag[None], reduce="none")
check("consistency: zigzag < straight", cz[0] < c[0], f"zigzag {cz[0]:.4f}")

static = np.zeros((50, 2))
cs = trajectory_consistency(static[None], reduce="none")
check("consistency: static -> NaN", np.isnan(cs[0]))

# acceleration term: gentle ramp vs constant speed
ramp = np.stack([0.5 * t ** 2, np.zeros_like(t)], axis=1)          # v=1*t linear
cr = trajectory_consistency(ramp[None], reduce="none")
check("consistency: linear accel ramp < 1", cr[0] < 1.0, f"got {cr[0]:.4f}")

# --- 2. quality group ----------------------------------------------------------
cf = comfort_score_norm(straight[None], reduce="none")
check("comfort: straight line -> ~1", cf[0] > 0.99, f"got {cf[0]:.6f}")
cfz = comfort_score_norm(zigzag[None], reduce="none")
check("comfort: zigzag < straight", cfz[0] < cf[0], f"zigzag {cfz[0]:.4f}")

cv = curvature_rms(straight[None], reduce="none")
check("curvature: straight -> ~1", cv[0] > 0.999, f"got {cv[0]:.6f}")

v6 = speed_score((6.0 * t[:, None] * np.array([[1.0, 0.0]])).squeeze(), reduce="none")
# note: build (T,2) with v=6 m/s along x
traj_v6 = np.stack([6.0 * t, np.zeros_like(t)], axis=1)
s6 = speed_score(traj_v6[None], reduce="none")
expect6 = float(np.clip(np.log1p(6.0) / np.log1p(15.0), 0, 1))
check("speed_score: v=6 -> log1p(6)/log1p(15)",
      abs(s6[0] - expect6) < 1e-9, f"got {s6[0]:.6f} expect {expect6:.6f}")

# --- 3. smoothing --------------------------------------------------------------
sm = smooth_traj_sg(zigzag, dt=0.1)
check("smooth_traj_sg: shape preserved", sm.shape == zigzag.shape)
check("smooth_traj_sg: short input returns raw",
      np.array_equal(smooth_traj_sg(np.zeros((2, 2))), np.zeros((2, 2))))

# --- 4. structured report ------------------------------------------------------
rep = evaluate_trajectories([straight, zigzag], dt=0.1, units="m",
                            v_static=0.1, length_eps=1.0)
check("report: aggregates computed",
      all(np.isfinite(rep["aggregate"][k]) for k in rep["aggregate"]),
      str({k: round(v, 4) for k, v in rep["aggregate"].items()}))

# --- 5. event checks (synthetic) ----------------------------------------------
def mkdet(fi, bbox, cat="person", tid=1, conf=0.9, w=1920, h=1080):
    x1, y1, x2, y2 = bbox
    return Detection(frame_index=fi, timestamp=fi / 30.0, track_id=tid,
                     class_id=0, category=cat, bbox=bbox, conf=conf,
                     norm_bbox=(x1 / w, y1 / h, x2 / w, y2 / h))

# person walks from left edge (outside road region) into center-road region
dets = []
for fi in range(0, 60):
    x = 40 + fi * 8          # px
    dets.append(mkdet(fi, (x, 500, x + 60, 700), tid=1))
# pedestrian vanishes at frame 60 in mid-frame (no edge, no occluder)
track_person = Track(1, "person", 0, dets)
# a car track that persists to the end (detector demonstrably works)
dets_car = [mkdet(f, (900, 600, 1500, 900), cat="car", tid=2, conf=0.95)
            for f in range(0, 100)]
track_car = Track(2, "car", 2, dets_car)
tracks = [track_person, track_car]

pr = check_presence(tracks, 100, actor="person")
check("presence: person found", pr.satisfied is True)

re_ = check_region_entry(tracks, 100, event_id="ev", actor="person",
                         region=(0.30, 0.40, 0.70, 0.70))
# centers: x from 70..520 px /1920 = 0.036..0.27 -> never enters [0.30,0.70]!
check("region_entry: outside region -> False", re_.satisfied is False)

dets_in = []
for fi in range(0, 60):
    x = 600 + fi * 8          # starts inside region [0.31..0.60]
    dets_in.append(mkdet(fi, (x, 500, x + 60, 700), tid=3))
track_in = Track(3, "person", 0, dets_in)
re2 = check_region_entry([track_in], 100, event_id="ev", actor="person",
                         region=(0.30, 0.40, 0.70, 0.70), require_from_outside=True)
check("region_entry: starts inside (no entry from outside) -> False",
      re2.satisfied is False, str(re2.evidence))

# person starts outside then enters
dets_cross = []
for fi in range(0, 60):
    x = 200 + fi * 10         # 0.10 -> 0.47 crosses 0.30 at frame ~37
    dets_cross.append(mkdet(fi, (x, 500, x + 60, 700), tid=4))
track_cross = Track(4, "person", 0, dets_cross)
re3 = check_region_entry([track_cross], 100, event_id="ev", actor="person",
                         region=(0.30, 0.40, 0.70, 0.70))
check("region_entry: crosses into region -> True with entry frame",
      re3.satisfied is True, str(re3.evidence.get("entry_frame")))

# --- 6. disappearance + uncertainty -------------------------------------------
dis = analyze_disappearances(tracks, num_frames=100)
check("disappearance: person mid-air vanish flagged unexplained",
      len(dis) == 1 and dis[0].unexplained and dis[0].track_id == 1,
      dis[0].note if dis else "none")
check("disappearance: car surviving to end not flagged",
      all(d.track_id != 2 for d in dis))

endings = attribute_endings(tracks, dis)
check("attribution: unexplained + confident -> anomaly",
      endings[0].attribution == "anomaly", endings[0].confidence_note)

checks_battery, dis_b = run_rule_battery(tracks, num_frames=100)
verd = adjudicate(checks_battery, endings, [d for t_ in tracks for d in t_.detections], 100,
                  actor="person", interrupt_regions=[(0.25, 0.55, 0.75, 0.95)])
check("verdict: INVALID on actor vanishing inside event region",
      verd.verdict == "invalid", f"{verd.verdict}: {verd.reasons}")
check("battery: region_presence detects person inside road region",
      any(c.event_id == "ped_in_road" and c.satisfied for c in checks_battery))

# same but person ending near border -> explained
dets_edge = []
for fi in range(0, 60):
    x = 1500 + fi * 8         # center exits right border area
    dets_edge.append(mkdet(fi, (x, 500, x + 60, 700), tid=5))
track_edge = Track(5, "person", 0, dets_edge)
dis2 = analyze_disappearances([track_edge, track_car], 100)
end2 = attribute_endings([track_edge, track_car], dis2)
check("attribution: border exit -> exited_frame",
      all(e.attribution == "exited_frame" for e in end2))

# low-confidence track -> detection_quality, uncertain
dets_low = [mkdet(fi, (600, 500, 660, 700), tid=6, conf=0.30)
            for fi in range(0, 60)]
track_low = Track(6, "person", 0, dets_low)
dis3 = analyze_disappearances([track_low, track_car], 100)
end3 = attribute_endings([track_low, track_car], dis3)
cb3, _ = run_rule_battery([track_low, track_car], 100)
v3 = adjudicate(cb3, end3, [d for t_ in [track_low, track_car] for d in t_.detections], 100)
check("verdict: low-conf ending -> UNCERTAIN not INVALID",
      v3.verdict == "uncertain", f"{v3.verdict}: {v3.reasons[:1]}")

# re-detection case: person re-acquired nearby -> redetection, uncertain
dets_a = [mkdet(fi, (600, 500, 660, 700), tid=7) for fi in range(0, 40)]
dets_b = [mkdet(fi, (610, 505, 670, 705), tid=8) for fi in range(44, 100)]
t7, t8 = Track(7, "person", 0, dets_a), Track(8, "person", 0, dets_b)
dis4 = analyze_disappearances([t7, t8, track_car], 100)
end4 = attribute_endings([t7, t8, track_car], dis4)
attr7 = [e for e in end4 if e.track_id == 7]
check("attribution: nearby re-acquisition -> redetection",
      attr7 and attr7[0].attribution == "redetection",
      attr7[0].attribution if attr7 else "none")

# --- 7. regression tests for E1 v3 confirmed bugs -----------------------------
# Bug A (order precision): two events triggered by the SAME frame must
# satisfy an order check.  presence stores full-precision time (0.71812080...)
# while region_* stores round(t, 4) (0.7181); raw comparison made
# 0.71812 > 0.7181 -> spurious failure (0024 real, E1 v3).
from wmeval.event_checks import check_order
r_ord = check_order({"ev_a": 0.7181208053691275, "ev_b": 0.7181},
                    ["ev_a", "ev_b"])
check("order: same-frame events -> True (precision fix)",
      r_ord.satisfied is True, str(r_ord.evidence.get("sequence")))
r_ord2 = check_order({"ev_a": 0.9, "ev_b": 0.1}, ["ev_a", "ev_b"])
check("order: reversed events still -> False",
      r_ord2.satisfied is False, str(r_ord2.evidence.get("sequence")))

# Bug B (id fragmentation): track 9 covers f12-62 near x=0.34; track 10 is an
# EXISTING track (first_frame=45 < 62) whose detections interleave
# (f45-54, gap f55-62, re-appears f63 at the vanish point).  Case-(a)-only
# redetection logic misread this handover as an anomaly -> false INVALID
# (0016 gen, E1 v3).  Expected: redetection (actor persisted).
dets_9 = [mkdet(fi, (500 + fi * 2, 500, 560 + fi * 2, 700), tid=9)
          for fi in range(12, 63)]
dets_10 = [mkdet(fi, (520 + fi * 2, 500, 580 + fi * 2, 700), tid=10)
           for fi in list(range(45, 55)) + list(range(63, 93))]
t9 = Track(9, "person", 0, dets_9)
t10 = Track(10, "person", 0, dets_10)
dis5 = analyze_disappearances([t9, t10, track_car], 93)
end5 = attribute_endings([t9, t10, track_car], dis5)
a9 = [e for e in end5 if e.track_id == 9]
check("attribution: id-fragmentation handover -> redetection",
      a9 and a9[0].attribution == "redetection",
      a9[0].attribution if a9 else "none")

# --- 8. regression: order None-propagation (E2 phase-2 bug report) ----------
# A 3-event sequence with an UNDECIDED middle event must NOT pass as
# ordered: appears(0.1) -> crosses(None) -> exits(0.8) returned True
# pre-fix because the missing event was silently skipped.
r_ord3 = check_order({"appears": 0.1, "crosses": None, "exits": 0.8},
                     ["appears", "crosses", "exits"])
check("order: undecided middle event -> None (not True)",
      r_ord3.satisfied is None, str(r_ord3.evidence))
r_ord3b = check_order({"appears": 0.1, "crosses": 0.4, "exits": 0.8},
                      ["appears", "crosses", "exits"])
check("order: fully-decided in-order 3-sequence -> True",
      r_ord3b.satisfied is True, str(r_ord3b.evidence.get("sequence")))
r_ord3c = check_order({"appears": 0.5, "crosses": 0.4, "exits": 0.8},
                      ["appears", "crosses", "exits"])
check("order: fully-decided out-of-order 3-sequence -> False",
      r_ord3c.satisfied is False, str(r_ord3c.evidence.get("sequence")))
r_ord3d = check_order({"appears": None, "crosses": 0.4, "exits": 0.8},
                      ["appears", "crosses", "exits"])
check("order: undecided FIRST event -> None (not True)",
      r_ord3d.satisfied is None, str(r_ord3d.evidence.get("missing")))

# coexistence guard: track 12 is present AT the vanish frame near the
# point (a different standing actor) -> must NOT be treated as a
# re-detection of track 11; track 11's ending stays anomaly
dets_11 = [mkdet(fi, (500, 500, 560, 700), tid=11) for fi in range(0, 40)]
dets_12 = [mkdet(fi, (640, 500, 700, 700), tid=12) for fi in range(0, 93)]
t11 = Track(11, "person", 0, dets_11)
t12 = Track(12, "person", 0, dets_12)
dis6 = analyze_disappearances([t11, t12, track_car], 93)
end6 = attribute_endings([t11, t12, track_car], dis6)
a11 = [e for e in end6 if e.track_id == 11]
check("attribution: coexisting actor at vanish point -> not redetection",
      a11 and a11[0].attribution != "redetection",
      a11[0].attribution if a11 else "none")

# --- 9. regression: target_fps sampling semantics (E2 phase-6 audit) --------
# VideoReader must yield NATIVE frame indices + native timestamps when
# resampling; the analyzed-frame count formula used by run_minimal must
# match what the reader actually yields (detection-coverage denominator).
import os as _os
from wmeval.video_io import VideoReader

_VID = "repos/SynPeDAS/generated_video/video_0016_0_16_72b_f0-150.mp4"
if _os.path.exists(_VID):
    vr = VideoReader(_VID, target_fps=8.0)
    idxs, tss, _ = vr.read_all()
    n = vr.meta.num_frames
    step = max(1.0, vr.meta.fps / 8.0)
    n_expect = int(round((n - 1) / step)) + 1
    check("video_io: target_fps yield count matches analyzed-frame formula",
          len(idxs) == n_expect, f"yielded {len(idxs)} expect {n_expect}")
    check("video_io: resampled indices stay NATIVE (monotone, spaced by step)",
          idxs == sorted(idxs) and all(
              0 < b - a <= round(step) + 1 for a, b in zip(idxs[:-1], idxs[1:])),
          f"first {idxs[:4]} last {idxs[-2:]}")
    check("video_io: timestamps are native (idx/native_fps)",
          all(abs(t - i / vr.meta.fps) < 1e-6 for i, t in zip(idxs, tss)),
          f"ts[0]={tss[0]:.3f} native idx0={idxs[0]}")
    vr_native = VideoReader(_VID)
    check("video_io: target_fps=None yields every native frame",
          len(vr_native.read_all()[0]) == n, f"native n={n}")
else:
    print("SKIP  video_io target_fps tests (local video not found)")

print(f"\n{len(ok)} passed, {len(fail)} failed")
if fail:
    print("FAILED:", fail)
    sys.exit(1)
print("ALL SELF-TESTS PASSED")
