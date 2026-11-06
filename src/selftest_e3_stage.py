#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Self-tests for the E3 stage modules — run BEFORE any real E3 run.

E3-D0 (src/e3_d0_temporal_diagnostics.py, pure functions only — no torch):
  D0-1  cosine similarity basics
  D0-2  constant prediction vs varying truth -> delta_dir > 0, R_delta == 0
  D0-3  identical sequences -> delta_dir == 0, R_delta == 1
  D0-4  partial collapse -> 0 < R_delta < 1, delta_dir > 0
  D0-5  fewer than 2 timesteps raises InsufficientTimesteps
  D0-6  pre-registered verdict rule (supported / borderline / no_evidence)
  D0-7  bootstrap CI deterministic under fixed seed

E3-A (src/e3_a_event_state_baseline.py):
  A-1   walk -> stop -> resume -> walk yields CROSSING -> STOPPED -> RESUMING
  A-2   stationary from the start -> WAITING only (no STOPPED/CROSSING)
  A-3   alternating jitter speeds -> no state flapping (no CROSSING)
  A-4   frame gap beyond tolerance -> UNKNOWN around the gap
  A-5   real tracks.json smoke: valid states, contiguous sorted segments
  A-6   CLI end-to-end into a temp dir produces the output contract

Usage: python3 -I src/selftest_e3_stage.py   (repo root; stdlib only)
"""
from __future__ import annotations

import json
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e3_d0_temporal_diagnostics as d0  # noqa: E402
import e3_a_event_state_baseline as e3a  # noqa: E402
import run_e3_d0_1_controls as d01  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "runs" / "e3_stage_selftest"

TESTS = []


def register(name):
    def deco(fn):
        TESTS.append((name, fn))
        return fn
    return deco


def make_seq(dim=8, base=1.0, spread=0.0, steps=6, seed=1):
    """A feature sequence whose per-step perturbation magnitude ~ spread
    in RANDOM directions (so the sequence genuinely changes direction)."""
    rng = random.Random(seed)
    vecs = []
    cur = [base + rng.uniform(-0.01, 0.01) for _ in range(dim)]
    for _ in range(steps):
        vecs.append(list(cur))
        cur = [v + spread * rng.uniform(-1.0, 1.0) for v in cur]
    return vecs


# -------------------------------- E3-D0 -------------------------------------

@register("D0-1_cosine_basics")
def d0_1():
    a = [1.0, 0.0, 0.0]
    b = [0.0, 1.0, 0.0]
    ok = (abs(d0.cosine_similarity(a, a) - 1.0) < 1e-9
          and abs(d0.cosine_similarity(a, b)) < 1e-9
          and d0.cosine_similarity([0.0] * 3, [0.0] * 3) == 1.0)
    return ok, "same->1, orthogonal->0, zero-vectors->1"


@register("D0-2_constant_prediction_collapses")
def d0_2():
    true = make_seq(spread=1.0, seed=2)
    pred = [list(true[0])] * len(true)
    d = d0.temporal_diagnostics(pred, true)
    ok = d["delta_dir"] > 0.1 and d["R_delta"] < 1e-6
    return ok, (f"constant pred: delta_dir={d['delta_dir']:.3f} > 0, "
                f"R_delta={d['R_delta']:.2e} ~ 0")


@register("D0-3_identical_sequences")
def d0_3():
    true = make_seq(spread=0.3, seed=3)
    d = d0.temporal_diagnostics([list(v) for v in true], true)
    ok = abs(d["delta_dir"]) < 1e-12 and abs(d["R_delta"] - 1.0) < 1e-9
    return ok, f"delta_dir={d['delta_dir']:.1e}, R_delta={d['R_delta']:.6f}"


@register("D0-4_partial_collapse")
def d0_4():
    true = make_seq(spread=0.8, seed=4)
    pred = make_seq(spread=0.15, seed=4)
    d = d0.temporal_diagnostics(pred, true)
    ok = 0.0 < d["R_delta"] < 1.0 and d["delta_dir"] > 0
    return ok, (f"pred spread 0.15 vs true 0.8: R_delta={d['R_delta']:.3f} in (0,1), "
                f"delta_dir={d['delta_dir']:.3f} > 0")


@register("D0-5_insufficient_timesteps")
def d0_5():
    raised = False
    try:
        d0.temporal_diagnostics([[1.0, 0.0]], [[1.0, 0.0]])
    except d0.InsufficientTimesteps:
        raised = True
    return raised, "single timestep raises InsufficientTimesteps"


@register("D0-6_verdict_rule_preregistered")
def d0_6():
    def diag(dd, rd):
        return {"delta_dir": dd, "R_delta": rd}
    supported = d0.summarize_diagnostics(
        [diag(0.30 + 0.02 * i, 0.20 + 0.01 * i) for i in range(40)])
    borderline = d0.summarize_diagnostics(
        [diag(0.40 + 0.02 * i, 0.90 + 0.01 * i) for i in range(40)])
    none = d0.summarize_diagnostics(
        [diag(-0.30 + 0.01 * i, 0.95 + 0.005 * i) for i in range(40)])
    ok = (supported["evidence_rule"]["verdict"] == "temporal_oversmoothing_supported"
          and borderline["evidence_rule"]["verdict"] == "borderline"
          and none["evidence_rule"]["verdict"] == "no_evidence")
    return ok, (f"supported={supported['evidence_rule']['verdict']}, "
                f"borderline={borderline['evidence_rule']['verdict']}, "
                f"none={none['evidence_rule']['verdict']}")


@register("D0-7_bootstrap_deterministic")
def d0_7():
    vals = [random.Random(9).uniform(-1, 1) for _ in range(30)]
    ci1 = d0.bootstrap_ci_median(vals)
    ci2 = d0.bootstrap_ci_median(vals)
    ok = ci1 == ci2 and ci1[0] <= ci1[1]
    return ok, f"same seed -> identical CI {ci1}"


# -------------------------------- E3-A --------------------------------------

def track_from_centers(centers):
    frames = list(range(len(centers)))
    return {"track_id": 0, "category": "person", "frames": frames,
            "norm_centers": [[c[0], c[1]] for c in centers]}


def walk_stop_walk(n_pre=30, n_stop=20, n_post=15, speed=0.005):
    centers = []
    x = 0.4
    y = 0.6
    for _ in range(n_pre):
        x += speed
        centers.append((x, y))
    for _ in range(n_stop):
        centers.append((x, y))
    for _ in range(n_post):
        x += speed
        centers.append((x, y))
    return track_from_centers(centers)


@register("A-1_walk_stop_resume")
def a_1():
    res = e3a.analyze_track(walk_stop_walk(), dict(e3a.DEFAULT_CONFIG))
    labels = res["labels"]
    segs = [s["state"] for s in res["segments"]]
    order_ok = ("CROSSING" in segs and "STOPPED" in segs and "RESUMING" in segs
                and segs.index("CROSSING") < segs.index("STOPPED")
                < segs.index("RESUMING"))
    ok = (order_ok and res["summary"]["n_stop_events"] == 1
          and res["summary"]["n_resume_events"] == 1)
    return ok, (f"segments={segs}, stop_events=1, resume_events=1")


@register("A-2_stationary_from_start")
def a_2():
    centers = [(0.4 + 1e-6 * i, 0.6) for i in range(40)]
    res = e3a.analyze_track(track_from_centers(centers), dict(e3a.DEFAULT_CONFIG))
    segs = set(s["state"] for s in res["segments"])
    ok = "WAITING" in segs and segs <= {"UNKNOWN", "WAITING"}
    return ok, f"segments={sorted(segs)} (no CROSSING/STOPPED/RESUMING)"


@register("A-3_jitter_no_flapping")
def a_3():
    centers = []
    x, y = 0.4, 0.6
    for i in range(60):
        step = 0.004 if i % 2 == 0 else 0.0005
        x += step
        centers.append((x, y))
    res = e3a.analyze_track(track_from_centers(centers), dict(e3a.DEFAULT_CONFIG))
    segs = [s["state"] for s in res["segments"]]
    ok = "CROSSING" not in segs and "STOPPED" not in segs and "RESUMING" not in segs
    return ok, (f"alternating 0.004/0.0005 speeds -> no motion state; "
                f"segments={segs}")


@register("A-4_gap_yields_unknown")
def a_4():
    t = walk_stop_walk(n_pre=25, n_stop=0, n_post=25, speed=0.005)
    # remove frames 25..34 (create a 10-frame gap)
    centers = t["norm_centers"]
    kept = centers[:25] + centers[35:]
    frames = list(range(25)) + list(range(35, len(centers)))
    tr = {"track_id": 0, "category": "person", "frames": frames,
          "norm_centers": kept}
    res = e3a.analyze_track(tr, dict(e3a.DEFAULT_CONFIG))
    labels = res["labels"]
    has_unknown = "UNKNOWN" in labels[25:36]
    resumes = "CROSSING" in labels[-10:]
    ok = has_unknown and resumes
    return ok, "10-frame gap -> UNKNOWN around the gap, CROSSING re-established"


@register("A-5_real_tracks_smoke")
def a_5():
    tj = REPO_ROOT / "runs/synpedas_smoke_v2/video_0016_0_16_72b_f0-150__real/tracks.json"
    if not tj.exists():
        return False, f"missing {tj}"
    tracks = json.loads(tj.read_text())
    persons = [t for t in tracks if t.get("category") == "person"
               and len(t.get("frames", [])) >= 10]
    ok = bool(persons)
    for t in persons[:3]:
        res = e3a.analyze_track(t, dict(e3a.DEFAULT_CONFIG))
        ok = ok and set(res["labels"]) <= set(e3a.STATES)
        segs = res["segments"]
        ok = ok and all(s["n_frames_observed"] >= 1 for s in segs)
        ok = ok and all(s["n_frames_span"] >= s["n_frames_observed"] for s in segs)
        ok = ok and [s["start_frame"] for s in segs] == sorted(s["start_frame"] for s in segs)
        total = sum(s["n_frames_observed"] for s in segs)
        ok = ok and total == len(res["labels"])
    return ok, f"{len(persons)} real person tracks analyzed; states/segments valid"


@register("A-6_cli_contract")
def a_6():
    tmp = Path(tempfile.mkdtemp(prefix="e3a_selftest_"))
    try:
        script = str(Path(__file__).resolve().parent / "e3_a_event_state_baseline.py")
        out = tmp / "out"
        r = subprocess.run([sys.executable, "-I", script,
                            "--tracks-root", "runs/synpedas_smoke_v2",
                            "--out-dir", str(out)],
                           capture_output=True, text=True, cwd=str(REPO_ROOT))
        files_ok = all((out / n).exists() for n in
                       ("per_frame_phases.csv", "track_summaries.csv",
                        "summary.json", "run_log.txt"))
        n_rows = (sum(1 for _ in (out / "per_frame_phases.csv").open()) - 1
                  if files_ok else -1)
        summary = json.loads((out / "summary.json").read_text()) if files_ok else {}
        image_plane = (summary.get("observation_level") == "image_plane"
                       or summary.get("config", {}).get("observation_level") == "image_plane")
        ok = (r.returncode == 0 and files_ok and n_rows > 0 and image_plane)
        return ok, (f"CLI exit={r.returncode}, files={files_ok}, "
                    f"per_frame rows={n_rows}, image_plane declaration present")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@register("A-7_reestablish_is_sustained")
def a_7():
    # Regression for the independent audit: with reestablish=10, n_move=2,
    # the first nine usable post-gap observations must remain UNKNOWN even
    # though the motion run itself is already long enough.
    cfg = dict(e3a.DEFAULT_CONFIG)
    cfg.update({"reestablish": 10, "n_move": 2, "warmup": 1,
                "smooth_window": 1, "gap_tolerance": 4})
    frames = list(range(5)) + list(range(12, 27))
    centers = []
    x = 0.4
    for _ in frames:
        x += 0.01
        centers.append((x, 0.6))
    tr = {"track_id": 0, "category": "person", "frames": frames,
          "norm_centers": centers}
    res = e3a.analyze_track(tr, cfg)
    post = res["labels"][5:]
    # The first post-gap frame has no computable displacement, hence no
    # speed; the following nine usable observations must also stay UNKNOWN.
    ok = (post[0] == "UNKNOWN"
          and all(s == "UNKNOWN" for s in post[1:10])
          and post[10] == "CROSSING")
    return ok, f"post-gap phases={post}; 10 usable observations required"


@register("D0-8_cluster_bootstrap")
def d0_8():
    diags = []
    for source in range(12):
        for pair in range(20):
            diags.append({"delta_dir": 0.2 + source * 0.001,
                          "R_delta": 0.1 + source * 0.001,
                          "jaad_video": f"video_{source:04d}"})
    s = d0.summarize_diagnostics(diags)
    method = s["uncertainty"]["method"]
    ok = (method == "cluster_bootstrap_by_jaad_video"
          and s["uncertainty"]["n_clusters"] == 12
          and s["evidence_rule"]["verdict"] == "temporal_oversmoothing_supported")
    return ok, f"correlated pairs resampled by JAAD source: {method}, 12 clusters"


@register("D0-9_crop_controls_deterministic")
def d0_9():
    boxes = [[0, 0, 10, 20], [2, 0, 12, 20], [4, 0, 14, 20]]
    stable = d01.stabilize_boxes(boxes, window=3)
    j1 = d01.jitter_boxes(boxes, seed=42)
    j2 = d01.jitter_boxes(boxes, seed=42)
    ok = (stable[1] == [2.0, 0.0, 12.0, 20.0]
          and stable[0] == [1.0, 0.0, 11.0, 20.0]
          and j1 == j2 and j1 != boxes
          and all(b[2] > b[0] and b[3] > b[1] for b in j1))
    return ok, "stable crop averages neighbours; jitter is seeded, valid, target-only"


@register("D0-10_delta_shuffle_is_not_reversal")
def d0_10():
    vals = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
    a = d01.shuffled_deltas(vals, seed=9101)
    b = d01.shuffled_deltas(vals, seed=9101)
    ok = a == b and sorted(a) == vals and a != vals and a != list(reversed(vals))
    return ok, f"deterministic non-reversal order control: {a}"


# --------------------------------- main --------------------------------------
def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["E3 stage selftest",
             f"run at: {time.strftime('%Y-%m-%dT%H:%M:%S')}",
             ""]
    n_pass = 0
    for name, fn in TESTS:
        try:
            ok, detail = fn()
        except Exception as exc:  # noqa: BLE001
            ok, detail = False, f"exception: {type(exc).__name__}: {exc}"
        n_pass += int(ok)
        line = f"{name}: {'PASS' if ok else 'FAIL'} — {detail}"
        print(line)
        lines.append(line)
    summary = f"SUMMARY: {n_pass}/{len(TESTS)} PASS"
    print(summary)
    lines += ["", summary]
    (OUT_DIR / "selftest_results.txt").write_text("\n".join(lines) + "\n",
                                                  encoding="utf-8")
    print(f"[selftest] results -> {OUT_DIR / 'selftest_results.txt'}")
    return 0 if n_pass == len(TESTS) else 1


if __name__ == "__main__":
    sys.exit(main())
