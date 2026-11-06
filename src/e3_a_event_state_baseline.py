#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E3-A: no-JEPA event-state rule baseline over image-plane observations.

Task (docs/e3_event_state_jepa_plan.md §3): NOT to predict the future, but to
determine what the video actually shows.  This module consumes per-frame
person tracks (norm_centers from tracks.json, produced by wmeval
detect_track) and labels each frame with an event phase:

    UNKNOWN  warm-up insufficient / frame gap beyond tolerance /
             not yet re-established after a gap
    WAITING  stationary from the start (never observed moving)
    CROSSING moving (sustained speed above the moving threshold)
    STOPPED  stationary after having been moving
    RESUMING motion resumed after STOPPED (transition window)

Design constraints:
- hysteresis band (v_stop < v_move) + minimum-run counters so detection
  jitter cannot flap the state machine;
- frame gaps (occlusion / missed detections) -> UNKNOWN, then a
  re-establishment window before classification resumes;
- **image-plane declaration**: image-plane speed != ground-plane speed.
  Without ego-motion / camera-geometry compensation all stop/motion
  judgements here are image-observation-level evidence ONLY
  (`observation_level: "image_plane"` in every output).  Do not upgrade
  them to world-level event claims.

Defaults for the thresholds are derived from the observed norm_centers scale
(~5e-3 normalized units per frame while walking); they are rule parameters,
not research conclusions — any retuning against labels must be logged in
docs/failure_log.md.

CLI (stdlib only, any python):
    python3 -I src/e3_a_event_state_baseline.py \
        [--tracks-root runs/synpedas_smoke_v2] [--out-dir runs/e3_a_baseline]

Outputs: per_frame_phases.csv, track_summaries.csv, summary.json, run_log.txt
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACKS = REPO_ROOT / "runs/synpedas_smoke_v2"
DEFAULT_OUT = REPO_ROOT / "runs/e3_a_baseline"

STATES = ("UNKNOWN", "WAITING", "CROSSING", "STOPPED", "RESUMING")

DEFAULT_CONFIG = {
    "v_stop": 0.0015,        # normalized units / frame: below -> stationary
    "v_move": 0.0030,        # above -> moving (hysteresis band in between)
    "n_stop": 5,             # consecutive stationary frames to enter STOPPED
    "n_move": 5,             # consecutive moving frames to enter motion states
    "warmup": 3,             # frames required before any classification
    "gap_tolerance": 4,      # missing-frame run treated as a gap -> UNKNOWN
    "reestablish": 3,        # valid frames needed after a gap to classify again
    "resume_window": 6,      # frames labelled RESUMING after motion resumes
    "smooth_window": 3,      # centred moving average on speed
    "smoothing_temporal_scope": "centered_offline_noncausal",
    "observation_level": "image_plane",
    "phase_semantics": {
        "UNKNOWN": "insufficient or interrupted image-plane evidence",
        "WAITING": "image-plane stationary since first usable observation",
        "CROSSING": "IMAGE_PLANE_MOVING (not verified road crossing)",
        "STOPPED": "image-plane stationary after earlier image-plane motion",
        "RESUMING": "image-plane motion resumes after STOPPED",
    },
}


def code_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


# ------------------------------- kinematics ---------------------------------

def frame_observations(frames: Sequence[int],
                       norm_centers: Sequence[Sequence[float]]) -> List[dict]:
    """Sorted per-frame observations with gap detection (by frame index).

    Duplicate detections on the same frame are dropped (first occurrence
    kept) — real YOLO track JSONs occasionally contain them and they would
    otherwise produce zero-length segments.
    """
    pairs = sorted(zip([int(f) for f in frames],
                       [(float(c[0]), float(c[1])) for c in norm_centers]),
                   key=lambda t: t[0])
    obs = []
    for frame, (x, y) in pairs:
        if obs and obs[-1]["frame"] == frame:
            continue  # duplicate detection on this frame
        gap_from_prev = (frame - obs[-1]["frame"] - 1) if obs else 0
        obs.append({"frame": frame, "x": x, "y": y, "gap_from_prev": gap_from_prev})
    return obs


def compute_kinematics(obs: Sequence[dict], config: dict) -> List[dict]:
    """Per-frame smoothed speed and acceleration with gap flags."""
    speeds_raw = []
    for i, o in enumerate(obs):
        if i == 0 or o["gap_from_prev"] > 0:
            speeds_raw.append(None)  # undefined across a gap / at start
        else:
            dx = o["x"] - obs[i - 1]["x"]
            dy = o["y"] - obs[i - 1]["y"]
            speeds_raw.append(math.hypot(dx, dy))
    w = config["smooth_window"]
    half = w // 2
    recs = []
    for i, o in enumerate(obs):
        window = [s for s in speeds_raw[max(0, i - half):i + half + 1]
                  if s is not None]
        speed = sum(window) / len(window) if window else None
        recs.append({**o, "speed_raw": speeds_raw[i], "speed": speed})
    for i, r in enumerate(recs):
        if i == 0 or r["gap_from_prev"] > 0 or recs[i - 1]["speed"] is None:
            r["dspeed"] = None
        elif r["speed"] is None:
            r["dspeed"] = None
        else:
            r["dspeed"] = r["speed"] - recs[i - 1]["speed"]
    return recs


# ------------------------------ state machine -------------------------------

def classify_phases(kin: Sequence[dict], config: dict) -> List[str]:
    """Image-plane event phase per frame (hysteresis + gap handling).

    A gap larger than gap_tolerance starts a recovery interval.  The state
    remains UNKNOWN until `reestablish` CONSECUTIVE usable observations have
    accumulated after that gap.  This is deliberately independent of warmup:
    using reestablish only on the gap frame would let classification resume
    on the very next frame.
    """
    v_stop, v_move = config["v_stop"], config["v_move"]
    n_stop, n_move = config["n_stop"], config["n_move"]
    warmup, gap_tol, reest = config["warmup"], config["gap_tolerance"], config["reestablish"]
    resume_window = config["resume_window"]

    labels: List[str] = []
    state = "UNKNOWN"
    stationary_run = 0
    moving_run = 0
    valid_run = 0          # consecutive usable frames (warmup)
    recovery_valid = 0     # consecutive usable frames since the last large gap
    recovering = False
    ever_moved = False
    resume_countdown = 0

    for r in kin:
        gap = r["gap_from_prev"] > gap_tol
        if gap:
            state = "UNKNOWN"
            stationary_run = moving_run = 0
            valid_run = 0
            recovery_valid = 0
            recovering = True
            resume_countdown = 0

        sp = r["speed"]
        usable = sp is not None
        if usable:
            valid_run += 1
            if recovering:
                recovery_valid += 1
        else:
            valid_run = 0
            if recovering:
                recovery_valid = 0
            state = "UNKNOWN"

        if sp is not None:
            if sp >= v_move:
                moving_run += 1
                stationary_run = 0
            elif sp < v_stop:
                stationary_run += 1
                moving_run = 0
            # hysteresis band: runs unchanged

        if recovering and recovery_valid >= reest:
            recovering = False
        ready = usable and valid_run >= warmup and not recovering

        if ready:
            if moving_run >= n_move:
                if state == "STOPPED":
                    state = "RESUMING"
                    resume_countdown = resume_window
                    ever_moved = True
                elif state == "RESUMING":
                    pass
                else:
                    state = "CROSSING"
                    ever_moved = True
            elif stationary_run >= n_stop:
                if state in ("CROSSING", "RESUMING"):
                    state = "STOPPED"
                elif state == "UNKNOWN" and not ever_moved:
                    state = "WAITING"
        else:
            state = "UNKNOWN"

        if resume_countdown > 0 and ready:
            resume_countdown -= 1
            if resume_countdown == 0 and state == "RESUMING":
                state = "CROSSING"
        labels.append(state)
    return labels


def phase_segments(frames: Sequence[int], labels: Sequence[str]) -> List[dict]:
    """Segments of constant phase.  n_frames_span counts the frame-index
    span (gaps included); n_frames_observed counts actual observations —
    they differ exactly when the track contains missing frames."""
    segs = []
    cur_state, cur_start, cur_count, prev_f = None, None, 0, None
    for f, s in zip(frames, labels):
        if s != cur_state:
            if cur_state is not None:
                segs.append({"state": cur_state, "start_frame": cur_start,
                             "end_frame": prev_f,
                             "n_frames_span": prev_f - cur_start + 1,
                             "n_frames_observed": cur_count})
            cur_state, cur_start, cur_count = s, f, 0
        cur_count += 1
        prev_f = f
    if cur_state is not None:
        segs.append({"state": cur_state, "start_frame": cur_start,
                     "end_frame": prev_f,
                     "n_frames_span": prev_f - cur_start + 1,
                     "n_frames_observed": cur_count})
    return segs


def summarize_track(labels: Sequence[str]) -> dict:
    counts = {s: 0 for s in STATES}
    for s in labels:
        counts[s] = counts.get(s, 0) + 1
    stop_events = sum(1 for i in range(1, len(labels))
                      if labels[i] == "STOPPED" and labels[i - 1] != "STOPPED")
    resume_events = sum(1 for i in range(1, len(labels))
                        if labels[i] == "RESUMING" and labels[i - 1] != "RESUMING")
    return {"n_frames": len(labels), **{f"frames_{s.lower()}": counts[s] for s in STATES},
            "n_stop_events": stop_events, "n_resume_events": resume_events}


# --------------------------------- CLI run ----------------------------------

def analyze_track(track: dict, config: dict) -> dict:
    frames = track["frames"]
    centers = track["norm_centers"]
    obs = frame_observations(frames, centers)
    kin = compute_kinematics(obs, config)
    labels = classify_phases(kin, config)
    segs = phase_segments([r["frame"] for r in kin], labels)
    return {"kin": kin, "labels": labels, "segments": segs,
            "summary": summarize_track(labels)}


def run_cli(tracks_root: Path, out_dir: Path, config: dict) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    log = [f"started {time.strftime('%Y-%m-%dT%H:%M:%S')} code={code_sha256()[:12]}",
           f"tracks_root={tracks_root} config={json.dumps(config, sort_keys=True)}"]
    per_frame = []
    track_rows = []
    n_videos = n_tracks = n_person = 0
    for tj in sorted(tracks_root.glob("*__*/tracks.json")):
        side = "gen" if tj.parent.name.endswith("__gen") else "real"
        video = tj.parent.name.rsplit("__", 1)[0]
        tracks = json.loads(tj.read_text())
        n_videos += 1
        for t in tracks:
            n_tracks += 1
            if t.get("category") != "person":
                continue
            if len(t.get("frames", [])) < 2:
                continue
            n_person += 1
            res = analyze_track(t, config)
            for r, lab in zip(res["kin"], res["labels"]):
                per_frame.append({
                    "video": video, "side": side, "track_id": t["track_id"],
                    "frame": r["frame"], "x": round(r["x"], 6), "y": round(r["y"], 6),
                    "speed": None if r["speed"] is None else round(r["speed"], 6),
                    "gap_from_prev": r["gap_from_prev"], "phase": lab,
                    "observation_level": config["observation_level"],
                })
            s = res["summary"]
            track_rows.append({
                "video": video, "side": side, "track_id": t["track_id"],
                "n_frames": s["n_frames"], "n_segments": len(res["segments"]),
                **{k: v for k, v in s.items() if k != "n_frames"},
                "observation_level": config["observation_level"],
            })
        log.append(f"{tj.parent.name}: person tracks analyzed")

    with (out_dir / "per_frame_phases.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_frame[0].keys()))
        w.writeheader()
        w.writerows(per_frame)
    with (out_dir / "track_summaries.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(track_rows[0].keys()))
        w.writeheader()
        w.writerows(track_rows)
    summary = {
        "code_sha256": code_sha256(),
        "tracks_root": str(tracks_root),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": config,
        "n_videos": n_videos, "n_tracks_total": n_tracks, "n_person_tracks": n_person,
        "n_person_tracks_used": len(track_rows),
        "n_per_frame_rows": len(per_frame),
        "states": list(STATES),
        "declaration": "image-plane observation-level event evidence only; "
                       "ego-motion / camera-geometry NOT compensated",
        "label_acceptance": "research conclusions gated on e1_manual_labels.csv "
                            "human ground truth (currently pending)",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2),
                                          encoding="utf-8")
    log.append(f"finished: videos={n_videos} person_tracks={n_person} "
               f"per_frame_rows={len(per_frame)}")
    (out_dir / "run_log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    print(f"[E3-A] videos={n_videos} person_tracks={n_person} "
          f"per_frame_rows={len(per_frame)}")
    print(f"[E3-A] outputs -> {out_dir}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="E3-A event-state rule baseline")
    ap.add_argument("--tracks-root", default=str(DEFAULT_TRACKS))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = ap.parse_args(argv)
    return run_cli(Path(args.tracks_root), Path(args.out_dir), dict(DEFAULT_CONFIG))


if __name__ == "__main__":
    sys.exit(main())
