"""Minimal end-to-end experiment entry point.

Reads a small set of videos (real JAAD clips and/or SynPeDAS generated
videos), runs detection+tracking, computes the DrivingGen trajectory
consistency/quality formulas (px-space, clearly labelled), executes the
event rule battery and the uncertainty-aware adjudication, and writes
inspectable intermediate artifacts per video.

Usage
-----
    PYTHONPATH=src python -m wmeval.run_minimal \
        --videos path/a.mp4 path/b.mp4 --outdir runs/smoke

    PYTHONPATH=src python -m wmeval.run_minimal \
        --manifest datasets/annotations/manifest.csv \
        --num-samples 10 --outdir runs/synpedas_smoke

Outputs (per video, under outdir/<stem>/)
    meta.json         video facts (fps, frames, resolution — measured)
    detections.jsonl  one JSON per detection (track_id/category/bbox/conf/t)
    tracks.json       grouped tracks with normalized centers
    metrics.json      official-formula trajectory metrics (units: px)
    events.json       rule checks + disappearance attributions + verdict
    annotated.mp4     visualization with track boxes
    keyframes/        first/middle/last frame of each person track
    … and outdir/summary.csv across all processed videos
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from .detect_track import (DetectTrackPipeline, group_tracks,
                           save_detections_jsonl, save_tracks_json)
from .event_checks import DEFAULT_CROSSING_SPEC, run_rule_battery
from .traj_metrics import evaluate_trajectories
from .uncertainty import adjudicate, attribute_endings
from .video_io import VideoReader, write_annotated_video, probe


def _interrupt_regions(spec: dict | None = None) -> list[tuple]:
    """Actor anomaly inside these regions interrupts the in-progress event."""
    spec = spec or DEFAULT_CROSSING_SPEC
    regions = []
    for eid, c in spec.items():
        if isinstance(c, dict) and c.get("check") in ("region_entry",
                                                      "region_presence") \
                and "region" in c:
            regions.append(tuple(c["region"]))
    return regions


def _px_thresholds(meta) -> dict:
    """Adapt official unit-space thresholds to image space (documented).

    v_static   : 0.5 % of frame diagonal per second
    length_eps : 1.0 % of frame diagonal
    """
    diag = float(np.hypot(meta.width, meta.height))
    return {"v_static": round(0.005 * diag, 2),
            "length_eps": round(0.01 * diag, 1)}


def process_one_video(
    video_path: str,
    out_dir: Path,
    weights: str,
    imgsz: int,
    conf: float,
    target_fps: float | None,
    actor: str = "person",
    device: int | str = 0,
    keyframes: bool = True,
) -> dict:
    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = probe(video_path)
    meta_facts = meta.to_dict()
    meta_facts["fps_reported_by_container"] = getattr(meta, "fps_reported", True)

    # ---- detection + tracking -------------------------------------------
    reader = VideoReader(video_path, target_fps=target_fps)
    pipe = DetectTrackPipeline(weights=weights, conf=conf, imgsz=imgsz,
                               device=device)
    detections = pipe.run(reader)
    tracks = group_tracks(detections)
    actor_tracks = [t for t in tracks if t.category == actor]

    save_detections_jsonl(detections, out_dir / "detections.jsonl")
    save_tracks_json(tracks, out_dir / "tracks.json")

    # ---- effective analysis rate ----------------------------------------
    # target_fps resampling: VideoReader yields native indices but at a
    # reduced rate.  Three things must follow the ANALYZED rate, not the
    # container rate (E2 phase-6 audit; latent bug — E1 ran with
    # target_fps=None so v3.1 outputs are unaffected):
    #   * trajectory dt (detection spacing is 1/target_fps, not 1/fps)
    #   * annotated-video playback fps (fewer frames written)
    #   * detection-coverage denominator (frames actually analyzed)
    analysis_fps = target_fps if (target_fps and target_fps > 0) else meta.fps
    if target_fps and meta.fps > 0:
        step = max(1.0, meta.fps / target_fps)
        n_analyzed = int(round((meta.num_frames - 1) / step)) + 1
    else:
        n_analyzed = meta.num_frames

    # ---- event rule battery ----------------------------------------------
    checks, disappearances = run_rule_battery(tracks, meta.num_frames)
    endings = attribute_endings(tracks, disappearances)
    verdict = adjudicate(checks, endings, detections, n_analyzed,
                         actor=actor, interrupt_regions=_interrupt_regions())

    # ---- official trajectory metric formulas (px space) ------------------
    # NOTE: pixel centers + pixel-unit thresholds (diag-scaled) — consistent
    # units. The exp(-sigma/mu) terms are scale-invariant, so cross-resolution
    # comparison is not distorted; only v_static/length_eps are absolute and
    # are scaled per-video by frame diagonal.
    # dt follows the ANALYZED frame spacing (target_fps), not container fps.
    dt = 1.0 / analysis_fps
    thr = _px_thresholds(meta)
    trajs = [t.centers for t in actor_tracks if t.length >= 5]
    metrics = evaluate_trajectories(
        trajs, dt=dt, units="px", v_static=thr["v_static"],
        length_eps=thr["length_eps"])
    metrics["thresholds_px"] = thr
    metrics["note"] = ("formula-faithful port on image-space (pixel) tracks; "
                       "NOT comparable with paper numbers computed on "
                       "world-coordinate meter trajectories")

    events = {
        "video": str(video_path),
        "num_frames": meta.num_frames,
        "fps": meta.fps,
        "checks": [c.to_dict() for c in checks],
        "disappearances": [d.to_dict() for d in disappearances],
        "ending_attributions": [e.to_dict() for e in endings],
        "verdict": verdict.to_dict(),
    }
    (out_dir / "events.json").write_text(
        json.dumps(events, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "meta.json").write_text(
        json.dumps(meta_facts, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- visualization ----------------------------------------------------
    try:
        frames_with_boxes = []
        det_by_frame: dict[int, list] = {}
        for d in detections:
            det_by_frame.setdefault(d.frame_index, []).append(d)
        for idx, _ts, frame in VideoReader(video_path,
                                           target_fps=target_fps).iter_frames():
            boxes = [{"track_id": d.track_id, "bbox": d.bbox,
                      "label": d.category, "conf": d.conf}
                     for d in det_by_frame.get(idx, [])]
            frames_with_boxes.append((frame, boxes))
        if frames_with_boxes:
            write_annotated_video(out_dir / "annotated.mp4", frames_with_boxes,
                                  fps=analysis_fps)
        if keyframes:
            for t in actor_tracks:
                kf_idx = sorted(set(
                    [t.detections[0].frame_index,
                     t.detections[len(t.detections) // 2].frame_index,
                     t.detections[-1].frame_index]))
                VideoReader(video_path).save_keyframes(
                    out_dir / "keyframes", kf_idx,
                    prefix=f"track{t.track_id:02d}")
    except Exception as e:  # noqa: BLE001 — visualization is best-effort
        events["visualization_error"] = str(e)

    elapsed = time.perf_counter() - t0
    return {
        "video": str(video_path),
        "stem": Path(video_path).stem,
        "num_frames": meta.num_frames,
        "fps": meta.fps,
        "resolution": meta.resolution,
        "num_detections": len(detections),
        "num_tracks": len(tracks),
        "num_actor_tracks": len(actor_tracks),
        "verdict": verdict.verdict,
        "verdict_reasons": "; ".join(verdict.reasons),
        "person_missing": bool(verdict.reasons and "no pedestrian found" in
                               verdict.reasons[0]),
        "traj_consistency": metrics.get("aggregate", {}).get(
            "velocity_accel_consistency"),
        "traj_comfort": metrics.get("aggregate", {}).get("comfort"),
        "elapsed_s": round(elapsed, 1),
    }


def _load_manifest_samples(manifest_csv: str, num_samples: int,
                           require_complete: bool = True) -> list[dict]:
    rows = []
    with open(manifest_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if require_complete and (not r["source_video"]
                                     or not r["generated_video"]):
                continue
            rows.append(r)
    rows = rows[:num_samples]
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", nargs="*", default=[], help="video files")
    ap.add_argument("--manifest", default=None,
                    help="manifest.csv from wmeval.build_manifest")
    ap.add_argument("--num-samples", type=int, default=10)
    ap.add_argument("--sides", choices=["real", "gen", "both"], default="both")
    ap.add_argument("--outdir", default="runs/minimal")
    ap.add_argument("--weights", default="yolov10x.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--fps", type=float, default=None,
                    help="analysis fps cap (None = native)")
    ap.add_argument("--device", default="0")
    ap.add_argument("--no-keyframes", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[str, Path]] = []

    for v in args.videos:
        jobs.append((v, outdir / Path(v).stem))
    if args.manifest:
        rows = _load_manifest_samples(args.manifest, args.num_samples)
        print(f"[run] {len(rows)} manifest samples loaded")
        for r in rows:
            sid = r["sample_id"]
            if args.sides in ("real", "both") and r["source_video"]:
                jobs.append((r["source_video"], outdir / f"{sid}__real"))
            if args.sides in ("gen", "both") and r["generated_video"]:
                jobs.append((r["generated_video"], outdir / f"{sid}__gen"))

    if not jobs:
        ap.error("no inputs: pass --videos and/or --manifest")

    summary = []
    for video, od in jobs:
        print(f"[run] processing {video} -> {od}")
        try:
            summary.append(process_one_video(
                video, od, args.weights, args.imgsz, args.conf,
                args.fps, device=args.device,
                keyframes=not args.no_keyframes))
            print(f"[run] done: {summary[-1]['verdict']} "
                  f"({summary[-1]['elapsed_s']}s)")
        except Exception as e:  # noqa: BLE001 — keep batch alive, log failure
            print(f"[run] FAILED {video}: {e}")
            summary.append({"video": video, "stem": Path(video).stem,
                            "error": str(e)})

    if summary:
        keys = sorted({k for s in summary for k in s})
        with open(outdir / "summary.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, restval="")
            w.writeheader()
            w.writerows(summary)
        (outdir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[run] summary -> {outdir/'summary.csv'}")
        ok = [s for s in summary if "error" not in s]
        verdicts = {}
        for s in ok:
            verdicts[s.get("verdict", "?")] = verdicts.get(s.get("verdict", "?"), 0) + 1
        print(f"[run] processed {len(ok)}/{len(summary)} videos; "
              f"verdicts: {verdicts}")


if __name__ == "__main__":
    main()
