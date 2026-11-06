#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build blind-safe E3-VLM visual evidence from local real/generated videos.

Full-video sheets are neutral (frame index + timestamp only) and contain no
prompt or E3-A labels.  Candidate-context sheets are isolated and marked
blind_safe=false so they are not consumed during V2 blind observation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/e3_vlm_assisted_review"
CONFIG = {
    "full_video_frames_per_sheet": 12,
    "full_video_sheets_per_video": 3,
    "full_video_tile": [480, 270],
    "candidate_context_frames": 24,
    "candidate_buffer_frames": 15,
    "candidate_tile": [320, 180],
    "image_format": "jpg",
    "jpeg_quality": 92,
    "time_alignment": "none; per-video frame_index/fps only",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sample_ids() -> list[str]:
    with (ROOT / "datasets/annotations/e1_manual_labels.csv").open(newline="") as f:
        return sorted({r["sample_id"] for r in csv.DictReader(f)})


def manifest_rows() -> dict:
    with (ROOT / "datasets/annotations/manifest.csv").open(newline="") as f:
        return {r["sample_id"]: r for r in csv.DictReader(f)}


def sheet(frames: list[tuple[int, np.ndarray]], fps: float, tile: tuple[int, int],
          cols: int) -> np.ndarray:
    tw, th = tile
    rows = (len(frames) + cols - 1) // cols
    canvas = np.zeros((rows * th, cols * tw, 3), dtype=np.uint8)
    for i, (idx, image) in enumerate(frames):
        thumb = cv2.resize(image, (tw, th))
        cv2.rectangle(thumb, (0, 0), (tw, 24), (0, 0, 0), -1)
        cv2.putText(thumb, f"f={idx} t={idx / fps:.3f}s", (6, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        canvas[(i // cols) * th:(i // cols + 1) * th,
               (i % cols) * tw:(i % cols + 1) * tw] = thumb
    return canvas


def contiguous_candidates() -> list[dict]:
    # One ID per contiguous phase segment; STOPPED and following RESUMING
    # are separate events.  This uses the unit-tested implementation.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from e3_vlm_review_utils import event_segments
    with (ROOT / "runs/e3_a_1_baseline/per_frame_phases.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    return [{"event_id": e["event_id"], "video": e["video"], "side": e["side"],
             "track_id": e["track_id"], "phase": e["phase"],
             "start": e["start_frame"], "end": e["end_frame"]}
            for e in event_segments(rows)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    samples = sample_ids()
    manifests = manifest_rows()
    inventory = []
    evidence = []
    skipped = []
    candidate_events = contiguous_candidates()
    for sid in samples:
        m = manifests[sid]
        for side, key in (("real", "source_video"), ("generated_video", "generated_video")):
            source_type = "real" if side == "real" else "gen"
            video = ROOT / m[key]
            rec = {"sample_id": sid, "source_type": source_type,
                   "video_path": m[key], "exists": video.exists()}
            if not video.exists():
                rec["status"] = "MEDIA_UNAVAILABLE"
                skipped.append(rec)
                inventory.append(rec)
                continue
            cap = cv2.VideoCapture(str(video))
            if not cap.isOpened():
                rec["status"] = "MEDIA_UNAVAILABLE"
                skipped.append(rec)
                inventory.append(rec)
                continue
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_sha = sha256(video)
            rec.update({"status": "OK", "fps": fps, "num_frames": count,
                        "width": width, "height": height,
                        "duration_seconds": count / fps if fps else None,
                        "video_sha256": video_sha})
            inventory.append(rec)
            frames = {}
            wanted = set(range(count))
            for idx in wanted:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, image = cap.read()
                if ok:
                    frames[idx] = image
            cap.release()
            if len(frames) != count:
                rec["status"] = "MEDIA_DECODE_INCOMPLETE"
                rec["decoded_frames"] = len(frames)
                skipped.append(rec)
                continue
            per_sheet = CONFIG["full_video_frames_per_sheet"]
            total = per_sheet * CONFIG["full_video_sheets_per_video"]
            indices = sorted({round(i * (count - 1) / (total - 1)) for i in range(total)})
            for sheet_idx in range(CONFIG["full_video_sheets_per_video"]):
                chosen = indices[sheet_idx * per_sheet:(sheet_idx + 1) * per_sheet]
                image = sheet([(i, frames[i]) for i in chosen], fps,
                              tuple(CONFIG["full_video_tile"]), 3)
                eid = f"{sid}__{source_type}__fullseq_{sheet_idx:02d}"
                path = out / "blind_full_video" / f"{eid}.jpg"
                path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY,
                                               CONFIG["jpeg_quality"]])
                evidence.append({
                    "evidence_id": eid, "sample_id": sid, "source_type": source_type,
                    "path": path.relative_to(ROOT).as_posix(), "blind_safe": True,
                    "kind": "full_video_uniform_contact_sheet",
                    "frame_indices": chosen,
                    "timestamps_seconds": [round(i / fps, 6) for i in chosen],
                    "original_video_sha256": video_sha,
                    "image_sha256": sha256(path), "extraction_config": CONFIG,
                })
            # V0: exact consecutive transition frames, without boxes or labels.
            if sid == "video_0016_0_16_72b_f0-150" and source_type == "gen":
                for idx in (54, 55, 62, 63):
                    path = out / "pilot_v0" / f"{sid}__gen__f{idx:03d}.jpg"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(path), frames[idx])
                    evidence.append({"evidence_id": path.stem, "sample_id": sid,
                                     "source_type": source_type,
                                     "path": path.relative_to(ROOT).as_posix(),
                                     "blind_safe": True, "kind": "raw_single_frame",
                                     "frame_indices": [idx],
                                     "timestamps_seconds": [round(idx / fps, 6)],
                                     "original_video_sha256": video_sha,
                                     "image_sha256": sha256(path),
                                     "extraction_config": CONFIG})
    # Candidate contexts remain outside blind_full_video.
    for event in candidate_events:
        sid, source_type = event["video"], event["side"]
        inv = next((x for x in inventory if x["sample_id"] == sid
                    and x["source_type"] == source_type), None)
        if not inv or inv.get("status") != "OK":
            continue
        video = ROOT / inv["video_path"]
        cap = cv2.VideoCapture(str(video))
        fps = inv["fps"]
        lo = max(0, event["start"] - CONFIG["candidate_buffer_frames"])
        hi = min(inv["num_frames"] - 1, event["end"] + CONFIG["candidate_buffer_frames"])
        chosen = sorted({round(lo + i * (hi - lo) /
                               (CONFIG["candidate_context_frames"] - 1))
                         for i in range(CONFIG["candidate_context_frames"])})
        images = []
        for idx in chosen:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, image = cap.read()
            if ok:
                images.append((idx, image))
        cap.release()
        image = sheet(images, fps, tuple(CONFIG["candidate_tile"]), 4)
        path = out / "candidate_context" / f"{event['event_id']}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), image)
        evidence.append({"evidence_id": event["event_id"], "sample_id": sid,
                         "source_type": source_type,
                         "path": path.relative_to(ROOT).as_posix(), "blind_safe": False,
                         "kind": "candidate_context_with_buffer",
                         "candidate_start": event["start"], "candidate_end": event["end"],
                         "context_start": lo, "context_end": hi,
                         "frame_indices": [i for i, _ in images],
                         "timestamps_seconds": [round(i / fps, 6) for i, _ in images],
                         "original_video_sha256": inv["video_sha256"],
                         "image_sha256": sha256(path), "extraction_config": CONFIG})
    (out / "media_inventory.json").write_text(json.dumps({
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "num_records": len(inventory), "records": inventory}, indent=2))
    (out / "evidence_manifest.json").write_text(json.dumps({
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "num_records": len(evidence), "records": evidence}, indent=2))
    (out / "run_manifest.json").write_text(json.dumps({
        "script": "src/build_e3_vlm_evidence.py",
        "script_sha256": sha256(Path(__file__)),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "extraction_config": CONFIG, "inventory_records": len(inventory),
        "evidence_records": len(evidence), "unavailable_or_incomplete": skipped,
        "candidate_context_events": len(candidate_events),
        "original_labels_modified": False}, indent=2))
    print(f"[E3-VLM evidence] videos={len(inventory)} evidence={len(evidence)} "
          f"candidate_events={len(candidate_events)} unavailable={len(skipped)} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
