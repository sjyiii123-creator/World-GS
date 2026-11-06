#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build an explicit E3 input/provenance manifest (no raw media included)."""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/e3_reproducibility/input_manifest.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def record(path: Path, included: bool) -> dict:
    rel = path.relative_to(ROOT).as_posix()
    return {"path": rel, "exists": path.exists(),
            "bytes": path.stat().st_size if path.exists() else None,
            "sha256": sha256(path) if path.exists() else None,
            "included_in_e3_followup_zip": included}


def main() -> int:
    samples = ROOT / "datasets/processed/actor_sequences/samples.csv"
    used_samples = set()
    with (ROOT / "runs/e3_d0_diagnostics_v2/per_pair_diagnostics.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            used_samples.add(row["sample"])
    videos = sorted(ROOT / "repos/SynPeDAS/JAAD_crossing_moments" / f"{s}.mp4"
                    for s in used_samples)
    tracks = sorted((ROOT / "runs/synpedas_smoke_v2").glob("*__*/tracks.json"))
    for tj in tracks:
        video = tj.parent.name.rsplit("__", 1)[0]
        side = tj.parent.name.rsplit("__", 1)[1]
        base = ("JAAD_crossing_moments" if side == "real" else "generated_video")
        videos.append(ROOT / "repos/SynPeDAS" / base / f"{video}.mp4")
    code = [ROOT / "src" / n for n in (
        "e3_d0_temporal_diagnostics.py", "run_e3_d0_1_controls.py",
        "e3_a_event_state_baseline.py", "selftest_e3_stage.py",
        "build_e3_a_1_review.py")]
    checkpoints = [ROOT / "runs/d4_experimental/d4_lam0.5.pt",
                   ROOT / "runs/j2_2_experimental/b4_residual_expanded.pt",
                   ROOT / "ckpt/dinov2-small/model.safetensors",
                   ROOT / "ckpt/dinov2-small/config.json",
                   ROOT / "ckpt/dinov2-small/preprocessor_config.json",
                   ROOT / "ckpt/yolov10x.pt"]
    caches = [ROOT / "runs/j1_baseline/feature_cache/features.npz",
              ROOT / "runs/j1_baseline/feature_cache/meta.json"]
    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "scope": "E3-D0 v2, E3-D0.1 and E3-A.1 follow-up",
        "split_policy": "D0 follow-ups use train,val only; split=test excluded by default",
        "raw_media_included": False,
        "model_weights_included": False,
        "feature_cache_included": False,
        "note": ("The follow-up ZIP permits independent recomputation from result CSVs "
                 "and track observations. Full DINO/video inference additionally "
                 "requires the exact raw media, checkpoints and cache indexed here."),
        "environment": {"python_wm_eval": "3.9.23", "torch": "2.7.1+cu128",
                        "ultralytics": "8.4.202", "platform": platform.platform()},
        "code": [record(p, True) for p in code],
        "tabular_inputs": [record(samples, True)] + [record(p, True) for p in tracks],
        "checkpoints": [record(p, False) for p in checkpoints],
        "feature_cache": [record(p, False) for p in caches],
        "raw_media_index": [record(p, False) for p in videos],
        "pending_human_truth": [record(ROOT / "datasets/annotations/e1_manual_labels.csv", True)],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[E3 manifest] code={len(code)} tracks={len(tracks)} "
          f"raw_media_index={len(videos)} -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
