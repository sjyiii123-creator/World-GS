#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E3-D0.1: delta-time sensitivity and target-crop controls.

This is a follow-up diagnostic, not an event-recognition result.  For the
same history and future times it compares:
  - predictions with the actual per-frame deltas;
  - predictions with every delta forced to zero;
  - predictions with deltas deterministically shuffled (not merely reversed).
It also compares DINOv2 target features extracted from the original boxes,
temporally smoothed boxes, and deterministic translated/scaled boxes.

The three crop variants are TARGET-ONLY controls: model inputs (history and
deltas) are unchanged.  split=test is excluded by default.  Uncertainty is
cluster-bootstrapped by jaad_video.  No threshold is tuned here and no claim
is made about STOPPED/RESUMING events until human-aligned windows exist.

Run (wm_eval env):
  /home/moz/miniconda3/envs/wm_eval/bin/python -I src/run_e3_d0_1_controls.py \
      --stride 20 --out-dir runs/e3_d0_1_controls
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e3_d0_temporal_diagnostics as d0

REPO_ROOT = Path(__file__).resolve().parents[1]


def stabilize_boxes(boxes: Sequence[Sequence[float]], window: int = 3) -> List[List[float]]:
    """Centred offline moving-average box coordinates (target-only control)."""
    half = window // 2
    out = []
    for i in range(len(boxes)):
        chunk = boxes[max(0, i - half):min(len(boxes), i + half + 1)]
        out.append([sum(b[j] for b in chunk) / len(chunk) for j in range(4)])
    return out


def jitter_boxes(boxes: Sequence[Sequence[float]], seed: int) -> List[List[float]]:
    """Deterministic small box translation/scale perturbation (target only)."""
    rng = random.Random(seed)
    out = []
    for b in boxes:
        x1, y1, x2, y2 = map(float, b)
        w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
        scale = rng.uniform(0.95, 1.05)
        dx, dy = rng.uniform(-0.02, 0.02) * w, rng.uniform(-0.02, 0.02) * h
        nw, nh = w * scale, h * scale
        cx, cy = (x1 + x2) / 2 + dx, (y1 + y2) / 2 + dy
        out.append([max(0.0, cx - nw / 2), max(0.0, cy - nh / 2),
                    cx + nw / 2, cy + nh / 2])
    return out


def shuffled_deltas(deltas: Sequence[float], seed: int) -> List[float]:
    """Deterministically shuffle time arguments without reversing them.

    Reversing would preserve the adjacent unordered pairs and therefore make
    both C and D diagnostics trivially identical.  A non-reversal shuffle is
    required for an order-sensitivity control.
    """
    out = list(deltas)
    random.Random(seed).shuffle(out)
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def median_ci(diags: Sequence[dict], field: str) -> dict:
    vals = [float(x[field]) for x in diags]
    return {"median": statistics.median(vals),
            "cluster_bootstrap_median_ci95": list(
                d0.cluster_bootstrap_ci_median(diags, field))}


def run(models: Sequence[str], stride: int, limit: Optional[int], out_dir: Path,
        samples_csv: Path, splits: Sequence[str]) -> int:
    import numpy as np
    import torch
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from wmeval.latent_features import LatentFeatureExtractor, CropConfig
    from wmeval.latent_world_state.residual_predictor import ResidualPredictor

    out_dir.mkdir(parents=True, exist_ok=True)
    log = [f"started {time.strftime('%Y-%m-%dT%H:%M:%S')}",
           f"models={list(models)} stride={stride} splits={list(splits)} limit={limit}",
           "controls: actual deltas / zero deltas / shuffled deltas; "
           "target boxes: original / temporally stabilized / deterministic jitter",
           "split=test excluded; descriptive controls, no event-performance claim"]
    predictors = {}
    for name in models:
        m = ResidualPredictor()
        ckpt = (REPO_ROOT / "runs/d4_experimental/d4_lam0.5.pt" if name == "d4"
                else REPO_ROOT / "runs/j2_2_experimental/b4_residual_expanded.pt")
        m.load_state_dict(torch.load(ckpt, map_location="cpu"))
        m.eval()
        predictors[name] = (m, ckpt)
    ex = LatentFeatureExtractor(model_id="ckpt/dinov2-small", crop_cfg=CropConfig(),
                                cache_dir="runs/j1_baseline/feature_cache")
    rows = list(csv.DictReader(samples_csv.open(newline="")))
    eligible = [r for r in rows if r["label"] == "same_identity"
                and r["split"] in set(splits)]
    pairs = eligible[::max(1, stride)]
    if limit:
        pairs = pairs[:limit]
    log.append(f"eligible={len(eligible)} selected={len(pairs)} (deterministic stride)")
    output = []
    skipped = 0
    with torch.no_grad():
        for r in pairs:
            sid = r["synpedas_sample"]
            f0 = int(sid.split("_f")[-1].split("-")[0])
            vp = REPO_ROOT / "repos/SynPeDAS/JAAD_crossing_moments" / f"{sid}.mp4"
            fps = float(r["fps"]) or 30.0
            anchor = d0.parse_frame_list(r["anchor_frames"])
            future = d0.parse_frame_list(r["cand_frames"])
            if len(anchor) < 2 or len(future) < 2:
                skipped += 1
                continue
            boxes = [b for _, b in future]
            variants = {"original": boxes, "stable": stabilize_boxes(boxes),
                        "jitter": jitter_boxes(boxes, seed=int(r["pair_id"]) + 4100)}
            try:
                hist = [ex.frame_feature(str(vp), f - f0, b, sid, r["anchor_ped_id"])
                        for f, b in anchor]
                targets = {
                    key: [ex.frame_feature(str(vp), f - f0, b, sid, r["cand_ped_id"])
                          for (f, _), b in zip(future, bs)]
                    for key, bs in variants.items()
                }
            except RuntimeError:
                skipped += 1
                continue
            hist_t = torch.from_numpy(np.stack(hist).astype(np.float32))
            deltas = [(f - anchor[-1][0]) / fps for f, _ in future]
            for name, (model, ckpt) in predictors.items():
                pred_varied = [model(hist_t[None], torch.tensor([dt])).numpy()[0]
                               for dt in deltas]
                pred_zero = [model(hist_t[None], torch.tensor([0.0])).numpy()[0]
                             for _ in deltas]
                pred_shuffled = [model(hist_t[None], torch.tensor([dt])).numpy()[0]
                                 for dt in shuffled_deltas(deltas, int(r["pair_id"]) + 9100)]
                rec = {
                    "pair_id": r["pair_id"], "split": r["split"],
                    "jaad_video": r["jaad_video"], "sample": sid, "model": name,
                    "D_pred_varied": d0.mean_adjacent_change(pred_varied),
                    "D_pred_zero_delta": d0.mean_adjacent_change(pred_zero),
                    "D_pred_shuffled_delta": d0.mean_adjacent_change(pred_shuffled),
                    "C_pred_varied": d0.mean_pairwise_cosine(pred_varied),
                    "C_pred_zero_delta": d0.mean_pairwise_cosine(pred_zero),
                    "C_pred_shuffled_delta": d0.mean_pairwise_cosine(pred_shuffled),
                }
                for key, target in targets.items():
                    rec[f"D_true_{key}"] = d0.mean_adjacent_change(target)
                    rec[f"C_true_{key}"] = d0.mean_pairwise_cosine(target)
                rec["R_delta_original"] = (rec["D_pred_varied"] /
                                            max(rec["D_true_original"], d0.EPS))
                rec["delta_vs_zero_ratio"] = (rec["D_pred_varied"] /
                                               max(rec["D_pred_zero_delta"], d0.EPS))
                rec["target_stable_ratio"] = (rec["D_true_stable"] /
                                               max(rec["D_true_original"], d0.EPS))
                rec["target_jitter_ratio"] = (rec["D_true_jitter"] /
                                               max(rec["D_true_original"], d0.EPS))
                output.append({k: (round(v, 8) if isinstance(v, float) else v)
                               for k, v in rec.items()})
            log.append(f"pair {r['pair_id']} done; skipped={skipped}")
    ex.flush_cache()
    if not output:
        raise RuntimeError("no pairs evaluated")
    with (out_dir / "per_pair_controls.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(output[0]))
        w.writeheader()
        w.writerows(output)
    for name, (_, ckpt) in predictors.items():
        subset = [r for r in output if r["model"] == name]
        summary = {
            "model": name, "n_pairs": len(subset), "n_skipped": skipped,
            "code_sha256": d0.code_sha256(),
            "control_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "checkpoint": str(ckpt), "checkpoint_sha256": sha256_file(ckpt),
            "samples_csv": str(samples_csv), "samples_sha256": sha256_file(samples_csv),
            "split_policy": {"included": list(splits), "test_excluded": "test" not in splits,
                             "stride": stride, "eligible_pairs": len(eligible)},
            "metrics": {field: median_ci(subset, field) for field in (
                "D_pred_varied", "D_pred_zero_delta", "D_pred_shuffled_delta",
                "D_true_original", "D_true_stable", "D_true_jitter",
                "R_delta_original", "delta_vs_zero_ratio",
                "target_stable_ratio", "target_jitter_ratio")},
            "interpretation_limits": [
                "delta_vs_zero_ratio and shuffled-delta controls measure sensitivity "
                "to the time argument and its order, not event information retention.",
                "stable/jitter controls measure target-feature crop sensitivity only.",
                "No STOPPED/RESUMING event conclusion without human-aligned windows.",
            ],
        }
        (out_dir / f"summary_{name}.json").write_text(json.dumps(summary, indent=2),
                                                       encoding="utf-8")
        log.append(f"model {name}: median R_delta_original="
                   f"{summary['metrics']['R_delta_original']['median']:.5f}, "
                   f"delta_vs_zero={summary['metrics']['delta_vs_zero_ratio']['median']:.4f}, "
                   f"jitter_ratio={summary['metrics']['target_jitter_ratio']['median']:.4f}")
    log.append(f"finished {time.strftime('%Y-%m-%dT%H:%M:%S')}; skipped={skipped}")
    (out_dir / "run_log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    for line in log:
        if line.startswith(("model ", "eligible=", "finished")):
            print(f"[E3-D0.1] {line}")
    print(f"[E3-D0.1] outputs -> {out_dir}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="E3-D0.1 time/crop controls")
    ap.add_argument("--samples", default=str(d0.DEFAULT_SAMPLES))
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "runs/e3_d0_1_controls"))
    ap.add_argument("--models", default="d4,b4")
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--include-test", action="store_true")
    args = ap.parse_args(argv)
    splits = ("train", "val", "test") if args.include_test else ("train", "val")
    return run([m.strip() for m in args.models.split(",") if m.strip()],
               args.stride, args.limit, Path(args.out_dir), Path(args.samples), splits)


if __name__ == "__main__":
    sys.exit(main())
