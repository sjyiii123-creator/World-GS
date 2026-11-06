#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E3-D0: temporal-variation / over-smoothing diagnostics for the predictors.

Motivation (docs/jepa_paper_audit.md §3, docs/e3_event_state_jepa_plan.md §2):
WA-JEPA's temporal diagnostics are worth borrowing, but a Participation-Ratio
decline does NOT establish temporal over-smoothing.  What is needed is
per-timestep evidence: predicted future sequences vs TRUE future sequences.

Diagnostics (defined within OUR protocol only — 384-d pooled DINOv2 features
and deterministic predictors; numbers are NOT comparable to WA-JEPA's):

    C(z)    = mean cosine similarity between all distinct-timestep pairs
    D(z)    = mean L2 change between adjacent timesteps
    Δ_dir   = C(pred) - C(true)        > 0  -> predicted futures collapse
                                         toward each other (directional)
    R_Δ     = D(pred) / max(D(true), eps)   << 1 -> predicted temporal
                                         change magnitude is insufficient

Pre-registered reading rule (fixed BEFORE any run; see plan §2):
    supported  : median(Δ_dir) > 0 AND median(R_Δ) < 0.5, both bootstrap
                 95% CIs excluding the critical value
    borderline : exactly one of the two holds
    no_evidence: neither holds -> must NOT motivate multi-future prediction
                 or temporal-loss changes on this basis alone

Real run (needs the wm_eval conda env: torch + DINOv2 + D4/B4 checkpoints):

    /home/moz/miniconda3/envs/wm_eval/bin/python -I \
        src/e3_d0_temporal_diagnostics.py [--limit N] [--models d4,b4]

Data contract: datasets/processed/actor_sequences/samples.csv rows with
label == "same_identity" — anchor_frames (12 past boxes of the actor) and
cand_frames (12 future boxes of the SAME actor).  Future window is
[gap, gap+12) frames (~0.5-0.9 s at ~30 fps): narrow but >= 2 timesteps,
which is the minimum the diagnostic requires.

Pure functions (this section) are stdlib-only and testable without torch.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

EPS = 1e-8
R_DELTA_EVIDENCE_BAR = 0.5   # pre-registered: R_Δ "clearly below 1"
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20261106

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLES = REPO_ROOT / "datasets/processed/actor_sequences/samples.csv"
DEFAULT_OUT = REPO_ROOT / "runs/e3_d0_diagnostics"


class InsufficientTimesteps(Exception):
    """Temporal diagnostics need >= 2 timesteps."""


# ------------------------------ pure functions ------------------------------

def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na < EPS and nb < EPS:
        return 1.0  # two zero vectors are identical
    if na < EPS or nb < EPS:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (na * nb)


def mean_pairwise_cosine(features: Sequence[Sequence[float]]) -> float:
    """C(z): average cosine similarity across distinct timesteps."""
    k = len(features)
    if k < 2:
        raise InsufficientTimesteps(f"need >=2 timesteps, got {k}")
    vals = [cosine_similarity(features[i], features[j])
            for i in range(k) for j in range(i + 1, k)]
    return sum(vals) / len(vals)


def mean_adjacent_change(features: Sequence[Sequence[float]]) -> float:
    """D(z): average L2 change between adjacent timesteps."""
    k = len(features)
    if k < 2:
        raise InsufficientTimesteps(f"need >=2 timesteps, got {k}")
    dists = [math.sqrt(sum((x - y) ** 2 for x, y in zip(features[i], features[i + 1])))
             for i in range(k - 1)]
    return sum(dists) / len(dists)


def temporal_diagnostics(pred: Sequence[Sequence[float]],
                         true: Sequence[Sequence[float]]) -> dict:
    """Δ_dir and R_Δ for one (predicted, true) future sequence pair.

    All outputs are coerced to plain Python floats so results remain
    JSON-serializable even when the inputs are numpy arrays."""
    c_pred = mean_pairwise_cosine(pred)
    c_true = mean_pairwise_cosine(true)
    d_pred = mean_adjacent_change(pred)
    d_true = mean_adjacent_change(true)
    return {
        "n_steps_pred": len(pred),
        "n_steps_true": len(true),
        "C_pred": float(c_pred),
        "C_true": float(c_true),
        "D_pred": float(d_pred),
        "D_true": float(d_true),
        "delta_dir": float(c_pred - c_true),
        "R_delta": float(d_pred / max(d_true, EPS)),
    }


def bootstrap_ci_median(values: Sequence[float], n_boot: int = N_BOOTSTRAP,
                        seed: int = BOOTSTRAP_SEED) -> Tuple[float, float]:
    """Percentile bootstrap 95% CI of the median (deterministic seed)."""
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    meds = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(len(values))] for _ in values]
        meds.append(statistics.median(sample))
    meds.sort()
    lo_idx = max(0, math.floor(0.025 * len(meds)))
    hi_idx = min(len(meds) - 1, math.floor(0.975 * len(meds)))
    return (meds[lo_idx], meds[hi_idx])


def cluster_bootstrap_ci_median(diags: Sequence[dict], field: str,
                                cluster_key: str = "jaad_video",
                                n_boot: int = N_BOOTSTRAP,
                                seed: int = BOOTSTRAP_SEED) -> Tuple[float, float]:
    """Bootstrap whole JAAD source videos, not individual correlated pairs.

    Each resample draws n_clusters source IDs with replacement and includes
    every pair belonging to the selected sources.  This preserves within-
    source correlation and avoids the optimistic pair-bootstrap interval.
    """
    groups = {}
    for d in diags:
        groups.setdefault(d[cluster_key], []).append(float(d[field]))
    sources = sorted(groups)
    if not sources:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    meds = []
    for _ in range(n_boot):
        vals = []
        for _ in sources:
            vals.extend(groups[sources[rng.randrange(len(sources))]])
        meds.append(statistics.median(vals))
    meds.sort()
    lo_idx = max(0, math.floor(0.025 * len(meds)))
    hi_idx = min(len(meds) - 1, math.floor(0.975 * len(meds)))
    return (meds[lo_idx], meds[hi_idx])


def summarize_diagnostics(diags: Sequence[dict]) -> dict:
    """Aggregate per-pair diagnostics and apply the pre-registered rule."""
    if not diags:
        return {"n_pairs": 0, "verdict": "no_data"}
    dd = [float(d["delta_dir"]) for d in diags]
    rd = [float(d["R_delta"]) for d in diags]
    clustered = all("jaad_video" in d for d in diags)
    if clustered:
        dd_ci = cluster_bootstrap_ci_median(diags, "delta_dir")
        rd_ci = cluster_bootstrap_ci_median(diags, "R_delta")
    else:
        dd_ci = bootstrap_ci_median(dd)
        rd_ci = bootstrap_ci_median(rd)
    med_dd = statistics.median(dd)
    med_rd = statistics.median(rd)
    dd_holds = med_dd > 0 and dd_ci[0] > 0
    rd_holds = med_rd < R_DELTA_EVIDENCE_BAR and rd_ci[1] < R_DELTA_EVIDENCE_BAR
    if dd_holds and rd_holds:
        verdict = "temporal_oversmoothing_supported"
    elif dd_holds or rd_holds:
        verdict = "borderline"
    else:
        verdict = "no_evidence"
    return {
        "n_pairs": len(diags),
        "delta_dir": {
            "mean": statistics.fmean(dd),
            "median": med_dd,
            "q1": statistics.quantiles(dd, n=4)[0] if len(dd) >= 4 else None,
            "q3": statistics.quantiles(dd, n=4)[2] if len(dd) >= 4 else None,
            "bootstrap_median_ci95": [dd_ci[0], dd_ci[1]],
        },
        "R_delta": {
            "mean": statistics.fmean(rd),
            "median": med_rd,
            "q1": statistics.quantiles(rd, n=4)[0] if len(rd) >= 4 else None,
            "q3": statistics.quantiles(rd, n=4)[2] if len(rd) >= 4 else None,
            "bootstrap_median_ci95": [rd_ci[0], rd_ci[1]],
        },
        "evidence_rule": {
            "delta_dir_positive_ci": bool(dd_holds),
            "R_delta_below_bar_ci": bool(rd_holds),
            "R_delta_bar": R_DELTA_EVIDENCE_BAR,
            "verdict": verdict,
            "note": "pre-registered before any run (docs/e3_event_state_jepa_plan.md §2)",
        },
        "protocol": "384-d pooled DINOv2 features, deterministic predictors; "
                    "NOT comparable to WA-JEPA's spatial-token protocol",
        "uncertainty": {
            "method": ("cluster_bootstrap_by_jaad_video" if clustered
                       else "pair_bootstrap (no source IDs available)"),
            "n_clusters": len({d["jaad_video"] for d in diags}) if clustered else None,
            "n_bootstrap": N_BOOTSTRAP,
            "seed": BOOTSTRAP_SEED,
        },
    }


def code_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


# ------------------------------ real run (torch) ----------------------------

def parse_frame_list(raw: str) -> List[Tuple[int, List[float]]]:
    """anchor_frames/cand_frames cell -> [(frame, [x1,y1,x2,y2]), ...]."""
    items = json.loads(raw)
    out = []
    for it in items:
        frame = int(it[0])
        bbox = [float(v) for v in it[1]]
        out.append((frame, bbox))
    return out


def run_real(models: Sequence[str], limit: Optional[int], out_dir: Path,
             samples_csv: Path, stride: int = 1,
             splits: Sequence[str] = ("train", "val")) -> int:
    import numpy as np  # noqa: import inside real run only
    import torch  # noqa: import inside real run only
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from wmeval.latent_features import LatentFeatureExtractor, CropConfig
    from wmeval.latent_world_state.residual_predictor import ResidualPredictor

    out_dir.mkdir(parents=True, exist_ok=True)
    log_lines: List[str] = [
        f"started {time.strftime('%Y-%m-%dT%H:%M:%S')} code={code_sha256()[:12]}",
        f"samples={samples_csv} models={list(models)} limit={limit}",
        "pre-registered rule: median(delta_dir)>0 AND median(R_delta)<0.5 "
        "(both bootstrap CI excluding the critical value)",
    ]

    predictors = {}
    for name in models:
        if name == "d4":
            m = ResidualPredictor()
            m.load_state_dict(torch.load(
                REPO_ROOT / "runs/d4_experimental/d4_lam0.5.pt", map_location="cpu"))
            predictors["d4"] = (m, "frozen D4 (discriminative)")
        elif name == "b4":
            m = ResidualPredictor()
            m.load_state_dict(torch.load(
                REPO_ROOT / "runs/j2_2_experimental/b4_residual_expanded.pt",
                map_location="cpu"))
            predictors["b4"] = (m, "frozen B4 (regression)")
        elif name == "persistence":
            predictors["persistence"] = (None, "repeat last history frame (lower bound reference)")
        else:
            raise ValueError(f"unknown model {name!r}")
    for m, _ in predictors.values():
        if m is not None:
            m.eval()

    ex = LatentFeatureExtractor(model_id="ckpt/dinov2-small",
                                crop_cfg=CropConfig(),
                                cache_dir="runs/j1_baseline/feature_cache")

    rows = list(csv.DictReader(samples_csv.open(newline="")))
    pairs_all = [r for r in rows
                 if r["label"] == "same_identity" and r["split"] in set(splits)]
    pairs = pairs_all[::max(1, stride)]
    if limit:
        pairs = pairs[:limit]
    split_counts = {}
    for r in pairs_all:
        split_counts[r["split"]] = split_counts.get(r["split"], 0) + 1
    log_lines.append(
        f"samples.csv rows={len(rows)} eligible same_identity pairs={len(pairs_all)} "
        f"splits={dict(sorted(split_counts.items()))} (split=test excluded by default; "
        f"use --include-test only for a separately justified final diagnostic) "
        f"stride={max(1, stride)} -> evaluating {len(pairs)} pairs "
        f"(deterministic subsample, explicitly logged; not a silent cap)")

    per_pair_rows = []
    diags_by_model = {name: [] for name in models}
    skipped = 0
    with torch.no_grad():
        for r in pairs:
            sid = r["synpedas_sample"]
            f0 = int(sid.split("_f")[-1].split("-")[0])
            vp = REPO_ROOT / "repos/SynPeDAS/JAAD_crossing_moments" / f"{sid}.mp4"
            fps = float(r["fps"]) or 30.0
            anchor = parse_frame_list(r["anchor_frames"])
            future = parse_frame_list(r["cand_frames"])
            if len(anchor) < 2 or len(future) < 2:
                skipped += 1
                continue
            try:
                hist = [ex.frame_feature(str(vp), f - f0, bbox, sid,
                                         r["anchor_ped_id"]) for f, bbox in anchor]
                true_future = [ex.frame_feature(str(vp), f - f0, bbox, sid,
                                                r["cand_ped_id"]) for f, bbox in future]
            except RuntimeError:
                skipped += 1
                continue
            hist_t = torch.from_numpy(np.stack(hist).astype(np.float32))
            last_anchor_frame = anchor[-1][0]
            deltas = [(f - last_anchor_frame) / fps for f, _ in future]
            base = {
                "pair_id": r["pair_id"], "split": r["split"], "sample": sid,
                "jaad_video": r["jaad_video"],
                "future_window_s": [round(deltas[0], 4), round(deltas[-1], 4)],
                "n_future_steps": len(future),
            }
            for name, (model, desc) in predictors.items():
                if name == "persistence":
                    pred = [hist[-1] for _ in future]
                else:
                    pred = [model(hist_t[None], torch.tensor([d])).numpy()[0]
                            for d in deltas]
                d = temporal_diagnostics(pred, true_future)
                d["jaad_video"] = r["jaad_video"]  # cluster bootstrap unit
                rec = {**base, "model": name,
                       "C_pred": round(d["C_pred"], 6), "C_true": round(d["C_true"], 6),
                       "D_pred": round(d["D_pred"], 6), "D_true": round(d["D_true"], 6),
                       "delta_dir": round(d["delta_dir"], 6),
                       "R_delta": round(d["R_delta"], 6)}
                per_pair_rows.append(rec)
                diags_by_model[name].append(d)
            log_lines.append(f"pair {r['pair_id']} ({sid}) done; skipped so far={skipped}")
    ex.flush_cache()

    # write outputs
    with (out_dir / "per_pair_diagnostics.csv").open("w", newline="",
                                                     encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_pair_rows[0].keys()))
        w.writeheader()
        w.writerows(per_pair_rows)
    for name in models:
        summary = summarize_diagnostics(diags_by_model[name])
        summary["model"] = name
        summary["description"] = predictors[name][1]
        summary["code_sha256"] = code_sha256()
        summary["samples_csv"] = str(samples_csv)
        summary["n_pairs_evaluated"] = len(diags_by_model[name])
        summary["n_pairs_skipped"] = skipped
        summary["split_policy"] = {
            "included_splits": list(splits),
            "eligible_pairs_by_split": split_counts,
            "test_protection": "split=test excluded unless --include-test is explicit",
            "stride": max(1, stride),
        }
        summary["future_window_note"] = ("cand_frames window [gap, gap+12) frames; "
                                         "narrow (~0.5-0.9 s) but >= 2 timesteps")
        (out_dir / f"summary_{name}.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")
        log_lines.append(
            f"model {name}: n={len(diags_by_model[name])} "
            f"median delta_dir={summary['delta_dir']['median']:.4f} "
            f"median R_delta={summary['R_delta']['median']:.4f} "
            f"verdict={summary['evidence_rule']['verdict']}")
    log_lines.append(f"finished {time.strftime('%Y-%m-%dT%H:%M:%S')}; "
                     f"pairs evaluated={len(diags_by_model[models[0]])} skipped={skipped}")
    (out_dir / "run_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    for line in log_lines:
        if line.startswith(("model ", "samples.csv", "finished")):
            print(f"[E3-D0] {line}")
    print(f"[E3-D0] outputs -> {out_dir}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="E3-D0 temporal diagnostics")
    ap.add_argument("--samples", default=str(DEFAULT_SAMPLES))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--models", default="d4,b4,persistence",
                    help="comma-separated: d4,b4,persistence")
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate only the first N selected pairs")
    ap.add_argument("--stride", type=int, default=1,
                    help="evaluate every N-th same_identity pair "
                         "(deterministic subsample, logged in outputs)")
    ap.add_argument("--include-test", action="store_true",
                    help="also include split=test (protected by default; "
                         "not for architecture/threshold selection)")
    args = ap.parse_args(argv)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    splits = ("train", "val", "test") if args.include_test else ("train", "val")
    return run_real(models, args.limit, Path(args.out_dir), Path(args.samples),
                    stride=args.stride, splits=splits)


if __name__ == "__main__":
    sys.exit(main())
