"""J1.1 §五: occlusion-sampling strategy comparison (A / B / C).

A: all frames regardless of occlusion
B: prefer_visible (drop occ==2 when alternates exist)   [J1 legacy]
C: weighted aggregation by visibility quality (1.0/0.7/0.3)

Subset: 200 train + 300 test pairs (sampled from the J1 pair set,
same pair_ids across strategies for a paired comparison).
Each strategy uses its OWN cache dir (config in cache identity).
Thresholds re-calibrated per strategy on TRAIN (strict FPR<=5%).
Output: runs/j1_1_audited/occlusion_strategies.json
"""
from __future__ import annotations

import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from wmeval.latent_features import CropConfig, LatentFeatureExtractor
from wmeval.latent_association import (FeatureAssociation,
                                       calibrate_threshold, pair_metrics)

SEQ = Path("datasets/processed/actor_sequences")
OUT = Path("runs/j1_1_audited")
KIN = 0.15
SEED = 31


def extrapolation_error(anchor, cand):
    def centers(frames):
        return [((b[0] + b[2]) / 2 / 1920.0, (b[1] + b[3]) / 2 / 1080.0)
                for _, b, _ in frames]
    ac, cc = centers(anchor), centers(cand)
    import statistics
    steps = [(ac[i + 1][0] - ac[i][0], ac[i + 1][1] - ac[i][1])
             for i in range(max(0, len(ac) - 6), len(ac) - 1)]
    v = (statistics.median(s[0] for s in steps),
         statistics.median(s[1] for s in steps))
    gap = cand[0][0] - anchor[-1][0]
    pred = (ac[-1][0] + v[0] * gap, ac[-1][1] + v[1] * gap)
    return ((cc[0][0] - pred[0]) ** 2 + (cc[0][1] - pred[1]) ** 2) ** 0.5


def main() -> None:
    random.seed(SEED)
    rows = list(csv.DictReader(open(SEQ / "samples.csv",
                                    newline="", encoding="utf-8")))
    # J1 pair ids (same pairs across strategies)
    j1 = list(csv.DictReader(open("runs/j1_baseline/pair_results.csv")))
    j1_ids = {r["pair_id"] for r in j1}
    pool = [r for r in rows if r["pair_id"] in j1_ids]
    tr = [r for r in pool if r["split"] == "train"]
    te = [r for r in pool if r["split"] == "test"]
    sub_tr = random.sample(tr, min(200, len(tr)))
    sub_te = random.sample(te, min(300, len(te)))
    sub = sub_tr + sub_te
    print(f"[occ] pairs: train {len(sub_tr)} / test {len(sub_te)}")

    results = {}
    for strat in ("all", "prefer_visible", "weighted"):
        cache = OUT / f"occlusion_cache_{strat}"
        cfg = CropConfig(occlusion_strategy=strat)
        ex = LatentFeatureExtractor(model_id="ckpt/dinov2-small",
                                    crop_cfg=cfg, cache_dir=cache)
        fa = FeatureAssociation(extractor=ex)
        t0 = time.perf_counter()
        recs = []
        for r in sub:
            anchor = json.loads(r["anchor_frames"])
            cand = json.loads(r["cand_frames"])
            f0 = int(r["synpedas_sample"].split("_f")[-1].split("-")[0])
            pf = fa.pair_features(r["synpedas_sample"], f0, anchor,
                                  r["anchor_ped_id"], cand, r["cand_ped_id"])
            recs.append({"pair_id": r["pair_id"], "split": r["split"],
                         "label": r["label"], "cos": pf.cos,
                         "kin": round(extrapolation_error(anchor, cand), 4)})
        ex.flush_cache()
        pos = [x["cos"] for x in recs if x["label"] == "same_identity"]
        neg = [x["cos"] for x in recs if x["label"] == "different_identity"]
        # strict calibration on the strategy's own train subset
        thr, rep = calibrate_threshold(
            [x["cos"] for x in recs if x["split"] == "train"
             and x["label"] == "same_identity"],
            [x["cos"] for x in recs if x["split"] == "train"
             and x["label"] == "different_identity"], 0.05)
        test = [x for x in recs if x["split"] == "test"]
        lab = [x["label"] == "same_identity" for x in test]
        m_cos = pair_metrics(lab, [x["cos"] >= (thr or 1.0) for x in test])
        m_fuse = pair_metrics(lab, [x["kin"] <= KIN and
                                    x["cos"] >= (thr or 1.0) for x in test])
        results[strat] = {
            "n_pairs": len(recs),
            "seconds": round(time.perf_counter() - t0),
            "separability": {"same_mean": round(float(np.mean(pos)), 4),
                             "diff_mean": round(float(np.mean(neg)), 4),
                             "same_p5": round(float(np.percentile(pos, 5)), 4),
                             "diff_p95": round(float(np.percentile(neg, 95)), 4)},
            "calibration": rep,
            "test_cos_strict": m_cos,
            "test_fused_strict": m_fuse,
        }
        print(f"[occ] {strat}: sep {results[strat]['separability']} "
              f"thr={thr} ({time.perf_counter()-t0:.0f}s)")
        # save per-strategy records
        with open(OUT / f"occlusion_pairs_{strat}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
            w.writeheader()
            w.writerows(recs)

    (OUT / "occlusion_strategies.json").write_text(
        json.dumps(results, indent=2))
    print(f"[occ] -> {OUT}/occlusion_strategies.json")


if __name__ == "__main__":
    main()
