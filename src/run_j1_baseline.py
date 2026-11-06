"""Run J1 baseline: frozen DINOv2 appearance association vs JAAD identity.

Pipeline:
  1. benchmark (32-crop batch): VRAM + ms/frame;
  2. extract features for a sample of pairs (train for calibration,
     test for evaluation; val untouched for later);
  3. separability report (same vs diff cosine distributions);
  4. calibrate cos threshold on TRAIN only;
  5. evaluate: appearance-only, kinematic-only (J0 rule), fused (AND);
  6. competition ranking by cosine.

Output: runs/j1_baseline/{pair_results.csv, metrics.json} + cache.
"""
from __future__ import annotations

import csv
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np

from wmeval.latent_features import CropConfig, LatentFeatureExtractor
from wmeval.latent_association import FeatureAssociation

SEQ = Path("datasets/processed/actor_sequences")
OUT = Path("runs/j1_baseline")
CACHE = OUT / "feature_cache"
KIN_THRESH = 0.15
N_PER_SPLIT = {"train": 800, "test": 800, "val": 200}
SEED = 21


def extrapolation_error(anchor, cand):
    # same kinematic rule as J0 (real gap from frame indices)
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
    OUT.mkdir(parents=True, exist_ok=True)

    ex = LatentFeatureExtractor(
        model_id="ckpt/dinov2-small",       # local weights (curl-proxied)
        crop_cfg=CropConfig(), cache_dir=CACHE)
    fa = FeatureAssociation(extractor=ex)
    print(f"[j1] model on {ex.device}, dim={ex.dim}")

    # ---- 1. benchmark -------------------------------------------------
    r0 = rows[0]
    anchor = json.loads(r0["anchor_frames"])
    f0 = int(r0["synpedas_sample"].split("_f")[-1].split("-")[0])
    bm = ex.benchmark(fa.video_path(r0["synpedas_sample"]),
                      anchor[0][0] - f0, anchor[0][1], n=32)
    print(f"[j1] benchmark: {bm}")

    # ---- 2/3. feature extraction + separability -----------------------
    sub = []
    for split, n in N_PER_SPLIT.items():
        pool = [r for r in rows if r["split"] == split]
        sub.extend(random.sample(pool, min(n, len(pool))))
    t0 = time.perf_counter()
    results = []
    for i, r in enumerate(sub):
        anchor = json.loads(r["anchor_frames"])
        cand = json.loads(r["cand_frames"])
        f0 = int(r["synpedas_sample"].split("_f")[-1].split("-")[0])
        try:
            pf = fa.pair_features(r["synpedas_sample"], f0, anchor,
                                  r["anchor_ped_id"], cand, r["cand_ped_id"])
        except RuntimeError as e:
            print(f"  skip pair {r['pair_id']}: {e}")
            continue
        kin = extrapolation_error(anchor, cand)
        results.append({
            "pair_id": r["pair_id"], "split": r["split"],
            "label": r["label"], "gap_frames": r["gap_frames"],
            "cos": pf.cos, "l2": pf.l2,
            "kin_err": round(kin, 4),
        })
        if (i + 1) % 400 == 0:
            ex.flush_cache()
            print(f"  {i+1}/{len(sub)} pairs, {time.perf_counter()-t0:.0f}s")
    ex.flush_cache()
    print(f"[j1] extracted {len(results)} pairs in "
          f"{time.perf_counter()-t0:.0f}s")

    pos = [x["cos"] for x in results if x["label"] == "same_identity"]
    neg = [x["cos"] for x in results if x["label"] == "different_identity"]
    sep = {
        "same_identity": {"n": len(pos),
                          "cos_mean": round(float(np.mean(pos)), 4),
                          "cos_p5": round(float(np.percentile(pos, 5)), 4),
                          "cos_p50": round(float(np.median(pos)), 4)},
        "different_identity": {"n": len(neg),
                               "cos_mean": round(float(np.mean(neg)), 4),
                               "cos_p50": round(float(np.median(neg)), 4),
                               "cos_p95": round(float(np.percentile(neg, 95)), 4)},
    }
    print(f"[j1] separability: {json.dumps(sep)}")

    # ---- 4. calibrate on train ----------------------------------------
    tr = [x for x in results if x["split"] == "train"]
    cos_thr = FeatureAssociation.calibrate_threshold(
        [x["cos"] for x in tr if x["label"] == "same_identity"],
        [x["cos"] for x in tr if x["label"] == "different_identity"])
    print(f"[j1] calibrated cos threshold (train, FPR<=5%): {cos_thr}")

    # ---- 5. evaluate on test -------------------------------------------
    te = [x for x in results if x["split"] == "test"]

    def eval_rule(name, accept):
        tp = sum(1 for x in te if x["label"] == "same_identity" and accept(x))
        fp = sum(1 for x in te if x["label"] == "different_identity" and accept(x))
        fn = sum(1 for x in te if x["label"] == "same_identity" and not accept(x))
        n = len(te)
        return {"rule": name, "n": n,
                "accuracy": round((n - fp - fn) / n, 4),
                "error_merge_rate": round(fp / n, 4),
                "missed_relink_rate": round(fn / max(1, tp + fn), 4)}

    out = {
        "benchmark": bm,
        "n_extracted": len(results),
        "separability": sep,
        "cos_threshold_train": cos_thr,
        "kin_threshold": KIN_THRESH,
        "rules": {
            "J0_kinematic_only": eval_rule(
                "kin", lambda x: x["kin_err"] <= KIN_THRESH),
            "J1_appearance_only": eval_rule(
                "cos", lambda x: x["cos"] >= cos_thr),
            "J1_fused_and": eval_rule(
                "and", lambda x: x["kin_err"] <= KIN_THRESH
                and x["cos"] >= cos_thr),
        },
        "test_by_gap": {},
    }
    for g in (15, 30, 45):
        sub_te = [x for x in te if int(x["gap_frames"]) == g]

        def er(name, accept, sub=sub_te):
            tp = sum(1 for x in sub if x["label"] == "same_identity" and accept(x))
            fp = sum(1 for x in sub if x["label"] == "different_identity" and accept(x))
            fn = sum(1 for x in sub if x["label"] == "same_identity" and not accept(x))
            return round((len(sub) - fp - fn) / len(sub), 4), round(fp / len(sub), 4)
        out["test_by_gap"][g] = {
            "kin_acc,merge": er("kin", lambda x: x["kin_err"] <= KIN_THRESH),
            "cos_acc,merge": er("cos", lambda x: x["cos"] >= cos_thr),
            "fused_acc,merge": er("and", lambda x: x["kin_err"] <= KIN_THRESH
                                  and x["cos"] >= cos_thr),
        }

    with open(OUT / "pair_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    (OUT / "metrics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["rules"], indent=2))
    print(json.dumps(out["test_by_gap"], indent=2))
    print(f"[j1] -> {OUT}")


if __name__ == "__main__":
    main()
