"""J2.3.2 §七: baseline rerun on the FULLY-ALIGNED candidate set.

P0-A: candidate boxes come from the CANONICAL observation index
      (exact (src,ped,frame) lookup; missing -> counted, never silent).
P0-B: kinematic threshold uses calibrate_lower_is_better (distance
      semantics; accept = kin <= thr; FPR = frac(negatives <= thr)).
P0-C: M_hard uses hard_negative_tier == 1 ONLY; Tier-2 and all-gated
      margins reported separately.
P0-D: candidates from the fixed-query manifest (no oracle time, no
      forced true match).

A1 (kin rank+reject) | B0 (static) | B4 (residual predictor, frozen
checkpoint, per-candidate delta) — identical candidate availability.
"""
from __future__ import annotations

import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")

from wmeval.latent_features import CropConfig, LatentFeatureExtractor
from wmeval.latent_world_state.residual_predictor import ResidualPredictor
from wmeval.latent_association import (calibrate_lower_is_better,
                                       calibrate_threshold)

OUT = Path("runs/j2_3_2_alignment")
CAND = OUT / "candidates_fixed_query.csv"
V2 = Path("runs/j2_experimental")
J22 = Path("runs/j2_2_experimental")
CACHE = Path("runs/j1_baseline/feature_cache")
SEQCSV = Path("datasets/processed/actor_sequences/samples.csv")
N_BOOT = 1000


def main() -> None:
    cands = list(csv.DictReader(open(CAND, newline="")))
    by_anchor = defaultdict(list)
    for r in cands:
        by_anchor[r["anchor_id"]].append(r)
    seqrows = {r["pair_id"]: r for r in csv.DictReader(
        open(SEQCSV, newline=""))}
    v2man = {m["pair_id"]: m for m in csv.DictReader(
        open(V2 / "j2_manifest.csv", newline=""))}
    ex = LatentFeatureExtractor(model_id="ckpt/dinov2-small",
                                crop_cfg=CropConfig(), cache_dir=CACHE)
    b4 = ResidualPredictor()
    b4.load_state_dict(torch.load(J22 / "b4_residual_expanded.pt",
                                  map_location="cpu"))
    b4.eval()

    coverage = {"rows_total": len(cands), "rows_with_feature": 0,
                "rows_missing_feature": 0,
                "missing_examples": []}
    rows = []
    with torch.no_grad():
        for aid, lst in by_anchor.items():
            srow = seqrows[aid]
            sid = srow["synpedas_sample"]
            f0 = int(sid.split("_f")[-1].split("-")[0])
            vp = f"repos/SynPeDAS/JAAD_crossing_moments/{sid}.mp4"
            anchor = json.loads(srow["anchor_frames"])
            try:
                hist = np.stack([ex.frame_feature(vp, f[0] - f0, f[1], sid,
                                                  srow["anchor_ped_id"])
                                 for f in anchor])
            except RuntimeError as e:
                coverage["missing_examples"].append(f"{aid}:anchor:{e}")
                continue
            z_last = hist[-1]
            entry = {"anchor_id": aid, "split": lst[0]["split"],
                     "source": srow["jaad_video"], "cands": []}
            for c in lst:
                # P0-A: box from the manifest's canonical provenance
                bbox = json.loads(c["bbox"])
                fr = int(c["candidate_timestamp_frame"])
                try:
                    feat = ex.frame_feature(vp, fr - f0, bbox, sid,
                                            c["candidate_id"])
                except RuntimeError as e:
                    coverage["rows_missing_feature"] += 1
                    coverage["missing_examples"].append(
                        f"{aid}:{c['candidate_id']}:{fr}:{e}")
                    continue      # recorded, never silent
                coverage["rows_with_feature"] += 1
                d = float(c["candidate_delta_s"])
                z_pred = b4(torch.from_numpy(hist[None].astype(np.float32)),
                            torch.tensor([max(d, 1e-3)])).numpy()[0]
                entry["cands"].append({
                    "is_true": bool(int(c["is_true_match"])),
                    "tier": int(c["hard_negative_tier"]),
                    "kin": float(c["kinematic_distance"]),
                    "static_cos": float(np.dot(z_last, feat)),
                    "pred_cos": float(np.dot(z_pred, feat)),
                    "delta_s": d})
            if entry["cands"]:
                rows.append(entry)
    ex.flush_cache()
    print(f"[base] coverage: {coverage['rows_with_feature']}/"
          f"{coverage['rows_total']} candidate features "
          f"({coverage['rows_missing_feature']} missing, recorded)")

    tr = [r for r in rows if r["split"] == "train"]
    # P0-B: kin threshold with LOWER-is-better calibration
    kin_pos = [c["kin"] for r in tr for c in r["cands"] if c["is_true"]]
    kin_neg = [c["kin"] for r in tr for c in r["cands"]
               if not c["is_true"] and c["tier"] == 1]
    kin_thr, kin_rep = calibrate_lower_is_better(kin_pos, kin_neg, 0.05)
    # appearance thresholds (higher-better) unchanged protocol
    stat_thr, _ = calibrate_threshold(
        [c["static_cos"] for r in tr for c in r["cands"] if c["is_true"]],
        [c["static_cos"] for r in tr for c in r["cands"]
         if not c["is_true"] and c["tier"] == 1], 0.05)
    pred_thr, _ = calibrate_threshold(
        [c["pred_cos"] for r in tr for c in r["cands"] if c["is_true"]],
        [c["pred_cos"] for r in tr for c in r["cands"]
         if not c["is_true"] and c["tier"] == 1], 0.05)
    print(f"[base] kin_thr(lower-better)={kin_thr} rep={kin_rep}")

    # P0-C: pure-tier margins
    def margin(rowset, key, tier=1):
        vals = []
        for r in rowset:
            pos = [c[key] for c in r["cands"] if c["is_true"]]
            negs = [c[key] for c in r["cands"] if c["tier"] == tier]
            if pos and negs:
                vals.append(pos[0] - max(negs))
        return round(float(np.mean(vals)), 4) if vals else None

    def evaluate(rowset, rule):
        n = top1 = acc = unc = wa = ta = 0
        for r in rowset:
            by = sorted(r["cands"], key=lambda c: c["kin"])
            top = by[0]
            n += 1
            top1 += top["is_true"]
            if rule == "A1":
                d = ("accept" if top["kin"] <= kin_thr else
                     "reject" if top["kin"] >= 2 * kin_thr else "uncertain")
            else:
                s = top["static_cos" if rule == "B0" else "pred_cos"]
                thr = stat_thr if rule == "B0" else pred_thr
                d = ("accept" if top["kin"] <= kin_thr and s >= thr else
                     "reject" if top["kin"] >= 2 * kin_thr or s < 0.45
                     else "uncertain")
            if d == "accept":
                acc += 1
                ta += top["is_true"]
                wa += not top["is_true"]
            elif d == "uncertain":
                unc += 1
        return {"n": n, "top1": round(top1 / n, 4),
                "coverage": round(acc / n, 4),
                "wrong_acc": round(wa / acc, 4) if acc else None,
                "uncertain": round(unc / n, 4)}

    val = [r for r in rows if r["split"] == "val"]
    expl = [r for r in rows if r["split"] == "exploratory_test"]
    results = {
        "protocol": "J2.3.2 aligned: canonical boxes + fixed-query(1.0s) "
                    "+ lower-is-better kin + pure-tier margins",
        "feature_coverage": coverage,
        "thresholds": {"kin_lower_better": kin_thr, "kin_report": kin_rep,
                       "static": stat_thr, "pred": pred_thr},
        "margins_tier1": {
            "B0_static_val": margin(val, "static_cos", 1),
            "B0_static_expl": margin(expl, "static_cos", 1),
            "B4_pred_val": margin(val, "pred_cos", 1),
            "B4_pred_expl": margin(expl, "pred_cos", 1)},
        "margins_tier2_separate": {
            "B0_static_val": margin(val, "static_cos", 2),
            "B4_pred_val": margin(val, "pred_cos", 2)},
        "margins_all_gated": {
            "B0_static_val": margin(
                [{"cands": [dict(c, tier=1 if (not c["is_true"] and
                                               c["kin"] <= (kin_thr or 9))
                             else c["tier"]) for c in r["cands"]]}
                 for r in val], "static_cos", 1),
        },
        "association_val": {r: evaluate(val, r) for r in ("A1", "B0", "B4")},
        "association_expl": {r: evaluate(expl, r) for r in ("A1", "B0", "B4")},
        "gap_strata_margins_tier1_expl": {},
    }
    for g0, g1, tag in ((0.6, 1.4, "1.0s-neighborhood"),
                        (1.4, 2.0, "late")):
        sub = [r for r in expl
               if any(g0 <= c["delta_s"] <= g1 for c in r["cands"])]
        results["gap_strata_margins_tier1_expl"][tag] = {
            "B0": margin(sub, "static_cos", 1),
            "B4": margin(sub, "pred_cos", 1),
            "n_anchors": len(sub)}

    def boot(rule, metric):
        rng = random.Random(51)
        by_src = defaultdict(list)
        for r in val + expl:
            by_src[r["source"]].append(r)
        srcs = list(by_src)
        vals_ = [evaluate([r for s in [rng.choice(srcs) for _ in srcs]
                           for r in by_src[s]], rule)[metric]
                 for _ in range(N_BOOT)]
        vals_ = [v for v in vals_ if v is not None]
        lo, hi = np.percentile(vals_, [2.5, 97.5])
        return [round(float(lo), 4), round(float(hi), 4)]

    results["source_ci_top1"] = {r: boot(r, "top1")
                                 for r in ("A1", "B0", "B4")}
    # candidate availability per method (identical by construction)
    results["availability"] = {
        "n_anchors_evaluated": len(rows),
        "per_method_candidate_counts": "identical (same manifest rows; "
                                       "A1 uses kin only, B0/B4 add cos "
                                       "scores computed on the SAME rows)"}
    (OUT / "baseline_aligned.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results["margins_tier1"], indent=1))
    print(json.dumps(results["association_val"], indent=1))
    print(f"[base] -> {OUT}")


if __name__ == "__main__":
    main()
