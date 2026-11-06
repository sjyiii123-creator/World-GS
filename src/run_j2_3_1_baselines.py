"""J2.3.1 §九: baseline recalibration on the FIXED candidate manifest.

B0 static last-frame | B4 residual predictor (per-candidate delta) |
A1 kinematic ranking+rejection — all on the corrected Tier manifest
(runs/j2_3_1_alignment/tier1_candidates.csv): correct attribution,
deduplicated observations, causal extrapolation gate.
Old M_hard=0.112 (same-source approximation) is NOT carried over.
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
from wmeval.latent_association import calibrate_threshold, pair_metrics

V2 = Path("runs/j2_experimental")
J22 = Path("runs/j2_2_experimental")
OUT = Path("runs/j2_3_1_alignment")
CAND = OUT / "tier1_candidates.csv"
CACHE = Path("runs/j1_baseline/feature_cache")
SEQCSV = Path("datasets/processed/actor_sequences/samples.csv")
N_BOOT = 1000


def main() -> None:
    cands = list(csv.DictReader(open(CAND, newline="")))
    by_anchor = defaultdict(list)
    for r in cands:
        by_anchor[r["anchor_id"]].append(r)
    v2man = {m["pair_id"]: m for m in csv.DictReader(
        open(V2 / "j2_manifest.csv", newline=""))}
    seqrows = {r["pair_id"]: r for r in csv.DictReader(
        open(SEQCSV, newline=""))}
    ex = LatentFeatureExtractor(model_id="ckpt/dinov2-small",
                                crop_cfg=CropConfig(), cache_dir=CACHE)
    b4 = ResidualPredictor()
    b4.load_state_dict(torch.load(J22 / "b4_residual_expanded.pt",
                                  map_location="cpu"))
    b4.eval()

    rows = []
    with torch.no_grad():
        for aid, lst in by_anchor.items():
            m = v2man.get(aid)
            if not m:
                continue
            srow = seqrows[aid]
            sid = srow["synpedas_sample"]
            f0 = int(sid.split("_f")[-1].split("-")[0])
            fps = float(srow["fps"])
            vp = f"repos/SynPeDAS/JAAD_crossing_moments/{sid}.mp4"
            anchor = json.loads(srow["anchor_frames"])
            hist = np.stack([ex.frame_feature(vp, f[0] - f0, f[1], sid,
                                              srow["anchor_ped_id"])
                             for f in anchor])
            z_last = hist[-1]
            entry = {"anchor_id": aid, "split": m["split"],
                     "source": srow["jaad_video"], "cands": []}
            for c in lst:
                src, ped, fr = c["candidate_feature_key"].split("|")
                fr = int(fr)
                # candidate feature from the dedup index's real box:
                # recover from the samples rows of THAT actor
                bb = None
                for r2 in seqrows.values():
                    if r2["jaad_video"] == src and r2["anchor_ped_id"] == ped:
                        for f, b, o in json.loads(r2["anchor_frames"]) + \
                                json.loads(r2["cand_frames"]):
                            if f == fr:
                                bb = b
                                break
                    if bb:
                        break
                if bb is None:
                    continue
                feat = ex.frame_feature(vp, fr - f0, bb, sid, ped)
                d = (fr - anchor[-1][0]) / fps
                z_pred = b4(torch.from_numpy(hist[None].astype(np.float32)),
                            torch.tensor([max(d, 1e-3)])).numpy()[0]
                entry["cands"].append({
                    "is_true": bool(int(c["is_true_match"])),
                    "kin": float(c["kinematic_distance"]),
                    "static_cos": float(np.dot(z_last, feat)),
                    "pred_cos": float(np.dot(z_pred, feat)),
                    "delta_s": round(d, 3)})
            if entry["cands"]:
                rows.append(entry)
    ex.flush_cache()
    print(f"[base] anchors: {len(rows)} (train "
          f"{sum(1 for r in rows if r['split']=='train')}, "
          f"val {sum(1 for r in rows if r['split']=='val')}, "
          f"expl {sum(1 for r in rows if r['split']=='exploratory_test')})")

    def calib(key):
        right = [c[key] for r in rows if r["split"] == "train"
                 for c in r["cands"] if c["is_true"]]
        wrong = [c[key] for r in rows if r["split"] == "train"
                 for c in r["cands"] if not c["is_true"]]
        t, rep = calibrate_threshold(right, wrong, 0.05)
        return t, rep

    kin_thr, _ = calib("kin")
    pred_thr, _ = calib("pred_cos")
    stat_thr, _ = calib("static_cos")
    print(f"[base] kin={kin_thr} static={stat_thr} pred={pred_thr}")

    def m_hard(rowset, key):
        vals = []
        for r in rowset:
            pos = [c[key] for c in r["cands"] if c["is_true"]]
            negs = [c[key] for c in r["cands"]
                    if not c["is_true"] and c["kin"] <= (kin_thr or 9)]
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
        "protocol": "corrected Tier manifest (dedup+attribution+causal gate)",
        "thresholds": {"kin": kin_thr, "static": stat_thr, "pred": pred_thr},
        "M_hard": {
            "B0_static_val": m_hard(val, "static_cos"),
            "B0_static_expl": m_hard(expl, "static_cos"),
            "B4_pred_val": m_hard(val, "pred_cos"),
            "B4_pred_expl": m_hard(expl, "pred_cos")},
        "old_M_hard_invalidated": "0.112 used same-source approximation "
                                  "with misattributed observations",
        "association_val": {r: evaluate(val, r) for r in ("A1", "B0", "B4")},
        "association_expl": {r: evaluate(expl, r) for r in ("A1", "B0", "B4")},
        "no_match_val": {r: evaluate(
            [{"cands": [c for c in r_["cands"] if not c["is_true"]],
              "split": r_["split"], "source": r_["source"],
              "anchor_id": r_["anchor_id"]}
             for r_ in val if any(not c["is_true"] for c in r_["cands"])],
            r) for r in ("A1", "B0", "B4")},
        "gap_strata_M_hard_expl": {},
    }
    for g0, g1, tag in ((0.4, 0.75, "0.5s"), (0.9, 1.25, "1.0s"),
                        (1.4, 1.9, "1.5s")):
        sub = [r for r in expl if any(g0 <= c["delta_s"] <= g1
                                      for c in r["cands"])]
        results["gap_strata_M_hard_expl"][tag] = {
            "B0": m_hard(sub, "static_cos"), "B4": m_hard(sub, "pred_cos")}

    # source bootstrap CI on top1 (A1, val+expl pooled)
    def boot(rule, metric):
        rng = random.Random(51)
        by_src = defaultdict(list)
        for r in val + expl:
            by_src[r["source"]].append(r)
        srcs = list(by_src)
        vals_ = []
        for _ in range(N_BOOT):
            sample = [r for s in [rng.choice(srcs) for _ in srcs]
                      for r in by_src[s]]
            vals_.append(evaluate(sample, rule)[metric])
        lo, hi = np.percentile(vals_, [2.5, 97.5])
        return [round(float(lo), 4), round(float(hi), 4)]

    results["source_ci_top1"] = {r: boot(r, "top1")
                                 for r in ("A1", "B0", "B4")}
    (OUT / "baseline_recalibration.json").write_text(
        json.dumps(results, indent=2))
    print(json.dumps(results["M_hard"], indent=1))
    print(json.dumps(results["association_val"], indent=1))
    print(f"[base] -> {OUT}")


if __name__ == "__main__":
    main()
