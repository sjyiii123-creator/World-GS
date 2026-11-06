"""D4 three-gate evaluation + LOSO analysis.

Gates (from frozen_baseline.json):
  1 Identity: val Tier-1 M_hard > B0's 0.0339
  2 Ranking: multi-candidate Top-1 with D4 scoring >= kin-only 0.650
  3 Selective: fixed coverage >=0.50, wrong-identity < A1's 0.258

Plus Leave-One-Source-Out: exclude each val source, recompute metrics.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")

from wmeval.latent_features import CropConfig, LatentFeatureExtractor
from wmeval.latent_world_state.residual_predictor import ResidualPredictor
from wmeval.latent_association import calibrate_lower_is_better, calibrate_threshold

CAND = Path("runs/j2_3_2a_alignment/candidates_fixed_query.csv")
SEQCSV = Path("datasets/processed/actor_sequences/samples.csv")
D4 = Path("runs/d4_experimental")
OUT = D4
CACHE = Path("runs/j1_baseline/feature_cache")


def main() -> None:
    cands = list(csv.DictReader(open(CAND, newline="")))
    seqrows = {r["pair_id"]: r for r in csv.DictReader(
        open(SEQCSV, newline=""))}
    ex = LatentFeatureExtractor(model_id="ckpt/dinov2-small",
                                crop_cfg=CropConfig(), cache_dir=CACHE)
    model = ResidualPredictor()
    model.load_state_dict(torch.load(D4 / "d4_lam0.5.pt",
                                     map_location="cpu"))
    model.eval()

    anchors = defaultdict(lambda: {"cands": []})
    for c in cands:
        a = anchors[c["anchor_id"]]
        a["split"] = c["split"]
        a["source"] = c["source_video_id"]
        a["cands"].append({
            "ped": c["candidate_id"], "is_true": bool(int(c["is_true_match"])),
            "tier": int(c["hard_negative_tier"]),
            "delta_s": float(c["candidate_delta_s"]),
            "bbox": json.loads(c["bbox"]),
            "frame": int(c["candidate_timestamp_frame"]),
            "kin": float(c["kinematic_distance"])})

    data = []
    for aid, a in anchors.items():
        srow = seqrows[aid]
        sid = srow["synpedas_sample"]
        f0 = int(sid.split("_f")[-1].split("-")[0])
        vp = f"repos/SynPeDAS/JAAD_crossing_moments/{sid}.mp4"
        try:
            hist = np.stack([ex.frame_feature(vp, f[0] - f0, f[1], sid,
                                              srow["anchor_ped_id"])
                            for f in json.loads(srow["anchor_frames"])])
        except RuntimeError:
            continue
        entry = {"anchor_id": aid, "split": a["split"],
                 "source": a["source"], "hist": hist, "cands": []}
        for c in a["cands"]:
            try:
                feat = ex.frame_feature(vp, c["frame"] - f0, c["bbox"],
                                        sid, c["ped"])
            except RuntimeError:
                continue
            entry["cands"].append({**c, "feat": feat})
        if any(c["is_true"] for c in entry["cands"]):
            data.append(entry)
    ex.flush_cache()
    val = [d for d in data if d["split"] == "val"]
    expl = [d for d in data if d["split"] == "exploratory_test"]
    tr = [d for d in data if d["split"] == "train"]

    # compute scores for every candidate
    with torch.no_grad():
        for e in data:
            tc = next(c for c in e["cands"] if c["is_true"])
            z = model(torch.from_numpy(e["hist"][None].astype(np.float32)),
                      torch.tensor([tc["delta_s"]])).numpy()[0]
            z_last = e["hist"][-1]
            for c in e["cands"]:
                c["pred_cos"] = float(np.dot(z, c["feat"]))
                c["static_cos"] = float(np.dot(z_last, c["feat"]))
                # D4 joint score: -kin + pred_cos (normalized)
                c["d4_score"] = -c["kin"] + c["pred_cos"]

    # ---- Gate 1: M_hard ----------------------------------------------------
    def m_hard(entries, key="pred_cos"):
        vals = []
        for e in entries:
            tc = next(c for c in e["cands"] if c["is_true"])
            negs = [c[key] for c in e["cands"] if c["tier"] == 1]
            if negs:
                vals.append(tc[key] - max(negs))
        return round(float(np.mean(vals)), 4) if vals else None

    # ---- Gate 2: Top-1 with different ranking keys -------------------------
    def top1(entries, rank_key):
        correct = 0
        for e in entries:
            ranked = sorted(e["cands"], key=lambda c: -c[rank_key])
            if ranked[0]["is_true"]:
                correct += 1
        return round(correct / len(entries), 4)

    # ---- Gate 3: Fixed coverage selective -----------------------------------
    def selective(entries, score_key, thr, cov_target=0.50):
        # accept top-1 if score >= thr; adjust thr for target coverage
        scores = []
        for e in entries:
            ranked = sorted(e["cands"], key=lambda c: -c["d4_score"
                              if score_key == "d4" else "kin"])
            top = ranked[0]
            scores.append((top["d4_score"] if score_key == "d4"
                           else -top["kin"], top["is_true"]))
        # find threshold that gives ~cov_target
        sorted_scores = sorted(scores, key=lambda x: -x[0])
        n_accept = int(len(scores) * cov_target)
        if n_accept == 0:
            return None
        accepted = sorted_scores[:n_accept]
        wrong = sum(1 for s, t in accepted if not t)
        return {"n_accepted": n_accept, "n_wrong": wrong,
                "wrong_rate": round(wrong / n_accept, 4),
                "coverage": round(n_accept / len(scores), 4)}

    # ---- LOSO ---------------------------------------------------------------
    def loso(entries, metric_fn):
        sources = list({e["source"] for e in entries})
        results = {}
        for src in sources:
            subset = [e for e in entries if e["source"] != src]
            if len(subset) >= 5:
                results[src] = metric_fn(subset)
        return results

    print(f"[eval] val={len(val)} expl={len(expl)}")
    print(f"\n=== Gate 1: Identity (M_hard) ===")
    mh_d4 = m_hard(val, "pred_cos")
    mh_b0 = m_hard(val, "static_cos")
    print(f"  D4 pred: {mh_d4} | B0 static: {mh_b0} | "
          f"gate: {'PASS' if mh_d4 > 0.0339 else 'FAIL'} (threshold 0.0339)")

    print(f"\n=== Gate 2: Top-1 Ranking ===")
    t_kin = top1(val, "kin")  # lower is better — invert
    # kin is distance: sort ascending
    def top1_kin(entries):
        c = 0
        for e in entries:
            ranked = sorted(e["cands"], key=lambda x: x["kin"])
            if ranked[0]["is_true"]:
                c += 1
        return round(c / len(entries), 4)
    t_kin = top1_kin(val)
    t_static = top1(val, "static_cos")
    t_d4 = top1(val, "pred_cos")
    t_d4_joint = top1(val, "d4_score")
    print(f"  kin-only: {t_kin} | static: {t_static} | "
          f"D4 pred: {t_d4} | D4 joint: {t_d4_joint}")
    print(f"  gate: {'PASS' if t_d4_joint >= t_kin else 'FAIL'} "
          f"(threshold {t_kin})")

    print(f"\n=== Gate 3: Selective (fixed coverage 0.50) ===")
    s_a1 = selective(val, "kin", None)
    s_d4 = selective(val, "d4", None)
    print(f"  A1 (kin): {s_a1}")
    print(f"  D4 (joint): {s_d4}")
    gate3 = (s_d4 and s_a1 and
             s_d4["wrong_rate"] < s_a1["wrong_rate"])
    print(f"  gate: {'PASS' if gate3 else 'FAIL'}")

    print(f"\n=== LOSO (val) ===")
    loso_mh = loso(val, lambda e: m_hard(e, "pred_cos"))
    loso_t1 = loso(val, top1_kin)
    print(f"  M_hard without each source:")
    for src in sorted(loso_mh):
        print(f"    -{src}: {loso_mh[src]}")
    print(f"  Top-1 without each source:")
    for src in sorted(loso_t1):
        print(f"    -{src}: {loso_t1[src]}")

    # Gate 4: stability — max deviation from overall
    overall_mh = mh_d4
    max_dev = max(abs(v - overall_mh) for v in loso_mh.values())
    print(f"\n=== Gate 4: LOSO Stability ===")
    print(f"  overall M_hard: {overall_mh} | max LOSO deviation: {max_dev}")
    gate4 = max_dev < 0.03  # not driven by single source
    print(f"  gate: {'PASS' if gate4 else 'FAIL'}")

    out = {
        "gates": {
            "G1_identity": {"D4": mh_d4, "B0": mh_b0,
                            "threshold": 0.0339,
                            "passed": mh_d4 > 0.0339},
            "G2_ranking": {"kin": t_kin, "static": t_static,
                           "D4_pred": t_d4, "D4_joint": t_d4_joint,
                           "threshold": t_kin,
                           "passed": t_d4_joint >= t_kin},
            "G3_selective": {"A1": s_a1, "D4": s_d4,
                             "passed": bool(gate3)},
            "G4_stability": {"overall": overall_mh,
                             "loso": loso_mh, "max_dev": round(max_dev, 4),
                             "passed": bool(gate4)}},
        "loso_top1": loso_t1,
        "expl_results": {
            "M_hard_d4": m_hard(expl, "pred_cos"),
            "M_hard_b0": m_hard(expl, "static_cos"),
            "top1_kin": top1_kin(expl),
            "top1_d4_joint": top1(expl, "d4_score")},
    }
    (OUT / "d4_evaluation.json").write_text(json.dumps(out, indent=2))
    print(f"\n[eval] -> {OUT}")
    all_pass = all(g["passed"] for g in out["gates"].values())
    print(f"\n{'='*50}")
    print(f"D4 OVERALL: {'ALL GATES PASS' if all_pass else 'NOT ALL GATES PASS'}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
