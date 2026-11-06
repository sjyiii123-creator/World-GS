"""D4.1: Fix G3 (kin sort direction), time contract, joint baselines.

Fixes:
  1. kin sorts ASCENDING (best match first); similarity sorts DESCENDING.
  2. Each candidate uses its OWN delta_s for prediction (not the true
     match's delta for all candidates).
  3. Joint baseline comparison: Kin+Static and Kin+B4.
  4. G4: compare D4 vs Kin, D4 vs Kin+Static, D4 vs Kin+B4.
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

CAND = Path("runs/j2_3_2a_alignment/candidates_fixed_query.csv")
SEQCSV = Path("datasets/processed/actor_sequences/samples.csv")
D4 = Path("runs/d4_experimental")
OUT = Path("runs/d4_1_reevaluation")
CACHE = Path("runs/j1_baseline/feature_cache")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cands = list(csv.DictReader(open(CAND, newline="")))
    seqrows = {r["pair_id"]: r for r in csv.DictReader(
        open(SEQCSV, newline=""))}
    ex = LatentFeatureExtractor(model_id="ckpt/dinov2-small",
                                crop_cfg=CropConfig(), cache_dir=CACHE)

    # load D4 model (λ=0.5) and B4 model (frozen)
    d4 = ResidualPredictor()
    d4.load_state_dict(torch.load(D4 / "d4_lam0.5.pt", map_location="cpu"))
    d4.eval()
    b4 = ResidualPredictor()
    b4.load_state_dict(torch.load(
        Path("runs/j2_2_experimental/b4_residual_expanded.pt"),
        map_location="cpu"))
    b4.eval()

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

    # compute scores for every candidate: static, b4, d4
    with torch.no_grad():
        for e in data:
            z_last = e["hist"][-1]
            for c in e["cands"]:
                # each candidate uses its OWN delta_s
                delta_c = c["delta_s"]
                z_d4 = d4(torch.from_numpy(
                    e["hist"][None].astype(np.float32)),
                    torch.tensor([delta_c])).numpy()[0]
                z_b4 = b4(torch.from_numpy(
                    e["hist"][None].astype(np.float32)),
                    torch.tensor([delta_c])).numpy()[0]
                c["d4_cos"] = float(np.dot(z_d4, c["feat"]))
                c["b4_cos"] = float(np.dot(z_b4, c["feat"]))
                c["static_cos"] = float(np.dot(z_last, c["feat"]))

    def rank_and_select(entries, score_key, top_n=None):
        """Returns (top1_correct, n_accept, n_wrong, accepted_list)."""
        n = len(entries)
        top1 = 0
        accepted = []
        for e in entries:
            # ascending for kin (distance), descending for similarity
            if score_key == "kin":
                ranked = sorted(e["cands"], key=lambda c: c["kin"])
            else:
                ranked = sorted(e["cands"], key=lambda c: -c[score_key])
            top = ranked[0]
            top1 += top["is_true"]
            accepted.append((top[score_key], top["is_true"]))
        # sort accepted by score (descending for similarity, ascending for kin)
        if score_key == "kin":
            accepted.sort(key=lambda x: x[0])   # ascending kin
        else:
            accepted.sort(key=lambda x: -x[0])  # descending similarity
        if top_n:
            accepted = accepted[:top_n]
        n_wrong = sum(1 for _, t in accepted if not t)
        return top1, len(accepted), n_wrong, accepted

    def evaluate(entries):
        val = [e for e in entries if e["split"] == "val"]
        expl = [e for e in entries if e["split"] == "exploratory_test"]
        results = {}
        for key in ("kin", "static_cos", "b4_cos", "d4_cos"):
            t1, na, nw, acc = rank_and_select(val, key, top_n=None)
            # fixed 50% coverage
            _, na50, nw50, acc50 = rank_and_select(val, key,
                                                   top_n=int(len(val) * 0.5))
            results[key] = {"top1": round(t1 / len(val), 4),
                            "coverage_50": round(na50 / len(val), 4),
                            "wrong_50": round(nw50 / na50, 4)
                            if na50 else None}
        return results

    # compute joint scores: kin+static and kin+b4
    with torch.no_grad():
        for e in data:
            for c in e["cands"]:
                c["kin_static"] = -c["kin"] + c["static_cos"]
                c["kin_b4"] = -c["kin"] + c["b4_cos"]
                c["kin_d4"] = -c["kin"] + c["d4_cos"]
    results_joint = evaluate(data)

    # save per-candidate audit CSV
    audit_rows = []
    for e in data:
        for c in e["cands"]:
            audit_rows.append({
                "anchor_id": e["anchor_id"],
                "split": e["split"],
                "source": e["source"],
                "ped": c["ped"],
                "is_true": int(c["is_true"]),
                "tier": c["tier"],
                "kin": c["kin"],
                "static_cos": round(c["static_cos"], 4),
                "b4_cos": round(c["b4_cos"], 4),
                "d4_cos": round(c["d4_cos"], 4),
                "delta_s": round(c["delta_s"], 3),
            })
    import csv as _csv
    with open(OUT / "audit_candidates.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(audit_rows[0].keys()))
        w.writeheader()
        w.writerows(audit_rows)

    # evaluate all seven methods
    methods = {
        "kin": "kin",
        "static": "static_cos",
        "kin_static": "kin_static",
        "b4": "b4_cos",
        "kin_b4": "kin_b4",
        "d4": "d4_cos",
        "kin_d4": "kin_d4",
    }
    out = {"methods": {}, "per_anchor": []}
    for name, key in methods.items():
        out["methods"][name] = evaluate(data)
    # per-anchor breakdown for val
    val_entries = [e for e in data if e["split"] == "val"]
    for e in val_entries:
        row = {"anchor_id": e["anchor_id"], "source": e["source"]}
        for name, key in methods.items():
            ranked = sorted(e["cands"], key=lambda c: c["kin"]) if key == "kin" \
                else sorted(e["cands"], key=lambda c: -c[key])
            row[f"{name}_top1"] = int(ranked[0]["is_true"])
        out["per_anchor"].append(row)

    (OUT / "reeval_results.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["methods"], indent=1))
    print(f"[reeval] -> {OUT}")


if __name__ == "__main__":
    main()
