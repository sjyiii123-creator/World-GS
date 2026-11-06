"""D4.1a: Proper seven-method evaluator (no nested duplication).

Each method is evaluated INDEPENDENTLY:
  kin, static, kin_static, b4, kin_b4, d4, kin_d4.
Output per method: top1, coverage_50, wrong_50, accepted list.
Plus per-anchor per-source breakdown with full precision scores.
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
OUT = Path("runs/d4_1a_independent_reevaluation")
CACHE = Path("runs/j1_baseline/feature_cache")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cands = list(csv.DictReader(open(CAND, newline="")))
    seqrows = {r["pair_id"]: r for r in csv.DictReader(
        open(SEQCSV, newline=""))}
    ex = LatentFeatureExtractor(model_id="ckpt/dinov2-small",
                                crop_cfg=CropConfig(), cache_dir=CACHE)

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

    # compute per-candidate scores for each method
    with torch.no_grad():
        for e in data:
            z_last = e["hist"][-1]
            for c in e["cands"]:
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
                c["kin_static"] = -c["kin"] + c["static_cos"]
                c["kin_b4"] = -c["kin"] + c["b4_cos"]
                c["kin_d4"] = -c["kin"] + c["d4_cos"]

    def evaluate_method(entries, score_key, split="val", coverage=0.5):
        """Evaluate ONE method independently."""
        subset = [e for e in entries if e["split"] == split]
        top1 = 0
        accepted = []
        for e in subset:
            if score_key == "kin":
                ranked = sorted(e["cands"], key=lambda c: c["kin"])
            else:
                ranked = sorted(e["cands"], key=lambda c: -c[score_key])
            top = ranked[0]
            top1 += top["is_true"]
            accepted.append((top[score_key], top["is_true"], e["anchor_id"]))
        n = len(subset)
        top1_acc = round(top1 / n, 4) if n else None
        # fixed coverage: take top-K by score
        if score_key == "kin":
            accepted.sort(key=lambda x: x[0])   # ascending kin
        else:
            accepted.sort(key=lambda x: -x[0])  # descending similarity
        n_accept = max(1, int(n * coverage))
        accepted_top = accepted[:n_accept]
        wrong = sum(1 for _, t, _ in accepted_top if not t)
        return {
            "top1_count": top1, "top1_accuracy": top1_acc,
            "n_accepted": n_accept, "n_wrong": wrong,
            "wrong_rate": round(wrong / n_accept, 4) if n_accept else None,
            "coverage": round(n_accept / n, 4),
            "accepted": accepted_top,
        }

    # evaluate all 7 methods
    methods = {
        "kin": "kin",
        "static": "static_cos",
        "kin_static": "kin_static",
        "b4": "b4_cos",
        "kin_b4": "kin_b4",
        "d4": "d4_cos",
        "kin_d4": "kin_d4",
    }
    results = {}
    for name, key in methods.items():
        results[name] = {
            "val": evaluate_method(data, key, "val", 0.5),
            "expl": evaluate_method(data, key, "exploratory_test", 0.5),
        }

    # per-anchor breakdown for val
    val_entries = [e for e in data if e["split"] == "val"]
    per_anchor = []
    for e in val_entries:
        row = {"anchor_id": e["anchor_id"], "source": e["source"]}
        for name, key in methods.items():
            if key == "kin":
                ranked = sorted(e["cands"], key=lambda c: c["kin"])
            else:
                ranked = sorted(e["cands"], key=lambda c: -c[key])
            top = ranked[0]
            row[f"{name}_top1"] = int(top["is_true"])
            row[f"{name}_score"] = round(float(top[key]), 4)
            row[f"{name}_ped"] = top["ped"]
        per_anchor.append(row)

    # per-source breakdown
    by_src = defaultdict(list)
    for e in val_entries:
        by_src[e["source"]].append(e)
    per_source = []
    for src in sorted(by_src):
        rows = by_src[src]
        rec = {"source": src, "n_anchors": len(rows)}
        for name, key in methods.items():
            correct = 0
            for e in rows:
                if key == "kin":
                    ranked = sorted(e["cands"], key=lambda c: c["kin"])
                else:
                    ranked = sorted(e["cands"], key=lambda c: -c[key])
                if ranked[0]["is_true"]:
                    correct += 1
            rec[f"{name}_correct"] = correct
            rec[f"{name}_top1"] = round(correct / len(rows), 4)
        per_source.append(rec)

    (OUT / "seven_methods.json").write_text(json.dumps(results, indent=2))
    with open(OUT / "per_anchor.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_anchor[0].keys()))
        w.writeheader()
        w.writerows(per_anchor)
    with open(OUT / "per_source.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_source[0].keys()))
        w.writeheader()
        w.writerows(per_source)

    print("=== Seven methods (val) ===")
    for name, r in results.items():
        v = r["val"]
        print(f"  {name}: top1={v['top1_accuracy']} wrong={v['wrong_rate']} "
              f"({v['n_wrong']}/{v['n_accepted']})")
    print("\n=== Per-source (val) ===")
    for rec in per_source:
        print(f"  {rec['source']}: n={rec['n_anchors']} "
              f"kin={rec['kin_correct']} static={rec['static_correct']} "
              f"kin_static={rec['kin_static_correct']} "
              f"d4={rec['d4_correct']} joint={rec['kin_d4_correct']}")
    print(f"\n[eval] -> {OUT}")


if __name__ == "__main__":
    main()
