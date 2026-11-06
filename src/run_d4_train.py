"""D4: Actor-Conditioned Discriminative Query training.

Architecture: ResidualPredictor (same as B4 — frozen DINOv2 + Residual
GRU 384→128→384, causal (history, delta) input).

Loss: L_prediction (cosine to true future) + λ_rank * L_rank
      (Tier-1 hardest-negative margin ranking, correct sign).

Training data: J2.3.2a frozen manifest — per anchor:
  history (12 causal frames), true match feature, Tier-1 negative
  features (variable count, masked ranking).

Model selection: val Tier-1 M_hard (NOT val loss).
λ ∈ {0.1, 0.3, 0.5, 1.0} — selected on val only, Final Test untouched.

Protocol: offline, fixed-query(1.0s) with 0.2s max decision delay.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, "src")

from wmeval.latent_features import CropConfig, LatentFeatureExtractor
from wmeval.latent_world_state.residual_predictor import ResidualPredictor
from wmeval.latent_world_state.ranking_loss import (
    prediction_loss, ranking_loss_masked)

CAND = Path("runs/j2_3_2a_alignment/candidates_fixed_query.csv")
V2 = Path("runs/j2_experimental")
OUT = Path("runs/d4_experimental")
CACHE = Path("runs/j1_baseline/feature_cache")
SEQCSV = Path("datasets/processed/actor_sequences/samples.csv")
CFG = {"seed": 51, "batch_size": 16, "lr": 3e-4, "weight_decay": 0.01,
       "epochs_max": 60, "patience": 8, "grad_clip": 1.0,
       "lambdas": [0.1, 0.3, 0.5, 1.0], "margin": 0.1}
MAX_NEG_K = 8       # padding size for variable negatives


def main() -> None:
    torch.manual_seed(CFG["seed"])
    np.random.seed(CFG["seed"])
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- load candidates + extract features (cache-first) --------------
    cands = list(csv.DictReader(open(CAND, newline="")))
    seqrows = {r["pair_id"]: r for r in csv.DictReader(
        open(SEQCSV, newline=""))}
    ex = LatentFeatureExtractor(model_id="ckpt/dinov2-small",
                                crop_cfg=CropConfig(), cache_dir=CACHE)

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

    # extract features per anchor: history + all candidates
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
    print(f"[D4] {len(data)} anchors with features "
          f"({sum(1 for d in data if d['split']=='train')} train / "
          f"{sum(1 for d in data if d['split']=='val')} val)")

    # ---- prepare batch tensors ------------------------------------------
    def make_batch(entries):
        """Returns (hist, delta_true, true_feat, neg_feats, neg_mask)."""
        B = len(entries)
        hist = np.stack([e["hist"] for e in entries])
        deltas, trues, negs, masks = [], [], [], []
        for e in entries:
            tc = next(c for c in e["cands"] if c["is_true"])
            deltas.append(tc["delta_s"])
            trues.append(tc["feat"])
            neg_list = [c["feat"] for c in e["cands"]
                        if c["tier"] == 1]  # Tier-1 hardest only
            n_neg = len(neg_list)
            # pad to MAX_NEG_K (zero-vec padding, mask=False)
            while len(neg_list) < MAX_NEG_K:
                neg_list.append(np.zeros(384, dtype=np.float32))
            neg_list = neg_list[:MAX_NEG_K]
            negs.append(np.stack(neg_list))
            m = np.zeros(MAX_NEG_K, dtype=bool)
            m[:min(n_neg, MAX_NEG_K)] = True
            masks.append(m)
        return (torch.from_numpy(hist.astype(np.float32)),
                torch.tensor(deltas, dtype=torch.float32),
                torch.from_numpy(np.stack(trues).astype(np.float32)),
                torch.from_numpy(np.stack(negs).astype(np.float32)),
                torch.from_numpy(np.stack(masks)))

    train_data = [d for d in data if d["split"] == "train"]
    val_data = [d for d in data if d["split"] == "val"]

    # ---- M_hard evaluation (pure Tier-1, J2.3.2a protocol) --------------
    def m_hard(model, entries, key="pred_cos"):
        model.eval()
        vals = []
        with torch.no_grad():
            for e in entries:
                tc = next(c for c in e["cands"] if c["is_true"])
                z = model(torch.from_numpy(e["hist"][None].astype(np.float32)),
                          torch.tensor([tc["delta_s"]])).numpy()[0]
                negs = [c["feat"] for c in e["cands"] if c["tier"] == 1]
                if negs:
                    vals.append(float(np.dot(z, tc["feat"])) -
                                max(float(np.dot(z, n)) for n in negs))
        return round(float(np.mean(vals)), 4) if vals else None

    # ---- train for each lambda -------------------------------------------
    results = {}
    best_overall = {"lambda": None, "val_m_hard": -1.0}
    for lam in CFG["lambdas"]:
        tag = f"d4_lam{lam}"
        print(f"\n[D4] training λ={lam} ...")
        model = ResidualPredictor()
        opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"],
                                weight_decay=CFG["weight_decay"])
        t0 = time.perf_counter()
        best_mh, best_state, wait = -1.0, None, 0
        log = []
        for ep in range(CFG["epochs_max"]):
            model.train()
            perm = np.random.permutation(len(train_data))
            losses = []
            for k in range(0, len(perm), CFG["batch_size"]):
                b = [train_data[i] for i in perm[k:k + CFG["batch_size"]]]
                if len(b) < 2:
                    continue
                hist, delta, true_f, neg_f, mask = make_batch(b)
                omega = model(hist, delta)
                lp = prediction_loss(omega, true_f)
                s_pos = (omega * true_f).sum(-1)
                s_neg = torch.einsum("bd,bkd->bk", omega, neg_f)
                lr_ = ranking_loss_masked(s_pos, s_neg, mask, CFG["margin"])
                loss = lp + lam * lr_
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
                opt.step()
                losses.append(float(loss))
            mh = m_hard(model, val_data)
            log.append({"epoch": ep,
                        "train_loss": round(float(np.mean(losses)), 5),
                        "val_m_hard": mh})
            if mh is not None and mh > best_mh:
                best_mh, wait = mh, 0
                best_state = {kk: vv.detach().cpu().clone()
                              for kk, vv in model.state_dict().items()}
            else:
                wait += 1
                if wait >= CFG["patience"]:
                    break
        if best_state:
            model.load_state_dict(best_state)
        torch.save(model.state_dict(), OUT / f"{tag}.pt")
        elapsed = round(time.perf_counter() - t0)
        vr = round(torch.cuda.max_memory_allocated() / 1e9, 3) \
            if torch.cuda.is_available() else 0
        results[str(lam)] = {
            "best_val_m_hard": best_mh, "epochs": len(log),
            "seconds": elapsed, "vram_gb": vr, "log": log[-5:]}
        print(f"  λ={lam}: best M_hard={best_mh} ({len(log)} epochs, "
              f"{elapsed}s)")
        if best_mh > best_overall["val_m_hard"]:
            best_overall = {"lambda": lam, "val_m_hard": best_mh,
                            "model": model}
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

    (OUT / "d4_training.json").write_text(json.dumps({
        "config": CFG, "results": results,
        "best_lambda": best_overall["lambda"],
        "best_val_m_hard": best_overall["val_m_hard"],
        "baseline_B0": 0.0339,
        "gate_1_passed": best_overall["val_m_hard"] > 0.0339,
    }, indent=2))
    print(f"\n[D4] best: λ={best_overall['lambda']} "
          f"M_hard={best_overall['val_m_hard']} "
          f"(B0=0.0339, gate={'PASS' if best_overall['val_m_hard'] > 0.0339 else 'FAIL'})")
    print(f"[D4] -> {OUT}")


if __name__ == "__main__":
    main()
