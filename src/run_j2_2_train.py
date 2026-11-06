"""J2.2 stage-4/6: train B2/B3/B4 + evaluate B0-B4 under one protocol.

B0: zero-residual static baseline (prediction = last history frame;
    evaluation only — exact by construction, verified in selftest).
B1: original GRU @ v2-240 data  (REFERENCE: the v1 checkpoint, already
    re-evaluated under the fixed protocol in J2.1 — cited, not retrained).
B2: ResidualPredictor @ v2-240 data.
B3: original GRU (ActorPredictor) @ expanded-800 data.
B4: ResidualPredictor @ expanded-800 data.

Training config identical to v1 where applicable (seed 51, batch 32,
AdamW 3e-4/wd 0.01, <=40 epochs, patience 6, clip 1.0, cosine loss).
Monitoring (task §五, all logged per epoch):
  train/val loss; same-identity cos; different-identity cos;
  hard (same-source) negative cos; predicted-feature variance;
  predicted-feature effective rank; cos(pred, batch-mean feature);
  ||pred - static|| difference; peak VRAM; wall time.
Model selection: best VAL DISCRIMINATION GAP (not val loss alone).

Unified evaluation on the v2 val split (60 sequences, source-isolated,
fixed identity masks). Output: runs/j2_2_experimental/j2_2_train.json
+ checkpoints b2_residual.pt / b3_gru_expanded.pt / b4_residual_expanded.pt.
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
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, "src")

from run_j2_train import ActorPredictor
from wmeval.latent_world_state.residual_predictor import ResidualPredictor

V2 = Path("runs/j2_experimental")
OUT = Path("runs/j2_2_experimental")
CFG = {"seed": 51, "batch_size": 32, "lr": 3e-4, "weight_decay": 0.01,
       "epochs_max": 40, "patience": 6, "grad_clip": 1.0,
       "objective": "cosine", "hidden": 128}
VAL_NPZ = V2 / "j2_sequences_v2.npz"


def load(npz_path):
    with np.load(npz_path, allow_pickle=False) as z:
        return (z["seq"].astype(np.float32), z["future"].astype(np.float32),
                z["delta"].astype(np.float32), z["split"],
                z["source_video_id"], z["actor_id"])


def identity_keys(srcs, actors):
    return [(str(s), str(a)) for s, a in zip(srcs, actors)]


def monitoring(pred, target, keys, delta, srcs):
    """Ten monitoring metrics (task §五)."""
    p = torch.from_numpy(pred)
    t = torch.from_numpy(target)
    cos = (p @ t.T).numpy()
    n = len(keys)
    same_m = np.zeros((n, n), dtype=bool)
    hard_m = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j:
                if keys[i] == keys[j]:
                    same_m[i, j] = True
                elif srcs[i] == srcs[j]:
                    hard_m[i, j] = True
    same_cos = cos[same_m].mean() if same_m.any() else None
    diff_cos = cos[~same_m & ~np.eye(n, dtype=bool)].mean()
    hard_cos = cos[hard_m].mean() if hard_m.any() else None
    # feature-space health
    centred = pred - pred.mean(0, keepdims=True)
    sv = np.linalg.svd(centred, compute_uv=False)
    eff_rank = int((sv > sv[0] * 1e-2).sum())
    batch_mean = F.normalize(p.mean(dim=0), dim=-1).numpy()
    pred_var = float(pred.std(axis=0).mean())
    mean_cos = float((pred @ batch_mean).mean())
    static_diff = float(np.linalg.norm(pred - target, axis=1).mean())
    return {"same_identity_cos": round(float(same_cos), 4) if same_cos
            is not None else None,
            "diff_identity_cos": round(float(diff_cos), 4),
            "hard_negative_cos": round(float(hard_cos), 4)
            if hard_cos is not None else None,
            "discrimination_gap": round(float(same_cos - diff_cos), 4)
            if same_cos is not None else None,
            "pred_feature_variance": round(pred_var, 6),
            "pred_effective_rank": eff_rank,
            "cos_pred_to_batchmean": round(mean_cos, 4),
            "pred_minus_target_l2": round(static_diff, 4)}


def train_model(model, seq_tr, fut_tr, d_tr, seq_va, fut_va, d_va, tag):
    torch.manual_seed(CFG["seed"])
    np.random.seed(CFG["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"],
                            weight_decay=CFG["weight_decay"])
    t0 = time.perf_counter()
    best_gap, best_state, wait = -9.0, None, 0
    hist = []
    n = len(seq_tr)
    for ep in range(CFG["epochs_max"]):
        model.train()
        perm = np.random.permutation(n)
        losses = []
        for k in range(0, n, CFG["batch_size"]):
            b = perm[k:k + CFG["batch_size"]]
            x = torch.from_numpy(seq_tr[b]).to(device)
            y = torch.from_numpy(fut_tr[b]).to(device)
            d = torch.from_numpy(d_tr[b]).to(device)
            z = model(x, d)
            loss = (1 - F.cosine_similarity(z, y, dim=-1)).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
            opt.step()
            losses.append(float(loss))
        model.eval()
        with torch.no_grad():
            x = torch.from_numpy(seq_va).to(device)
            d = torch.from_numpy(d_va).to(device)
            pred = model(x, d).cpu().numpy()
        # model selection by VAL DISCRIMINATION GAP (not loss alone)
        keys_va = VAL_KEYS
        mon = monitoring(pred, VAL_FUT, keys_va, d_va, VAL_SRCS)
        val_loss = float((1 - F.cosine_similarity(
            torch.from_numpy(pred), torch.from_numpy(VAL_FUT),
            dim=-1)).mean())
        gap = mon["discrimination_gap"]
        if gap is not None and gap > best_gap:
            best_gap, wait = gap, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            wait += 1
        hist.append({"epoch": ep,
                     "train_loss": round(float(np.mean(losses)), 5),
                     "val_loss": round(val_loss, 5), **mon})
        if wait >= CFG["patience"]:
            break
    if best_state:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), OUT / f"{tag}.pt")
    vr = (torch.cuda.max_memory_allocated() / 1e9
          if device == "cuda" else 0.0)
    torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
    return {"tag": tag, "device": device,
            "seconds": round(time.perf_counter() - t0),
            "epochs_run": len(hist), "best_val_gap": round(best_gap, 4),
            "peak_vram_gb": round(vr, 3), "log": hist,
            "final_monitoring": hist[-1]}


def predict_with(model, seq, delta):
    model.eval()
    with torch.no_grad():
        z = model(torch.from_numpy(seq), torch.from_numpy(delta))
    return z.numpy()


# globals for val monitoring
VAL_KEYS = VAL_FUT = VAL_SRCS = None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    v2seq, v2fut, v2d, v2sp, v2src, v2act = load(VAL_NPZ)
    global VAL_KEYS, VAL_FUT, VAL_SRCS
    tr = np.flatnonzero(v2sp == "train")
    va = np.flatnonzero(v2sp == "val")
    VAL_KEYS = identity_keys(v2src[va], v2act[va])
    VAL_FUT = v2fut[va]
    VAL_SRCS = [str(s) for s in v2src[va]]

    results = {"config": CFG, "protocol": {
        "identity": "(source_video_id, actor_id) fixed mask (J2.1)",
        "validation": "v2 val split (60 seq, source-isolated)",
        "model_selection": "best val discrimination gap (not loss)",
        "monitoring": "ten metrics per epoch (task J2.2 §五)"}}

    # ---- B0: zero-residual static baseline (evaluation only) -----------
    static_pred = v2seq[va][:, -1, :]
    b0 = monitoring(static_pred, VAL_FUT, VAL_KEYS, v2d[va], VAL_SRCS)
    results["B0_static_zero_residual"] = b0
    print("[B0]", json.dumps(b0))

    # ---- B1: v1 GRU @ v2-240 (reference, cited from J2.1) --------------
    b1_model = ActorPredictor()
    b1_model.load_state_dict(torch.load(V2 / "j2_predictor.pt",
                                        map_location="cpu"))
    b1_pred = predict_with(b1_model, v2seq[va], v2d[va])
    results["B1_v1_GRU_v2data_REFERENCE"] = monitoring(
        b1_pred, VAL_FUT, VAL_KEYS, v2d[va], VAL_SRCS)
    print("[B1]", json.dumps(results["B1_v1_GRU_v2data_REFERENCE"]))

    # ---- B2: residual GRU @ v2-240 --------------------------------------
    results["B2_residual_v2data"] = train_model(
        ResidualPredictor(), v2seq[tr], v2fut[tr], v2d[tr],
        v2seq[va], v2fut[va], v2d[va], "b2_residual")
    print("[B2] best_gap", results["B2_residual_v2data"]["best_val_gap"],
          "final", results["B2_residual_v2data"]["final_monitoring"])

    # ---- expanded data (B3/B4) -------------------------------------------
    exp = OUT / "j2_sequences_expanded.npz"
    if exp.exists():
        eseq, efut, ed, esp, esrc, eact = load(exp)
        # val stays v2-val (never touched by expanded sources —
        # expansion excluded val/expl/final sources by construction)
        results["B3_v1arch_GRU_expanded"] = train_model(
            ActorPredictor(), eseq, efut, ed,
            v2seq[va], v2fut[va], v2d[va], "b3_gru_expanded")
        print("[B3] best_gap", results["B3_v1arch_GRU_expanded"]["best_val_gap"])
        results["B4_residual_expanded"] = train_model(
            ResidualPredictor(), eseq, efut, ed,
            v2seq[va], v2fut[va], v2d[va], "b4_residual_expanded")
        print("[B4] best_gap", results["B4_residual_expanded"]["best_val_gap"])
    else:
        results["B3_v1arch_GRU_expanded"] = "NOT RUN — expanded data absent"
        results["B4_residual_expanded"] = "NOT RUN — expanded data absent"

    (OUT / "j2_2_train.json").write_text(json.dumps(results, indent=2))
    print(f"[j2.2-train] -> {OUT}")


if __name__ == "__main__":
    main()
