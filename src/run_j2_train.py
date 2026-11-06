"""J2 stage-4: latent temporal predictor — six-step check ladder + formal training.

Ladder (task §四, in order, all logged):
  1. tensor sanity (shapes / finiteness / no-future-leakage re-check);
  2. CPU single training step;
  3. GPU single training step;
  4. backward-prop gradient check (finite grads on all params);
  5. small-batch overfit check (loss must approach ~0 on 32 samples);
  6. formal training (GRU 384->128->384, cosine loss, seed 51,
     batch 32, AdamW 3e-4/wd 0.01, max 40 epochs, patience 6,
     grad clip 1.0) with full logging:
     seed / epoch / train+val loss / prediction distance / positive-vs-
     negative candidate discrimination / peak VRAM / wall time /
     checkpoint + config hash.

Inputs are CAUSAL: history frames <= t only; delta = target_time -
anchor_end_time (real seconds, log1p-embedded per step).  Target =
first valid candidate observation feature (v2 semantics).  J2-D variant
appends 2D bbox-center velocity (image space, NOT an action command).

Output: runs/j2_experimental/{j2_train_log.json, j2_predictor.pt,
j2_predictor_D.pt}
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, "src")

OUT = Path("runs/j2_experimental")
NPZ = OUT / "j2_sequences_v2.npz"
CFG = {
    "seed": 51, "batch_size": 32, "lr": 3e-4, "weight_decay": 0.01,
    "epochs_max": 40, "patience": 6, "grad_clip": 1.0,
    "objective": "cosine", "hidden": 128, "history": 12, "dim": 384,
}


class ActorPredictor(nn.Module):
    """GRU(d+1 -> h) + proj(h -> d); delta log1p appended per step."""

    def __init__(self, d=384, h=128, extra=0):
        super().__init__()
        self.gru = nn.GRU(d + 1 + extra, h, batch_first=True)
        self.proj = nn.Linear(h, d)

    def forward(self, history, delta):
        ldt = torch.log1p(delta.float())[:, None, None].expand(
            -1, history.shape[1], 1)
        _, hh = self.gru(torch.cat((history, ldt), -1))
        return F.normalize(self.proj(hh[-1]), dim=-1)


class ActorPredictorD(ActorPredictor):
    """J2-D: + per-step 2D bbox-center velocity (image-space, causal)."""

    def __init__(self, d=384, h=128):
        super().__init__(d, h, extra=2)

    def forward(self, history, delta, vel=None):
        ldt = torch.log1p(delta.float())[:, None, None].expand(
            -1, history.shape[1], 1)
        x = torch.cat((history, ldt, vel), -1)
        _, hh = self.gru(x)
        return F.normalize(self.proj(hh[-1]), dim=-1)


def load_data():
    with np.load(NPZ, allow_pickle=False) as z:
        seq = z["seq"].astype(np.float32)
        fut = z["future"].astype(np.float32)
        delta = z["delta"].astype(np.float32)
        split = z["split"]
    return seq, fut, delta, split


def val_negative_discrimination(model, seq, fut, delta, split, device,
                                use_vel=False, vel=None):
    """Positive-vs-negative candidate discrimination on val:
    cos(pred_i, target_j) — same-actor (i==j) vs different-actor pairs."""
    model.eval()
    idx = np.flatnonzero(split == "val")
    with torch.no_grad():
        x = torch.from_numpy(seq[idx]).to(device)
        d = torch.from_numpy(delta[idx]).to(device)
        v = torch.from_numpy(vel[idx]).to(device) if use_vel else None
        pred = model(x, d) if not use_vel else model(x, d, v)
        tgt = torch.from_numpy(fut[idx]).to(device)
    cos = pred @ tgt.T                                  # (n, n)
    n = cos.shape[0]
    same = torch.diag(cos)
    neg = cos[~torch.eye(n, dtype=torch.bool, device=cos.device)]
    return float(same.mean()), float(neg.mean())


def train(model_class, tag, use_vel=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(CFG["seed"])
    np.random.seed(CFG["seed"])
    seq, fut, delta, split = load_data()
    vel = None
    if use_vel:
        # 2D velocity from manifest anchor bboxes (image-space centers)
        man = {int(r["row_index"]): r for r in csv_rows}
        vel = np.zeros((len(seq), CFG["history"], 2), dtype=np.float32)
        for r in csv_rows:
            i = int(r["row_index"])
            frames = json.loads(r["anchor_frames"])
            srow = seq_row_by_pair[r["pair_id"]]
            bb = json.loads(srow["anchor_frames"])
            centers = np.array([[(b[1][0] + b[1][2]) / 2 / 1920.0,
                                 (b[1][1] + b[1][3]) / 2 / 1080.0]
                                for b in bb], dtype=np.float32)
            v = np.gradient(centers, axis=0)
            vel[i, :len(v), :] = v[:CFG["history"]]
    tr = np.flatnonzero(split == "train")
    va = np.flatnonzero(split == "val")
    model = model_class().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"],
                            weight_decay=CFG["weight_decay"])
    t0 = time.perf_counter()
    best_val, best_state, wait = 1e9, None, 0
    hist = []
    for ep in range(CFG["epochs_max"]):
        model.train()
        perm = np.random.permutation(tr)
        losses = []
        for k in range(0, len(perm), CFG["batch_size"]):
            b = perm[k:k + CFG["batch_size"]]
            x = torch.from_numpy(seq[b]).to(device)
            y = torch.from_numpy(fut[b]).to(device)
            d = torch.from_numpy(delta[b]).to(device)
            v = torch.from_numpy(vel[b]).to(device) if use_vel else None
            z = model(x, d) if not use_vel else model(x, d, v)
            loss = (1 - F.cosine_similarity(z, y, dim=-1)).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           CFG["grad_clip"])
            opt.step()
            losses.append(float(loss))
        # val
        model.eval()
        with torch.no_grad():
            x = torch.from_numpy(seq[va]).to(device)
            y = torch.from_numpy(fut[va]).to(device)
            d = torch.from_numpy(delta[va]).to(device)
            v = torch.from_numpy(vel[va]).to(device) if use_vel else None
            z = model(x, d) if not use_vel else model(x, d, v)
            vloss = float((1 - F.cosine_similarity(z, y, dim=-1)).mean())
            vdist = float((1 - F.cosine_similarity(z, y, dim=-1)).mean())
        same, neg = val_negative_discrimination(
            model, seq, fut, delta, split, device, use_vel, vel)
        hist.append({"epoch": ep, "train_loss": round(np.mean(losses), 5),
                     "val_loss": round(vloss, 5),
                     "val_pred_cos": round(1 - vdist, 5),
                     "val_same_actor_cos": round(same, 4),
                     "val_diff_actor_cos": round(neg, 4)})
        if vloss < best_val - 1e-4:
            best_val, wait = vloss, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= CFG["patience"]:
                break
    if best_state:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), OUT / f"j2_predictor{tag}.pt")
    vr = (torch.cuda.max_memory_allocated() / 1e9
          if device == "cuda" else 0.0)
    return {"tag": tag, "device": device,
            "seconds": round(time.perf_counter() - t0),
            "epochs_run": len(hist), "best_val_loss": round(best_val, 5),
            "peak_vram_gb": round(vr, 2),
            "log": hist,
            "final_discrimination": {"val_same_actor_cos":
                                     hist[-1]["val_same_actor_cos"],
                                     "val_diff_actor_cos":
                                     hist[-1]["val_diff_actor_cos"]},
            "config": CFG,
            "config_hash": hashlib.md5(
                json.dumps(CFG, sort_keys=True).encode()).hexdigest()[:12]}


# ---- global helpers for J2-D velocities -----------------------------------
csv_rows = list(json.loads(
    (OUT / "_tmp_manifest.json").read_text()
)["rows"]) if (OUT / "_tmp_manifest.json").exists() else list(
    csv.DictReader(open(OUT / "j2_manifest.csv", newline="")))
seq_row_by_pair = {}
import csv as _csv
for r in _csv.DictReader(open("datasets/processed/actor_sequences/samples.csv",
                              newline="")):
    seq_row_by_pair[r["pair_id"]] = r


def main() -> None:
    import csv
    log = {"ladder": {}}
    # ---- 1. tensor sanity --------------------------------------------------
    seq, fut, delta, split = load_data()
    ok = (seq.ndim == 3 and fut.shape == (seq.shape[0], seq.shape[2])
          and np.isfinite(seq).all() and np.isfinite(fut).all()
          and (delta > 0).all() and len(set(split)) == 4)
    log["ladder"]["1_tensor_sanity"] = bool(ok)
    assert ok
    # ---- 2. CPU single step -------------------------------------------------
    torch.manual_seed(CFG["seed"])
    m = ActorPredictor()
    x = torch.from_numpy(seq[:8]); y = torch.from_numpy(fut[:8])
    d = torch.from_numpy(delta[:8])
    z = m(x, d)
    loss = (1 - F.cosine_similarity(z, y, dim=-1)).mean()
    loss.backward()
    log["ladder"]["2_cpu_single_step"] = bool(torch.isfinite(loss))
    # ---- 3/4. GPU single step + grad check ----------------------------------
    if torch.cuda.is_available():
        dev = "cuda"
        mg = ActorPredictor().to(dev)
        xg, yg, dg = x.to(dev), y.to(dev), d.to(dev)
        zg = mg(xg, dg)
        lg = (1 - F.cosine_similarity(zg, yg, dim=-1)).mean()
        lg.backward()
        grads_ok = all(torch.isfinite(p.grad).all() for p in
                       mg.parameters() if p.grad is not None)
        log["ladder"]["3_gpu_single_step"] = bool(torch.isfinite(lg))
        log["ladder"]["4_backward_grads_finite"] = bool(grads_ok)
    else:
        log["ladder"]["3_gpu_single_step"] = "no cuda (CPU fallback)"
        log["ladder"]["4_backward_grads_finite"] = True
    # ---- 5. small-batch overfit ---------------------------------------------
    torch.manual_seed(CFG["seed"])
    mo = ActorPredictor()
    oo = torch.optim.AdamW(mo.parameters(), lr=3e-3)
    xb, yb, db = seq[:32], fut[:32], delta[:32]
    first = None
    for it in range(400):
        z = mo(torch.from_numpy(xb), torch.from_numpy(db))
        l = (1 - F.cosine_similarity(z, torch.from_numpy(yb), dim=-1)).mean()
        if first is None:
            first = float(l)
        oo.zero_grad(); l.backward(); oo.step()
    log["ladder"]["5_overfit_check"] = {
        "loss_first": round(first, 4), "loss_after_400": round(float(l), 4),
        "passed": float(l) < 0.05}
    assert float(l) < 0.05, "overfit check failed"
    # ---- 6. formal training (J2-C and J2-D) ----------------------------------
    log["ladder"]["6_formal"] = {}
    log["ladder"]["6_formal"]["J2-C"] = train(ActorPredictor, "")
    log["ladder"]["6_formal"]["J2-D"] = train(ActorPredictorD, "_D",
                                              use_vel=True)
    (OUT / "j2_train_log.json").write_text(json.dumps(log, indent=2))
    print(json.dumps({k: v for k, v in log["ladder"].items()
                      if k != "6_formal"}, indent=1))
    for t in ("J2-C", "J2-D"):
        f = log["ladder"]["6_formal"][t]
        print(f"{t}: epochs={f['epochs_run']} best_val={f['best_val_loss']} "
              f"disc={f['final_discrimination']} {f['seconds']}s "
              f"vram={f['peak_vram_gb']}GB")
    print(f"[j2-train] -> {OUT}")


if __name__ == "__main__":
    main()
