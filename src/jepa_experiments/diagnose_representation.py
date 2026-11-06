"""J2.3 G1: read-only representation diagnostics for B0/B1/B2/B3/B4.

Ten metrics (task §四) on the SAME validation samples (v2 val split,
actor-balanced sampling so repeated actors don't dominate):
  1 channel variance (mean over dims, RAW features)
  2 off-diagonal covariance energy (RAW, centered)
  3 Participation Ratio of the covariance spectrum (FULL val set —
    never mini-batch rank, which is min(n-1, d) by construction)
  4 mean similarity of predictions to the common mean direction
  5 same-identity cosine
  6 different-identity cosine
  7 same-scene hard-negative cosine (different actor, same source)
  8 Positive-Hard Margin  M_hard = E[s(zhat, z+) - max_hard s(zhat, z-)]
  9 gap strata (0.5/1.0/1.5 s)
 10 output validity (finite, unit norm)

Constant-feature counterexample (task §四 MUST): a fully collapsed
representation z_i = c scores ZERO covariance loss — demonstrating
that covariance decorrelation alone cannot prevent collapse; identity
separation must be measured directly (metrics 5-8).

Raw vs L2-normalized statistics are reported separately (task §四).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")

from run_j2_train import ActorPredictor
from wmeval.latent_world_state.residual_predictor import ResidualPredictor

V2 = Path("runs/j2_experimental")
J22 = Path("runs/j2_2_experimental")
OUT = Path("runs/j2_3_diagnostics")


def load_val():
    with np.load(V2 / "j2_sequences_v2.npz", allow_pickle=False) as z:
        seq = z["seq"].astype(np.float32)
        fut = z["future"].astype(np.float32)
        delta = z["delta"].astype(np.float32)
        split = z["split"]
        srcs = [str(x) for x in z["source_video_id"]]
        actors = [str(x) for x in z["actor_id"]]
    idx = np.flatnonzero(split == "val")
    return seq[idx], fut[idx], delta[idx], \
        [srcs[i] for i in idx], [actors[i] for i in idx]


def actor_balanced(keys):
    """One representative sequence per (source, actor)."""
    seen = set()
    keep = []
    for i, k in enumerate(keys):
        if k not in seen:
            seen.add(k)
            keep.append(i)
    return np.array(keep)


def load_models():
    m = {}
    b1 = ActorPredictor()
    b1.load_state_dict(torch.load(V2 / "j2_predictor.pt", map_location="cpu"))
    b1.eval()
    m["B1_v1_GRU"] = b1
    b2 = ResidualPredictor()
    b2.load_state_dict(torch.load(J22 / "b2_residual.pt", map_location="cpu"))
    b2.eval()
    m["B2_residual_240"] = b2
    b3 = ActorPredictor()
    b3.load_state_dict(torch.load(J22 / "b3_gru_expanded.pt",
                                  map_location="cpu"))
    b3.eval()
    m["B3_GRU_800"] = b3
    b4 = ResidualPredictor()
    b4.load_state_dict(torch.load(J22 / "b4_residual_expanded.pt",
                                  map_location="cpu"))
    b4.eval()
    m["B4_residual_800"] = b4
    return m


@torch.no_grad()
def predict(model, seq, delta):
    return model(torch.from_numpy(seq), torch.from_numpy(delta)).numpy()


def raw_stats(feat):
    x = torch.from_numpy(feat).float()
    x = x - x.mean(dim=0, keepdim=True)
    n = x.shape[0]
    cov = x.T @ x / (n - 1)
    offdiag = cov - torch.diag(torch.diag(cov))
    variance = torch.diag(cov)
    offdiag_energy = (offdiag.square().sum() / x.shape[1]).item()
    eig = torch.linalg.eigvalsh(cov).clamp_min(0)
    pr = (eig.sum() ** 2 / eig.square().sum().clamp_min(1e-12)).item()
    return {"mean_channel_variance_raw": round(variance.mean().item(), 8),
            "offdiag_covariance_raw": round(offdiag_energy, 8),
            "participation_ratio_full_set": round(pr, 2)}


def identity_stats(pred, target, keys, srcs):
    n = len(keys)
    cos = pred @ target.T
    same_m = np.zeros((n, n), dtype=bool)
    hard_m = np.zeros((n, n), dtype=bool)
    for i in range(n):
        same_m[i, i] = True          # prediction vs its OWN target
        for j in range(n):
            if i != j:
                if keys[i] == keys[j]:
                    same_m[i, j] = True
                elif srcs[i] == srcs[j]:
                    hard_m[i, j] = True
    same = cos[same_m].mean() if same_m.any() else None
    diff = cos[~same_m & ~np.eye(n, dtype=bool)].mean()
    hard = cos[hard_m].mean() if hard_m.any() else None
    # M_hard: E[s(zhat, z+) - max_hard s(zhat, z-)]
    margins = []
    for i in range(n):
        hard_js = [j for j in range(n) if hard_m[i, j]]
        if hard_js:
            margins.append(cos[i, i] - max(cos[i, j] for j in hard_js))
    m_hard = float(np.mean(margins)) if margins else None
    return {"same_identity_cos": round(float(same), 4) if same is not None else None,
            "diff_identity_cos": round(float(diff), 4),
            "hard_negative_cos": round(float(hard), 4) if hard is not None else None,
            "positive_hard_margin": round(m_hard, 4) if m_hard is not None else None,
            "n_hard_rows": len(margins)}


def mean_direction_cos(pred):
    p = torch.from_numpy(pred).float()
    mean_dir = torch.nn.functional.normalize(p.mean(dim=0), dim=0)
    return round(float((p @ mean_dir).mean()), 4)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    seq, fut, delta, srcs, actors = load_val()
    keys = [(s, a) for s, a in zip(srcs, actors)]
    bal = actor_balanced(keys)
    print(f"[diag] val n={len(keys)}, actor-balanced n={len(bal)}")
    sb, fb, db = seq[bal], fut[bal], delta[bal]
    kb = [keys[i] for i in bal]
    vb = [srcs[i] for i in bal]

    models = load_models()
    results = {"protocol": {
        "samples": "v2 val split, actor-balanced (one seq per identity)",
        "n_full": int(len(keys)), "n_balanced": int(len(bal)),
        "raw_vs_normalized": "raw stats on un-normalized; identity stats "
                             "on L2-normalized (model outputs are "
                             "normalized by construction)"},
        "models": {}}

    # B0 static: prediction = last history frame (raw stats on RAW
    # features as they come from the encoder cache = already L2-normed;
    # report as-is with note)
    b0_pred = sb[:, -1, :]
    results["models"]["B0_static"] = {
        **raw_stats(b0_pred),
        **identity_stats(b0_pred, fb, kb, vb),
        "mean_direction_cos": mean_direction_cos(b0_pred),
        "gap_strata": {},
        "validity": "finite/unit-norm (encoder output)"}

    for name, model in models.items():
        pred = predict(model, sb, db)
        strata = {}
        for g0, g1, tag in ((0.4, 0.75, "0.5s"), (0.9, 1.25, "1.0s"),
                            (1.4, 1.9, "1.5s")):
            m_ = (db >= g0) & (db <= g1)
            if m_.sum() >= 3:
                sub = identity_stats(pred[m_], fb[m_],
                                     [kb[i] for i in np.flatnonzero(m_)],
                                     [vb[i] for i in np.flatnonzero(m_)])
                strata[tag] = {"same_cos": sub["same_identity_cos"],
                               "hard_cos": sub["hard_negative_cos"]}
        finite = bool(np.isfinite(pred).all())
        unit = bool(np.allclose(np.linalg.norm(pred, axis=1), 1.0, atol=1e-3))
        results["models"][name] = {
            **raw_stats(pred), **identity_stats(pred, fb, kb, vb),
            "mean_direction_cos": mean_direction_cos(pred),
            "gap_strata": strata,
            "validity": f"finite={finite} unit_norm={unit}"}

    # constant-feature counterexample (task §四 MUST)
    c = torch.nn.functional.normalize(torch.randn(1, 384), dim=-1
                                      ).repeat(len(sb), 1).numpy()
    const_stats = raw_stats(c)
    const_stats["note"] = ("CONSTANT-feature counterexample: fully "
                           "collapsed representation scores ZERO "
                           "off-diagonal covariance (and zero variance) "
                           "— covariance decorrelation ALONE cannot "
                           "prevent identity collapse; identity "
                           "separation must be measured directly")
    const_stats["identity"] = identity_stats(c, fb, kb, vb)
    results["constant_feature_counterexample"] = const_stats

    (OUT / "representation_diagnostics.json").write_text(
        json.dumps(results, indent=2))
    for name, r in results["models"].items():
        print(f"[{name}] var={r['mean_channel_variance_raw']:.2e} "
              f"PR={r['participation_ratio_full_set']} "
              f"meandir={r['mean_direction_cos']} "
              f"same={r['same_identity_cos']} hard={r['hard_negative_cos']} "
              f"M_hard={r['positive_hard_margin']}")
    print(f"[const] {const_stats['mean_channel_variance_raw']} "
          f"offdiag={const_stats['offdiag_covariance_raw']} "
          f"(zero-cov collapse demo)")
    print(f"[diag] -> {OUT}")


if __name__ == "__main__":
    main()
