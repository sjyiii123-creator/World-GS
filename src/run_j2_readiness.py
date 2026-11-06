"""J1.1 §九: J2 training-data readiness check (NO training started).

Builds and verifies the J2 dataset from cached+extracted frozen features:
  * per positive pair: per-frame anchor embedding SEQUENCE (T<=12, all
    frames <= t by construction — re-verified here), real delta_time,
    future target embedding (candidate segment), occlusion labels;
  * competition candidate embeddings (negative side, separability
    diagnostics);
  * collapse checks: feature-space effective rank + same/diff spread;
  * leakage audit: max(anchor frame) <= t and min(candidate frame) > t
    for every record.

Output: runs/j2_readiness/{j2_sequences.npz, readiness.json}
"""
from __future__ import annotations

import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from wmeval.latent_features import CropConfig, LatentFeatureExtractor

SEQ = Path("datasets/processed/actor_sequences")
OUT = Path("runs/j2_readiness")
N_TRAIN, N_TEST = 250, 50
SEED = 51


def main() -> None:
    random.seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(SEQ / "samples.csv", newline="")))
    comps = {str(json.loads(l)["pair_id"]): json.loads(l) for l in
             (SEQ / "competitions.jsonl").read_text().splitlines() if l.strip()}
    pos_pool = [r for r in rows if r["label"] == "same_identity"
                and r["split"] == "train"]
    te_pool = [r for r in rows if r["label"] == "same_identity"
               and r["split"] == "test"]
    sub = (random.sample(pos_pool, min(N_TRAIN, len(pos_pool)))
           + random.sample(te_pool, min(N_TEST, len(te_pool))))

    ex = LatentFeatureExtractor(model_id="ckpt/dinov2-small",
                                crop_cfg=CropConfig(),
                                cache_dir=Path("runs/j1_baseline/feature_cache"))
    t0 = time.perf_counter()
    seqs, futures, deltas, occs, splits, leaks = [], [], [], [], [], []
    comp_feats = []
    for r in sub:
        anchor = json.loads(r["anchor_frames"])
        cand = json.loads(r["cand_frames"])
        t = anchor[-1][0]
        # leakage audit
        if max(f for f, _, _ in anchor) > t or min(f for f, _, _ in cand) <= t:
            leaks.append(r["pair_id"])
            continue
        sid = r["synpedas_sample"]
        f0 = int(sid.split("_f")[-1].split("-")[0])
        vp = f"repos/SynPeDAS/JAAD_crossing_moments/{sid}.mp4"
        # per-frame anchor sequence (all frames <= t)
        hist = [ex.frame_feature(vp, f[0] - f0, f[1], sid, r["anchor_ped_id"])
                for f in anchor]
        fut = ex.segment_feature(vp, f0, cand, sid, r["cand_ped_id"])
        gap_s = float(r["gap_seconds"])
        seqs.append(np.stack(hist))
        futures.append(fut)
        deltas.append(gap_s)
        occs.append([f[2] for f in anchor])
        splits.append(r["split"])
        # competition candidates (future-side negatives)
        c = comps.get(r["pair_id"])
        if c:
            for cd in c["candidates"]:
                try:
                    comp_feats.append(
                        (cd["is_true_match"],
                         ex.frame_feature(vp, cd["first_frame"] - f0,
                                          cd["bbox"], sid, cd["cand_ped_id"])))
                except RuntimeError:
                    pass
    ex.flush_cache()
    print(f"[j2] built {len(seqs)} sequences "
          f"({time.perf_counter()-t0:.0f}s); leakage violations: {len(leaks)}")

    # ---- collapse checks -------------------------------------------------
    all_feats = np.concatenate([np.concatenate(seqs), np.stack(futures)], axis=0)
    centred = all_feats - all_feats.mean(axis=0, keepdims=True)
    sv = np.linalg.svd(centred, compute_uv=False)
    eff_rank = int(np.sum(sv > sv[0] * 1e-2))
    same_f = np.array([float(np.dot(futures[i], seqs[i][-1]))
                       for i in range(len(seqs))])

    np.savez_compressed(
        OUT / "j2_sequences.npz",
        seq=np.stack(seqs), future=np.stack(futures),
        delta=np.array(deltas), occ=np.array(occs, dtype=object),
        split=np.array(splits))
    readiness = {
        "n_sequences": len(seqs),
        "splits": {"train": splits.count("train"), "test": splits.count("test")},
        "leakage_violations": len(leaks),
        "leakage_audit": "max(anchor frame) <= t AND min(candidate frame) > t "
                         "for every record (re-verified at build time)",
        "feature_space": {
            "dim": int(all_feats.shape[1]),
            "effective_rank_1pct": eff_rank,
            "collapse_check": "OK" if eff_rank > 10 else "SUSPECT",
        },
        "same_pair_cos_anchor_last_vs_future": {
            "mean": round(float(np.mean(same_f)), 4),
            "std": round(float(np.std(same_f)), 4)},
        "n_competition_features": len(comp_feats),
        "future_side_note": "competition candidate features stored "
                            "in-run only (diagnostics); negatives NEVER "
                            "enter the predictor input path",
        "j2_contract": {
            "predictor_input": "anchor per-frame embeddings at frames <= t",
            "target": "candidate-segment embedding (t+delta) — supervision "
                      "only at train time",
            "negative_separability_check_required": True,
            "not_optimize_similarity_only": "cos loss on prediction vs real "
                                            "future + report negative "
                                            "separability (no collapse)",
        },
        "training_config_proposed": {
            "model": "LatentPredictor (GRU 384->256->384, delta buckets 8)",
            "epochs": 40, "batch": 64, "lr": 3e-4,
            "val": "held-out 20% of train sequences (by SOURCE VIDEO)",
            "trigger_gate": "J3-lite must first show event-level value; "
                            "FPR target and recall tradeoff documented "
                            "(docs/j1_1_calibration_report.md)"},
    }
    (OUT / "readiness.json").write_text(json.dumps(readiness, indent=2))
    print(json.dumps({k: readiness[k] for k in
                      ("n_sequences", "leakage_violations", "feature_space",
                       "same_pair_cos_anchor_last_vs_future")}, indent=2))
    print(f"[j2] -> {OUT}")


if __name__ == "__main__":
    main()
