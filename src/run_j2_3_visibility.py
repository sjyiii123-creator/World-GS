"""J2.3 G3 experiment: V0-V3 reference-selection comparison (no training).

For each v2-val sequence: build the anchor's (timestamp, occ, feature)
history, select the reference by V0/V1/V2/V3, then compare the
reference against the sequence's own target (positive) and same-source
different-actor targets (hard negatives) — same identity-mask protocol
as G1.  Memory records (age/status/skip) are emitted for the audit
contract.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from wmeval.latent_features import CropConfig, LatentFeatureExtractor
from wmeval.latent_world_state.visibility_memory import (
    ReferenceMemory, select_reference, update_memory, visibility_class)

V2 = Path("runs/j2_experimental")
OUT = Path("runs/j2_3_diagnostics")
CACHE = Path("runs/j1_baseline/feature_cache")
SEQCSV = Path("datasets/processed/actor_sequences/samples.csv")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with np.load(V2 / "j2_sequences_v2.npz", allow_pickle=False) as z:
        seq = z["seq"].astype(np.float32)
        fut = z["future"].astype(np.float32)
        delta = z["delta"].astype(np.float32)
        split = z["split"]
        srcs = [str(x) for x in z["source_video_id"]]
        actors = [str(x) for x in z["actor_id"]]
    man = {r["pair_id"]: r for r in csv.DictReader(
        open(V2 / "j2_manifest.csv", newline=""))}
    seqrows = {r["pair_id"]: r for r in csv.DictReader(
        open(SEQCSV, newline=""))}
    ex = LatentFeatureExtractor(model_id="ckpt/dinov2-small",
                                crop_cfg=CropConfig(), cache_dir=CACHE)
    idx = np.flatnonzero(split == "val")
    keys = [(srcs[i], actors[i]) for i in idx]
    vs = [srcs[i] for i in idx]

    # per-strategy references
    refs = {s: [] for s in ("V0", "V1", "V2", "V3")}
    records = []
    for i in idx:
        pr = seqrows  # lookup by pair id via manifest
        m = None
        for mm in man.values():
            if int(mm["row_index"]) == i:
                m = mm
                break
        sid = m["synpedas_sample"]
        f0 = int(sid.split("_f")[-1].split("-")[0])
        fps = float(m["fps"])
        vp = f"repos/SynPeDAS/JAAD_crossing_moments/{sid}.mp4"
        srow = seqrows[m["pair_id"]]
        anchor = json.loads(srow["anchor_frames"])
        frames = []
        for f, bb, occ in anchor:
            feat = ex.frame_feature(vp, f - f0, bb, sid,
                                    srow["anchor_ped_id"])
            frames.append((f / fps, occ, feat))
        for s in refs:
            ref, meta = select_reference(frames, s)
            refs[s].append(ref)
            records.append({"pair_id": m["pair_id"], "strategy": s, **meta})
        # audit-contract record for the memory module (V1 semantics)
        mem = ReferenceMemory(actor_key=keys[len(records) // 4 - 1
                                             if False else 0]
                              if False else (srcs[i], actors[i]),
                              reference_feature=None,
                              reference_timestamp=None)
        for t, occ, feat in frames:
            mem = update_memory(mem, {"timestamp": t,
                                      "visibility": visibility_class(occ),
                                      "feature": feat})
        records[-1]["memory_record"] = mem.to_record()
    ex.flush_cache()

    results = {}
    n = len(idx)
    for s, R in refs.items():
        R = np.stack(R)
        cos = R @ fut[idx].T
        same_m = np.eye(n, dtype=bool)
        hard_m = np.zeros((n, n), dtype=bool)
        for a in range(n):
            for b in range(n):
                if a != b and vs[a] == vs[b] and keys[a] != keys[b]:
                    hard_m[a, b] = True
        margins = [cos[i, i] - max(cos[i, j] for j in range(n)
                                   if hard_m[i, j])
                   for i in range(n) if hard_m[i].any()]
        results[s] = {
            "positive_cos": round(float(cos[same_m].mean()), 4),
            "hard_negative_cos": round(float(cos[hard_m].mean()), 4)
            if hard_m.any() else None,
            "positive_hard_margin": round(float(np.mean(margins)), 4)
            if margins else None,
            "n_hard_rows": len(margins)}
    (OUT / "visibility_v0_v3.json").write_text(json.dumps(
        {"results": results,
         "memory_records_sample": records[:8],
         "contract": "reference never fabricated: skipped updates keep "
                     "the OLD reference timestamp; age/status always "
                     "advance"}, indent=2))
    print(json.dumps(results, indent=1))
    print(f"[vis] -> {OUT}")


if __name__ == "__main__":
    main()
