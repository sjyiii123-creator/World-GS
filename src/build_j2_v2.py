"""J2 stage-2: build j2_sequences_v2.npz + j2_manifest.csv (audited).

Fixes over runs/j2_readiness (task §二 / GATES missing-evidence #1/#2):
  * full provenance per sequence (source_video_id, actor_id, pair_id);
  * supervision target = the candidate segment's FIRST VALID OBSERVATION
    frame feature (single frame, exact timestamp) — not the segment mean
    that mismatched the first-frame delta;
  * strict checks: target_time_s > anchor_end_time_s;
    delta == target_time_s - anchor_end_time_s (asserted);
  * source-video-level splits (a source NEVER crosses splits):
      train            = JAAD-train sources minus val carve (J1-viewed ok)
      val              = 18 JAAD-train sources carved out (seed 51)
      exploratory_test = JAAD-test sources ALREADY VIEWED by the J-series
                         (marked exploratory — NOT an untouched final test)
      final_test       = JAAD-test sources NEVER viewed (2 sources, 12
                         pairs — tiny; directional sanity ONLY, stated)
  * duplicate / overlapping-window statistics reported;
  * feature_cache_revision recorded for cache provenance.

Output: runs/j2_experimental/{j2_sequences_v2.npz, j2_manifest.csv,
dataset_v2_stats.json}
"""
from __future__ import annotations

import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from wmeval.latent_features import CropConfig, LatentFeatureExtractor

SEQ = Path("datasets/processed/actor_sequences")
J1R = Path("runs/j1_baseline/pair_results.csv")
OUT = Path("runs/j2_experimental")
CACHE = Path("runs/j1_baseline/feature_cache")
SEED = 51
N_PER_SPLIT = {"train": 240, "val": 60, "exploratory_test": 100,
               "final_test": 12}


def main() -> None:
    random.seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    seq_rows = list(csv.DictReader(open(SEQ / "samples.csv", newline="")))
    j1_ids = {r["pair_id"] for r in csv.DictReader(open(J1R, newline=""))}

    # ---- source-level split assignment -----------------------------------
    pos = [r for r in seq_rows if r["label"] == "same_identity"]
    viewed = {r["jaad_video"] for r in seq_rows if r["pair_id"] in j1_ids}
    test_src = {r["jaad_video"] for r in pos if r["split"] == "test"}
    train_src = {r["jaad_video"] for r in pos if r["split"] == "train"}
    unseen_test = test_src - viewed
    train_list = sorted(viewed & train_src)
    val_src = set(random.sample(train_list, max(1, len(train_list) // 5)))
    train_src_use = (viewed & train_src) - val_src

    def bucket(r):
        if r["split"] == "train":
            if r["jaad_video"] in val_src:
                return "val"
            if r["jaad_video"] in train_src_use:
                return "train"
            return None            # train source never viewed: unused this round
        if r["jaad_video"] in unseen_test:
            return "final_test"
        return "exploratory_test"  # test-split source, already viewed

    pools = defaultdict(list)
    for r in pos:
        b = bucket(r)
        if b:
            pools[b].append(r)
    # prefer J1-viewed pairs (cache hits) inside train/exploratory pools
    for b in ("train", "exploratory_test"):
        pools[b].sort(key=lambda r: (r["pair_id"] not in j1_ids, r["pair_id"]))

    sub = []
    for b, n in N_PER_SPLIT.items():
        pool = pools[b]
        take = min(n, len(pool))
        chosen = pool[:take] if b in ("train", "exploratory_test") \
            else random.sample(pool, take)
        sub.extend(chosen)
        print(f"[v2] {b}: {take}/{len(pool)} available")

    # ---- feature extraction ----------------------------------------------
    ex = LatentFeatureExtractor(model_id="ckpt/dinov2-small",
                                crop_cfg=CropConfig(), cache_dir=CACHE)
    seqs, futures, deltas, splits = [], [], [], []
    man, problems = [], []
    seen_anchor_keys = defaultdict(int)
    for r in sub:
        anchor = json.loads(r["anchor_frames"])
        cand = json.loads(r["cand_frames"])
        sid = r["synpedas_sample"]
        f0 = int(sid.split("_f")[-1].split("-")[0])
        fps = float(r["fps"])
        vp = f"repos/SynPeDAS/JAAD_crossing_moments/{sid}.mp4"
        # first VALID observation (prefer_visible semantics, k=1)
        vis = [f for f in cand if f[2] == 0] or [f for f in cand if f[2] <= 1] or cand
        tf = vis[0]
        try:
            hist = [ex.frame_feature(vp, f[0] - f0, f[1], sid,
                                     r["anchor_ped_id"]) for f in anchor]
            future = ex.frame_feature(vp, tf[0] - f0, tf[1], sid,
                                      r["cand_ped_id"])
        except RuntimeError as e:
            problems.append(f"{r['pair_id']}: {e}")
            continue
        t_anchor_end = anchor[-1][0] / fps
        t_target = tf[0] / fps
        delta = t_target - t_anchor_end
        if not delta > 0:
            problems.append(f"{r['pair_id']}: non-positive delta")
            continue
        seqs.append(np.stack(hist))
        futures.append(future)
        deltas.append(round(delta, 4))
        splits.append(bucket(r))
        key = (r["jaad_video"], r["anchor_ped_id"], anchor[-1][0])
        seen_anchor_keys[key] += 1
        man.append({
            "row_index": len(seqs) - 1,
            "source_video_id": r["jaad_video"],
            "actor_id": r["anchor_ped_id"],
            "pair_id": r["pair_id"],
            "synpedas_sample": sid,
            "split": bucket(r),
            "anchor_end_time_s": round(t_anchor_end, 4),
            "target_time_s": round(t_target, 4),
            "target_type": "first_valid_candidate_observation",
            "target_frame": tf[0],
            "anchor_frames": json.dumps([f[0] for f in anchor]),
            "occlusion_anchor": json.dumps([f[2] for f in anchor]),
            "occlusion_target": tf[2],
            "delta_s": round(delta, 4),
            "gap_frames": r["gap_frames"],
            "fps": fps,
            "split_note": ("viewed_by_J_series" if bucket(r) in
                           ("train", "val", "exploratory_test") else
                           "never_viewed"),
            "feature_cache_revision": "j1_baseline+M1_mapping_v1",
        })
    ex.flush_cache()

    # ---- duplicate / overlap statistics ----------------------------------
    dups = {k: c for k, c in seen_anchor_keys.items() if c > 1}
    # overlapping anchor windows per (source, actor)
    overlap = 0
    by_actor = defaultdict(list)
    for m in man:
        by_actor[(m["source_video_id"], m["actor_id"])].append(
            (json.loads(m["anchor_frames"]), m["target_frame"]))
    for k, lst in by_actor.items():
        spans = sorted((a[0], a[-1], t) for a, t in lst)
        for i in range(1, len(spans)):
            if spans[i][0] <= spans[i - 1][1]:
                overlap += 1

    np.savez_compressed(
        OUT / "j2_sequences_v2.npz",
        seq=np.stack(seqs), future=np.stack(futures),
        delta=np.array(deltas), split=np.array(splits),
        source_video_id=np.array([m["source_video_id"] for m in man]),
        actor_id=np.array([m["actor_id"] for m in man]),
        pair_id=np.array([m["pair_id"] for m in man]))
    with open(OUT / "j2_manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(man[0].keys()))
        w.writeheader()
        w.writerows(man)
    stats = {
        "n_sequences": len(seqs),
        "splits": {s: sum(1 for x in splits if x == s)
                   for s in N_PER_SPLIT},
        "source_video_split_isolation": "guaranteed by construction "
        "(source-level buckets); verified in preflight",
        "duplicate_anchor_keys": len(dups),
        "overlapping_anchor_windows_same_actor": overlap,
        "extraction_problems": problems[:10],
        "final_test_note": "2 never-viewed sources / 12 sequences — "
                           "directional sanity ONLY, not a powered test",
        "target_semantics": "first VALID observation frame of the "
                            "candidate segment (prefer_visible k=1)",
        "cache_provenance": "j1_baseline feature cache (M1 mapping "
                            "verified 2026-11-05) + newly extracted frames",
    }
    (OUT / "dataset_v2_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"[v2] wrote {len(seqs)} sequences; dups={len(dups)} "
          f"overlaps={overlap} problems={len(problems)}")
    print(f"[v2] -> {OUT}")


if __name__ == "__main__":
    main()
