"""J2.2 stage-3: expanded training data (dedup + source isolation).

Protocol (task §三):
  * full positive-pair pool from the sequence dataset, EXCLUDING the
    v2 val / exploratory / final-test source videos (never re-split);
  * dedup: one representative (anchor, gap) per independent anchor key
    (source, actor, anchor_end_frame) — v2 kept all 3 gap variants per
    anchor (design dimension, 45 dup keys); J2.2 keeps the middle gap
    (30 frames) as representative and reports the removal;
  * statistics: independent sources / actors added vs v2; repeated
    anchors; overlapping windows; shared-real-frame flags;
  * explicit target + delta per sequence (first-valid-observation
    semantics, v2 protocol);
  * feature budget: sample up to N_SEQ independent anchors (default
    800) — extraction ~25 min; the FULL dedup protocol & manifest are
    fixed here so a later full-extraction run extends, not redefines.

Output: runs/j2_2_experimental/{j2_sequences_expanded.npz,
j2_manifest_expanded.csv, data_expansion_stats.json}
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
V2MAN = Path("runs/j2_experimental/j2_manifest.csv")
OUT = Path("runs/j2_2_experimental")
CACHE = Path("runs/j1_baseline/feature_cache")
SEED = 51
N_SEQ = 800           # feature-budget sample of independent anchors
REP_GAP = 30          # representative gap for dedup


def main() -> None:
    random.seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(SEQ / "samples.csv", newline="")))
    pos = [r for r in rows if r["label"] == "same_identity"]
    # v2 split sources: EXCLUDE val/expl/final from training expansion
    v2man = list(csv.DictReader(open(V2MAN, newline="")))
    excluded = {m["source_video_id"] for m in v2man
                if m["split"] in ("val", "exploratory_test", "final_test")}
    v2_train_src = {m["source_video_id"] for m in v2man
                    if m["split"] == "train"}

    pool = [r for r in pos if r["split"] == "train"
            and r["jaad_video"] not in excluded]
    # dedup: one representative per (source, actor, anchor_end)
    by_anchor = defaultdict(list)
    for r in pool:
        anchor_end = json.loads(r["anchor_frames"])[-1][0]
        by_anchor[(r["jaad_video"], r["anchor_ped_id"], anchor_end)].append(r)
    dedup = []
    for key, lst in sorted(by_anchor.items()):
        rep = next((r for r in lst if int(r["gap_frames"]) == REP_GAP),
                   lst[len(lst) // 2])
        dedup.append(rep)

    full_srcs = {r["jaad_video"] for r in dedup}
    full_actors = {(r["jaad_video"], r["anchor_ped_id"]) for r in dedup}
    new_srcs = full_srcs - v2_train_src
    new_actors = full_actors - {(m["source_video_id"], m["actor_id"])
                                for m in v2man if m["split"] == "train"}
    print(f"[expand] full dedup pool: {len(dedup)} independent anchors "
          f"({len(full_srcs)} sources / {len(full_actors)} actors)")
    print(f"[expand] vs v2 train (53 src/160 actor): +{len(new_srcs)} "
          f"new sources, +{len(new_actors)} new actors")

    # feature-budget sampling: keep ALL v2-train-covered anchors whose
    # features are cached plus new ones up to N_SEQ
    j1_ids = {r["pair_id"] for r in csv.DictReader(
        open("runs/j1_baseline/pair_results.csv", newline=""))}
    cached_first = [r for r in dedup if r["pair_id"] in j1_ids]
    uncached = [r for r in dedup if r["pair_id"] not in j1_ids]
    random.shuffle(uncached)
    take = cached_first + uncached
    sub = take[:N_SEQ]
    print(f"[expand] sampled {len(sub)} anchors "
          f"({len(cached_first)} cache-prefilled available, "
          f"{len(sub) - len([r for r in sub if r['pair_id'] in j1_ids])} "
          f"new-extraction)")

    ex = LatentFeatureExtractor(model_id="ckpt/dinov2-small",
                                crop_cfg=CropConfig(), cache_dir=CACHE)
    seqs, futures, deltas, splits = [], [], [], []
    man = []
    problems = []
    for r in sub:
        anchor = json.loads(r["anchor_frames"])
        cand = json.loads(r["cand_frames"])
        sid = r["synpedas_sample"]
        f0 = int(sid.split("_f")[-1].split("-")[0])
        fps = float(r["fps"])
        vp = f"repos/SynPeDAS/JAAD_crossing_moments/{sid}.mp4"
        vis = ([f for f in cand if f[2] == 0]
               or [f for f in cand if f[2] <= 1] or cand)
        tf = vis[0]
        try:
            hist = [ex.frame_feature(vp, f[0] - f0, f[1], sid,
                                     r["anchor_ped_id"]) for f in anchor]
            future = ex.frame_feature(vp, tf[0] - f0, tf[1], sid,
                                      r["cand_ped_id"])
        except RuntimeError as e:
            problems.append(f"{r['pair_id']}: {e}")
            continue
        t_ae = anchor[-1][0] / fps
        t_t = tf[0] / fps
        d = t_t - t_ae
        if not d > 0:
            problems.append(f"{r['pair_id']}: delta<=0")
            continue
        seqs.append(np.stack(hist))
        futures.append(future)
        deltas.append(round(d, 4))
        splits.append("train_expanded")
        man.append({
            "row_index": len(seqs) - 1,
            "source_video_id": r["jaad_video"],
            "actor_id": r["anchor_ped_id"], "pair_id": r["pair_id"],
            "synpedas_sample": sid, "split": "train_expanded",
            "anchor_end_time_s": round(t_ae, 4),
            "target_time_s": round(t_t, 4),
            "target_type": "first_valid_candidate_observation",
            "target_frame": tf[0],
            "anchor_frames": json.dumps([f[0] for f in anchor]),
            "delta_s": round(d, 4), "fps": fps,
            "feature_cache_revision": "j1_baseline+M1_mapping_v1+expand",
        })
    ex.flush_cache()

    np.savez_compressed(
        OUT / "j2_sequences_expanded.npz",
        seq=np.stack(seqs), future=np.stack(futures),
        delta=np.array(deltas), split=np.array(splits),
        source_video_id=np.array([m["source_video_id"] for m in man]),
        actor_id=np.array([m["actor_id"] for m in man]),
        pair_id=np.array([m["pair_id"] for m in man]))
    with open(OUT / "j2_manifest_expanded.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(man[0].keys()))
        w.writeheader()
        w.writerows(man)

    # shared-real-frame flags + overlap stats
    by_actor = defaultdict(list)
    for m in man:
        by_actor[(m["source_video_id"], m["actor_id"])].append(m)
    shared_frame_pairs = 0
    for k, lst in by_actor.items():
        spans = sorted((json.loads(m["anchor_frames"]),
                        m["target_frame"]) for m in lst)
        for i in range(1, len(spans)):
            if spans[i][0][0] <= spans[i - 1][0][-1]:
                shared_frame_pairs += 1
    stats = {
        "full_dedup_pool": {"independent_anchors": len(dedup),
                            "sources": len(full_srcs),
                            "actors": len(full_actors)},
        "vs_v2_train": {"new_sources": len(new_srcs),
                        "new_actors": len(new_actors)},
        "sampled_for_training": {"n": len(seqs),
                                 "sources": len({m['source_video_id']
                                                 for m in man}),
                                 "actors": len({(m['source_video_id'],
                                                 m['actor_id'])
                                                for m in man}),
                                 "budget_note": f"feature-extraction "
                                 f"budget sample of {N_SEQ} independent "
                                 f"anchors; full protocol fixed, "
                                 f"extendable"},
        "dedup_rule": f"one representative gap ({REP_GAP}f) per "
                      "(source,actor,anchor_end); v2 kept 3 gap variants",
        "overlapping_windows_same_actor": shared_frame_pairs,
        "extraction_problems": len(problems),
        "identity_truth": "JAAD official ped ids (never ByteTrack)",
        "excluded_sources": f"{len(excluded)} (v2 val/expl/final, "
                            "not re-split)",
    }
    (OUT / "data_expansion_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"[expand] wrote {len(seqs)} sequences "
          f"({stats['sampled_for_training']['sources']} src / "
          f"{stats['sampled_for_training']['actors']} actors); "
          f"problems={len(problems)}")
    print(f"[expand] -> {OUT}")


if __name__ == "__main__":
    main()
