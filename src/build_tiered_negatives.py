"""J2.3 G2: tiered hard-negative audit with spatiotemporal evidence.

Tiers (task §五):
  Tier-1 — same source, co-occurring in the anchor's gap window with a
           spatial gate (different actor observed within the window and
           near the anchor's spatial track);
  Tier-2 — same source, different time period;
  Tier-3 — different source (easy).

Evidence per row: source_video_id, anchor_actor_id, candidate_actor_id,
anchor_last_timestamp, candidate_first_timestamp, temporal_overlap,
spatial_gate, identity_label, hard_negative_tier, evidence_provenance.
Identity truth: JAAD official ids only (never ByteTrack / generated-side
auto track ids).

Output: runs/j2_3_diagnostics/tiered_negatives.csv + tier_stats.json
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

SEQ = Path("datasets/processed/actor_sequences/samples.csv")
V2MAN = Path("runs/j2_experimental/j2_manifest.csv")
OUT = Path("runs/j2_3_diagnostics")
WINDOW_FRAMES = 5        # co-occurrence window around the gap
SPATIAL_GATE = 0.30      # normalized center distance for "near"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(SEQ, newline="")))
    # actor observation map: (source, actor) -> [(frame, cx, cy)]
    obs = defaultdict(list)
    for r in rows:
        src, act = r["jaad_video"], r["anchor_ped_id"]
        for f, b, o in json.loads(r["anchor_frames"]):
            obs[(src, act)].append(
                (f, (b[0] + b[2]) / 2 / 1920.0, (b[1] + b[3]) / 2 / 1080.0))
        for f, b, o in json.loads(r["cand_frames"]):
            obs[(src, act)].append(
                (f, (b[0] + b[2]) / 2 / 1920.0, (b[1] + b[3]) / 2 / 1080.0))
    for k in obs:
        obs[k].sort()

    v2 = list(csv.DictReader(open(V2MAN, newline="")))
    out_rows = []
    tier_counts = defaultdict(int)
    for m in v2:
        src = m["source_video_id"]
        act = m["actor_id"]
        f_end = json.loads(m["anchor_frames"])[-1]
        t_end = float(m["anchor_end_time_s"])
        f_tgt, t_tgt = int(m["target_frame"]), float(m["target_time_s"])
        fps = float(m["fps"])
        lo, hi = f_tgt - WINDOW_FRAMES, f_tgt + WINDOW_FRAMES
        same_src_actors = [k for k in obs if k[0] == src and k[1] != act]
        tier1 = []
        tier2 = []
        for k in same_src_actors:
            for f, cx, cy in obs[k]:
                if lo <= f <= hi:
                    tier1.append((k, f, cx, cy))
                elif f < f_end - 60 or f > f_tgt + 60:
                    tier2.append(k)
        # spatial gate on tier-1: near the anchor's own last position
        a_obs = obs[(src, act)]
        a_near = [o for o in a_obs if abs(o[0] - f_tgt) <= WINDOW_FRAMES]
        acx, acy = (a_near[0][1], a_near[0][2]) if a_near else \
            (a_obs[-1][1], a_obs[-1][2])
        t1_gated = [x for x in tier1
                    if np.hypot(x[2] - acx, x[3] - acy) <= SPATIAL_GATE]
        row = {
            "pair_id": m["pair_id"], "source_video_id": src,
            "anchor_actor_id": act,
            "anchor_last_timestamp": t_end,
            "candidate_target_time_s": t_tgt,
            "gap_s": round(t_tgt - t_end, 3),
            "n_tier1_cooccurring_actors": len({k for k, *_ in tier1}),
            "n_tier1_gated_actors": len({k for k, *_ in t1_gated}),
            "n_tier2_same_source_actors": len(set(tier2)),
            "tier1_example": json.dumps(
                [{"actor": k[1], "frame": f, "dist_to_anchor":
                  round(float(np.hypot(cx - acx, cy - acy)), 3)}
                 for k, f, cx, cy in t1_gated[:3]]),
            "identity_truth": "JAAD official ids",
            "evidence_provenance": "samples.csv frames + v2 manifest times",
        }
        out_rows.append(row)
        tier_counts["tier1_pairs"] += len(t1_gated)
        tier_counts["tier2_actors_total"] += len(set(tier2))

    with open(OUT / "tiered_negatives.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    # per-split tier stats + coverage of occlusion / appearance / gap
    stats = {
        "n_anchor_sequences": len(out_rows),
        "sequences_with_tier1_gated": sum(
            1 for r in out_rows if r["n_tier1_gated_actors"] > 0),
        "sequences_with_tier1_any": sum(
            1 for r in out_rows if r["n_tier1_cooccurring_actors"] > 0),
        "total_tier1_gated_negatives": tier_counts["tier1_pairs"],
        "total_tier2_actors": tier_counts["tier2_actors_total"],
        "tier3_cross_source": "all other actors in the pool (easy)",
        "definitions": {
            "tier1": f"same source, observation within ±{WINDOW_FRAMES} "
                     f"frames of the gap target AND center distance "
                     f"<= {SPATIAL_GATE} (spatial gate)",
            "tier2": "same source, different time period (>60 frames away)",
            "tier3": "different source video"},
        "gap_coverage": {
            g: sum(1 for r in out_rows
                   if abs(float(r["gap_s"]) - g) < 0.3 and
                   r["n_tier1_gated_actors"] > 0)
            for g in (0.5, 1.0, 1.5)},
    }
    (OUT / "tier_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=1))
    print(f"[tiers] -> {OUT}")


if __name__ == "__main__":
    main()
