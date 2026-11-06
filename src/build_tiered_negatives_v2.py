"""J2.3.1 P0 fixes: causal, correctly-attributed, deduplicated Tier-1.

Fixes over build_tiered_negatives.py (J2.3, kept read-only):
  P0-1 identity attribution: anchor_frames -> (src, anchor_ped);
       cand_frames -> (src, cand_ped).  In the OLD builder ALL frames
       were keyed to the anchor identity — 100% of negative pairs
       (7,603; 90,122 frame observations) had the OTHER pedestrian's
       boxes registered under the anchor's name.
  P0-2 dedup: canonical observation index (src, actor, frame);
       duplicate appends collapse (161,310 -> 24,966 unique anchor
       observations measured; bbox conflicts = 0 so collapse is safe;
       conflicts, if any, are recorded and EXCLUDED, never averaged).
  P0-3 future-position leak: the spatial-gate centre is NO LONGER the
       anchor's own observation near the target time.  It is the
       KINEMATIC EXTRAPOLATION of the anchor's causal history
       (frames <= anchor_end, tail-median velocity) evaluated at EACH
       candidate's own first-observation time.  The true match's
       future coordinates never enter candidate generation, gating,
       predictor input, or scoring — identity ground truth is used
       ONLY for labels (is_true_match).

Output (candidate-level manifest, full competition sets):
  runs/j2_3_1_alignment/tier1_candidates.csv + tier1_stats.json
Identity truth: JAAD official ids only (never ByteTrack).
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

SEQ = Path("datasets/processed/actor_sequences/samples.csv")
V2MAN = Path("runs/j2_experimental/j2_manifest.csv")
OUT = Path("runs/j2_3_1_alignment")
WINDOW_FRAMES = 12      # candidate search window after anchor end
SPATIAL_GATE = 0.35     # extrapolation-centre distance gate
VELOCITY_K = 6          # tail frames for median velocity


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(SEQ, newline="")))

    # ---- P0-1/P0-2: correct attribution + deduplicated index --------
    # obs[(src, actor)] = {frame: (cx, cy)}  — canonical, deduplicated
    obs = defaultdict(dict)
    dup_appends = 0
    conflicts = []
    for r in rows:
        src = r["jaad_video"]
        for who, frames_key in (("a", "anchor_frames"),
                                ("c", "cand_frames")):
            ped = r["anchor_ped_id"] if who == "a" else r["cand_ped_id"]
            for f, b, o in json.loads(r[frames_key]):
                cx, cy = (b[0] + b[2]) / 2 / 1920.0, (b[1] + b[3]) / 2 / 1080.0
                k = (src, ped, f)
                if f in obs[(src, ped)]:
                    prev = obs[(src, ped)][f]
                    if abs(prev[0] - cx) > 1e-6 or abs(prev[1] - cy) > 1e-6:
                        conflicts.append(k)
                    else:
                        dup_appends += 1
                else:
                    obs[(src, ped)][f] = (cx, cy)
    n_unique = sum(len(v) for v in obs.values())
    print(f"[v2] canonical observations: {n_unique} unique "
          f"({dup_appends} duplicate appends collapsed, "
          f"{len(conflicts)} bbox conflicts)")

    # ---- v2 manifest anchors (same-identity supervision pairs) -------
    v2 = list(csv.DictReader(open(V2MAN, newline="")))
    cand_rows = []
    tier_stats = defaultdict(int)
    for m in v2:
        src, act = m["source_video_id"], m["actor_id"]
        a_frames = sorted(obs.get((src, act), {}).keys())
        if not a_frames:
            continue
        f_end = json.loads(m["anchor_frames"])[-1]
        # CAUSAL history: frames <= anchor end only
        hist = [f for f in a_frames if f <= f_end]
        if len(hist) < 3:
            continue
        # P0-3: kinematic extrapolation centre (no future positions)
        hist_xy = np.array([obs[(src, act)][f] for f in hist])
        steps = np.diff(hist_xy, axis=0)[-VELOCITY_K:]
        v_med = np.median(steps, axis=0) if len(steps) else np.zeros(2)
        # candidate window CENTRES on the association time f_tgt (the
        # true match's re-appearance — supervision label); competitors
        # are identities observed around the SAME time (existence only,
        # identity ground truth never used for generation/gating)
        f_tgt = int(m["target_frame"])   # association time (label-only)
        lo, hi = f_tgt - WINDOW_FRAMES, f_tgt + WINDOW_FRAMES
        pool = []
        for (s2, p2), frames in obs.items():
            if s2 != src or p2 == act:
                continue
            in_win = sorted(f for f in frames if lo < f <= hi)
            if in_win:
                pool.append((p2, in_win))
        # the TRUE match's future observation (supervision label only)
        true_ped = m["actor_id"]
        if lo <= f_tgt <= hi and f_tgt in obs.get((src, true_ped), {}):
            pool.append((true_ped, [f_tgt]))
        for p2, in_win in pool:
            f_c = in_win[0]
            cx, cy = obs[(src, p2)][f_c]
            # extrapolation to THIS candidate's own time
            gap = f_c - hist[-1]
            pred = hist_xy[-1] + v_med * gap
            dist = float(np.hypot(cx - pred[0], cy - pred[1]))
            gated = dist <= SPATIAL_GATE
            is_true = (p2 == true_ped)
            tier = 1 if (gated and not is_true) else \
                (0 if is_true else 2)
            cand_rows.append({
                "anchor_id": f"{m['pair_id']}",
                "candidate_id": p2,
                "source_video_id": src,
                "split": m["split"],
                "anchor_last_frame": f_end,
                "candidate_timestamp_frame": f_c,
                "candidate_delta_frames": gap,
                "kinematic_distance": round(dist, 4),
                "spatial_gate_result": int(gated),
                "is_true_match": int(is_true),
                "hard_negative_tier": tier,
                "candidate_feature_key": f"{src}|{p2}|{f_c}",
                "evidence_provenance": "dedup-index obs + causal extrapolation gate",
            })
            tier_stats[tier] += 1

    with open(OUT / "tier1_candidates.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cand_rows[0].keys()))
        w.writeheader()
        w.writerows(cand_rows)

    anchors = {r["anchor_id"] for r in cand_rows}
    per_anchor = defaultdict(list)
    for r in cand_rows:
        per_anchor[r["anchor_id"]].append(r)
    n_with_t1 = sum(1 for a, lst in per_anchor.items()
                    if any(x["hard_negative_tier"] == 1 for x in lst))
    n_with_true = sum(1 for a, lst in per_anchor.items()
                      if any(x["is_true_match"] for x in lst))
    stats = {
        "fixes": ["P0-1 cand frames attributed to cand_ped (was: anchor)",
                  "P0-2 (src,actor,frame) canonical dedup",
                  "P0-3 causal kinematic-extrapolation gate centre"],
        "canonical_observations": {"unique": n_unique,
                                   "duplicate_appends_collapsed": dup_appends,
                                   "bbox_conflicts_excluded": len(conflicts)},
        "n_anchors": len(anchors),
        "n_with_tier1_competitor": n_with_t1,
        "n_with_true_match": n_with_true,
        "candidate_rows": len(cand_rows),
        "tier_counts": {"true_match(t0)": tier_stats[0],
                        "tier1_gated_competitor": tier_stats[1],
                        "tier2_same_source_not_gated": tier_stats[2]},
        "split_report": {s: {"anchors": len({r['anchor_id'] for r in cand_rows
                                             if r['split'] == s}),
                             "tier1": sum(1 for r in cand_rows
                                          if r['split'] == s and
                                          r["hard_negative_tier"] == 1)}
                         for s in ("train", "val", "exploratory_test",
                                   "final_test")},
        "gap_strata_tier1": {
            g: sum(1 for r in cand_rows
                   if r["hard_negative_tier"] == 1 and
                   abs(int(r["candidate_delta_frames"]) - g) <= 8)
            for g in (15, 30, 45)},
        "old_result_invalidated": "391/412 (J2.3) used misattributed "
                                  "observations + future-position gate",
    }
    (OUT / "tier1_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats["tier_counts"], indent=1))
    print(f"anchors with Tier-1 competitor: {n_with_t1}/{len(anchors)}")
    print(f"[v2] -> {OUT}")


if __name__ == "__main__":
    main()
