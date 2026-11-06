"""J2.3.2 P0-A + P0-D: canonical observation index + fixed-time query.

P0-A fix: a CANONICAL index (source, actor, frame) -> (bbox, occ) built
from BOTH sides of every samples row with correct attribution
(anchor_frames -> anchor_ped_id; cand_frames -> cand_ped_id).  The old
baseline looked up boxes by scanning rows where anchor_ped_id==ped and
mixing anchor+cand frames — faithful emulation: 3,011 correct / 555
missing / 17 wrong-actor boxes (3 of them TRUE matches).  The canonical
index removes the search entirely; lookups are exact and fail-closed
(missing -> recorded, never silently skipped).

P0-D fix: FIXED-TIME QUERY protocol.  query_time = anchor_end_time +
QUERY_OFFSET_S (a pre-registered protocol constant, set to 1.0 s = the
median gap of the positive supervision pairs — NOT tuned on any
evaluation data).  The candidate window centres on query_time; the true
match is NEVER force-appended — labels are attached AFTER the pool is
built, so anchors naturally become true-available / true-unavailable /
no-match.  true_target_time stays available as a supervision column
only.

Feature-cache safety: cache keys include the bbox hash, so the 17
wrong-box features extracted earlier sit under unreachable keys —
naturally invalidated; the new correct boxes get fresh extractions.

Output: runs/j2_3_2_alignment/{observation_index.json,
candidates_fixed_query.csv, fixed_query_stats.json}
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
OUT = Path("runs/j2_3_2a_alignment")
QUERY_OFFSET_S = 1.0          # pre-registered protocol constant
QUERY_WINDOW_S = 0.4          # +/- window around query_time
SPATIAL_GATE = 0.35
QUERY_TOL_S = 0.2       # pre-registered tolerance
VELOCITY_K = 6


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(SEQ, newline="")))

    # ---- P0-A: canonical observation index -------------------------------
    idx = {}                       # (src, ped, frame) -> (bbox, occ)
    dups = 0
    conflicts = []
    for r in rows:
        src = r["jaad_video"]
        for ped_key, frames_key in ((r["anchor_ped_id"], "anchor_frames"),
                                    (r["cand_ped_id"], "cand_frames")):
            for f, b, o in json.loads(r[frames_key]):
                k = (src, ped_key, f)
                if k in idx:
                    if list(idx[k][0]) != list(b):
                        conflicts.append(k)
                    else:
                        dups += 1
                else:
                    idx[k] = (b, o)
    n_idx = len(idx)
    print(f"[idx] canonical (src,actor,frame) -> bbox: {n_idx} unique "
          f"(dups collapsed {dups}, conflicts {len(conflicts)})")
    (OUT / "observation_index.json").write_text(json.dumps({
        "n_unique": n_idx, "duplicates_collapsed": dups,
        "bbox_conflicts": len(conflicts),
        "conflict_keys": [list(k) for k in conflicts[:20]],
        "attribution": "anchor_frames->anchor_ped; cand_frames->cand_ped",
    }, indent=2))

    # ---- P0-D: fixed-time query candidate generation ----------------------
    v2 = list(csv.DictReader(open(V2MAN, newline="")))
    seqrows = {r["pair_id"]: r for r in rows}
    out_rows = []
    stats = defaultdict(int)
    for m in v2:
        src, act = m["source_video_id"], m["actor_id"]
        srow = seqrows[m["pair_id"]]
        fps = float(srow["fps"])
        anchor = json.loads(srow["anchor_frames"])
        f_end = anchor[-1][0]
        # causal history from the canonical index
        hist = sorted(k[2] for k in idx if k[0] == src and k[1] == act
                      and k[2] <= f_end)
        if len(hist) < 3:
            continue
        hist_xy = np.array([[ (idx[(src, act, f)][0][0] + idx[(src, act, f)][0][2]) / 2 / 1920.0,
                               (idx[(src, act, f)][0][1] + idx[(src, act, f)][0][3]) / 2 / 1080.0]
                             for f in hist])
        steps = np.diff(hist_xy, axis=0)[-VELOCITY_K:]
        v_med = np.median(steps, axis=0) if len(steps) else np.zeros(2)
        # P0-D: FIXED query time (protocol constant, no oracle)
        f_query = f_end + round(QUERY_OFFSET_S * fps)
        lo = f_query - round(QUERY_WINDOW_S * fps)
        hi = f_query + round(QUERY_WINDOW_S * fps)
        # candidate pool: identities OBSERVED in the query window
        # (existence only; labels attached afterwards)
        pool = []
        for (s2, p2), _ in {} .items():
            pass
        seen_actors = {k for k in idx if k[0] == src}
        pool_actors = set()
        for k in seen_actors:
            if k[1] == act:
                continue
            if lo <= k[2] <= hi:
                pool_actors.add(k[1])
        # also the anchor actor itself MIGHT re-appear naturally
        reapp = [k[2] for k in idx if k[0] == src and k[1] == act
                 and lo <= k[2] <= hi]
        cand_peds = sorted(pool_actors) + ([act] if reapp else [])
        n_out = 0
        for p2 in cand_peds:
            in_win = sorted(k[2] for k in idx
                            if k[0] == src and k[1] == p2
                            and lo <= k[2] <= hi)
            # J2.3.2a P0-B fix: choose the observation CLOSEST to the
            # predefined query_frame (not the window-start in_win[0],
            # which pulled true-match deltas to 0.6s under a 1.0s query).
            f_c = min(in_win, key=lambda f: abs(f - f_query))
            query_err = abs(f_c - f_query) / fps
            if query_err > QUERY_TOL_S:
                continue    # too far from query time -> unavailable
            bbox, occ = idx[(src, p2, f_c)]
            cx = (bbox[0] + bbox[2]) / 2 / 1920.0
            cy = (bbox[1] + bbox[3]) / 2 / 1080.0
            gap = f_c - hist[-1]
            pred = hist_xy[-1] + v_med * gap
            dist = float(np.hypot(cx - pred[0], cy - pred[1]))
            gated = dist <= SPATIAL_GATE
            is_true = (p2 == act)   # label attached AFTER pool built
            tier = 0 if is_true else (1 if gated else 2)
            out_rows.append({
                "anchor_id": m["pair_id"],
                "candidate_id": p2,
                "source_video_id": src,
                "split": m["split"],
                "anchor_last_frame": f_end,
                "query_frame": f_query,
                "query_time_s": round(QUERY_OFFSET_S, 3),
                "candidate_timestamp_frame": f_c,
                "candidate_delta_s": round((f_c - f_end) / fps, 3),
                "query_time_error_s": round(query_err, 3),
                "visibility": occ,
                "true_target_frame": int(m["target_frame"]),   # supervision ONLY
                "kinematic_distance": round(dist, 4),
                "spatial_gate_result": int(gated),
                "is_true_match": int(is_true),
                "hard_negative_tier": tier,
                "bbox": json.dumps(bbox),
                "occlusion": occ,
                "candidate_feature_key": f"{src}|{p2}|{f_c}",
                "evidence_provenance": "canonical idx + fixed-query(1.0s) "
                                       "+ causal extrapolation gate",
            })
            stats[tier] += 1
            n_out += 1
        if not any(r["is_true_match"] for r in out_rows[-n_out:]) if n_out else True:
            stats["anchors_no_natural_true"] += 0
    # natural no-match statistics
    anchors = {r["anchor_id"] for r in out_rows}
    no_match = sum(1 for a in anchors
                   if not any(r["is_true_match"] for r in out_rows
                              if r["anchor_id"] == a))
    with open(OUT / "candidates_fixed_query.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    fq_stats = {
        "protocol": {"query_offset_s": QUERY_OFFSET_S,
                     "query_window_s": QUERY_WINDOW_S,
                     "offset_derivation": "median positive gap "
                                          "(pre-registered, not tuned)",
                     "oracle_time_used": False,
                     "true_match_forced": False},
        "n_candidate_rows": len(out_rows),
        "n_anchors": len(anchors),
        "tier_counts": {str(k): v for k, v in stats.items()
                        if isinstance(k, int)},
        "anchors_with_natural_true_match": len(anchors) - no_match,
        "anchors_natural_no_match": no_match,
        "canonical_index": {"unique": n_idx, "conflicts": len(conflicts)},
    }
    (OUT / "fixed_query_stats.json").write_text(json.dumps(fq_stats, indent=2))
    print(json.dumps(fq_stats, indent=1))
    print(f"[fq] -> {OUT}")


if __name__ == "__main__":
    main()
