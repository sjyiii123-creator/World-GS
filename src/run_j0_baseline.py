"""J0 baseline: kinematic (motion-extrapolation) association evaluation.

Reuses the SAME kinematic rule as wmeval.actor_binding (tail-median
velocity extrapolation, normalized distance gate) — evaluated against
JAAD real identity labels on the synthetic-gap actor-sequence dataset.
v3.1 adjudication semantics are NOT touched (standalone evaluation).

Metrics:
  * pair-level: accuracy, error-merge rate (different-identity accepted),
    missed-relink rate (same-identity rejected);
  * competition-level: top-1 identity accuracy (ranking by extrapolation
    error), ambiguity rate (top-2 errors within ratio).

Output: runs/j0_baseline/{pair_results.csv, metrics.json}
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, "src")

SEQ = Path("datasets/processed/actor_sequences")
OUT = Path("runs/j0_baseline")

THRESH = 0.15          # same gate as actor_binding max_pred_err
AMBIG_RATIO = 1.5      # same ambiguity ratio as actor_binding
W, H = 1920.0, 1080.0


def centers(frames):
    # frames: [[fi, bbox, occ], ...]
    return [((b[0] + b[2]) / 2 / W, (b[1] + b[3]) / 2 / H) for _, b, _ in frames]


def tail_velocity(frames, k=5):
    c = centers(frames)
    if len(c) < 2:
        return (0.0, 0.0)
    steps = [(c[i + 1][0] - c[i][0], c[i + 1][1] - c[i][1])
             for i in range(max(0, len(c) - k - 1), len(c) - 1)]
    return (statistics.median(s[0] for s in steps),
            statistics.median(s[1] for s in steps))


def extrapolation_error(anchor, cand, gap_frames):
    """actor_binding's case-(a) kinematic rule on a pair (normalized)."""
    v = tail_velocity(anchor)
    last = centers(anchor)[-1]
    first = centers(cand)[0]
    pred = (last[0] + v[0] * gap_frames, last[1] + v[1] * gap_frames)
    return ((first[0] - pred[0]) ** 2 + (first[1] - pred[1]) ** 2) ** 0.5


def main() -> None:
    rows = list(csv.DictReader(open(SEQ / "samples.csv",
                                    newline="", encoding="utf-8")))
    results = []
    for r in rows:
        anchor = json.loads(r["anchor_frames"])
        cand = json.loads(r["cand_frames"])
        gap = int(r["gap_frames"])
        # real gap in frames between anchor end and cand start (integers)
        true_gap = cand[0][0] - anchor[-1][0]
        err = extrapolation_error(anchor, cand, true_gap)
        accepted = err <= THRESH
        results.append({
            "pair_id": r["pair_id"], "split": r["split"],
            "label": r["label"], "gap_frames": r["gap_frames"],
            "true_gap_frames": true_gap,
            "pred_err_norm": round(err, 4),
            "accepted": int(accepted),
            "correct": int(accepted == (r["label"] == "same_identity")),
        })

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "pair_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    def metrics(sub):
        n = len(sub)
        if not n:
            return {}
        tp = sum(1 for x in sub if x["label"] == "same_identity" and x["accepted"])
        fp = sum(1 for x in sub if x["label"] == "different_identity" and x["accepted"])
        fn = sum(1 for x in sub if x["label"] == "same_identity" and not x["accepted"])
        return {
            "n": n,
            "accuracy": round((n - fp - fn) / n, 4),
            "error_merge_rate": round(fp / n, 4),      # wrong identity merged
            "missed_relink_rate": round(fn / max(1, tp + fn), 4),
        }

    # competition ranking: top-1 identity by extrapolation error
    comps = [json.loads(l) for l in
             (SEQ / "competitions.jsonl").read_text().splitlines() if l.strip()]
    pair_row = {r["pair_id"]: r for r in rows}
    comp_rows = []
    for c in comps:
        pr = pair_row.get(str(c["pair_id"]))
        if not pr:
            continue
        anchor = json.loads(pr["anchor_frames"])
        errs = []
        for cand in c["candidates"]:
            # candidate's own first frame as the cand segment start
            qtr = [cand["first_frame"], cand["bbox"], 0]
            gap = cand["first_frame"] - anchor[-1][0]
            if gap <= 0:
                continue
            errs.append((extrapolation_error(anchor, [qtr], gap),
                         cand["is_true_match"], cand["cand_ped_id"]))
        if len(errs) < 2:
            continue
        errs.sort()
        top1 = errs[0][1]
        amb = (len(errs) > 1 and errs[1][0] < AMBIG_RATIO * max(errs[0][0], 1e-6)
               and errs[1][0] <= THRESH)
        comp_rows.append({"comp_id": c["comp_id"], "split": pr["split"],
                          "gap_frames": pr["gap_frames"],
                          "n_candidates": len(errs),
                          "top1_correct": int(top1),
                          "ambiguous": int(amb)})

    def comp_metrics(sub):
        n = len(sub)
        return {"n": n,
                "top1_identity_accuracy": round(
                    sum(x["top1_correct"] for x in sub) / max(n, 1), 4),
                "ambiguity_rate": round(
                    sum(x["ambiguous"] for x in sub) / max(n, 1), 4)}

    out = {"threshold_norm": THRESH,
           "protocol": "actor_binding kinematic rule vs JAAD identity labels",
           "pair_level": {"all": metrics(results)},
           "by_split": {s: metrics([x for x in results if x["split"] == s])
                        for s in ("train", "val", "test")},
           "by_gap": {g: metrics([x for x in results
                                  if int(x["gap_frames"]) == g])
                      for g in (15, 30, 45)},
           "competition_level": {"all": comp_metrics(comp_rows)},
           "competition_by_split": {s: comp_metrics(
               [x for x in comp_rows if x["split"] == s])
               for s in ("train", "val", "test")}}
    (OUT / "metrics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["pair_level"]["all"], indent=2))
    print("by_gap:", json.dumps(out["by_gap"]))
    print("competition all:", json.dumps(out["competition_level"]["all"]))
    print(f"[j0] -> {OUT}")


if __name__ == "__main__":
    main()
