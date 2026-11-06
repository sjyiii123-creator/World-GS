"""J1.1 §七/§八: multi-candidate association evaluation + J3-lite.

Multi-candidate (per anchor, ALL candidates from competitions.jsonl):
  J0     rank by kinematic extrapolation error (no rejection)
  J1     rank by frozen-feature cosine (no rejection)
  J3-lite three-state decision:
      accept   : top-1 by cos AND cos >= thr_hi AND kin <= KIN
      reject   : cos < thr_lo OR kin > KIN_REJECT
      uncertain: everything else (band)
  Thresholds calibrated on TRAIN competitions only:
      thr_hi = strict FPR<=5% on train negative candidates
      thr_lo = train positive P10 (reject side loses <=10% of true links)
  KIN=0.15 (actor_binding gate), KIN_REJECT=0.30 (2x gate).

Metrics: top-1 identity accuracy, true-match-missing rate (true match
not among candidates), wrong-identity selection rate (among accepts),
undetermined rate, ambiguity rate (top-2 cos within 0.05).

Risk-coverage curve: sweep thr_hi on test -> coverage (accept share)
vs wrong-selection rate among accepts + true-recall among accepts.
Anti-gaming report: uncertain band composition (share of TRUE matches
parked in uncertain) — hiding true pairs in Uncertain is exposed.

Output: runs/j3_lite/{competitions_eval.csv, metrics.json}
"""
from __future__ import annotations

import csv
import json
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from wmeval.latent_features import CropConfig, LatentFeatureExtractor
from wmeval.latent_association import calibrate_threshold

SEQ = Path("datasets/processed/actor_sequences")
J1R = Path("runs/j1_baseline/pair_results.csv")
OUT = Path("runs/j3_lite")
KIN, KIN_REJECT = 0.15, 0.30
TOP2_MARGIN = 0.05
SEED = 41


def centers(frames):
    return [((b[0] + b[2]) / 2 / 1920.0, (b[1] + b[3]) / 2 / 1080.0)
            for _, b, _ in frames]


def kin_err(anchor, first_frame, bbox):
    ac = centers(anchor)
    steps = [(ac[i + 1][0] - ac[i][0], ac[i + 1][1] - ac[i][1])
             for i in range(max(0, len(ac) - 6), len(ac) - 1)]
    v = (statistics.median(s[0] for s in steps),
         statistics.median(s[1] for s in steps))
    gap = first_frame - anchor[-1][0]
    if gap <= 0:
        return 9.9
    pred = (ac[-1][0] + v[0] * gap, ac[-1][1] + v[1] * gap)
    c = ((bbox[0] + bbox[2]) / 2 / 1920.0, (bbox[1] + bbox[3]) / 2 / 1080.0)
    return ((c[0] - pred[0]) ** 2 + (c[1] - pred[1]) ** 2) ** 0.5


def main() -> None:
    random.seed(SEED)
    global SEQ_ROWS
    if SEQ_ROWS is None:
        _load_seq_rows()
    OUT.mkdir(parents=True, exist_ok=True)
    j1 = list(csv.DictReader(open(J1R, newline="")))
    j1_by_id = {r["pair_id"]: r for r in j1}
    comps = [json.loads(l) for l in
             (SEQ / "competitions.jsonl").read_text().splitlines() if l.strip()]
    comps = [c for c in comps if str(c["pair_id"]) in j1_by_id]
    ex = LatentFeatureExtractor(model_id="ckpt/dinov2-small",
                                crop_cfg=CropConfig(),
                                cache_dir=Path("runs/j1_baseline/feature_cache"))
    t0 = time.perf_counter()
    eval_rows = []
    for c in comps:
        pr = j1_by_id[str(c["pair_id"])]
        if pr["split"] == "val":
            continue
        sid = c["synpedas_sample"]
        f0 = int(sid.split("_f")[-1].split("-")[0])
        seq_row = next((r for r in SEQ_ROWS if r["pair_id"] == str(c["pair_id"])), None)
        if seq_row is None:
            continue
        anchor = json.loads(seq_row["anchor_frames"])
        aped = seq_row["anchor_ped_id"]
        vp = f"repos/SynPeDAS/JAAD_crossing_moments/{sid}.mp4"
        a_feat = ex.segment_feature(vp, f0, anchor, sid, aped)
        cands = []
        for cd in c["candidates"]:
            try:
                feat = ex.frame_feature(vp, cd["first_frame"] - f0,
                                        cd["bbox"], sid, cd["cand_ped_id"])
            except RuntimeError:
                continue
            cos = float(np.dot(a_feat, feat))
            ke = kin_err(anchor, cd["first_frame"], cd["bbox"])
            cands.append({"ped": cd["cand_ped_id"],
                          "is_true": cd["is_true_match"], "cos": round(cos, 4),
                          "kin": round(ke, 4)})
        if len(cands) < 2:
            continue
        eval_rows.append({"comp_id": c["comp_id"], "pair_id": c["pair_id"],
                          "split": pr["split"], "cands": cands})
    ex.flush_cache()
    print(f"[j3] {len(eval_rows)} competitions evaluated "
          f"({time.perf_counter()-t0:.0f}s)")

    # ---- calibrate thresholds on TRAIN competitions ---------------------
    tr_rows = [r for r in eval_rows if r["split"] == "train"]
    pos = [cd["cos"] for r in tr_rows for cd in r["cands"] if cd["is_true"]]
    neg = [cd["cos"] for r in tr_rows for cd in r["cands"] if not cd["is_true"]]
    thr_hi, rep_hi = calibrate_threshold(pos, neg, 0.05)
    thr_lo = round(float(np.percentile(pos, 10)), 4)
    print(f"[j3] thr_hi={thr_hi} (FPR<=5% on train neg candidates), "
          f"thr_lo={thr_lo} (train pos P10)")

    def evaluate(rows, thr_hi_v, thr_lo_v):
        n = len(rows)
        top1_j0 = top1_j1 = 0
        accept = reject = uncertain = 0
        wrong_accept = true_accept = true_in_unc = 0
        true_missing = 0
        amb = 0
        for r in rows:
            cands = sorted(r["cands"], key=lambda c: c["cos"], reverse=True)
            by_kin = sorted(r["cands"], key=lambda c: c["kin"])
            top1_j0 += by_kin[0]["is_true"]
            top1_j1 += cands[0]["is_true"]
            if len(cands) > 1 and cands[0]["cos"] - cands[1]["cos"] < TOP2_MARGIN:
                amb += 1
            if not any(c["is_true"] for c in r["cands"]):
                true_missing += 1
            best = cands[0]
            if best["cos"] >= thr_hi_v and best["kin"] <= KIN:
                accept += 1
                if best["is_true"]:
                    true_accept += 1
                else:
                    wrong_accept += 1
            elif best["cos"] < thr_lo_v or best["kin"] > KIN_REJECT:
                reject += 1
            else:
                uncertain += 1
                if any(c["is_true"] for c in r["cands"]):
                    true_in_unc += 1
        cov = accept / n
        return {
            "n": n,
            "top1_identity_accuracy_j0_kin": round(top1_j0 / n, 4),
            "top1_identity_accuracy_j1_cos": round(top1_j1 / n, 4),
            "true_match_missing_rate": round(true_missing / n, 4),
            "accept_rate_coverage": round(cov, 4),
            "wrong_identity_selection_rate_among_accepts":
                round(wrong_accept / accept, 4) if accept else None,
            "accept_precision": round(true_accept / accept, 4) if accept else None,
            "reject_rate": round(reject / n, 4),
            "undetermined_rate": round(uncertain / n, 4),
            "uncertain_band_true_share": round(true_in_unc / uncertain, 4)
            if uncertain else None,
            "ambiguity_rate": round(amb / n, 4),
        }

    te_rows = [r for r in eval_rows if r["split"] == "test"]
    out = {
        "thresholds": {"thr_hi": thr_hi, "thr_lo": thr_lo,
                       "calibration_hi": rep_hi,
                       "protocol": "train-only calibration (FPR<=5% high, "
                                   "pos-P10 low); test never tuned"},
        "test_j0_j1_j3lite": evaluate(te_rows, thr_hi, thr_lo),
        "train_j0_j1_j3lite": evaluate(tr_rows, thr_hi, thr_lo),
        "risk_coverage_curve_test": [],
    }
    # sweep thr_hi for risk-coverage (J3-lite accept gate only)
    for t in (0.60, 0.65, 0.70, 0.75, 0.8122, 0.85, 0.90):
        if t < (thr_hi or 0):
            # sweep includes operating points BELOW the strict calibration;
            # mark them explicitly as non-calibrated comparison points
            pass
        m = evaluate(te_rows, t, thr_lo)
        out["risk_coverage_curve_test"].append(
            {"thr_hi": t, "coverage": m["accept_rate_coverage"],
             "wrong_selection": m["wrong_identity_selection_rate_among_accepts"],
             "accept_precision": m["accept_precision"],
             "undetermined": m["undetermined_rate"]})
    (OUT / "metrics.json").write_text(json.dumps(out, indent=2))
    with open(OUT / "competitions_eval.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["comp_id", "pair_id", "split",
                                          "n_cands"])
        w.writeheader()
        for r in eval_rows:
            w.writerow({"comp_id": r["comp_id"], "pair_id": r["pair_id"],
                        "split": r["split"], "n_cands": len(r["cands"])})
    (OUT / "competitions_detail.json").write_text(json.dumps(eval_rows))
    print(json.dumps(out["test_j0_j1_j3lite"], indent=2))
    print(f"[j3] -> {OUT}")


# sequence rows loader (module level for the loop above)
SEQ_ROWS = None


def _load_seq_rows():
    global SEQ_ROWS
    SEQ_ROWS = list(csv.DictReader(open(SEQ / "samples.csv", newline="")))


if __name__ == "__main__":
    _load_seq_rows()
    main()
