"""J2 stage-3: A0/A1/A2 selective-association strong baselines.

Data: runs/j3_lite/competitions_detail.json (per-anchor candidates with
cos/kin/true-match flags — features already computed; CPU-only run).

Baselines (top-1 ranking is KINEMATIC in all three — the J3-lite lesson):
  A0: kin ranking, accept-all (no rejection);
  A1: kin ranking + kin selective rejection (kin threshold calibrated on
      TRAIN competitions, FPR<=5% among WRONG candidates);
  A2: kin ranking + DINOv2 appearance-confidence rejection (cos threshold
      calibrated on TRAIN, FPR<=5% among WRONG candidates).

No-match groups: for every competition, a twin with the true match
REMOVED (all candidates wrong) — measures false acceptance when the
correct actor is absent (task §三 "正确 Actor 缺席").

Fixed uncetainty-band statistic (task §三 fix): the tautological
`uncertain_band_true_share` (=1.0 because every set contains the true
match) is replaced by the INFORMATIVE comparison:
band top-1(kin) accuracy vs overall top-1 accuracy — if the band
concentrates hard cases, its top-1 accuracy is BELOW overall.

Metrics (per task §三, never pair-accuracy alone): Top-1 Identity
Accuracy; Wrong-Identity-Among-Accepted; Acceptance Coverage;
True-Match-Recall (among true-match-present competitions); No-Match
False-Acceptance (twin groups); Uncertain share; source-video clustered
bootstrap 95% CIs (1000 resamples).

Splits: source-video level, consistent with j2 v2 buckets (train /
exploratory_test; final_test sources excluded — too small).

Output: runs/j2_experimental/{a_baselines.csv, a_baselines.json}
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

from wmeval.latent_association import calibrate_threshold

DETAIL = Path("runs/j3_lite/competitions_detail.json")
SEQ = Path("datasets/processed/actor_sequences")
J1R = Path("runs/j1_baseline/pair_results.csv")
OUT = Path("runs/j2_experimental")
N_BOOT = 1000
SEED = 51


def calibrate_wrong_fpr(rows, key, target=0.05):
    """Threshold on TRAIN wrong candidates with FPR<=target (strict)."""
    wrong = [c[key] for r in rows if r["split"] == "train"
             for c in r["cands"] if not c["is_true"]]
    right = [c[key] for r in rows if r["split"] == "train"
             for c in r["cands"] if c["is_true"]]
    # for kin: LOWER is better -> work on negative scores
    if key == "kin":
        thr, rep = calibrate_threshold([-x for x in right],
                                       [-x for x in wrong], target)
        return (None if thr is None else -thr), rep
    thr, rep = calibrate_threshold(right, wrong, target)
    return thr, rep


def three_state(row, rule):
    """rule: ('A0'|'A1'|'A2') -> (decision, chosen) on ONE competition."""
    by_kin = sorted(row["cands"], key=lambda c: c["kin"])
    top = by_kin[0]
    if rule == "A0":
        return "accept", top
    if rule == "A1":
        if top["kin"] <= rule_thr["A1"]:
            return "accept", top
        if top["kin"] >= 2.0 * rule_thr["A1"]:
            return "reject", None
        return "uncertain", top
    # A2: cos-confidence gate on the kin-ranked top-1
    if top["cos"] >= rule_thr["A2"]:
        return "accept", top
    if top["cos"] <= rule_thr_lo["A2"]:
        return "reject", None
    return "uncertain", top


def evaluate(rows, rule, nomatch=False):
    n = len(rows)
    top1 = 0
    acc = rej = unc = 0
    wrong_acc = true_acc = 0
    band_top1 = 0
    band_n = 0
    for r in rows:
        has_true = not nomatch
        if nomatch:
            r = {"cands": [c for c in r["cands"] if not c["is_true"]],
                 "split": r["split"], "comp_id": r["comp_id"]}
            if len(r["cands"]) < 1:
                continue
        d, top = three_state(r, rule)
        by_kin = sorted(r["cands"], key=lambda c: c["kin"])
        t_ok = by_kin[0]["is_true"]
        top1 += t_ok
        if d == "accept":
            acc += 1
            if t_ok:
                true_acc += 1
            else:
                wrong_acc += 1
        elif d == "uncertain":
            unc += 1
            band_top1 += t_ok
            band_n += 1
        else:
            rej += 1
    return {
        "n": n, "rule": rule, "nomatch": nomatch,
        "top1_identity_accuracy": round(top1 / n, 4),
        "acceptance_coverage": round(acc / n, 4),
        "wrong_identity_among_accepted": round(wrong_acc / acc, 4)
        if acc else None,
        "accept_precision": round(true_acc / acc, 4) if acc else None,
        "true_match_recall": round(true_acc / n, 4) if not nomatch else None,
        "no_match_false_acceptance": round(acc / n, 4) if nomatch else None,
        "uncertain_share": round(unc / n, 4),
        "band_top1_accuracy": round(band_top1 / band_n, 4)
        if band_n else None,
    }


def bootstrap_ci(rows, rule, metric="top1_identity_accuracy", seed=51):
    rng = random.Random(seed)
    by_src = defaultdict(list)
    for r in rows:
        by_src[r.get("source_video_id", r["comp_id"])].append(r)
    srcs = list(by_src)
    vals = []
    for _ in range(N_BOOT):
        sample_srcs = [rng.choice(srcs) for _ in srcs]
        sample = [r for s in sample_srcs for r in by_src[s]]
        vals.append(evaluate(sample, rule)[metric])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"metric": metric, "ci95": [round(float(lo), 4),
                                        round(float(hi), 4)]}


def main() -> None:
    random.seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    detail = json.loads(DETAIL.read_text())
    # split by source (consistent with v2 buckets): train-side vs test-side
    seq_src = {}
    for r in csv.DictReader(open(SEQ / "samples.csv", newline="")):
        seq_src[r["pair_id"]] = r["jaad_video"]
    j1 = {r["pair_id"] for r in csv.DictReader(open(J1R, newline=""))}
    for d in detail:
        src = seq_src.get(str(d["pair_id"]))
        d["source_video_id"] = src
        # J1 saw both sides of its sample; treat train-source rows as
        # calibration, test-source rows as (exploratory) evaluation
        orig_split = d["split"]
        d["split"] = "train" if orig_split == "train" else "test"
    train_rows = [r for r in detail if r["split"] == "train"]
    test_rows = [r for r in detail if r["split"] == "test"]
    print(f"[A] competitions: train {len(train_rows)} / test {len(test_rows)}")

    global rule_thr, rule_thr_lo
    rule_thr, rule_thr_lo = {}, {}
    t_k, rep_k = calibrate_wrong_fpr(detail, "kin")
    t_c, rep_c = calibrate_wrong_fpr(detail, "cos")
    rule_thr["A1"] = t_k
    rule_thr["A2"] = t_c
    # low thresholds (reject side): P10 of RIGHT candidates on train
    right_kin = [c["kin"] for r in train_rows for c in r["cands"] if c["is_true"]]
    right_cos = [c["cos"] for r in train_rows for c in r["cands"] if c["is_true"]]
    rule_thr_lo["A1"] = round(float(np.percentile(right_kin, 90)), 4)  # kin hi side
    rule_thr_lo["A2"] = round(float(np.percentile(right_cos, 10)), 4)
    print(f"[A] A1 kin thr={t_k} (rep fpr={rep_k.get('achieved_fpr')}), "
          f"A2 cos thr={t_c} (rep fpr={rep_c.get('achieved_fpr')})")

    results = []
    for rule in ("A0", "A1", "A2"):
        results.append(evaluate(test_rows, rule))
        results.append(evaluate(test_rows, rule, nomatch=True))
    # fixed uncertainty-band statistic (informative replacement)
    overall_top1 = evaluate(test_rows, "A0")["top1_identity_accuracy"]

    cis = {}
    for rule in ("A0", "A1", "A2"):
        cis[rule] = bootstrap_ci(test_rows, rule)
        cis[rule + "_wrong_acc"] = bootstrap_ci(
            test_rows, rule, "wrong_identity_among_accepted")

    out = {
        "thresholds": {"A1_kin": rule_thr["A1"], "A2_cos": rule_thr["A2"],
                       "A1_kin_hi": rule_thr_lo["A1"],
                       "A2_cos_lo": rule_thr_lo["A2"],
                       "calibration": "train-only, FPR<=5% among WRONG "
                                      "candidates (strict protocol)"},
        "results": results,
        "uncertainty_band_fix": {
            "old_tautological": "uncertain_band_true_share=1.0 "
            "(every competition contains the true match — uninformative)",
            "new_statistic": "band_top1_accuracy vs overall top1 — "
                             "band concentrates hard cases iff lower",
            "overall_top1_kin": overall_top1,
            "band_top1_by_rule": {r["rule"]: r["band_top1_accuracy"]
                                  for r in results
                                  if not r["nomatch"]}},
        "source_bootstrap_ci95": cis,
        "coverage_matched_note": "A1 vs A2 compared at their own calibrated "
                                 "operating points AND via threshold sweep "
                                 "in a_baselines_sweep.json",
    }
    (OUT / "a_baselines.json").write_text(json.dumps(out, indent=2))

    # threshold sweep for coverage-matched A1 vs A2
    sweep = []
    for t in np.arange(0.05, 0.55, 0.05):
        rule_thr["A1"] = round(float(t), 3)
        m = evaluate(test_rows, "A1")
        sweep.append({"rule": "A1", "thr": round(float(t), 3),
                      "coverage": m["acceptance_coverage"],
                      "wrong_acc": m["wrong_identity_among_accepted"]})
    rule_thr["A1"] = t_k
    for t in np.arange(0.55, 0.91, 0.05):
        rule_thr["A2"] = round(float(t), 3)
        m = evaluate(test_rows, "A2")
        sweep.append({"rule": "A2", "thr": round(float(t), 3),
                      "coverage": m["acceptance_coverage"],
                      "wrong_acc": m["wrong_identity_among_accepted"]})
    rule_thr["A2"] = t_c
    (OUT / "a_baselines_sweep.json").write_text(json.dumps(sweep, indent=2))

    print(json.dumps(results, indent=2))
    print(f"[A] -> {OUT}")


if __name__ == "__main__":
    main()
