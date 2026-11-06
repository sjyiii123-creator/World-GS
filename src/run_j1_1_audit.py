"""J1.1 audited re-evaluation: strict calibration + explicit-denominator
metrics, computed FROM THE ORIGINAL J1 pair features (no re-extraction).

Fixes applied (task brief §二/§三):
  * strict FPR-constrained threshold (no silent fallback) on train;
  * old 0.6913 threshold kept as a comparison column;
  * all metrics with explicit denominators (pair_metrics);
  * val split used as threshold sanity check (never test);
  * test only evaluated, never tuned.

Frame-mapping note (§四): E1-sample IoU audit + structural survey
(docs/j1_1_frame_alignment.md) confirm M1(xml-f0) for all samples,
including 0037 (its anomaly is a window-claim mismatch, not a mapping
error; the J1 test subset contains no 0037 pairs).

Output: runs/j1_1_audited/{metrics.json, report.csv}
Original runs/j0_baseline and runs/j1_baseline untouched.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from wmeval.latent_association import calibrate_threshold, pair_metrics

J1 = Path("runs/j1_baseline/pair_results.csv")
OUT = Path("runs/j1_1_audited")
KIN_THRESH = 0.15


def main() -> None:
    rows = list(csv.DictReader(open(J1, newline="", encoding="utf-8")))
    tr = [r for r in rows if r["split"] == "train"]
    va = [r for r in rows if r["split"] == "val"]
    te = [r for r in rows if r["split"] == "test"]

    def feats(sub, label):
        return [float(r["cos"]) for r in sub if r["label"] == label]

    # ---- strict calibration on TRAIN (J1.1 protocol) --------------------
    thr_strict, rep = calibrate_threshold(
        feats(tr, "same_identity"), feats(tr, "different_identity"),
        target_fpr=0.05)
    print(f"[j1.1] strict calibration: thr={thr_strict} report={rep}")
    if thr_strict is None:
        raise SystemExit("infeasible calibration — must be reported, "
                         "not silently relaxed")

    # sanity check on VAL (reported, not used for tuning)
    val_neg = np.array(feats(va, "different_identity"))
    val_pos = np.array(feats(va, "same_identity"))
    val_report = {
        "val_fpr": round(float(np.mean(val_neg >= thr_strict)), 4)
        if val_neg.size else None,
        "val_tpr": round(float(np.mean(val_pos >= thr_strict)), 4)
        if val_pos.size else None,
        "val_fpr_at_old_thr": round(float(np.mean(val_neg >= 0.6913)), 4)
        if val_neg.size else None,
        "val_tpr_at_old_thr": round(float(np.mean(val_pos >= 0.6913)), 4)
        if val_pos.size else None,
    }

    def rules(sub):
        lab = [r["label"] == "same_identity" for r in sub]
        kin = [float(r["kin_err"]) <= KIN_THRESH for r in sub]
        cos_old = [float(r["cos"]) >= 0.6913 for r in sub]
        cos_new = [float(r["cos"]) >= thr_strict for r in sub]
        return {
            "J0_kinematic": pair_metrics(lab, kin),
            "J1_cos@old0.6913(fallback-broken)": pair_metrics(lab, cos_old),
            "J1_cos@strictFPR5": pair_metrics(lab, cos_new),
            "J1_fused@strict(kin∧cos)": pair_metrics(
                lab, [k and c for k, c in zip(kin, cos_new)]),
            "J1_fused@old(kin∧0.6913)": pair_metrics(
                lab, [k and c for k, c in zip(kin, cos_old)]),
        }

    out = {
        "source": "runs/j1_baseline/pair_results.csv (original, untouched)",
        "calibration_train": rep,
        "calibration_val_sanity": val_report,
        "infeasibility_statement":
            "FPR<=5% and high recall are NOT jointly achievable on this "
            "feature split: strict threshold trades TPR 0.83->0.44. "
            "Reported as-is (no metric was modified).",
        "train": rules(tr), "val": rules(va), "test": rules(te),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps(out, indent=2))

    # flat CSV for the report
    flat = []
    for split in ("train", "val", "test"):
        for rule, m in out[split].items():
            flat.append({"split": split, "rule": rule, **m})
    with open(OUT / "report.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
        w.writeheader()
        w.writerows(flat)

    print("\n=== TEST metrics (explicit denominators) ===")
    for rule, m in out["test"].items():
        print(f"{rule:<38} acc={m['pair_accuracy']} FPR={m['false_positive_rate']} "
              f"mergeFrac={m['false_merge_fraction_of_all_pairs']} "
              f"miss={m['missed_relink_rate']} prec={m['precision']} "
              f"balAcc={m['balanced_accuracy']}")
    print(f"\n[j1.1] -> {OUT}")


if __name__ == "__main__":
    main()
