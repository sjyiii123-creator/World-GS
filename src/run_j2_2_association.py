"""J2.2 stage-7: association under the unified protocol.

A1: kinematic ranking + kinematic selective rejection (baseline).
A2: kinematic ranking + frozen-appearance (static) confidence rejection.
A3: kinematic ranking + RESIDUAL-predictor evidence rejection (per-
    candidate delta; unusable time -> unknown evidence -> uncertain).

All candidates use their OWN delta; thresholds calibrated train-only
(FPR<=5% among wrong candidates, strict protocol); test bed = the v2
exploratory competitions (82, source-isolated); no-match twins;
hard-negative and gap strata; source-grouped bootstrap CIs;
fixed-coverage comparison via threshold sweeps.
"""
from __future__ import annotations

import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")

from wmeval.latent_features import CropConfig, LatentFeatureExtractor
from wmeval.latent_association import calibrate_threshold
from run_j2_train import ActorPredictor
from wmeval.latent_world_state.residual_predictor import ResidualPredictor

V2 = Path("runs/j2_experimental")
OUT = Path("runs/j2_2_experimental")
J2X = V2 / "j2_experimental" if False else V2
DETAIL = Path("runs/j3_lite/competitions_detail.json")
RAW = Path("datasets/processed/actor_sequences/competitions.jsonl")
SEQCSV = Path("datasets/processed/actor_sequences/samples.csv")
CACHE = Path("runs/j1_baseline/feature_cache")
N_BOOT = 1000
KIN_HI_MULT = 2.0


def load_model(kind):
    if kind == "A2":
        return None
    if kind == "A3":
        m = ResidualPredictor()
        m.load_state_dict(torch.load(OUT / "b4_residual_expanded.pt",
                                     map_location="cpu"))
        m.eval()
        return m


def main() -> None:
    random.seed(51)
    ex = LatentFeatureExtractor(model_id="ckpt/dinov2-small",
                                crop_cfg=CropConfig(), cache_dir=CACHE)
    seqrows = {r["pair_id"]: r for r in csv.DictReader(
        open(SEQCSV, newline=""))}
    v2man = {r["pair_id"]: r for r in csv.DictReader(
        open(V2 / "j2_manifest.csv", newline=""))}
    raw_comps = {c["comp_id"]: c for c in
                 (json.loads(l) for l in RAW.read_text().splitlines()
                  if l.strip())}
    detail = json.loads(DETAIL.read_text())

    residual = load_model("A3")

    def prep(pair_filter):
        rows = []
        with torch.no_grad():
            for c in detail:
                pid = str(c["pair_id"])
                if pid not in v2man or not pair_filter(v2man[pid]):
                    continue
                pr = seqrows[pid]
                sid = pr["synpedas_sample"]
                f0 = int(sid.split("_f")[-1].split("-")[0])
                fps = float(pr["fps"])
                vp = f"repos/SynPeDAS/JAAD_crossing_moments/{sid}.mp4"
                anchor = json.loads(pr["anchor_frames"])
                hist = np.stack([ex.frame_feature(vp, f[0] - f0, f[1], sid,
                                                  pr["anchor_ped_id"])
                                 for f in anchor])
                anchor_last_t = anchor[-1][0] / fps
                z_last = hist[-1]
                raw = raw_comps.get(c["comp_id"], {})
                cands = []
                for cd in c["cands"]:
                    rc = next((x for x in raw.get("candidates", [])
                               if x["cand_ped_id"] == cd["ped"]), None)
                    if rc is None:
                        continue
                    try:
                        feat = ex.frame_feature(
                            vp, rc["first_frame"] - f0, rc["bbox"], sid,
                            rc["cand_ped_id"])
                    except RuntimeError:
                        continue
                    cand_t = rc["first_frame"] / fps
                    d = cand_t - anchor_last_t
                    usable = d > 0 and np.isfinite(d)
                    pc = None
                    if usable and residual is not None:
                        z = residual(
                            torch.from_numpy(hist[None].astype(np.float32)),
                            torch.tensor([float(d)])).numpy()[0]
                        pc = round(float(np.dot(z, feat)), 4)
                    cands.append({
                        "ped": cd["ped"], "is_true": cd["is_true"],
                        "kin": cd["kin"],
                        "static_cos": round(float(np.dot(z_last, feat)), 4),
                        "pred_cos": pc,
                        "cand_delta_s": round(float(d), 4) if usable
                        else None})
                if len(cands) >= 2:
                    rows.append({"comp_id": c["comp_id"],
                                 "source_video_id": pr["jaad_video"],
                                 "cands": cands})
        return rows

    eval_rows = prep(lambda m: m["split"] == "exploratory_test")
    tr_rows = prep(lambda m: m["split"] == "train")
    ex.flush_cache()
    print(f"[assoc] competitions: eval {len(eval_rows)} / train {len(tr_rows)}")

    def calib(key):
        right = [c[key] for r in tr_rows for c in r["cands"]
                 if c["is_true"] and c[key] is not None]
        wrong = [c[key] for r in tr_rows for c in r["cands"]
                 if not c["is_true"] and c[key] is not None]
        if not right or not wrong:
            return None, None
        t, rep = calibrate_threshold(right, wrong, 0.05)
        return t, rep

    kin_right = [c["kin"] for r in tr_rows for c in r["cands"] if c["is_true"]]
    kin_wrong = [c["kin"] for r in tr_rows for c in r["cands"]
                 if not c["is_true"]]
    kt, _ = calibrate_threshold([-v for v in kin_right],
                                [-v for v in kin_wrong], 0.05)
    kin_thr = -kt
    static_thr, _ = calib("static_cos")
    pred_thr, _ = calib("pred_cos")
    print(f"[assoc] kin={kin_thr} static={static_thr} pred={pred_thr}")

    def evaluate(rows, rule, thr=None, nomatch=False):
        n = top1 = acc = unc = wrong_acc = true_acc = 0
        for r in rows:
            cands = r["cands"]
            if nomatch:
                cands = [c for c in cands if not c["is_true"]]
                if not cands:
                    continue
            n += 1
            by = sorted(cands, key=lambda c: c["kin"])
            top = by[0]
            top1 += top["is_true"]
            if rule == "A1":
                base = kin_thr if thr is None else thr
                d = ("accept" if top["kin"] <= base else
                     "reject" if top["kin"] >= KIN_HI_MULT * base
                     else "uncertain")
            elif rule == "A2":
                s = top["static_cos"]
                d = ("accept" if top["kin"] <= kin_thr and
                     s is not None and s >= (static_thr if thr is None else thr) else
                     "reject" if top["kin"] >= KIN_HI_MULT * kin_thr or
                     (s is not None and s < 0.45) else "uncertain")
            else:  # A3 residual
                p = top["pred_cos"]
                if p is None:
                    d = "uncertain"
                else:
                    d = ("accept" if top["kin"] <= kin_thr and
                         p >= (pred_thr if thr is None else thr) else
                         "reject" if top["kin"] >= KIN_HI_MULT * kin_thr or
                         p < 0.45 else "uncertain")
            if d == "accept":
                acc += 1
                if top["is_true"]:
                    true_acc += 1
                else:
                    wrong_acc += 1
            elif d == "uncertain":
                unc += 1
        return {"rule": rule, "thr": thr, "nomatch": nomatch, "n": n,
                "top1_identity_accuracy": round(top1 / n, 4),
                "acceptance_coverage": round(acc / n, 4),
                "wrong_identity_among_accepted":
                    round(wrong_acc / acc, 4) if acc else None,
                "no_match_false_acceptance": round(acc / n, 4) if nomatch
                else None,
                "uncertain_share": round(unc / n, 4)}

    # calibrated operating points
    results = []
    for rule in ("A1", "A2", "A3"):
        results.append(evaluate(eval_rows, rule))
        results.append(evaluate(eval_rows, rule, nomatch=True))

    # fixed-coverage sweeps (threshold -> coverage curve, normal+nomatch)
    sweep = []
    for t in np.arange(0.01, 0.30, 0.02):
        sweep.append(evaluate(eval_rows, "A1", thr=round(float(t), 3)))
    if static_thr:
        for t in np.arange(0.50, 0.92, 0.04):
            sweep.append(evaluate(eval_rows, "A2", thr=round(float(t), 3)))
    if pred_thr:
        for t in np.arange(0.50, 0.92, 0.04):
            sweep.append(evaluate(eval_rows, "A3", thr=round(float(t), 3)))

    # hard-negative & gap strata at calibrated points
    strata = {}
    for rule in ("A1", "A2", "A3"):
        sub_hard = [r for r in eval_rows
                    if any(not c["is_true"] and c["kin"] <= kin_thr
                           for c in r["cands"])]
        m_hard = evaluate(sub_hard, rule) if sub_hard else None
        strata[rule] = {"hard_neg_competitions": m_hard}

    def boot_ci(rule, metric):
        rng = random.Random(51)
        by_src = defaultdict(list)
        for r in eval_rows:
            by_src[r["source_video_id"]].append(r)
        srcs = list(by_src)
        vals = [evaluate([r for s in [rng.choice(srcs) for _ in srcs]
                          for r in by_src[s]], rule)[metric]
                for _ in range(N_BOOT)]
        vals = [v for v in vals if v is not None]
        if not vals:
            return None
        lo, hi = np.percentile(vals, [2.5, 97.5])
        return [round(float(lo), 4), round(float(hi), 4)]

    cis = {r: {m: boot_ci(r, m) for m in
               ("top1_identity_accuracy",
                "wrong_identity_among_accepted")}
           for r in ("A1", "A2", "A3")}

    out = {"thresholds": {"kin": kin_thr, "static_cos": static_thr,
                          "residual_pred_cos": pred_thr,
                          "calibration": "train-only FPR<=5% (strict)"},
           "results": results, "sweep": sweep, "strata": strata,
           "source_bootstrap_ci95": cis,
           "note": "A3 uses b4_residual_expanded predictions with "
                   "per-candidate delta; unusable time -> unknown -> "
                   "uncertain"}
    (OUT / "j2_2_association.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(results, indent=1))
    print(f"[assoc] -> {OUT}")


if __name__ == "__main__":
    main()
