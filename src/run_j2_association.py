"""J2 stage-5/6: ablation (prediction quality) + association fusion.

Stage-5 ablation — same protocol for every method (predict the target
feature from causal history; evaluate on val + exploratory_test):
  J1 / J2-A : last history feature as prediction (static);
  J2-B      : mean of history features;
  J2-C      : trained GRU (j2_predictor.pt);
  J2-D      : trained GRU + 2D motion state (j2_predictor_D.pt).
  Metrics: mean cos(pred, target); same-vs-different actor discrimination
  (val cross-cos); gap strata (0.5/1.0/1.5s).

Stage-6 fusion — kinematic ranking stays the MAIN ranker; candidates get
evidence columns:
  static_similarity   (anchor last frame vs candidate first frame)
  predicted_similarity(J2-C prediction vs candidate feature)
Rules evaluated on exploratory competitions (never-viewed final_test
kept untouched for directional sanity only):
  A1  : kin rank + kin rejection (re-baselined here);
  A3  : kin rank + kin ∧ predicted-cos rejection;
  JOINT: z(kin)+z(pred_cos) ranking + selective rejection + ambiguity
  band (top-2 margin) -> Uncertain.
Metrics: identical to stage-3 A baselines (incl. no-match twins,
bootstrap CIs).  Never auto-merge tracks on prediction similarity alone;
ambiguity must stay Uncertain (task §六).

Output: runs/j2_experimental/{j2_ablation.json, j2_association.json}
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
from torch.nn import functional as F

sys.path.insert(0, "src")

from wmeval.latent_features import CropConfig, LatentFeatureExtractor
from run_j2_train import ActorPredictor, ActorPredictorD

OUT = Path("runs/j2_experimental")
CACHE = Path("runs/j1_baseline/feature_cache")
DETAIL = Path("runs/j3_lite/competitions_detail.json")
SEQCSV = Path("datasets/processed/actor_sequences/samples.csv")
N_BOOT = 1000


def load_predictor(tag):
    m = ActorPredictorD() if tag == "D" else ActorPredictor()
    m.load_state_dict(torch.load(OUT / f"j2_predictor{'' if tag=='C' else '_D'}.pt",
                                  map_location="cpu"))
    m.eval()
    return m


def main() -> None:
    random.seed(51)
    with np.load(OUT / "j2_sequences_v2.npz", allow_pickle=False) as z:
        seq, fut = z["seq"].astype(np.float32), z["future"].astype(np.float32)
        delta, split = z["delta"].astype(np.float32), z["split"]

    # ---------- stage 5: ablation -----------------------------------------
    def predict_all(kind):
        if kind in ("J1", "J2-A"):
            return seq[:, -1, :]
        if kind == "J2-B":
            return seq.mean(axis=1)
        m = load_predictor("C" if kind == "J2-C" else "D")
        with torch.no_grad():
            x = torch.from_numpy(seq)
            if kind == "J2-D":
                # velocities from manifest (same source as training)
                vel = build_vel()
                z = m(x, torch.from_numpy(delta), torch.from_numpy(vel))
            else:
                z = m(x, torch.from_numpy(delta))
        return z.numpy()

    def build_vel():
        man = list(csv.DictReader(open(OUT / "j2_manifest.csv", newline="")))
        seqrows = {r["pair_id"]: r for r in csv.DictReader(
            open(SEQCSV, newline=""))}
        vel = np.zeros((len(seq), seq.shape[1], 2), dtype=np.float32)
        for r in man:
            i = int(r["row_index"])
            bb = json.loads(seqrows[r["pair_id"]]["anchor_frames"])
            centers = np.array([[(b[1][0] + b[1][2]) / 2 / 1920.0,
                                 (b[1][1] + b[1][3]) / 2 / 1080.0]
                                for b in bb], dtype=np.float32)
            v = np.gradient(centers, axis=0)
            vel[i, :len(v), :] = v[:seq.shape[1]]
        return vel

    ab = {}
    for kind in ("J1", "J2-A", "J2-B", "J2-C", "J2-D"):
        pred = predict_all(kind)
        cosv = (pred * fut).sum(-1)
        va = np.flatnonzero(split == "val")
        p = torch.from_numpy(pred[va]); t = torch.from_numpy(fut[va])
        cm = (p @ t.T).numpy()
        n = cm.shape[0]
        same = np.diag(cm)
        neg = cm[~np.eye(n, dtype=bool)]
        gap_strata = {}
        for g0, g1, name in ((0.4, 0.75, "0.5s"), (0.9, 1.25, "1.0s"),
                             (1.4, 1.9, "1.5s")):
            m_ = (delta[va] >= g0) & (delta[va] <= g1)
            if m_.sum() >= 3:
                gap_strata[name] = round(float(cosv[va][m_].mean()), 4)
        ab[kind] = {
            "mean_cos_to_target_val": round(float(cosv[va].mean()), 4),
            "same_actor_cos": round(float(same.mean()), 4),
            "diff_actor_cos": round(float(neg.mean()), 4),
            "discrimination_gap": round(float(same.mean() - neg.mean()), 4),
            "gap_strata_cos_val": gap_strata,
        }
    (OUT / "j2_ablation.json").write_text(json.dumps(ab, indent=2))
    print("[ablation]", json.dumps(ab, indent=1))

    # ---------- stage 6: association fusion --------------------------------
    ex = LatentFeatureExtractor(model_id="ckpt/dinov2-small",
                                crop_cfg=CropConfig(), cache_dir=CACHE)
    seqrows = {r["pair_id"]: r for r in csv.DictReader(
        open(SEQCSV, newline=""))}
    man_by_pair = {r["pair_id"]: r for r in csv.DictReader(
        open(OUT / "j2_manifest.csv", newline=""))}
    detail = json.loads(DETAIL.read_text())
    raw_comps = {c["comp_id"]: c for c in
                 (json.loads(l) for l in
                  Path("datasets/processed/actor_sequences/"
                       "competitions.jsonl").read_text().splitlines()
                  if l.strip())}
    # v2 exploratory pairs (their competitions are the fusion testbed)
    expl_pairs = {m["pair_id"] for m in man_by_pair.values()
                  if m["split"] == "exploratory_test"}
    comps = [d for d in detail
             if str(d["pair_id"]) in expl_pairs and len(d["cands"]) >= 2]
    model = load_predictor("C")
    eval_rows = []
    with torch.no_grad():
        for c in comps:
            pr = seqrows[str(c["pair_id"])]
            man = man_by_pair[str(c["pair_id"])]
            sid = pr["synpedas_sample"]
            f0 = int(sid.split("_f")[-1].split("-")[0])
            vp = f"repos/SynPeDAS/JAAD_crossing_moments/{sid}.mp4"
            anchor = json.loads(pr["anchor_frames"])
            hist = np.stack([ex.frame_feature(vp, f[0] - f0, f[1], sid,
                                              pr["anchor_ped_id"])
                             for f in anchor])
            dt = float(man["delta_s"])
            z = model(torch.from_numpy(hist[None].astype(np.float32)),
                      torch.tensor([dt])).numpy()[0]
            cands = []
            raw = raw_comps.get(c["comp_id"], {})
            for cd in c["cands"]:
                rc = next((x for x in raw.get("candidates", [])
                           if x["cand_ped_id"] == cd["ped"]), None)
                if rc is None:
                    continue
                try:
                    feat = ex.frame_feature(vp, rc["first_frame"] - f0,
                                            rc["bbox"], sid, rc["cand_ped_id"])
                except RuntimeError:
                    continue
                cands.append({"ped": cd["ped"],
                              "is_true": cd["is_true"],
                              "kin": cd["kin"], "cos": cd["cos"],
                              "pred_cos": round(float(np.dot(z, feat)), 4)})
            if len(cands) >= 2:
                eval_rows.append({"comp_id": c["comp_id"],
                                  "source_video_id": pr["jaad_video"],
                                  "cands": cands})
    ex.flush_cache()
    print(f"[fusion] competitions prepared: {len(eval_rows)}")

    # calibrate thresholds on TRAIN-side competitions (same as stage 3)
    tr_pairs = {m["pair_id"] for m in man_by_pair.values()
                if m["split"] == "train"}
    tr_comps = [d for d in detail
                if str(d["pair_id"]) in tr_pairs and len(d["cands"]) >= 2]
    # pred_cos thresholds need predictions for train comps too
    tr_eval = []
    with torch.no_grad():
        for c in tr_comps:
            pr = seqrows.get(str(c["pair_id"]))
            if pr is None or str(c["pair_id"]) not in man_by_pair:
                continue
            man = man_by_pair[str(c["pair_id"])]
            sid = pr["synpedas_sample"]
            f0 = int(sid.split("_f")[-1].split("-")[0])
            vp = f"repos/SynPeDAS/JAAD_crossing_moments/{sid}.mp4"
            anchor = json.loads(pr["anchor_frames"])
            hist = np.stack([ex.frame_feature(vp, f[0] - f0, f[1], sid,
                                              pr["anchor_ped_id"])
                             for f in anchor])
            z = model(torch.from_numpy(hist[None].astype(np.float32)),
                      torch.tensor([float(man["delta_s"])])).numpy()[0]
            raw = raw_comps.get(c["comp_id"], {})
            for cd in c["cands"]:
                rc = next((x for x in raw.get("candidates", [])
                           if x["cand_ped_id"] == cd["ped"]), None)
                if rc is None:
                    continue
                try:
                    feat = ex.frame_feature(vp, rc["first_frame"] - f0,
                                            rc["bbox"], sid, rc["cand_ped_id"])
                    tr_eval.append((cd["is_true"],
                                    round(float(np.dot(z, feat)), 4)))
                except RuntimeError:
                    pass
    ex.flush_cache()
    from wmeval.latent_association import calibrate_threshold
    right = [v for t, v in tr_eval if t]
    wrong = [v for t, v in tr_eval if not t]
    pred_thr, rep = calibrate_threshold(right, wrong, 0.05)
    kin_wrong = [cd["kin"] for d in tr_comps for cd in d["cands"]
                 if not cd["is_true"]]
    kin_right = [cd["kin"] for d in tr_comps for cd in d["cands"]
                 if cd["is_true"]]
    kt, _ = calibrate_threshold([-v for v in kin_right],
                                [-v for v in kin_wrong], 0.05)
    kin_thr = -kt
    print(f"[fusion] kin_thr={kin_thr} pred_thr={pred_thr} "
          f"(fpr={rep.get('achieved_fpr')})")

    def zsc(vals):
        a = np.array(vals, dtype=float)
        return (a - a.mean()) / (a.std() + 1e-9)

    def evaluate(rows, rule, nomatch=False):
        n = top1 = acc = rej = unc = wrong_acc = true_acc = 0
        amb = 0
        for r in rows:
            cands = r["cands"]
            if nomatch:
                cands = [c for c in cands if not c["is_true"]]
                if not cands:
                    continue
            n += 1
            if rule == "A1":
                by = sorted(cands, key=lambda c: c["kin"])
                top = by[0]
                d = ("accept" if top["kin"] <= kin_thr else
                     "reject" if top["kin"] >= 2 * kin_thr else "uncertain")
            elif rule == "A3":
                by = sorted(cands, key=lambda c: c["kin"])
                top = by[0]
                d = ("accept" if top["kin"] <= kin_thr and
                     top["pred_cos"] >= pred_thr else
                     "reject" if top["kin"] >= 2 * kin_thr or
                     top["pred_cos"] < 0.45 else "uncertain")
            else:  # JOINT
                zk = zsc([-c["kin"] for c in cands])
                zp = zsc([c["pred_cos"] for c in cands])
                order = sorted(range(len(cands)),
                               key=lambda i: -(zk[i] + zp[i]))
                top = cands[order[0]]
                s_top = zk[order[0]] + zp[order[0]]
                s_2nd = zk[order[1]] + zp[order[1]] if len(order) > 1 else -9
                if s_top - s_2nd < 0.5:
                    amb += 1
                    d = "uncertain"
                    top_ok = top["is_true"]
                    top1 += top_ok
                    unc += 1
                    continue
                d = "accept" if s_top >= 0.5 else (
                    "reject" if s_top <= -1.0 else "uncertain")
            top_ok = top["is_true"]
            top1 += top_ok
            if d == "accept":
                acc += 1
                if top_ok:
                    true_acc += 1
                else:
                    wrong_acc += 1
            elif d == "uncertain":
                unc += 1
            else:
                rej += 1
        return {"rule": rule, "nomatch": nomatch, "n": n,
                "top1_identity_accuracy": round(top1 / n, 4),
                "acceptance_coverage": round(acc / n, 4),
                "wrong_identity_among_accepted":
                    round(wrong_acc / acc, 4) if acc else None,
                "accept_precision": round(true_acc / acc, 4) if acc else None,
                "no_match_false_acceptance": round(acc / n, 4) if nomatch
                else None,
                "uncertain_share": round(unc / n, 4),
                "ambiguity_share": round(amb / n, 4) if rule == "JOINT"
                else None}

    results = []
    for rule in ("A1", "A3", "JOINT"):
        results.append(evaluate(eval_rows, rule))
        results.append(evaluate(eval_rows, rule, nomatch=True))

    def boot_ci(rule, metric):
        rng = random.Random(51)
        by_src = defaultdict(list)
        for r in eval_rows:
            by_src[r["source_video_id"]].append(r)
        srcs = list(by_src)
        vals = [evaluate([r for s in
                          [rng.choice(srcs) for _ in srcs]
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
           for r in ("A1", "A3", "JOINT")}

    out = {"fusion_testbed": {"n_competitions": len(eval_rows),
                              "split": "exploratory_test (v2 buckets)"},
           "thresholds": {"kin": kin_thr, "pred_cos": pred_thr,
                          "calibration": "train-only, FPR<=5% (wrong "
                                         "candidates)"},
           "results": results, "source_bootstrap_ci95": cis,
           "safety": {"no_auto_merge": "prediction similarity alone never "
                       "merges track ids",
                      "ambiguity_band": "JOINT top-2 margin < 0.5 -> "
                                        "Uncertain (task §六)"}}
    (OUT / "j2_association.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(results, indent=1))
    print(f"[fusion] -> {OUT}")


if __name__ == "__main__":
    main()
