"""J2.1 re-evaluation: fixed identity masks + per-candidate time (Gates 0-1).

Fixes applied relative to the J2 round (task §二/§三):
  * identity relation = (source_video_id, actor_id) — the OLD
    cross-cos evaluation used matrix off-diagonals as "different actor",
    mislabeling SAME-actor pairs from different sequences as negatives
    (measured: 42/3540 val pairs = 1.2%; they inflate the negative mean
    and thus UNDERSTATE the discrimination gap);
  * easy vs hard negatives: hard = different actor in the SAME source
    video (co-occurring pedestrians); easy = different source;
  * identical numeric actor ids in different videos are DIFFERENT
    pedestrians (keyed by source);
  * per-candidate time: each competition candidate gets its OWN delta
    (candidate_first_time_s - anchor_last_time_s) — the old fusion used
    the TRUE match's delta for ALL candidates, leaking which candidate
    is correct via the time input; now the predictor is queried once
    per candidate with that candidate's own time. Invalid intervals
    (<=0 or missing) are marked unusable and excluded from accept.

Old checkpoints are loaded READ-ONLY (j2_predictor.pt / j2_predictor_D.pt);
old results (j2_ablation.json / j2_association.json) untouched.
Output: runs/j2_1_reeval/{pair_reeval.json, competition_reeval.json}
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")

from wmeval.latent_features import CropConfig, LatentFeatureExtractor
from run_j2_train import ActorPredictor, ActorPredictorD

OUT = Path("runs/j2_1_reeval")
J2X = Path("runs/j2_experimental")
DETAIL = Path("runs/j3_lite/competitions_detail.json")
RAW = Path("datasets/processed/actor_sequences/competitions.jsonl")
SEQCSV = Path("datasets/processed/actor_sequences/samples.csv")
CACHE = Path("runs/j1_baseline/feature_cache")


def identity_key(row):
    return (row["source_video_id"], row["actor_id"])


def load_predictor(tag):
    m = ActorPredictorD() if tag == "D" else ActorPredictor()
    m.load_state_dict(torch.load(
        J2X / ("j2_predictor_D.pt" if tag == "D" else "j2_predictor.pt"),
        map_location="cpu"))
    m.eval()
    return m


def predict_batch(model, seq, deltas, use_vel=False, vel=None):
    with torch.no_grad():
        x = torch.from_numpy(seq.astype(np.float32))
        d = torch.from_numpy(np.asarray(deltas, dtype=np.float32))
        if use_vel:
            z = model(x, d, torch.from_numpy(vel))
        else:
            z = model(x, d)
    return z.numpy()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with np.load(J2X / "j2_sequences_v2.npz", allow_pickle=False) as z:
        seq, fut = z["seq"].astype(np.float32), z["future"].astype(np.float32)
        delta, split = z["delta"].astype(np.float32), z["split"]
        srcs = z["source_video_id"]
        actors = z["actor_id"]
    man = list(csv.DictReader(open(J2X / "j2_manifest.csv", newline="")))
    # velocities for J2-D (same construction as training)
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

    # ============ pair-level re-evaluation (fixed identity masks) ========
    def gap_stats(pred_kind):
        if pred_kind in ("J1", "J2-A"):
            pred = seq[:, -1, :]
        elif pred_kind == "J2-B":
            pred = seq.mean(axis=1)
        else:
            model = load_predictor("C" if pred_kind == "J2-C" else "D")
            use_vel = pred_kind == "J2-D"
            pred = predict_batch(model, seq, delta, use_vel,
                                 vel if use_vel else None)
        results = {}
        for sp in ("val", "exploratory_test"):
            idx = np.flatnonzero(split == sp)
            if len(idx) < 4:
                continue
            p = pred[idx]; t = fut[idx]
            cos = p @ t.T
            ident = [identity_key({"source_video_id": srcs[i],
                                   "actor_id": actors[i]}) for i in idx]
            same_mask = np.zeros((len(idx), len(idx)), dtype=bool)
            for a in range(len(idx)):
                for b in range(len(idx)):
                    same_mask[a, b] = (a == b) or (ident[a] == ident[b])
            same_pairs = cos[same_mask]
            diff_mask = (~same_mask)
            # split negatives: hard = same source, easy = different source
            hard_mask = np.zeros_like(diff_mask)
            for a in range(len(idx)):
                for b in range(len(idx)):
                    if diff_mask[a, b] and srcs[idx[a]] == srcs[idx[b]]:
                        hard_mask[a, b] = True
            easy_mask = diff_mask & (~hard_mask)
            results[sp] = {
                "n_sequences": int(len(idx)),
                "same_identity_cos": round(float(same_pairs.mean()), 4),
                "diff_identity_cos_ALL": round(float(cos[diff_mask].mean()), 4),
                "diff_identity_cos_hard_same_source":
                    round(float(cos[hard_mask].mean()), 4)
                if hard_mask.any() else None,
                "diff_identity_cos_easy_cross_source":
                    round(float(cos[easy_mask].mean()), 4)
                if easy_mask.any() else None,
                "gap_ALL": round(float(same_pairs.mean()
                                        - cos[diff_mask].mean()), 4),
                "gap_hard": round(float(same_pairs.mean()
                                        - cos[hard_mask].mean()), 4)
                if hard_mask.any() else None,
            }
        return results

    pair = {}
    for kind in ("J1", "J2-B", "J2-C", "J2-D"):
        pair[kind] = gap_stats(kind)
    # old values for the diff report
    try:
        old = json.loads((J2X / "j2_ablation.json").read_text())
        old_summary = {k: {"old_gap_val": v["discrimination_gap"],
                           "old_same": v["same_actor_cos"],
                           "old_diff": v["diff_actor_cos"]}
                       for k, v in old.items() if k in pair}
    except FileNotFoundError:
        old_summary = {}
    (OUT / "pair_reeval.json").write_text(json.dumps(
        {"fixed_identity_protocol": pair, "old_values": old_summary,
         "fix_note": "old negatives = matrix off-diagonal (contained "
                     "same-actor cross-sequence pairs -> negative mean "
                     "inflated -> gap UNDERSTATED); new negatives = "
                     "different (source,actor), split hard/easy"},
        indent=2))
    print("[pair]", json.dumps(pair, indent=1)[:1200])

    # ============ competition re-evaluation (per-candidate time) =========
    ex = LatentFeatureExtractor(model_id="ckpt/dinov2-small",
                                crop_cfg=CropConfig(), cache_dir=CACHE)
    raw_comps = {c["comp_id"]: c for c in
                 (json.loads(l) for l in RAW.read_text().splitlines()
                  if l.strip())}
    detail = json.loads(DETAIL.read_text())
    man_by_pair = {r["pair_id"]: r for r in man}
    model = load_predictor("C")

    def prep_rows(pair_filter):
        rows = []
        with torch.no_grad():
            for c in detail:
                pid = str(c["pair_id"])
                if pid not in man_by_pair or not pair_filter(man_by_pair[pid]):
                    continue
                pr = seqrows[pid]
                man_r = man_by_pair[pid]
                sid = pr["synpedas_sample"]
                f0 = int(sid.split("_f")[-1].split("-")[0])
                fps = float(pr["fps"])
                vp = f"repos/SynPeDAS/JAAD_crossing_moments/{sid}.mp4"
                anchor = json.loads(pr["anchor_frames"])
                hist = np.stack([ex.frame_feature(vp, f[0] - f0, f[1], sid,
                                                  pr["anchor_ped_id"])
                                 for f in anchor])
                anchor_last_t = anchor[-1][0] / fps
                raw = raw_comps.get(c["comp_id"], {})
                cands = []
                for cd in c["cands"]:
                    rc = next((x for x in raw.get("candidates", [])
                               if x["cand_ped_id"] == cd["ped"]), None)
                    if rc is None:
                        continue
                    # per-candidate delta (GATE 1 FIX): this candidate's
                    # OWN first-observation time minus anchor end
                    cand_t = rc["first_frame"] / fps
                    d = cand_t - anchor_last_t
                    usable = d > 0 and np.isfinite(d)
                    try:
                        feat = ex.frame_feature(
                            vp, rc["first_frame"] - f0, rc["bbox"], sid,
                            rc["cand_ped_id"])
                    except RuntimeError:
                        continue
                    if usable:
                        z = model(torch.from_numpy(hist[None].astype(np.float32)),
                                  torch.tensor([float(d)])).numpy()[0]
                        pred_cos = round(float(np.dot(z, feat)), 4)
                    else:
                        pred_cos = None  # unusable time -> unknown evidence
                    cands.append({"ped": cd["ped"], "is_true": cd["is_true"],
                                  "kin": cd["kin"], "cos": cd["cos"],
                                  "cand_delta_s": round(float(d), 4)
                                  if usable else None,
                                  "pred_cos": pred_cos})
                if len(cands) >= 2:
                    rows.append({"comp_id": c["comp_id"],
                                 "source_video_id": pr["jaad_video"],
                                 "cands": cands})
        return rows

    eval_rows = prep_rows(lambda m: m["split"] == "exploratory_test")
    tr_rows = prep_rows(lambda m: m["split"] == "train")
    ex.flush_cache()
    print(f"[comp] competitions: eval {len(eval_rows)} / train {len(tr_rows)}")

    # thresholds: kin (same as before, features-independent) and pred_cos
    # calibrated on train with PER-CANDIDATE predictions
    from wmeval.latent_association import calibrate_threshold
    right = [c["pred_cos"] for r in tr_rows for c in r["cands"]
             if c["is_true"] and c["pred_cos"] is not None]
    wrong = [c["pred_cos"] for r in tr_rows for c in r["cands"]
             if not c["is_true"] and c["pred_cos"] is not None]
    pred_thr, rep = calibrate_threshold(right, wrong, 0.05)
    kin_right = [c["kin"] for r in tr_rows for c in r["cands"] if c["is_true"]]
    kin_wrong = [c["kin"] for r in tr_rows for c in r["cands"]
                 if not c["is_true"]]
    kt, _ = calibrate_threshold([-v for v in kin_right],
                                [-v for v in kin_wrong], 0.05)
    kin_thr = -kt
    print(f"[comp] kin_thr={kin_thr} pred_thr={pred_thr} "
          f"(fpr={rep.get('achieved_fpr')})")

    def evaluate(rows, rule, nomatch=False):
        n = top1 = acc = rej = unc = wrong_acc = true_acc = 0
        unusable = 0
        for r in rows:
            cands = r["cands"]
            if nomatch:
                cands = [c for c in cands if not c["is_true"]]
                if not cands:
                    continue
            n += 1
            by = sorted(cands, key=lambda c: c["kin"])
            top = by[0]
            if rule == "A1":
                d = ("accept" if top["kin"] <= kin_thr else
                     "reject" if top["kin"] >= 2 * kin_thr else "uncertain")
            else:  # A3 with per-candidate times; unusable -> uncertain
                pc = top["pred_cos"]
                if pc is None:
                    d = "uncertain"
                    unusable += 1
                elif top["kin"] <= kin_thr and pc >= pred_thr:
                    d = "accept"
                elif top["kin"] >= 2 * kin_thr or (pc is not None and
                                                   pc < 0.45):
                    d = "reject"
                else:
                    d = "uncertain"
            top1 += top["is_true"]
            if d == "accept":
                acc += 1
                if top["is_true"]:
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
                "no_match_false_acceptance": round(acc / n, 4)
                if nomatch else None,
                "uncertain_share": round(unc / n, 4),
                "unusable_time_evidence": unusable}

    results = []
    for rule in ("A1", "A3"):
        results.append(evaluate(eval_rows, rule))
        results.append(evaluate(eval_rows, rule, nomatch=True))
    out = {"fix": "per-candidate delta (candidate_first_time_s - "
                  "anchor_last_time_s); no true-match time is used for "
                  "other candidates; unusable intervals -> Unknown",
           "thresholds": {"kin": kin_thr, "pred_cos": pred_thr,
                          "calibration": "train-only FPR<=5% (wrong "
                                         "candidates), per-candidate times"},
           "results": results}
    (OUT / "competition_reeval.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
