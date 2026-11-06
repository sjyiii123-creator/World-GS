"""J1: feature-based actor association (frozen-appearance baseline).

Association score: cosine distance between L2-normalized segment
features (segment = mean of sampled per-frame DINOv2 CLS embeddings).

Fusion rule v1 (thresholds calibrated on TRAIN split only):
  accepted = (kinematic_err <= kin_thresh) AND (cos >= cos_thresh)
  — kinematics still gates gross spatial impossibility; appearance
    decides IDENTITY (the 18.2% error-merge problem of J0).
Competition ranking: candidates sorted by cosine distance.

No v3.1 semantics touched; standalone evaluation.
"""
from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np

from .latent_features import LatentFeatureExtractor


@dataclasses.dataclass
class PairFeatures:
    anchor_feat: np.ndarray
    cand_feat: np.ndarray
    cos: float
    l2: float


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# ---------------------------------------------------------------------------
# J1.1 calibration (P0 fix): explicit protocol, NO silent fallback
# ---------------------------------------------------------------------------
def calibrate_threshold(feats_pos, feats_neg, target_fpr: float = 0.05):
    """Strict FPR-constrained threshold selection (J1.1 repaired).

    Protocol (deterministic, regression-tested, no ULP crawling):
      * max allowed negatives passing = k = floor(target_fpr * n_neg)
        (floor, NOT ceil: k/n_neg <= target_fpr by construction — the
        v2 bug was ceil(21/406=5.17%>5%) plus a 1-ULP while-loop that
        could never cross a 4-decimal tie gap);
      * threshold = smallest unique negative score u such that
        count(neg >= u) <= k  (unique-value scan, ties handled exactly;
        TPR-friendly among all FPR-feasible choices);
      * if no unique value satisfies the constraint, the threshold is
        nextafter(max(neg)) (accepts nothing) — always FPR-feasible;
      * NO silent fallback: report carries achieved train FPR/TPR;
        infeasible (no data) returns (None, report).

    Returns (threshold, report_dict).
    """
    neg = np.sort(np.asarray(list(feats_neg), dtype=float))
    pos = np.asarray(list(feats_pos), dtype=float)
    report = {"n_pos": int(pos.size), "n_neg": int(neg.size),
              "target_fpr": target_fpr,
              "protocol": "floor-quantile-unique-scan-no-fallback"}
    if neg.size == 0 or pos.size == 0 or not (0 < target_fpr < 1):
        report["reason"] = ("no negatives/positives or invalid target — "
                            "FPR/TPR undefined")
        return None, report
    k = int(np.floor(target_fpr * neg.size))
    uniq = np.unique(neg)
    t = None
    for u in uniq:
        if float(np.count_nonzero(neg >= u)) <= k:
            t = float(u)
            break
    if t is None:
        t = float(np.nextafter(uniq[-1], np.inf))   # accepts nothing
    fpr = float(np.mean(neg >= t))
    if fpr > target_fpr + 1e-12:
        report.update({"reason": "infeasible: constraint unsatisfiable",
                       "achieved_fpr": round(fpr, 4)})
        return None, report
    # round for readability ONLY if it does not re-admit tied negatives
    t_out = round(t, 4)
    if float(np.mean(neg >= t_out)) <= target_fpr + 1e-12:
        t = t_out
    tpr = float(np.mean(pos >= t))
    report.update({"threshold": round(t, 6), "achieved_fpr": round(fpr, 4),
                   "achieved_tpr": round(tpr, 4),
                   "max_allowed_negatives": k,
                   "tradeoff_note": "FPR target met; TPR is whatever it "
                                    "is (report infeasibility instead of "
                                    "silently relaxing)"})
    return t, report


# ---------------------------------------------------------------------------
# J1.1 metrics (P1 fix): explicit denominators
# ---------------------------------------------------------------------------
def pair_metrics(labels, accepted):
    """Binary association metrics with EXPLICIT denominators.

    labels/accepted: sequences of bool (True = same_identity / accepted).
    Definitions:
      FP (false merge count)      = accepted & ~label
      false_merge_fraction_of_all = FP / n_all_pairs
      FPR                          = FP / n_negatives      (all diff-identity)
      missed_relink_rate           = FN / n_positives
      pair_accuracy                = correct / n_all_pairs
      balanced_accuracy            = (TPR + TNR) / 2
      precision                    = TP / (TP + FP)
    """
    lab = np.asarray(list(labels), dtype=bool)
    acc = np.asarray(list(accepted), dtype=bool)
    n = lab.size
    tp = int(np.sum(acc & lab)); fp = int(np.sum(acc & ~lab))
    fn = int(np.sum(~acc & lab)); tn = int(np.sum(~acc & ~lab))
    n_pos, n_neg = tp + fn, fp + tn
    return {
        "n_all_pairs": n,
        "false_merge_count": fp,
        "false_merge_fraction_of_all_pairs": round(fp / n, 4) if n else None,
        "false_positive_rate": round(fp / n_neg, 4) if n_neg else None,
        "missed_relink_rate": round(fn / n_pos, 4) if n_pos else None,
        "pair_accuracy": round((tp + tn) / n, 4) if n else None,
        "balanced_accuracy": round(
            (tp / n_pos + tn / n_neg) / 2, 4) if n_pos and n_neg else None,
        "precision": round(tp / (tp + fp), 4) if (tp + fp) else None,
        "true_positive_rate_recall": round(tp / n_pos, 4) if n_pos else None,
        "counts": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


class FeatureAssociation:
    def __init__(self, extractor: LatentFeatureExtractor,
                 video_root: str = "repos/SynPeDAS/JAAD_crossing_moments",
                 ext: str = ".mp4") -> None:
        self.ex = extractor
        self.video_root = video_root
        self.ext = ext

    def video_path(self, synpedas_sample: str) -> str:
        return f"{self.video_root}/{synpedas_sample}{self.ext}"

    def pair_features(self, sample: str, window_start: int,
                      anchor_frames: list, anchor_ped: str,
                      cand_frames: list, cand_ped: str) -> PairFeatures:
        vp = self.video_path(sample)
        a = self.ex.segment_feature(vp, window_start, anchor_frames,
                                    sample, anchor_ped)
        c = self.ex.segment_feature(vp, window_start, cand_frames,
                                    sample, cand_ped)
        return PairFeatures(a, c, round(cos_sim(a, c), 4),
                            round(float(np.linalg.norm(a - c)), 4))

    @staticmethod
    def calibrate_threshold(feats_pos: list, feats_neg: list,
                            target_fpr: float = 0.05) -> float:
        """DEPRECATED (J1.1 audit finding): the midpoint fallback below
        silently violated the FPR constraint (measured train FPR 28.33%
        at the claimed <=5% target on the J1 split).  Kept ONLY so the
        original run_j1_baseline.py stays reproducible verbatim; new code
        MUST use the module-level calibrate_threshold (strict, no
        fallback, returns a checkable report)."""
        if not feats_neg:
            return 0.0
        neg = sorted(feats_neg)
        k = max(0, int(len(neg) * (1 - target_fpr)) - 1)
        thr = neg[k]
        # ensure it separates at least the median positive
        pos_med = float(np.median(feats_pos)) if feats_pos else thr
        if thr > pos_med:  # degenerate: no separation, fall back to midpoint
            lo = float(np.percentile(feats_neg, 25)) if feats_neg else 0
            hi = float(np.percentile(feats_pos, 75)) if feats_pos else 1
            thr = (lo + hi) / 2
        return round(float(thr), 4)

    @staticmethod
    def fused_decision(kin_err: float, kin_thresh: float,
                       cos: float, cos_thresh: float) -> bool:
        return (kin_err <= kin_thresh) and (cos >= cos_thresh)


def calibrate_lower_is_better(scores_pos, scores_neg,
                              target_fpr: float = 0.05):
    """J2.3.2 P0-B: threshold calibration for DISTANCE-like scores
    (LOWER = better; accept rule is score <= threshold).

    Protocol: smallest threshold t such that the fraction of TRAIN
    negatives with score <= t is <= target_fpr.  No fallback; the
    report carries achieved FPR/TPR.  Returns (threshold, report).
    """
    neg = np.sort(np.asarray(list(scores_neg), dtype=float))
    pos = np.asarray(list(scores_pos), dtype=float)
    rep = {"n_pos": int(pos.size), "n_neg": int(neg.size),
           "target_fpr": target_fpr,
           "protocol": "lower-is-better-max-feasible-no-fallback"}
    if neg.size == 0 or pos.size == 0 or not (0 < target_fpr < 1):
        rep["reason"] = "no positives/negatives or invalid target"
        return None, rep
    k = int(np.floor(target_fpr * neg.size))
    uniq = np.unique(neg)
    # J2.3.2a P0-A fix: collect ALL feasible thresholds (FPR<=target)
    # and select the MAXIMUM — maximum TPR under the FPR constraint.
    # (The old first-hit scan returned the MINIMUM feasible threshold,
    # e.g. 0.0036/TPR 22% where 0.0183/TPR 71% was equally feasible.)
    feasible = [float(u) for u in uniq
                if float(np.count_nonzero(neg <= u)) <= k]
    t = max(feasible) if feasible else \
        float(np.nextafter(uniq[0], -np.inf))  # accepts nothing
    fpr = float(np.mean(neg <= t))
    if fpr > target_fpr + 1e-12:
        rep.update({"reason": "infeasible", "achieved_fpr": round(fpr, 4)})
        return None, rep
    tpr = float(np.mean(pos <= t))
    t_out = round(t, 4)
    if float(np.mean(neg <= t_out)) > target_fpr + 1e-12:
        t_out = t
    rep.update({"threshold": t_out, "achieved_fpr": round(fpr, 4),
                "achieved_tpr": round(tpr, 4),
                "max_allowed_negatives": k})
    return t_out, rep
