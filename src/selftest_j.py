"""J-series self-tests (CPU-runnable without model weights).

Covers:
  * crop logic of LatentFeatureExtractor (small-object padding, clamping,
    context expansion, square resize) — pure OpenCV;
  * FeatureAssociation threshold calibration (degenerate separation case);
  * LatentPredictor forward shape + cosine loss + no-leakage API shape;
  * LatentPredictor.delta_bucket monotonicity.

Full GPU integration (DINOv2 benchmark/extraction) is exercised by
src/run_j1_baseline.py once weights are present.

Run:
    PYTHONPATH=src /home/moz/miniconda3/envs/wm_eval/bin/python src/selftest_j.py
"""
from __future__ import annotations

import sys

sys.path.insert(0, "src")

import numpy as np
import torch
import cv2

from wmeval.latent_features import CropConfig, LatentFeatureExtractor
from wmeval.latent_association import FeatureAssociation
from wmeval.latent_predictor import LatentPredictor, assemble_history

ok, fail = [], []


def check(name, cond, detail=""):
    (ok if cond else fail).append(name)
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")


# --- crop logic (no model needed: instantiate __new__) -------------------
cfg = CropConfig()
ex = LatentFeatureExtractor.__new__(LatentFeatureExtractor)  # skip __init__
ex.cfg = cfg

frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
frame[:] = (40, 120, 200)

# 1. tiny box -> clamped to min_crop with context
tiny = ex._crop(frame, (900, 500, 940, 560))          # 40x60 box
check("crop: tiny box (40x60) resized to square crop_size",
      tiny.shape[:2] == (cfg.crop_size, cfg.crop_size), str(tiny.shape))

# 2. context expansion actually widens the crop region: compare a stripe
# pattern visibility — center stripe kept, surroundings included
# (use a large bbox so min_crop does not dominate the comparison)
frame2 = np.zeros((1080, 1920, 3), dtype=np.uint8)
frame2[:, 700:900] = 255                                # vertical white stripe
bbox_center = (700, 400, 820, 660)                      # flush with stripe edge
c1 = ex._crop(frame2, bbox_center)
white_frac_with_ctx = (c1 > 128).mean()
cfg_narrow = CropConfig(context_pad=0.0)
ex.cfg = cfg_narrow
c2 = ex._crop(frame2, bbox_center)
white_frac_no_ctx = (c2 > 128).mean()
ex.cfg = cfg
check("crop: context_pad widens observed region (stripe fraction drops)",
      white_frac_with_ctx < white_frac_no_ctx,
      f"{white_frac_with_ctx:.2f} < {white_frac_no_ctx:.2f}")

# 3. boundary clamp: box near right edge does not wrap
edge = ex._crop(frame, (1900, 500, 1930, 560))
check("crop: edge box clamps to image boundary (no exception, square)",
      edge.shape[:2] == (cfg.crop_size, cfg.crop_size))

# --- threshold calibration ------------------------------------------------
from wmeval.latent_association import (calibrate_threshold as strict_cal,
                                       pair_metrics)
pos = np.random.RandomState(0).normal(0.9, 0.02, 200)
neg = np.random.RandomState(1).normal(0.5, 0.05, 200)
thr, rep = strict_cal(list(pos), list(neg))
check("calibration: separable case -> threshold between modes",
      0.55 < thr < 0.85 and rep["achieved_fpr"] <= 0.05 + 1e-9,
      f"thr={thr} fpr={rep['achieved_fpr']}")
# degenerate: no separation (complete overlap)
rng = np.random.RandomState(2)
thr2, rep2 = strict_cal(list(rng.normal(0.5, 0.05, 100)),
                        list(rng.normal(0.5, 0.05, 100)))
check("calibration: complete overlap -> still FPR-feasible (no fallback), "
      "TPR reported as-is",
      thr2 is not None and rep2["achieved_fpr"] <= 0.05 + 1e-9,
      f"thr={thr2} fpr={rep2.get('achieved_fpr')} tpr={rep2.get('achieved_tpr')}")
# no negatives
thr3, rep3 = strict_cal([0.9, 0.8], [])
check("calibration: no negatives -> (None, reason), never a fake number",
      thr3 is None and "no negatives" in rep3["reason"])
# heavy ties (equal scores): FPR constraint must hold exactly on ties
pos_t = [0.9] * 100
neg_t = [0.7] * 20 + [0.5] * 180
thr4, rep4 = strict_cal(pos_t, neg_t)
passed_neg = sum(1 for x in neg_t if x >= thr4)
check("calibration: tied scores respect FPR (<= floor(0.05*neg))",
      thr4 is not None and passed_neg <= 10,
      f"thr={thr4} passed={passed_neg}/200")
# infeasible-by-target (target so small that even top value over-counts
# with ties covering everything)
thr5, rep5 = strict_cal([0.6] * 50, [0.6] * 50, target_fpr=0.0)
check("calibration: target=0 with full ties at max -> accepts nothing "
      "(threshold above max) or None",
      thr5 is None or all(x < thr5 for x in [0.6] * 50))
# achieved TPR is honest (not inflated)
check("calibration: report carries achieved TPR (train)",
      "achieved_tpr" in rep and 0 <= rep["achieved_tpr"] <= 1)

# --- pair metrics: explicit denominators -----------------------------------
m = pair_metrics([True, True, False, False], [True, False, False, True])
check("metrics: hand-computed case tp=1 fp=1 fn=1 tn=1",
      m["counts"] == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}
      and m["false_positive_rate"] == 0.5          # fp/(fp+tn)=1/2
      and m["false_merge_fraction_of_all_pairs"] == 0.25   # fp/4
      and m["missed_relink_rate"] == 0.5           # fn/(tp+fn)=1/2
      and m["pair_accuracy"] == 0.5
      and m["precision"] == 0.5
      and m["balanced_accuracy"] == 0.5, str(m["counts"]))
m2 = pair_metrics([True, True], [True, True])
check("metrics: degenerate (no negatives) -> FPR None, not zero",
      m2["false_positive_rate"] is None and m2["precision"] == 1.0)

# --- occlusion strategies (config semantics, CPU) ---------------------------
from wmeval.latent_features import CropConfig, select_frames
c_all = CropConfig(occlusion_strategy="all")
c_pref = CropConfig(occlusion_strategy="prefer_visible")
c_w = CropConfig(occlusion_strategy="weighted")
check("occlusion: strategy field in cache identity (asdict differs)",
      c_all != c_pref and c_pref != c_w)
check("occlusion: visibility classes mapped (0/1/2)",
      (c_pref.visibility_class(0), c_pref.visibility_class(1),
       c_pref.visibility_class(2)) ==
      ("visible", "partially_occluded", "heavily_occluded"))

# frame selection semantics (pure function, J1.1 §五 fix)
mix = [[i, (0, 0, 1, 1), 0] for i in range(6)] + \
      [[6 + i, (0, 0, 1, 1), 1] for i in range(6)]     # 6 visible + 6 occ1
sel_a = select_frames(mix, "all", 3)
sel_b = select_frames(mix, "prefer_visible", 3)
check("selection: prefer_visible picks VISIBLE frames when available "
      "(B differs from A on occ==1 data)",
      all(f[2] == 0 for f in sel_b) and len(sel_b) == 3
      and sel_a != sel_b,
      f"A={[f[2] for f in sel_a]} B={[f[2] for f in sel_b]}")
only_occ1 = [[i, (0, 0, 1, 1), 1] for i in range(6)]
sel_b2 = select_frames(only_occ1, "prefer_visible", 3)
check("selection: no visible frames -> falls back, nothing deleted",
      len(sel_b2) == 3 and all(f[2] == 1 for f in sel_b2))
occ_mix3 = [[i, (0, 0, 1, 1), min(i // 3, 2)] for i in range(9)]
sel_b3 = select_frames(occ_mix3, "prefer_visible", 3)
check("selection: tier priority 0 -> 1 -> 2 (never empty)",
      len(sel_b3) == 3 and max(f[2] for f in sel_b3) <= 1)
sel_c = select_frames(mix, "weighted", 3)
check("selection: weighted uses ALL frames (weights at aggregation)",
      len(sel_c) == 3 and max(f[2] for f in sel_c) == 1)
sel_short = select_frames(mix[:2], "prefer_visible", 3)
check("selection: fewer frames than k -> all kept",
      len(sel_short) == 2)

# --- latent predictor -----------------------------------------------------
pred = LatentPredictor()
B, T, D = 3, 12, pred.cfg.input_dim
hist = torch.nn.functional.normalize(
    torch.randn(B, T, D), dim=-1)
z = pred(hist, delta_seconds=1.5)
check("predictor: forward shape (B, D) matches input dim",
      z.shape == (B, D), str(tuple(z.shape)))
check("predictor: output L2-normalized (unit sphere)",
      torch.allclose(z.norm(dim=-1), torch.ones(B), atol=1e-4))
target = torch.nn.functional.normalize(torch.randn(B, D), dim=-1)
loss = LatentPredictor.loss(z, target)
check("predictor: cosine loss finite and in [0, 2]",
      0 <= float(loss) <= 2.0, f"loss={float(loss):.4f}")
# no-leakage API: forward accepts ONLY history
import inspect
sig = inspect.signature(LatentPredictor.forward)
check("predictor: forward() takes no future argument (no-leakage API)",
      list(sig.parameters) == ["self", "history", "delta_seconds"],
          str(list(sig.parameters)))
# delta bucket monotone
buckets = [LatentPredictor.delta_bucket(d) for d in
           (0.1, 0.5, 1.0, 2.0, 4.0, 8.0)]
check("predictor: delta_bucket monotone non-decreasing over time",
      all(a <= b for a, b in zip(buckets[:-1], buckets[1:])), str(buckets))
# assemble_history from numpy
h = assemble_history([np.ones(D) / np.sqrt(D)] * 5)
check("assemble_history: (1, T, D) tensor from numpy list",
      h.shape == (1, 5, D))

print(f"\n{len(ok)} passed, {len(fail)} failed")
if fail:
    print("FAILED:", fail)
    sys.exit(1)
print("ALL J-SERIES SELF-TESTS PASSED")
