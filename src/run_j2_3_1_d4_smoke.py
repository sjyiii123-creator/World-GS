"""J2.3.1 §十: D4 training-plan smoke (forward/backward ONLY, no training).

Five admission gates (task §十一.5 subset):
  1 gradient direction on rank loss (7/7 selftest passed separately);
  2 provenance: every training row carries a dedup candidate_feature_key;
  3 time causality: per-candidate delta, predictor input excludes
    candidates/future;
  4 single-batch overfit on a tiny D4 loss;
  5 no-negative boundary: batch without negatives contributes zero
    rank loss (no crash).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, "src")

from wmeval.latent_world_state.residual_predictor import ResidualPredictor
from wmeval.latent_world_state.ranking_loss import (
    prediction_loss, ranking_loss)

CAND = Path("runs/j2_3_1_alignment/tier1_candidates.csv")


def main() -> None:
    gates = {}
    # G1 gradient direction (synthetic, mirrors selftest)
    sp = torch.tensor([0.3], requires_grad=True)
    ranking_loss(sp, torch.tensor([[0.55]]), 0.1).backward()
    gates["G1_gradient_direction"] = bool(sp.grad is not None
                                          and float(sp.grad) < 0)
    # G2 provenance: every row has a dedup feature key
    rows = list(csv.DictReader(open(CAND, newline="")))
    keys = [r["candidate_feature_key"] for r in rows]
    # uniqueness is PER ANCHOR (the same competitor legitimately serves
    # several anchors' competition sets); global uniqueness was a wrong
    # assertion — a competitor is ONE observation serving many windows
    from collections import defaultdict
    per_anchor = defaultdict(list)
    for r in rows:
        per_anchor[r["anchor_id"]].append(r["candidate_feature_key"])
    dup_in_anchor = sum(1 for a, ks in per_anchor.items()
                        if len(set(ks)) != len(ks))
    gates["G2_provenance_keys"] = bool(
        all("|" in k for k in keys) and dup_in_anchor == 0
        and len(set(keys)) >= 2000)  # dedup'd distinct observations
    # G3 causality: predictor input = (history, delta) only
    import inspect
    sig = list(inspect.signature(ResidualPredictor.forward).parameters)
    gates["G3_causal_api"] = sig == ["self", "history", "delta"]
    # G4 single-batch overfit of the FULL D4 loss
    torch.manual_seed(51)
    D, T = 384, 12
    model = ResidualPredictor()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    hist = F.normalize(torch.randn(8, T, D), dim=-1)
    dts = torch.rand(8) + 0.4
    pos = F.normalize(torch.randn(8, D), dim=-1)
    negs = F.normalize(torch.randn(8, 3, D), dim=-1)
    first = None
    for it in range(300):
        omega = model(hist, dts)
        lp = prediction_loss(omega, pos)
        lr_ = ranking_loss(torch.einsum("bd,bd->b", omega, pos),
                           torch.einsum("bd,bkd->bk", omega, negs), 0.1)
        loss = lp + 0.5 * lr_
        if first is None:
            first = float(loss)
        opt.zero_grad(); loss.backward(); opt.step()
    gates["G4_overfit"] = {"first": round(first, 4),
                           "after_300": round(float(loss), 4),
                           "passed": float(loss) < 0.10}
    # G5 no-negative boundary
    l5 = ranking_loss(torch.tensor([0.5]), torch.zeros(1, 0))
    gates["G5_no_negative_boundary"] = bool(
        float(l5) == 0.0 and not l5.requires_grad)
    out = {"gates": gates,
           "config": {"lambda_rank_search": [0.1, 0.3, 0.5, 1.0],
                      "selection": "val M_hard only (final_test untouched)",
                      "note": "SMOKE ONLY — no formal training started"}}
    (Path("runs/j2_3_1_alignment") / "d4_smoke.json").write_text(
        json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))
    all_pass = (gates["G1_gradient_direction"] and gates["G2_provenance_keys"]
                and gates["G3_causal_api"] and gates["G4_overfit"]["passed"]
                and gates["G5_no_negative_boundary"])
    print("ALL_D4_GATES_PASS:", all_pass)


if __name__ == "__main__":
    main()
