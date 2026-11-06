"""J2.3.1 P0-4: ranking loss with CORRECT sign + gradient contract.

Correct form (task §八):
    L_rank = max(0, margin + max_j s_j^-  -  s^+)
score HIGHER = more credible identity association.  Zero when
s^+ > max s^- + margin; positive gradient pushes s^+ UP and the
hardest negative DOWN.

J2.3 design doc had the sign REVERSED (max(0, m + s^+ - max s^-))
which would penalize the correct candidate OUTSCORING negatives —
optimization direction opposite to the task.  Errata recorded in
docs/j2_3_1_ranking_loss_tests.md; no training ever ran with the wrong
formula (D4 not started — design-only).
"""
from __future__ import annotations

from typing import Optional

import torch
from torch.nn import functional as F


def ranking_loss(score_pos: torch.Tensor,
                  scores_neg: torch.Tensor,
                  margin: float = 0.1) -> torch.Tensor:
    """score_pos: (B,) positive candidate scores.
    scores_neg: (B, K) negative candidate scores (K >= 1; K == 0 rows
    must be handled by the caller — see no_negative behavior below).

    Hardest-negative aggregation: max over the K negatives.
    Identity rule: sequences of the SAME identity must never appear in
    scores_neg (caller contract, regression-tested at the sampler level).
    """
    if scores_neg.shape[-1] == 0:
        # no negatives in the batch: no ranking constraint — return a
        # DETACHED zero so the loss graph stays valid but contributes
        # nothing (explicit, never silently nan/zero-with-grad)
        return score_pos.new_zeros(())
    hardest = scores_neg.max(dim=-1).values
    return (margin + hardest - score_pos).clamp_min(0).mean()


def ranking_loss_masked(score_pos: torch.Tensor,
                        scores_neg: torch.Tensor,
                        valid_negative_mask: torch.Tensor,
                        margin: float = 0.1) -> torch.Tensor:
    """J2.3.2 P0-8: variable-length negative sets per anchor.

    scores_neg: (B, K) padded; valid_negative_mask: (B, K) bool where
    True = a REAL negative (False = padding — excluded from the
    hardest-negative max; same-identity candidates must NEVER be marked
    True, caller contract).  Rows with zero valid negatives contribute
    zero rank loss (excluded from the mean, not silently zero-filled
    into the gradient).
    """
    if scores_neg.shape != valid_negative_mask.shape:
        raise ValueError("mask shape mismatch")
    if not valid_negative_mask.any():
        return score_pos.new_zeros(())
    # masked max: -inf on padding so it never wins the hardest slot
    neg_masked = scores_neg.masked_fill(~valid_negative_mask, float("-inf"))
    hardest = neg_masked.max(dim=-1).values
    has_neg = valid_negative_mask.any(dim=-1)            # (B,)
    per_row = (margin + hardest - score_pos).clamp_min(0)
    per_row = torch.where(has_neg, per_row, torch.zeros_like(per_row))
    denom = has_neg.float().sum().clamp_min(1.0)
    return (per_row.sum() / denom)


def prediction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (1 - F.cosine_similarity(pred, target, dim=-1)).mean()
