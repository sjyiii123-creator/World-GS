"""J2.2 Static Residual Predictor: z_pred = normalize(z_last + residual).

Research hypothesis (j2_v2 design B): v1's direct GRU converged to a
"mean-face" solution (commonness).  Residual parameterization hands
the EASY part (appearance near-invariance) to the static last frame
and asks the GRU only for the DRIFT CORRECTION — a degenerate solution
in residual space is ZERO output (safe fallback == static baseline),
whereas v1's degenerate solution was the batch-mean feature (harmful).

Contract (task §四 verification items):
  1. zero residual == static last-frame baseline (exact);
  2. output finite and L2-normalized (always);
  3. causal inputs only (history, delta) — no future anywhere;
  4. no future candidate features enter the input path;
  5/6. single-step gradients + overfit checked in selftest_j2_2.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ResidualPredictor(nn.Module):
    """GRU(d+1 -> h) residual head; prediction = z_last + residual."""

    def __init__(self, d: int = 384, h: int = 128):
        super().__init__()
        self.gru = nn.GRU(d + 1, h, batch_first=True)
        self.proj = nn.Linear(h, d)

    def forward(self, history: torch.Tensor,
                delta: torch.Tensor) -> torch.Tensor:
        """history (B,T,D) frames <= t ONLY; delta (B,) seconds."""
        ldt = torch.log1p(delta.float())[:, None, None].expand(
            -1, history.shape[1], 1)
        _, hh = self.gru(torch.cat((history, ldt), -1))
        residual = self.proj(hh[-1])
        z_last = history[:, -1, :]
        return F.normalize(z_last + residual, dim=-1)

    @torch.no_grad()
    def zero_residual_output(self, history: torch.Tensor) -> torch.Tensor:
        """Verification hook: prediction with residual forced to zero —
        MUST equal normalize(z_last) exactly (static baseline)."""
        return F.normalize(history[:, -1, :], dim=-1)

    def kill_residual(self) -> None:
        """Zero the residual head (for the baseline-fallback test)."""
        torch.nn.init.zeros_(self.proj.weight)
        torch.nn.init.zeros_(self.proj.bias)
