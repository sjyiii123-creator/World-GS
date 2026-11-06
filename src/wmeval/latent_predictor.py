"""J2 (design-stage): lightweight latent temporal predictor — INTERFACE.

Form:  z_future = Predictor(z_history, delta_time)

  * encoder stays FROZEN (DINOv2); this head only operates in latent
    space — it NEVER predicts RGB pixels;
  * GRU (default) over per-frame L2-normalized embeddings + a
    delta-time embedding; output re-normalized to the unit sphere
    (cosine supervision);
  * NO future leakage: inference uses only embeddings at frames
    <= t; the training loop may see future embeddings as TARGETS only
    (enforced by API shape: ``forward(history)`` never receives the
    future);
  * prediction error is EVIDENCE, never a verdict: callers must not
    mark events Invalid from prediction error alone (module docstring
    is the contract; see docs/jepa_baseline_plan.md J3).

Runnable now (forward + loss are fully functional and tested);
training is intentionally NOT started this round (decision gate:
J1 results must first show appearance separability worth predicting).
"""
from __future__ import annotations

import dataclasses

import numpy as np
import torch
import torch.nn as nn


@dataclasses.dataclass
class PredictorConfig:
    input_dim: int = 384            # dinov2-small CLS dim
    hidden_dim: int = 256
    n_layers: int = 1
    delta_bins: int = 8             # log-spaced gap buckets (delta_time)
    dropout: float = 0.1


class LatentPredictor(nn.Module):
    """GRU head mapping a history of frozen embeddings to a future one."""

    def __init__(self, cfg: PredictorConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or PredictorConfig()
        self.delta_embed = nn.Embedding(self.cfg.delta_bins, self.cfg.hidden_dim)
        self.gru = nn.GRU(self.cfg.input_dim, self.cfg.hidden_dim,
                          num_layers=self.cfg.n_layers,
                          batch_first=True,
                          dropout=self.cfg.dropout if self.cfg.n_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.hidden_dim, self.cfg.input_dim))

    @staticmethod
    def delta_bucket(delta_seconds: float) -> int:
        """Map a real time gap (seconds) to a log bucket id."""
        if delta_seconds <= 0:
            return 0
        b = int(np.ceil(np.log2(max(delta_seconds, 1e-3) / 0.25) + 1))
        return int(np.clip(b, 0, 7))

    def forward(self, history: torch.Tensor,
                delta_seconds: float) -> torch.Tensor:
        """history: (B, T, D) embeddings at frames <= t ONLY.
        Returns (B, D) L2-normalized predicted embedding at t+delta."""
        # normalize inputs (defensive: caller may pass unnormalized)
        h = torch.nn.functional.normalize(history, dim=-1)
        out, _ = self.gru(h)
        state = out[:, -1]                              # (B, hidden)
        d = self.delta_embed(torch.tensor(
            self.delta_bucket(delta_seconds), device=h.device))
        z = self.head(state + d)
        return torch.nn.functional.normalize(z, dim=-1)

    @staticmethod
    def loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Cosine loss against the REAL future embedding (train-time
        supervision only; targets never enter forward())."""
        cos = (pred * torch.nn.functional.normalize(target, dim=-1)).sum(-1)
        return (1.0 - cos).mean()


def assemble_history(embeddings: list[np.ndarray]) -> torch.Tensor:
    """Stack per-frame embeddings -> (1, T, D) tensor for inference."""
    return torch.from_numpy(np.stack(embeddings)).float().unsqueeze(0)
