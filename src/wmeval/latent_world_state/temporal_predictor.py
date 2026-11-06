"""Temporal predictor: future latent feature from causal history + gap.

Wraps the EXISTING trained GRU checkpoints (j2_predictor.pt /
j2_predictor_D.pt) behind a contract that makes leakage structurally
impossible: forward() accepts ONLY (history, delta_seconds) — the
future/target NEVER appears in the input path.  delta is a PREDICTION
request ("what will this actor look like at t+delta"), one call per
candidate with that candidate's OWN delta (task J2.1 §三).
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclasses.dataclass
class PredictionRequest:
    history: np.ndarray             # (T, D) causal, L2-normalized
    delta_seconds: float            # target time - anchor end time


@dataclasses.dataclass
class PredictionResult:
    predicted_feature: np.ndarray   # (D,) L2-normalized
    delta_seconds: float
    model_id: str
    note: str = ""


class TemporalPredictor:
    """JEPA-inspired latent predictor (frozen-encoder downstream head)."""

    def __init__(self, checkpoint: Path, model_id: str = "GRU-384-128-384",
                 with_motion: bool = False) -> None:
        self.model_id = model_id
        self.with_motion = with_motion
        self.net = self._load(checkpoint)

    def _load(self, checkpoint: Path) -> nn.Module:
        from run_j2_train import ActorPredictor, ActorPredictorD  # noqa
        net = ActorPredictorD() if self.with_motion else ActorPredictor()
        net.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        net.eval()
        return net

    @torch.no_grad()
    def predict(self, request: PredictionRequest,
                motion_state=None) -> PredictionResult:
        x = torch.from_numpy(request.history[None].astype(np.float32))
        d = torch.tensor([float(request.delta_seconds)])
        z = (self.net(x, d, motion_state) if self.with_motion
             else self.net(x, d))
        return PredictionResult(predicted_feature=z.numpy()[0],
                                 delta_seconds=request.delta_seconds,
                                 model_id=self.model_id)

    @staticmethod
    def predict_is_causal(history_frames, anchor_end_frame) -> bool:
        """Guard: every history frame must be <= the anchor end."""
        return all(f <= anchor_end_frame for f in history_frames)
