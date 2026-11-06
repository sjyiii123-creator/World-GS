"""Encoder adapter: reuses the EXISTING frozen DINOv2 pipeline (no reimpl).

Exposes: frozen feature extraction + provenance, so every downstream
module records WHICH encoder produced its evidence.  The adapter wraps
`wmeval.latent_features.LatentFeatureExtractor` — a single
implementation, single cache, single preprocessing identity.
"""
from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np

from ..latent_features import CropConfig, LatentFeatureExtractor


@dataclasses.dataclass
class EncoderProvenance:
    model_id: str
    frame_mapping_version: str
    crop_config: dict
    frozen: bool = True


class EncoderAdapter:
    """Frozen visual encoder + provenance (delegates, never re-implements)."""

    def __init__(self, model_id: str = "ckpt/dinov2-small",
                 crop_cfg: Optional[CropConfig] = None,
                 cache_dir=None, device: str = "cuda") -> None:
        self._ex = LatentFeatureExtractor(model_id=model_id,
                                          crop_cfg=crop_cfg or CropConfig(),
                                          cache_dir=cache_dir, device=device)
        self.provenance = EncoderProvenance(
            model_id=model_id,
            frame_mapping_version=self._ex._cache_meta.get(
                "frame_mapping_version", "unknown"),
            crop_config=dataclasses.asdict(
                crop_cfg or CropConfig()),
            frozen=not any(
                p.requires_grad for p in self._ex.model.parameters()))

    # ---- feature contract -------------------------------------------------
    def frame_feature(self, video_path: str, frame_index: int, bbox,
                      sample: str, ped: str) -> np.ndarray:
        """L2-normalized frozen embedding of one actor crop (M1 mapping)."""
        return self._ex.frame_feature(video_path, frame_index, bbox,
                                      sample, ped)

    def segment_feature(self, video_path: str, window_start: int,
                        frames: list, sample: str, ped: str) -> np.ndarray:
        return self._ex.segment_feature(video_path, window_start, frames,
                                        sample, ped)

    def flush(self) -> None:
        self._ex.flush_cache()

    @property
    def dim(self) -> int:
        return self._ex.dim
