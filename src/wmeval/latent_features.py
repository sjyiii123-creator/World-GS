"""J1: frozen DINOv2 actor-crop feature extraction with offline cache.

Design decisions (task §四 checklist addressed explicitly):
  * small-object crops  : actor bboxes can be tiny; crops are padded by
                          ``context_pad`` (fraction of bbox size) and
                          clamped to a ``min_crop`` pixel size;
  * crop context        : context_pad=0.30 adds surrounding pixels so the
                          encoder sees torso+scene, not a 40px blob;
  * resolution change   : all crops resized to ``crop_size`` square
                          (aspect ignored BY DESIGN — pedestrian crops are
                          near-vertical; recorded in cache meta);
  * blur / occlusion    : segment features average multiple sampled
                          frames; per-frame JAAD occlusion flag stored
                          alongside; occluded frames excluded when
                          ``skip_occluded`` and alternates exist;
  * normalization       : ImageNet mean/std via HF image processor, then
                          L2-normalized output embeddings;
  * separability check  : ``separability_report`` returns same/diff
                          cosine distributions (run on labelled pairs).

Cache: single .npz keyed by "sample|ped|frame|bboxhash" + meta.json
(model id, revision, preprocessing params) — reruns are no-ops.

Model: facebook/dinov2-small (ViT-S/14, 22M) — FROZEN, eval mode.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import torch


@dataclasses.dataclass
class CropConfig:
    context_pad: float = 0.30       # fraction of bbox w/h added around
    min_crop_px: int = 48           # clamp tiny boxes up (keep center)
    crop_size: int = 224            # resize target (square)
    frames_per_segment: int = 3     # first/middle/last sampled
    # ---- J1.1 occlusion sampling strategy (task §五) --------------------
    # JAAD occluded semantics (official README): 0 = visible,
    # 1 = partially occluded (>25%), 2 = heavily occluded (>75%).
    # "all"            : use every frame regardless of occlusion
    # "prefer_visible" : drop heavily-occluded frames when alternates
    #                    exist; partially-occluded frames stay eligible
    #                    but visible frames are preferred in sampling
    # "weighted"       : aggregate frame features weighted by visibility
    #                    quality (w=1.0/0.7/0.3 for occ 0/1/2)
    occlusion_strategy: str = "prefer_visible"
    visibility_weights: tuple = (1.0, 0.7, 0.3)  # occ 0/1/2

    def visibility_class(self, occ: int) -> str:
        return {0: "visible", 1: "partially_occluded",
                2: "heavily_occluded"}.get(occ, "unknown")


DEFAULT_MODEL = "facebook/dinov2-small"


def select_frames(frames: list, strategy: str, k: int) -> list:
    """Pure frame-selection logic per occlusion strategy (unit-testable).

    frames: [[xml_frame, bbox, occ], ...]
    Semantics:
      all            — every frame, time-ordered;
      prefer_visible — sample on the highest-visibility tier that has
                       >= k frames (occ==0 first, then occ<=1, then all);
                       nothing is deleted outright when a segment is
                       fully occluded (falls back to the whole segment).
                       (J1.1 fix: the earlier `occ < 2` filter kept ALL
                       occ==1 frames, making B identical to A on data
                       with zero occ==2 frames — measured 294/500 pairs
                       carry occ==1, so the tier priority now matters.)
      weighted       — every frame (weights applied at aggregation).
    Then uniform-in-time sampling of k frames on the selected pool.
    """
    if strategy == "prefer_visible":
        use = None
        for max_occ in (0, 1, 2):
            cand = [f for f in frames if f[2] <= max_occ]
            if len(cand) >= k:
                use = cand
                break
        use = use if use is not None else frames
    else:  # all / weighted
        use = frames
    if len(use) > k:
        idx = np.linspace(0, len(use) - 1, k).astype(int)
        use = [use[i] for i in idx]
    return use


class LatentFeatureExtractor:
    """Frozen DINOv2 features for actor crops, with on-disk cache."""

    def __init__(self, model_id: str = DEFAULT_MODEL,
                 device: str = "cuda", crop_cfg: Optional[CropConfig] = None,
                 cache_dir: Optional[Path] = None) -> None:
        from transformers import AutoModel, AutoImageProcessor
        self.model_id = model_id
        self.device = device if torch.cuda.is_available() else "cpu"
        self.cfg = crop_cfg or CropConfig()
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._cache: dict = {}
        self._cache_meta = {"model_id": model_id, "device": self.device,
                            "crop": dataclasses.asdict(self.cfg),
                            # J1.1 cache protection: mapping version is part
                            # of cache identity — a future mapping fix
                            # invalidates old caches via meta mismatch
                            # (docs/j1_1_frame_alignment.md).
                            "frame_mapping_version":
                                "M1_xml_minus_window_start_verified_20261105"}
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            meta_p = self.cache_dir / "meta.json"
            if meta_p.exists():
                prev = json.loads(meta_p.read_text())
                if not self._meta_compatible(prev):
                    raise RuntimeError(
                        f"cache mismatch: {prev} vs {self._cache_meta} "
                        "— use a new cache_dir (J1.1: mapping/strategy/"
                        "preprocessing changes must invalidate caches)")
                # legacy cache accepted: mark provenance at runtime
                self._cache_meta["legacy_cache_accepted"] = prev
            else:
                meta_p.write_text(json.dumps(self._cache_meta, indent=2))
            npz = self.cache_dir / "features.npz"
            if npz.exists():
                z = np.load(npz)
                self._cache = {k: z[k] for k in z.files}
        self.dim = self.model.config.hidden_size
        self._frame_caps: dict = {}

    def _meta_compatible(self, prev: dict) -> bool:
        """Cache compatibility with J1 legacy caches (J1.1 §六).

        JSON round-trips tuples to lists — normalize before comparing.
        A legacy cache (pre-J1.1) is accepted ONLY when:
          * model/device identical;
          * crop fields shared by both versions identical;
          * legacy skip_occluded=True maps to the current
            occlusion_strategy='prefer_visible' (behaviour-equivalent);
          * frame mapping was verified unchanged (M1) — legacy caches
            carry no mapping field, acceptance is recorded in
            legacy_cache_accepted for auditability.
        Any real change (strategy, crop params, model) still invalidates.
        """
        if prev.get("model_id") != self.model_id:
            return False
        def _norm(x):
            if isinstance(x, tuple):
                return [_norm(v) for v in x]
            if isinstance(x, list):
                return [_norm(v) for v in x]
            if isinstance(x, dict):
                return {k: _norm(v) for k, v in x.items()}
            return x
        legacy_crop = _norm(dict(prev.get("crop", {})))
        cur = _norm(dict(self._cache_meta["crop"]))
        legacy_skip = legacy_crop.pop("skip_occluded", None)
        # compare only fields shared by both generations; the new
        # identity fields are checked separately below
        for f in ("occlusion_strategy", "visibility_weights"):
            legacy_crop.pop(f, None)
            cur.pop(f, None)
        if legacy_crop != cur:
            return False
        if legacy_skip is not None:
            if legacy_skip is True and \
                    self.cfg.occlusion_strategy != "prefer_visible":
                return False
            if legacy_skip is False and \
                    self.cfg.occlusion_strategy != "all":
                return False
        # non-legacy caches must match on the new identity fields
        if "occlusion_strategy" in prev.get("crop", {}) or \
                "frame_mapping_version" in prev:
            return _norm(prev) == _norm(self._cache_meta)
        return True

    # ---------------- cropping -----------------------------------------
    def _crop(self, frame: np.ndarray, bbox) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        pw, ph = w * self.cfg.context_pad, h * self.cfg.context_pad
        x1, y1 = x1 - pw, y1 - ph
        x2, y2 = x2 + pw, y2 + ph
        # clamp tiny boxes to min crop (keep center)
        if x2 - x1 < self.cfg.min_crop_px:
            c = (x1 + x2) / 2
            x1, x2 = c - self.cfg.min_crop_px / 2, c + self.cfg.min_crop_px / 2
        if y2 - y1 < self.cfg.min_crop_px:
            c = (y1 + y2) / 2
            y1, y2 = c - self.cfg.min_crop_px / 2, c + self.cfg.min_crop_px / 2
        H, W = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        crop = frame[int(y1):int(y2), int(x1):int(x2)]
        return cv2.resize(crop, (self.cfg.crop_size, self.cfg.crop_size),
                          interpolation=cv2.INTER_AREA)

    def _video(self, path: str) -> cv2.VideoCapture:
        if path not in self._frame_caps:
            self._frame_caps[path] = cv2.VideoCapture(path)
        return self._frame_caps[path]

    def _read_frame(self, video_path: str, frame_idx: int) -> np.ndarray:
        cap = self._video(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:  # fall back to sequential scan
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            for _ in range(frame_idx + 1):
                ok, frame = cap.read()
                if not ok:
                    break
        if not ok or frame is None:
            raise RuntimeError(f"frame read failed {video_path}@{frame_idx}")
        return frame

    # ---------------- features -----------------------------------------
    @staticmethod
    def _key(sample: str, ped: str, frame: int, bbox) -> str:
        h = hashlib.md5(f"{sample}|{ped}|{frame}|{tuple(round(float(v), 1) for v in bbox)}".encode()).hexdigest()[:16]
        return h

    @torch.no_grad()
    def frame_feature(self, video_path: str, frame_idx: int, bbox,
                      sample: str, ped: str) -> np.ndarray:
        k = self._key(sample, ped, frame_idx, bbox)
        if k in self._cache:
            return self._cache[k]
        crop = self._crop(self._read_frame(video_path, frame_idx), bbox)
        inputs = self.processor(images=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
                                return_tensors="pt").to(self.device)
        out = self.model(**inputs)
        emb = out.last_hidden_state[:, 0]        # CLS token
        emb = torch.nn.functional.normalize(emb, dim=-1)
        v = emb.squeeze(0).float().cpu().numpy()
        self._cache[k] = v
        return v

    def segment_feature(self, video_path: str, window_start: int,
                        frames: list, sample: str, ped: str) -> np.ndarray:
        """Segment feature under the configured occlusion strategy.

        frames: [[xml_frame, bbox, occ], ...] — pixel indices are
        XML-frame minus window_start in the SynPeDAS clip (M1 mapping,
        verified per-sample in docs/j1_1_frame_alignment.md).
        Frame selection: wmeval.latent_features.select_frames (pure,
        unit-tested). Aggregation: mean (all/prefer_visible) or
        visibility-weighted mean (weighted, 1.0/0.7/0.3 for occ 0/1/2).
        """
        strat = self.cfg.occlusion_strategy
        use = select_frames(frames, strat, self.cfg.frames_per_segment)
        feats = [self.frame_feature(video_path, f[0] - window_start, f[1],
                                    sample, ped) for f in use]
        if strat == "weighted":
            w = np.array([self.cfg.visibility_weights[min(f[2], 2)]
                          for f in use], dtype=float)
            w = w / w.sum() if w.sum() > 0 else np.full(len(w), 1 / len(w))
            v = np.einsum("i,ij->j", w, np.stack(feats))
        else:
            v = np.mean(feats, axis=0)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def flush_cache(self) -> None:
        if not self.cache_dir or not self._cache:
            return
        npz = self.cache_dir / "features.npz"
        old = {}
        if npz.exists():
            z = np.load(npz)
            old = {k: z[k] for k in z.files}
        old.update(self._cache)
        np.savez_compressed(npz, **old)
        self._cache = {}

    # ---------------- diagnostics --------------------------------------
    @torch.no_grad()
    def benchmark(self, video_path: str, frame_idx: int, bbox,
                  n: int = 32) -> dict:
        crop = self._crop(self._read_frame(video_path, frame_idx), bbox)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        batch = [rgb] * n
        t0 = time.perf_counter()
        inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
        out = self.model(**inputs)
        dt = time.perf_counter() - t0
        return {"batch": n, "seconds": round(dt, 3),
                "ms_per_frame": round(dt / n * 1000, 1),
                "device": self.device,
                "vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)
                if self.device == "cuda" else 0.0}
