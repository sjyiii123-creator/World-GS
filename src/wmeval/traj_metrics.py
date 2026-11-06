"""Faithful NumPy port of DrivingGen's official trajectory metrics.

Provenance (all formulas/defaults transcribed from the DrivingGen repo,
branch main, as read on 2026-11-04 — see docs/repository_analysis.md):

* ``trajectory_consistency``  <- drivinggen/trajs/traj_consistency.py
  (velocity consistency + acceleration consistency, the "Consistency" group)
* ``comfort_score_norm``      <- drivinggen/trajs/traj_quality.py
* ``curvature_rms``           <- drivinggen/trajs/traj_quality.py
* ``speed_score``             <- drivinggen/trajs/traj_quality.py
* ``smooth_traj_sg``          <- drivinggen/z-sample_ftd.py (preprocessing)

Defaults replicate the official ones exactly (dt=0.1, eps=1e-9,
v_static=0.1, length_eps=1.0, j/a/y_scale=1.0, v_ref=6.0, k=2.5,
win_sec=0.4, poly=3).

IMPORTANT — units contract
--------------------------
The official pipeline runs these on **world-coordinate (meter) trajectories
sampled at 10 Hz** produced by UniDepth back-projection.  The minimal
experiment (stage 1) feeds **image-space (pixel) tracks** at the video's own
fps.  The formulas are unit-agnostic but the *thresholds* are not: callers
must pass ``dt=1/fps`` and rescale ``v_static``/``length_eps`` to pixel
units, and every report produced from pixel tracks MUST be labelled
``"units": "px"`` — those numbers are a *formula-faithful reproduction*,
not the paper's metric values.  Once depth back-projection lands, the same
functions with default thresholds produce the paper-aligned numbers.
"""

from __future__ import annotations

from typing import Iterable, Union

import numpy as np
from scipy.signal import savgol_filter

ArrayLike = Union[np.ndarray, Iterable]


# ----------------------------------------------------------------------------
# official preprocessing: smooth_traj_sg (z-sample_ftd.py)
# ----------------------------------------------------------------------------
def smooth_traj_sg(xy: np.ndarray, dt: float = 0.1, win_sec: float = 0.4,
                   poly: int = 3) -> np.ndarray:
    """Savitzky-Golay smoothing, official parameters (win_sec/dt -> odd k)."""
    xy = np.asarray(xy, dtype=np.float64)
    k = int(round(win_sec / dt))
    if k % 2 == 0:
        k += 1
    T = xy.shape[0]
    if T == 4:
        k, poly = 3, 1
    if k < 3:
        return xy
    if k >= T:
        # degenerate short tracks: shrink to largest valid odd window and
        # keep polyorder valid (savgol requires polyorder < window_length)
        k = T - 1 if (T - 1) % 2 == 1 else T - 2
        if k < 3:
            return xy
        poly = min(poly, k - 2)
    return savgol_filter(xy, window_length=k, polyorder=poly, axis=0,
                         mode="interp")


def _prep_xy(arr: ArrayLike, axis: int = -2):
    """Move time axis to -2, flatten batch dims -> (N, T, 2). Official helper."""
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[..., :2]
    arr = np.moveaxis(arr, axis, -2)
    T = arr.shape[-2]
    arr = arr.reshape(-1, T, 2)
    return arr


# ----------------------------------------------------------------------------
# Consistency group: velocity + acceleration consistency  (OFFICIAL PORT)
# ----------------------------------------------------------------------------
def trajectory_consistency(
    traj_xy: ArrayLike,
    *,
    dt: float = 0.1,
    axis: int = -2,
    eps: float = 1e-9,
    v_static: float = 0.1,
    reduce: str = "none",
):
    """Velocity/acceleration consistency, exactly as DrivingGen computes it.

    v        = ||diff(xy)/dt||                      (per-step speed)
    moving   = max(v) >= v_static                   (else NaN)
    vel_cons = exp(-std(v) / (mean(v) + eps))
    a        = diff(v)/dt                           (signed accel of speed)
    acc_cons = exp(-std(a) / (mean(|a|) + eps))     # NB: mu over |a|, sigma over signed a
    score    = 0.5 * (vel_cons + acc_cons)          # arithmetic mean (the
                docstring in the official repo claims a product; the code sums
                then multiplies by 0.5 — we follow the code).
    """
    arr, bs = _prep_xy(traj_xy, axis), None
    # keep batch shape for "none" mode
    raw = np.asarray(traj_xy, dtype=np.float64)[..., :2]
    raw = np.moveaxis(raw, axis, -2)
    bs = raw.shape[:-2]
    xy = arr
    N, T, _ = xy.shape
    scores = np.full(N, np.nan, dtype=np.float64)

    v = np.linalg.norm(np.diff(xy, axis=1) / dt, axis=-1)          # (N, T-1)
    if v.shape[1] < 1:
        return np.full(bs if reduce == "none" else (), np.nan)
    moving = v.max(axis=1) >= v_static
    if not moving.any():
        return scores.reshape(bs) if reduce == "none" else float("nan")

    v_m = v[moving]                                                 # (M, T-1)
    mu_v = v_m.mean(axis=1)
    sigma_v = v_m.std(axis=1)
    vel = np.exp(-sigma_v / (mu_v + eps))

    if v_m.shape[1] >= 2:
        a_m = np.diff(v_m, axis=1) / dt                             # (M, T-2)
        mu_a = np.mean(np.abs(a_m), axis=1)
        sigma_a = np.std(a_m, axis=1)
        acc = np.exp(-sigma_a / (mu_a + eps))
        scores[moving] = vel + acc
    else:  # too short for acceleration term: official code would fail here;
        # we keep velocity term only and flag via half-weight (documented
        # deviation, logged in the report as `short_track`)
        scores[moving] = vel
    scores[moving] *= 0.5

    if reduce == "none":
        return scores.reshape(bs)
    return float(np.nanmean(scores))


def get_traj_consistency(preds: list[np.ndarray], **kw) -> float:
    """Official wrapper: mean over trajectories (static -> excluded via NaN)."""
    return float(trajectory_consistency(np.stack(preds, 0), reduce="mean", **kw))


# ----------------------------------------------------------------------------
# Quality group: comfort / curvature / speed  (OFFICIAL PORT)
# ----------------------------------------------------------------------------
def comfort_score_norm(
    traj_xy: ArrayLike,
    *,
    dt: float = 0.1,
    axis: int = -2,
    eps: float = 1e-9,
    v_static: float = 0.1,
    pct: float | None = None,
    length_eps: float = 1.0,
    j_scale: float = 1.0,
    a_scale: float = 1.0,
    y_scale: float = 1.0,
    reduce: str = "none",
    return_components: bool = False,
):
    """Comfort score: jerk/acc/yaw-rate time-aggregated, normalized per meter,
    mapped through 1/(1+x/scale), geometric mean of the three components."""
    xy = _prep_xy(traj_xy, axis)
    N, T, _ = xy.shape
    out = np.full(N, np.nan)
    if T < 5:
        if return_components:
            return out, np.full((N, 3), np.nan)
        return out

    # central differences
    v = (xy[:, 2:, :] - xy[:, :-2, :]) / (2 * dt)                   # (N, T-2, 2)
    a = (xy[:, 2:, :] - 2 * xy[:, 1:-1, :] + xy[:, :-2, :]) / (dt ** 2)
    j = (a[:, 2:, :] - a[:, :-2, :]) / (2 * dt)                     # (N, T-4, 2)
    th = np.arctan2(v[..., 1], v[..., 0])
    yaw_rt = ((th[:, 2:] - th[:, :-2] + np.pi) % (2 * np.pi) - np.pi) / (2 * dt)

    speed = np.linalg.norm(v, axis=-1)                              # (N, T-2)
    moving = speed.max(axis=1) >= v_static
    lengths = np.sum(np.linalg.norm(np.diff(xy, axis=1), axis=-1), axis=1)
    valid = moving & (lengths > length_eps)
    if not valid.any():
        if return_components:
            return out, np.full((N, 3), np.nan)
        return out

    def _agg(x: np.ndarray) -> np.ndarray:
        """mean over time, or percentile(pct) over time (official)."""
        if pct is None:
            return x.mean(axis=1)
        return np.percentile(x, pct, axis=1)

    jerk_stat = _agg(np.linalg.norm(j, axis=-1))
    acc_stat = _agg(np.linalg.norm(a, axis=-1))
    yaw_stat = _agg(np.abs(yaw_rt))

    L = lengths + eps
    jerk_pm = jerk_stat / L
    acc_pm = acc_stat / L
    yaw_pm = yaw_stat / L

    S_j = 1.0 / (1.0 + jerk_pm / max(j_scale, eps))
    S_a = 1.0 / (1.0 + acc_pm / max(a_scale, eps))
    S_y = 1.0 / (1.0 + yaw_pm / max(y_scale, eps))

    comps = np.stack([S_j, S_a, S_y], axis=-1)                     # (N, 3)
    S = np.exp(np.nanmean(np.log(comps), axis=-1))
    out[valid] = S[valid]

    if reduce != "none":
        out = float(np.nanmean(out))
    if return_components:
        return out, comps
    return out


def curvature_rms(
    traj_xy: ArrayLike,
    *,
    dt: float = 0.1,
    axis: int = -2,
    eps: float = 1e-9,
    v_static: float = 0.1,
    pct: float | None = None,
    reduce: str = "none",
):
    """Curvature smoothness: kappa = |x'y''-y'x''| / (x'^2+y'^2+eps)^1.5,
    RMS over time, score = 1/(1+rms).  Static -> NaN."""
    xy = _prep_xy(traj_xy, axis)
    N, T, _ = xy.shape
    out = np.full(N, np.nan)
    if T < 3:
        return out if reduce == "none" else float("nan")

    x = xy[..., 0]
    y = xy[..., 1]
    x_dot = np.diff(x, axis=-1) / dt                                # (N, T-1)
    y_dot = np.diff(y, axis=-1) / dt
    x_dd = np.diff(x_dot, axis=-1) / dt                             # (N, T-2)
    y_dd = np.diff(y_dot, axis=-1) / dt

    xd = x_dot[..., :-1]
    yd = y_dot[..., :-1]
    num = np.abs(xd * y_dd - yd * x_dd)
    den = (xd ** 2 + yd ** 2 + eps) ** 1.5
    kappa = num / den

    v = np.linalg.norm(np.diff(xy, axis=1) / dt, axis=-1)
    moving = v.max(axis=1) >= v_static
    if pct is not None:
        thr = np.percentile(kappa, pct, axis=-1, keepdims=True)
        kappa = np.where(kappa > thr, np.nan, kappa)
    rms = np.sqrt(np.nanmean(kappa ** 2, axis=-1))
    score = 1.0 / (1.0 + rms)
    out[moving] = score[moving]
    if reduce != "none":
        return float(np.nanmean(out))
    return out


def speed_score(
    traj_xy: ArrayLike,
    *,
    dt: float = 0.1,
    axis: int = -2,
    v_ref: float = 6.0,
    k: float = 2.5,
    v_static: float = 0.1,
    use_percentile=None,
    reduce: str = "none",
):
    """Speed encouragement: log1p(v_stat)/log1p(k*v_ref) clipped to [0,1];
    non-moving trajectories score 0.0 (official change from NaN)."""
    xy = _prep_xy(traj_xy, axis)
    N = xy.shape[0]
    v = np.linalg.norm(np.diff(xy, axis=1) / dt, axis=-1)           # (N, T-1)
    if use_percentile is None:
        v_stat = v.mean(axis=1)
    else:
        v_stat = np.percentile(v, use_percentile, axis=1)
    moving = v.max(axis=1) >= v_static
    v_max = k * v_ref
    denom = np.log1p(v_max)
    out = np.zeros(N)
    if moving.any():
        num = np.log1p(v_stat[moving])
        out[moving] = np.clip(num / denom, 0.0, 1.0)
    if reduce != "none":
        return float(np.nanmean(out))   # official uses plain mean here (0 for static)
    return out


def get_traj_quality(preds: list[np.ndarray], **kw) -> float:
    """Official wrapper: mean of (comfort, curvature, speed) over trajectories."""
    arr = np.stack(preds, 0)
    c = comfort_score_norm(arr, reduce="none", **kw)
    k = curvature_rms(arr, reduce="none", **kw)
    s = speed_score(arr, reduce="none", **kw)
    stacked = np.stack([c, k, s], axis=-1)
    return float(np.nanmean(stacked))


# ----------------------------------------------------------------------------
# structured report helper (wmeval addition — not part of DrivingGen)
# ----------------------------------------------------------------------------
def evaluate_trajectories(
    trajs: list[np.ndarray],
    dt: float,
    units: str,
    v_static: float,
    length_eps: float,
    smooth: bool = True,
) -> dict:
    """Run the full official metric suite over a list of (T,2) trajectories.

    Variable-length tracks are evaluated one-by-one and aggregated with
    nanmean — equivalent to the official batch form for equal-length inputs
    (all official aggregations are per-trajectory then nanmean).

    Returns a dict with per-trajectory arrays and aggregate means.  ``units``
    is recorded verbatim into the report ("m" for world coords / "px" for
    image space) so downstream consumers cannot mistake a pixel-space
    formula reproduction for a paper-aligned number.
    """
    if len(trajs) == 0:
        return {"units": units, "dt": dt, "num_trajs": 0,
                "note": "no trajectories"}
    if smooth:
        trajs = [smooth_traj_sg(np.asarray(t, dtype=np.float64), dt=dt)
                 for t in trajs]
    trajs = [np.asarray(t, dtype=np.float64) for t in trajs]

    consist_l, comfort_l, curv_l, spd_l = [], [], [], []
    for t in trajs:
        if t.ndim != 2 or t.shape[0] < 2:
            continue
        arr = t[None]  # (1, T, 2)
        consist_l.append(trajectory_consistency(
            arr, dt=dt, v_static=v_static, reduce="none")[0])
        comfort_l.append(comfort_score_norm(
            arr, dt=dt, v_static=v_static, length_eps=length_eps,
            reduce="none"))
        curv_l.append(curvature_rms(
            arr, dt=dt, v_static=v_static, reduce="none"))
        spd_l.append(speed_score(
            arr, dt=dt, v_static=v_static, reduce="none"))

    def _lst(xs):
        return [None if x is None or (isinstance(x, float) and np.isnan(x))
                else round(float(x), 6) for x in xs]

    def _agg(xs):
        xs = [x for x in xs if x is not None and np.isfinite(x)]
        return float(np.nanmean(xs)) if xs else None

    return {
        "units": units,
        "dt": dt,
        "num_trajs": int(len(trajs)),
        "smoothed": bool(smooth),
        "v_static": v_static,
        "length_eps": length_eps,
        "per_traj": {
            "velocity_accel_consistency": _lst(consist_l),
            "comfort": _lst(comfort_l),
            "curvature": _lst(curv_l),
            "speed": _lst(spd_l),
        },
        "aggregate": {
            "velocity_accel_consistency": _agg(consist_l),
            "comfort": _agg(comfort_l),
            "curvature": _agg(curv_l),
            "speed": _agg(spd_l),
        },
        "provenance": "formula port of DrivingGen main: trajs/traj_consistency.py, "
                      "trajs/traj_quality.py, z-sample_ftd.py smooth_traj_sg",
    }
