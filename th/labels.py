"""Failure labels for dreams. Fixed before any Phase 1 data was collected.

Rule (decided 2026-11-23 from the Phase 0 smoke test, where single-step dips of
dPSNR below 0 were mostly noise; median chosen over mean after a unit test showed one
extreme frame could pull a 5-step mean below 0):

* smooth  : trailing MEDIAN of dPSNR over the last 5 steps (fewer at the start), so a
            single outlier frame, however extreme, cannot trigger a failure
* failure : the smoothed dPSNR is <= 0 for 5 steps in a row
* onset   : the first step of that 5-step run (1-based step number)
* censored: no failure within the horizon; we only know the dream survived
            at least `horizon` steps (survival analysis handles this)

Weak-baseline flag (per task): if, over its first 8 steps, the dream beats the
frozen-frame baseline by less than 1 dB (median across windows), dPSNR cannot
separate good dreams from bad ones for this task (Phase 0: walker-walk).
Such tasks are reported but excluded from the failure-time models.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

SMOOTH = 5
SUSTAIN = 5
WEAK_EARLY_STEPS = 8
WEAK_MIN_DB = 1.0


def smooth_dpsnr(dpsnr: np.ndarray, window: int = SMOOTH) -> np.ndarray:
    """Causal trailing median along the last axis."""
    x = np.asarray(dpsnr, dtype=np.float64)
    out = np.empty_like(x)
    for t in range(x.shape[-1]):
        out[..., t] = np.median(x[..., max(0, t - window + 1): t + 1], axis=-1)
    return out


def failure_onset(dpsnr_row: np.ndarray, smooth: int = SMOOTH, sustain: int = SUSTAIN) -> Optional[int]:
    """1-based step at which the dream breaks, or None if it never does."""
    bad = smooth_dpsnr(dpsnr_row, smooth) <= 0
    run = 0
    for t, b in enumerate(bad):
        run = run + 1 if b else 0
        if run == sustain:
            return t - sustain + 2   # first step of the run, 1-based
    return None


def onsets(dpsnr: np.ndarray, **kw) -> np.ndarray:
    """(n_windows,) float array of onset steps, NaN where censored."""
    return np.array([np.nan if (o := failure_onset(r, **kw)) is None else o for r in dpsnr], dtype=float)


def weak_baseline(dpsnr: np.ndarray, early: int = WEAK_EARLY_STEPS, min_db: float = WEAK_MIN_DB) -> bool:
    return bool(np.median(np.asarray(dpsnr)[:, :early].mean(axis=1)) < min_db)
