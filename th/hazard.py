"""Phase 2: turn per-step signals into failure forecasts.

Everything here is CPU-only and reads the .npz files written by phase1.py.

Two targets, both built from the pre-registered failure rule (th/labels.py):

* forecast  y_t = 1 if the dream breaks within the next k steps (t+1 .. t+k),
            using only what is known at step t. Rows exist while the dream is
            still alive at t, and only when the label is knowable (a censored
            dream needs t + k <= horizon).
* hazard    y_t = 1 if the dream breaks exactly at step t, given it survived
            until t. Used for survival curves (S(t) = prod (1 - h)).

Features at step t are causal (steps <= t only), label-free, and computed per
signal s in (u_r, u_f, u_s) after dividing by the task's mean latent motion:
    level   log s_t
    recent  mean of log s over the last 4 steps
    trend   recent mean minus the mean of the 4 steps before that
plus log motion ratio and log t.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from th.labels import chance_error, dpsnr_np, error_onsets, onsets, weak_baseline

SIGNALS = ("u_r", "u_f", "u_s")
EPS = 1e-6


def feature_names() -> List[str]:
    names = []
    for s in SIGNALS:
        names += [f"{s}_level", f"{s}_recent", f"{s}_trend"]
    return names + ["motion_level", "log_step"]


def _trailing_mean(x: np.ndarray, w: int, shift: int = 0) -> np.ndarray:
    """Mean of x[t-shift-w+1 .. t-shift] along axis 1 (fewer at the start; NaN-free)."""
    T = x.shape[1]
    out = np.empty_like(x)
    for t in range(T):
        hi = max(0, t - shift)
        lo = max(0, hi - w + 1)
        out[:, t] = x[:, lo:hi + 1].mean(axis=1)
    return out


def window_features(d: Dict[str, np.ndarray]) -> np.ndarray:
    """(n_windows, T, n_features) causal features for one task file."""
    m = max(float(d["motion"].mean()), EPS)
    feats = []
    for s in SIGNALS:
        lv = np.log(np.maximum(d[s] / m, EPS))
        recent = _trailing_mean(lv, 4)
        before = _trailing_mean(lv, 4, shift=4)
        feats += [lv, recent, recent - before]
    feats.append(np.log(np.maximum(d["motion"] / m, EPS)))
    T = d["dpsnr"].shape[1]
    feats.append(np.broadcast_to(np.log(np.arange(1, T + 1)), d["dpsnr"].shape))
    return np.stack(feats, axis=-1).astype(np.float64)


@dataclass
class Split:
    X: np.ndarray            # forecast rows
    y: np.ndarray
    task: np.ndarray         # task name per row
    win: np.ndarray          # global window id per row
    step: np.ndarray         # 1-based step per row
    Xh: np.ndarray           # hazard rows
    yh: np.ndarray
    taskh: np.ndarray
    steph: np.ndarray
    onsets: Dict[str, np.ndarray] = field(default_factory=dict)   # per task, NaN = censored
    horizons: Dict[str, int] = field(default_factory=dict)
    dpsnr: Dict[str, np.ndarray] = field(default_factory=dict)
    feats: Dict[str, np.ndarray] = field(default_factory=dict)
    weak: List[str] = field(default_factory=list)
    chance: Dict[str, float] = field(default_factory=dict)
    Xd: np.ndarray = field(default_factory=lambda: np.empty(0))    # detection rows: every step
    yd: np.ndarray = field(default_factory=lambda: np.empty(0))    # 1 if already broken at t
    win_onset: np.ndarray = field(default_factory=lambda: np.empty(0))   # onset per window id (NaN = censored)
    win_task: List[str] = field(default_factory=list)


def load_split(stage_dir: Path, k: int, weak_tasks: Optional[List[str]] = None, label: str = "dpsnr",
               tau: float = 0.5, chance: Optional[Dict[str, float]] = None) -> Split:
    """Build forecast, hazard and detection rows for every usable task in a stage directory.
    label "dpsnr": the paper's frozen-frame criterion (weak-baseline tasks left out).
    label "error": direct latent error >= tau x the task's chance error (all tasks kept).
    weak_tasks / chance: pass the fit split's values to apply them to another split."""
    stage_dir = Path(stage_dir)
    if stage_dir.name == "unseen" and not (stage_dir / "UNSEALED").exists():
        raise SystemExit("unseen is sealed until Phase 4 (create runs/phase1/unseen/UNSEALED to open it)")
    X, y, task, win, step = [], [], [], [], []
    Xd, yd = [], []
    Xh, yh, taskh, steph = [], [], [], []
    sp = Split(*[np.empty(0)] * 9)
    wid = 0
    win_onset = []
    for f in sorted(stage_dir.glob("*.npz")):
        if f.name.endswith(".tmp.npz"):
            continue
        name = f.stem
        with np.load(f) as z:
            d = {kk: z[kk] for kk in z.files}
        if "mse" in d:
            d["dpsnr"] = dpsnr_np(d["mse"], d["mse_floor"])
        if label == "dpsnr":
            is_weak = (name in weak_tasks) if weak_tasks is not None else weak_baseline(d["dpsnr"])
            if is_weak:
                sp.weak.append(name)
                continue
            on = onsets(d["dpsnr"])
        else:
            c = chance[name] if chance and name in chance else chance_error(d["z_start"])
            sp.chance[name] = c
            on = error_onsets(d["lat_err"], c, tau)
        F = window_features(d)
        n, T, _ = F.shape
        sp.onsets[name], sp.horizons[name], sp.dpsnr[name], sp.feats[name] = on, T, d["dpsnr"], F
        for i in range(n):
            o = on[i]
            for t in range(1, T + 1):
                alive = np.isnan(o) or o > t
                if alive and (not np.isnan(o) or t + k <= T):
                    X.append(F[i, t - 1]); step.append(t); task.append(name); win.append(wid)
                    y.append(int(not np.isnan(o) and t < o <= t + k))
                if np.isnan(o) or t <= o:
                    Xh.append(F[i, t - 1]); steph.append(t); taskh.append(name)
                    yh.append(int(not np.isnan(o) and t == o))
                Xd.append(F[i, t - 1]); yd.append(int(not np.isnan(o) and t >= o))
            win_onset.append(o)
            sp.win_task.append(name)
            wid += 1
    sp.win_onset = np.array(win_onset, dtype=float)
    sp.X, sp.y, sp.task, sp.win, sp.step = (np.array(X), np.array(y), np.array(task), np.array(win), np.array(step))
    sp.Xh, sp.yh, sp.taskh, sp.steph = np.array(Xh), np.array(yh), np.array(taskh), np.array(steph)
    sp.Xd, sp.yd = np.array(Xd), np.array(yd)
    return sp


# ---------------------------------------------------------------- models

def fit_logistic(X: np.ndarray, y: np.ndarray, C: float = 1.0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
    model.fit(X, y)
    return model


def signal_columns(sig: str) -> List[int]:
    return [i for i, n in enumerate(feature_names()) if n.startswith(sig)]


class ThresholdRule:
    """Paper and demo style: one signal, one fixed threshold (Youden's J on the fit split).
    To compare by Brier score, each side of the threshold predicts its fit-split base rate."""

    def __init__(self, col: int):
        self.col = col

    def fit(self, X, y):
        from sklearn.metrics import roc_curve
        fpr, tpr, thr = roc_curve(y, X[:, self.col])
        j = int(np.argmax(tpr - fpr))
        self.thr = float(thr[j])
        flag = X[:, self.col] >= self.thr
        self.p_hi = float(y[flag].mean()) if flag.any() else float(y.mean())
        self.p_lo = float(y[~flag].mean()) if (~flag).any() else float(y.mean())
        return self

    def predict_proba(self, X):
        p = np.where(X[:, self.col] >= self.thr, self.p_hi, self.p_lo)
        return np.stack([1 - p, p], axis=1)


# ---------------------------------------------------------------- metrics

def brier(y, p) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def auroc(y, p) -> Optional[float]:
    from sklearn.metrics import roc_auc_score
    return None if len(np.unique(y)) < 2 else float(roc_auc_score(y, p))


def reliability(y, p, n_bins: int = 10, min_count: int = 50) -> Dict:
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    bins = []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        bins.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "n": int(m.sum()),
                     "mean_pred": float(p[m].mean()), "frac_pos": float(y[m].mean())})
    big = [b for b in bins if b["n"] >= min_count]
    max_gap = max((abs(b["mean_pred"] - b["frac_pos"]) for b in big), default=None)
    return {"bins": bins, "max_gap_bins_ge_min": max_gap, "min_count": min_count}


def warning_times(sp: Split, p: np.ndarray, thr: float = 0.5) -> List[int]:
    """For every dream that breaks after step 1: steps between the forecast first reaching thr
    and the break (0 if it never does before the break)."""
    out = []
    for w, onset in enumerate(sp.win_onset):
        if np.isnan(onset) or onset <= 1:
            continue
        m = sp.win == w
        steps, probs = sp.step[m], p[m]
        hit = steps[(probs >= thr) & (steps < onset)]
        out.append(int(onset - hit.min()) if hit.size else 0)
    return out


def kaplan_meier(on: np.ndarray, horizon: int) -> np.ndarray:
    """S(t) for t = 1..horizon from onsets (NaN = censored at horizon)."""
    S, s = [], 1.0
    for t in range(1, horizon + 1):
        at_risk = np.sum(np.isnan(on) | (on >= t))
        events = np.sum(on == t)
        if at_risk > 0:
            s *= 1 - events / at_risk
        S.append(s)
    return np.array(S)


def median_survival(S: np.ndarray) -> Optional[int]:
    below = np.nonzero(S <= 0.5)[0]
    return int(below[0] + 1) if below.size else None
