"""Four further forecast tests (PROTOCOL item 23). CPU only, on the Phase 1 data.

    python forecast_tests.py                     # primary label: latent error >= 0.5 x chance
    python forecast_tests.py --label error --tau 1
    python forecast_tests.py --label dpsnr       # the paper's frozen-frame label

Fit on runs/phase1/fit, evaluate on runs/phase1/test; unseen reported as secondary.
Output: runs/forecast_tests/<tag>/report.json

T1  forecast window sweep, k = 1, 2, 4, 8, 16
T2  signals aligned at the break, against matched unbroken controls, bootstrap intervals
T3  per-dream survival from landmark steps (C-index and calibration)
T4  gradient-boosted trees with recent raw levels, against the logistic model, k = 8

Every forecast AUROC is reported three ways: plain, beyond time (within dream-step groups) and
within task and step group. Cells need >= MIN_CELL rows with both outcomes present; the
average is weighted by rows.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from th.hazard import SIGNALS, feature_names, fit_logistic, load_split, reliability  # noqa: E402

STEP_GROUPS = [(1, 8), (9, 16), (17, 32), (33, 48), (49, 64), (65, 96)]
MIN_CELL_TIME = 50       # same rule as phase4.py
MIN_CELL_TASK = 20
KS = (1, 2, 4, 8, 16)
LANDMARKS = (1, 8, 24)
HORIZONS_U = (4, 8, 16, 32)
OFFSETS = list(range(-10, 5))
N_BOOT = 2000
LAGS = 8


def _r(x, n=3):
    return None if x is None else round(float(x), n)


# ---------------------------------------------------------------- AUROC, three views

def _auc(y, p):
    from sklearn.metrics import roc_auc_score
    return None if len(np.unique(y)) < 2 else float(roc_auc_score(y, p))


def _weighted(cells):
    from sklearn.metrics import roc_auc_score
    vals, w = [], []
    for y, p, min_n in cells:
        if len(y) >= min_n and 0 < y.mean() < 1:
            vals.append(roc_auc_score(y, p)); w.append(len(y))
    return None if not vals else float(np.average(vals, weights=w))


def three_aurocs(y, p, steps, tasks):
    groups = [((steps >= lo) & (steps <= hi)) for lo, hi in STEP_GROUPS]
    by_time = _weighted([(y[g], p[g], MIN_CELL_TIME) for g in groups])
    cells = []
    for t in np.unique(tasks):
        mt = tasks == t
        cells += [(y[mt & g], p[mt & g], MIN_CELL_TASK) for g in groups]
    return {"auroc": _r(_auc(y, p)), "beyond_time": _r(by_time), "within_task_and_time": _r(_weighted(cells))}


# ---------------------------------------------------------------- loading

def load_three(args, k):
    p1 = Path(args.phase1)
    tr = load_split(p1 / "fit", k, label=args.label, tau=args.tau)
    te = load_split(p1 / "test", k, weak_tasks=tr.weak, label=args.label, tau=args.tau, chance=tr.chance)
    un = None
    if (p1 / "unseen" / "UNSEALED").exists() and not args.no_unseen:
        # same as phase4.py: unseen tasks get their own weak-baseline check and chance error
        un = load_split(p1 / "unseen", k, label=args.label, tau=args.tau)
    return tr, te, un


# ---------------------------------------------------------------- T1

def t1_sweep(args):
    out = {}
    for k in KS:
        tr, te, un = load_three(args, k)
        row = {}
        if len(set(tr.y)) < 2:
            out[k] = {"note": "no positives in fit"}; continue
        m = fit_logistic(tr.X, tr.y)
        for name, sp in (("test", te), ("unseen", un)):
            if sp is None or not len(sp.y):
                continue
            p = m.predict_proba(sp.X)[:, 1]
            row[name] = {"positive_rate": _r(sp.y.mean(), 4), "n_rows": int(len(sp.y)),
                         **three_aurocs(sp.y, p, sp.step, sp.task)}
        out[k] = row
    return out


# ---------------------------------------------------------------- T2

def _aligned_diffs(sp):
    """Per breaking dream: (n_offsets, n_signals) of z-scored log level minus matched controls."""
    names = feature_names()
    cols = [names.index(f"{s}_level") for s in SIGNALS]
    rows = []
    for task, F in sp.feats.items():
        lv = F[..., cols]                                       # (n, T, S) log level / task motion
        mu, sd = lv.reshape(-1, len(cols)).mean(0), lv.reshape(-1, len(cols)).std(0) + 1e-9
        z = (lv - mu) / sd
        on = sp.onsets[task]
        n, T, _ = z.shape
        for i in range(n):
            o = on[i]
            if np.isnan(o) or o <= 12:
                continue
            o = int(o)
            d = np.full((len(OFFSETS), len(cols)), np.nan)
            for j, off in enumerate(OFFSETS):
                t = o + off
                if not 1 <= t <= T:
                    continue
                ctrl = [c for c in range(n) if c != i and (np.isnan(on[c]) or on[c] > t)]
                if not ctrl:
                    continue
                d[j] = z[i, t - 1] - z[ctrl, t - 1].mean(0)
            rows.append(d)
    return np.array(rows)                                       # (n_dreams, n_offsets, S)


def t2_aligned(sp):
    D = _aligned_diffs(sp)
    if not len(D):
        return {"n_dreams": 0}
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(D), size=(N_BOOT, len(D)))
    boots = np.stack([np.nanmean(D[b], axis=0) for b in idx])   # (N_BOOT, offsets, S)
    mean = np.nanmean(D, axis=0)
    lo, hi = np.nanpercentile(boots, 2.5, axis=0), np.nanpercentile(boots, 97.5, axis=0)
    out = {"n_dreams": int(len(D)), "offsets": OFFSETS, "signals": {}}
    for s_i, s in enumerate(SIGNALS):
        excl = [(off, bool(lo[j, s_i] > 0 or hi[j, s_i] < 0)) for j, off in enumerate(OFFSETS)]
        lead = [off for off, e in excl if e and off <= -2]
        out["signals"][s] = {"mean": np.round(mean[:, s_i], 3).tolist(), "ci_lo": np.round(lo[:, s_i], 3).tolist(),
                             "ci_hi": np.round(hi[:, s_i], 3).tolist(),
                             "offsets_ci_excludes_0_at_or_before_-2": lead, "lead_claimed": bool(lead)}
    return out


# ---------------------------------------------------------------- T3

def _landmark_rows(sp, s, u):
    """Dreams unbroken at s; y = 1 if still unbroken at s + u (unknown if censored before s + u)."""
    X, y, rem, ev, task = [], [], [], [], []
    for t_name, F in sp.feats.items():
        on, T = sp.onsets[t_name], sp.horizons[t_name]
        if s > T:
            continue
        for i in range(len(on)):
            o = on[i]
            if not np.isnan(o) and o <= s:
                continue
            broke = not np.isnan(o)
            known = broke or s + u <= T
            X.append(F[i, s - 1]); task.append(t_name)
            y.append(np.nan if not known else float((not broke) or o > s + u))
            rem.append((o - s) if broke else (T - s)); ev.append(broke)
    return np.array(X), np.array(y), np.array(rem, float), np.array(ev, bool), np.array(task)


def c_index(risk, time, event, groups=None):
    """Harrell's C. A pair counts when the shorter time is an event; higher risk should break sooner."""
    conc = tot = 0.0
    n = len(risk)
    for i in range(n):
        if not event[i]:
            continue
        for j in range(n):
            if i == j or time[j] <= time[i] or (groups is not None and groups[i] != groups[j]):
                continue
            tot += 1
            conc += 1.0 if risk[i] > risk[j] else 0.5 if risk[i] == risk[j] else 0.0
    return None if tot == 0 else conc / tot


def t3_landmark(tr, others):
    out = {}
    for s in LANDMARKS:
        models = {}
        for u in HORIZONS_U:
            X, y, *_ = _landmark_rows(tr, s, u)
            k = ~np.isnan(y)
            if k.sum() and len(set(y[k])) == 2:
                models[u] = fit_logistic(X[k], y[k].astype(int))
        row = {"fit_dreams_at_risk": int(len(_landmark_rows(tr, s, 16)[0]))}
        for name, sp in others:
            if sp is None:
                continue
            X, y16, rem, ev, task = _landmark_rows(sp, s, 16)
            if not len(X) or 16 not in models:
                continue
            p16 = models[16].predict_proba(X)[:, 1]
            risk = 1 - p16
            k = ~np.isnan(y16)
            cal = reliability(y16[k], p16[k], n_bins=5, min_count=20) if k.any() else None
            per_u = {}
            for u, m in models.items():
                Xu, yu, *_ = _landmark_rows(sp, s, u)
                ku = ~np.isnan(yu)
                per_u[u] = {"n": int(ku.sum()), "observed_unbroken": _r(yu[ku].mean()),
                            "mean_predicted": _r(m.predict_proba(Xu[ku])[:, 1].mean()) if ku.any() else None,
                            "auroc": _r(_auc(yu[ku], m.predict_proba(Xu[ku])[:, 1])) if ku.any() else None}
            row[name] = {"dreams_at_risk": int(len(X)), "events": int(ev.sum()),
                         "c_index": _r(c_index(risk, rem, ev)),
                         "c_index_within_task": _r(c_index(risk, rem, ev, task)),
                         "u16_mean_predicted": _r(p16[k].mean()) if k.any() else None,
                         "u16_observed": _r(y16[k].mean()) if k.any() else None,
                         "u16_max_calibration_gap": _r(cal["max_gap_bins_ge_min"]) if cal else None,
                         "per_u": per_u}
        out[s] = row
    return out


# ---------------------------------------------------------------- T4

def lag_features(sp):
    """Each signal's log level at steps t, t-1, ..., t-7 for every forecast row (repeat step 1 at the start)."""
    names = feature_names()
    cols = [names.index(f"{s}_level") for s in SIGNALS]
    first = {}
    for w, t in enumerate(sp.win_task):
        first.setdefault(t, w)
    out = np.empty((len(sp.y), LAGS * len(cols)))
    for r in range(len(sp.y)):
        t_name, t = sp.task[r], sp.step[r]
        i = sp.win[r] - first[t_name]
        F = sp.feats[t_name][i]
        steps = np.maximum(np.arange(t, t - LAGS, -1), 1) - 1
        out[r] = F[np.ix_(steps, cols)].T.reshape(-1)
    return out


def t4_trees(args):
    from sklearn.ensemble import HistGradientBoostingClassifier
    tr, te, un = load_three(args, 8)
    Ltr = lag_features(tr)
    Xtr_ext = np.hstack([tr.X, Ltr])
    models = {
        "logistic (Phase 2 features)": (fit_logistic(tr.X, tr.y), False),
        "logistic (+ last 8 levels)": (fit_logistic(Xtr_ext, tr.y), True),
        "boosted trees (+ last 8 levels)": (HistGradientBoostingClassifier(random_state=0).fit(Xtr_ext, tr.y), True),
    }
    out = {}
    for name, sp in (("test", te), ("unseen", un)):
        if sp is None or not len(sp.y):
            continue
        Xe = np.hstack([sp.X, lag_features(sp)])
        out[name] = {}
        for mname, (m, ext) in models.items():
            p = m.predict_proba(Xe if ext else sp.X)[:, 1]
            out[name][mname] = {"brier": _r(np.mean((p - sp.y) ** 2), 5), **three_aurocs(sp.y, p, sp.step, sp.task)}
    return out


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1", default=str(ROOT / "runs" / "phase1"))
    ap.add_argument("--label", choices=["dpsnr", "error"], default="error")
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-unseen", action="store_true")
    args = ap.parse_args()
    tag = "dpsnr" if args.label == "dpsnr" else f"error_tau{args.tau:g}"
    out = Path(args.out) if args.out else ROOT / "runs" / "forecast_tests" / tag
    out.mkdir(parents=True, exist_ok=True)

    tr, te, un = load_three(args, 8)
    rep = {"label": tag, "fit_tasks": len(tr.onsets), "test_tasks": len(te.onsets),
           "unseen_tasks": None if un is None else len(un.onsets), "weak_tasks_left_out": tr.weak,
           "min_rows_per_cell": {"beyond_time": MIN_CELL_TIME, "within_task_and_time": MIN_CELL_TASK}}
    print(f"label {tag}: fit {len(tr.win_onset)} dreams, test {len(te.win_onset)}, "
          f"unseen {None if un is None else len(un.win_onset)}")

    rep["T1_window_sweep"] = t1_sweep(args)
    print("\nT1  breaks within k steps (test): AUROC / beyond time / within task and time")
    for k, r in rep["T1_window_sweep"].items():
        t = r.get("test", {})
        print(f"  k={k:<3} pos {t.get('positive_rate')}  {t.get('auroc')} / {t.get('beyond_time')} / "
              f"{t.get('within_task_and_time')}")

    rep["T2_aligned"] = {"test": t2_aligned(te), "fit (more dreams, no model involved)": t2_aligned(tr)}
    if un is not None:
        rep["T2_aligned"]["unseen"] = t2_aligned(un)
    print("\nT2  signal vs matched controls around the break (z units); lead claimed?")
    for split, r in rep["T2_aligned"].items():
        if not r.get("n_dreams"):
            print(f"  {split}: no dreams break after step 12"); continue
        j2 = OFFSETS.index(-2)
        print(f"  {split} ({r['n_dreams']} dreams)")
        for s, v in r["signals"].items():
            print(f"    {s}: at -5 {v['mean'][OFFSETS.index(-5)]:+.2f}, at -2 {v['mean'][j2]:+.2f} "
                  f"[{v['ci_lo'][j2]:+.2f}, {v['ci_hi'][j2]:+.2f}], at 0 {v['mean'][OFFSETS.index(0)]:+.2f}  "
                  f"lead: {v['lead_claimed']} {v['offsets_ci_excludes_0_at_or_before_-2']}")

    rep["T3_landmark"] = t3_landmark(tr, [("test", te), ("unseen", un)])
    print("\nT3  landmark survival: C-index (plain / within task), u=16 predicted vs observed unbroken")
    for s, r in rep["T3_landmark"].items():
        for split in ("test", "unseen"):
            if split in r:
                v = r[split]
                print(f"  s={s:<3}{split:<7} n {v['dreams_at_risk']:<4} C {v['c_index']} / {v['c_index_within_task']}"
                      f"   u16 {v['u16_mean_predicted']} vs {v['u16_observed']}")

    rep["T4_trees"] = t4_trees(args)
    print("\nT4  k=8 forecast: Brier, AUROC / beyond time / within task and time")
    for split, r in rep["T4_trees"].items():
        for m, v in r.items():
            print(f"  {split:<7}{m:<34}{v['brier']:<9}{v['auroc']} / {v['beyond_time']} / {v['within_task_and_time']}")

    (out / "report.json").write_text(json.dumps(rep, indent=2, default=str))
    print(f"\nwritten {out / 'report.json'}")


if __name__ == "__main__":
    main()
