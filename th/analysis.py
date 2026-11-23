"""Phase 1 checks: reproduce the paper's correlation, and summarise long dreams.

    python -m th.analysis repro      # runs/phase1/repro -> repro_report.json
    python -m th.analysis fit        # runs/phase1/fit   -> long_report.json
    python -m th.analysis test

Never run this on runs/phase1/unseen before Phase 4.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from th.labels import MSE_NOISE_FLOOR, dpsnr_np, onsets, weak_baseline

SIGNALS = ("u_r", "u_f", "u_s")


def _load(path: Path) -> dict:
    with np.load(path) as z:
        d = {k: z[k] for k in z.files}
    d["dpsnr"] = dpsnr_np(d["mse"], d["mse_floor"])   # always the noise-floored version
    return d


def _rho(x, y):
    r = spearmanr(x, y).correlation
    return None if np.isnan(r) else round(float(r), 3)


def _seq_err(d, h, floor_c):
    """Sequence error = -dPSNR over the first h steps from mean MSEs (paper: run_dynamics_eval)."""
    mse, fl = d["mse"][:, :h].mean(1), d["mse_floor"][:, :h].mean(1)
    if floor_c:
        return -10 * np.log10((fl + floor_c) / (mse + floor_c))
    with np.errstate(divide="ignore"):
        return -10 * np.log10(fl / np.maximum(mse, 1e-12))


def _signal(d, s, h, agg, norm):
    """Per-sequence signal: aggregate over steps, divided by the task's motion scale."""
    x = d[s][:, :h]
    x = {"mean": x.mean(1), "max": x.max(1), "last": x[:, -1]}[agg]
    if norm == "real" and "motion_real" in d:
        m = float(d["motion_real"][:, :h].mean())
    else:
        m = float(d["motion"][:, :h].mean())
    return x / max(m, 1e-6)


VARIANTS = [
    # name,                            floor_c,          agg,    norm
    ("paper-exact (as far as stated)", 0.0,              "mean", "real"),
    ("noise-floored dPSNR",            MSE_NOISE_FLOOR,  "mean", "real"),
    ("signal = max over steps",        0.0,              "max",  "real"),
    ("signal = last step",             0.0,              "last", "real"),
    ("normalized by predicted motion", 0.0,              "mean", "pred"),
]


def repro_report(stage_dir: Path) -> dict:
    """Paper protocol: 8 context + 16 dreamed frames, base model, pooled Spearman over sequences.
    We correlate each signal with the error (-dPSNR), so a good predictor gives a positive rho.
    Paper (Figure 5): rho about 0.8, pooled over 9k sequences from 200 tasks."""
    stage_dir = Path(stage_dir)
    data = {}
    for f in sorted(stage_dir.glob("*.npz")):
        if not f.name.endswith(".tmp.npz"):
            with np.load(f) as z:
                data[f.stem] = {k: z[k] for k in z.files}
    if not data:
        print("no repro files found")
        return {}
    sig_names = [s for s in ("u_r", "u_f", "u_s", "u_s4", "u_r_s0") if s in next(iter(data.values()))]

    rep = {"n_tasks": len(data), "variants": {}}
    for vname, c, agg, norm in VARIANTS:
        xs = {s: [] for s in sig_names}; ys = []; task_x = {s: [] for s in sig_names}; task_y = []
        within = {s: [] for s in sig_names}; dropped = 0
        for t, d in data.items():
            h = min(16, d["dpsnr"].shape[1])
            y = _seq_err(d, h, c)
            ok = np.isfinite(y)
            dropped += int((~ok).sum())
            if ok.sum() < 3:
                continue
            task_y.append(float(np.median(y[ok])))
            for s in sig_names:
                x = _signal(d, s, h, agg, norm)[ok]
                xs[s].append(x); task_x[s].append(float(np.median(x)))
                within[s].append(_rho(x, y[ok]))
            ys.append(y[ok])
        Y = np.concatenate(ys)
        rep["variants"][vname] = {
            "n_sequences": int(len(Y)), "dropped_undefined": dropped,
            "pooled": {s: _rho(np.concatenate(xs[s]), Y) for s in sig_names},
            "within_task_median": {s: (round(float(np.median([r for r in within[s] if r is not None])), 3)
                                       if any(r is not None for r in within[s]) else None) for s in sig_names},
            "task_level": {s: _rho(task_x[s], task_y) for s in sig_names},
        }

    # per-task table for the paper-exact variant
    per_task = {}
    for t, d in data.items():
        h = min(16, d["dpsnr"].shape[1])
        y = _seq_err(d, h, 0.0); ok = np.isfinite(y)
        row = {"n": int(ok.sum()), "static_windows": int((~ok).sum()),
               "mean_seq_dpsnr": None if not ok.any() else round(float(-y[ok].mean()), 2)}
        for s in ("u_r", "u_f", "u_s"):
            row[s] = _rho(_signal(d, s, h, "mean", "real")[ok], y[ok]) if ok.sum() >= 3 else None
        per_task[t] = row
    rep["per_task_paper_exact"] = per_task
    (stage_dir / "repro_report.json").write_text(json.dumps(rep, indent=2))

    print("\n=========== reproduction: Spearman(signal, rollout error); paper: about 0.8 pooled ===========")
    for vname, v in rep["variants"].items():
        print(f"\n{vname}   ({v['n_sequences']} sequences, {v['dropped_undefined']} static ones dropped)")
        print(f"  {'':<22}" + "".join(f"{s:>8}" for s in sig_names))
        for k, lab in (("pooled", "pooled (paper's way)"), ("task_level", "task level (20 pts)"),
                       ("within_task_median", "within task, median")):
            print(f"  {lab:<22}" + "".join(f"{str(v[k][s]):>8}" for s in sig_names))
    print("\nper task, paper-exact:")
    for t, r in per_task.items():
        print(f"  {t:<26}" + "".join(f"{str(r[s]):>8}" for s in ("u_r", "u_f", "u_s"))
              + f"   dPSNR {str(r['mean_seq_dpsnr']):>6}" + (f"   ({r['static_windows']} static)" if r["static_windows"] else ""))
    return rep


def long_report(stage_dir: Path) -> dict:
    """Failure onsets under the pre-registered rule, censoring, weak-baseline flags."""
    stage_dir = Path(stage_dir)
    if stage_dir.name == "unseen":
        raise SystemExit("unseen is sealed until Phase 4")
    per_task = {}
    for f in sorted(stage_dir.glob("*.npz")):
        if f.name.endswith(".tmp.npz"):
            continue
        d = _load(f)
        on = onsets(d["dpsnr"])
        per_task[f.stem] = {
            "n": int(len(on)), "horizon": int(d["dpsnr"].shape[1]),
            "weak_baseline": weak_baseline(d["dpsnr"]),
            "frac_failed": round(float(np.mean(~np.isnan(on))), 2),
            "median_onset_if_failed": None if np.all(np.isnan(on)) else float(np.nanmedian(on)),
            "onsets": [None if np.isnan(o) else int(o) for o in on],
        }
    usable = {t: r for t, r in per_task.items() if not r["weak_baseline"]}
    frac = np.mean([o is not None for r in usable.values() for o in r["onsets"]]) if usable else float("nan")
    rep = {"n_tasks": len(per_task), "weak_baseline_tasks": [t for t, r in per_task.items() if r["weak_baseline"]],
           "frac_failed_usable": round(float(frac), 3), "per_task": per_task}
    (stage_dir / "long_report.json").write_text(json.dumps(rep, indent=2))

    print(f"\n=========== long dreams: {stage_dir.name} ===========")
    print(f"weak-baseline tasks (excluded from failure models): {rep['weak_baseline_tasks']}")
    print(f"fraction of usable dreams that break within the horizon: {rep['frac_failed_usable']}"
          "   (plan check: want at least 0.20)")
    for t, r in per_task.items():
        flag = "  weak baseline" if r["weak_baseline"] else ""
        print(f"  {t:<26} failed {r['frac_failed']:>5}  median onset {str(r['median_onset_if_failed']):>6}"
              f"  horizon {r['horizon']}{flag}")
    return rep


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "repro"
    d = Path(__file__).resolve().parents[1] / "runs" / "phase1" / which
    repro_report(d) if which == "repro" else long_report(d)
