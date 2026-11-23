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

from th.labels import onsets, weak_baseline

SIGNALS = ("u_r", "u_f", "u_s")


def _load(path: Path) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def _rho(x, y):
    r = spearmanr(x, y).correlation
    return None if np.isnan(r) else round(float(r), 3)


def repro_report(stage_dir: Path) -> dict:
    """Paper protocol: 8 context + 16 dreamed frames. Signals divided by the task's average
    latent motion, averaged over the 16 steps, vs sequence-level dPSNR. We correlate each
    signal with -dPSNR, so a good predictor gives a positive rho (paper: about 0.8)."""
    stage_dir = Path(stage_dir)
    per_task, pooled = {}, {s: [] for s in SIGNALS}
    pooled_y = []
    for f in sorted(stage_dir.glob("*.npz")):
        if f.name.endswith(".tmp.npz"):
            continue
        d = _load(f)
        h = min(16, d["dpsnr"].shape[1])
        y = -10 * np.log10(d["mse_floor"][:, :h].mean(1) / d["mse"][:, :h].mean(1))   # higher = worse
        m = max(float(d["motion"][:, :h].mean()), 1e-6)
        row = {"n": int(len(y)), "mean_seq_dpsnr": round(float(-y.mean()), 2)}
        for s in SIGNALS:
            x = d[s][:, :h].mean(1) / m
            row[s] = _rho(x, y)
            pooled[s].append(x)
        pooled_y.append(y)
        per_task[f.stem] = row

    if not per_task:
        print("no repro files found")
        return {}
    Y = np.concatenate(pooled_y)
    rep = {
        "n_tasks": len(per_task), "n_sequences": int(len(Y)),
        "pooled_rho": {s: _rho(np.concatenate(pooled[s]), Y) for s in SIGNALS},
        "median_within_task_rho": {
            s: (round(float(np.median(v)), 3) if (v := [r[s] for r in per_task.values() if r[s] is not None]) else None)
            for s in SIGNALS
        },
        "per_task": per_task,
    }
    (stage_dir / "repro_report.json").write_text(json.dumps(rep, indent=2))

    print("\n=========== reproduction: signal vs rollout error (paper: rho about 0.8) ===========")
    print(f"{rep['n_sequences']} sequences from {rep['n_tasks']} tasks")
    print(f"{'':<28}{'u_r':>8}{'u_f':>8}{'u_s':>8}")
    print(f"{'pooled across tasks':<28}" + "".join(f"{str(rep['pooled_rho'][s]):>8}" for s in SIGNALS))
    print(f"{'median within a task':<28}" + "".join(f"{str(rep['median_within_task_rho'][s]):>8}" for s in SIGNALS))
    print()
    for t, r in per_task.items():
        print(f"  {t:<26}" + "".join(f"{str(r[s]):>8}" for s in SIGNALS) + f"   dPSNR {r['mean_seq_dpsnr']:>6}")
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
