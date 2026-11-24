"""Phase 4: open the sealed unseen-task dreams, once, and test everything on them. CPU only.

    python phase4.py            # checks the seal, unseals, evaluates, writes runs/phase4/report.json

For each break definition (PROTOCOL items 16 and 18: the paper's frozen-frame label, the direct-error
label at tau 0.5 = primary, and tau 1.0), models are fit on Phase 1 `fit` dreams only (seen tasks),
exactly as in Phase 2, then evaluated on the seen `test` split and on the 10 unseen tasks side by side.

Reported per label and split: detection ("already broken at step t") and forecast ("breaks within
k steps"), each as plain AUROC and as AUROC beyond time (within groups of similar dream step, so the
fact that later steps break more cannot inflate it), Brier score, largest calibration gap, median
warning, and the trust horizon (Kaplan-Meier median) per task.

Guess 5 (pre-registered): forecast AUROC on unseen within 0.05 of seen. Scored on the primary label,
using plain AUROC as pre-registered; beyond-time AUROC is reported next to it.

A second run is refused, so the unseen result cannot be quietly re-tuned.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from th.hazard import (brier, feature_names, fit_logistic, kaplan_meier, load_split,  # noqa: E402
                       median_survival, reliability, warning_times)

LABELS = [("dpsnr", 0.5, "frozen frame (paper)"), ("error", 0.5, "halfway to a random frame (primary)"),
          ("error", 1.0, "as far as a random frame (strict check)")]
STEP_GROUPS = [(1, 8), (9, 16), (17, 32), (33, 48), (49, 64), (65, 96)]


def check_seal(unseen: Path) -> dict:
    man = json.loads((unseen / "manifest.json").read_text())
    bad, ok = [], 0
    for task, meta in man["tasks"].items():
        if "sha256" not in meta:
            continue
        f = unseen / f"{task}.npz"
        if not f.exists() or hashlib.sha256(f.read_bytes()).hexdigest() != meta["sha256"]:
            bad.append(task)
        else:
            ok += 1
    return {"intact": ok, "changed_or_missing": bad}


def _auc(y, p):
    from sklearn.metrics import roc_auc_score
    return None if len(np.unique(y)) < 2 else round(float(roc_auc_score(y, p)), 3)


def _auc_beyond_time(y, p, steps):
    from sklearn.metrics import roc_auc_score
    vals, w = [], []
    for lo, hi in STEP_GROUPS:
        m = (steps >= lo) & (steps <= hi)
        if m.sum() > 50 and 0 < y[m].mean() < 1:
            vals.append(roc_auc_score(y[m], p[m])); w.append(m.sum())
    return None if not vals else round(float(np.average(vals, weights=w)), 3)


def evaluate(tr, sp, k):
    step_col = feature_names().index("log_step")
    out = {"n_dreams": int(len(sp.win_onset)), "weak_tasks": sp.weak,
           "frac_dreams_broken": None if not len(sp.win_onset) else round(float(np.mean(~np.isnan(sp.win_onset))), 3)}
    # forecast
    if len(sp.y) and len(set(tr.y)) == 2:
        m = fit_logistic(tr.X, tr.y)
        p = m.predict_proba(sp.X)[:, 1]
        warn = warning_times(sp, p)
        out["forecast"] = {"positive_rate": round(float(sp.y.mean()), 4), "auroc": _auc(sp.y, p),
                           "auroc_beyond_time": _auc_beyond_time(sp.y, p, sp.step),
                           "brier": round(brier(sp.y, p), 5),
                           "max_calibration_gap": reliability(sp.y, p)["max_gap_bins_ge_min"],
                           "median_warning": None if not warn else float(np.median(warn))}
    # detection
    if len(sp.yd) and len(set(tr.yd)) == 2:
        md = fit_logistic(tr.Xd, tr.yd)
        pd = md.predict_proba(sp.Xd)[:, 1]
        st = np.round(np.exp(sp.Xd[:, step_col])).astype(int)
        out["detection"] = {"positive_rate": round(float(sp.yd.mean()), 4), "auroc": _auc(sp.yd, pd),
                            "auroc_beyond_time": _auc_beyond_time(sp.yd, pd, st)}
    out["trust_horizon"] = {t: {"median": median_survival(kaplan_meier(on, sp.horizons[t])),
                                "frac_broken": round(float(np.mean(~np.isnan(on))), 3)} for t, on in sp.onsets.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--again", action="store_true", help="regenerate the same report after unsealing")
    args = ap.parse_args()
    p1 = ROOT / "runs" / "phase1"
    unseen = p1 / "unseen"
    out = ROOT / "runs" / "phase4"
    out.mkdir(parents=True, exist_ok=True)

    if (unseen / "UNSEALED").exists() and not args.again:
        raise SystemExit("Already unsealed on " + (unseen / "UNSEALED").read_text().strip()
                         + ". Results are in runs/phase4/report.json. Use --again only to regenerate the same report.")
    seal = check_seal(unseen)
    print(f"seal check: {seal['intact']} files intact, changed or missing: {seal['changed_or_missing'] or 'none'}")
    if seal["changed_or_missing"]:
        raise SystemExit("Seal broken: stop and investigate before opening anything.")
    if not (unseen / "UNSEALED").exists():
        (unseen / "UNSEALED").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))

    report = {"seal": seal, "k": args.k, "labels": {}}
    for label, tau, desc in LABELS:
        tag = "dpsnr" if label == "dpsnr" else f"error_tau{tau:g}"
        tr = load_split(p1 / "fit", args.k, label=label, tau=tau)
        te = load_split(p1 / "test", args.k, weak_tasks=tr.weak, label=label, tau=tau, chance=tr.chance)
        un = load_split(unseen, args.k, label=label, tau=tau)   # unseen: weak flags and chance error from its own dreams
        report["labels"][tag] = {"description": desc, "seen_test": evaluate(tr, te, args.k), "unseen": evaluate(tr, un, args.k)}

    prim = report["labels"]["error_tau0.5"]
    a_s, a_u = prim["seen_test"].get("forecast", {}).get("auroc"), prim["unseen"].get("forecast", {}).get("auroc")
    report["guess5_unseen_forecast_auroc_within_0.05"] = None if a_s is None or a_u is None else abs(a_s - a_u) <= 0.05
    (out / "report.json").write_text(json.dumps(report, indent=2))

    print(f"\n========== seen (held-out test) vs UNSEEN tasks; AUROC, 0.5 = no skill ==========")
    for tag, r in report["labels"].items():
        print(f"\n{r['description']}")
        print(f"  {'':<10}{'dreams':>7}{'broken':>8}{'detect':>8}{'beyond t':>9}{'forecast':>9}{'beyond t':>9}{'cal gap':>8}{'warn':>6}")
        for split in ("seen_test", "unseen"):
            v = r[split]; d = v.get("detection", {}); f = v.get("forecast", {})
            gap = f.get("max_calibration_gap")
            print(f"  {split:<10}{v['n_dreams']:>7}{str(v['frac_dreams_broken']):>8}{str(d.get('auroc')):>8}"
                  f"{str(d.get('auroc_beyond_time')):>9}{str(f.get('auroc')):>9}{str(f.get('auroc_beyond_time')):>9}"
                  f"{str(None if gap is None else round(gap, 2)):>8}{str(f.get('median_warning')):>6}")
    print(f"\n5. unseen forecast AUROC within 0.05 of seen (primary label): "
          f"{report['guess5_unseen_forecast_auroc_within_0.05']}   (you said 40%)")
    print("\nTrust horizon on unseen tasks, primary label (median steps; None = beyond horizon):")
    for t, v in prim["unseen"]["trust_horizon"].items():
        print(f"  {t:<28} {str(v['median']):>5}   broken {v['frac_broken']}")
    print(f"\nSaved {out / 'report.json'}")


if __name__ == "__main__":
    main()
