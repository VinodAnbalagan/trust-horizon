"""Phase 4: open the sealed unseen-task dreams, once, and test everything on them. CPU only.

    python phase4.py            # checks the seal, unseals, evaluates, writes runs/phase4/report.json

The forecast model is fit on Phase 1 `fit` dreams only (seen tasks), exactly as in Phase 2,
then evaluated on the seen `test` split and on the unseen tasks side by side.
Running it a second time refuses, so the unseen result cannot be quietly re-tuned.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from th.hazard import (auroc, brier, fit_logistic, kaplan_meier, load_split, median_survival,  # noqa: E402
                       reliability, warning_times)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--again", action="store_true", help="re-run after unsealing (results are logged, not hidden)")
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
    if not (unseen / "UNSEALED").exists():
        import time
        (unseen / "UNSEALED").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))

    tr = load_split(p1 / "fit", args.k)
    te = load_split(p1 / "test", args.k, weak_tasks=tr.weak)
    un = load_split(unseen, args.k)        # weak-baseline flags decided from the unseen tasks' own dreams

    model = fit_logistic(tr.X, tr.y)
    res = {}
    for name, sp in (("seen test", te), ("unseen", un)):
        p = model.predict_proba(sp.X)[:, 1] if len(sp.y) else np.empty(0)
        rel = reliability(sp.y, p) if len(sp.y) else {"max_gap_bins_ge_min": None, "bins": []}
        warn = warning_times(sp, p) if len(sp.y) else []
        surv = {}
        for task, on in sp.onsets.items():
            S = kaplan_meier(on, sp.horizons[task])
            surv[task] = {"median_trust_horizon": median_survival(S), "frac_failed": round(float(np.mean(~np.isnan(on))), 3)}
        res[name] = {"n_rows": int(len(sp.y)), "positive_rate": round(float(sp.y.mean()), 4) if len(sp.y) else None,
                     "brier": round(brier(sp.y, p), 5) if len(sp.y) else None,
                     "auroc": None if not len(sp.y) else (None if (a := auroc(sp.y, p)) is None else round(a, 3)),
                     "max_calibration_gap": rel["max_gap_bins_ge_min"], "reliability": rel["bins"],
                     "median_warning": None if not warn else float(np.median(warn)),
                     "weak_tasks": sp.weak, "survival": surv}
    a_seen, a_un = res["seen test"]["auroc"], res["unseen"]["auroc"]
    guess5 = None if a_seen is None or a_un is None else abs(a_seen - a_un) <= 0.05
    report = {"seal": seal, "k": args.k, "results": res,
              "guess5_unseen_auroc_within_0.05": guess5}
    (out / "report.json").write_text(json.dumps(report, indent=2))

    print(f"\n========== forecast P(break within {args.k} steps): seen vs unseen ==========")
    print(f"{'':<12}{'rows':>8}{'pos rate':>10}{'Brier':>9}{'AUROC':>8}{'cal gap':>9}{'warning':>9}")
    for name, r in res.items():
        print(f"{name:<12}{r['n_rows']:>8}{str(r['positive_rate']):>10}{str(r['brier']):>9}{str(r['auroc']):>8}"
              f"{str(None if r['max_calibration_gap'] is None else round(r['max_calibration_gap'], 3)):>9}"
              f"{str(r['median_warning']):>9}")
    print(f"\n5. unseen AUROC within 0.05 of seen: {guess5}   (you said 40%)")
    print("\nTrust horizon on unseen tasks:")
    for task, v in res["unseen"]["survival"].items():
        print(f"  {task:<28} median {str(v['median_trust_horizon']):>5}   failed {v['frac_failed']}")
    print(f"\nSaved {out / 'report.json'}")


if __name__ == "__main__":
    main()
