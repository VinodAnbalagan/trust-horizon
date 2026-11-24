"""Phase 2: hazard model, calibration, warning time, survival curves. CPU only.

    python phase2.py                 # fit on runs/phase1/fit, evaluate on runs/phase1/test
    python phase2.py --k 8           # forecast window (steps ahead); 8 is the pre-registered value
    python phase2.py --label error   # direct-error label (PROTOCOL item 18), tau 0.5 by default

Outputs in runs/phase2/: report.json and four figures
(reliability.png, warning.png, survival.png, forecasts.png).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from th.hazard import (SIGNALS, ThresholdRule, auroc, brier, feature_names, fit_logistic,  # noqa: E402,F401
                       kaplan_meier, load_split, median_survival, reliability, signal_columns,
                       warning_times)


def evaluate(name, model, tr, te, cols=None):
    Xtr = tr.X if cols is None else tr.X[:, cols]
    Xte = te.X if cols is None else te.X[:, cols]
    model.fit(Xtr, tr.y)
    p = model.predict_proba(Xte)[:, 1]
    return {"name": name, "brier": round(brier(te.y, p), 5), "auroc": _r(auroc(te.y, p))}, p


def _r(x, n=3):
    return None if x is None else round(float(x), n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-dir", default=str(ROOT / "runs" / "phase1" / "fit"))
    ap.add_argument("--test-dir", default=str(ROOT / "runs" / "phase1" / "test"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--label", choices=["dpsnr", "error"], default="dpsnr")
    ap.add_argument("--tau", type=float, default=0.5)
    args = ap.parse_args()
    tag = "dpsnr" if args.label == "dpsnr" else f"error_tau{args.tau:g}"
    out = Path(args.out) if args.out else ROOT / "runs" / "phase2" / tag
    out.mkdir(parents=True, exist_ok=True)

    tr = load_split(Path(args.fit_dir), args.k, label=args.label, tau=args.tau)
    te = load_split(Path(args.test_dir), args.k, weak_tasks=tr.weak, label=args.label, tau=args.tau, chance=tr.chance)
    print(f"label: {tag}")
    print(f"fit:  {len(tr.onsets)} tasks, {len(tr.win_onset)} dreams, {len(tr.y)} forecast rows, "
          f"positive rate {tr.y.mean():.3f}")
    print(f"test: {len(te.onsets)} tasks, {len(te.win_onset)} dreams, {len(te.y)} forecast rows, "
          f"positive rate {te.y.mean():.3f}")
    print(f"weak-baseline tasks left out: {tr.weak}")

    # ---------------- models
    results, probs = [], {}
    probs["base rate"] = np.full(len(te.y), tr.y.mean())
    results.append({"name": "base rate", "brier": round(brier(te.y, probs["base rate"]), 5), "auroc": None})
    step_col = [feature_names().index("log_step")]
    r, probs["time only"] = evaluate("time only (no signals)", _Lazy(), tr, te, step_col)
    results.append(r)
    for s in SIGNALS:
        r, probs[f"{s} threshold"] = evaluate(f"{s} threshold rule", ThresholdRule(feature_names().index(f"{s}_level")), tr, te)
        results.append(r)
        r, probs[f"{s} logistic"] = evaluate(f"{s} logistic", _Lazy(), tr, te, signal_columns(s) + step_col)
        results.append(r)
    combined = _Lazy()
    r, probs["combined"] = evaluate("combined (all signals)", combined, tr, te)
    results.append(r)

    # ---------------- detection: is the dream ALREADY broken at step t? (not a forecast)
    detection = {}
    if len(set(tr.yd)) == 2 and len(set(te.yd)) == 2:
        dm = _Lazy().fit(tr.Xd, tr.yd)
        detection["combined"] = _r(auroc(te.yd, dm.predict_proba(te.Xd)[:, 1]))
        for s_ in SIGNALS:
            detection[f"{s_} level"] = _r(auroc(te.yd, te.Xd[:, feature_names().index(f"{s_}_level")]))
        detection["positive_rate_test"] = _r(te.yd.mean())

    # ---------------- pre-registered checks
    thr_rows = [x for x in results if x["name"].endswith("threshold rule")]
    best_thr = min(thr_rows, key=lambda x: x["brier"])
    comb = next(x for x in results if x["name"].startswith("combined"))
    rel = reliability(te.y, probs["combined"])
    warn = warning_times(te, probs["combined"], 0.5)
    lr = combined.model.named_steps["logisticregression"]
    coefs = dict(zip(feature_names(), lr.coef_[0].tolist()))
    weight = {s: float(sum(abs(coefs[n]) for n in feature_names() if n.startswith(s))) for s in SIGNALS}
    # Robust importance (PROTOCOL item 19): drop in test AUROC when a signal's features are removed.
    # Weights of correlated features are unstable; a known-answer test picked the wrong signal by weight.
    full_auc = auroc(te.y, probs["combined"]) or 0.5
    ablation = {}
    for s in SIGNALS:
        keep = [i for i, n in enumerate(feature_names()) if not n.startswith(s)]
        m_ = _Lazy().fit(tr.X[:, keep], tr.y)
        ablation[s] = round(full_auc - (auroc(te.y, m_.predict_proba(te.X[:, keep])[:, 1]) or 0.5), 4)
    step1 = int(np.sum(te.win_onset == 1))

    checks = {
        "1_combined_beats_best_threshold": {
            "combined_brier": comb["brier"], "best_threshold": best_thr["name"], "best_threshold_brier": best_thr["brier"],
            "result": comb["brier"] < best_thr["brier"]},
        "2_calibrated_within_0.1": {"max_gap": _r(rel["max_gap_bins_ge_min"]),
                                    "result": rel["max_gap_bins_ge_min"] is not None and rel["max_gap_bins_ge_min"] <= 0.1},
        "3_median_warning_ge_3": {"median_warning": None if not warn else float(np.median(warn)), "n_breaks": len(warn),
                                  "breaks_at_step_1_excluded": step1,
                                  "result": bool(warn) and float(np.median(warn)) >= 3},
        "6_largest_weight": {"weights": {k: round(v, 3) for k, v in weight.items()},
                             "result_by_weight": max(weight, key=weight.get),
                             "auroc_drop_when_removed": ablation,
                             "result": max(ablation, key=ablation.get)},
    }

    # ---------------- survival (fit and test together; both are seen tasks)
    surv = {}
    for name in tr.onsets:
        on = np.concatenate([tr.onsets[name], te.onsets.get(name, np.empty(0))])
        S = kaplan_meier(on, tr.horizons[name])
        surv[name] = {"median_trust_horizon": median_survival(S), "n": int(len(on)),
                      "frac_failed": round(float(np.mean(~np.isnan(on))), 3), "S": S.round(4).tolist()}

    report = {"label": tag, "k": args.k, "fit_tasks": list(tr.onsets), "weak_tasks": tr.weak,
              "chance_error": {k_: round(v, 4) for k_, v in tr.chance.items()}, "detection_auroc": detection,
              "dreams_broken": {"fit": round(float(np.mean(~np.isnan(tr.win_onset))), 3),
                                "test": round(float(np.mean(~np.isnan(te.win_onset))), 3)},
              "positive_rate": {"fit": round(float(tr.y.mean()), 4), "test": round(float(te.y.mean()), 4)},
              "models": results, "preregistered_checks": checks, "reliability_combined": rel,
              "warning_times": warn, "coefficients_combined": {k: round(v, 4) for k, v in coefs.items()},
              "survival": surv}
    (out / "report.json").write_text(json.dumps(report, indent=2))

    # ---------------- print
    print(f"dreams that break: fit {report['dreams_broken']['fit']}, test {report['dreams_broken']['test']}")
    print(f"\n========== detection: is the dream already broken at step t? (AUROC on test) ==========")
    for k_, v in detection.items():
        print(f"  {k_:<26}{v}")
    print(f"\n========== forecast: P(dream breaks within {args.k} steps), evaluated on test ==========")
    print(f"{'model':<28}{'Brier':>9}{'AUROC':>8}")
    for x in results:
        print(f"{x['name']:<28}{x['brier']:>9}{str(x['auroc']):>8}")
    print("\n========== pre-registered guesses ==========")
    c = checks
    print(f"1. combined beats best threshold ({c['1_combined_beats_best_threshold']['best_threshold']}): "
          f"{c['1_combined_beats_best_threshold']['result']}   (you said 70%)")
    print(f"2. calibrated within 0.1 (max gap {c['2_calibrated_within_0.1']['max_gap']}): "
          f"{c['2_calibrated_within_0.1']['result']}   (you said 55%)")
    print(f"3. median warning >= 3 steps (median {c['3_median_warning_ge_3']['median_warning']}, "
          f"{c['3_median_warning_ge_3']['n_breaks']} breaks): {c['3_median_warning_ge_3']['result']}   (you said 40%)")
    print(f"6. most important signal (AUROC drop when removed): {c['6_largest_weight']['result']} "
          f"{c['6_largest_weight']['auroc_drop_when_removed']}   (you said u_s, 50%)")
    print(f"   by raw weight (unreliable with correlated signals): {c['6_largest_weight']['result_by_weight']}")
    print("   (4 is tested in Phase 3, 5 in Phase 4)")
    print("\n========== trust horizon (median steps before a dream breaks; None = beyond horizon) ==========")
    for name, v in surv.items():
        print(f"  {name:<26} {str(v['median_trust_horizon']):>5}   failed {v['frac_failed']:>5}  of {v['n']}")

    _figures(out, te, probs, rel, warn, surv, args.k)
    print(f"\nSaved report and figures to {out}")


class _Lazy:
    """Standardized logistic regression with a sklearn-like fit/predict_proba."""

    def fit(self, X, y):
        self.model = fit_logistic(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)


def _figures(out, te, probs, rel, warn, surv, k):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # reliability: combined vs best single logistic
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=0.8)
    for label, color in (("combined", "black"), ("u_s logistic", "tab:green"), ("u_r logistic", "tab:red")):
        rl = reliability(te.y, probs[label])
        xs = [b["mean_pred"] for b in rl["bins"] if b["n"] >= 20]
        ys = [b["frac_pos"] for b in rl["bins"] if b["n"] >= 20]
        ax.plot(xs, ys, "o-", color=color, lw=1.5 if label == "combined" else 1, ms=4, label=label)
    ax.set_xlabel(f"forecast P(break within {k} steps)")
    ax.set_ylabel("observed fraction that broke")
    ax.set_title("Calibration on held-out dreams", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "reliability.png", dpi=130); plt.close(fig)

    # warning times
    fig, ax = plt.subplots(figsize=(5, 3.2))
    if warn:
        ax.hist(warn, bins=np.arange(0, max(warn) + 2) - 0.5, color="tab:blue")
        ax.axvline(np.median(warn), color="black", ls="--", lw=1, label=f"median {np.median(warn):.0f}")
        ax.legend(fontsize=8)
    ax.set_xlabel("steps of warning before the dream broke (forecast >= 0.5)")
    ax.set_ylabel("dreams")
    fig.tight_layout(); fig.savefig(out / "warning.png", dpi=130); plt.close(fig)

    # survival curves
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, v in surv.items():
        ax.step(np.arange(1, len(v["S"]) + 1), v["S"], where="post", lw=1, alpha=0.7, label=name)
    ax.axhline(0.5, color="grey", ls=":", lw=0.8)
    ax.set_xlabel("dream step")
    ax.set_ylabel("fraction of dreams still trustworthy")
    ax.set_ylim(0, 1.02)
    ax.set_title("Trust horizon per task (Kaplan-Meier)", fontsize=10)
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(out / "survival.png", dpi=130); plt.close(fig)

    # example forecasts on test dreams that break
    fail, seen_tasks = [], set()
    for w, o in enumerate(te.win_onset):   # first breaking dream of up to 4 different tasks
        if not np.isnan(o) and o > 8 and te.win_task[w] not in seen_tasks:
            fail.append(w); seen_tasks.add(te.win_task[w])
        if len(fail) == 4:
            break
    if fail:
        fig, axes = plt.subplots(len(fail), 1, figsize=(8, 2.1 * len(fail)), squeeze=False)
        offsets, t0 = {}, 0
        for name, on in te.onsets.items():
            offsets[name] = t0; t0 += len(on)
        for ax, w in zip(axes[:, 0], fail):
            name = te.win_task[w]
            dp = te.dpsnr[name][w - offsets[name]]
            m = te.win == w
            ax.plot(np.arange(1, len(dp) + 1), dp, color="black", lw=1.2, label="dPSNR (dB)")
            ax.axhline(0, color="grey", ls="--", lw=0.6)
            ax.axvline(te.win_onset[w], color="tab:red", lw=1, label="break (rule)")
            ax2 = ax.twinx()
            ax2.plot(te.step[m], probs["combined"][m], color="tab:blue", lw=1.5, label=f"P(break within {k})")
            ax2.set_ylim(0, 1)
            ax.set_title(f"{name}", fontsize=8, loc="left")
        from matplotlib.lines import Line2D
        axes[0, 0].legend(handles=[Line2D([], [], color="black", label="dPSNR (dB, left axis)"),
                                   Line2D([], [], color="tab:red", label="break (rule)"),
                                   Line2D([], [], color="tab:blue", label=f"forecast P(break within {k}), right axis")],
                          fontsize=7, loc="upper right")
        axes[-1, 0].set_xlabel("dream step")
        fig.tight_layout(); fig.savefig(out / "forecasts.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
