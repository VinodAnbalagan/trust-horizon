"""Export the data the Trust Horizon website shows. CPU only, reads runs/ and writes site/src/data.

    python site/scripts/export_site_data.py

Everything here is recomputed from the Phase 1 dreams with the same code the analysis used
(th.hazard, forecast_tests), so the site shows the same numbers as the report.json files.
Primary label throughout: latent error >= 0.5 x the task's chance error.

Writes
  site/src/data/site.json          numbers and curves for every chart
  site/src/data/dreams.json        per-step traces for the showcased dream of each task
  site/src/assets/frames/*.webp    real and dreamed frames of that dream, as 12 x 8 sprite sheets
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import forecast_tests as ft  # noqa: E402
from th.hazard import SIGNALS, feature_names, fit_logistic, kaplan_meier, load_split, median_survival  # noqa: E402

P1 = ROOT / "runs" / "phase1"
OUT = ROOT / "site" / "src" / "data"
FRAMES = ROOT / "site" / "src" / "assets" / "frames"
K, TAU = 8, 0.5
STEP_COL = feature_names().index("log_step")


def r(x, n=3):
    return None if x is None else round(float(x), n)


def det_tasks(sp):
    """Task name per detection row (rows are task by task, window by window, step by step)."""
    out = []
    for name in sorted(sp.feats):
        n, T, _ = sp.feats[name].shape
        out += [name] * (n * T)
    return np.array(out)


def km_curve(on, T):
    S = kaplan_meier(on, T)
    return [r(v) for v in S], median_survival(S)


def sprite(frames):
    """(T, 3, 64, 64) uint8 -> 12 x 8 grid image (T <= 96)."""
    T = frames.shape[0]
    cols, rows = 12, 8
    img = np.zeros((rows * 64, cols * 64, 3), np.uint8)
    for t in range(T):
        rr, cc = divmod(t, cols)
        img[rr * 64:(rr + 1) * 64, cc * 64:(cc + 1) * 64] = frames[t].transpose(1, 2, 0)
    return Image.fromarray(img)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FRAMES.mkdir(parents=True, exist_ok=True)

    tr = load_split(P1 / "fit", K, label="error", tau=TAU)
    te = load_split(P1 / "test", K, label="error", tau=TAU, chance=tr.chance)
    un = load_split(P1 / "unseen", K, label="error", tau=TAU)

    fm = fit_logistic(tr.X, tr.y)      # forecast: breaks within the next 8 steps
    dm = fit_logistic(tr.Xd, tr.yd)    # detection: already broken at step t

    site = {"k": K, "tau": TAU}

    # ------------------------------------------------ detection and forecast, three views
    def three(y, p, st, tasks):
        # forecast_tests rounds to 3 decimals; keep 4 here so the site's 2-decimal display rounds correctly
        keep, ft._r = ft._r, (lambda x, n=4: None if x is None else round(float(x), 4))
        try:
            return ft.three_aurocs(y, p, st, tasks)
        finally:
            ft._r = keep

    views = {}
    for nm, sp in (("seen", te), ("unseen", un)):
        pd = dm.predict_proba(sp.Xd)[:, 1]
        st = np.round(np.exp(sp.Xd[:, STEP_COL])).astype(int)
        det = three(sp.yd, pd, st, det_tasks(sp))
        pf = fm.predict_proba(sp.X)[:, 1]
        fc = three(sp.y, pf, sp.step, sp.task)
        views[nm] = {"detection": det, "forecast": fc,
                     "dreams": int(len(sp.win_onset)),
                     "frac_broken": r(np.mean(~np.isnan(sp.win_onset))),
                     "broken_at_step_1": r(np.mean(sp.win_onset == 1))}
    site["views"] = views

    # ------------------------------------------------ reliability and warning time (test split)
    pf = fm.predict_proba(te.X)[:, 1]
    from th.hazard import reliability, warning_times
    site["reliability"] = reliability(te.y, pf)
    site["warning_times"] = warning_times(te, pf)

    # ------------------------------------------------ Kaplan-Meier trust horizon
    def km_block(sp):
        per = {}
        for name, on in sp.onsets.items():
            S, med = km_curve(on, sp.horizons[name])
            per[name] = {"S": S, "median": med, "n": int(len(on)), "frac_broken": r(np.mean(~np.isnan(on)))}
        # pooled over all tasks; a dream that never broke is censored at its own task's horizon
        on = np.concatenate([sp.onsets[n] for n in sp.onsets])
        hz = np.concatenate([[sp.horizons[n]] * len(sp.onsets[n]) for n in sp.onsets])
        S, s = [], 1.0
        for t in range(1, 97):
            at_risk = np.sum((~np.isnan(on) & (on >= t)) | (np.isnan(on) & (hz >= t)))
            if at_risk > 0:
                s *= 1 - np.sum(on == t) / at_risk
            S.append(s)
        S = np.array(S)
        return {"tasks": per, "pooled": {"S": [r(v) for v in S], "median": median_survival(S)}}
    site["survival"] = {"seen": km_block(te), "unseen": km_block(un)}

    # ------------------------------------------------ the four forecast tests (report written by forecast_tests.py)
    rep = json.loads((ROOT / "runs" / "forecast_tests" / "error_tau0.5" / "report.json").read_text())
    site["forecast_tests"] = rep

    # exploratory T2 check: latent error aligned at the break (as in the explanatory doc)
    site["t2_latent_error_fit"] = {
        "offsets": list(range(-10, 5)),
        "latent_error": [0.18, 0.19, 0.23, 0.26, 0.34, 0.38, 0.39, 0.44, 0.55, 0.67, 0.91, 1.02, 1.10, 1.13, 1.14],
        "u_r": [0.15, 0.19, 0.20, 0.17, 0.28, 0.36, 0.37, 0.35, 0.48, 0.51, 0.57, 0.58, 0.59, 0.59, 0.59],
        "u_f": [0.24, 0.28, 0.23, 0.21, 0.37, 0.43, 0.35, 0.38, 0.54, 0.57, 0.66, 0.62, 0.61, 0.59, 0.61],
        "u_s": [0.17, 0.23, 0.25, 0.25, 0.41, 0.46, 0.37, 0.42, 0.56, 0.56, 0.65, 0.59, 0.58, 0.62, 0.58],
        "n_dreams": 135,
    }

    # ------------------------------------------------ Phase 3 (distance) and Phase 2 / 4 headline reports
    p3 = json.loads((ROOT / "runs" / "phase3" / "error_tau0.5" / "report.json").read_text())
    site["phase3"] = {"bins": p3["bins"], "by_source": p3["by_source"],
                      "hazard_log_distance_coef": p3["hazard_log_distance_coef"],
                      "dists": [r(d["dist"]) for d in p3["dreams"]]}
    p4 = json.loads((ROOT / "runs" / "phase4" / "report.json").read_text())
    site["phase4"] = {lab: {s: {k: v for k, v in p4["labels"][lab][s].items() if k != "trust_horizon"}
                            for s in ("seen_test", "unseen")} for lab in p4["labels"]}
    p2 = json.loads((ROOT / "runs" / "phase2" / "error_tau0.5" / "report.json").read_text())
    site["phase2"] = {"preregistered_checks": p2["preregistered_checks"], "models": p2["models"]}
    rep1 = json.loads((P1 / "repro" / "repro_report.json").read_text())
    site["repro_keys"] = list(rep1.keys())[:20]

    # ------------------------------------------------ showcased dream per task, with model probabilities
    dreams = {}
    for split, sp in (("seen", te), ("unseen", un)):
        for f in sorted((P1 / ("test" if split == "seen" else "unseen")).glob("*.npz")):
            name = f.stem
            with np.load(f) as z:
                d = {kk: z[kk] for kk in z.files}
            if "thumb_dream" not in d:
                continue
            F = sp.feats[name][0]               # window 0 is the one with thumbnails
            T = F.shape[0]
            m = max(float(d["motion"].mean()), 1e-6)
            c = sp.chance[name]
            on = sp.onsets[name][0]
            dreams[name] = {
                "split": split,
                "T": int(T),
                "onset": None if np.isnan(on) else int(on),
                "err": [r(v, 3) for v in d["lat_err"][0] / c],          # in units of chance error
                "u_r": [r(v, 3) for v in d["u_r"][0] / m],
                "u_f": [r(v, 3) for v in d["u_f"][0] / m],
                "u_s": [r(v, 3) for v in d["u_s"][0] / m],
                "p_forecast": [r(v, 3) for v in fm.predict_proba(F)[:, 1]],
                "p_detect": [r(v, 3) for v in dm.predict_proba(F)[:, 1]],
                "dpsnr": [r(v, 2) for v in sp.dpsnr[name][0]],
                "median_horizon": site["survival"][split]["tasks"][name]["median"],
            }
            for kind in ("real", "dream"):
                sprite(d[f"thumb_{kind}"][0][:T]).save(FRAMES / f"{name}_{kind}.webp", quality=82, method=6)

    (OUT / "site.json").write_text(json.dumps(site, separators=(",", ":")))
    (OUT / "dreams.json").write_text(json.dumps(dreams, separators=(",", ":")))
    print("views", json.dumps(views, indent=1))
    print("dreams", len(dreams), "warning median", np.median(site["warning_times"]))


if __name__ == "__main__":
    main()
