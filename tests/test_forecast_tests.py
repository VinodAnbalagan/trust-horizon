"""Known-answer check for forecast_tests.py.

Planted: signals ramp up LEAD=6 steps before every break, so T1-T4 must find skill and T2 must
claim a lead. Null: the same data with each task's signal rows shuffled across dreams, so signals
carry no information about that dream's break: AUROCs near 0.5 and no lead claimed.
"""
import os, subprocess, sys, json, shutil
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
import make_synthetic_phase1 as syn


def build(root, shuffle, level=False):
    rng = np.random.default_rng(1)
    for split, n in (("fit", 16), ("test", 8)):
        d = Path(root) / split; d.mkdir(parents=True, exist_ok=True)
        for t in range(8):
            task = syn.make_task(rng, n, 96)
            if level:   # every dream's signals carry a constant offset: earlier break, higher signal
                on = syn_onsets(task)
                for s in ("u_r", "u_f", "u_s"):
                    task[s] = task[s] * np.exp(2.0 * np.exp(-np.nan_to_num(on, nan=200.0)[:, None] / 30))
            if shuffle:
                for s in ("u_r", "u_f", "u_s"):
                    task[s] = task[s][rng.permutation(n)]
            np.savez_compressed(d / f"task{t}.npz", **task)


def syn_onsets(task):
    from th.labels import chance_error, error_onsets
    return error_onsets(task["lat_err"], chance_error(task["z_start"]), 0.5)


def run(root):
    out = Path(root) / "out"
    subprocess.run([sys.executable, str(ROOT / "forecast_tests.py"), "--phase1", str(root), "--out", str(out)],
                   check=True, capture_output=True)
    return json.loads((out / "report.json").read_text())


def main():
    res = {}
    for name, shuffle, level in (("planted", False, False), ("null", True, False), ("level", False, True),
                                 ("level_null", True, True)):
        root = Path(f"/tmp/ft_{name}"); shutil.rmtree(root, ignore_errors=True)
        build(root, shuffle, level); res[name] = run(root)
    P, N = res["planted"], res["null"]
    ok = True
    def check(cond, msg):
        nonlocal ok; ok &= bool(cond); print(("PASS " if cond else "FAIL ") + msg)
    t1p, t1n = P["T1_window_sweep"]["8"]["test"], N["T1_window_sweep"]["8"]["test"]
    check(t1p["beyond_time"] > 0.7, f"T1 planted k=8 beyond-time AUROC high ({t1p['beyond_time']})")
    check(abs(t1n["beyond_time"] - 0.5) < 0.1, f"T1 null k=8 beyond-time AUROC near 0.5 ({t1n['beyond_time']})")
    check(P["T2_aligned"]["test"]["signals"]["u_r"]["lead_claimed"], "T2 planted: lead claimed for u_r")
    lp = P["T2_aligned"]["test"]["signals"]["u_r"]["offsets_ci_excludes_0_at_or_before_-2"]
    check(min(lp) >= -7, f"T2 planted: lead not earlier than the planted ramp ({lp})")
    check(not any(v["lead_claimed"] for v in N["T2_aligned"]["test"]["signals"].values()), "T2 null: no lead claimed")
    a4 = P["T3_landmark"]["24"]["test"]["per_u"]["4"]["auroc"]
    check(a4 > 0.9, f"T3 ramp data, s=24: breaks within 4 steps are seen coming ({a4})")
    L, LN = res["level"]["T3_landmark"], res["level_null"]["T3_landmark"]
    for s_ in ("1", "8", "24"):
        cl, cln = L[s_]["test"]["c_index"], LN[s_]["test"]["c_index"]
        check(cl > 0.7, f"T3 level data s={s_}: C-index high ({cl})")
        check(abs(cln - 0.5) < 0.12, f"T3 level null s={s_}: C-index near 0.5 ({cln})")
        g = L[s_]["test"]["u16_max_calibration_gap"]
        print(f"     calibration gap s={s_}: {g}, predicted {L[s_]['test']['u16_mean_predicted']} vs observed {L[s_]['test']['u16_observed']}")
    t4 = P["T4_trees"]["test"]["boosted trees (+ last 8 levels)"]
    check(t4["beyond_time"] > 0.7, f"T4 planted trees beyond-time AUROC high ({t4['beyond_time']})")
    t4n = N["T4_trees"]["test"]["boosted trees (+ last 8 levels)"]
    check(abs(t4n["beyond_time"] - 0.5) < 0.1, f"T4 null trees beyond-time AUROC near 0.5 ({t4n['beyond_time']})")
    print("ALL PASS" if ok else "SOME FAILED"); sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
