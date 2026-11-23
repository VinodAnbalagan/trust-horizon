"""Phase 3: does the trust horizon shrink as a dream starts further from the training data?

Five of our seen tasks also have trajectories collected AFTER the base model was trained
(random actions, no-op actions, a fresh expert). Their start states range from familiar to
unfamiliar. We dream from them, measure each start's distance to the training data, and
compare survival curves across distance bins.

    python phase3.py bank   --usd-per-hour 0.50    # GPU, short: encode training clips per task
    python phase3.py dreams --usd-per-hour 0.50    # GPU: long dreams from the new start states
    python phase3.py analyze                       # CPU: distances, survival by distance bin
    python phase3.py estimate

Uses Phase 1 `fit` dreams (expert, held out) as the in-distribution group.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
UPSTREAM = Path(os.environ.get("MMBENCH2", ROOT / "third_party" / "mmbench2")).resolve()
sys.path[:0] = [str(UPSTREAM / "src"), str(ROOT)]

import numpy as np  # noqa: E402

TASKS = ["cup-catch", "finger-turn-easy", "og-point-maze", "pygame-point-maze-var1", "pygame-pong"]
NEW_STARTS = ["active_random", "active_zero", "active_expert"]   # collected after base training
TRAIN_PARTS = ["expert", "mixed-small"]                          # part of base's training data
N_BANK = int(os.environ.get("TH_N_BANK", 2000))        # env overrides are for tests only
N_WINDOWS = int(os.environ.get("TH_N_WINDOWS", 16))
SEC_PER_WINDOW_STEP = 0.435


def estimate(price):
    meta = json.load(open(UPSTREAM / "tasks.json"))
    steps = sum(N_WINDOWS * min(96, int(meta[t]["max_episode_steps"]) - 8) for t in TASKS) * len(NEW_STARTS)
    h_dream = steps * SEC_PER_WINDOW_STEP / 3600 + len(TASKS) * len(NEW_STARTS) * 30 / 3600
    h_bank = len(TASKS) * 4 / 60   # about 4 minutes per task (download + encode)
    print(f"bank    {h_bank:.2f} h  ${h_bank * price:.2f}")
    print(f"dreams  {h_dream:.2f} h  ${h_dream * price:.2f}   ({steps} window-steps)")


def gpu_stage(stage, args):
    os.environ.setdefault("HF_HOME", "/workspace/hf_cache" if Path("/workspace").exists() else str(ROOT / ".hf_cache"))
    import torch
    from huggingface_hub import hf_hub_download

    from th.bank import encode_clips, save_bank
    from th.collect import collect_task
    from th.rollout import load_wm
    from th.timing import Clock, Progress, append_ledger, fmt_secs, save_json

    clock = Clock(args.usd_per_hour)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"
    tasks_json, cache = str(UPSTREAM / "tasks.json"), os.path.join(os.environ["HF_HOME"], "datasets")
    out_dir = ROOT / "runs" / "phase3" / stage
    out_dir.mkdir(parents=True, exist_ok=True)
    man_path = out_dir / "manifest.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {"stage": stage, "items": {}}

    jobs = [(t, None) for t in TASKS] if stage == "bank" else [(t, p) for t in TASKS for p in NEW_STARTS]
    todo = [(t, p) for t, p in jobs if not (out_dir / (f"{t}.npz" if p is None else f"{t}__{p}.npz")).exists()]
    print(f"phase3 {stage}: {len(jobs) - len(todo)} done, {len(todo)} to run on {gpu}")
    if not todo:
        return
    with clock.stage("model"):
        tok = hf_hub_download("nicklashansen/mmbench2-models", f"{args.variant}/tokenizer.pt")
        dyn = hf_hub_download("nicklashansen/mmbench2-models", f"{args.variant}/dynamics.pt")
        wm = load_wm(tok, dyn, device=dev)

    for i, (task, part) in enumerate(todo, 1):
        key = task if part is None else f"{task}__{part}"
        print(f"\n[{i}/{len(todo)}] {key}   (elapsed {fmt_secs(clock.elapsed())}, est. ${clock.cost():.2f})")
        try:
            with clock.stage(stage):
                if stage == "bank":
                    bank = encode_clips(wm, task, TRAIN_PARTS, tasks_json=tasks_json, cache_dir=cache, n_clips=N_BANK)
                    save_bank(out_dir / f"{task}.npz", task, bank)
                    meta = {"task": task, "n": int(len(bank)), "dim": int(bank.shape[1])}
                else:
                    meta = collect_task(wm, task, part, out_dir / f"{key}.npz", tasks_json=tasks_json, cache_dir=cache,
                                        ctx=8, horizon=96, n_windows=N_WINDOWS, n_seeds=4, tau_ctx=0.1, seed=0,
                                        n_thumb_windows=1, progress=Progress(key, every=32))
        except Exception as e:
            meta = {"error": f"{type(e).__name__}: {e}"}
        manifest["items"][key] = meta
        save_json(man_path, manifest)
        print(f"  {meta.get('error', 'saved')}")

    rep = clock.report()
    total_usd, total_h = append_ledger(ROOT / "runs", time.strftime(f"phase3-{stage}-%Y%m%d-%H%M%S"), "phase3", rep, gpu, stage)
    print(f"\n{stage} done in {rep['total_readable']}, est. ${rep['est_cost_usd']}. "
          f"All runs so far: {total_h:.2f} GPU-hours, est. ${total_usd:.2f}")


def analyze(args):
    from th.bank import knn_distance
    from th.hazard import kaplan_meier, median_survival
    from th.labels import onsets, weak_baseline

    bank_dir, dream_dir = ROOT / "runs" / "phase3" / "bank", ROOT / "runs" / "phase3" / "dreams"
    fit_dir = Path(args.fit_dir)
    rows = []   # one per dream: task, source, distance, onset, horizon
    for task in TASKS:
        bp = bank_dir / f"{task}.npz"
        if not bp.exists():
            print(f"  {task}: no bank, skipped"); continue
        bank = np.load(bp)["bank"]
        sources = [("val (expert, held out)", fit_dir / f"{task}.npz")] + \
                  [(p, dream_dir / f"{task}__{p}.npz") for p in NEW_STARTS]
        loaded = [(s, dict(np.load(f))) for s, f in sources if f.exists()]
        if not loaded:
            continue
        ref = next((d for s, d in loaded if s.startswith("val")), loaded[0][1])
        if weak_baseline(ref["dpsnr"]):
            print(f"  {task}: weak baseline, skipped"); continue
        dist = {s: knn_distance(d["z_start"].reshape(len(d["z_start"]), -1), bank) for s, d in loaded}
        scale = float(np.median(dist[loaded[0][0]]))   # in-distribution reference = 1.0
        for s, d in loaded:
            on = onsets(d["dpsnr"])
            for i in range(len(on)):
                rows.append({"task": task, "source": s, "dist": float(dist[s][i] / max(scale, 1e-9)),
                             "onset": None if np.isnan(on[i]) else int(on[i]), "horizon": int(d["dpsnr"].shape[1])})
    if not rows:
        raise SystemExit("no Phase 3 data found")

    dist = np.array([r["dist"] for r in rows])
    edges = np.quantile(dist, [0, 0.25, 0.5, 0.75, 1.0])
    bins = np.clip(np.digitize(dist, edges[1:-1]), 0, 3)
    H = min(r["horizon"] for r in rows)
    per_bin = []
    for b in range(4):
        on = np.array([np.nan if rows[i]["onset"] is None else min(rows[i]["onset"], H + 1)
                       for i in np.nonzero(bins == b)[0]], dtype=float)
        on[on > H] = np.nan
        S = kaplan_meier(on, H)
        per_bin.append({"bin": b, "dist_range": [round(float(edges[b]), 3), round(float(edges[b + 1]), 3)],
                        "n": int(len(on)), "frac_failed": round(float(np.mean(~np.isnan(on))), 3),
                        "median_trust_horizon": median_survival(S), "S": S.round(4).tolist()})
    meds = [pb["median_trust_horizon"] if pb["median_trust_horizon"] is not None else H + 1 for pb in per_bin]
    monotone = all(meds[i] >= meds[i + 1] for i in range(3)) and meds[0] > meds[3]

    # discrete-time hazard with log distance as covariate (task as fixed effect)
    from th.hazard import fit_logistic
    X, y = [], []
    tasks = sorted({r["task"] for r in rows})
    for r in rows:
        last = r["onset"] if r["onset"] is not None else r["horizon"]
        for t in range(1, last + 1):
            X.append([np.log(r["dist"]), np.log(t)] + [1.0 if r["task"] == tk else 0.0 for tk in tasks[1:]])
            y.append(int(r["onset"] == t))
    coef = None
    if len(set(y)) == 2:
        m = fit_logistic(np.array(X), np.array(y))
        coef = float(m.named_steps["logisticregression"].coef_[0][0] / m.named_steps["standardscaler"].scale_[0])

    out = ROOT / "runs" / "phase3"
    rep = {"bins": per_bin, "guess4_monotone_decrease": bool(monotone),
           "hazard_log_distance_coef": None if coef is None else round(coef, 3),
           "by_source": {s: {"n": int(sum(r["source"] == s for r in rows)),
                             "median_dist": round(float(np.median([r["dist"] for r in rows if r["source"] == s])), 3)}
                         for s in sorted({r["source"] for r in rows})},
           "dreams": rows}
    (out / "report.json").write_text(json.dumps(rep, indent=2))

    print("\n========== trust horizon vs distance from training data ==========")
    for s, v in rep["by_source"].items():
        print(f"  {s:<26} {v['n']:>4} dreams   median distance {v['median_dist']}")
    print()
    for pb in per_bin:
        print(f"  distance {pb['dist_range'][0]:>6} to {pb['dist_range'][1]:<7} n={pb['n']:<4} "
              f"failed {pb['frac_failed']:>5}   median trust horizon {pb['median_trust_horizon']}")
    print(f"\n4. median time to failure falls monotonically with distance: {monotone}   (you said 60%)")
    if coef is not None:
        print(f"   hazard per unit log distance: {coef:+.3f} (positive = farther breaks sooner)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#1f4e79", "#2e86c1", "#e67e22", "#c0392b"]
    for pb, c in zip(per_bin, colors):
        ax.step(np.arange(1, H + 1), pb["S"], where="post", color=c, lw=1.8,
                label=f"distance {pb['dist_range'][0]:.2f} to {pb['dist_range'][1]:.2f}")
    ax.axhline(0.5, color="grey", ls=":", lw=0.8)
    ax.set_xlabel("dream step"); ax.set_ylabel("fraction of dreams still trustworthy"); ax.set_ylim(0, 1.02)
    ax.set_title("Trust horizon by distance of the start state from training data", fontsize=10)
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "survival_by_distance.png", dpi=130)
    print(f"\nSaved {out / 'report.json'} and survival_by_distance.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["bank", "dreams", "analyze", "estimate"])
    ap.add_argument("--usd-per-hour", type=float, default=0.50)
    ap.add_argument("--variant", default="base")
    ap.add_argument("--fit-dir", default=str(ROOT / "runs" / "phase1" / "fit"))
    args = ap.parse_args()
    if args.stage == "estimate":
        estimate(args.usd_per_hour)
    elif args.stage == "analyze":
        analyze(args)
    else:
        gpu_stage(args.stage, args)


if __name__ == "__main__":
    main()
