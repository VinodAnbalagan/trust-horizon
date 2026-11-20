"""Phase 0 smoke test: does the pipeline work on the real model, and what does it cost?

Loads a released MMBench2 world model, dreams long open-loop rollouts on a few
validation trajectories, logs the three hallucination signals and the true error
at every step, and reports time, GPU memory and estimated cost.

    python smoke.py --usd-per-hour 0.69                       # full smoke test
    python smoke.py --tasks walker-walk --n-windows 2 --horizon 32 --usd-per-hour 0.69

Outputs in runs/<run_id>/:
    summary.json       timing, cost, memory, per-task sanity checks, Phase 1 projections
    rollouts.npz       per-step signals and errors, shape (n_windows, horizon) per task
    signals.png        signals vs error over the horizon, one row per window
    strip_<task>.png   real (top) vs dream (bottom), every 8th dreamed frame
and one row appended to runs/ledger.csv (running total of GPU spend).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
UPSTREAM = Path(os.environ.get("MMBENCH2", ROOT / "third_party" / "mmbench2")).resolve()
sys.path[:0] = [str(UPSTREAM / "src"), str(ROOT)]
os.environ.setdefault("HF_HOME", "/workspace/hf_cache" if Path("/workspace").exists() else str(ROOT / ".hf_cache"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from th.timing import Clock, Progress, append_ledger, fmt_secs, save_json  # noqa: E402

DEFAULT_TASKS = "walker-walk,cup-catch,og-point-maze,pygame-point-maze-var1"


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="combined", choices=["base", "coverage_aware", "combined"])
    p.add_argument("--tasks", default=DEFAULT_TASKS)
    p.add_argument("--partition", default="val")
    p.add_argument("--n-windows", type=int, default=4)
    p.add_argument("--ctx", type=int, default=8)
    p.add_argument("--horizon", type=int, default=96)
    p.add_argument("--n-seeds", type=int, default=4)
    p.add_argument("--tau-ctx", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--usd-per-hour", type=float, default=0.0, help="your pod's price, for cost estimates")
    p.add_argument("--note", default="")
    return p.parse_args()


def spearman(a, b):
    from scipy.stats import spearmanr
    r = spearmanr(a, b).correlation
    return None if np.isnan(r) else round(float(r), 3)


def main():
    args = parse()
    from huggingface_hub import hf_hub_download

    from th.data import fetch_task_files, load_windows, task_lang_emb
    from th.rollout import dream_batch, load_wm

    tasks = [t for t in args.tasks.split(",") if t]
    clock = Clock(args.usd_per_hour)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda = dev.type == "cuda"
    gpu = torch.cuda.get_device_name(0) if cuda else "cpu"
    tasks_json = str(UPSTREAM / "tasks.json")
    run_id = time.strftime("smoke-%Y%m%d-%H%M%S")
    out_dir = ROOT / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"run {run_id} on {gpu}, HF cache {os.environ['HF_HOME']}")

    with clock.stage("checkpoint_download"):
        tok = hf_hub_download("nicklashansen/mmbench2-models", f"{args.variant}/tokenizer.pt")
        dyn = hf_hub_download("nicklashansen/mmbench2-models", f"{args.variant}/dynamics.pt")
    with clock.stage("model_load"):
        wm = load_wm(tok, dyn, device=dev)
    print(f"model ready after {fmt_secs(clock.elapsed())}")

    summary = {
        "run_id": run_id, "gpu": gpu, "variant": args.variant, "ctx": args.ctx, "horizon": args.horizon,
        "n_seeds": args.n_seeds, "tau_ctx": args.tau_ctx, "sched_K": wm.sched["K"], "tasks": {},
    }
    arrays, strips, step_rates = {}, {}, []

    for task in tasks:
        print(f"\n== {task}")
        try:
            with clock.stage("data"):
                paths = fetch_task_files(task, args.partition,
                                         cache_dir=os.path.join(os.environ["HF_HOME"], "datasets"))
                root = os.path.dirname(paths[0])
                wins = load_windows(task, args.partition, root, length=args.ctx + args.horizon,
                                    n_windows=args.n_windows, tasks_json=tasks_json, seed=args.seed)
        except Exception as e:  # one bad task should not waste the whole paid run
            summary["tasks"][task] = {"error": f"{type(e).__name__}: {e}"}
            print(f"  data error, skipping: {e}")
            continue
        if not wins:
            summary["tasks"][task] = {"skipped": f"no episode with >= {args.ctx + args.horizon} frames"}
            print("  skipped: episodes too short")
            continue

        frames = torch.stack([w.frames for w in wins])
        acts = torch.stack([w.actions for w in wins])
        mask = torch.stack([w.act_mask for w in wins])
        le = task_lang_emb(task, tasks_json, wm.lang_dim)
        lang = None if le is None else le.unsqueeze(0).expand(len(wins), -1)

        if cuda:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.time()
        with clock.stage("dream"):
            out = dream_batch(wm, frames, acts, mask, lang, ctx=args.ctx, n_seeds=args.n_seeds,
                              tau_ctx=args.tau_ctx, seed=args.seed, keep_frames=True,
                              progress=Progress(task, every=16))
            if cuda:
                torch.cuda.synchronize()
        t_dream = time.time() - t0
        sec_per_window_step = t_dream / (args.horizon * len(wins))
        step_rates.append(sec_per_window_step)

        dream = out.pop("dream_frames")
        for k, v in out.items():
            arrays[f"{task}/{k}"] = v
        arrays[f"{task}/starts"] = np.array([w.start for w in wins])

        # Sanity check against the paper's protocol (8 context + 16 dreamed frames):
        # sequence dPSNR vs each motion-normalized signal, Spearman over windows.
        # Two normalizations: per-step motion, and the task-average motion the paper uses.
        h16 = min(16, args.horizon)
        seq = 10 * np.log10(out["mse_floor"][:, :h16].mean(1) / out["mse"][:, :h16].mean(1))
        m_step = np.maximum(out["motion"][:, :h16], 1e-3)
        m_task = max(float(out["motion"].mean()), 1e-3)
        rho = {}
        if len(wins) >= 3:
            for sig in ("u_r", "u_f", "u_s"):
                rho[sig] = {"per_step_norm": spearman((out[sig][:, :h16] / m_step).mean(1), seq),
                            "task_norm": spearman(out[sig][:, :h16].mean(1) / m_task, seq)}
        first_bad = [int(np.argmax(d <= 0)) + 1 if (d <= 0).any() else None for d in out["dpsnr"]]

        summary["tasks"][task] = {
            "n_windows": len(wins),
            "dream_secs": round(t_dream, 1),
            "ms_per_dream_step_per_window": round(1000 * sec_per_window_step, 1),
            "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if cuda else None,
            "seq16_dpsnr": [round(float(x), 2) for x in seq],
            "spearman_signal_vs_seq16_dpsnr": rho,
            "first_step_dpsnr_le_0": first_bad,
            "mean_dpsnr_by_quarter": [round(float(q.mean()), 2) for q in np.array_split(out["dpsnr"], 4, axis=1)],
        }
        print(f"  dream {fmt_secs(t_dream)}, {1000 * sec_per_window_step:.0f} ms per step per window, "
              f"first dPSNR<=0 at {first_bad}")

        idx = list(range(0, args.horizon, 8))
        top = np.concatenate(list(frames[0, args.ctx:][idx].numpy()), axis=2)
        bot = np.concatenate(list(dream[0][idx]), axis=2)
        strips[task] = np.concatenate([top, bot], axis=1).transpose(1, 2, 0)

    # ---------- projections for Phase 1 ----------
    if step_rates:
        r = float(np.median(step_rates))  # seconds per (window x dream step), at this batch size
        plans = {
            "small: 10 tasks x 8 windows x 96 steps": 10 * 8 * 96,
            "medium: 20 tasks x 16 windows x 96 steps": 20 * 16 * 96,
            "large: 30 tasks x 32 windows x 96 steps": 30 * 32 * 96,
        }
        summary["phase1_projection"] = {
            "sec_per_window_step": round(r, 4),
            "note": "Measured at this run's batch size; bigger batches are usually faster per window, "
                    "so these are upper bounds. Add about 1 min per task for data download.",
            "plans": {k: {"hours": round(n * r / 3600, 2), "usd": round(n * r / 3600 * args.usd_per_hour, 2)}
                      for k, n in plans.items()},
        }

    # ---------- figures ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    done = [t for t in tasks if f"{t}/dpsnr" in arrays]
    rows = max(1, sum(arrays[f"{t}/dpsnr"].shape[0] for t in done))
    fig, axes = plt.subplots(rows, 1, figsize=(11, 2.2 * rows), squeeze=False)
    r_i = 0
    for t in done:
        for w in range(arrays[f"{t}/dpsnr"].shape[0]):
            ax = axes[r_i, 0]
            x = np.arange(1, args.horizon + 1)
            ax.plot(x, arrays[f"{t}/dpsnr"][w], color="black", lw=1.5, label="dPSNR (dB, >0 beats frozen frame)")
            ax.axhline(0, color="grey", lw=0.8, ls="--")
            ax2 = ax.twinx()
            mot = max(float(arrays[f"{t}/motion"].mean()), 1e-3)
            for sig, c in (("u_r", "tab:red"), ("u_f", "tab:blue"), ("u_s", "tab:green")):
                v = arrays[f"{t}/{sig}"][w] / mot
                ax2.plot(x, v / (np.median(v) + 1e-8), color=c, lw=0.9, alpha=0.8, label=f"{sig}_norm / median")
            ax.set_title(f"{t}  window {w}", fontsize=9, loc="left")
            if r_i == 0:
                ax.legend(loc="upper left", fontsize=7)
                ax2.legend(loc="upper right", fontsize=7)
            r_i += 1
    axes[-1, 0].set_xlabel("dream step")
    fig.tight_layout()
    fig.savefig(out_dir / "signals.png", dpi=110)
    for t, s in strips.items():
        Image.fromarray(s).save(out_dir / f"strip_{t}.png")
    np.savez_compressed(out_dir / "rollouts.npz", **{k.replace("/", "__"): v for k, v in arrays.items()})

    # ---------- time and cost ----------
    rep = clock.report()
    summary["timing"] = rep
    total_usd, total_h = append_ledger(ROOT / "runs", run_id, "phase0", rep, gpu, args.note)
    save_json(out_dir / "summary.json", summary)

    print("\n================ time and cost ================")
    for k, v in rep["stages_secs"].items():
        print(f"  {k:<22} {fmt_secs(v)}")
    print(f"  {'total':<22} {rep['total_readable']}   est. ${rep['est_cost_usd']} at ${args.usd_per_hour}/h")
    print(f"  all runs so far: {total_h:.2f} GPU-hours, est. ${total_usd:.2f}  (runs/ledger.csv)")
    if "phase1_projection" in summary:
        print("\n  Phase 1 projections (upper bounds):")
        for k, v in summary["phase1_projection"]["plans"].items():
            print(f"    {k:<44} {v['hours']} h   ${v['usd']}")
    print(f"\nSaved to {out_dir}")
    print("Reminder: the pod bills while it is running, even when idle. Stop it when you are done.")


if __name__ == "__main__":
    main()
