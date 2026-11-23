"""Phase 1: collect dreams with per-step signals and errors.

Four stages, each resumable (tasks already saved are skipped):

  repro   8 context + 16 dreamed frames, 32 windows per seen task (test split, as the paper).
          Checks we reproduce the paper's correlation of about 0.8.
  fit     long dreams (up to 96 steps), 16 windows per seen task (val split).
          Training data for the Phase 2 hazard model.
  test    long dreams, 8 windows per seen task, from the held-out `test` split.
          Seen-task evaluation for Phase 2.
  unseen  long dreams, 16 windows per unseen task (active_test split). SEALED:
          nothing about these results is printed or analysed until Phase 4.
          Only file hashes are recorded, so the seal can be checked later.

Usage on the pod (run inside tmux or with nohup, see HANDOFF.md):
    python phase1.py repro  --usd-per-hour 0.50
    python phase1.py fit    --usd-per-hour 0.50
    python phase1.py test   --usd-per-hour 0.50
    python phase1.py unseen --usd-per-hour 0.50
    python phase1.py estimate                      # time and cost of each stage, no GPU needed

Outputs: runs/phase1/<stage>/<task>.npz, runs/phase1/<stage>/manifest.json,
and one row per invocation in runs/ledger.csv.
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

SEEN_TASKS = [
    # DMControl
    "cup-catch", "finger-turn-easy", "reacher-easy", "cheetah-run", "walker-walk",
    # Meta-World, ManiSkill, MuJoCo
    "mw-assembly", "mw-door-open", "ms-pick-cube", "ms-ant-walk", "mujoco-halfcheetah",
    # Box2D, RoboDesk, OGBench
    "bipedal-walker-uneven", "lunarlander-land", "rd-push-red", "og-point-maze", "og-antball",
    # MiniArcade, Atari
    "pygame-point-maze-var1", "pygame-pong", "pygame-coinrun", "atari-pong", "atari-ms-pacman",
]

STAGES = {
    #          partition      tasks      windows  horizon
    "repro":  ("test",        "seen",    32,      16),   # paper: held-out test sequences
    "fit":    ("val",         "seen",    16,      96),
    "test":   ("test",        "seen",    8,       96),
    "unseen": ("active_test", "unseen",  16,      96),
}
SEC_PER_WINDOW_STEP = 0.20    # first repro run on an RTX 4090 (8 denoising steps); 4 steps should be faster


def task_list(kind: str):
    from task_set import TASK_SET, UNSEEN_TASK_SET
    if kind == "seen":
        missing = [t for t in SEEN_TASKS if t not in TASK_SET]
        assert not missing, f"not in TASK_SET: {missing}"
        return list(SEEN_TASKS)
    return list(UNSEEN_TASK_SET)


def estimate(usd_per_hour: float):
    import json as _j
    meta = _j.load(open(UPSTREAM / "tasks.json"))
    total_h = 0.0
    print(f"{'stage':<8} {'tasks':>5} {'window-steps':>13} {'hours':>7} {'usd':>7}")
    for stage, (_, kind, nw, hz) in STAGES.items():
        steps = 0
        for t in task_list(kind):
            ep = int(meta.get(t, {}).get("max_episode_steps", 1000))
            steps += nw * max(16, min(hz, ep - 8))
        h = steps * SEC_PER_WINDOW_STEP / 3600 + len(task_list(kind)) * 40 / 3600
        total_h += h
        print(f"{stage:<8} {len(task_list(kind)):>5} {steps:>13} {h:>7.2f} {h * usd_per_hour:>7.2f}")
    print(f"{'total':<8} {'':>5} {'':>13} {total_h:>7.2f} {total_h * usd_per_hour:>7.2f}")
    print("Includes about 40 s per task for data download. Actual windows per task can be fewer than requested.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=list(STAGES) + ["estimate"])
    p.add_argument("--usd-per-hour", type=float, default=0.50)
    p.add_argument("--variant", default="base",
                   help="base = trained on the 200 seen tasks only (see PROTOCOL.md); combined also saw the unseen tasks")
    p.add_argument("--n-seeds", type=int, default=4)
    p.add_argument("--tau-ctx", type=float, default=0.1)
    p.add_argument("--batch-windows", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tasks", default="", help="comma list to override the stage's task list (debugging)")
    p.add_argument("--max-windows", type=int, default=0, help="cap windows per task (debugging)")
    args = p.parse_args()

    if args.stage == "estimate":
        estimate(args.usd_per_hour)
        return

    os.environ.setdefault("HF_HOME", "/workspace/hf_cache" if Path("/workspace").exists() else str(ROOT / ".hf_cache"))
    import torch
    from huggingface_hub import hf_hub_download

    from th.collect import collect_task
    from th.rollout import load_wm
    from th.timing import Clock, Progress, append_ledger, fmt_secs, save_json

    partition, kind, n_windows, horizon = STAGES[args.stage]
    if args.max_windows:
        n_windows = min(n_windows, args.max_windows)
    tasks = [t for t in args.tasks.split(",") if t] or task_list(kind)
    sealed = args.stage == "unseen"
    out_dir = ROOT / "runs" / "phase1" / args.stage
    out_dir.mkdir(parents=True, exist_ok=True)
    man_path = out_dir / "manifest.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {"stage": args.stage, "tasks": {}}

    clock = Clock(args.usd_per_hour)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"
    todo = [t for t in tasks if not (out_dir / f"{t}.npz").exists()]
    print(f"phase1 {args.stage}: {len(tasks)} tasks, {len(tasks) - len(todo)} already done, {len(todo)} to run on {gpu}")
    if not todo:
        return

    with clock.stage("model"):
        tok = hf_hub_download("nicklashansen/mmbench2-models", f"{args.variant}/tokenizer.pt")
        dyn = hf_hub_download("nicklashansen/mmbench2-models", f"{args.variant}/dynamics.pt")
        wm = load_wm(tok, dyn, device=dev)

    tasks_json = str(UPSTREAM / "tasks.json")
    cache = os.path.join(os.environ["HF_HOME"], "datasets")
    for i, task in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)}] {task}   (stage elapsed {fmt_secs(clock.elapsed())}, est. ${clock.cost():.2f})")
        try:
            with clock.stage("tasks"):
                meta = collect_task(
                    wm, task, partition, out_dir / f"{task}.npz",
                    tasks_json=tasks_json, cache_dir=cache, ctx=8, horizon=horizon, n_windows=n_windows,
                    n_seeds=args.n_seeds, tau_ctx=args.tau_ctx, seed=args.seed,
                    batch_windows=args.batch_windows, n_thumb_windows=0 if args.stage == "repro" else 1,
                    progress=None if sealed else Progress(task, every=32),
                )
        except Exception as e:  # one bad task must not end a paid run
            meta = {"task": task, "error": f"{type(e).__name__}: {e}"}
        manifest["tasks"][task] = meta
        save_json(man_path, manifest)   # save after every task
        if "error" in meta or "skipped" in meta:
            print(f"  {meta.get('error') or meta.get('skipped')}")
        else:
            print(f"  saved {meta['n_windows']} windows x {meta['horizon']} steps in {fmt_secs(meta['secs_dream'])}"
                  + ("  (sealed)" if sealed else ""))

    rep = clock.report()
    total_usd, total_h = append_ledger(ROOT / "runs", time.strftime(f"phase1-{args.stage}-%Y%m%d-%H%M%S"),
                                       "phase1", rep, gpu, args.stage)
    print(f"\n{args.stage} done in {rep['total_readable']}, est. ${rep['est_cost_usd']}. "
          f"All runs so far: {total_h:.2f} GPU-hours, est. ${total_usd:.2f}")

    if args.stage == "repro" and not args.tasks:
        from th.analysis import repro_report
        repro_report(out_dir)


if __name__ == "__main__":
    main()
