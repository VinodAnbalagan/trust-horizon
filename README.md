# Trust Horizon: When Do World Model Dreams Break?

A probabilistic follow-up to *Hallucination in World Models is Predictable and Preventable*
(Hansen and Wang, 2026; [paper](https://arxiv.org/abs/2606.27326),
[interactive page](https://www.nicklashansen.com/mmbench2), [code](https://github.com/nicklashansen/mmbench2)).

That paper shows three label-free signals (tokenizer round-trip residual `u_r`, flow
instability `u_f`, inter-seed variance `u_s`) rank hallucinations well
(Spearman about 0.8, per-task AUROC 0.87 to 0.94). This project asks three follow-up questions:

1. **Forecast, not alarm.** Can the signals give a calibrated probability that a dream
   breaks within the next k steps, and how much warning does that give?
2. **Trust horizon vs distance from data.** How does the expected time to failure shrink
   as a rollout starts further from the training data? This is the quantity a method like
   Mind Dreamer (ICML 2026), which deliberately dreams from novel latent states, would need.
3. **Unseen tasks.** Do the signals and the forecast hold up on the 10 held-out tasks?
   These are the locked test set and are opened once, in Phase 4.

Course: Stanford probability for AI, final project. Target submission **December 15, 2026**
(hard deadline December 21).

## Plan

Phases run one at a time. After each one we look at the time and cost it took and decide
whether the next phase is worth it.

| Phase | Work | Checkpoint question | GPU? |
|---|---|---|---|
| 0 | Smoke test on the real checkpoint | Does it work, and what does one dream step cost? | short |
| 1 | Long rollouts (about 96 steps) on seen tasks; unseen tasks generated sealed | Do we reproduce Spearman of about 0.8? Do failures happen within the horizon? | main cost |
| 2 | Discrete-time hazard model on the signals | Calibrated? Better than a single-signal threshold? How much warning? | no (CPU) |
| 3 | Survival curves by distance to training data | Does the trust horizon shrink with distance? | some |
| 4 | Open the locked unseen-task set once | Do results transfer? Compare with pre-registration | short |
| 5 | React app, project site, HF Space, thumbnail, video | Submit Dec 15 | no |

Phases 2 and 5 run on a laptop, so GPU spend is Phases 0, 1, 3 and 4.

## Time and cost tracking

Every GPU script takes `--usd-per-hour` (your pod's price) and:
- prints elapsed time and ETA while it runs,
- prints a time and cost breakdown per stage at the end,
- appends one row to `runs/ledger.csv`, and prints the running total across all runs,
- for Phase 0, projects the time and cost of small, medium and large Phase 1 runs.

The ledger counts script time only. The pod bills while it is running, idle or not, so stop
it when a run finishes.

## Running on RunPod

1. **Create a pod.** PyTorch template (torch 2.4 or newer). One GPU with 24 GB or more is
   plenty for a 350M model, for example an RTX 4090, RTX A5000 or L4. Container or volume disk
   of at least 30 GB. Note the hourly price shown on the pod page.
2. **Get the code** (in the pod's terminal or Jupyter terminal):
   ```
   cd /workspace
   git clone https://github.com/<your-username>/trust-horizon.git
   cd trust-horizon
   bash scripts/setup_runpod.sh
   ```
3. **Smoke test.** Start small, then the full run:
   ```
   python smoke.py --tasks walker-walk --n-windows 2 --horizon 32 --usd-per-hour <price>
   python smoke.py --usd-per-hour <price>
   ```
4. **Bring results back.** Commit the small files (summary, plots, ledger), or download
   `runs/` from the Jupyter file browser:
   ```
   git add -f runs/ledger.csv runs/*/summary.json runs/*/*.png
   git commit -m "Phase 0 results" && git push
   ```
5. **Stop the pod.**

The first run downloads the checkpoint (about 1.4 GB) into `/workspace/hf_cache`, which
survives pod restarts if you use a network volume.

## Layout

```
smoke.py              Phase 0: smoke test with timing, cost and Phase 1 projections
th/sampler.py         batched next-latent sampler, u_f per sample (adapted from upstream, MIT)
th/data.py            load raw MMBench2 trajectories (png strips + .pt), windows inside one episode
th/rollout.py         open-loop dreams with per-step u_r, u_f, u_s, motion, and true error
th/timing.py          stage timing, cost estimates, run ledger
scripts/setup_runpod.sh   one-time pod setup
tests/test_pipeline.py    CPU test with tiny random models and synthetic data
PREREGISTRATION.md    guesses written before seeing data
third_party/          upstream code, cloned by the setup script (git-ignored)
runs/                 outputs (git-ignored except what you add on purpose)
```

## Rules

- The unseen tasks stay sealed until Phase 4. No tuning on them.
- `PREREGISTRATION.md` is committed before looking at any rollout data. The git timestamp
  is the proof.
- Every result is reproducible from one command.

## Design decisions so far

- **Ground truth from recorded trajectories, not live simulators.** The `val` split has
  real frames and actions for every seen task, and `active_test` covers the unseen tasks.
  This avoids installing ten simulator stacks.
- **Paper protocol for the sanity check:** 8 context frames, 16 dreamed frames, shortcut
  schedule with d = 0.125 (K = 8), context corruption tau_ctx = 0.1. Long rollouts slide a
  context window of 24 latents, as the live demo does.
- **The dream follows seed 0.** Other seeds only feed `u_s`, so the dream is a real sample,
  not a blurry average.
- **Motion normalization.** The paper divides each signal by the task's average latent motion.
  We log raw signals and motion per step, and report both the paper's task-average
  normalization and a per-step one.
- **Per-step failure signal:** `dpsnr = 10 log10(mse_floor / mse)`, where the floor repeats
  the last context frame. `dpsnr <= 0` is the paper's scene-divergence criterion, applied per
  step. The exact failure-onset rule is fixed in Phase 1, before modelling.
- **One change to upstream code:** the sampler returns `u_f` per batch element instead of
  averaged over the batch, so many rollouts can share one forward pass.

Upstream code pinned at commit `3dda6ea`.
