# Protocol decisions

Decisions about how results are measured, recorded before the data they apply to.
Git history timestamps each one.

## Before Phase 1 data (2026-11-23)

Based on the Phase 0 smoke test (12 dreams, 3 tasks):

1. **Failure rule** (`th/labels.py`). A dream breaks when the centered 5-step median of
   dPSNR (steps t-2 to t+2) stays at or below 0 for 5 steps in a row. Onset is the first step
   of that run. Dreams that never break are censored at their horizon. A median, because one
   extreme frame can pull a mean below 0. Centered, because a trailing median detects breaks
   about 2 steps late, which gave fake warning time in a synthetic test.
2. **Weak-baseline tasks.** If the median dream beats the frozen-frame baseline by less than
   1 dB over its first 8 steps, the task is reported but left out of the failure-time models.
3. **Tasks.** 20 seen tasks over all 10 domains (`phase1.py`, `SEEN_TASKS`), chosen before
   seeing their dreams. Horizon is up to 96 steps, shorter where episodes are shorter.
4. **Splits.** Hazard model fit on `val` windows, evaluated on the held-out `test` split of the
   same seen tasks. The 10 unseen tasks (`active_test`) are collected now but sealed: only
   file hashes are recorded until Phase 4.
5. **Reproduction check.** Paper protocol (8 + 16 frames), signals divided by the task's mean
   latent motion, Spearman against sequence dPSNR, pooled and within task.
6. **u_s noise.** Keep 4 seeds; smooth signals over steps in Phase 2 if needed.
7. **Model: `base`, not `combined`.** The checkpoint README says `combined` was finetuned with the
   targeted data of all 210 tasks, including the 10 held-out ones, so for `combined` the unseen
   tasks are not unseen. `base` was pretrained on the 200 seen tasks only. Phase 0 used `combined`
   (pipeline check only); all Phase 1 to 4 results use `base`.

## After the first repro run, before any long dreams (2026-11-23)

The first `repro` run (20 tasks, 640 sequences) gave pooled Spearman 0.26 to 0.42 and about 0 within
tasks, against the paper's 0.8. Checking our setup line by line against the paper's own evaluation
code (`collect_data.py`, `uncertainty.py`, `train_dynamics.py`) found that we had copied settings
from the interactive demo instead. Fixed before collecting any long dreams:

8. **Precision bfloat16** (was float16). The model was trained in bfloat16, and the paper's
   evaluation sampler uses bfloat16.
9. **4 denoising steps** (`eval_d = 0.25`, was 0.125 with 8 steps), as in the paper's evaluation code.
   Flow instability is measured across these steps.
10. **u_r on the mean of the seeds' predictions**, as in the paper's `URNormScorer` (was seed 0).
    The seed-0 version is still stored as `u_r_s0`. The dream itself still follows seed 0.
11. **Noise-floored dPSNR**: `10 log10((mse_floor + c) / (mse + c))` with c = 1e-4 (40 dB). Without
    it, scenes that stay perfectly still give dPSNR = -inf (3 of 20 tasks in the first run), and a
    perfect dream of a still scene would count as broken. Analysis recomputes this from stored
    errors, so it also applies to the first run.

12. **Signals from 2 seeds**, as the paper (`n_samples_unc = 2`): `u_s` is the variance over seeds
    0 and 1, and `u_r` uses the mean of seeds 0 and 1. We still sample 4 seeds and store the 4-seed
    variance as `u_s4`.
13. **Motion scale from real data.** The paper divides by "the per-step RMS change of the latent
    representation, computed as a per-task average over the dataset". We store the real latent
    change (`motion_real`) and use its task average; the predicted-motion version is reported too.
14. **Reproduction on the `test` split**, as the paper ("held-out" test sequences), 32 windows per task.
15. **What the paper does not state, so we report every option:** how per-step signals become one
    number per sequence (we report mean, max and last step), and the pooling (we report pooled over
    all sequences as in its Figure 5, at task level, and within task).

Checked against upstream at commit 3dda6ea and matching: 8 context + 16 dreamed frames
(`eval_ctx`, `eval_horizon`), shortcut schedule, `tau_ctx` 0.1, single-sample autoregressive dreams,
delta PSNR from the mean pixel MSE over the horizon against a repeated last context frame
(`run_dynamics_eval`), u_f as the mean step-to-step RMS change of the predicted clean latent over
the second half of the denoising steps, base checkpoint.

The first run is kept as `runs/phase1/repro_fp16` for comparison. `repro` is re-run with all fixes.
