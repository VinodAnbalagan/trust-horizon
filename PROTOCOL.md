# Protocol decisions

Decisions about how results are measured, recorded before the data they apply to.
Git history timestamps each one.

## Before Phase 1 data (2026-11-23)

Based on the Phase 0 smoke test (12 dreams, 3 tasks):

1. **Failure rule** (`th/labels.py`). A dream breaks when the trailing 5-step median of
   dPSNR stays at or below 0 for 5 steps in a row. Onset is the first step of that run.
   Dreams that never break are censored at their horizon.
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
