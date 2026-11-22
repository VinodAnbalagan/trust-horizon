# Pre-registration

Write this before looking at any rollout data (the Phase 0 smoke test output counts as data,
so fill this in first). Commit it, and do not edit it afterwards. Add corrections below the
line at the end instead.

Date written:22-November-2026
Written by: Vinod

## Guesses

For each, state the claim and how sure you are (a probability). The suggestions in brackets
are only prompts; replace them with your own numbers.

1. **Combined forecast vs single signal.** A hazard model using all three signals has a lower
   Brier score than the best single signal with a fixed threshold.
   Confidence: [70%]

2. **Calibration.** On held-out seen-task windows, the forecast's reliability curve stays within
   0.1 of the diagonal in every bin with at least 50 samples.
   Confidence: [55%]

3. **Warning time.** The median warning (steps between the forecast first exceeding 0.5 and the
   actual failure onset) is at least 3 steps.
   Confidence: [40%]

4. **Trust horizon vs distance.** Median time to failure decreases monotonically across distance
   bins (kNN distance from the start state to the training data).
   Confidence: [60%]

5. **Transfer.** On the unseen tasks, the forecast's AUROC for "fails within 8 steps" is within
   0.05 of its seen-task value.
   Confidence: [40%]

6. **Which signal matters most.** [u_s] gets the largest weight in the
   combined model.
   Confidence: [50%]

## What would change the plan

- If fewer than 20% of 96-step rollouts fail at all, extend the horizon or pick harder tasks.
- If we cannot reproduce Spearman of at least 0.6 on the 24-frame protocol, stop and debug the
  pipeline before modelling.

---
Corrections (dated, below this line only):
