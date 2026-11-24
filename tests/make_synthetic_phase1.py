"""Synthetic Phase 1 outputs with a KNOWN answer, to test Phase 2 end to end.

Each dream breaks at a random step (or never). The signals start rising LEAD steps
before the break, so a working pipeline should find a median warning close to LEAD,
a high AUROC, and decent calibration. One task has a weak baseline (dPSNR near 0).

    python tests/make_synthetic_phase1.py /tmp/syn && python phase2.py \
        --fit-dir /tmp/syn/fit --test-dir /tmp/syn/test --out /tmp/syn/phase2
"""
import sys
from pathlib import Path

import numpy as np

import os
LEAD = int(os.environ.get("LEAD", 6))


def make_task(rng, n, T, weak=False, p_fail=0.7):
    dps = np.zeros((n, T)); sig = {s: np.zeros((n, T)) for s in ("u_r", "u_f", "u_s")}
    lat = np.zeros((n, T)); z_start = rng.normal(0, 1, (n, 64))        # chance error about 1.41
    for i in range(n):
        onset = rng.integers(15, T - 5) if rng.random() < p_fail else None
        base = 0.3 if weak else 4.0
        d = base + rng.normal(0, 1.0, T)
        if onset is not None and not weak:
            d[onset - 1:] = -3 + rng.normal(0, 1.0, T - onset + 1)
        dps[i] = d
        e = 0.1 + 0.02 * rng.normal(size=T)
        if onset is not None and not weak:
            e[onset - 1:] = 1.2 + 0.05 * rng.normal(size=T - onset + 1)     # above 0.5 x chance
        lat[i] = e
        for k, s in enumerate(sig):
            x = np.exp(rng.normal(0, 0.25, T))
            if onset is not None and not weak:
                ramp = np.clip((np.arange(1, T + 1) - (onset - LEAD)) / LEAD, 0, 1)
                x *= 1 + (2.0 - 0.5 * k) * ramp
            sig[s][i] = x
    motion = np.ones((n, T))
    mse_floor = np.full((n, T), 0.01)
    mse = mse_floor / 10 ** (dps / 10)
    return dict(dpsnr=dps.astype(np.float32), motion=motion.astype(np.float32), mse=mse.astype(np.float32),
                lat_err=lat.astype(np.float32), z_start=z_start.astype(np.float16),
                mse_floor=mse_floor.astype(np.float32), **{k: v.astype(np.float32) for k, v in sig.items()})


def main(root):
    rng = np.random.default_rng(0)
    for split, n in (("fit", 16), ("test", 8)):
        d = Path(root) / split
        d.mkdir(parents=True, exist_ok=True)
        for t in range(8):
            np.savez_compressed(d / f"task{t}.npz", **make_task(rng, n, 96))
        np.savez_compressed(d / "weaktask.npz", **make_task(rng, n, 96, weak=True))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/syn")
