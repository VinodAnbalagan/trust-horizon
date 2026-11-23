"""Unit tests for the pre-registered failure rule.   python tests/test_labels.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from th.labels import failure_onset, onsets, smooth_dpsnr, weak_baseline  # noqa: E402


def main():
    # centered median
    assert np.allclose(smooth_dpsnr(np.array([2.0, 4.0, 6.0, -50.0, 6.0]), 3)[:4], [3.0, 4.0, 4.0, 6.0])

    # a single dip is not a failure
    row = np.full(40, 3.0)
    row[10] = -20.0
    assert failure_onset(row) is None, "one bad frame must not count"
    row[10:12] = -20.0
    assert failure_onset(row) is None, "two bad frames must not count either"

    # a sustained drop is; onset is the first step of the 5-step run (1-based)
    row = np.concatenate([np.full(20, 3.0), np.full(20, -3.0)])
    o = failure_onset(row)
    # drop starts at index 20 (step 21); a centered median sees it with no lag
    assert o == 21, o

    # never breaking is censored
    assert failure_onset(np.full(30, 1.0)) is None
    assert np.isnan(onsets(np.full((2, 30), 1.0))).all()

    # bad from the very start: onset at step 1
    assert failure_onset(np.full(30, -1.0)) == 1

    # weak baseline: early dPSNR below 1 dB
    assert weak_baseline(np.full((4, 30), 0.3))
    assert not weak_baseline(np.full((4, 30), 4.0))
    print("label checks passed")


if __name__ == "__main__":
    main()
