# Trust Horizon: When Do World Model Dreams Break?

World models can imagine futures that look realistic but drift from reality.
[Hansen and Wang (2026)](https://www.nicklashansen.com/mmbench2) showed that three internal
signals can detect these hallucinations. This project turns those signals into a
**probability forecast**: how likely a dream is to break within the next few steps, and how
long it can be trusted.

Questions:
1. Can the signals give a calibrated probability that a dream breaks soon, and with how much warning?
2. How does the trust horizon shrink as a dream starts further from the training data?
3. Does it hold on the 10 unseen tasks (locked test set, opened once)?

## Quick start (RunPod)

```
git clone https://github.com/VinodAnbalagan/trust-horizon.git
cd trust-horizon
bash scripts/setup_runpod.sh
python smoke.py --usd-per-hour <pod price>
```

Each run prints its time and estimated cost, and logs it to `runs/ledger.csv`.

## Layout

```
smoke.py           Phase 0 smoke test
th/                rollouts, data loading, sampler, timing
scripts/           pod setup
tests/             CPU pipeline test
PREREGISTRATION.md guesses made before seeing data
```

## Credits

Built on the [MMBench2 code, checkpoints and dataset](https://github.com/nicklashansen/mmbench2)
by Nicklas Hansen and Xiaolong Wang (MIT and CC-BY-4.0). `th/sampler.py` adapts their sampler
to return flow instability per sample.
