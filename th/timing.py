"""Wall-clock timing and GPU cost tracking.

Every run records how long each stage took and what it cost at your pod's
hourly price, and appends one line to runs/ledger.csv so the total spend
across all phases is always one file away.
"""
from __future__ import annotations

import csv
import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Optional


def fmt_secs(s: float) -> str:
    s = float(s)
    if s < 60:
        return f"{s:.1f}s"
    if s < 3600:
        return f"{s / 60:.1f}min"
    return f"{s / 3600:.2f}h"


class Clock:
    def __init__(self, usd_per_hour: float = 0.0):
        self.usd_per_hour = float(usd_per_hour)
        self.t0 = time.time()
        self.stages: Dict[str, float] = {}

    @contextmanager
    def stage(self, name: str):
        t = time.time()
        try:
            yield
        finally:
            self.stages[name] = self.stages.get(name, 0.0) + time.time() - t

    def elapsed(self) -> float:
        return time.time() - self.t0

    def cost(self, secs: Optional[float] = None) -> float:
        secs = self.elapsed() if secs is None else secs
        return secs / 3600.0 * self.usd_per_hour

    def report(self) -> Dict:
        total = self.elapsed()
        return {
            "total_secs": round(total, 1),
            "total_readable": fmt_secs(total),
            "usd_per_hour": self.usd_per_hour,
            "est_cost_usd": round(self.cost(total), 3),
            "stages_secs": {k: round(v, 1) for k, v in self.stages.items()},
        }


class Progress:
    """Prints elapsed time and ETA during a long loop."""

    def __init__(self, label: str, every: int = 8):
        self.label, self.every, self.t0 = label, every, time.time()

    def __call__(self, done: int, total: int):
        if done % self.every and done != total:
            return
        el = time.time() - self.t0
        eta = el / done * (total - done)
        print(f"  [{self.label}] step {done}/{total}  elapsed {fmt_secs(el)}  eta {fmt_secs(eta)}", flush=True)


def append_ledger(runs_dir: Path, run_id: str, phase: str, report: Dict, gpu: str, note: str = ""):
    """One row per run, so total spend is easy to track."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / "ledger.csv"
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["run_id", "phase", "gpu", "usd_per_hour", "total_secs", "est_cost_usd", "note"])
        w.writerow([run_id, phase, gpu, report["usd_per_hour"], report["total_secs"], report["est_cost_usd"], note])
    rows = list(csv.DictReader(path.open()))
    total_usd = sum(float(r["est_cost_usd"] or 0) for r in rows)
    total_h = sum(float(r["total_secs"] or 0) for r in rows) / 3600
    return total_usd, total_h


def save_json(path: Path, obj: Dict):
    path.write_text(json.dumps(obj, indent=2))
