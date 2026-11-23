"""End-to-end test of Phase 2 on synthetic dreams with a known answer.   python tests/test_analysis.py

LEAD=6: signals rise 6 steps before each break  -> the forecast must give real warning.
LEAD=1: signals jump at the break               -> the forecast must give NO warning
        (guards against labels or features leaking the future).
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(lead, d):
    env = {**os.environ, "LEAD": str(lead)}
    subprocess.run([sys.executable, str(ROOT / "tests" / "make_synthetic_phase1.py"), d], env=env, check=True)
    subprocess.run([sys.executable, str(ROOT / "phase2.py"), "--fit-dir", f"{d}/fit", "--test-dir", f"{d}/test",
                    "--out", f"{d}/p2"], check=True, capture_output=True)
    return json.loads(Path(f"{d}/p2/report.json").read_text())


def main():
    with tempfile.TemporaryDirectory() as d:
        r = run(6, d + "/lead6")
        c = r["preregistered_checks"]
        assert r["weak_tasks"] == ["weaktask"], r["weak_tasks"]
        assert c["3_median_warning_ge_3"]["median_warning"] >= 3, c["3_median_warning_ge_3"]
        assert c["1_combined_beats_best_threshold"]["result"]
        assert c["6_largest_weight"]["result"] == "u_r"          # planted strongest
        r = run(1, d + "/lead1")
        mw = r["preregistered_checks"]["3_median_warning_ge_3"]["median_warning"]
        assert mw == 0, f"lockstep signals must give no warning, got {mw}"
    print("analysis checks passed")


if __name__ == "__main__":
    main()
