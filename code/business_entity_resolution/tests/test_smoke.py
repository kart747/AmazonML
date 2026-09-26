"""End-to-end smoke test on synthetic data (≈1–2 min, 4 threads, <2 GB RAM).

    python tests/test_smoke.py            # or: python -m pytest -q tests/

Runs prepare → block → features → evaluate (A1, A2) → predict (A2) in a temporary
directory, then checks that the official validator passes and the local scores are sane.
Needs the official validator (BER_VALIDATOR or the default student_resource path).
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
sys.path.insert(0, str(HERE))
from synth_data import make  # noqa: E402


def run(stage, env, *args):
    r = subprocess.run([sys.executable, str(SRC / "run.py"), stage, *args], env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-4000:], r.stderr[-4000:])
        raise RuntimeError(f"stage {stage} failed")
    return r.stdout


def test_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        make(tmp / "dataset")
        env = dict(os.environ, BER_DATA_DIR=str(tmp / "dataset"), BER_WORK_DIR=str(tmp / "work"),
                   BER_OUT_DIR=str(tmp / "output"), BER_THREADS="4")
        for stage, args in (("prepare", ()), ("block", ()), ("features", ()),
                            ("evaluate", ("--approach", "1")), ("evaluate", ("--approach", "2")),
                            ("predict", ("--approach", "2"))):
            out = run(stage, env, *args)
        assert "validator exit 0" in out, out[-3000:]
        for a in (1, 2):
            rep = json.load(open(tmp / "work" / "reports" / f"eval_A{a}.json"))
            f = rep["eval"]["macro_f05"]
            print(f"A{a}: macro F0.5 {f}, blocking oracle {rep['eval_blocking_oracle']['macro_f05']}")
            assert f > 0.5, rep["eval"]
        m = (tmp / "output" / "matching_results.tsv").read_text().splitlines()
        assert m[0] == "source1_entity_id\tmatched_entity_ids" and len(m) == 1501


if __name__ == "__main__":
    test_end_to_end()
    print("SMOKE TEST PASSED")
