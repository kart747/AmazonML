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

        # experiment scripts (src/experiments/*) on the same synthetic work dir
        exp = SRC / "experiments"

        def run_exp(script, *args, extra_env=None):
            r = subprocess.run([sys.executable, str(exp / script), *args], env={**env, **(extra_env or {})},
                               capture_output=True, text=True)
            if r.returncode != 0:
                print(r.stdout[-4000:], r.stderr[-4000:])
                raise RuntimeError(f"{script} failed")
            return r.stdout

        run_exp("features_v2.py")
        fx = {"BER_EXTRA": "featx", "BER_TAG": "_fx"}
        r = subprocess.run([sys.executable, str(SRC / "run.py"), "evaluate", "--approach", "1"],
                           env={**env, **fx}, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-3000:]
        assert (tmp / "work" / "reports" / "eval_A1_fx.json").exists()
        print(run_exp("decoder_lab.py", "--approach", "2")[-1500:])
        run_exp("density.py", "--approach", "2")
        assert (tmp / "work" / "reports" / "density_A2.json").exists()
        run_exp("error_analysis.py", "--approach", "2")
        assert (tmp / "work" / "reports" / "errors_A2.md").exists()
        run_exp("loco.py", "--features", "context")
        assert (tmp / "work" / "reports" / "loco_context.json").exists()


if __name__ == "__main__":
    test_end_to_end()
    print("SMOKE TEST PASSED")
