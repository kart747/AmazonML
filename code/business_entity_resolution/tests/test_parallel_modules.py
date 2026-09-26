"""Smoke test for the parallel-track modules (xscript.py, redecode.py) on synthetic data.

    python tests/test_parallel_modules.py      # ≈2–3 min, 4 threads, <1.5 GB RAM

Builds a synthetic dataset (India copies partly in Devanagari), runs the base pipeline in
a temporary work dir, then:
  xscript build (train+test) → measure (must recover base misses) → merge into an
  isolated work dir → features + evaluate A1 there (base artifacts untouched);
  redecode diagnose → write a variant submission with a per-country tau (validator must pass).
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


def sh(args, env):
    r = subprocess.run([sys.executable, *map(str, args)], env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-4000:], r.stderr[-4000:])
        raise RuntimeError(f"failed: {' '.join(map(str, args))}")
    return r.stdout


def test_parallel_modules():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        make(tmp / "dataset")
        env = dict(os.environ, BER_DATA_DIR=str(tmp / "dataset"), BER_WORK_DIR=str(tmp / "work"),
                   BER_OUT_DIR=str(tmp / "output"), BER_THREADS="4")
        for stage, args in (("prepare", ()), ("block", ()), ("features", ()),
                            ("evaluate", ("--approach", "2")), ("predict", ("--approach", "2"))):
            sh([SRC / "run.py", stage, *args], env)
        base_files = sorted(p.name for p in (tmp / "work" / "block" / "train").iterdir())

        # ---- xscript: supplementary cross-script view
        xs = SRC / "experiments" / "xscript.py"
        sh([xs, "build", "--split", "train"], env)
        sh([xs, "build", "--split", "test"], env)
        sh([xs, "measure"], env)
        rep = json.load(open(tmp / "work" / "reports" / "xscript_recall.json"))
        print("xscript measure:", json.dumps(rep))
        assert "India" in rep and "US" not in rep, rep           # only non-Latin-script targets
        r = rep["India"]
        assert r["union_recall_by_k"]["8"] >= r["base_recall"], r
        dst = tmp / "work_xs"
        for split in ("train", "test"):
            sh([xs, "merge", "--split", split, "--dst", dst], env)
        env_xs = dict(env, BER_WORK_DIR=str(dst))
        sh([SRC / "run.py", "features", "--split", "train"], env_xs)
        sh([SRC / "run.py", "evaluate", "--approach", "1"], env_xs)
        ev = json.load(open(dst / "reports" / "eval_A1.json"))
        print("A1 on merged candidates:", ev["eval"]["macro_f05"], "oracle", ev["eval_blocking_oracle"]["macro_f05"])
        after = sorted(p.name for p in (tmp / "work" / "block" / "train").iterdir())
        assert set(base_files) <= set(after) and all(n.startswith("xs_") for n in set(after) - set(base_files)), \
            "xscript must only add xs_* files to the base block dir"

        # ---- redecode: diagnostics + a variant submission
        print(sh([SRC / "experiments" / "redecode.py", "diagnose", "--approach", "2"], env)[-1500:])
        out = tmp / "variant"
        o = sh([SRC / "experiments" / "redecode.py", "write", "--approach", "2", "--tau", "0.8",
                "--country-tau", "France=0.9", "--out", out], env)
        assert "validator exit 0" in o, o[-3000:]
        dec = json.load(open(out / "decision.json"))
        assert dec["country_tau"] == {"France": 0.9}, dec
        assert (tmp / "output" / "matching_results.tsv").exists(), "main output must be untouched"


if __name__ == "__main__":
    test_parallel_modules()
    print("PARALLEL MODULES SMOKE TEST PASSED")
