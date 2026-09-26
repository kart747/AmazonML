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
        assert r["union_recall"] >= r["base_recall"] and "xs" in r["union_recall_by_k_per_view"], r
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

        # ---- dense view (optional: needs BER_DENSE_PY = a python with torch + transformers)
        dpy = os.environ.get("BER_DENSE_PY")
        if dpy and Path(dpy).exists():
            r = subprocess.run([dpy, str(SRC / "experiments" / "dense_view.py"), "build", "--split", "train"],
                               env=env, capture_output=True, text=True)
            assert r.returncode == 0, r.stderr[-3000:]
            sh([xs, "measure", "--views", "xs,dn"], env)
            rep = json.load(open(tmp / "work" / "reports" / "supp_recall_xs_dn.json"))
            print("xs+dn measure:", json.dumps(rep["India"]))
            assert rep["India"]["union_recall"] >= rep["India"]["base_recall"]
            dst2 = tmp / "work_xd"
            sh([xs, "merge", "--split", "train", "--dst", dst2, "--views", "xs,dn", "--cap", "dn=8"], env)
            env_xd = dict(env, BER_WORK_DIR=str(dst2))
            sh([SRC / "run.py", "features", "--split", "train"], env_xd)
            r = subprocess.run([dpy, str(SRC / "experiments" / "dense_view.py"), "features", "--split", "train"],
                               env=env_xd, capture_output=True, text=True)
            assert r.returncode == 0, r.stderr[-3000:]
            sh([SRC / "run.py", "evaluate", "--approach", "1"], dict(env_xd, BER_EXTRA="featdn", BER_TAG="_dn"))
            ev = json.load(open(dst2 / "reports" / "eval_A1_dn.json"))
            assert "dn_cos" in ev["feature_gain_top20"] or ev["n_features"] > 29, ev
            print("A1 xs+dn@8 + dn_cos:", ev["eval"]["macro_f05"])
        else:
            print("dense view skipped (set BER_DENSE_PY to a python with torch + transformers)")

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
