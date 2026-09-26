"""Paths and parameters shared by every stage.

All paths can be overridden with environment variables so the package can run on
any machine: BER_DATA_DIR (contains train/ and test/), BER_WORK_DIR (artifacts),
BER_OUT_DIR (submission files).
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]  # .../AmazonML
DATA_DIR = Path(os.environ.get("BER_DATA_DIR", ROOT / "student_resource" / "dataset"))
WORK_DIR = Path(os.environ.get("BER_WORK_DIR", ROOT / "work"))
OUT_DIR = Path(os.environ.get("BER_OUT_DIR", ROOT / "output"))
VALIDATOR = Path(os.environ.get("BER_VALIDATOR", ROOT / "student_resource" / "utils" / "validate_submission.py"))

N_THREADS = int(os.environ.get("BER_THREADS", os.cpu_count() or 8))
SEED = 42

# ---- validation protocol (S1-grouped folds; see RESEARCH_DESIGN.md §6) ----
N_FOLDS = 10
TRAIN_FOLDS = (1, 2)   # model fitting
TUNE_FOLD = 3          # early stopping + threshold selection
EVAL_FOLD = 0          # reported score (never used for any choice)
QUERY_FOLDS = (0, 1, 2, 3)  # train S1 folds that are retrieved/scored
if os.environ.get("BER_QUERY_FOLDS"):  # e.g. "4,5,6,7,8,9" for extra stage-2 training data (isolated work dir)
    QUERY_FOLDS = tuple(int(x) for x in os.environ["BER_QUERY_FOLDS"].split(","))

# ---- blocking (candidate generation) ----
BLOCK = dict(
    fit_sample=2_000_000,   # strings used to fit each TF-IDF vocabulary (label-free)
    name_max_df=0.05,       # drop char n-grams present in >5% of records (speed + noise)
    addr_max_df=0.05,
    k_name=5,               # S1 -> target top-K per view
    k_addr=5,
    k_combo=10,
    k_rev=8,                # target -> S1 top-K (reverse view)
    item_block=65_536,      # items per column block in top-K matmul (cache locality)
)

# ---- models ----
LGB_PARAMS = dict(
    objective="binary",
    learning_rate=0.08,
    num_leaves=127,
    min_data_in_leaf=200,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    max_bin=127,
    verbose=-1,
    seed=SEED,
    num_threads=N_THREADS,
)
LGB_ROUNDS = 1500
LGB_EARLY_STOP = 50

THRESHOLD_GRID = [round(0.30 + 0.01 * i, 2) for i in range(66)]  # 0.30 .. 0.95

# ---- experiment variants (so teammates can run variants without overwriting each other) ----
# BER_TAG:   suffix for model/report/stage-2 artifact names, e.g. "_v2feats"
# BER_EXTRA: comma-separated extra feature sets (work/<name>/<split>/*.parquet, row-aligned
#            with work/feat), e.g. "featx" produced by `python src/experiments/features_v2.py`
RUN_TAG = os.environ.get("BER_TAG", "")
EXTRA_SETS = [s for s in os.environ.get("BER_EXTRA", "").split(",") if s]


def ensure_dirs():
    for d in (WORK_DIR, OUT_DIR):
        d.mkdir(parents=True, exist_ok=True)
