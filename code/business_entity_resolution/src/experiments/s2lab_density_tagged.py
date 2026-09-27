"""Density check (P-dense, 19% S1 drop) for stage2_lab models whose stage 1 is a TAGGED A2 run.

stage2_lab.density() scores stage 1 with the untagged A2_stage1_fold{A,B} models on
PAIR+CONTEXT features only. A stage-2-lab run built on another stage 1 (e.g. work_xd, whose
stage 1 is A2_xd_* trained with BER_EXTRA=featdn -> +dn_cos) needs those models and their
extra stage-1 columns. This wrapper redirects both, then calls stage2_lab.density unchanged,
so the protocol (seed, drop, recomputed context, tau re-selection) is identical.

    BER_WORK_DIR=../work_xd python src/experiments/s2lab_density_tagged.py \
        --stage1-tag _xd --stage1-extra dn_cos --extra featx,featdn --variant all
Output: <work>/reports/s2lab_density.json (same format as stage2_lab density).
Peak RAM ~12 GB (like stage2_lab density): run it solo.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ber.approaches as ap  # noqa: E402
import stage2_lab  # noqa: E402


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--stage1-tag", required=True, help="tag of the A2 stage-1 models, e.g. _xd")
    a.add_argument("--stage1-extra", default="", help="extra stage-1 input columns, e.g. dn_cos")
    a.add_argument("--extra", default="", help="feature sets to hstack (must provide the stage-1 extra "
                                                "columns and the stage-2 model's extra columns)")
    a.add_argument("--variant", default="all")
    a.add_argument("--drop", type=float, default=0.19)
    x = a.parse_args()
    orig = ap.load_model
    ap.load_model = lambda n: orig(n.replace("A2_stage1", f"A2{x.stage1_tag}_stage1"))
    ap.S1_FEATS = ap.S1_FEATS + [c for c in x.stage1_extra.split(",") if c]
    stage2_lab.density([f"s2lab_{v}" for v in x.variant.split(",")], [s for s in x.extra.split(",") if s], x.drop)


if __name__ == "__main__":
    main()
