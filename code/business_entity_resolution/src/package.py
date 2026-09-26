"""Build the final submission zip in the structure required by the challenge.

    python src/package.py --team TEAMNAME [--doc ../../Documentation.md]

<team>_submission.zip
├── output/matching_results.tsv, output/candidate_pairs.tsv
├── code/business_entity_resolution/{src/, tests/, README.md, requirements.txt}
└── Documentation_template.md   (the filled-in methodology document)

Runs the official validator on the output files first and refuses to package on failure.
"""
import argparse
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ber import config  # noqa: E402
from ber.approaches import run_validator  # noqa: E402

PKG = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True)
    ap.add_argument("--doc", default=str(config.ROOT / "Documentation.md"))
    a = ap.parse_args()
    if run_validator() != 0:
        sys.exit("validator failed: not packaging")
    doc = Path(a.doc)
    if not doc.exists():
        sys.exit(f"methodology document not found: {doc}")
    dst = config.ROOT / f"{a.team}_submission.zip"
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        for f in ("matching_results.tsv", "candidate_pairs.tsv"):
            z.write(config.OUT_DIR / f, f"output/{f}")
        for f in sorted(PKG.rglob("*")):
            rel = f.relative_to(PKG)
            if f.is_file() and "__pycache__" not in rel.parts and f.suffix != ".pyc":
                z.write(f, f"code/business_entity_resolution/{rel}")
        z.write(doc, "Documentation_template.md")
    print(f"wrote {dst} ({dst.stat().st_size / 2**20:.1f} MB)")
    with zipfile.ZipFile(dst) as z:
        for n in z.namelist()[:40]:
            print("  ", n)


if __name__ == "__main__":
    main()
