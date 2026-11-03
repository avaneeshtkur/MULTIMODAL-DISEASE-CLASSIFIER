"""
Package this repo's training outputs into trained_outputs.zip (run on Colab after training).

    python scripts/package_artifacts.py --version 2026-11-02 --out trained_outputs.zip
"""
import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.artifact_packager import package  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="trained_outputs.zip")
    ap.add_argument("--version", default=date.today().isoformat())
    ap.add_argument("--only", default="", help="comma-separated subset of tb,dr,oral,gatekeeper")
    ap.add_argument("--dr-input-domain", default="ben_graham_filtered",
                    help="declared DR training domain; use 'none' if trained on raw fundus photos")
    args = ap.parse_args()
    keys = [k.strip() for k in args.only.split(",") if k.strip()] or None
    domain = None if args.dr_input_domain.lower() == "none" else args.dr_input_domain
    out, packed = package(args.out, args.version, keys, domain)
    print(f"Wrote {out} with models: {', '.join(packed) or 'none'}")
    return 0 if packed else 1


if __name__ == "__main__":
    raise SystemExit(main())
