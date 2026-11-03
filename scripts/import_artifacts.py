"""
Import trained model artifacts (trained_outputs.zip) into artifacts/.

    python scripts/import_artifacts.py --archive trained_outputs.zip --dry-run
    python scripts/import_artifacts.py --archive trained_outputs.zip
    python scripts/import_artifacts.py --archive trained_outputs.zip --only tb,dr

Only models that pass checksum and validation checks are installed; the
previously installed version of every other model is left untouched.
Exit code 0 = at least one model valid/installed and no archive-level errors.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.artifact_importer import import_archive  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive", default="trained_outputs.zip", help="path to trained_outputs.zip")
    ap.add_argument("--dry-run", action="store_true", help="validate only; install nothing")
    ap.add_argument("--only", default="", help="comma-separated subset: tb,dr,oral,gatekeeper")
    ap.add_argument("--allow-missing-manifest", action="store_true",
                    help="import even if the archive has no SHA-256 manifest (models are still validated)")
    args = ap.parse_args()
    only = [k.strip() for k in args.only.split(",") if k.strip()] or None
    res = import_archive(args.archive, dry_run=args.dry_run, only=only,
                         allow_missing_manifest=args.allow_missing_manifest)
    print("\n".join(res.lines()))
    return 0 if res.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
