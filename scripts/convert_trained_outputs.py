"""
Convert an externally produced trained_outputs/ folder (Gemini/Colab format)
into a verified MediScan archive, then (optionally) import it.

    python scripts/convert_trained_outputs.py --src trained_outputs --out trained_outputs_mediscan.zip --import

Every inferred field is backed by evidence recorded in metadata.json["conversion"];
see src/core/external_adapter.py.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.artifact_importer import import_archive  # noqa: E402
from src.core.external_adapter import convert  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="trained_outputs")
    ap.add_argument("--out", default="trained_outputs_mediscan.zip")
    ap.add_argument("--import", dest="do_import", action="store_true", help="import into artifacts/ afterwards")
    args = ap.parse_args()
    out, report = convert(args.src, args.out)
    print(f"Converted archive: {out}")
    for key, r in report.items():
        print(f"  [{key}] {'OK' if r['ok'] else 'FAILED: ' + r['error']}")
        for n in r.get("notes", []):
            print(f"      - {n}")
    if not any(r["ok"] for r in report.values()):
        return 1
    if args.do_import:
        res = import_archive(out)
        print("\n".join(res.lines()))
        return 0 if res.ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
