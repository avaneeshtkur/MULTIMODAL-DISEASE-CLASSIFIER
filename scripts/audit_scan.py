"""Audit real routing and direct specialist inference without changing patient records.
Usage: python scripts/audit_scan.py scan.png --module auto --output audit.json
"""
import argparse
import json
import tempfile
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from src.core.pipeline import build_context, run_screening
from src.core.image_preprocessor import preprocess
from src.core.inference_engine import output_to_score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('--module', choices=['auto', 'tb', 'dr', 'oral'], default='auto')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as td:
        ctx = build_context(db_path=Path(td) / 'audit.db')
        pid = ctx.patients.register('AUDIT', is_demo=True)
        out = run_screening(ctx, args.image.read_bytes(), args.image.name, pid,
                            args.module, explain=False)
        audit = {'image_sha256': out.image_sha256, 'status': out.status,
                 'routing': vars(out.modality) if out.modality else None,
                 'result': out.result, 'rejection': out.rejection.code if out.rejection else None}
        if out.result and out.result.get('model_score') is not None:
            h = ctx.registry.get(out.result['disease'])
            with Image.open(args.image) as im:
                raw = h.run(preprocess(im, h.metadata['preprocessing']))
            score = output_to_score(raw, h.metadata)
            audit.update(direct_raw_output=raw.tolist(), direct_score=score,
                         matches_displayed_score=abs(score - out.result['model_score']) <= 0.000051)
        text = json.dumps(audit, indent=2, default=str)
        if args.output:
            args.output.write_text(text)
        print(text)

if __name__ == '__main__':
    main()
