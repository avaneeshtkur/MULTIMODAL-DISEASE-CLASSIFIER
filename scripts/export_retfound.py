"""Strictly export a fine-tuned RETFound head from a reviewed, pinned checkout.

Run in the upstream model's environment on the remote build machine. No network
downloads, missing-key fallback, classifier replacement or local installation.
"""
import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-dir', type=Path, required=True)
    p.add_argument('--revision', required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--constructor', default='RETFound_mae')
    p.add_argument('--num-classes', type=int, default=5)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    revision = subprocess.check_output(['git', '-C', str(args.reference_dir), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(args.reference_dir), 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
    if revision != args.revision or dirty:
        p.error('Reference checkout must be clean and match the full reviewed commit SHA.')
    if args.out.exists():
        p.error('Choose a new output path; existing artifacts are never overwritten.')
    import torch
    sys.path.insert(0, str(args.reference_dir.resolve()))
    spec = importlib.util.spec_from_file_location('retfound_reference_models', args.reference_dir / 'models_vit.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = getattr(module, args.constructor)(num_classes=args.num_classes, global_pool=True)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    state = state.get('model', state)
    if 'head.weight' not in state or state['head.weight'].shape[0] != args.num_classes:
        raise ValueError('A matching disease-fine-tuned classifier head is required.')
    model.load_state_dict(state, strict=True)
    model.eval()
    with torch.inference_mode():
        probe = torch.zeros(1, 3, 224, 224)
        original = model(probe)
        if tuple(original.shape) != (1, args.num_classes):
            raise ValueError('Unexpected output shape; inspect the selected upstream architecture.')
        traced = torch.jit.trace(model, probe, strict=True)
        torch.testing.assert_close(original, traced(probe), rtol=0, atol=1e-5)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(args.out))
    print(json.dumps({'export': str(args.out), 'reference_revision': revision,
                      'status': 'Exported; independent real-image reference fixtures still required for serving.'}))


if __name__ == '__main__':
    main()
