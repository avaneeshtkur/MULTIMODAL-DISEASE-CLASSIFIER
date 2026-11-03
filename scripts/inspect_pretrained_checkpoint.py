"""Inspect public state dictionaries without executing pickle-provided classes.

Uses memory mapping so training checkpoints containing optimizer states do not
need to fit in RAM. This does not enable a model or certify its performance.
"""
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    args = parser.parse_args()
    import torch
    with torch.serialization.safe_globals([argparse.Namespace]):
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True, mmap=True)
    state = checkpoint.get('model', checkpoint)
    shapes = {key: list(value.shape) for key, value in state.items() if isinstance(value, torch.Tensor)}
    print(json.dumps({'keys': list(checkpoint), 'state_shapes': shapes}, indent=2))


if __name__ == '__main__':
    main()
