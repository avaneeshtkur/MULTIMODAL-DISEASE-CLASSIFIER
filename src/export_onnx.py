"""
export_onnx.py — convert a trained model to ONNX for fast CPU inference.

Usage:
    python -m src.export_onnx              # exports TB model
    python -m src.export_onnx --disease dr

ONNX Runtime is typically faster than PyTorch on CPU; this script measures both
on the current machine and writes results/<disease>/cpu_benchmark.json.
"""
import argparse
import json
import time

import numpy as np
import onnxruntime as ort
import torch

import config
from src.data import eval_tfms, load_image
from src.model import load_trained


def export(disease: str = "tb", runs: int = 30):
    dcfg = config.task_config(disease)
    model = load_trained("cpu", disease=disease)

    # Use a real test image for correctness check (random noise gives unrealistic diffs)
    splits_csv = dcfg["splits_csv"]
    if splits_csv.exists():
        import pandas as pd
        from src.data import resolve_paths
        df = resolve_paths(pd.read_csv(splits_csv), dcfg["data_dir"])
        path = df.query("split == 'test'")["path"].iloc[0]
        dummy = eval_tfms(load_image(path, dcfg["image_mode"])).unsqueeze(0)
    else:
        dummy = torch.randn(1, 3, config.IMG_SIZE, config.IMG_SIZE)

    onnx_path = dcfg["onnx_file"]
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(model, dummy, str(onnx_path),
                      input_names=["image"], output_names=["logit"],
                      dynamic_axes={"image": {0: "batch"}, "logit": {0: "batch"}},
                      opset_version=17, dynamo=False)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = model(dummy).numpy()
    out = sess.run(None, {"image": dummy.numpy()})[0]
    max_diff = float(np.abs(ref - out).max())

    def bench(fn):
        fn()  # warm-up
        t0 = time.perf_counter()
        for _ in range(runs):
            fn()
        return (time.perf_counter() - t0) / runs * 1000

    with torch.no_grad():
        torch_ms = bench(lambda: model(dummy))
    onnx_ms = bench(lambda: sess.run(None, {"image": dummy.numpy()}))
    size_mb = onnx_path.stat().st_size / 1e6

    res = {"disease": disease, "onnx_vs_torch_max_diff": max_diff,
           "pytorch_cpu_ms": torch_ms, "onnx_cpu_ms": onnx_ms, "onnx_size_mb": size_mb}
    results_dir = dcfg["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "cpu_benchmark.json").write_text(json.dumps(res, indent=2))
    print(f"[{disease}] ONNX saved ({size_mb:.1f} MB). Max diff vs PyTorch: {max_diff:.2e}")
    print(f"[{disease}] CPU latency — PyTorch: {torch_ms:.1f} ms | ONNX: {onnx_ms:.1f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--disease", default="tb", choices=["tb", "dr", "oral", "gatekeeper"])
    args = parser.parse_args()
    export(args.disease)
