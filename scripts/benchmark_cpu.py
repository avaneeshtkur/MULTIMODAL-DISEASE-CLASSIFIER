"""
Measure CPU inference latency of the EfficientNet-B0 ONNX pipeline on THIS machine.

With installed models, benchmarks them. Without, it exports a RANDOMLY INITIALISED
EfficientNet-B0 (latency does not depend on weight values) and says so in the output.

    python scripts/benchmark_cpu.py --runs 50
"""
import argparse
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import config  # noqa: E402
from src.core.image_preprocessor import default_spec, preprocess  # noqa: E402


def bench(path: Path, spec: dict, runs: int, label: str):
    import onnxruntime as ort
    t0 = time.perf_counter()
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    load_ms = (time.perf_counter() - t0) * 1000
    name = sess.get_inputs()[0].name
    img = Image.open(config.SAMPLES_DIR / "sample_chest_xray.png") if (config.SAMPLES_DIR / "sample_chest_xray.png").exists() \
        else Image.fromarray(np.random.default_rng(0).integers(0, 255, (512, 512), dtype=np.uint8))
    pre, inf = [], []
    for i in range(runs + 3):
        t = time.perf_counter(); x = preprocess(img, spec); t1 = time.perf_counter()
        sess.run(None, {name: x}); t2 = time.perf_counter()
        if i >= 3:
            pre.append((t1 - t) * 1000); inf.append((t2 - t1) * 1000)
    print(f"{label}: session load {load_ms:.0f} ms | preprocessing median {statistics.median(pre):.1f} ms | "
          f"inference median {statistics.median(inf):.1f} ms (p90 {np.percentile(inf, 90):.1f} ms) | "
          f"model {path.stat().st_size / 1e6:.1f} MB | {runs} runs")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", type=int, default=50); a = ap.parse_args()
    print(f"Machine: {platform.platform()} | {platform.processor() or platform.machine()} | Python {platform.python_version()}")
    from src.core.model_registry import ModelRegistry
    reg = ModelRegistry()
    done = False
    for key in config.MODEL_KEYS:
        h = reg.get(key)
        if h.available:
            bench(h.report.onnx_path, h.metadata["preprocessing"], a.runs, f"[{key}] {h.version}"); done = True
    if not done:
        import torch
        from src.model import build_model
        m = build_model(pretrained=False).eval()
        p = Path(tempfile.mkdtemp()) / "effb0.onnx"
        torch.onnx.export(m, torch.randn(1, 3, 224, 224), str(p), input_names=["image"], output_names=["logit"],
                          opset_version=17, dynamo=False)
        bench(p, default_spec("grayscale_to_rgb"), a.runs, "[random-init EfficientNet-B0, no trained models installed]")


if __name__ == "__main__":
    main()
