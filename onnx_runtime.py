import json
import os
import statistics
import time

import numpy as np
import onnxruntime as ort
import torch

from data import get_loaders
from model import ResNet20

ONNX_PATH = "resnet20_fp32.onnx"


def export_model():
    model = ResNet20()
    model.load_state_dict(torch.load("resnet20_fp32.pth", map_location="cpu",
                                     weights_only=True))
    model.eval()
    dummy = torch.rand(1, 3, 32, 32)
    torch.onnx.export(
        model, dummy, ONNX_PATH,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},  # any batch size
        opset_version=17,
    )
    print("Exported to", ONNX_PATH)
    return model


def make_session(threads):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(ONNX_PATH, opts, providers=["CPUExecutionProvider"])


def latency_ms(sess, batch_size, runs=100, warmup=20):
    x = np.random.rand(batch_size, 3, 32, 32).astype(np.float32)
    for _ in range(warmup):
        sess.run(None, {"input": x})
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        sess.run(None, {"input": x})
        times.append((time.perf_counter() - start) * 1000)
    times.sort()
    return statistics.median(times), times[int(0.95 * runs) - 1]


def accuracy(sess, loader):
    correct, total = 0, 0
    for images, labels in loader:
        logits = sess.run(None, {"input": images.numpy()})[0]
        correct += int((logits.argmax(1) == labels.numpy()).sum())
        total += len(labels)
    return 100.0 * correct / total


if __name__ == "__main__":
    model = export_model()
    sess1 = make_session(threads=1)    # for batch-1 latency (same as our baseline)
    sess6 = make_session(threads=6)    # for throughput

    # Sanity check: ONNX output must match PyTorch output
    x = torch.rand(4, 3, 32, 32)
    with torch.no_grad():
        torch_out = model(x).numpy()
    ort_out = sess1.run(None, {"input": x.numpy()})[0]
    print("Max difference PyTorch vs ONNX:", np.abs(torch_out - ort_out).max())

    _, test_loader = get_loaders()
    acc = accuracy(sess6, test_loader)
    lat_med, lat_p95 = latency_ms(sess1, 1)
    b_med, _ = latency_ms(sess6, 128, runs=20, warmup=5)

    result = {
        "name": "onnx_fp32",
        "accuracy_pct": round(acc, 2),
        "size_mb": round(os.path.getsize(ONNX_PATH) / 1e6, 3),
        "latency_ms_median_bs1": round(lat_med, 3),
        "latency_ms_p95_bs1": round(lat_p95, 3),
        "throughput_img_per_s": round(128 / (b_med / 1000), 1),
    }
    print(json.dumps(result, indent=2))

    data = {}
    if os.path.exists("results.json"):
        with open("results.json") as f:
            data = json.load(f)
    data["onnx_fp32"] = result
    with open("results.json", "w") as f:
        json.dump(data, f, indent=2)