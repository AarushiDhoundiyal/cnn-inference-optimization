import json
import os
import statistics
import time

import torch

from data import get_loaders
from model import ResNet20


def model_size_mb(model, path="tmp_size.pth"):
    torch.save(model.state_dict(), path)
    size = os.path.getsize(path) / 1e6
    os.remove(path)
    return size


def measure_latency_ms(model, batch_size, threads, runs=100, warmup=20):
    torch.set_num_threads(threads)
    model.eval()
    x = torch.rand(batch_size, 3, 32, 32)
    times = []
    with torch.no_grad():
        for _ in range(warmup):                 # discard slow first runs
            model(x)
        for _ in range(runs):
            start = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - start) * 1000)
    times.sort()
    return statistics.median(times), times[int(0.95 * runs) - 1]


def evaluate_accuracy(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total


def benchmark(model, name, test_loader, results_file="results.json"):
    acc = evaluate_accuracy(model, test_loader)
    size = model_size_mb(model)
    lat_med, lat_p95 = measure_latency_ms(model, batch_size=1, threads=1)
    b_med, _ = measure_latency_ms(model, batch_size=128, threads=6, runs=20, warmup=5)
    throughput = 128 / (b_med / 1000)

    result = {
        "name": name,
        "accuracy_pct": round(acc, 2),
        "size_mb": round(size, 3),
        "latency_ms_median_bs1": round(lat_med, 3),
        "latency_ms_p95_bs1": round(lat_p95, 3),
        "throughput_img_per_s": round(throughput, 1),
    }
    print(json.dumps(result, indent=2))

    # Save so we can build comparison tables later
    data = {}
    if os.path.exists(results_file):
        with open(results_file) as f:
            data = json.load(f)
    data[name] = result
    with open(results_file, "w") as f:
        json.dump(data, f, indent=2)
    return result


if __name__ == "__main__":
    _, test_loader = get_loaders()
    model = ResNet20()
    model.load_state_dict(torch.load("resnet20_fp32.pth", map_location="cpu", weights_only=True))
    benchmark(model, "fp32_baseline", test_loader)