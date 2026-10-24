import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Median of all runs (6 FP32, 5 INT8, 3 ONNX); raw numbers in results_log.md
MODELS = {
    "PyTorch FP32": {"acc": 90.42, "size": 1.136, "lat": 7.262, "thr": 1017.6},
    "PyTorch INT8": {"acc": 90.38, "size": 0.334, "lat": 7.104, "thr": 1340.9},
    "ONNX Runtime FP32": {"acc": 90.42, "size": 1.096, "lat": 2.087, "thr": 2083.0},
}
# Pruning: (sparsity %, accuracy %, gzip size MB) -- single run each
PRUNING = [(0, 90.42, None), (50, 90.29, 0.611), (70, 89.91, 0.412), (90, 85.31, 0.200)]
DENSE_GZIP_MB = 1.028   

base = MODELS["PyTorch FP32"]

# ---------- Markdown table ----------
lines = ["| Model | Accuracy (%) | Size (MB) | Latency bs=1 (ms) | Throughput (img/s) | Latency speed-up | Throughput gain |",
         "|---|---|---|---|---|---|---|"]
for name, m in MODELS.items():
    lines.append(f"| {name} | {m['acc']:.2f} | {m['size']:.3f} | {m['lat']:.2f} | {m['thr']:.0f} | "
                 f"{base['lat'] / m['lat']:.2f}x | {m['thr'] / base['thr']:.2f}x |")
lines += ["", "| Pruning (global L1, 3 fine-tune epochs) | Accuracy (%) | Gzip size (MB) |", "|---|---|---|"]
for sp, acc, gz in PRUNING:
    gz_txt = f"{gz:.3f}" if gz is not None else (f"{DENSE_GZIP_MB:.3f}" if DENSE_GZIP_MB else "n/a")
    lines.append(f"| {sp}% sparsity | {acc:.2f} | {gz_txt} |")
open("results_table.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

# ---------- Chart 1: size / latency / throughput ----------
names = list(MODELS)
colors = ["#888888", "#2a7de1", "#e1572a"]
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, key, title in zip(axes, ["size", "lat", "thr"],
                          ["Model size (MB)", "Latency, batch 1 (ms)", "Throughput, batch 128 (img/s)"]):
    vals = [MODELS[n][key] for n in names]
    bars = ax.bar(range(3), vals, color=colors)
    ax.set_xticks(range(3)); ax.set_xticklabels(["PyTorch\nFP32", "PyTorch\nINT8", "ONNX RT\nFP32"])
    ax.set_title(title)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3g}", ha="center", va="bottom")
fig.suptitle("ResNet-20 on CIFAR-10, Ryzen 5 5500U CPU (median of 3 runs)")
plt.tight_layout(); plt.savefig("chart_performance.png", dpi=150); plt.close()

# ---------- Chart 2: accuracy vs sparsity ----------
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot([p[0] for p in PRUNING], [p[1] for p in PRUNING], marker="o")
for sp, acc, _ in PRUNING:
    ax.annotate(f"{acc:.2f}", (sp, acc), textcoords="offset points", xytext=(0, 8), ha="center")
ax.set_xlabel("Sparsity (% of weights pruned)"); ax.set_ylabel("Test accuracy (%)")
ax.set_title("Accuracy vs sparsity (after fine-tuning)")
plt.tight_layout(); plt.savefig("chart_pruning.png", dpi=150); plt.close()
print("Saved chart_performance.png and chart_pruning.png")