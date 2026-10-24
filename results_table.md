| Model | Accuracy (%) | Size (MB) | Latency bs=1 (ms) | Throughput (img/s) | Latency speed-up | Throughput gain |
|---|---|---|---|---|---|---|
| PyTorch FP32 | 90.42 | 1.136 | 7.26 | 1018 | 1.00x | 1.00x |
| PyTorch INT8 | 90.38 | 0.334 | 7.10 | 1341 | 1.02x | 1.32x |
| ONNX Runtime FP32 | 90.42 | 1.096 | 2.09 | 2083 | 3.48x | 2.05x |

| Pruning (global L1, 3 fine-tune epochs) | Accuracy (%) | Gzip size (MB) |
|---|---|---|
| 0% sparsity | 90.42 | 1.028 |
| 50% sparsity | 90.29 | 0.611 |
| 70% sparsity | 89.91 | 0.412 |
| 90% sparsity | 85.31 | 0.200 |
