	                                    FP32	            INT8	            Change
Accuracy	                            90.42%	            90.38%	            −0.04 points
Model size	                            1.136 MB	        0.334 MB	        3.4x smaller (−71%)
Throughput (batch 128, 6 threads)	    1,022.5 img/s	    1,340.9 img/s	    +31%
Latency (batch 1, median)	            7.264 ms	        7.146 ms	        −1.6% (within noise)


First run :-

python benchmark.py

Files already downloaded and verified
Files already downloaded and verified
{
  "name": "fp32_baseline",
  "accuracy_pct": 90.42,
  "size_mb": 1.136,
  "latency_ms_median_bs1": 7.184,
  "latency_ms_p95_bs1": 8.335,
  "throughput_img_per_s": 1009.3
}

python quantize.py

C:\Users\aarus\cnn-inference-optimization\venv\Lib\site-packages\torch\ao\quantization\observer.py:229: UserWarning: Please use quant_min and quant_max to specify the range for observers.                     reduce_range will be deprecated in a future release of PyTorch.
  warnings.warn(
Files already downloaded and verified
Files already downloaded and verified
Files already downloaded and verified
{
  "name": "int8_static",
  "accuracy_pct": 90.38,
  "size_mb": 0.334,
  "latency_ms_median_bs1": 7.08,
  "latency_ms_p95_bs1": 8.081,
  "throughput_img_per_s": 1312.4
}

Second run:-

python benchmark.py

Files already downloaded and verified
Files already downloaded and verified
{
  "name": "fp32_baseline",
  "accuracy_pct": 90.42,
  "size_mb": 1.136,
  "latency_ms_median_bs1": 7.264,
  "latency_ms_p95_bs1": 7.992,
  "throughput_img_per_s": 1025.2
}

python quantize.py 

C:\Users\aarus\cnn-inference-optimization\venv\Lib\site-packages\torch\ao\quantization\observer.py:229: UserWarning: Please use quant_min and quant_max to specify the range for observers.                     reduce_range will be deprecated in a future release of PyTorch.
  warnings.warn(
Files already downloaded and verified
Files already downloaded and verified
Files already downloaded and verified
{
  "name": "int8_static",
  "accuracy_pct": 90.38,
  "size_mb": 0.334,
  "latency_ms_median_bs1": 7.22,
  "latency_ms_p95_bs1": 9.524,
  "throughput_img_per_s": 1340.9
}


Third run:-

python benchmark.py

Files already downloaded and verified
Files already downloaded and verified
{
  "name": "fp32_baseline",
  "accuracy_pct": 90.42,
  "size_mb": 1.136,
  "latency_ms_median_bs1": 7.321,
  "latency_ms_p95_bs1": 11.239,
  "throughput_img_per_s": 1022.5
}

python quantize.py

C:\Users\aarus\cnn-inference-optimization\venv\Lib\site-packages\torch\ao\quantization\observer.py:229: UserWarning: Please use quant_min and quant_max to specify the range for observers.                     reduce_range will be deprecated in a future release of PyTorch.
  warnings.warn(
Files already downloaded and verified
Files already downloaded and verified
Files already downloaded and verified
{
  "name": "int8_static",
  "accuracy_pct": 90.38,
  "size_mb": 0.334,
  "latency_ms_median_bs1": 7.146,
  "latency_ms_p95_bs1": 9.82,
  "throughput_img_per_s": 1341.6
}


prune.py output:-


Files already downloaded and verified
Files already downloaded and verified

=== Pruning 50% of weights ===
Sparsity 50.0% | accuracy right after pruning: 87.65%
Fine-tune epoch 1/3 | accuracy 89.10%                                                                                                                                        
Fine-tune epoch 2/3 | accuracy 90.01%                                                                                                                                        
Fine-tune epoch 3/3 | accuracy 90.29%                                                                                                                                        
Final sparsity: 50.0%
Compressed (gzip) size: 0.611 MB
{
  "name": "pruned_50",
  "accuracy_pct": 90.29,
  "size_mb": 1.136,
  "latency_ms_median_bs1": 7.312,
  "latency_ms_p95_bs1": 8.842,
  "throughput_img_per_s": 981.4
}

=== Pruning 70% of weights ===
Sparsity 70.0% | accuracy right after pruning: 51.91%
Fine-tune epoch 1/3 | accuracy 88.04%                                                                                                                                        
Fine-tune epoch 2/3 | accuracy 88.93%                                                                                                                                        
Fine-tune epoch 3/3 | accuracy 89.91%                                                                                                                                        
Final sparsity: 70.0%
Compressed (gzip) size: 0.412 MB
{
  "name": "pruned_70",
  "accuracy_pct": 89.91,
  "size_mb": 1.136,
  "latency_ms_median_bs1": 7.461,
  "latency_ms_p95_bs1": 9.882,
  "throughput_img_per_s": 1008.9
}

=== Pruning 90% of weights ===
Sparsity 90.0% | accuracy right after pruning: 11.38%
Fine-tune epoch 1/3 | accuracy 82.91%                                                                                                                                        
Fine-tune epoch 2/3 | accuracy 84.12%                                                                                                                                        
Fine-tune epoch 3/3 | accuracy 85.31%                                                                                                                                        
Final sparsity: 90.0%
Compressed (gzip) size: 0.200 MB
{
  "name": "pruned_90",
  "accuracy_pct": 85.31,
  "size_mb": 1.136,
  "latency_ms_median_bs1": 7.633,
  "latency_ms_p95_bs1": 9.57,
  "throughput_img_per_s": 1023.6
}


(venv) C:\Users\aarus\cnn-inference-optimization>python onnx_runtime.py
Exported to resnet20_fp32.onnx
Max difference PyTorch vs ONNX: 2.3841858e-06
Files already downloaded and verified
Files already downloaded and verified
{
  "name": "onnx_fp32",
  "accuracy_pct": 90.42,
  "size_mb": 1.096,
  "latency_ms_median_bs1": 2.151,
  "latency_ms_p95_bs1": 2.755,
  "throughput_img_per_s": 2072.2
}

(venv) C:\Users\aarus\cnn-inference-optimization>python onnx_runtime.py
Exported to resnet20_fp32.onnx
Max difference PyTorch vs ONNX: 4.2915344e-06
Files already downloaded and verified
Files already downloaded and verified
{
  "name": "onnx_fp32",
  "accuracy_pct": 90.42,
  "size_mb": 1.096,
  "latency_ms_median_bs1": 2.039,
  "latency_ms_p95_bs1": 2.406,
  "throughput_img_per_s": 2088.4
}

(venv) C:\Users\aarus\cnn-inference-optimization>python onnx_runtime.py
Exported to resnet20_fp32.onnx
Max difference PyTorch vs ONNX: 1.9073486e-06
Files already downloaded and verified
Files already downloaded and verified
{
  "name": "onnx_fp32",
  "accuracy_pct": 90.42,
  "size_mb": 1.096,
  "latency_ms_median_bs1": 2.087,
  "latency_ms_p95_bs1": 2.623,
  "throughput_img_per_s": 2083.0
}

(venv) C:\Users\aarus\cnn-inference-optimization>python benchmark.py
Files already downloaded and verified
Files already downloaded and verified
{
  "name": "fp32_baseline",
  "accuracy_pct": 90.42,
  "size_mb": 1.136,
  "latency_ms_median_bs1": 7.336,
  "latency_ms_p95_bs1": 9.35,
  "throughput_img_per_s": 1001.2
}

(venv) C:\Users\aarus\cnn-inference-optimization>python benchmark.py
Files already downloaded and verified
Files already downloaded and verified
{
  "name": "fp32_baseline",
  "accuracy_pct": 90.42,
  "size_mb": 1.136,
  "latency_ms_median_bs1": 7.26,
  "latency_ms_p95_bs1": 8.446,
  "throughput_img_per_s": 1012.6
}

(venv) C:\Users\aarus\cnn-inference-optimization>python benchmark.py
Files already downloaded and verified
Files already downloaded and verified
{
  "name": "fp32_baseline",
  "accuracy_pct": 90.42,
  "size_mb": 1.136,
  "latency_ms_median_bs1": 7.243,
  "latency_ms_p95_bs1": 8.895,
  "throughput_img_per_s": 1023.3
}

(venv) C:\Users\aarus\cnn-inference-optimization>python quantize.py 
C:\Users\aarus\cnn-inference-optimization\venv\Lib\site-packages\torch\ao\quantization\observer.py:229: UserWarning: Please use quant_min and quant_max to specify the range for observers.                     reduce_range will be deprecated in a future release of PyTorch.
  warnings.warn(
Files already downloaded and verified
Files already downloaded and verified
Files already downloaded and verified
{
  "name": "int8_static",
  "accuracy_pct": 90.38,
  "size_mb": 0.334,
  "latency_ms_median_bs1": 7.104,
  "latency_ms_p95_bs1": 10.327,
  "throughput_img_per_s": 1310.4
}

(venv) C:\Users\aarus\cnn-inference-optimization>python quantize.py 
C:\Users\aarus\cnn-inference-optimization\venv\Lib\site-packages\torch\ao\quantization\observer.py:229: UserWarning: Please use quant_min and quant_max to specify the range for observers.                     reduce_range will be deprecated in a future release of PyTorch.
  warnings.warn(
Files already downloaded and verified
Files already downloaded and verified
Files already downloaded and verified
{
  "name": "int8_static",
  "accuracy_pct": 90.38,
  "size_mb": 0.334,
  "latency_ms_median_bs1": 7.081,
  "latency_ms_p95_bs1": 8.029,
  "throughput_img_per_s": 1343.8
}

(venv) C:\Users\aarus\cnn-inference-optimization>python gzip_baseline.py
Dense FP32 gzip size: 1.028 MB

