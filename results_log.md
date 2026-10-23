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
