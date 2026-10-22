import torch

print("PyTorch:", torch.__version__)
print("CPU threads:", torch.get_num_threads())

x = torch.rand(3, 3)
print(x)