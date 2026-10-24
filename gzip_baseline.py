import torch
from model import ResNet20
from prune import gzip_size_mb

model = ResNet20()
model.load_state_dict(torch.load("resnet20_fp32.pth", map_location="cpu", weights_only=True))
print(f"Dense FP32 gzip size: {gzip_size_mb(model):.3f} MB")