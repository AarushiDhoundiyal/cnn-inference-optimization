import copy

import torch
import torch.nn as nn
from torch.ao.nn.quantized import FloatFunctional
from torch.ao.quantization import (QuantStub, DeQuantStub, fuse_modules,
                                   get_default_qconfig, prepare, convert)
from torchvision import datasets, transforms

from data import CIFAR_MEAN, CIFAR_STD
from benchmark import benchmark
from data import get_loaders


class QBasicBlock(nn.Module):
    """Same as BasicBlock, but written so it can be quantized."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.skip_add = FloatFunctional()   # quantization-friendly addition

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.skip_add.add_relu(out, self.shortcut(x))  # add, then ReLU


class QResNet20(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.quant = QuantStub()        # float -> int8 at the input
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(16, 16, 3, 1)
        self.layer2 = self._make_layer(16, 32, 3, 2)
        self.layer3 = self._make_layer(32, 64, 3, 2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)
        self.dequant = DeQuantStub()    # int8 -> float at the output

    def _make_layer(self, in_c, out_c, num_blocks, stride):
        layers = [QBasicBlock(in_c, out_c, stride)]
        for _ in range(num_blocks - 1):
            layers.append(QBasicBlock(out_c, out_c, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.quant(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer3(self.layer2(self.layer1(out)))
        out = self.pool(out).flatten(1)
        out = self.fc(out)
        return self.dequant(out)


def fuse(model):
    """Merge Conv+BN(+ReLU) into single layers."""
    fuse_modules(model, [["conv1", "bn1", "relu"]], inplace=True)
    for m in model.modules():
        if isinstance(m, QBasicBlock):
            fuse_modules(m, [["conv1", "bn1", "relu1"], ["conv2", "bn2"]], inplace=True)
            if len(m.shortcut) > 0:
                fuse_modules(m, [["shortcut.0", "shortcut.1"]], inplace=True)


def calibration_loader(num_images=1000):
    # Training images WITHOUT augmentation, so we never calibrate on the test set
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
    train_set = datasets.CIFAR10("./data", train=True, download=True, transform=tf)
    subset = torch.utils.data.Subset(train_set, range(num_images))
    return torch.utils.data.DataLoader(subset, batch_size=100, shuffle=False)


if __name__ == "__main__":
    torch.backends.quantized.engine = "fbgemm"   # the x86 CPU backend

    # 1. Load your trained FP32 weights into the quantizable model
    model = QResNet20()
    model.load_state_dict(torch.load("resnet20_fp32.pth", map_location="cpu",
                                     weights_only=True))
    model.eval()

    # 2. Fuse layers, attach the quantization settings, insert observers
    fuse(model)
    model.qconfig = get_default_qconfig("fbgemm")
    prepare(model, inplace=True)

    # 3. Calibrate: run images through so observers record value ranges
    with torch.no_grad():
        for images, _ in calibration_loader():
            model(images)

    # 4. Convert to a real INT8 model
    convert(model, inplace=True)

    # 5. Benchmark it and save the results next to the baseline
    _, test_loader = get_loaders()
    benchmark(model, "int8_static", test_loader)