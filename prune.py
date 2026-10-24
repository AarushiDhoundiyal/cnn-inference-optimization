import argparse
import gzip
import os

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.optim as optim

from benchmark import benchmark
from data import get_loaders
from model import ResNet20
from train import train_one_epoch, evaluate


def prunable(model):
    # Prune the weights of every Conv and Linear layer
    return [(m, "weight") for m in model.modules()
            if isinstance(m, (nn.Conv2d, nn.Linear))]


def sparsity_pct(model):
    zeros, total = 0, 0
    for m, _ in prunable(model):
        zeros += (m.weight == 0).sum().item()
        total += m.weight.numel()
    return 100.0 * zeros / total


def gzip_size_mb(model, path="tmp_prune.pth"):
    torch.save(model.state_dict(), path)
    with open(path, "rb") as f:
        raw = f.read()
    os.remove(path)
    return len(gzip.compress(raw)) / 1e6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetune_epochs", type=int, default=3)
    args = parser.parse_args()

    train_loader, test_loader = get_loaders()
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cpu")

    for amount in [0.5, 0.7, 0.9]:
        print(f"\n=== Pruning {int(amount * 100)}% of weights ===")
        model = ResNet20()
        model.load_state_dict(torch.load("resnet20_fp32.pth", map_location="cpu",
                                         weights_only=True))

        # 1. Zero out the smallest weights across the whole network
        prune.global_unstructured(prunable(model),
                                  pruning_method=prune.L1Unstructured,
                                  amount=amount)
        _, acc = evaluate(model, test_loader, criterion, device)
        print(f"Sparsity {sparsity_pct(model):.1f}% | accuracy right after pruning: {acc:.2f}%")

        # 2. Fine-tune with a small learning rate (the pruning mask stays applied)
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.finetune_epochs)
        for epoch in range(1, args.finetune_epochs + 1):
            train_one_epoch(model, train_loader, criterion, optimizer, device)
            _, acc = evaluate(model, test_loader, criterion, device)
            scheduler.step()
            print(f"Fine-tune epoch {epoch}/{args.finetune_epochs} | accuracy {acc:.2f}%")

        # 3. Make the pruning permanent (bake the zeros into the weights)
        for m, _ in prunable(model):
            prune.remove(m, "weight")

        print(f"Final sparsity: {sparsity_pct(model):.1f}%")
        print(f"Compressed (gzip) size: {gzip_size_mb(model):.3f} MB")
        benchmark(model, f"pruned_{int(amount * 100)}", test_loader)


if __name__ == "__main__":
    main()