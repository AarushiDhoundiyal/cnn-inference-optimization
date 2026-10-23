import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from data import get_loaders
from model import ResNet20


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()                      # training mode (BatchNorm updates)
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="train", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()          # clear old gradients
        outputs = model(images)        # forward pass
        loss = criterion(outputs, labels)
        loss.backward()                # backpropagation
        optimizer.step()               # update the weights

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, 100.0 * correct / total


def evaluate(model, loader, criterion, device):
    model.eval()                       # evaluation mode
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():              # no gradients needed when testing
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    device = torch.device("cpu")
    train_loader, test_loader = get_loaders()

    model = ResNet20().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()               # lower the learning rate

        print(f"Epoch {epoch}/{args.epochs} | "
              f"train loss {train_loss:.3f} acc {train_acc:.2f}% | "
              f"test loss {test_loss:.3f} acc {test_acc:.2f}% | "
              f"time {time.time() - start:.0f}s")

        if test_acc > best_acc:        # save the best model so far
            best_acc = test_acc
            torch.save(model.state_dict(), "resnet20_fp32.pth")

    print(f"Best test accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()