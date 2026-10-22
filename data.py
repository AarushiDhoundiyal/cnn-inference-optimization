import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Per-channel mean/std of the CIFAR-10 training set
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def get_loaders(batch_size=128, num_workers=2, data_dir="./data"):
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),   # augmentation
        transforms.RandomHorizontalFlip(),      # augmentation
        transforms.ToTensor(),                  # image -> tensor with values in [0,1]
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    test_tf = transforms.Compose([              # no augmentation for testing
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    train_set = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_tf)
    test_set = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False,
                             num_workers=num_workers)
    return train_loader, test_loader


if __name__ == "__main__":
    # This guard is required on Windows when num_workers > 0
    train_loader, test_loader = get_loaders()
    images, labels = next(iter(train_loader))
    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
    print("First 10 labels:", labels[:10])