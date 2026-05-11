from __future__ import annotations

import os

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from torchvision.models import ResNet18_Weights
from tqdm import tqdm


_CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
_CIFAR100_STD = (0.2675, 0.2565, 0.2761)

_SAMPLES_PER_CLASS = 81


def _make_balanced_subset(dataset, n_classes: int, samples_per_class: int):
    class_counts = [0 for _ in range(n_classes)]
    indices = []

    for idx, label in enumerate(dataset.targets):
        if class_counts[label] < samples_per_class:
            indices.append(idx)
            class_counts[label] += 1

        if all(c == samples_per_class for c in class_counts):
            break

    missing = [
        c for c, count in enumerate(class_counts)
        if count < samples_per_class
    ]

    if missing:
        raise RuntimeError(
            f"Not enough samples for classes: {missing}. "
            f"Counts: {class_counts}"
        )

    return Subset(dataset, indices), class_counts


def init_last_layer(layer: nn.Linear) -> None:
    """Initialize CIFAR100 head using 81 normalized features per class."""

    torch.manual_seed(42)

    n_classes = layer.out_features      # 100
    in_features = layer.in_features     # 512

    data_dir = os.environ.get("CIFAR100_DATA_DIR", "./data")
    batch_size = int(os.environ.get("HEAD_INIT_BATCH_SIZE", "256"))
    num_workers = int(os.environ.get("HEAD_INIT_NUM_WORKERS", "0"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = T.Compose(
        [
            T.Resize(224),
            T.ToTensor(),
            T.Normalize(mean=_CIFAR100_MEAN, std=_CIFAR100_STD),
        ]
    )

    train_dataset = datasets.CIFAR100(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    subset, class_counts = _make_balanced_subset(
        dataset=train_dataset,
        n_classes=n_classes,
        samples_per_class=_SAMPLES_PER_CLASS,
    )

    train_loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    backbone.fc = nn.Identity()
    backbone.eval()
    backbone.to(device)

    sums = torch.zeros(n_classes, in_features, device=device)
    counts = torch.zeros(n_classes, device=device)

    print(
        f"[head_init] Computing prototypes from balanced CIFAR100 subset: "
        f"{_SAMPLES_PER_CLASS} samples/class, total={len(subset)} | "
        f"feature_norm_before_mean=True"
    )

    with torch.no_grad():
        for images, labels in tqdm(
            train_loader,
            desc="  Computing prototypes",
            unit="batch",
            leave=False,
        ):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            features = backbone(images)

            # Normalize every image feature before class averaging.
            features = features / (features.norm(dim=1, keepdim=True) + 1e-12)

            sums.index_add_(0, labels, features)

            counts.index_add_(
                0,
                labels,
                torch.ones_like(labels, dtype=counts.dtype, device=device),
            )

    missing_classes = (counts == 0).nonzero(as_tuple=False).flatten().tolist()
    if missing_classes:
        raise RuntimeError(f"No selected training examples for classes: {missing_classes}")

    if not torch.all(counts == _SAMPLES_PER_CLASS):
        raise RuntimeError(
            f"Expected {_SAMPLES_PER_CLASS} samples per class, got counts={counts.cpu().tolist()}"
        )

    prototypes = sums / counts[:, None]

    # Normalize final class prototypes.
    prototypes = prototypes / (prototypes.norm(dim=1, keepdim=True) + 1e-12)

    with torch.no_grad():
        layer.weight.copy_(
            prototypes.to(
                device=layer.weight.device,
                dtype=layer.weight.dtype,
            )
        )

        if layer.bias is not None:
            layer.bias.zero_()