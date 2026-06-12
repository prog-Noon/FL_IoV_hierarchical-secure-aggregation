"""
datasets/__init__.py

Dataset loading and partitioning for federated learning.

Supports:
  - MNIST
  - FashionMNIST
  - CIFAR-10

Partitioning strategies:
  - IID:     random uniform partition across vehicles
  - Non-IID: Dirichlet(alpha) distribution over class labels

torchvision is imported LAZILY inside load_dataset() so that the project
can be imported even if torchvision is not yet installed (it will only fail
at the moment you actually try to download a real dataset, not at startup).
If torchvision is absent, install it with:
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
"""

from __future__ import annotations
import os
import warnings
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset


# ── Transforms ────────────────────────────────────────────────────────────────

def _get_transforms(dataset_name: str):
    """Return torchvision transforms for the named dataset."""
    try:
        import torchvision.transforms as T
    except ImportError:
        return None   # synthetic path — no transforms needed

    if dataset_name in ("MNIST", "FashionMNIST"):
        return T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    elif dataset_name == "CIFAR10":
        return T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
    raise ValueError(f"Unknown dataset: {dataset_name}")


# ── Dataset loading ────────────────────────────────────────────────────────────

def load_dataset(
    name: str,
    data_dir: str = "./data",
) -> Tuple[Dataset, Dataset]:
    """
    Download and return (train_dataset, test_dataset).

    Falls back to a synthetic random dataset if:
      - torchvision is not installed, OR
      - the download fails (network not available).

    The synthetic dataset has identical tensor shapes and API so all
    downstream code runs unchanged.  Test accuracy will be ~10% (random),
    which is expected and correct for a smoke test.
    """
    os.makedirs(data_dir, exist_ok=True)

    try:
        import torchvision
        import torchvision.datasets as tvd
    except ImportError:
        warnings.warn(
            "torchvision is not installed.  Using synthetic random dataset.\n"
            "To install: pip install torch torchvision "
            "--index-url https://download.pytorch.org/whl/cpu"
        )
        from datasets.synthetic import make_synthetic_dataset
        return make_synthetic_dataset(name)

    cls_map = {
        "MNIST":        tvd.MNIST,
        "FashionMNIST": tvd.FashionMNIST,
        "CIFAR10":      tvd.CIFAR10,
    }
    if name not in cls_map:
        raise ValueError(f"Dataset '{name}' not supported. "
                         f"Choose from: {list(cls_map)}")

    tfm = _get_transforms(name)
    try:
        Cls   = cls_map[name]
        train = Cls(data_dir, train=True,  download=True, transform=tfm)
        test  = Cls(data_dir, train=False, download=True, transform=tfm)
        return train, test
    except Exception as exc:
        warnings.warn(
            f"Could not download {name} ({exc}).\n"
            "Falling back to synthetic random dataset.\n"
            "For real results, ensure internet access and re-run."
        )
        from datasets.synthetic import make_synthetic_dataset
        return make_synthetic_dataset(name)


# ── IID partitioning ──────────────────────────────────────────────────────────

def iid_partition(
    dataset: Dataset,
    num_vehicles: int,
    seed: int = 42,
) -> Dict[int, List[int]]:
    """
    Randomly shuffle indices and split uniformly across vehicles.
    Each vehicle gets approximately len(dataset)/num_vehicles samples.
    """
    rng     = np.random.default_rng(seed)
    indices = rng.permutation(len(dataset)).tolist()
    shard   = max(1, len(indices) // num_vehicles)
    return {
        v: indices[v * shard: (v + 1) * shard]
        for v in range(num_vehicles)
    }


# ── Non-IID Dirichlet partitioning ────────────────────────────────────────────

def dirichlet_partition(
    dataset:      Dataset,
    num_vehicles: int,
    alpha:        float = 0.5,
    seed:         int   = 42,
    min_samples:  int   = 10,
) -> Dict[int, List[int]]:
    """
    Partition using Dirichlet(alpha) over class labels.

    Smaller alpha → more heterogeneous (each vehicle sees mostly one class).
    Each vehicle is guaranteed at least `min_samples` examples.
    """
    rng = np.random.default_rng(seed)

    # Extract labels (handles both torchvision datasets and TensorDatasets)
    if hasattr(dataset, "targets"):
        labels = np.asarray(dataset.targets)
    else:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])

    num_classes = int(labels.max()) + 1
    class_indices: Dict[int, List[int]] = {
        c: np.where(labels == c)[0].tolist() for c in range(num_classes)
    }
    for c in class_indices:
        rng.shuffle(class_indices[c])   # in-place shuffle

    # Draw per-class proportions for each vehicle
    proportions = rng.dirichlet(
        alpha * np.ones(num_vehicles), size=num_classes
    )   # shape: (num_classes, num_vehicles)

    vehicle_indices: Dict[int, List[int]] = {v: [] for v in range(num_vehicles)}
    for c in range(num_classes):
        idxs   = class_indices[c]
        splits = (proportions[c] * len(idxs)).astype(int)
        splits[-1] = len(idxs) - splits[:-1].sum()   # fix rounding
        ptr = 0
        for v in range(num_vehicles):
            vehicle_indices[v].extend(idxs[ptr: ptr + splits[v]])
            ptr += splits[v]

    # Guarantee minimum samples per vehicle
    for v in range(num_vehicles):
        while len(vehicle_indices[v]) < min_samples:
            donor = max(vehicle_indices, key=lambda x: len(vehicle_indices[x]))
            if len(vehicle_indices[donor]) <= min_samples:
                break
            vehicle_indices[v].append(vehicle_indices[donor].pop())

    return vehicle_indices


# ── DataLoader builders ───────────────────────────────────────────────────────

def build_vehicle_loaders(
    dataset:     Dataset,
    partition:   Dict[int, List[int]],
    batch_size:  int = 32,
    num_workers: int = 0,
) -> Dict[int, DataLoader]:
    """Wrap each vehicle's index list in a DataLoader."""
    return {
        v: DataLoader(
            Subset(dataset, idxs),
            batch_size  = batch_size,
            shuffle     = True,
            num_workers = num_workers,
            drop_last   = False,
        )
        for v, idxs in partition.items()
    }


def build_test_loader(
    test_dataset: Dataset,
    batch_size:   int = 256,
    num_workers:  int = 0,
) -> DataLoader:
    return DataLoader(
        test_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
    )


def get_num_classes(dataset_name: str) -> int:
    return 10   # MNIST, FashionMNIST, CIFAR10 all have 10 classes


def get_input_shape(dataset_name: str) -> Tuple[int, ...]:
    if dataset_name in ("MNIST", "FashionMNIST"):
        return (1, 28, 28)
    return (3, 32, 32)
