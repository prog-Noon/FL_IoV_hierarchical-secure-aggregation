"""
datasets/synthetic.py

Generates a synthetic image dataset matching the API of MNIST/CIFAR10
so that all experiment code runs without network access.

Used for:
  - CI smoke tests
  - Offline development
  - Environments where torchvision download is blocked

The synthetic dataset uses random tensors with random integer labels.
Test accuracy will be ~10% (random chance), which is expected and correct.
"""

from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import TensorDataset


def make_synthetic_dataset(
    dataset_name: str,
    n_train:      int = 2000,
    n_test:       int = 400,
    seed:         int = 0,
) -> tuple:
    """
    Return (train_dataset, test_dataset) with .targets attribute
    for compatibility with the partitioner.
    """
    rng = np.random.default_rng(seed)

    if dataset_name in ("MNIST", "FashionMNIST"):
        C, H, W = 1, 28, 28
    elif dataset_name == "CIFAR10":
        C, H, W = 3, 32, 32
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    N_CLS = 10

    X_tr = torch.from_numpy(rng.random((n_train, C, H, W), dtype=np.float32))
    y_tr = torch.from_numpy(rng.integers(0, N_CLS, n_train).astype(np.int64))
    X_te = torch.from_numpy(rng.random((n_test,  C, H, W), dtype=np.float32))
    y_te = torch.from_numpy(rng.integers(0, N_CLS, n_test ).astype(np.int64))

    train_ds = TensorDataset(X_tr, y_tr)
    test_ds  = TensorDataset(X_te, y_te)

    # Attach .targets so dirichlet_partition / iid_partition work unchanged
    train_ds.targets = y_tr.numpy()   # type: ignore[attr-defined]
    test_ds.targets  = y_te.numpy()   # type: ignore[attr-defined]

    return train_ds, test_ds
