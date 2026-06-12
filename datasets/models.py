"""
datasets/models.py

Neural network architectures used for FL experiments.

MNIST / FashionMNIST → LeNet-style CNN  (~21 K parameters)
CIFAR-10             → Compact ResNet   (~62 K parameters)

Both architectures are intentionally small so that 100-vehicle simulations
run in reasonable time on CPU.  They are standard enough that accuracy
numbers are comparable to published baselines.
"""

from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── LeNet-style CNN for MNIST / FashionMNIST ─────────────────────────────────

class LeNet(nn.Module):
    """
    Two conv layers followed by two fully-connected layers.
    Input: (B, 1, 28, 28)   Output: (B, 10)
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)   # → (B,16,28,28)
        self.pool  = nn.MaxPool2d(2)                                # → (B,16,14,14)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)  # → (B,32,14,14)
        # after second pool: (B,32,7,7)
        self.fc1   = nn.Linear(32 * 7 * 7, 120)
        self.fc2   = nn.Linear(120, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ── Compact ResNet for CIFAR-10 ───────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.net(x) + x)


class CompactResNet(nn.Module):
    """
    3-stage residual network for CIFAR-10.
    Input: (B, 3, 32, 32)   Output: (B, 10)
    ~62 K parameters.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.stem   = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.stage1 = ResBlock(16)
        self.down1  = nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False)
        self.stage2 = ResBlock(32)
        self.down2  = nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False)
        self.stage3 = ResBlock(64)
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.fc     = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)


# ── Factory ───────────────────────────────────────────────────────────────────

def build_model(dataset_name: str, num_classes: int = 10) -> nn.Module:
    if dataset_name in ("MNIST", "FashionMNIST"):
        return LeNet(num_classes)
    elif dataset_name == "CIFAR10":
        return CompactResNet(num_classes)
    raise ValueError(f"No model defined for dataset {dataset_name}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def get_flat_params(model: nn.Module) -> torch.Tensor:
    """Return all model parameters as a single 1-D float32 tensor."""
    return torch.cat([p.data.view(-1) for p in model.parameters()]).float()


def set_flat_params(model: nn.Module, flat: torch.Tensor) -> None:
    """Overwrite model parameters from a 1-D tensor (in-place)."""
    ptr = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[ptr: ptr + n].view(p.shape))
        ptr += n


def compute_update(
    old_params: torch.Tensor, new_params: torch.Tensor
) -> torch.Tensor:
    """wi(t) = new_params - old_params (gradient direction)."""
    return new_params - old_params


def apply_update(model: nn.Module, update: torch.Tensor) -> None:
    """W(t+1) = W(t) + update (in-place)."""
    current = get_flat_params(model)
    set_flat_params(model, current + update)
