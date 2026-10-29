"""
model.py — Lightweight CNN for MNIST classification.

Provides the network architecture and helper utilities to flatten / unflatten
model parameters into a single vector, which is required for gossip-based
neighbor averaging.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import List


class MNISTNet(nn.Module):
    """
    Lightweight CNN for MNIST (28×28 grayscale images, 10 classes).

    Architecture:
        Conv(1→16, 3×3) → ReLU → MaxPool(2)
        Conv(16→32, 3×3) → ReLU → MaxPool(2)
        FC(32*5*5 → 128) → ReLU
        FC(128 → 10)
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))   # (B, 16, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))   # (B, 32,  7,  7)
        x = x.view(x.size(0), -1)              # (B, 32*7*7)
        x = F.relu(self.fc1(x))                # (B, 128)
        x = self.fc2(x)                        # (B, 10)
        return x


# ── Parameter vector helpers ─────────────────────────────────────────────────


def get_model_params(model: nn.Module) -> torch.Tensor:
    """Return all model parameters concatenated into a single 1-D tensor."""
    return torch.cat([p.data.clone().detach().flatten() for p in model.parameters()])


def set_model_params(model: nn.Module, flat_params: torch.Tensor) -> None:
    """
    Overwrite model parameters from a single flat 1-D tensor.

    The tensor is split according to each parameter's original shape
    and loaded in the same order as ``model.parameters()``.
    """
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(flat_params[offset: offset + numel].view(p.shape))
        offset += numel


def get_param_count(model: nn.Module) -> int:
    """Return the total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters())


def get_param_bytes(model: nn.Module) -> int:
    """Return the size in bytes of the flat parameter vector (float32)."""
    return get_param_count(model) * 4  # 4 bytes per float32
