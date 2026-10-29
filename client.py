"""
client.py — Federated Learning client (virtual device).

Each FLClient holds its own MNISTNet, a local DataLoader, and exposes
methods for local SGD training, parameter I/O, and evaluation on the
global test set.

GPU acceleration:
    - Model and data are moved to the specified device (CPU or CUDA).
    - When on CUDA, AMP (Automatic Mixed Precision) is used for ~2x
      speedup on RTX GPUs with negligible accuracy impact.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from typing import Tuple

from model import MNISTNet, get_model_params, set_model_params


class FLClient:
    """
    A single federated-learning client.

    Parameters
    ----------
    client_id : int
        Unique identifier for this client.
    train_loader : DataLoader
        Local training data.
    lr : float
        Learning rate for SGD.
    momentum : float
        Momentum for SGD.
    device : torch.device
        CPU or CUDA device.
    """

    def __init__(
        self,
        client_id: int,
        train_loader: DataLoader,
        lr: float = 0.01,
        momentum: float = 0.9,
        device: torch.device = torch.device("cpu"),
        model_factory=MNISTNet,
    ):
        self.client_id = client_id
        self.train_loader = train_loader
        self.device = device
        self._use_amp = device.type == "cuda"   # AMP only on GPU

        # Model and optimizer
        self.model = model_factory().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=lr, momentum=momentum
        )
        self.criterion = nn.CrossEntropyLoss()

        # GradScaler for AMP — no-op on CPU (enabled=False)
        self.scaler = GradScaler("cuda", enabled=self._use_amp)

    # ── Local training ───────────────────────────────────────────────────

    def local_train(self, epochs: int = 1) -> float:
        """
        Run SGD on local data for *epochs* epochs.

        Uses AMP (FP16) automatically when running on a CUDA device.
        Returns the average training loss over the last epoch.
        """
        self.model.train()
        loss_sum = 0.0
        n_batches = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_batches = 0
            for images, labels in self.train_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                self.optimizer.zero_grad(set_to_none=True)  # faster than zero_grad()

                # ── AMP forward pass
                with autocast("cuda", enabled=self._use_amp):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                # ── AMP backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                epoch_loss += loss.item()
                epoch_batches += 1

            # Track the last epoch's stats
            if epoch == epochs - 1:
                loss_sum = epoch_loss
                n_batches = epoch_batches

        avg_loss = loss_sum / max(n_batches, 1)
        return avg_loss

    # ── Parameter access ─────────────────────────────────────────────────

    def get_params(self) -> torch.Tensor:
        """Return a flat 1-D copy of all model parameters."""
        return get_model_params(self.model)

    def set_params(self, flat_params: torch.Tensor) -> None:
        """Overwrite model parameters from a flat 1-D tensor."""
        set_model_params(self.model, flat_params)

    # ── Evaluation ───────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader) -> Tuple[float, float]:
        """
        Evaluate the model on *test_loader*.

        Returns
        -------
        accuracy : float
            Fraction of correct predictions.
        avg_loss : float
            Average cross-entropy loss.
        """
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        n_batches = 0

        for images, labels in test_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with autocast("cuda", enabled=self._use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            n_batches += 1

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / max(total, 1)
        return accuracy, avg_loss

