"""
metrics.py — Metrics logger and plotting utilities for M1 experiments.

Tracks per-round:
  • Test accuracy and loss per client and global average
  • Consensus error: (1/N) Σ ‖wᵢ − w̄‖²
  • Communication cost: messages attempted/delivered/dropped, bytes
  • Packet-loss statistics

Saves results to JSON and generates matplotlib plots.
"""

import json
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple

from client import FLClient
from channel import ImperfectChannel
from torch.utils.data import DataLoader


class MetricsLogger:
    """
    Collects and stores per-round metrics for a single experiment run.

    Parameters
    ----------
    num_clients : int
        Number of FL clients.
    topology : str
        Topology name (for labelling).
    packet_loss_rate : float
        Configured packet-loss probability.
    """

    def __init__(
        self,
        num_clients: int,
        topology: str,
        packet_loss_rate: float,
        max_async_delay: int = 0,
    ):
        self.num_clients = num_clients
        self.topology = topology
        self.packet_loss_rate = packet_loss_rate
        self.max_async_delay = max_async_delay

        # Per-round storage
        self.rounds: List[int] = []
        self.client_accuracies: List[List[float]] = []   # [round][client]
        self.client_losses: List[List[float]] = []       # [round][client]
        self.avg_accuracies: List[float] = []
        self.avg_losses: List[float] = []
        self.consensus_errors: List[float] = []
        self.comm_stats: List[Dict[str, int]] = []

    # ── Recording ────────────────────────────────────────────────────────

    def record_round(
        self,
        round_num: int,
        clients: List[FLClient],
        test_loader: DataLoader,
        channel: ImperfectChannel,
    ) -> Dict[str, float]:
        """
        Evaluate all clients, compute consensus error, log communication
        stats, and store everything.

        Returns a summary dict for quick printing.
        """
        self.rounds.append(round_num)

        # ── Per-client accuracy & loss
        accs = []
        losses = []
        for client in clients:
            acc, loss = client.evaluate(test_loader)
            accs.append(acc)
            losses.append(loss)

        self.client_accuracies.append(accs)
        self.client_losses.append(losses)
        avg_acc = float(np.mean(accs))
        avg_loss = float(np.mean(losses))
        self.avg_accuracies.append(avg_acc)
        self.avg_losses.append(avg_loss)

        # ── Consensus error: (1/N) Σ ‖wᵢ − w̄‖²
        all_params = torch.stack([c.get_params() for c in clients])
        mean_params = all_params.mean(dim=0)
        diffs = all_params - mean_params.unsqueeze(0)
        consensus_err = float((diffs ** 2).sum(dim=1).mean())
        self.consensus_errors.append(consensus_err)

        # ── Communication stats for this round
        stats = channel.get_round_stats(round_num)
        self.comm_stats.append(stats)

        return {
            "round": round_num,
            "avg_accuracy": avg_acc,
            "avg_loss": avg_loss,
            "consensus_error": consensus_err,
            "msgs_attempted": stats["attempted"],
            "msgs_dropped": stats["dropped"],
        }

    # ── Summaries ────────────────────────────────────────────────────────

    def get_final_summary(self, channel: ImperfectChannel) -> Dict:
        """Return a JSON-serializable summary of the experiment."""
        total_stats = channel.get_total_stats()
        return {
            "topology": self.topology,
            "packet_loss_rate": self.packet_loss_rate,
            "max_async_delay": self.max_async_delay,
            "num_rounds": len(self.rounds),
            "final_avg_accuracy": self.avg_accuracies[-1] if self.avg_accuracies else 0,
            "final_avg_loss": self.avg_losses[-1] if self.avg_losses else 0,
            "final_consensus_error": self.consensus_errors[-1] if self.consensus_errors else 0,
            "total_communication": total_stats,
            "per_round_accuracy": self.avg_accuracies,
            "per_round_consensus_error": self.consensus_errors,
            "per_round_comm_stats": self.comm_stats,
        }

    # ── Persistence ──────────────────────────────────────────────────────

    def save_json(self, filepath: str, channel: ImperfectChannel) -> None:
        """Save the full experiment summary to a JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.get_final_summary(channel), f, indent=2)

    # ── Plotting ─────────────────────────────────────────────────────────

    @staticmethod
    def plot_accuracy_comparison(
        results: List[Dict],
        save_path: str,
        title: str = "Test Accuracy vs Communication Round",
    ) -> None:
        """
        Plot accuracy curves for multiple experiment configurations.

        Parameters
        ----------
        results : list of dict
            Each dict must have keys ``topology``, ``packet_loss_rate``,
            and ``per_round_accuracy``.
        save_path : str
            File path to save the figure.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.figure(figsize=(10, 6))
        for r in results:
            delay = r.get("max_async_delay", 0)
            delay_tag = f", delay={delay}" if delay else ""
            label = f"{r['topology']} (loss={r['packet_loss_rate']}{delay_tag})"
            plt.plot(r["per_round_accuracy"], label=label)
        plt.xlabel("Communication Round")
        plt.ylabel("Test Accuracy")
        plt.title(title)
        plt.legend(fontsize=8, ncol=2)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

    @staticmethod
    def plot_consensus_error(
        results: List[Dict],
        save_path: str,
        title: str = "Consensus Error vs Communication Round",
    ) -> None:
        """Plot consensus error curves for multiple configurations."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.figure(figsize=(10, 6))
        for r in results:
            delay = r.get("max_async_delay", 0)
            delay_tag = f", delay={delay}" if delay else ""
            label = f"{r['topology']} (loss={r['packet_loss_rate']}{delay_tag})"
            plt.plot(r["per_round_consensus_error"], label=label)
        plt.xlabel("Communication Round")
        plt.ylabel("Consensus Error")
        plt.title(title)
        plt.legend(fontsize=8, ncol=2)
        plt.grid(True, alpha=0.3)
        plt.yscale("log")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

    @staticmethod
    def plot_communication_cost(
        results: List[Dict],
        save_path: str,
        title: str = "Communication Cost Comparison",
    ) -> None:
        """Bar chart comparing total messages and bytes across configurations."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        labels = []
        attempted = []
        delivered = []

        for r in results:
            delay = r.get("max_async_delay", 0)
            delay_tag = f"\ndelay={delay}" if delay else ""
            labels.append(f"{r['topology']}\nloss={r['packet_loss_rate']}{delay_tag}")
            total = r["total_communication"]
            attempted.append(total["attempted"])
            delivered.append(total.get("delivered", total["delivered_immediate"] + total.get("delayed", 0)))

        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width / 2, attempted, width, label="Attempted", color="steelblue")
        ax.bar(x + width / 2, delivered, width, label="Delivered", color="seagreen")
        ax.set_xlabel("Configuration")
        ax.set_ylabel("Total Messages")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
