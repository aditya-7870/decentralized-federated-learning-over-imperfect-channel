"""
config.py — Centralized configuration for M1–M3.

All hyperparameters, topology settings, channel impairment parameters, and
reproducibility controls are defined here as a single dataclass. CLI flags
override these defaults without changing source files.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import List


VALID_TOPOLOGIES = ("ring", "grid", "random")


@dataclass
class Config:
    """Master configuration for D-FL experiments."""

    # ── Reproducibility ──────────────────────────────────────────────────
    seed: int = 42

    # ── Training ─────────────────────────────────────────────────────────
    num_clients: int = 5
    num_rounds: int = 20
    local_epochs: int = 2
    batch_size: int = 64
    learning_rate: float = 0.01
    momentum: float = 0.9

    # ── Dataset ──────────────────────────────────────────────────────────
    dataset_name: str = "MNIST"
    data_dir: str = "./data"
    iid: bool = True                 # True → IID partition, False → non-IID

    # ── Topology ─────────────────────────────────────────────────────────
    topology: str = "ring"
    random_edge_prob: float = 0.5    # p for Erdős–Rényi random graph

    # ── Imperfect channel ────────────────────────────────────────────────
    packet_loss_rate: float = 0.0    # probability of dropping an update
    max_async_delay: int = 0         # max random delay in rounds (0 = synchronous)

    # ── Experiment sweep ─────────────────────────────────────────────────
    topologies_to_test: List[str] = field(
        default_factory=lambda: ["ring", "grid", "random"]
    )
    packet_loss_rates_to_test: List[float] = field(
        default_factory=lambda: [0.0, 0.2]
    )
    async_delays_to_test: List[int] = field(
        default_factory=lambda: [0]
    )

    # ── Output ───────────────────────────────────────────────────────────
    results_dir: str = "./results/revised"
    save_plots: bool = True

    def __post_init__(self):
        """Basic validation."""
        assert 0.0 <= self.packet_loss_rate <= 1.0, "packet_loss_rate must be in [0, 1]"
        assert self.max_async_delay >= 0, "max_async_delay must be non-negative"
        assert self.num_clients > 0, "Need at least 1 client"
        assert self.topology in VALID_TOPOLOGIES, (
            f"Unknown topology '{self.topology}'"
        )
        for t in self.topologies_to_test:
            assert t in VALID_TOPOLOGIES, f"Unknown topology '{t}'"
        for d in self.async_delays_to_test:
            assert d >= 0, "async delays must be non-negative"


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    """Shared CLI for M1 / M2 / M3 entry points."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--num-clients", type=int, default=None)
    p.add_argument("--num-rounds", type=int, default=None)
    p.add_argument("--local-epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--momentum", type=float, default=None)
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--results-dir", type=str, default=None)
    p.add_argument("--non-iid", action="store_true",
                   help="Use shard-based non-IID data partition")
    p.add_argument("--iid", action="store_true",
                   help="Force IID partition (default)")
    p.add_argument("--topologies", nargs="+", choices=VALID_TOPOLOGIES, default=None)
    p.add_argument("--packet-loss", nargs="+", type=float, default=None,
                   help="Packet-loss rates to sweep, e.g. 0.0 0.2")
    p.add_argument("--async-delay", nargs="+", type=int, default=None,
                   help="Max async delays (rounds) to sweep, e.g. 0 2")
    p.add_argument("--edge-prob", type=float, default=None,
                   help="Erdős–Rényi edge probability for the random topology")
    p.add_argument("--plot-only", action="store_true",
                   help="Regenerate plots from saved JSON; do not train")
    p.add_argument("--no-plots", action="store_true",
                   help="Skip writing PNG files after a training run")
    return p


def config_from_args(args: argparse.Namespace) -> Config:
    """Build a Config, applying only the flags the user actually passed."""
    cfg = Config()
    if args.seed is not None:
        cfg.seed = args.seed
    if args.num_clients is not None:
        cfg.num_clients = args.num_clients
    if args.num_rounds is not None:
        cfg.num_rounds = args.num_rounds
    if args.local_epochs is not None:
        cfg.local_epochs = args.local_epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.learning_rate = args.lr
    if args.momentum is not None:
        cfg.momentum = args.momentum
    if args.data_dir is not None:
        cfg.data_dir = args.data_dir
    if args.results_dir is not None:
        cfg.results_dir = args.results_dir
    if args.non_iid:
        cfg.iid = False
    if args.iid:
        cfg.iid = True
    if args.topologies is not None:
        cfg.topologies_to_test = list(args.topologies)
        cfg.topology = args.topologies[0]
    if args.packet_loss is not None:
        cfg.packet_loss_rates_to_test = list(args.packet_loss)
        cfg.packet_loss_rate = args.packet_loss[0]
    if args.async_delay is not None:
        cfg.async_delays_to_test = list(args.async_delay)
        cfg.max_async_delay = args.async_delay[0]
    if args.edge_prob is not None:
        cfg.random_edge_prob = args.edge_prob
    if args.no_plots:
        cfg.save_plots = False
    cfg.__post_init__()
    if cfg.num_rounds <= 0 or cfg.local_epochs <= 0 or cfg.batch_size <= 0:
        raise ValueError("Rounds, epochs and batch size must be positive")
    if any(not 0 <= v <= 1 for v in cfg.packet_loss_rates_to_test):
        raise ValueError("Loss rates must be in [0, 1]")
    return cfg


def experiment_name(topology: str, packet_loss_rate: float, max_async_delay: int = 0) -> str:
    """Stable filename stem for one (topology, loss, delay) run."""
    if max_async_delay:
        return f"{topology}_loss{packet_loss_rate}_delay{max_async_delay}"
    return f"{topology}_loss{packet_loss_rate}"
