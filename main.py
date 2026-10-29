"""
main.py — M1 Experiment Orchestrator.

Runs Gossip-Based Federated Learning under Link Errors and Asynchrony
across topology × packet-loss × async-delay combinations, collects metrics,
and generates comparison plots.

Execution order (you train locally; this repo does not auto-train):
    python main.py                 # M1: gossip FL under imperfect channels
    python main_m2.py              # M2: event-triggered asynchronous gossip
    python main_m3.py              # M3: spectral-gap analysis (no training)

Usage:
    python main.py
    python main.py --plot-only     # regenerate plots from results/*.json
    python main.py --topologies ring --packet-loss 0.0 --num-rounds 20
"""

import os
import json
import time
import torch
import numpy as np

from config import Config, build_arg_parser, config_from_args, experiment_name
from dataset import prepare_dataloaders
from model import MNISTNet, get_model_params, get_param_bytes
from client import FLClient
from network import create_topology, get_adjacency_dict, get_mixing_matrix, get_spectral_gap
from channel import ImperfectChannel
from gossip import GossipAggregator
from metrics import MetricsLogger


def set_seed(seed: int) -> None:
    """Set random seeds for full reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_experiment(
    cfg: Config,
    topology_name: str,
    packet_loss_rate: float,
    train_loaders,
    test_loader,
    device: torch.device,
    max_async_delay: int = 0,
) -> dict:
    """
    Run a single M1 experiment for one topology, packet-loss rate, and delay.

    Returns the JSON-serializable result summary.
    """
    print(f"\n{'='*70}")
    print(
        f"  Topology: {topology_name} | Packet Loss: {packet_loss_rate} "
        f"| Async delay: {max_async_delay}"
    )
    print(f"{'='*70}")

    # ── Seed for reproducibility within each experiment
    set_seed(cfg.seed)
    for cid, loader in enumerate(train_loaders):
        if loader.generator is not None:
            loader.generator.manual_seed(cfg.seed + 1000 + cid)

    # ── Build topology
    graph = create_topology(
        topology_name, cfg.num_clients,
        p=cfg.random_edge_prob, seed=cfg.seed
    )
    adjacency = get_adjacency_dict(graph)
    W = get_mixing_matrix(graph)
    spectral_gap = get_spectral_gap(W)

    print(f"  Nodes: {graph.number_of_nodes()} | Edges: {graph.number_of_edges()}")
    print(f"  Spectral gap: {spectral_gap:.4f}")
    print(f"  Adjacency: {dict(adjacency)}")

    # ── Initialize clients with identical weights
    reference_model = MNISTNet().to(device)
    init_params = get_model_params(reference_model)
    param_bytes = get_param_bytes(reference_model)

    clients = []
    for i in range(cfg.num_clients):
        client = FLClient(
            client_id=i,
            train_loader=train_loaders[i],
            lr=cfg.learning_rate,
            momentum=cfg.momentum,
            device=device,
        )
        client.set_params(init_params.clone())
        clients.append(client)

    # ── Channel and aggregator
    channel = ImperfectChannel(
        packet_loss_rate=packet_loss_rate,
        max_async_delay=max_async_delay,
        seed=cfg.seed,
        param_bytes=param_bytes,
    )
    aggregator = GossipAggregator(clients, adjacency, W, channel)

    # ── Metrics logger
    logger = MetricsLogger(
        cfg.num_clients, topology_name, packet_loss_rate, max_async_delay
    )

    # ── Training loop
    for rnd in range(1, cfg.num_rounds + 1):
        # Local training
        for client in clients:
            client.local_train(epochs=cfg.local_epochs)

        # Gossip aggregation through imperfect channel
        aggregator.aggregate_round(rnd)

        # Record metrics
        summary = logger.record_round(rnd, clients, test_loader, channel)

        # Print progress
        if rnd % 5 == 0 or rnd == 1:
            print(
                f"  Round {rnd:3d}/{cfg.num_rounds} | "
                f"Acc: {summary['avg_accuracy']:.4f} | "
                f"Loss: {summary['avg_loss']:.4f} | "
                f"Consensus Err: {summary['consensus_error']:.6f} | "
                f"Msgs: {summary['msgs_attempted']} sent, "
                f"{summary['msgs_dropped']} dropped"
            )

    # ── Save per-experiment results
    result = logger.get_final_summary(channel)
    result["spectral_gap"] = spectral_gap
    from dataclasses import asdict
    result["config"] = asdict(cfg)
    result["mean_drift"] = aggregator.mean_drift_history

    exp_name = experiment_name(topology_name, packet_loss_rate, max_async_delay)
    json_path = os.path.join(cfg.results_dir, f"{exp_name}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"  → Saved results to {json_path}")

    ckpt_dir = os.path.join(cfg.results_dir, "checkpoints", exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    for client in clients:
        ckpt_path = os.path.join(ckpt_dir, f"client_{client.client_id}.pt")
        torch.save(client.model.state_dict(), ckpt_path)
    print(f"  → Saved {len(clients)} client checkpoints to {ckpt_dir}/")

    return result


def generate_m1_plots(all_results: list, cfg: Config) -> None:
    """Write comparison plots from already-computed M1 result dicts."""
    print("\nGenerating plots...")

    for topo in cfg.topologies_to_test:
        topo_results = [r for r in all_results if r["topology"] == topo]
        if not topo_results:
            continue
        MetricsLogger.plot_accuracy_comparison(
            topo_results,
            os.path.join(cfg.results_dir, f"accuracy_{topo}.png"),
            title=f"Test Accuracy — {topo.capitalize()} Topology",
        )

    for plr in cfg.packet_loss_rates_to_test:
        plr_results = [r for r in all_results if r["packet_loss_rate"] == plr]
        if not plr_results:
            continue
        MetricsLogger.plot_accuracy_comparison(
            plr_results,
            os.path.join(cfg.results_dir, f"accuracy_loss{plr}.png"),
            title=f"Test Accuracy — Packet Loss = {plr}",
        )

    MetricsLogger.plot_consensus_error(
        all_results,
        os.path.join(cfg.results_dir, "consensus_error_all.png"),
    )
    MetricsLogger.plot_communication_cost(
        all_results,
        os.path.join(cfg.results_dir, "communication_cost.png"),
    )
    print(f"Plots saved to {cfg.results_dir}/")


def print_m1_summary(all_results: list) -> None:
    print(f"\n{'='*70}")
    print("  M1 RESULTS SUMMARY")
    print(f"{'='*70}")
    print(
        f"{'Topology':<10} {'PL Rate':<10} {'Delay':<8} {'Final Acc':<12} "
        f"{'Consensus Err':<16} {'Spectral Gap':<14} {'Msgs Sent':<12} {'Msgs Dropped':<12}"
    )
    print("-" * 94)
    for r in all_results:
        total = r["total_communication"]
        print(
            f"{r['topology']:<10} {r['packet_loss_rate']:<10.1f} "
            f"{r.get('max_async_delay', 0):<8} "
            f"{r['final_avg_accuracy']:<12.4f} "
            f"{r['final_consensus_error']:<16.6f} "
            f"{r.get('spectral_gap', 0):<14.4f} "
            f"{total['attempted']:<12} "
            f"{total['dropped']:<12}"
        )


def load_m1_results(results_dir: str) -> list:
    combined_path = os.path.join(results_dir, "m1_all_results.json")
    if os.path.exists(combined_path):
        with open(combined_path, encoding="utf-8") as f:
            return json.load(f)
    return []


def main():
    """Run the full M1 experiment suite, or regenerate plots from saved JSON."""
    parser = build_arg_parser("M1: Gossip-based FL under link errors and asynchrony")
    args = parser.parse_args()
    cfg = config_from_args(args)
    os.makedirs(cfg.results_dir, exist_ok=True)

    if args.plot_only:
        all_results = load_m1_results(cfg.results_dir)
        if not all_results:
            print(
                f"No M1 results found at {os.path.join(cfg.results_dir, 'm1_all_results.json')}.\n"
                "Train first with: python main.py"
            )
            return
        print_m1_summary(all_results)
        if cfg.save_plots:
            generate_m1_plots(all_results, cfg)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_gpu = device.type == "cuda"

    print(f"Device: {device}")
    if use_gpu:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
    else:
        print("WARNING: No GPU detected — training on CPU (will be slow)")

    set_seed(cfg.seed)
    train_loaders, test_loader = prepare_dataloaders(
        num_clients=cfg.num_clients,
        batch_size=cfg.batch_size,
        iid=cfg.iid,
        data_dir=cfg.data_dir,
        seed=cfg.seed,
        use_gpu=use_gpu,
    )
    print(f"Dataset: {cfg.dataset_name} ({'IID' if cfg.iid else 'non-IID'})")
    print(
        f"Clients: {cfg.num_clients} | Rounds: {cfg.num_rounds} | "
        f"Local epochs: {cfg.local_epochs}"
    )
    print(f"Topologies: {cfg.topologies_to_test}")
    print(f"Packet loss: {cfg.packet_loss_rates_to_test}")
    print(f"Async delays: {cfg.async_delays_to_test}")

    all_results = []
    start_time = time.time()

    for topology_name in cfg.topologies_to_test:
        for plr in cfg.packet_loss_rates_to_test:
            for delay in cfg.async_delays_to_test:
                result = run_experiment(
                    cfg, topology_name, plr,
                    train_loaders, test_loader, device,
                    max_async_delay=delay,
                )
                all_results.append(result)

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"  All experiments complete in {elapsed:.1f}s")
    print(f"{'='*70}")

    combined_path = os.path.join(cfg.results_dir, "m1_all_results.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"Combined results saved to {combined_path}")

    if cfg.save_plots:
        generate_m1_plots(all_results, cfg)

    print_m1_summary(all_results)


if __name__ == "__main__":
    main()