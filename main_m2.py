"""
main_m2.py — M2 Experiment Orchestrator: Event-Triggered Asynchronous Gossip.

M2 builds on the M1 baseline by replacing the always-on synchronous gossip
protocol with an event-triggered variant: a client broadcasts its model
update only when the L2 change in its parameters since the last transmission
exceeds a configurable threshold delta.

PDF specification (Section 7 - Methodology Design):
    "After each local SGD step, a node's model state is checked by a change
     detector: if the change since the last transmission exceeds threshold d,
     the update is broadcast asynchronously to neighbors; otherwise
     transmission is suppressed to save bandwidth."

What M2 measures:
    • Communication reduction: how many transmissions are suppressed vs M1
    • Bytes saved: reduction in bytes sent compared to full gossip (M1)
    • Accuracy trade-off: does the threshold affect final test accuracy?
    • Convergence: does event-triggering slow convergence vs M1?
    • Consensus error: does reduced communication hurt synchronisation?

Experiment sweep:
    • Threshold delta in [0.0, 0.01, 0.05, 0.1, 0.3, 0.5]
      (0.0 = always transmit = equivalent to M1 synchronous gossip)
    • All 3 topologies tested (ring, grid, random)
    • Packet-loss rates [0.0, 0.2] tested

Usage:
    python main.py                 # M1 first (trains and saves results)
    python main_m2.py              # M2 event-triggered gossip (trains)
    python main_m2.py --plot-only  # plots only, from results/m2/*.json

Output:
    results/m2/m2_all_results.json
    results/m2/m2_accuracy_vs_threshold_<topology>.png
    results/m2/m2_communication_savings.png
    results/m2/m2_consensus_error_vs_threshold.png
    results/m2/m2_trigger_rate.png
    results/m2/m2_vs_m1_comparison.png
"""

import os
import json
import time
from typing import Optional

import torch
import numpy as np

from config import Config, build_arg_parser, config_from_args
from dataset import prepare_dataloaders
from model import MNISTNet, get_model_params, get_param_bytes
from client import FLClient
from network import create_topology, get_adjacency_dict, get_mixing_matrix, get_spectral_gap
from channel import ImperfectChannel
from gossip import GossipAggregator
from gossip_event_triggered import EventTriggeredGossipAggregator
from metrics import MetricsLogger


# ── M2 Configuration ──────────────────────────────────────────────────────────

M2_RESULTS_DIR = "./results/m2"
M1_RESULTS_PATH = "./results/m1_all_results.json"

# delta = 0.0 is the M1-equivalent baseline (always trigger)
DEFAULT_DELTAS = [0.0, 0.01, 0.05, 0.10, 0.30, 0.50]


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_m2_experiment(
    cfg: Config,
    delta: float,
    topology_name: str,
    packet_loss_rate: float,
    train_loaders,
    test_loader,
    device: torch.device,
    max_async_delay: int = 0,
    num_rounds: Optional[int] = None,
) -> dict:
    """
    Run one M2 experiment: event-triggered gossip with threshold *delta*.

    delta = 0.0 is identical to M1 (all transmissions triggered).
    """
    rounds = num_rounds if num_rounds is not None else cfg.num_rounds
    set_seed(cfg.seed)

    graph    = create_topology(topology_name, cfg.num_clients,
                               p=cfg.random_edge_prob, seed=cfg.seed)
    adjacency = get_adjacency_dict(graph)
    W         = get_mixing_matrix(graph)
    spectral_gap = get_spectral_gap(W)
    param_bytes  = get_param_bytes(MNISTNet())

    # Initialise clients with identical weights
    ref_params = get_model_params(MNISTNet().to(device))
    clients = []
    for i in range(cfg.num_clients):
        c = FLClient(
            client_id=i,
            train_loader=train_loaders[i],
            lr=cfg.learning_rate,
            momentum=cfg.momentum,
            device=device,
        )
        c.set_params(ref_params.clone())
        clients.append(c)

    channel = ImperfectChannel(
        packet_loss_rate=packet_loss_rate,
        max_async_delay=max_async_delay,
        seed=cfg.seed,
        param_bytes=param_bytes,
    )

    # Choose aggregator
    if delta == 0.0:
        # Baseline: always-on gossip (equivalent to M1)
        aggregator = GossipAggregator(clients, adjacency, W, channel)
        trigger_history = []
    else:
        aggregator = EventTriggeredGossipAggregator(
            clients, adjacency, W, channel, delta=delta
        )
        trigger_history = aggregator.trigger_history

    logger = MetricsLogger(
        cfg.num_clients, topology_name, packet_loss_rate, max_async_delay
    )

    total_triggered  = 0
    total_suppressed = 0

    for rnd in range(1, rounds + 1):
        # Local training
        for client in clients:
            client.local_train(epochs=cfg.local_epochs)

        # Gossip (event-triggered or full)
        if delta == 0.0:
            aggregator.aggregate_round(rnd)
            # All clients always send to all neighbors
            n_neighbors_per_client = len(adjacency.get(0, []))
            total_triggered  += cfg.num_clients * n_neighbors_per_client
        else:
            stats = aggregator.aggregate_round(rnd)
            total_triggered  += stats["triggered"]
            total_suppressed += stats["suppressed"]

        summary = logger.record_round(rnd, clients, test_loader, channel)

        if rnd % 5 == 0 or rnd == 1:
            trig_pct = (total_triggered / max(1, total_triggered + total_suppressed)) * 100
            print(
                f"  Round {rnd:3d}/{rounds} | "
                f"Acc={summary['avg_accuracy']:.4f} | "
                f"ConsErr={summary['consensus_error']:.6f} | "
                f"Triggered={total_triggered} ({trig_pct:.0f}%) | "
                f"Suppressed={total_suppressed}"
            )

    total_tx_in_m1   = channel.get_total_stats()["attempted"]
    total_bytes_sent = channel.get_total_stats()["bytes_attempted"]
    full_gossip_msgs = rounds * sum(len(v) for v in adjacency.values())
    suppression_rate = total_suppressed / max(1, total_triggered + total_suppressed)

    result = logger.get_final_summary(channel)
    result.update({
        "delta"             : delta,
        "spectral_gap"      : spectral_gap,
        "total_triggered"   : total_triggered,
        "total_suppressed"  : total_suppressed,
        "suppression_rate"  : suppression_rate,
        "full_gossip_msgs"  : full_gossip_msgs,
        "comm_reduction_pct": suppression_rate * 100.0,
        "bytes_saved_pct"   : suppression_rate * 100.0,  # proportional
        "trigger_history"   : trigger_history,
    })
    return result


def generate_m2_plots(
    all_results: list,
    m1_results: list,
    topologies: list,
    packet_loss_rates: list,
    delta_values: list,
) -> None:
    """Generate all M2 comparison plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(M2_RESULTS_DIR, exist_ok=True)

    topo_colors  = {"ring": "steelblue", "grid": "seagreen", "random": "coral"}
    delta_labels = {d: ("Baseline (δ=0)" if d == 0.0 else f"δ={d}") for d in delta_values}

    # ── 1. Accuracy vs Round for each topology (one plot per topo, lines = delta)
    for topo in topologies:
        for plr in packet_loss_rates:
            subset = [r for r in all_results
                      if r["topology"] == topo and r["packet_loss_rate"] == plr]
            if not subset:
                continue

            fig, ax = plt.subplots(figsize=(9, 5))
            # cmap = plt.cm.get_cmap("Blues", len(delta_values) + 2)
            cmap = plt.colormaps.get_cmap("Blues").resampled(len(delta_values) + 2)
            for i, r in enumerate(sorted(subset, key=lambda x: x["delta"])):
                label = delta_labels[r["delta"]]
                lw = 2.5 if r["delta"] == 0.0 else 1.5
                ls = "--" if r["delta"] == 0.0 else "-"
                ax.plot(r["per_round_accuracy"], label=label, linewidth=lw,
                        linestyle=ls, color=cmap(i + 2))
            ax.set_title(f"M2: Accuracy vs Round — {topo.capitalize()} (PLR={plr})",
                         fontsize=13)
            ax.set_xlabel("Communication Round", fontsize=11)
            ax.set_ylabel("Average Test Accuracy", fontsize=11)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            fname = f"m2_accuracy_{topo}_plr{plr}.png"
            plt.savefig(os.path.join(M2_RESULTS_DIR, fname), dpi=150)
            plt.close()
            print(f"  -> {fname}")

    # ── 2. Communication savings vs threshold (bar chart, one per topology)
    fig, axes = plt.subplots(1, max(1, len(topologies)), figsize=(14, 5), sharey=True)
    if len(topologies) == 1:
        axes = [axes]
    for ax, topo in zip(axes, topologies):
        subset = [r for r in all_results
                  if r["topology"] == topo and r["packet_loss_rate"] == 0.0]
        subset.sort(key=lambda x: x["delta"])
        deltas  = [r["delta"] for r in subset]
        savings = [r["comm_reduction_pct"] for r in subset]
        bars = ax.bar([str(d) for d in deltas], savings,
                      color=topo_colors.get(topo, "gray"), edgecolor="black")
        for bar, val in zip(bars, savings):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=8)
        ax.set_title(topo.capitalize(), fontsize=12)
        ax.set_xlabel("Threshold δ", fontsize=10)
        if ax == axes[0]:
            ax.set_ylabel("Communication Reduction (%)", fontsize=10)
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3, axis="y")
    plt.suptitle("M2: Communication Savings by Topology (PLR=0.0)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(M2_RESULTS_DIR, "m2_communication_savings.png"), dpi=150)
    plt.close()
    print("  -> m2_communication_savings.png")

    # ── 3. Consensus error vs threshold
    fig, ax = plt.subplots(figsize=(9, 5))
    for topo in topologies:
        subset = [r for r in all_results
                  if r["topology"] == topo and r["packet_loss_rate"] == 0.0]
        subset.sort(key=lambda x: x["delta"])
        deltas = [r["delta"] for r in subset]
        cerr   = [r["final_consensus_error"] for r in subset]
        ax.plot(deltas, cerr, "o-", color=topo_colors.get(topo, "gray"),
                label=topo.capitalize(), linewidth=2, markersize=7)
    ax.set_title("M2: Final Consensus Error vs Threshold δ", fontsize=13)
    ax.set_xlabel("Threshold δ", fontsize=11)
    ax.set_ylabel("Final Consensus Error", fontsize=11)
    ax.set_yscale("log")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(os.path.join(M2_RESULTS_DIR, "m2_consensus_error_vs_threshold.png"),
                dpi=150)
    plt.close()
    print("  -> m2_consensus_error_vs_threshold.png")

    # ── 4. Final accuracy vs threshold
    fig, ax = plt.subplots(figsize=(9, 5))
    for topo in topologies:
        subset = [r for r in all_results
                  if r["topology"] == topo and r["packet_loss_rate"] == 0.0]
        subset.sort(key=lambda x: x["delta"])
        deltas = [r["delta"] for r in subset]
        accs   = [r["final_avg_accuracy"] for r in subset]
        ax.plot(deltas, accs, "o-", color=topo_colors.get(topo, "gray"),
                label=topo.capitalize(), linewidth=2, markersize=7)
    ax.set_title("M2: Final Accuracy vs Threshold δ", fontsize=13)
    ax.set_xlabel("Threshold δ", fontsize=11)
    ax.set_ylabel("Final Average Test Accuracy", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.9, 1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(M2_RESULTS_DIR, "m2_accuracy_vs_threshold.png"), dpi=150)
    plt.close()
    print("  -> m2_accuracy_vs_threshold.png")

    # ── 5. Trigger rate over rounds (ring topology, PLR=0, various delta)
    fig, ax = plt.subplots(figsize=(9, 5))
    ring_results = [r for r in all_results
                    if r["topology"] == "ring" and r["packet_loss_rate"] == 0.0
                    and r["delta"] > 0.0]
    ring_results.sort(key=lambda x: x["delta"])
    # cmap = plt.cm.get_cmap("Reds", len(ring_results) + 2)
    plt.colormaps.get_cmap("Reds").resampled(len(delta_values) + 2)
    for i, r in enumerate(ring_results):
        hist = r.get("trigger_history", [])
        if hist:
            triggered_pct = [
                h["triggered"] / max(1, h["triggered"] + h["suppressed"]) * 100
                for h in hist
            ]
            ax.plot(triggered_pct, label=f"δ={r['delta']}",
                    color=cmap(i + 2), linewidth=1.8)
    ax.set_title("M2: Trigger Rate over Rounds (Ring, PLR=0.0)", fontsize=13)
    ax.set_xlabel("Round", fontsize=11)
    ax.set_ylabel("Clients Triggered (%)", fontsize=11)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(M2_RESULTS_DIR, "m2_trigger_rate.png"), dpi=150)
    plt.close()
    print("  -> m2_trigger_rate.png")

    # ── 6. M2 vs M1 comparison (if M1 results exist)
    if m1_results:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for topo in topologies:
            # M1 result (PLR=0)
            m1_r = next((r for r in m1_results
                         if r["topology"] == topo and r["packet_loss_rate"] == 0.0), None)
            # M2 baseline (delta=0) and best threshold
            m2_base = next((r for r in all_results
                            if r["topology"] == topo and r["delta"] == 0.0
                            and r["packet_loss_rate"] == 0.0), None)
            # Best threshold = highest comm reduction with acc >= M1 - 0.005
            m1_acc = m1_r["final_avg_accuracy"] if m1_r else 0
            candidates = [r for r in all_results
                          if r["topology"] == topo and r["delta"] > 0.0
                          and r["packet_loss_rate"] == 0.0
                          and r["final_avg_accuracy"] >= m1_acc - 0.005]
            m2_best = max(candidates, key=lambda x: x["comm_reduction_pct"]) \
                      if candidates else None

            color = topo_colors.get(topo, "gray")
            if m1_r:
                axes[0].plot(m1_r["per_round_accuracy"], color=color,
                             linestyle="--", linewidth=2, label=f"M1 {topo}")
            if m2_base:
                axes[0].plot(m2_base["per_round_accuracy"], color=color,
                             linestyle="-", linewidth=1.5,
                             label=f"M2 δ=0 {topo}", alpha=0.7)
            if m2_best:
                axes[0].plot(m2_best["per_round_accuracy"], color=color,
                             linestyle=":", linewidth=2,
                             label=f"M2 δ={m2_best['delta']} {topo}")

        axes[0].set_title("M2 vs M1: Accuracy Curves")
        axes[0].set_xlabel("Round")
        axes[0].set_ylabel("Average Test Accuracy")
        axes[0].legend(fontsize=7, ncol=2)
        axes[0].grid(True, alpha=0.3)

        # Bar chart: comm reduction for best threshold per topology
        topo_labels, m1_msgs, m2_msgs = [], [], []
        for topo in topologies:
            m1_r = next((r for r in m1_results
                         if r["topology"] == topo and r["packet_loss_rate"] == 0.0), None)
            m1_acc = m1_r["final_avg_accuracy"] if m1_r else 0
            candidates = [r for r in all_results
                          if r["topology"] == topo and r["delta"] > 0.0
                          and r["packet_loss_rate"] == 0.0
                          and r["final_avg_accuracy"] >= m1_acc - 0.005]
            m2_best = max(candidates, key=lambda x: x["comm_reduction_pct"]) \
                      if candidates else None
            if m1_r and m2_best:
                topo_labels.append(topo.capitalize())
                m1_msgs.append(m1_r["total_communication"]["attempted"])
                m2_msgs.append(m2_best["total_communication"]["attempted"])

        x = np.arange(len(topo_labels))
        w = 0.35
        axes[1].bar(x - w / 2, m1_msgs, w, label="M1 (Full Gossip)",
                    color="steelblue")
        axes[1].bar(x + w / 2, m2_msgs, w, label="M2 (Event-Triggered)",
                    color="seagreen")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(topo_labels)
        axes[1].set_title("M2 vs M1: Messages Transmitted")
        axes[1].set_ylabel("Total Messages")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis="y")

        plt.suptitle("M2 Event-Triggered Gossip vs M1 Full Gossip", fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(M2_RESULTS_DIR, "m2_vs_m1_comparison.png"), dpi=150)
        plt.close()
        print("  -> m2_vs_m1_comparison.png")


def print_m2_summary(all_results: list) -> None:
    """Print M2 summary table."""
    print(f"\n{'='*80}")
    print("  M2 RESULTS SUMMARY — Event-Triggered Gossip")
    print(f"{'='*80}")
    print(f"{'Topology':<10} {'PLR':<6} {'Delta':<8} {'FinalAcc':<12} "
          f"{'ConsErr':<14} {'CommReduc%':<12} {'Triggered':<12} {'Suppressed'}")
    print("-" * 80)
    for r in sorted(all_results, key=lambda x: (x["topology"], x["packet_loss_rate"],
                                                  x["delta"])):
        print(
            f"{r['topology']:<10} {r['packet_loss_rate']:<6} {r['delta']:<8.2f} "
            f"{r['final_avg_accuracy']:<12.4f} "
            f"{r['final_consensus_error']:<14.6f} "
            f"{r['comm_reduction_pct']:<12.1f} "
            f"{r['total_triggered']:<12} "
            f"{r['total_suppressed']}"
        )


def main():
    """Run full M2 event-triggered gossip experiment suite, or plot-only."""
    parser = build_arg_parser("M2: Event-triggered asynchronous gossip")
    parser.add_argument(
        "--deltas", nargs="+", type=float, default=None,
        help="Trigger thresholds to sweep (default: 0 0.01 0.05 0.1 0.3 0.5)",
    )
    args = parser.parse_args()
    cfg = config_from_args(args)
    delta_values = list(args.deltas) if args.deltas is not None else list(DEFAULT_DELTAS)
    topologies = cfg.topologies_to_test
    packet_loss_rates = cfg.packet_loss_rates_to_test
    os.makedirs(M2_RESULTS_DIR, exist_ok=True)

    m1_results = []
    if os.path.exists(M1_RESULTS_PATH):
        with open(M1_RESULTS_PATH, encoding="utf-8") as f:
            m1_results = json.load(f)

    if args.plot_only:
        out_path = os.path.join(M2_RESULTS_DIR, "m2_all_results.json")
        if not os.path.exists(out_path):
            print(f"No M2 results found at {out_path}.\nTrain first with: python main_m2.py")
            return
        with open(out_path, encoding="utf-8") as f:
            all_results = json.load(f)
        print_m2_summary(all_results)
        if cfg.save_plots:
            print(f"\nGenerating M2 plots in {M2_RESULTS_DIR}/...")
            generate_m2_plots(all_results, m1_results, topologies, packet_loss_rates, delta_values)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_gpu = device.type == "cuda"

    print(f"\n{'='*70}")
    print("  M2: EVENT-TRIGGERED ASYNCHRONOUS GOSSIP EXPERIMENTS")
    print(f"{'='*70}")
    print(f"  Device : {device}")
    if use_gpu:
        torch.backends.cudnn.benchmark = True
    print(f"  Rounds : {cfg.num_rounds} | Topologies: {topologies}")
    print(f"  Deltas : {delta_values}")
    print(f"  PLRs   : {packet_loss_rates}")
    print(f"  Async  : {cfg.async_delays_to_test}")

    set_seed(cfg.seed)
    train_loaders, test_loader = prepare_dataloaders(
        num_clients=cfg.num_clients,
        batch_size=cfg.batch_size,
        iid=cfg.iid,
        data_dir=cfg.data_dir,
        seed=cfg.seed,
        use_gpu=use_gpu,
    )
    print(f"  Dataset: {cfg.dataset_name} ({'IID' if cfg.iid else 'non-IID'})")

    if m1_results:
        print(f"  Loaded {len(m1_results)} M1 results for comparison")
    else:
        print("  M1 results not found — run python main.py first for M1 vs M2 plots")

    all_results = []
    start_time  = time.time()
    total_runs  = (
        len(delta_values) * len(topologies) * len(packet_loss_rates)
        * len(cfg.async_delays_to_test)
    )
    run_idx     = 0

    for topology_name in topologies:
        for plr in packet_loss_rates:
            for delay in cfg.async_delays_to_test:
                for delta in delta_values:
                    run_idx += 1
                    label = "Baseline (full gossip)" if delta == 0.0 \
                            else f"Event-triggered δ={delta}"
                    print(f"\n[{run_idx}/{total_runs}] "
                          f"Topology={topology_name} PLR={plr} delay={delay} {label}")
                    print(f"  {'─'*60}")

                    result = run_m2_experiment(
                        cfg=cfg,
                        delta=delta,
                        topology_name=topology_name,
                        packet_loss_rate=plr,
                        train_loaders=train_loaders,
                        test_loader=test_loader,
                        device=device,
                        max_async_delay=delay,
                        num_rounds=cfg.num_rounds,
                    )
                    all_results.append(result)
                    print(
                        f"  Final Acc={result['final_avg_accuracy']:.4f} | "
                        f"CommReduc={result['comm_reduction_pct']:.1f}% | "
                        f"Triggered={result['total_triggered']} "
                        f"Suppressed={result['total_suppressed']}"
                    )

    elapsed = time.time() - start_time
    print(f"\n  All M2 experiments complete in {elapsed:.1f}s")

    out_path = os.path.join(M2_RESULTS_DIR, "m2_all_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Results saved to {out_path}")

    print_m2_summary(all_results)

    if cfg.save_plots:
        print(f"\nGenerating M2 plots in {M2_RESULTS_DIR}/...")
        generate_m2_plots(all_results, m1_results, topologies, packet_loss_rates, delta_values)

    print(f"\n{'='*70}")
    print(f"  M2 complete. All outputs in: {M2_RESULTS_DIR}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
