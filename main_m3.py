"""
main_m3.py — M3 Experiment Orchestrator: Spectral Gap Analysis.

PDF specification (Section 7 - M3):
    "For each candidate topology (ring, grid, random graph), a row-stochastic
     mixing matrix W is constructed and its eigenvalues computed, yielding the
     spectral gap 1 - |lambda_2(W)|. This value feeds a theoretical convergence
     bound, which is compared against the convergence rate and consensus error
     observed empirically from a gossip simulation run on the same topology.
     The validation step checks whether the predicted ordering and rate of
     convergence across topologies matches what is actually observed, including
     under the packet-loss and asynchrony conditions introduced in M1."

What M3 produces:
    1. Mixing matrix W and full eigenvalue spectrum for each topology
    2. SLEM and spectral gap for ring / grid / random (n=5 nodes)
    3. Theoretical convergence bound E_t = E_0 * SLEM^t
    4. Empirical gossip simulation on the same topologies
    5. Validation: does spectral ordering predict M1 convergence ordering?
    6. Size sweep: spectral gap vs number of nodes
    7. Mixing time heatmap across topologies and epsilon targets

No model training required — M3 is pure graph/spectral analysis.

Usage:
    python main_m3.py                 # spectral analysis (no model training)
    python main_m3.py --plot-only     # regenerate plots from saved JSON

Output:
    results/m3/m3_spectral_profiles.json
    results/m3/m3_spectral_gap_bar.png
    results/m3/m3_eigenvalue_spectrum.png
    results/m3/m3_convergence_theoretical.png
    results/m3/m3_convergence_empirical.png
    results/m3/m3_theory_vs_empirical.png
    results/m3/m3_spectral_gap_vs_size.png
    results/m3/m3_mixing_time_heatmap.png
    results/m3/m3_m1_validation.png          (only if M1 results exist)
    results/m3/m3_report.md
"""

import os
import json
import time
import numpy as np

from config import Config, build_arg_parser, config_from_args
from network import create_topology, get_mixing_matrix
from spectral_analysis import (
    profile_topology,
    sweep_sizes,
    compute_eigenvalues,
    compute_slem,
    compute_spectral_gap,
    compute_mixing_time,
    compute_theoretical_convergence,
    simulate_gossip_consensus,
)


# ── M3 Configuration ──────────────────────────────────────────────────────────

M3_RESULTS_DIR  = "./results/m3"
M1_RESULTS_PATH = "./results/m1_all_results.json"
TOPOLOGIES      = ["ring", "grid", "random"]
NODE_SIZES      = [4, 5, 8, 10, 16, 20]
SIM_ROUNDS      = 60
EPSILON_TARGETS = [1e-2, 1e-3, 1e-4, 1e-6]
TOPO_COLORS     = {"ring": "steelblue", "grid": "seagreen", "random": "coral"}


# ── Plotting ──────────────────────────────────────────────────────────────────

def generate_m3_plots(profiles: list, size_sweep: list, m1_results: list,
                       cfg: Config) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(M3_RESULTS_DIR, exist_ok=True)
    main_profiles = [p for p in profiles if p["n_nodes"] == cfg.num_clients]

    # ── 1. Spectral gap bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [p["topology"].capitalize() for p in main_profiles]
    gaps   = [p["spectral_gap"] for p in main_profiles]
    slems  = [p["slem"]         for p in main_profiles]
    colors = [TOPO_COLORS.get(p["topology"], "gray") for p in main_profiles]
    x      = np.arange(len(labels))
    w      = 0.38

    bars1 = ax.bar(x - w/2, gaps,  w, label="Spectral Gap (1-|λ₂|)",
                   color=colors, edgecolor="black")
    bars2 = ax.bar(x + w/2, slems, w, label="SLEM |λ₂(W)|",
                   color=colors, edgecolor="black", alpha=0.5, hatch="//")
    for bar, val in zip(bars1, gaps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", fontsize=9, fontweight="bold")
    for bar, val in zip(bars2, slems):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Value", fontsize=11)
    ax.set_title(f"M3: Spectral Gap and SLEM by Topology (n={cfg.num_clients})",
                 fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(M3_RESULTS_DIR, "m3_spectral_gap_bar.png"), dpi=150)
    plt.close()
    print("  -> m3_spectral_gap_bar.png")

    # ── 2. Full eigenvalue spectrum
    fig, ax = plt.subplots(figsize=(9, 5))
    for p in main_profiles:
        evals = sorted(np.array(p["all_eigenvalues"]), reverse=True)
        color = TOPO_COLORS.get(p["topology"], "gray")
        ax.plot(evals, "o-", color=color, label=p["topology"].capitalize(),
                linewidth=2, markersize=6)
        ax.axhline(p["slem"], color=color, linestyle=":", linewidth=1, alpha=0.5)
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.2, label="|λ₁|=1")
    ax.set_xlabel("Eigenvalue Index i", fontsize=11)
    ax.set_ylabel("|λᵢ(W)|", fontsize=11)
    ax.set_title("M3: Mixing Matrix Eigenvalue Spectrum", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(M3_RESULTS_DIR, "m3_eigenvalue_spectrum.png"), dpi=150)
    plt.close()
    print("  -> m3_eigenvalue_spectrum.png")

    # ── 3. Theoretical convergence curves
    fig, ax = plt.subplots(figsize=(9, 5))
    rounds = np.arange(SIM_ROUNDS + 1)
    for p in main_profiles:
        curve = compute_theoretical_convergence(p["spectral_gap"], SIM_ROUNDS)
        color = TOPO_COLORS.get(p["topology"], "gray")
        label = (f"{p['topology'].capitalize()} "
                 f"(γ={p['spectral_gap']:.4f}, SLEM={p['slem']:.4f})")
        ax.semilogy(rounds, curve, color=color, linewidth=2.5, label=label)
    ax.set_xlabel("Gossip Round", fontsize=11)
    ax.set_ylabel("Consensus Error (log scale)", fontsize=11)
    ax.set_title(f"M3: Theoretical Convergence Bound E_t = E₀ · SLEM^t\n(n={cfg.num_clients})",
                 fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(os.path.join(M3_RESULTS_DIR, "m3_convergence_theoretical.png"), dpi=150)
    plt.close()
    print("  -> m3_convergence_theoretical.png")

    # ── 4. Empirical gossip simulation
    fig, ax = plt.subplots(figsize=(9, 5))
    for p in main_profiles:
        graph = create_topology(p["topology"], p["n_nodes"],
                                p=0.5, seed=42)
        W     = get_mixing_matrix(graph)
        emp   = simulate_gossip_consensus(W, SIM_ROUNDS)
        color = TOPO_COLORS.get(p["topology"], "gray")
        label = (f"{p['topology'].capitalize()} (empirical, "
                 f"γ={p['spectral_gap']:.4f})")
        ax.semilogy(emp, color=color, linewidth=2.5, label=label)
    ax.set_xlabel("Gossip Round", fontsize=11)
    ax.set_ylabel("Consensus Error = Var(x) (log scale)", fontsize=11)
    ax.set_title("M3: Empirical Gossip Consensus Simulation", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(os.path.join(M3_RESULTS_DIR, "m3_convergence_empirical.png"), dpi=150)
    plt.close()
    print("  -> m3_convergence_empirical.png")

    # ── 5. Theory vs Empirical comparison (one subplot per topology)
    fig, axes = plt.subplots(1, len(main_profiles),
                              figsize=(5 * len(main_profiles), 5))
    if len(main_profiles) == 1:
        axes = [axes]
    for ax, p in zip(axes, main_profiles):
        graph = create_topology(p["topology"], p["n_nodes"], p=0.5, seed=42)
        W     = get_mixing_matrix(graph)
        emp   = simulate_gossip_consensus(W, SIM_ROUNDS)
        # normalise theory to match empirical starting point
        theory = compute_theoretical_convergence(p["spectral_gap"], SIM_ROUNDS,
                                                  initial_error=emp[0])
        rounds = np.arange(SIM_ROUNDS + 1)
        color  = TOPO_COLORS.get(p["topology"], "gray")
        ax.semilogy(rounds, emp,    color=color, linewidth=2.5,
                    label="Empirical simulation")
        ax.semilogy(rounds, theory, color="black", linewidth=1.5,
                    linestyle="--", label=f"Theory: E₀·SLEM^t")
        ax.set_title(f"{p['topology'].capitalize()}\n"
                     f"γ={p['spectral_gap']:.4f}  SLEM={p['slem']:.4f}")
        ax.set_xlabel("Round")
        ax.set_ylabel("Consensus Error")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, which="both")
    plt.suptitle("M3: Empirical vs Theoretical Convergence per Topology", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(M3_RESULTS_DIR, "m3_theory_vs_empirical.png"), dpi=150)
    plt.close()
    print("  -> m3_theory_vs_empirical.png")

    # ── 6. Spectral gap vs network size
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for topo in TOPOLOGIES:
        pts = [p for p in size_sweep if p["topology"] == topo]
        pts.sort(key=lambda x: x["n_nodes"])
        ns   = [p["n_nodes"]     for p in pts]
        gaps = [p["spectral_gap"] for p in pts]
        tmix = [p["mixing_time_1e3"] for p in pts]
        color = TOPO_COLORS.get(topo, "gray")
        axes[0].plot(ns, gaps, "o-", color=color, label=topo.capitalize(),
                     linewidth=2, markersize=6)
        valid = [(n, t) for n, t in zip(ns, tmix) if np.isfinite(t)]
        if valid:
            vn, vt = zip(*valid)
            axes[1].plot(vn, vt, "o-", color=color, label=topo.capitalize(),
                         linewidth=2, markersize=6)

    axes[0].set_title("Spectral Gap vs Network Size")
    axes[0].set_xlabel("Number of Nodes n")
    axes[0].set_ylabel("Spectral Gap γ")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Mixing Time (ε=10⁻³) vs Network Size")
    axes[1].set_xlabel("Number of Nodes n")
    axes[1].set_ylabel("Rounds to ε-Consensus")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.suptitle("M3: Spectral Properties vs Network Size", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(M3_RESULTS_DIR, "m3_spectral_gap_vs_size.png"), dpi=150)
    plt.close()
    print("  -> m3_spectral_gap_vs_size.png")

    # ── 7. Mixing time heatmap
    fig, ax = plt.subplots(figsize=(9, 4))
    t_matrix = np.zeros((len(main_profiles), len(EPSILON_TARGETS)))
    for i, p in enumerate(main_profiles):
        for j, eps in enumerate(EPSILON_TARGETS):
            t = compute_mixing_time(p["spectral_gap"], eps)
            t_matrix[i, j] = t if np.isfinite(t) else 999

    im = ax.imshow(t_matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(EPSILON_TARGETS)))
    ax.set_xticklabels([f"ε=10⁻{int(-np.log10(e))}" for e in EPSILON_TARGETS])
    ax.set_yticks(range(len(main_profiles)))
    ax.set_yticklabels([p["topology"].capitalize() for p in main_profiles])
    for i in range(len(main_profiles)):
        for j in range(len(EPSILON_TARGETS)):
            val = t_matrix[i, j]
            txt = "∞" if val > 900 else f"{val:.1f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=11,
                    fontweight="bold",
                    color="white" if val > t_matrix.max() * 0.6 else "black")
    plt.colorbar(im, ax=ax, label="Rounds to ε-Consensus")
    ax.set_title(f"M3: Mixing Time Heatmap (n={cfg.num_clients} nodes)", fontsize=13)
    ax.set_xlabel("Target ε")
    ax.set_ylabel("Topology")
    plt.tight_layout()
    plt.savefig(os.path.join(M3_RESULTS_DIR, "m3_mixing_time_heatmap.png"), dpi=150)
    plt.close()
    print("  -> m3_mixing_time_heatmap.png")

    # ── 8. M1 Validation (if M1 results available)
    if m1_results:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Scatter: spectral gap vs M1 final accuracy
        for r in m1_results:
            tname = r.get("topology", "")
            mp    = next((p for p in main_profiles if p["topology"] == tname), None)
            if not mp:
                continue
            color = TOPO_COLORS.get(tname, "gray")
            plr   = r["packet_loss_rate"]
            marker = "o" if plr == 0.0 else "^"
            label  = f"{tname.capitalize()} PLR={plr}"
            axes[0].scatter(mp["spectral_gap"], r["final_avg_accuracy"],
                            color=color, marker=marker, s=120,
                            edgecolors="black", label=label, zorder=5)
            axes[0].annotate(f"PLR={plr}",
                             (mp["spectral_gap"], r["final_avg_accuracy"]),
                             textcoords="offset points", xytext=(5, 3), fontsize=8)

        axes[0].set_xlabel("Spectral Gap γ", fontsize=11)
        axes[0].set_ylabel("Final M1 Test Accuracy", fontsize=11)
        axes[0].set_title("Spectral Gap vs M1 Accuracy\n(validates theory: higher γ → better)")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        # Scatter: spectral gap vs M1 consensus error
        for r in m1_results:
            tname = r.get("topology", "")
            mp    = next((p for p in main_profiles if p["topology"] == tname), None)
            if not mp:
                continue
            color  = TOPO_COLORS.get(tname, "gray")
            plr    = r["packet_loss_rate"]
            marker = "o" if plr == 0.0 else "^"
            axes[1].scatter(mp["spectral_gap"], r["final_consensus_error"],
                            color=color, marker=marker, s=120,
                            edgecolors="black", zorder=5)

        axes[1].set_xlabel("Spectral Gap γ", fontsize=11)
        axes[1].set_ylabel("Final M1 Consensus Error (log)", fontsize=11)
        axes[1].set_yscale("log")
        axes[1].set_title("Spectral Gap vs M1 Consensus Error\n(validates: higher γ → lower error)")
        axes[1].grid(True, alpha=0.3, which="both")

        plt.suptitle("M3: Spectral Gap Validation Against M1 Results", fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(M3_RESULTS_DIR, "m3_m1_validation.png"), dpi=150)
        plt.close()
        print("  -> m3_m1_validation.png")


# ── Markdown Report ───────────────────────────────────────────────────────────

def generate_report(profiles: list, size_sweep: list, m1_results: list,
                    cfg: Config) -> str:
    main_profiles = [p for p in profiles if p["n_nodes"] == cfg.num_clients]
    lines = []
    lines.append("# M3 Spectral Gap Analysis Report\n\n")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    lines.append(f"Network size: **{cfg.num_clients} nodes**\n\n")
    lines.append("---\n\n")
    lines.append("## 1. Spectral Properties\n\n")
    lines.append("| Topology | Nodes | Edges | Avg Degree | "
                 "Spectral Gap γ | SLEM |λ₂(W)| | "
                 "T_mix(10⁻²) | T_mix(10⁻³) | T_mix(10⁻⁶) |\n")
    lines.append("|----------|-------|-------|------------|"
                 "----------------|-------------|"
                 "------------|------------|------------|\n")
    for p in main_profiles:
        lines.append(
            f"| {p['topology'].capitalize()} | {p['n_nodes']} | {p['n_edges']} | "
            f"{p['avg_degree']:.2f} | **{p['spectral_gap']:.6f}** | "
            f"{p['slem']:.6f} | "
            f"{p['mixing_time_1e2']:.1f} | "
            f"{p['mixing_time_1e3']:.1f} | "
            f"{p['mixing_time_1e6']:.1f} |\n"
        )
    lines.append("\n")

    ranked = sorted(main_profiles, key=lambda x: x["spectral_gap"], reverse=True)
    lines.append("## 2. Key Findings\n\n")
    if ranked:
        best  = ranked[0]
        worst = ranked[-1]
        lines.append(f"- **Fastest mixing:** {best['topology'].capitalize()} "
                     f"topology (γ = {best['spectral_gap']:.4f})\n")
        lines.append(f"- **Slowest mixing:** {worst['topology'].capitalize()} "
                     f"topology (γ = {worst['spectral_gap']:.4f})\n")
        t_best  = compute_mixing_time(best["spectral_gap"])
        t_worst = compute_mixing_time(worst["spectral_gap"])
        if np.isfinite(t_best) and np.isfinite(t_worst) and t_best > 0:
            speedup = t_worst / t_best
            lines.append(f"- **Speed-up factor:** {best['topology'].capitalize()} "
                         f"converges ~{speedup:.1f}x faster than "
                         f"{worst['topology'].capitalize()}\n")
    lines.append("\n")

    lines.append("## 3. Theoretical Background\n\n")
    lines.append("**Spectral gap:** γ = 1 − |λ₂(W)|\n\n")
    lines.append("**SLEM** = Second Largest Eigenvalue Modulus = |λ₂(W)|\n\n")
    lines.append("**Convergence bound:** E_t = E₀ · SLEM^t\n\n")
    lines.append("**Mixing time:** T_mix(ε) = log(1/ε) / log(1/SLEM)\n\n")

    if m1_results:
        lines.append("## 4. Validation Against M1\n\n")
        lines.append("| Topology | PLR | Spectral Gap γ | M1 Final Acc | M1 Consensus Err |\n")
        lines.append("|----------|-----|----------------|-------------|------------------|\n")
        for r in sorted(m1_results, key=lambda x: (x["topology"], x["packet_loss_rate"])):
            tname = r.get("topology", "")
            mp    = next((p for p in main_profiles if p["topology"] == tname), None)
            sg    = mp["spectral_gap"] if mp else float("nan")
            lines.append(
                f"| {tname.capitalize()} | {r['packet_loss_rate']} | {sg:.6f} | "
                f"{r['final_avg_accuracy']:.4f} | "
                f"{r['final_consensus_error']:.6f} |\n"
            )
        lines.append("\n")

    lines.append("## 5. Generated Plots\n\n")
    for f in [
        "m3_spectral_gap_bar.png     — Spectral gap + SLEM bar chart",
        "m3_eigenvalue_spectrum.png  — Full eigenvalue spectrum of W",
        "m3_convergence_theoretical.png — Theoretical convergence bound",
        "m3_convergence_empirical.png   — Empirical gossip simulation",
        "m3_theory_vs_empirical.png     — Theory vs empirical overlay",
        "m3_spectral_gap_vs_size.png    — Spectral gap vs network size",
        "m3_mixing_time_heatmap.png     — Mixing time for various ε",
        "m3_m1_validation.png           — Spectral gap vs M1 results (if M1 ran)",
    ]:
        lines.append(f"- `{f}`\n")

    return "".join(lines)


# ── Summary printing ──────────────────────────────────────────────────────────

def print_m3_summary(profiles: list, cfg: Config) -> None:
    main_profiles = sorted(
        [p for p in profiles if p["n_nodes"] == cfg.num_clients],
        key=lambda x: x["spectral_gap"], reverse=True
    )
    print(f"\n{'='*75}")
    print(f"  M3 SPECTRAL ANALYSIS SUMMARY  (n={cfg.num_clients} nodes)")
    print(f"{'='*75}")
    print(f"{'Topology':<10} {'Edges':<7} {'AvgDeg':<8} {'Gap γ':<12} "
          f"{'SLEM':<12} {'T_mix(1e-2)':<14} {'T_mix(1e-3)'}")
    print("-" * 75)
    for p in main_profiles:
        print(
            f"{p['topology'].capitalize():<10} "
            f"{p['n_edges']:<7} "
            f"{p['avg_degree']:<8.2f} "
            f"{p['spectral_gap']:<12.6f} "
            f"{p['slem']:<12.6f} "
            f"{p['mixing_time_1e2']:<14.2f} "
            f"{p['mixing_time_1e3']:.2f}"
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = build_arg_parser("M3: Spectral-gap analysis of mixing matrices")
    args = parser.parse_args()
    cfg = config_from_args(args)
    topologies = cfg.topologies_to_test
    os.makedirs(M3_RESULTS_DIR, exist_ok=True)

    json_path = os.path.join(M3_RESULTS_DIR, "m3_spectral_profiles.json")
    m1_results = []
    if os.path.exists(M1_RESULTS_PATH):
        with open(M1_RESULTS_PATH, encoding="utf-8") as f:
            m1_results = json.load(f)

    if args.plot_only:
        if not os.path.exists(json_path):
            print(f"No M3 results found at {json_path}.\nRun first with: python main_m3.py")
            return
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        main_profiles = data["main_profiles"]
        size_sweep = data["size_sweep"]
        all_profiles = main_profiles + [
            p for p in size_sweep if p["n_nodes"] != cfg.num_clients
        ]
        print_m3_summary(all_profiles, cfg)
        if cfg.save_plots:
            print(f"\nGenerating M3 plots in {M3_RESULTS_DIR}/...")
            generate_m3_plots(all_profiles, size_sweep, m1_results, cfg)
        return

    print(f"\n{'='*70}")
    print("  M3: SPECTRAL GAP ANALYSIS")
    print(f"{'='*70}")
    print(f"  Topologies  : {topologies}")
    print(f"  Node sizes  : {NODE_SIZES}")
    print(f"  Sim rounds  : {SIM_ROUNDS}")
    print(f"  Epsilon tgts: {EPSILON_TARGETS}")

    start = time.time()

    # Phase 1: Profile main node count from config
    print(f"\n--- Phase 1: Profile topologies (n={cfg.num_clients}) ---")
    main_profiles = []
    for topo in topologies:
        p = profile_topology(topo, cfg.num_clients,
                             seed=cfg.seed, p=cfg.random_edge_prob)
        main_profiles.append(p)
        print(f"  {topo:<8} γ={p['spectral_gap']:.4f}  "
              f"SLEM={p['slem']:.4f}  "
              f"T_mix(1e-3)={p['mixing_time_1e3']:.1f} rounds")

    # Phase 2: Size sweep
    print(f"\n--- Phase 2: Spectral gap vs network size ---")
    size_sweep = sweep_sizes(topologies, NODE_SIZES, seed=cfg.seed,
                             p=cfg.random_edge_prob)
    print(f"  Profiled {len(size_sweep)} (topology, size) pairs")

    # Phase 3: Load M1 results
    if m1_results:
        print(f"\n--- Phase 3: Loading M1 results for validation ---")
        print(f"  Loaded {len(m1_results)} M1 experiments")
        # Verify spectral ordering matches M1 ordering
        print("\n  Validating: spectral gap ordering vs M1 convergence ordering:")
        print(f"  {'Topology':<10} {'Spec.Gap':<12} {'M1 ConsErr(PLR=0)'}")
        for p in sorted(main_profiles, key=lambda x: x["spectral_gap"], reverse=True):
            m1r = next((r for r in m1_results
                        if r["topology"] == p["topology"]
                        and r["packet_loss_rate"] == 0.0), None)
            ce = m1r["final_consensus_error"] if m1r else float("nan")
            print(f"  {p['topology'].capitalize():<10} {p['spectral_gap']:.6f}     {ce:.6f}")
    else:
        print(f"\n--- Phase 3: M1 results not found (spectral validation skipped) ---")

    elapsed = time.time() - start

    # Combine all profiles
    all_profiles = main_profiles + [
        p for p in size_sweep if p["n_nodes"] != cfg.num_clients
    ]

    # Save JSON
    data = {
        "config": {
            "num_clients"     : cfg.num_clients,
            "random_edge_prob": cfg.random_edge_prob,
            "seed"            : cfg.seed,
        },
        "main_profiles": main_profiles,
        "size_sweep"   : size_sweep,
        "epsilon_targets": EPSILON_TARGETS,
        "mixing_times": {
            p["topology"]: {
                str(eps): compute_mixing_time(p["spectral_gap"], eps)
                for eps in EPSILON_TARGETS
            }
            for p in main_profiles
        },
    }
    json_path = os.path.join(M3_RESULTS_DIR, "m3_spectral_profiles.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"\n  Spectral profiles saved to {json_path}")

    # Print summary
    print_m3_summary(all_profiles, cfg)

    # Generate plots
    print(f"\nGenerating M3 plots in {M3_RESULTS_DIR}/...")
    generate_m3_plots(all_profiles, size_sweep, m1_results, cfg)

    # Generate report
    report = generate_report(all_profiles, size_sweep, m1_results, cfg)
    rpt_path = os.path.join(M3_RESULTS_DIR, "m3_report.md")
    with open(rpt_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  -> m3_report.md")

    print(f"\n  M3 analysis complete in {elapsed:.1f}s")
    print(f"\n{'='*70}")
    print(f"  M3 complete. All outputs in: {M3_RESULTS_DIR}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
