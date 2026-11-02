# M3 Spectral Gap Analysis Report

Generated: 2026-10-29 18:01:57

Network size: **5 nodes**

---

## 1. Spectral Properties

| Topology | Nodes | Edges | Avg Degree | Spectral Gap γ | SLEM ρ | T_mix(10⁻²) | T_mix(10⁻³) | T_mix(10⁻⁶) |
|----------|-------|-------|------------|----------------|-------------|------------|------------|------------|
| Ring | 5 | 5 | 2.00 | **0.460655** | 0.539345 | 4.0 | 6.0 | 12.0 |
| Grid | 5 | 5 | 2.00 | **0.211325** | 0.788675 | 10.0 | 15.0 | 30.0 |
| Random | 5 | 7 | 2.80 | **0.346447** | 0.653553 | 6.0 | 9.0 | 17.0 |

## 2. Key Findings

- **Fastest mixing:** Ring topology (γ = 0.4607)
- **Slowest mixing:** Grid topology (γ = 0.2113)
- **Speed-up factor:** Ring converges ~2.5x faster than Grid

## 3. Theoretical Background

**Spectral gap:** γ = 1 − |λ₂(W)|

**SLEM** = Second Largest Eigenvalue Modulus = |λ₂(W)|

**Squared-error upper bound (ideal fixed averaging only):** E_t <= B_t = E₀ · SLEM^(2t)

**Mixing time:** T_mix(ε) = ceil(log(1/ε) / (2 log(1/SLEM))) for relative squared error

## 5. Generated Plots

- `m3_spectral_gap_bar.png     — Spectral gap + SLEM bar chart`
- `m3_eigenvalue_spectrum.png  — Full eigenvalue spectrum of W`
- `m3_convergence_theoretical.png — Theoretical convergence bound`
- `m3_convergence_empirical.png   — Empirical gossip simulation`
- `m3_theory_vs_empirical.png     — Theory vs empirical overlay`
- `m3_spectral_gap_vs_size.png    — Spectral gap vs network size`
- `m3_mixing_time_heatmap.png     — Mixing time for various ε`
- `m3_m1_validation.png           — Spectral gap vs M1 results (if M1 ran)`
