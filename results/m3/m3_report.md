# M3 Spectral Gap Analysis Report

Generated: 2026-10-08 14:51:20

Network size: **5 nodes**

---

## 1. Spectral Properties

| Topology | Nodes | Edges | Avg Degree | Spectral Gap γ | SLEM |λ₂(W)| | T_mix(10⁻²) | T_mix(10⁻³) | T_mix(10⁻⁶) |
|----------|-------|-------|------------|----------------|-------------|------------|------------|------------|
| Ring | 5 | 5 | 2.00 | **0.460655** | 0.539345 | 7.5 | 11.2 | 22.4 |
| Grid | 5 | 5 | 2.00 | **0.211325** | 0.788675 | 19.4 | 29.1 | 58.2 |
| Random | 5 | 7 | 2.80 | **0.346447** | 0.653553 | 10.8 | 16.2 | 32.5 |

## 2. Key Findings

- **Fastest mixing:** Ring topology (γ = 0.4607)
- **Slowest mixing:** Grid topology (γ = 0.2113)
- **Speed-up factor:** Ring converges ~2.6x faster than Grid

## 3. Theoretical Background

**Spectral gap:** γ = 1 − |λ₂(W)|

**SLEM** = Second Largest Eigenvalue Modulus = |λ₂(W)|

**Convergence bound:** E_t = E₀ · SLEM^t

**Mixing time:** T_mix(ε) = log(1/ε) / log(1/SLEM)

## 4. Validation Against M1

| Topology | PLR | Spectral Gap γ | M1 Final Acc | M1 Consensus Err |
|----------|-----|----------------|-------------|------------------|
| Grid | 0.0 | 0.211325 | 0.9910 | 0.314697 |
| Grid | 0.2 | 0.211325 | 0.9909 | 0.661188 |
| Random | 0.0 | 0.346447 | 0.9917 | 0.130301 |
| Random | 0.2 | 0.346447 | 0.9912 | 0.160869 |
| Ring | 0.0 | 0.460655 | 0.9914 | 0.120685 |
| Ring | 0.2 | 0.460655 | 0.9908 | 0.301325 |

## 5. Generated Plots

- `m3_spectral_gap_bar.png     — Spectral gap + SLEM bar chart`
- `m3_eigenvalue_spectrum.png  — Full eigenvalue spectrum of W`
- `m3_convergence_theoretical.png — Theoretical convergence bound`
- `m3_convergence_empirical.png   — Empirical gossip simulation`
- `m3_theory_vs_empirical.png     — Theory vs empirical overlay`
- `m3_spectral_gap_vs_size.png    — Spectral gap vs network size`
- `m3_mixing_time_heatmap.png     — Mixing time for various ε`
- `m3_m1_validation.png           — Spectral gap vs M1 results (if M1 ran)`
