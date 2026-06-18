#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════╗
║        LORENZ '96 MODEL — N-Dimensional Atmospheric Dynamics          ║
║        Lyapunov Spectrum, Kaplan-Yorke Dimension & Attractors         ║
╚══════════════════════════════════════════════════════════════════════════╝

PHYSICAL BACKGROUND
───────────────────
E. N. Lorenz (1996, "Predictability: A problem partly solved", Proc. Seminar
on Predictability, ECMWF) introduced this model as a toy representation of
atmospheric dynamics on a latitude circle. Each variable x_i represents a
scalar atmospheric quantity (e.g., vorticity) at N equally spaced grid points
around a circle of constant latitude.

THE EQUATION:

    dx_i/dt = (x_{i+1} − x_{i−2}) · x_{i−1} − x_i + F

for i = 0, 1, ..., N−1, with PERIODIC boundary conditions (indices mod N).

TERMS:
  (x_{i+1} − x_{i−2}) · x_{i−1}  →  Quadratic advection. This is a
      simplified representation of the nonlinear advection term (v·∇)v
      in the Navier-Stokes equations. It conserves total energy:
          d/dt [Σ x_i²/2] = 0   (from advection term alone)
      and preserves the phase-space volume (Liouville property of advection).

  −x_i  →  Linear damping (friction/dissipation). This drives the system
      toward x_i = 0 in the absence of forcing.

  +F    →  External forcing (constant). Represents large-scale energy input
      (e.g., differential solar heating). F is the CONTROL PARAMETER.

DYNAMICAL REGIMES (for standard N = 40):
────────────────────────────────────────
  F < 1:      Decaying solutions → x_i → 0 (damping overwhelms forcing)

  F ≈ 1-3:    Stable fixed point x_i = F for all i (homogeneous state)

  F ≈ 3-5:    Periodic orbits emerge (Hopf bifurcation)

  F ≈ 5-6:    Quasi-periodic motion (torus dynamics)

  F ≈ 6-8:    Weakly chaotic (positive λ_1, small D_KY)

  F = 8:      STANDARD BENCHMARK in data assimilation and weather prediction.
              D_KY ≈ 27.1 for N=40. The system behaves like a "toy atmosphere"
              with a decorrelation time of ~0.25 model time units (≈ 5 real days
              if one model time unit = 5 days in operational meteorology).

  F > 8:      Increasingly chaotic. D_KY increases roughly linearly with F.

  F = 16:     Very turbulent. Nearly all Lyapunov exponents are positive.

MATHEMATICAL PROPERTIES
───────────────────────
1. ENERGY CONSERVATION BY ADVECTION:
   The quadratic terms satisfy: Σ_i (x_{i+1} − x_{i−2}) x_{i−1} x_i = 0
   Proof: summation by parts with periodic BC. This is the analog of
   the antisymmetry of (v·∇)v in Euler's equations.

2. BOUNDED SOLUTIONS:
   The total energy E = ½ Σ x_i² satisfies:
       dE/dt = −2E + F Σ x_i ≤ −2E + F √(2NE)   (Cauchy-Schwarz)
   So the attractor lies within a bounded ball in R^N.

3. PHASE-SPACE CONTRACTION:
   ∇·F = Σ ∂(dx_i/dt)/∂x_i = −N
   So the system is uniformly dissipative with contraction rate = −N.
   Consequence:  Σ λ_i = −N  (exact, for consistency check).

4. SYMMETRY:
   The system is equivariant under cyclic permutations:
       x_i → x_{i+1 mod N}
   (but the attractor may break this symmetry).

LYAPUNOV EXPONENTS FOR HIGH-DIMENSIONAL ODEs
─────────────────────────────────────────────
The Benettin QR algorithm generalizes directly. We track an N×N
perturbation matrix Y satisfying dY/dt = J(x) · Y.

For Lorenz-96, the Jacobian is SPARSE (tridiagonal + one off-diagonal):

    ∂(dx_i/dt)/∂x_j = { x_{i+1} − x_{i−2}   if j = i−1
                       { x_{i−1}               if j = i+1
                       { −x_{i−1}              if j = i−2
                       { −1                     if j = i
                       { 0                      otherwise

This sparsity is key to efficient computation when N is large.

KAPLAN-YORKE DIMENSION
──────────────────────
Same formula as always. For N=40, F=8:
    Approximately 13 positive exponents, 1 near-zero, 26 negative.
    D_KY ≈ 27.1 (the attractor is a ~27-dimensional fractal in R^40).

This very high dimensionality is what makes Lorenz '96 a challenging
benchmark for data assimilation and machine learning weather prediction.
"""

import numpy as np
from scipy.integrate import solve_ivp
import json
import csv
import time
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ══════════════════════════════════════════════════════════════════════════
# 1. GLOBAL PARAMETERS — Lorenz '96
# ══════════════════════════════════════════════════════════════════════════
N_DIM = 40          # Number of grid points (latitude circle)

# ── Scan parameter: forcing F ──
F_SCAN = [2, 4, 6, 8, 12, 16]

# ── Representative F values for attractor panels ──
F_ATTRACTORS = [2, 4, 8, 12, 16]

REGIME_LABELS = {
    2:   "Fixed Point",
    4:   "Periodic",
    8:   "Chaotic (Standard)",
    12:  "Strongly Chaotic",
    16:  "Turbulent",
}

# ── Integration parameters ──
T_TRANS    = 100.0    # Transient discard
T_LYAP    = 500.0     # Lyapunov integration time (shorter due to N=40)
DT_RENORM = 0.5       # QR renormalization interval
DT_TRAJ   = 0.01      # Trajectory sampling time step


# ══════════════════════════════════════════════════════════════════════════
# 2. LORENZ '96 EQUATIONS AND JACOBIAN
# ══════════════════════════════════════════════════════════════════════════
def lorenz96(t, x, F):
    """
    Lorenz '96 right-hand side:
        dx_i/dt = (x_{i+1} − x_{i−2}) · x_{i−1} − x_i + F

    Implemented with numpy roll operations for periodic boundaries:
      np.roll(x, -1) gives x_{i+1}
      np.roll(x,  1) gives x_{i−1}
      np.roll(x,  2) gives x_{i−2}
    """
    x = np.asarray(x)
    x_ip1 = np.roll(x, -1)  # x_{i+1}
    x_im1 = np.roll(x,  1)  # x_{i−1}
    x_im2 = np.roll(x,  2)  # x_{i−2}

    return (x_ip1 - x_im2) * x_im1 - x + F


def lorenz96_jacobian(x):
    """
    Jacobian of the Lorenz '96 system.

    J_{ij} = ∂(dx_i/dt)/∂x_j

    The nonzero entries are (using periodic indexing mod N):
      J[i, i]   = −1                         (from −x_i)
      J[i, i−1] = x_{i+1} − x_{i−2}         (from advection, ∂/∂x_{i-1})
      J[i, i+1] = x_{i−1}                    (from advection, ∂/∂x_{i+1})
      J[i, i−2] = −x_{i−1}                   (from advection, ∂/∂x_{i-2})

    The Jacobian is a SPARSE circulant-like matrix — only 4 diagonals
    are nonzero (wrapping around with periodic BC).
    """
    N = len(x)
    J = np.zeros((N, N))

    for i in range(N):
        ip1 = (i + 1) % N
        im1 = (i - 1) % N
        im2 = (i - 2) % N

        J[i, i]   = -1.0
        J[i, im1] = x[ip1] - x[im2]   # coefficient of δx_{i-1}
        J[i, ip1] = x[im1]              # coefficient of δx_{i+1}
        J[i, im2] = -x[im1]             # coefficient of δx_{i-2}

    return J


# ══════════════════════════════════════════════════════════════════════════
# 3. LYAPUNOV COMPUTATION — VARIATIONAL EQUATIONS + QR
# ══════════════════════════════════════════════════════════════════════════
def augmented_rhs(t, state_and_Y, F, N):
    """
    Augmented ODE for Lorenz '96 + variational equations.

    State vector: [x_0, ..., x_{N-1}, Y_00, Y_10, ..., Y_{N-1,N-1}]
                   \_______________/  \___________________________/
                     N ODE variables    N² variational variables

    The variational equation dY/dt = J(x) · Y tracks how perturbation
    vectors evolve. For N=40, this means 40 + 40² = 1640 coupled ODEs.
    """
    x = state_and_Y[:N]
    Y = state_and_Y[N:].reshape(N, N)

    dxdt = lorenz96(t, x, F)
    J = lorenz96_jacobian(x)
    dYdt = J @ Y

    return np.concatenate([dxdt, dYdt.flatten()])


def compute_lyapunov_ode(F, N=N_DIM):
    """
    Compute the full N-dimensional Lyapunov spectrum for Lorenz '96
    at forcing F, using QR renormalization.

    For N=40, we simultaneously integrate 1640 ODEs and periodically
    QR-decompose a 40×40 matrix. This is computationally intensive
    but gives the FULL spectrum, from which D_KY is exact.

    CONSISTENCY CHECK:
        Σ λ_i = −N  (divergence of Lorenz '96 is uniformly −N)
    """
    # ── Initial condition with small perturbation from x=F ──
    x0 = F * np.ones(N) + 0.01 * np.random.randn(N)

    # ── Discard transient ──
    sol_trans = solve_ivp(lorenz96, [0, T_TRANS], x0, args=(F,),
                          method='RK45', rtol=1e-8, atol=1e-10)
    x0 = sol_trans.y[:, -1]

    # ── Initialize perturbation matrix ──
    Y = np.eye(N)
    state_and_Y = np.concatenate([x0, Y.flatten()])

    # ── Lyapunov accumulation via QR ──
    n_steps = int(T_LYAP / DT_RENORM)
    lyap_sum = np.zeros(N)
    t_current = 0.0

    for step in range(n_steps):
        sol = solve_ivp(
            augmented_rhs,
            [t_current, t_current + DT_RENORM],
            state_and_Y,
            args=(F, N),
            method='RK45', rtol=1e-8, atol=1e-10,
        )
        state_and_Y = sol.y[:, -1]
        t_current += DT_RENORM

        Y = state_and_Y[N:].reshape(N, N)
        Q, R = np.linalg.qr(Y)

        # Fix sign convention
        signs = np.sign(np.diag(R))
        signs[signs == 0] = 1
        Q = Q * signs
        R = np.diag(signs) @ R

        lyap_sum += np.log(np.abs(np.diag(R)))
        state_and_Y[N:] = Q.flatten()

    lyap = lyap_sum / (n_steps * DT_RENORM)
    lyap = np.sort(lyap)[::-1]

    # Consistency check
    expected_sum = -float(N)
    actual_sum = np.sum(lyap)
    print(f"    Σλ = {actual_sum:.2f} (expected {expected_sum:.2f})")

    d_ky = kaplan_yorke_dimension(lyap)
    return lyap, d_ky


def kaplan_yorke_dimension(lyap_sorted):
    """D_KY from sorted Lyapunov spectrum. See MG script for derivation."""
    cum = np.cumsum(lyap_sorted)
    neg = np.where(cum < 0)[0]

    if len(neg) == 0:
        return float(len(lyap_sorted))
    elif neg[0] == 0:
        return 0.0
    else:
        j = neg[0]
        return j + cum[j - 1] / abs(lyap_sorted[j])


# ══════════════════════════════════════════════════════════════════════════
# 4. TRAJECTORY GENERATION
# ══════════════════════════════════════════════════════════════════════════
def get_trajectory(F, N=N_DIM, t_max=50.0, dt=0.01):
    """Integrate Lorenz '96 and return trajectory after transient."""
    x0 = F * np.ones(N) + 0.01 * np.random.randn(N)
    sol_trans = solve_ivp(lorenz96, [0, T_TRANS], x0, args=(F,),
                          method='RK45', rtol=1e-8, atol=1e-10)
    x0 = sol_trans.y[:, -1]

    t_eval = np.arange(0, t_max, dt)
    sol = solve_ivp(lorenz96, [0, t_max], x0, args=(F,),
                    method='RK45', t_eval=t_eval, rtol=1e-8, atol=1e-10)
    return sol.t, sol.y   # sol.y is (N, n_times)


# ══════════════════════════════════════════════════════════════════════════
# 5. DATA EXPORT
# ══════════════════════════════════════════════════════════════════════════
def export_data(results, base_filename="../../figures/lorenz_96/../data/lorenz_96/lorenz96_scan_data"):
    """Export results to JSON and CSV."""
    with open(f"{base_filename}.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(f"{base_filename}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["F", "D_KY", "Lambda_1", "Sum_Lambda",
                          "N_Positive", "Full_Spectrum_first10"])
        for key, r in sorted(results.items(), key=lambda x: float(x[0])):
            writer.writerow([
                r["F"], r["D_KY"], r["lambda_1"], r["sum_lambda"],
                r["n_positive_exponents"],
                [round(l, 4) for l in r["full_spectrum"][:10]]
            ])
    print(f"\nExported: {base_filename}.json and {base_filename}.csv")


# ══════════════════════════════════════════════════════════════════════════
# 6. PLOTTING
# ══════════════════════════════════════════════════════════════════════════
def plot_metrics(results_dict):
    """D_KY and λ_1 vs F."""
    Fs = sorted([float(k) for k in results_dict.keys()])
    d_kys = [results_dict[str(int(f)) if f == int(f) else str(f)]["D_KY"] for f in Fs]
    l1s = [results_dict[str(int(f)) if f == int(f) else str(f)]["lambda_1"] for f in Fs]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.set_xlabel(r'Forcing $F$', fontsize=12, fontweight='bold')
    ax1.set_ylabel(r'$D_{KY}$', color='tab:blue', fontsize=12, fontweight='bold')
    ax1.plot(Fs, d_kys, 'o-', color='tab:blue', linewidth=2.5, markersize=7)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax2 = ax1.twinx()
    ax2.set_ylabel(r'$\lambda_1$', color='tab:red', fontsize=12, fontweight='bold')
    ax2.plot(Fs, l1s, 's-', color='tab:red', linewidth=2.5, markersize=7)
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.axhline(0, color='black', linewidth=1.5, linestyle='--')

    plt.title("Lorenz '96 (N=40): Route to Chaos", fontsize=14,
              fontweight='bold', pad=15)
    fig.tight_layout()
    plt.savefig("../../figures/lorenz_96/lorenz96_manuscript_metrics.png", dpi=300)
    print("Saved: ../figures/lorenz_96/lorenz96_manuscript_metrics.png")


def plot_attractors(F_list, results_dict=None, N=N_DIM):
    """
    3-row figure for each F:
      Row 1: Hovmöller diagram (x_i vs t, heatmap) — natural for N-dim systems
      Row 2: Time series x_0(t)
      Row 3: Lyapunov spectrum bar chart
    """
    n = len(F_list)
    fig = plt.figure(figsize=(4.5 * n, 12))
    gs = GridSpec(3, n, figure=fig, height_ratios=[1, 0.7, 0.6], hspace=0.35)

    for i, F in enumerate(F_list):
        F_key = str(int(F)) if F == int(F) else str(float(F))
        print(f"Generating visuals for F = {F}...")

        t_arr, X = get_trajectory(F, N=N, t_max=20.0, dt=0.01)

        d_ky_val = None
        lyap_spectrum = None
        if results_dict is not None and F_key in results_dict:
            d_ky_val = results_dict[F_key]["D_KY"]
            lyap_spectrum = results_dict[F_key]["full_spectrum"]

        regime = REGIME_LABELS.get(int(F), "")
        title_dky = f"$D_{{KY}}={d_ky_val:.1f}$" if d_ky_val is not None else ""
        title_top = f"$F={F}$ — {regime}\n{title_dky}"

        # ═══ Row 1: Hovmöller diagram (space-time heatmap) ═══
        # For N-dimensional systems, a 2D heatmap of x_i(t) is more
        # informative than a single phase-space projection.
        ax_hov = fig.add_subplot(gs[0, i])
        # Subsample time for readability
        stride = max(1, len(t_arr) // 500)
        im = ax_hov.pcolormesh(t_arr[::stride], np.arange(N),
                                X[:, ::stride], cmap='RdBu_r',
                                shading='auto')
        ax_hov.set_title(title_top, fontsize=11, fontweight='bold')
        ax_hov.set_xlabel(r"$t$", fontsize=10)
        ax_hov.set_ylabel(r"Grid point $i$", fontsize=10)

        # ═══ Row 2: Time series x_0(t) ═══
        ax_ts = fig.add_subplot(gs[1, i])
        ax_ts.plot(t_arr, X[0, :], color='tab:blue', linewidth=0.5)
        ax_ts.set_xlabel(r"$t$", fontsize=10)
        ax_ts.set_ylabel(r"$x_0(t)$", fontsize=10)
        ax_ts.set_title("Time series (site 0)", fontsize=10)
        ax_ts.grid(True, linestyle=':', alpha=0.5)

        # ═══ Row 3: Lyapunov spectrum ═══
        ax_ly = fig.add_subplot(gs[2, i])
        if lyap_spectrum is not None:
            spec = [l for l in lyap_spectrum if np.isfinite(l)]
            indices = np.arange(len(spec))
            colors = ['tab:red' if l > 0 else ('gold' if abs(l) < 0.01
                       else 'tab:blue') for l in spec]
            ax_ly.bar(indices, spec, color=colors, edgecolor='none',
                      width=0.9)
            ax_ly.axhline(0, color='black', linewidth=1)
            ax_ly.set_xlabel("Index $i$", fontsize=9)
            ax_ly.set_ylabel(r"$\lambda_i$", fontsize=10)
            ax_ly.set_title(f"Lyapunov spectrum (N={N})", fontsize=10)
            ax_ly.grid(True, linestyle=':', alpha=0.5, axis='y')
        else:
            ax_ly.text(0.5, 0.5, "No data", ha='center', va='center',
                       transform=ax_ly.transAxes, fontsize=12, color='gray')

    fig.suptitle(f"Lorenz '96 (N={N}) — Route to Chaos",
                 fontsize=15, fontweight='bold', y=1.01)
    fig.tight_layout()
    plt.savefig("../../figures/lorenz_96/lorenz96_educational_attractors.png", dpi=300,
                bbox_inches='tight')
    print("Saved: ../figures/lorenz_96/lorenz96_educational_attractors.png")


# ══════════════════════════════════════════════════════════════════════════
# 7. MAIN
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    results = {}

    print("=" * 70)
    print(f"Lorenz '96 (N={N_DIM}) — Lyapunov Spectrum & Kaplan-Yorke Dimension")
    print("=" * 70)

    for F in F_SCAN:
        F_key = str(int(F)) if F == int(F) else str(float(F))
        print(f"F = {F:5.1f} | Computing {N_DIM} exponents... ",
              end="", flush=True)
        t0 = time.time()

        lyap, d_ky = compute_lyapunov_ode(F)

        elapsed = time.time() - t0
        n_pos = int(sum(1 for l in lyap if l > 0))

        results[F_key] = {
            "F": float(F),
            "N": N_DIM,
            "D_KY": round(d_ky, 3),
            "lambda_1": round(float(lyap[0]), 5),
            "sum_lambda": round(float(np.sum(lyap)), 3),
            "n_positive_exponents": n_pos,
            "full_spectrum": [float(l) for l in lyap],
        }

        print(f"  Done in {elapsed:6.1f}s | D_KY={d_ky:6.2f} | "
              f"λ_1={lyap[0]:7.4f} | #(λ>0)={n_pos}")

    export_data(results)
    plot_metrics(results)
    plot_attractors(F_ATTRACTORS, results_dict=results)

    print("\nAll tasks completed successfully!")
    plt.show()
