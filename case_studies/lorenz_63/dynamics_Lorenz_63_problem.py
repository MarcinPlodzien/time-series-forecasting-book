#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════╗
║        LORENZ '63 SYSTEM — 3-Variable Atmospheric Convection          ║
║        Lyapunov Spectrum, Kaplan-Yorke Dimension & Attractors         ║
╚══════════════════════════════════════════════════════════════════════════╝

PHYSICAL BACKGROUND
───────────────────
E. N. Lorenz (1963, "Deterministic Nonperiodic Flow", J. Atmos. Sci. 20:130)
derived a 3-variable truncation of the Navier-Stokes equations describing
2D Rayleigh-Bénard convection — a fluid layer heated from below.

The equations are:

    dx/dt = σ (y − x)
    dy/dt = x (ρ − z) − y
    dz/dt = x y − β z

where:
  x(t) ~ intensity of convective overturning
  y(t) ~ temperature difference between ascending/descending currents
  z(t) ~ deviation of vertical temperature profile from linearity

PARAMETERS:
  σ (sigma) = Prandtl number     (ratio of viscous to thermal diffusivity)
  ρ (rho)   = Rayleigh number    (driving force: bottom-to-top ΔT)
  β (beta)  = geometric factor   (aspect ratio of convection rolls)

Classic values: σ = 10, β = 8/3, ρ = 28

BIFURCATION STRUCTURE (scanning ρ with σ=10, β=8/3):
─────────────────────────────────────────────────────
  ρ < 1:        The origin is the only fixed point (globally stable).
                All fluid motion decays → heat conduction only.

  ρ = 1:        Pitchfork bifurcation. Two new fixed points appear:
                C± = (±√[β(ρ-1)], ±√[β(ρ-1)], ρ-1)
                representing steady convection rolls.

  1 < ρ < ρ_H:  C± are stable spirals. ρ_H = σ(σ+β+3)/(σ-β-1) ≈ 24.74
                (the Hopf bifurcation threshold for σ=10, β=8/3).

  ρ ≈ 13.93:   A homoclinic bifurcation occurs. For ρ slightly above this,
                transient chaos exists but eventually settles to C±.

  ρ ≈ 24.06:   The "crisis" point: the chaotic attractor becomes the
                global attractor. This is the true onset of sustained chaos.

  ρ = 28:       The classic Lorenz attractor. D_KY ≈ 2.06.
                One positive Lyapunov exponent (λ_1 ≈ 0.9056).

  ρ > 28:       Continued chaos with periodic windows. At ρ ≈ 100,
                D_KY ≈ 2.11. At ρ ≈ 200, more complex dynamics.

IMPORTANT PROPERTY: The Lorenz system is DISSIPATIVE. The phase-space
volume contracts at rate:
    ∇·F = ∂ẋ/∂x + ∂ẏ/∂y + ∂ż/∂z = −σ − 1 − β = −(10+1+8/3) ≈ −13.67

This means: Σ λ_i = −(σ + 1 + β) < 0 always. The sum of all Lyapunov
exponents equals the divergence. This is a CONSISTENCY CHECK for our
computation.

LYAPUNOV EXPONENTS — ODE VARIATIONAL EQUATIONS
───────────────────────────────────────────────
For an ODE system dx/dt = F(x), the variational (tangent-linear) equation
for a perturbation matrix Y is:

    dY/dt = J(x(t)) · Y

where J = ∂F/∂x is the Jacobian evaluated along the trajectory x(t).
For Lorenz '63:

         ⎡ −σ    σ    0 ⎤
    J =  ⎢ ρ−z  −1   −x ⎥
         ⎣  y    x   −β ⎦

ALGORITHM (Benettin et al., 1980):
  1. Integrate the ODE x(t) and the variational equation Y(t)
     simultaneously.
  2. Periodically (every Δt), QR-decompose Y:  Y = Q R
  3. Record λ_i^{local} = ln(R_ii) / Δt
  4. Reset Y ← Q  (keep directions, reset magnitudes)
  5. Average over time to get global Lyapunov exponents.

KAPLAN-YORKE DIMENSION
──────────────────────
Same formula as for any dynamical system:
    D_KY = j + S_j / |λ_{j+1}|

For the Lorenz attractor at ρ=28:
    λ_1 ≈ +0.906, λ_2 ≈ 0.000, λ_3 ≈ −14.57
    S_2 = λ_1 + λ_2 ≈ 0.906
    D_KY = 2 + 0.906/14.57 ≈ 2.062

This means the attractor is slightly "thicker" than a 2D surface —
a fractal embedded in 3D space.
"""

import numpy as np
from scipy.integrate import solve_ivp
import json
import csv
import time
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ══════════════════════════════════════════════════════════════════════════
# 1. GLOBAL PARAMETERS — Lorenz '63
# ══════════════════════════════════════════════════════════════════════════
SIGMA = 10.0       # Prandtl number
BETA  = 8.0 / 3.0  # Geometric factor
# ρ (rho) is the scan parameter — the Rayleigh number

# ── Parameter scan: ρ values ──
# We choose values that span all dynamical regimes
RHO_SCAN = [0.5, 5, 15, 24, 28, 50, 100, 150]

# ── Attractor visualization: representative ρ values ──
RHO_ATTRACTORS = [5, 15, 28, 100, 150]

REGIME_LABELS = {
    5:    "Stable Fixed Point",
    15:   "Transient Chaos → Fixed Pt",
    28:   "Classic Chaos",
    100:  "Developed Chaos",
    150:  "Strong Chaos",
}

# ── Integration parameters ──
T_TRANS  = 200.0     # Transient discard time
T_LYAP   = 2000.0    # Integration time for Lyapunov averaging
DT_RENORM = 1.0      # QR renormalization interval
DT_TRAJ   = 0.01     # Time step for trajectory output


# ══════════════════════════════════════════════════════════════════════════
# 2. LORENZ '63 EQUATIONS AND JACOBIAN
# ══════════════════════════════════════════════════════════════════════════
def lorenz63(t, state, rho):
    """
    Lorenz '63 right-hand side.

    dx/dt = σ(y − x)           ← rate of convective overturning
    dy/dt = x(ρ − z) − y       ← thermal forcing minus damping
    dz/dt = xy − βz            ← nonlinear heat transport minus diffusion
    """
    x, y, z = state
    return [
        SIGMA * (y - x),
        x * (rho - z) - y,
        x * y - BETA * z,
    ]


def lorenz63_jacobian(state, rho):
    """
    Jacobian matrix J = ∂F/∂x for the Lorenz system.

         ⎡ −σ    σ    0 ⎤      ⎡ −10   10    0 ⎤
    J =  ⎢ ρ−z  −1   −x ⎥  =   ⎢ ρ−z  −1   −x ⎥
         ⎣  y    x   −β ⎦      ⎣  y    x  −8/3 ⎦

    This matrix determines how infinitesimal perturbations evolve:
        dδx/dt = J · δx
    The eigenvalues of J at fixed points give local stability.
    Along a trajectory, J varies with time through x(t), y(t), z(t).
    """
    x, y, z = state
    return np.array([
        [-SIGMA,   SIGMA,   0    ],
        [rho - z, -1,      -x    ],
        [y,        x,      -BETA ],
    ])


# ══════════════════════════════════════════════════════════════════════════
# 3. LYAPUNOV EXPONENT COMPUTATION — VARIATIONAL EQUATIONS + QR
# ══════════════════════════════════════════════════════════════════════════
def augmented_rhs(t, state_and_Y, rho, ndim=3):
    """
    Right-hand side for the AUGMENTED system:
      [x(t), Y(t)]  where Y is the ndim×ndim perturbation matrix.

    The state vector is packed as:
      state_and_Y = [x, y, z, Y_00, Y_10, Y_20, Y_01, Y_11, Y_21, ...]
                     \_____/  \___________________________________/
                     3 ODE       ndim² variational equations
                     variables

    The variational equation is:
      dY/dt = J(x(t)) · Y

    where J is the Jacobian evaluated at the current trajectory point.
    Each column of Y tracks how one perturbation direction evolves.
    """
    x_state = state_and_Y[:ndim]
    Y = state_and_Y[ndim:].reshape(ndim, ndim)

    # Lorenz RHS
    dxdt = lorenz63(t, x_state, rho)

    # Jacobian at current point
    J = lorenz63_jacobian(x_state, rho)

    # Variational equation: dY/dt = J · Y
    dYdt = J @ Y

    return np.concatenate([dxdt, dYdt.flatten()])


def compute_lyapunov_ode(rho, ndim=3):
    """
    Compute the full Lyapunov spectrum for the Lorenz '63 system
    at parameter ρ, using the QR-renormalization method.

    ALGORITHM (Benettin, Galgani, Giorgilli, Strelcyn, 1980):
    ──────────────────────────────────────────────────────────
    1. Initialize: x₀ = random IC on attractor, Y₀ = I (identity).
    2. Integrate augmented system [x, Y] for Δt (renormalization interval).
    3. QR-decompose Y:  Y = Q · R
       - The columns of Q give the orthonormal Lyapunov directions.
       - R_ii measures the growth/shrinkage along direction i.
    4. Record:  λ_i += ln|R_ii|
       Set Y ← Q (renormalize to prevent overflow/collapse).
    5. Repeat 2–4 for N_steps.
    6. Average:  λ_i /= (N_steps · Δt)

    WHY QR? Without renormalization, all perturbation vectors would
    align with the most unstable direction (the one with λ_1 > 0),
    making it impossible to measure λ_2, λ_3, etc. QR decomposition
    keeps the vectors orthogonal while tracking their individual
    growth rates through the R diagonal.

    CONSISTENCY CHECK: For Lorenz '63,
        Σ λ_i = −σ − 1 − β = −(10 + 1 + 8/3) ≈ −13.667
    This is an exact relation (Liouville's theorem for dissipative systems).
    """
    # ── Discard transient ──
    ic = [1.0, 1.0, 1.0]
    sol_trans = solve_ivp(lorenz63, [0, T_TRANS], ic, args=(rho,),
                          method='RK45', rtol=1e-10, atol=1e-12)
    x0 = sol_trans.y[:, -1]

    # ── Initialize perturbation matrix as identity ──
    Y = np.eye(ndim)
    state_and_Y = np.concatenate([x0, Y.flatten()])

    # ── Lyapunov accumulation ──
    n_steps = int(T_LYAP / DT_RENORM)
    lyap_sum = np.zeros(ndim)
    t_current = 0.0

    for step in range(n_steps):
        # Integrate augmented system for one renormalization interval
        sol = solve_ivp(
            augmented_rhs,
            [t_current, t_current + DT_RENORM],
            state_and_Y,
            args=(rho, ndim),
            method='RK45', rtol=1e-10, atol=1e-12,
        )
        state_and_Y = sol.y[:, -1]
        t_current += DT_RENORM

        # Extract perturbation matrix
        Y = state_and_Y[ndim:].reshape(ndim, ndim)

        # QR decomposition: Y = Q R
        # Q columns = new orthonormal basis (Lyapunov directions)
        # R diagonal = stretch/compression factors
        Q, R = np.linalg.qr(Y)

        # Ensure positive diagonal of R (sign convention)
        signs = np.sign(np.diag(R))
        signs[signs == 0] = 1
        Q = Q * signs
        R = np.diag(signs) @ R

        # Accumulate: ln|R_ii| gives the logarithmic growth rate
        lyap_sum += np.log(np.abs(np.diag(R)))

        # Reset perturbation vectors to orthonormal Q
        state_and_Y[ndim:] = Q.flatten()

    # ── Average Lyapunov exponents ──
    lyap = lyap_sum / (n_steps * DT_RENORM)
    lyap = np.sort(lyap)[::-1]  # Descending order

    # ── Consistency check ──
    expected_sum = -(SIGMA + 1.0 + BETA)
    actual_sum = np.sum(lyap)
    print(f"    Σλ = {actual_sum:.4f} (expected {expected_sum:.4f}, "
          f"error = {abs(actual_sum - expected_sum):.4f})")

    # ── Kaplan-Yorke dimension ──
    d_ky = kaplan_yorke_dimension(lyap)

    return lyap, d_ky


def kaplan_yorke_dimension(lyap_sorted):
    """
    Kaplan-Yorke dimension from sorted (descending) Lyapunov exponents.

    D_KY = j + S_j / |λ_{j+1}|

    where j = max integer such that S_j = Σ_{i=1}^j λ_i ≥ 0.
    See the Mackey-Glass script docstring for full derivation.
    """
    cum = np.cumsum(lyap_sorted)
    negative_indices = np.where(cum < 0)[0]

    if len(negative_indices) == 0:
        return float(len(lyap_sorted))
    elif negative_indices[0] == 0:
        return 0.0
    else:
        j = negative_indices[0]
        return j + cum[j - 1] / abs(lyap_sorted[j])


# ══════════════════════════════════════════════════════════════════════════
# 4. TRAJECTORY GENERATION
# ══════════════════════════════════════════════════════════════════════════
def get_trajectory(rho, t_max=100.0, dt=0.01):
    """
    Integrate the Lorenz system and return x(t), y(t), z(t) after
    discarding transients. Used for attractor and time-series plots.
    """
    ic = [1.0, 1.0, 1.0]
    # Let transient decay
    sol_trans = solve_ivp(lorenz63, [0, T_TRANS], ic, args=(rho,),
                          method='RK45', rtol=1e-9, atol=1e-11)
    x0 = sol_trans.y[:, -1]

    # Production run
    t_span = [0, t_max]
    t_eval = np.arange(0, t_max, dt)
    sol = solve_ivp(lorenz63, t_span, x0, args=(rho,),
                    method='RK45', t_eval=t_eval, rtol=1e-9, atol=1e-11)
    return sol.t, sol.y  # sol.y is (3, N): [x, y, z]


# ══════════════════════════════════════════════════════════════════════════
# 5. DATA EXPORT
# ══════════════════════════════════════════════════════════════════════════
def export_data(results, base_filename="../../figures/lorenz_63/../data/lorenz_63/lorenz63_scan_data"):
    """Export results to JSON and CSV."""
    with open(f"{base_filename}.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(f"{base_filename}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Rho", "D_KY", "Lambda_1", "Sum_Lambda", "Full_Spectrum"])
        for key, r in sorted(results.items(), key=lambda x: float(x[0])):
            writer.writerow([
                r["rho"], r["D_KY"], r["lambda_1"],
                r["sum_lambda"],
                [round(l, 5) for l in r["full_spectrum"]]
            ])
    print(f"\nExported: {base_filename}.json and {base_filename}.csv")


# ══════════════════════════════════════════════════════════════════════════
# 6. PLOTTING
# ══════════════════════════════════════════════════════════════════════════
def plot_metrics(results_dict):
    """D_KY and λ_1 vs ρ — shows route to chaos."""
    rhos = sorted([float(k) for k in results_dict.keys()])
    d_kys = [results_dict[str(int(r) if r == int(r) else r)]["D_KY"] for r in rhos]
    lambda_1s = [results_dict[str(int(r) if r == int(r) else r)]["lambda_1"] for r in rhos]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.set_xlabel(r'Rayleigh Number $\rho$', fontsize=12, fontweight='bold')
    ax1.set_ylabel(r'$D_{KY}$', color='tab:blue', fontsize=12, fontweight='bold')
    ax1.plot(rhos, d_kys, 'o-', color='tab:blue', linewidth=2.5, markersize=7)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax2 = ax1.twinx()
    ax2.set_ylabel(r'$\lambda_1$', color='tab:red', fontsize=12, fontweight='bold')
    ax2.plot(rhos, lambda_1s, 's-', color='tab:red', linewidth=2.5, markersize=7)
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.axhline(0, color='black', linewidth=1.5, linestyle='--')

    plt.title("Lorenz '63: Route to Chaos", fontsize=14, fontweight='bold', pad=15)
    fig.tight_layout()
    plt.savefig("../../figures/lorenz_63/lorenz63_manuscript_metrics.png", dpi=300)
    print("Saved: ../figures/lorenz_63/lorenz63_manuscript_metrics.png")


def plot_attractors(rho_list, results_dict=None):
    """
    3-row figure for each ρ:
      Row 1: 3D phase-space projection (x vs z)
      Row 2: Time series x(t)
      Row 3: Lyapunov spectrum bar chart
    """
    n = len(rho_list)
    fig = plt.figure(figsize=(4.5 * n, 12))
    gs = GridSpec(3, n, figure=fig, height_ratios=[1, 0.7, 0.6], hspace=0.35)

    for i, rho in enumerate(rho_list):
        rho_key = str(int(rho)) if rho == int(rho) else str(float(rho))
        print(f"Generating visuals for ρ = {rho}...")

        t_arr, xyz = get_trajectory(rho, t_max=100.0, dt=0.01)

        # Look up results
        d_ky_val = None
        lyap_spectrum = None
        if results_dict is not None and rho_key in results_dict:
            d_ky_val = results_dict[rho_key]["D_KY"]
            lyap_spectrum = results_dict[rho_key]["full_spectrum"]

        regime = REGIME_LABELS.get(int(rho), "")
        title_dky = f"$D_{{KY}}={d_ky_val:.2f}$" if d_ky_val is not None else ""
        title_top = f"$\\rho={rho}$ — {regime}\n{title_dky}"

        # ═══ Row 1: Phase-space attractor (x vs z projection) ═══
        ax_att = fig.add_subplot(gs[0, i])
        ax_att.plot(xyz[0], xyz[2], color='black', linewidth=0.3, alpha=0.7)
        ax_att.set_title(title_top, fontsize=11, fontweight='bold')
        ax_att.set_xlabel(r"$x$", fontsize=10)
        ax_att.set_ylabel(r"$z$", fontsize=10)
        ax_att.grid(True, linestyle=':', alpha=0.5)

        # ═══ Row 2: Time series x(t) ═══
        ax_ts = fig.add_subplot(gs[1, i])
        ax_ts.plot(t_arr, xyz[0], color='tab:blue', linewidth=0.5)
        ax_ts.set_xlabel(r"$t$", fontsize=10)
        ax_ts.set_ylabel(r"$x(t)$", fontsize=10)
        ax_ts.set_title("Time series", fontsize=10)
        ax_ts.grid(True, linestyle=':', alpha=0.5)

        # ═══ Row 3: Lyapunov spectrum ═══
        ax_ly = fig.add_subplot(gs[2, i])
        if lyap_spectrum is not None:
            spec = [l for l in lyap_spectrum if np.isfinite(l)]
            indices = np.arange(len(spec))
            colors = ['tab:red' if l > 0 else ('gold' if abs(l) < 0.01
                       else 'tab:blue') for l in spec]
            ax_ly.bar(indices, spec, color=colors, edgecolor='black',
                      linewidth=0.5, width=0.8)
            ax_ly.axhline(0, color='black', linewidth=1)
            ax_ly.set_xlabel("Index $i$", fontsize=9)
            ax_ly.set_ylabel(r"$\lambda_i$", fontsize=10)
            ax_ly.set_title("Lyapunov spectrum", fontsize=10)
            ax_ly.grid(True, linestyle=':', alpha=0.5, axis='y')
        else:
            ax_ly.text(0.5, 0.5, "No data", ha='center', va='center',
                       transform=ax_ly.transAxes, fontsize=12, color='gray')

    fig.suptitle("Lorenz '63 — Route to Chaos",
                 fontsize=15, fontweight='bold', y=1.01)
    fig.tight_layout()
    plt.savefig("../../figures/lorenz_63/lorenz63_educational_attractors.png", dpi=300, bbox_inches='tight')
    print("Saved: ../figures/lorenz_63/lorenz63_educational_attractors.png")


# ══════════════════════════════════════════════════════════════════════════
# 7. MAIN
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    results = {}

    print("=" * 70)
    print("Lorenz '63 — Lyapunov Spectrum & Kaplan-Yorke Dimension")
    print("=" * 70)

    for rho in RHO_SCAN:
        rho_key = str(int(rho)) if rho == int(rho) else str(float(rho))
        print(f"ρ = {rho:6.1f} | Computing 3 exponents... ", end="", flush=True)
        t0 = time.time()

        lyap, d_ky = compute_lyapunov_ode(rho)

        elapsed = time.time() - t0
        n_pos = int(sum(1 for l in lyap if l > 0))

        results[rho_key] = {
            "rho": float(rho),
            "D_KY": round(d_ky, 3),
            "lambda_1": round(float(lyap[0]), 5),
            "sum_lambda": round(float(np.sum(lyap)), 5),
            "n_positive_exponents": n_pos,
            "full_spectrum": [float(l) for l in lyap],
        }

        print(f"  Done in {elapsed:5.1f}s | D_KY={d_ky:5.2f} | "
              f"λ_1={lyap[0]:8.5f} | Σλ={np.sum(lyap):8.4f}")

    export_data(results)
    plot_metrics(results)
    plot_attractors(RHO_ATTRACTORS, results_dict=results)

    print("\nAll tasks completed successfully!")
    plt.show()
