#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════╗
║        MACKEY-GLASS DELAY-DIFFERENTIAL EQUATION (DDE)                 ║
║        Lyapunov Spectrum, Kaplan-Yorke Dimension & Attractors         ║
╚══════════════════════════════════════════════════════════════════════════╝

PHYSICAL BACKGROUND
───────────────────
The Mackey-Glass equation was introduced by M. Mackey and L. Glass (1977,
"Oscillation and chaos in physiological control systems", Science 197:287)
as a model of blood cell regulation (hematopoiesis). The variable x(t)
represents the concentration of circulating blood cells, governed by:

    dx/dt = β x(t-τ) / [1 + x(t-τ)^c] − γ x(t)

where:
  - β  = 0.2    production rate
  - γ  = 0.1    destruction rate  (decay)
  - c  = 10     Hill coefficient   (nonlinearity of the feedback)
  - τ           time delay in the feedback loop (control parameter)

The delay τ is the key bifurcation parameter:
  - Small τ  → the system relaxes to a stable fixed point
  - τ ≈ 6-8  → stable limit cycles emerge (Hopf bifurcation)
  - τ ≈ 17   → period-doubling cascade → onset of chaos
  - τ > 17   → chaotic attractor with increasing complexity
  - τ >> 30  → "hyperchaos" (multiple positive Lyapunov exponents)

Because DDEs are infinite-dimensional systems (the state is the entire
function x(t') for t-τ ≤ t' ≤ t), they can in principle have infinitely
many Lyapunov exponents. In practice we compute a finite number n_lyap
that captures the most relevant ones.

LYAPUNOV EXPONENTS — THEORY
───────────────────────────
Lyapunov exponents λ_i measure the average exponential rate of divergence
(or convergence) of infinitesimally close trajectories in phase space.

For a dynamical system with trajectory x(t), consider a small perturbation
δx(0). After time T, this perturbation grows as:

    ‖δx(T)‖ ~ ‖δx(0)‖ · exp(λ · T)

More precisely, the i-th Lyapunov exponent is:

    λ_i = lim_{T→∞} (1/T) · ln( σ_i(T) )

where σ_i(T) are the singular values of the linearized flow map
(the fundamental matrix / propagator) Φ(0,T). Equivalently, they come
from the eigenvalues of the Oseledets matrix:

    Λ = lim_{T→∞} [ Φ(0,T)^T · Φ(0,T) ]^{1/(2T)}

PRACTICAL COMPUTATION IN CODE (jitcdde)
────────────────────────────────────────
The jitcdde_lyap library implements the standard Benettin algorithm
adapted for DDEs:

  1. The DDE is integrated alongside n_lyap tangent-linear (variational)
     equations — one for each exponent to be computed.
  2. These variational equations describe how infinitesimal perturbation
     vectors evolve under the linearized dynamics (the Jacobian of the
     DDE right-hand side, including delayed terms).
  3. At each step of length Δt, the perturbation vectors are QR-
     orthogonalized (Gram-Schmidt). The logarithms of the diagonal
     elements of R give the "local" Lyapunov exponents for that step:
         λ_i^{local} = ln(R_ii) / Δt
  4. These local exponents are time-averaged over a long trajectory
     (after discarding transients) to yield the global exponents.

The integrate() method of jitcdde_lyap returns a 3-tuple:
    (state, local_lyapunov_exponents, integration_dt)

We accumulate:  Σ_steps  λ_i^{local} · Δt_step   /   Σ_steps Δt_step

CHAOS CRITERION
───────────────
  - If λ_1 > 0  →  the system is CHAOTIC  (sensitive dependence on IC)
  - If λ_1 = 0  →  limit cycle / quasi-periodic  (neutral stability)
  - If λ_1 < 0  →  stable fixed point  (all perturbations decay)
  - If ≥2 λ_i > 0 → HYPERCHAOS  (chaos in multiple independent directions)

KAPLAN-YORKE DIMENSION — THEORY
────────────────────────────────
The Kaplan-Yorke (KY) dimension (also called the Lyapunov dimension)
estimates the information dimension of the attractor from the Lyapunov
spectrum. It was introduced by Kaplan & Yorke (1979).

DEFINITION. Order the Lyapunov exponents in decreasing order:
    λ_1 ≥ λ_2 ≥ ... ≥ λ_n

Define the partial sums:
    S_j = Σ_{i=1}^{j} λ_i

Find the largest integer j such that S_j ≥ 0. Then:

    D_KY = j + S_j / |λ_{j+1}|

INTERPRETATION. The Kaplan-Yorke dimension can be understood as follows:
  - The first j exponents "expand" phase-space volume along j directions,
    with total expansion rate S_j.
  - The (j+1)-th exponent contracts volume at rate |λ_{j+1}|.
  - The fractional part S_j/|λ_{j+1}| measures "how much" of the next
    direction is filled before contraction overwhelms expansion.
  - D_KY gives the effective dimensionality of the strange attractor.

EXAMPLES:
  - Fixed point:    D_KY = 0           (0-dimensional)
  - Limit cycle:    D_KY ≈ 1           (1-dimensional curve)
  - Torus:          D_KY ≈ 2           (2-dimensional surface)
  - Lorenz attr.:   D_KY ≈ 2.06        (fractal between 2D and 3D)
  - MG at τ=17:     D_KY ≈ 2.1         (onset of chaos)
  - MG at τ=30:     D_KY ≈ 3.6         (high-dimensional chaos)

THE KAPLAN-YORKE CONJECTURE states that D_KY equals the information
dimension D_1 for "typical" attractors. This has been proven for some
classes of systems (Ledrappier-Young, 1985) but remains a conjecture
in general.

COMPUTING D_KY FROM DISCRETE TIME SERIES (EXPERIMENTAL DATA)
──────────────────────────────────────────────────────────────
If you only have a measured scalar time series x(t_1), x(t_2), ..., x(t_N)
(e.g., from an experiment), you can still estimate D_KY via:

  1. DELAY EMBEDDING (Takens' theorem, 1981):
     Reconstruct the attractor in m-dimensional space using delay vectors:
         X_i = [x(t_i), x(t_i + τ_e), x(t_i + 2τ_e), ..., x(t_i + (m-1)τ_e)]
     where τ_e is the embedding delay (chosen via mutual information or
     autocorrelation) and m is the embedding dimension (chosen via false
     nearest neighbors). Takens' theorem guarantees that for m ≥ 2d+1
     (where d is the attractor's box-counting dimension), the reconstructed
     attractor is diffeomorphic to the true attractor.

  2. LYAPUNOV EXPONENTS FROM TIME SERIES:
     • Wolf's algorithm (Wolf et al., 1985): tracks divergence of nearby
       trajectories in the reconstructed phase space → estimates λ_1.
     • Rosenstein's algorithm (1993): more robust estimate of λ_1.
     • Sano-Sawada / Eckmann-Ruelle methods: estimate the full spectrum
       by fitting local linear maps (Jacobians) to clusters of nearby
       points in the embedding, then computing average growth rates.
     • Python packages: nolds, pyunicorn, tisean (C library with Python
       wrappers) all implement these algorithms.

  3. Apply the Kaplan-Yorke formula to the estimated spectrum.

  CAVEAT: These methods require long, low-noise time series and careful
  choice of embedding parameters. Results are less reliable than direct
  computation from known equations (what we do here).
"""

import numpy as np
from jitcdde import jitcdde_lyap, jitcdde, y, t
import json
import csv
import os
import time
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ══════════════════════════════════════════════════════════════════════════
# 1. GLOBAL PARAMETERS — Mackey-Glass DDE
# ══════════════════════════════════════════════════════════════════════════
#
# The Mackey-Glass equation:
#   dx/dt = β · x(t-τ) / [1 + x(t-τ)^c] − γ · x(t)
#
# The first term is a Hill function: it models a saturating production rate
# that depends on the delayed state x(t-τ). At low x, production ≈ β·x.
# At high x, the term x^c dominates the denominator and production → 0.
# The Hill coefficient c=10 makes the sigmoid transition very sharp.

BETA   = 0.2     # Production rate coefficient
GAMMA  = 0.1     # Linear decay rate (destruction of blood cells)
C_EXP  = 10.0    # Hill coefficient (c=10 → steep nonlinear feedback)

# ── Parameter scan range ──
# We scan τ over a range that covers ALL dynamical regimes:
#   τ = 5   → stable fixed point  (feedback too fast to destabilize)
#   τ = 10  → limit cycle          (Hopf bifurcation has occurred)
#   τ = 17  → onset of chaos       (period-doubling cascade complete)
#   τ = 30  → developed chaos      (D_KY ≈ 3.5, 2 positive λ's)
#   τ = 50  → high-dimensional chaos (D_KY > 5)
TAU_SCAN  = [5, 10, 15, 17, 20, 25, 30, 35, 40, 50]

# ── Integration parameters ──
T_TRANS  = 5000.0    # Transient time to discard (let attractor settle)
T_LYAP   = 20000.0   # Integration time for Lyapunov averaging
DT       = 10.0      # Sampling interval for Lyapunov accumulation

# ── Attractor visualization: 5 representative τ values ──
# These span: fixed-point → limit-cycle → chaos-onset → chaotic → hyperchaotic
TAU_ATTRACTORS = [5, 10, 17, 30, 50]

# Regime labels for each attractor τ
REGIME_LABELS = {
    5:  "Fixed Point",
    10: "Limit Cycle",
    17: "Onset of Chaos",
    30: "Chaotic",
    50: "Strongly Chaotic",
}


# ══════════════════════════════════════════════════════════════════════════
# 2. COMPUTATION: LYAPUNOV EXPONENTS
# ══════════════════════════════════════════════════════════════════════════
def compute_lyapunov(tau, n_lyap):
    """
    Compute the n_lyap largest Lyapunov exponents and the Kaplan-Yorke
    dimension for the Mackey-Glass DDE at delay τ.

    ALGORITHM (Benettin et al., 1980, adapted for DDEs by jitcdde):
    ─────────────────────────────────────────────────────────────────
    1. Define the DDE:  dx/dt = β x(t-τ)/[1+x(t-τ)^c] − γ x(t)
    2. Augment with n_lyap variational (tangent-linear) equations.
       These track how infinitesimal perturbation vectors δx_i(t)
       evolve under the LINEARIZED dynamics:
           dδx_i/dt = J(t) · δx_i(t) + J_τ(t) · δx_i(t-τ)
       where J and J_τ are the Jacobians w.r.t. current and delayed state.
    3. Integrate the augmented system. After each step Δt:
       a) QR-decompose the perturbation matrix: [δx_1|...|δx_n] = Q·R
       b) Record local exponents: λ_i^{local} = ln(R_ii) / Δt
       c) Replace perturbation vectors with the orthonormal columns Q.
    4. Average the local exponents over time → global Lyapunov exponents.

    Parameters
    ----------
    tau : float
        Time delay in the Mackey-Glass equation.
    n_lyap : int
        Number of largest Lyapunov exponents to compute.

    Returns
    -------
    lyap : np.ndarray
        Sorted (descending) array of Lyapunov exponents.
    d_ky : float
        Kaplan-Yorke dimension.
    """
    # ── Define the Mackey-Glass DDE ──
    # y(0) is the current state x(t), y(0, t-tau) is x(t-τ)
    f = [BETA * y(0, t - tau) / (1.0 + y(0, t - tau)**C_EXP) - GAMMA * y(0)]

    # ── Create DDE solver with Lyapunov computation ──
    # jitcdde_lyap automatically generates the variational equations
    # by symbolically differentiating f w.r.t. y(0) and y(0, t-tau).
    # The Jacobian entries are:
    #   ∂f/∂x(t)   = −γ
    #   ∂f/∂x(t-τ) = β [1 + x(t-τ)^c − c·x(t-τ)^c] / [1 + x(t-τ)^c]^2
    #              = β / [1 + x(t-τ)^c]^2 · [1 + (1-c)·x(t-τ)^c]
    DDE = jitcdde_lyap(f, n_lyap=n_lyap)
    DDE.constant_past([0.9])

    # Smooth out initial discontinuity (the constant past is not a
    # continuous derivative of the DDE solution — this resolves it)
    DDE.step_on_discontinuities()

    # ── Discard transient ──
    # The attractor is reached after ~5000 time units for most τ values.
    DDE.integrate(DDE.t + T_TRANS)

    # ── Accumulate Lyapunov exponents ──
    # We use time-weighted averaging because integration steps may vary:
    #   λ_i = Σ (λ_i^{local} · Δt) / Σ Δt
    times = np.arange(DDE.t + DT, DDE.t + T_LYAP, DT)

    lyap_accum = np.zeros(n_lyap)
    total_time = 0.0

    for current_time in times:
        # integrate() returns: (state, local_lyap_exponents, dt_step)
        # where local_lyap_exponents are ln(R_ii)/Δt from the QR step.
        _, local_lyap, dt_step = DDE.integrate(current_time)

        # Time-weighted accumulation:
        lyap_accum += local_lyap * dt_step
        total_time += dt_step

    # Time-averaged global exponents:
    lyap = lyap_accum / total_time if total_time > 0 else lyap_accum

    # ── Sort in descending order (λ_1 ≥ λ_2 ≥ ... ≥ λ_n) ──
    lyap = np.sort(lyap)[::-1]

    # ── Compute Kaplan-Yorke dimension ──
    d_ky = kaplan_yorke_dimension(lyap)

    return lyap, d_ky


def kaplan_yorke_dimension(lyap_sorted):
    """
    Compute the Kaplan-Yorke dimension from a SORTED (descending)
    Lyapunov spectrum.

    DERIVATION
    ──────────
    Given λ_1 ≥ λ_2 ≥ ... ≥ λ_n, define partial sums:

        S_j = Σ_{i=1}^{j} λ_i

    The Kaplan-Yorke dimension is:

        D_KY = j_max + S_{j_max} / |λ_{j_max + 1}|

    where j_max is the largest integer j such that S_j ≥ 0.

    GEOMETRIC MEANING: In phase space, the first j_max directions
    expand (or are neutral), and the (j_max+1)-th direction contracts.
    The attractor "fills" j_max full dimensions plus a fraction of
    the next dimension, determined by the ratio of expansion to
    contraction.

    SPECIAL CASES:
      - All λ_i < 0:  D_KY = 0 (stable fixed point, 0D attractor)
      - S_1 = λ_1 ≈ 0, S_2 < 0: D_KY ≈ 1 (limit cycle)
      - All S_j ≥ 0:  D_KY = n (attractor fills the full computed space;
        need more exponents to resolve the dimension)
    """
    cum = np.cumsum(lyap_sorted)

    # Find the first index where the cumulative sum goes negative
    negative_indices = np.where(cum < 0)[0]

    if len(negative_indices) == 0:
        # All partial sums are non-negative → attractor fills the
        # entire computed subspace. D_KY is at least n_lyap.
        return float(len(lyap_sorted))
    elif negative_indices[0] == 0:
        # Even λ_1 < 0 → stable fixed point
        return 0.0
    else:
        # j is the first index where S_j < 0, so j_max = j - 1
        # (using 0-indexed arrays: j_max in the formula = j here)
        j = negative_indices[0]
        # D_KY = j + S_{j-1} / |λ_j|
        #   where S_{j-1} = cum[j-1] and λ_j = lyap_sorted[j]
        return j + cum[j - 1] / abs(lyap_sorted[j])


# ══════════════════════════════════════════════════════════════════════════
# 3. PHASE-SPACE DATA GENERATION
# ══════════════════════════════════════════════════════════════════════════
def get_phase_space_data(tau, t_max=1000, dt=1.0):
    """
    Integrate the Mackey-Glass DDE and return the time series x(t)
    AFTER discarding T_TRANS transient time. This data is used for:
      - Phase-space reconstruction: plotting x(t) vs x(t-τ)
      - Time-series visualization: plotting x(t) vs t

    The delay-embedding x(t) vs x(t-τ) is physically natural here
    because τ already appears in the equation. For a general system
    one would use Takens embedding with a computed embedding delay.
    """
    f = [BETA * y(0, t - tau) / (1.0 + y(0, t - tau)**C_EXP) - GAMMA * y(0)]
    DDE = jitcdde(f)
    DDE.constant_past([0.9])
    DDE.step_on_discontinuities()
    DDE.integrate(DDE.t + T_TRANS)

    times = np.arange(DDE.t + dt, DDE.t + t_max, dt)
    data = [DDE.integrate(time_val)[0] for time_val in times]
    return np.array(data)


# ══════════════════════════════════════════════════════════════════════════
# 4. DATA EXPORT
# ══════════════════════════════════════════════════════════════════════════
def export_data(results, base_filename="../../data/mackiney_glass/mg_scan_data"):
    """Export results to JSON and CSV."""
    with open(f"{base_filename}.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(f"{base_filename}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Tau", "D_KY", "Lambda_1", "N_Positive", "Full_Spectrum"])
        for tau_str, r in sorted(results.items(), key=lambda x: float(x[0])):
            writer.writerow([
                r["tau"], r["D_KY"], r["lambda_1"],
                r["n_positive_exponents"],
                [round(l, 5) for l in r["full_spectrum"]]
            ])
    print(f"\nExported: {base_filename}.json and {base_filename}.csv")


# ══════════════════════════════════════════════════════════════════════════
# 5. PLOTTING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════
def plot_metrics(results_dict):
    """
    Plot D_KY and λ_1 vs τ on a dual-axis figure.
    This shows the route to chaos: as τ increases, λ_1 becomes positive
    (chaos onset) and D_KY rises from ~1 (limit cycle) to >3 (hyperchaos).
    """
    taus = sorted([float(k) for k in results_dict.keys()])
    d_kys = [results_dict[str(int(t) if t == int(t) else t)]["D_KY"] for t in taus]
    lambda_1s = [results_dict[str(int(t) if t == int(t) else t)]["lambda_1"] for t in taus]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color1 = 'tab:blue'
    ax1.set_xlabel(r'Time Delay ($\tau$)', fontsize=12, fontweight='bold')
    ax1.set_ylabel(r'Kaplan-Yorke Dimension ($D_{KY}$)', color=color1,
                   fontsize=12, fontweight='bold')
    ax1.plot(taus, d_kys, 'o-', color=color1, linewidth=2.5, markersize=7)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel(r'Max Lyapunov Exponent ($\lambda_1$)', color=color2,
                   fontsize=12, fontweight='bold')
    ax2.plot(taus, lambda_1s, 's-', color=color2, linewidth=2.5, markersize=7)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.axhline(0, color='black', linewidth=1.5, linestyle='--')

    plt.title("Mackey-Glass: Route to Chaos", fontsize=14, fontweight='bold', pad=15)
    fig.tight_layout()
    plt.savefig("../../figures/mackiney_glass/mg_manuscript_metrics.png", dpi=300)
    print("Saved: ../figures/mackiney_glass/mg_manuscript_metrics.png")


def plot_attractors(tau_list, results_dict=None):
    """
    Generates a 3-row figure for each τ value in tau_list:
      Row 1: Phase-space attractor x(t) vs x(t-τ), with D_KY in title
      Row 2: Time series x(t) vs t
      Row 3: Lyapunov spectrum bar chart (positive exponents highlighted)
    """
    n = len(tau_list)
    fig = plt.figure(figsize=(4.5 * n, 12))
    gs = GridSpec(3, n, figure=fig, height_ratios=[1, 0.7, 0.6], hspace=0.35)

    dt_sim = 0.5

    for i, tau in enumerate(tau_list):
        tau_key = str(int(tau)) if tau == int(tau) else str(float(tau))
        print(f"Generating visuals for τ = {tau}...")
        ts = get_phase_space_data(tau, t_max=2000, dt=dt_sim)

        # ── Look up D_KY and regime label ──
        d_ky_val = None
        lyap_spectrum = None
        if results_dict is not None and tau_key in results_dict:
            d_ky_val = results_dict[tau_key]["D_KY"]
            lyap_spectrum = results_dict[tau_key]["full_spectrum"]

        regime = REGIME_LABELS.get(int(tau), "")
        title_dky = f"$D_{{KY}}={d_ky_val:.2f}$" if d_ky_val is not None else ""
        title_top = f"$\\tau={tau}$ — {regime}\n{title_dky}"

        # ═══ Row 1: Phase-space attractor ═══
        delay_steps = int(tau / dt_sim)
        x_t = ts[delay_steps:]
        x_t_minus_tau = ts[:-delay_steps]

        ax_att = fig.add_subplot(gs[0, i])
        ax_att.plot(x_t_minus_tau, x_t, color='black', linewidth=0.4, alpha=0.8)
        ax_att.set_title(title_top, fontsize=11, fontweight='bold')
        ax_att.set_xlabel(r"$x(t - \tau)$", fontsize=10)
        ax_att.set_ylabel(r"$x(t)$", fontsize=10)
        ax_att.grid(True, linestyle=':', alpha=0.5)

        # ═══ Row 2: Time series x(t) vs t ═══
        t_axis = np.arange(len(ts)) * dt_sim
        ax_ts = fig.add_subplot(gs[1, i])
        ax_ts.plot(t_axis, ts, color='tab:blue', linewidth=0.5)
        ax_ts.set_xlabel(r"$t$", fontsize=10)
        ax_ts.set_ylabel(r"$x(t)$", fontsize=10)
        ax_ts.set_title(f"Time series", fontsize=10)
        ax_ts.grid(True, linestyle=':', alpha=0.5)

        # ═══ Row 3: Lyapunov spectrum ═══
        ax_ly = fig.add_subplot(gs[2, i])
        if lyap_spectrum is not None:
            # Filter out -inf values and take first 15 for display
            spec = [l for l in lyap_spectrum if np.isfinite(l)][:15]
            indices = np.arange(len(spec))
            colors = ['tab:red' if l > 0 else ('gold' if abs(l) < 1e-3
                       else 'tab:blue') for l in spec]
            ax_ly.bar(indices, spec, color=colors, edgecolor='black',
                      linewidth=0.5, width=0.8)
            ax_ly.axhline(0, color='black', linewidth=1, linestyle='-')
            ax_ly.set_xlabel("Exponent index $i$", fontsize=9)
            ax_ly.set_ylabel(r"$\lambda_i$", fontsize=10)
            ax_ly.set_title("Lyapunov spectrum", fontsize=10)
            ax_ly.grid(True, linestyle=':', alpha=0.5, axis='y')
        else:
            ax_ly.text(0.5, 0.5, "No data", ha='center', va='center',
                       transform=ax_ly.transAxes, fontsize=12, color='gray')

    fig.suptitle("Mackey-Glass DDE — Route to Chaos",
                 fontsize=15, fontweight='bold', y=1.01)
    fig.tight_layout()
    plt.savefig("../../figures/mackiney_glass/mg_educational_attractors.png", dpi=300, bbox_inches='tight')
    print("Saved: ../figures/mackiney_glass/mg_educational_attractors.png")


# ══════════════════════════════════════════════════════════════════════════
# 6. MAIN — PARAMETER SCAN & VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    results = {}

    print("=" * 70)
    print("Mackey-Glass DDE — Lyapunov Spectrum & Kaplan-Yorke Dimension")
    print("=" * 70)

    # ── Compute Lyapunov spectra ──
    # We pick n_lyap adaptively: more exponents for larger τ,
    # since higher τ → more "active" Lyapunov directions.
    for tau in TAU_SCAN:
        tau_clean = float(round(tau, 2))
        n_exp = max(15, int(tau_clean * 0.6))

        print(f"τ = {tau_clean:5.1f} | Computing {n_exp} exponents... ",
              end="", flush=True)
        t0 = time.time()

        lyap, d_ky = compute_lyapunov(tau_clean, n_lyap=n_exp)

        elapsed = time.time() - t0
        n_pos = int(sum(1 for l in lyap if l > 0))

        results[str(int(tau) if tau == int(tau) else tau)] = {
            "tau": tau_clean,
            "D_KY": round(d_ky, 3),
            "lambda_1": round(float(lyap[0]), 5),
            "n_positive_exponents": n_pos,
            "full_spectrum": [float(l) for l in lyap]
        }

        print(f"Done in {elapsed:6.1f}s | D_KY={d_ky:5.2f} | "
              f"λ_1={lyap[0]:8.5f} | #(λ>0)={n_pos}")

    # ── Export data ──
    export_data(results)

    # ── Plot metrics ──
    plot_metrics(results)

    # ── Plot attractors with all 5 regimes ──
    plot_attractors(TAU_ATTRACTORS, results_dict=results)

    print("\nAll tasks completed successfully!")
    plt.show()
