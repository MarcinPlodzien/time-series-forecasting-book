#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════╗
║   ENSO NIÑO 3.4 — El Niño-Southern Oscillation SST Anomalies         ║
║   D_KY Estimation from Experimental Time Series                       ║
╚══════════════════════════════════════════════════════════════════════════╝

DATASET: ENSO Niño 3.4 Index
─────────────────────────────
The Niño 3.4 index is the area-averaged Sea Surface Temperature (SST)
anomaly over the equatorial Pacific region 5°N–5°S, 170°W–120°W.
It is the PRIMARY index used to define El Niño and La Niña events:

  El Niño:   Niño 3.4 index > +0.5°C for ≥5 consecutive 3-month averages
  La Niña:   Niño 3.4 index < −0.5°C for ≥5 consecutive 3-month averages
  Neutral:   −0.5°C ≤ Niño 3.4 ≤ +0.5°C

THE PHYSICS:
  The El Niño-Southern Oscillation is a COUPLED OCEAN-ATMOSPHERE phenomenon
  in the tropical Pacific. It involves a feedback loop:

  1. NORMAL STATE: Trade winds blow westward along the equator, piling up
     warm water in the western Pacific (warm pool) and causing cold water
     upwelling along the South American coast.

  2. EL NIÑO: Trade winds weaken → warm water spreads eastward → further
     weakens trade winds (Bjerknes feedback). The Walker circulation
     collapses, and SST anomalies peak in the central-eastern Pacific.

  3. LA NIÑA: Trade winds strengthen → enhanced upwelling → cold anomalies
     in the eastern Pacific → stronger Walker circulation.

  The quasi-periodic oscillation (~2–7 years) arises from the interplay
  of oceanic Kelvin and Rossby waves that propagate across the Pacific,
  providing the "delayed negative feedback" that turns El Niño into La Niña
  and vice versa (the delayed-oscillator or recharge-oscillator theory).

DYNAMICAL CHARACTER:
  - The ENSO cycle is QUASI-PERIODIC (dominant period ~3–5 years)
  - There is IRREGULARITY in amplitude and period → possible low-dim chaos
  - The question "is ENSO chaotic?" is actively debated:
      • Simple delay-oscillator models → clearly chaotic for some params
      • Coupled GCMs → mix of chaos + stochastic weather noise forcing
      • From observations: D_KY ≈ 3–4 has been estimated, but the short
        record (~100–150 years) makes this uncertain
  - The D_KY estimate tells us: even if ENSO is chaotic, it lives on a
    LOW-DIMENSIONAL attractor (3–4D), meaning its dynamics can potentially
    be captured by a small set of coupled ODEs (Lorenz-like models)

REFERENCES:
  Bjerknes, J. (1969), "Atmospheric teleconnections from the equatorial
    Pacific", Mon. Wea. Rev.
  Suarez & Schopf (1988), "A delayed action oscillator for ENSO",
    J. Atmos. Sci.
  Jin (1997), "An equatorial ocean recharge paradigm for ENSO",
    J. Atmos. Sci.

════════════════════════════════════════════════════════════════════════════

ROSENSTEIN DIVERGENCE — DETAILED EXPLANATION
────────────────────────────────────────────
The "Rosenstein divergence" refers to the diagnostic plot produced by
Rosenstein's algorithm (Rosenstein, Collins & De Luca, 1993, "A practical
method for calculating largest Lyapunov exponents from small data sets",
Physica D 65:117-134).

WHAT IT SHOWS:
  The y-axis plots  ⟨ ln d(k) ⟩  —  the AVERAGE logarithmic distance
  between nearest-neighbor pairs in the reconstructed phase space, as a
  function of the time lag k (x-axis).

  If the system is chaotic:
    ⟨ ln d(k) ⟩ ≈ λ_1 · k · Δt + constant

    i.e., the curve rises LINEARLY at early times. The SLOPE of this
    linear region equals the largest Lyapunov exponent λ_1.

HOW TO READ THE PLOT:
  - LINEAR RISE at early k → the system is chaotic, slope = λ_1
  - FLAT / no rise → λ_1 ≈ 0, the system is periodic or quasi-periodic
  - NEGATIVE slope → λ_1 < 0, stable (nearby trajectories converge)
  - SATURATION at large k → neighboring trajectories have diverged to
    the full extent of the attractor (diameter of the attractor in log scale)

INTUITION:
  Imagine two nearby ants walking on the attractor. If it is chaotic,
  they start close but their paths exponentially diverge:
    distance(k) ~ d(0) · e^{λ_1 · k}
  Taking log:
    ln distance(k) ~ ln d(0) + λ_1 · k

  So plotting ⟨ln d(k)⟩ vs k gives a straight line whose slope IS λ_1.
  Once the ants are as far apart as the attractor allows, the curve
  saturates (they can't get farther apart than the attractor's diameter).

WHY ROSENSTEIN IS BETTER THAN WOLF:
  Wolf's (1985) algorithm re-selects neighbors periodically, which
  introduces selection bias. Rosenstein tracks the SAME pair over time,
  giving a cleaner estimate — especially for short, noisy data.

RAW DATA ATTRACTOR (DELAY PLOT)
───────────────────────────────
When we plot x(t) vs x(t + τ) directly from the raw time series, we are
constructing a 2D Takens delay embedding. This IS a legitimate attractor
reconstruction (Takens' theorem guarantees this for m ≥ 2d+1, so m=2 may
not fully unfold the attractor, but it reveals its qualitative structure).

For ENSO:
  - The raw delay plot shows the characteristic "noisy torus" or "chaotic
    limit cycle" structure of the ENSO oscillation
  - El Niño events appear as excursions to the right/top
  - La Niña events appear as excursions to the left/bottom
"""

import numpy as np
import os
import json
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.spatial import KDTree


# ══════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE  = os.path.join(SCRIPT_DIR, "../../data/enso_nino34", "enso_nino34.txt")
DATA_NAME  = "ENSO Niño 3.4"


# ══════════════════════════════════════════════════════════════════════════
# 2. EMBEDDING PARAMETER SELECTION
# ══════════════════════════════════════════════════════════════════════════
def autocorrelation(x, max_lag=None):
    """Normalized autocorrelation C(τ) = <x(t)·x(t+τ)> / <x²>."""
    x = x - np.mean(x)
    n = len(x)
    if max_lag is None:
        max_lag = n // 4
    c0 = np.dot(x, x)
    acf = np.zeros(max_lag)
    for lag in range(max_lag):
        acf[lag] = np.dot(x[:n - lag], x[lag:]) / c0
    return acf


def find_embedding_delay(x):
    """Embedding delay τ = first zero of autocorrelation."""
    acf = autocorrelation(x, max_lag=len(x) // 4)
    for i in range(1, len(acf)):
        if acf[i] <= 0:
            return i
    for i in range(1, len(acf) - 1):
        if acf[i] < acf[i - 1] and acf[i] < acf[i + 1]:
            return i
    return len(acf) // 10


def false_nearest_neighbors(x, tau, max_dim=10, rtol=15.0):
    """FNN algorithm (Kennel et al., 1992) for embedding dimension m."""
    n = len(x)
    fnn_fractions = []

    for m in range(1, max_dim + 1):
        n_valid = n - m * tau
        if n_valid < 10:
            break

        X_m = np.array([x[i:i + m * tau:tau] for i in range(n_valid)])
        x_next = np.array([x[i + m * tau] for i in range(n_valid)])

        tree = KDTree(X_m)
        dists, indices = tree.query(X_m, k=2)
        nn_dists = dists[:, 1]
        nn_indices = indices[:, 1]

        n_false = 0
        n_total = 0
        sigma = np.std(x)

        for j in range(len(X_m)):
            if nn_dists[j] < 1e-10:
                continue
            nn_idx = nn_indices[j]
            if nn_idx >= len(x_next):
                continue
            added_dist = abs(x_next[j] - x_next[nn_idx])
            ratio = added_dist / nn_dists[j]
            n_total += 1
            if ratio > rtol or (nn_dists[j] / sigma) < 1e-3:
                n_false += 1

        fnn_frac = n_false / max(n_total, 1)
        fnn_fractions.append(fnn_frac)
        if fnn_frac < 0.01:
            break

    for m, frac in enumerate(fnn_fractions, start=1):
        if frac < 0.01:
            return m, fnn_fractions
    return len(fnn_fractions), fnn_fractions


# ══════════════════════════════════════════════════════════════════════════
# 3. DELAY EMBEDDING
# ══════════════════════════════════════════════════════════════════════════
def delay_embed(x, m, tau):
    """Delay vectors: X_i = [x(i), x(i+τ), ..., x(i+(m-1)τ)]"""
    n_embed = len(x) - (m - 1) * tau
    return np.array([x[i:i + m * tau:tau] for i in range(n_embed)])


# ══════════════════════════════════════════════════════════════════════════
# 4. LYAPUNOV EXPONENT ESTIMATION
# ══════════════════════════════════════════════════════════════════════════
def rosenstein_lyapunov(x, m, tau, dt=1.0, max_iter=None):
    """
    Rosenstein's algorithm for the largest Lyapunov exponent λ_1.

    ALGORITHM:
      1. Embed x into R^m via delay vectors X_i.
      2. For each X_i, find its nearest neighbor X_j (|i-j| > mean_period).
      3. Track divergence: d_i(k) = ||X_{i+k} - X_{j+k}||.
      4. Compute ⟨ln d(k)⟩ (the "Rosenstein divergence curve").
      5. λ_1 = slope of the linear region.
    """
    X = delay_embed(x, m, tau)
    N = len(X)
    if max_iter is None:
        max_iter = N // 5

    mean_period = find_embedding_delay(x)
    tree = KDTree(X)

    nn_indices = np.zeros(N, dtype=int)
    for i in range(N):
        k_query = min(50, N)
        dists, idxs = tree.query(X[i], k=k_query)
        found = False
        for j_idx in range(1, len(idxs)):
            j = idxs[j_idx]
            if abs(i - j) > mean_period:
                nn_indices[i] = j
                found = True
                break
        if not found:
            nn_indices[i] = idxs[1]

    divergence = np.zeros(max_iter)
    counts = np.zeros(max_iter)

    for i in range(N):
        j = nn_indices[i]
        for k in range(max_iter):
            if i + k >= N or j + k >= N:
                break
            dist = np.linalg.norm(X[i + k] - X[j + k])
            if dist > 0:
                divergence[k] += np.log(dist)
                counts[k] += 1

    valid = counts > 0
    avg_div = np.zeros(max_iter)
    avg_div[valid] = divergence[valid] / counts[valid]

    fit_end = max(10, max_iter // 5)
    t_axis = np.arange(max_iter) * dt
    valid_fit = valid[:fit_end]
    if np.sum(valid_fit) > 5:
        coeffs = np.polyfit(t_axis[:fit_end][valid_fit[:fit_end]],
                           avg_div[:fit_end][valid_fit[:fit_end]], 1)
        lambda_1 = coeffs[0]
    else:
        lambda_1 = 0.0

    return lambda_1, t_axis, avg_div, valid


def sano_sawada_spectrum(x, m, tau, k_neighbors=None, dt=1.0):
    """
    Sano-Sawada: full Lyapunov spectrum from data-driven local Jacobians.
    Fits local linear maps A_i to neighbor clusters, then applies QR
    renormalization along the trajectory (Benettin with data-derived A).
    """
    X = delay_embed(x, m, tau)
    N = len(X)

    if k_neighbors is None:
        k_neighbors = max(m + 5, int(np.sqrt(N)))
        k_neighbors = min(k_neighbors, N // 10)

    tree = KDTree(X)
    Q = np.eye(m)
    lyap_sum = np.zeros(m)
    n_maps = 0
    step = max(1, tau // 2)

    for i in range(0, N - step - 1, step):
        dists, idx = tree.query(X[i], k=k_neighbors + 1)
        neighbors = idx[1:]
        valid_nn = neighbors[(neighbors + step) < N]
        if len(valid_nn) < m + 1:
            continue

        X_local = X[valid_nn] - X[i]
        X_next = X[valid_nn + step] - X[i + step]

        try:
            A, _, _, _ = np.linalg.lstsq(X_local, X_next, rcond=None)
            A = A.T
        except np.linalg.LinAlgError:
            continue

        Q_new = A @ Q
        Q_new, R = np.linalg.qr(Q_new)
        signs = np.sign(np.diag(R))
        signs[signs == 0] = 1
        Q_new = Q_new * signs
        R = np.diag(signs) @ R

        diag_R = np.abs(np.diag(R))
        if np.all(diag_R > 1e-20):
            lyap_sum += np.log(diag_R)
            n_maps += 1
            Q = Q_new

    if n_maps > 0:
        lyap = lyap_sum / (n_maps * step * dt)
        lyap = np.sort(lyap)[::-1]
    else:
        lyap = np.zeros(m)

    return lyap


# ══════════════════════════════════════════════════════════════════════════
# 5. KAPLAN-YORKE DIMENSION
# ══════════════════════════════════════════════════════════════════════════
def kaplan_yorke_dimension(lyap_sorted):
    """D_KY = j + S_j / |λ_{j+1}|."""
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
# 6. PLOTTING
# ══════════════════════════════════════════════════════════════════════════
def plot_results(data, data_norm, tau, m, d_ky, lyap, lambda_1,
                 t_div, avg_div, valid_div):
    """
    2×3 figure:
      Row 1: (0) Time series  (1) Raw delay plot x(t) vs x(t+τ)
             (2) Reconstructed attractor (Takens, m-dim → 2D projection)
      Row 2: (3) Lyapunov spectrum  (4) Rosenstein divergence  (5) FNN curve
    """
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle(
        f"{DATA_NAME}  —  SST Anomaly Dynamics\n"
        f"$D_{{KY}} = {d_ky:.2f}$,  $m = {m}$,  "
        f"$\\tau = {tau}$ months,  $\\lambda_1 = {lyap[0]:.4f}$",
        fontsize=14, fontweight='bold'
    )

    # ═══ (0,0): Time series ═══
    ax_ts = fig.add_subplot(gs[0, 0])
    t_months = np.arange(len(data))
    ax_ts.plot(t_months / 12.0, data, color='tab:blue', linewidth=0.8)
    ax_ts.axhline(0.5, color='red', linewidth=0.8, linestyle='--',
                  alpha=0.5, label='El Niño threshold')
    ax_ts.axhline(-0.5, color='blue', linewidth=0.8, linestyle='--',
                  alpha=0.5, label='La Niña threshold')
    ax_ts.set_xlabel("Year (relative)", fontsize=11)
    ax_ts.set_ylabel("SST Anomaly (°C)", fontsize=11)
    ax_ts.set_title("Time Series", fontsize=12, fontweight='bold')
    ax_ts.legend(fontsize=8)
    ax_ts.grid(True, linestyle=':', alpha=0.5)

    # ═══ (0,1): Raw delay plot — x(t) vs x(t+τ) ═══
    # This is the simplest attractor visualization: a 2D delay embedding
    # directly from the raw data, NO reconstruction needed.
    ax_raw = fig.add_subplot(gs[0, 1])
    x_t = data_norm[:-tau]
    x_t_tau = data_norm[tau:]
    ax_raw.plot(x_t, x_t_tau, color='black', linewidth=0.3, alpha=0.6)
    ax_raw.set_xlabel(f"$x(t)$", fontsize=11)
    ax_raw.set_ylabel(f"$x(t + {tau})$", fontsize=11)
    ax_raw.set_title(
        f"Raw Delay Plot ($\\tau = {tau}$)",
        fontsize=12, fontweight='bold'
    )
    ax_raw.grid(True, linestyle=':', alpha=0.5)

    # ═══ (0,2): Reconstructed attractor (higher-dim projection) ═══
    ax_att = fig.add_subplot(gs[0, 2])
    X_embed = delay_embed(data_norm, m, tau)
    if m >= 3:
        # Use dimensions 0 and 2 for a different view
        ax_att.plot(X_embed[:, 0], X_embed[:, 2], color='darkgreen',
                   linewidth=0.3, alpha=0.6)
        ax_att.set_ylabel(f"$x(t + {2*tau})$", fontsize=11)
    else:
        ax_att.plot(X_embed[:, 0], X_embed[:, 1], color='darkgreen',
                   linewidth=0.3, alpha=0.6)
        ax_att.set_ylabel(f"$x(t + {tau})$", fontsize=11)
    ax_att.set_xlabel(f"$x(t)$", fontsize=11)
    ax_att.set_title(
        f"Takens Embedding (m={m}),  $D_{{KY}}={d_ky:.2f}$",
        fontsize=12, fontweight='bold'
    )
    ax_att.grid(True, linestyle=':', alpha=0.5)

    # ═══ (1,0): Lyapunov spectrum ═══
    ax_ly = fig.add_subplot(gs[1, 0])
    indices = np.arange(len(lyap))
    colors = ['tab:red' if l > 0 else ('gold' if abs(l) < 0.005
               else 'tab:blue') for l in lyap]
    ax_ly.bar(indices, lyap, color=colors, edgecolor='black',
              linewidth=0.5, width=0.8)
    ax_ly.axhline(0, color='black', linewidth=1)
    ax_ly.set_xlabel("Exponent index $i$", fontsize=11)
    ax_ly.set_ylabel(r"$\lambda_i$", fontsize=11)
    ax_ly.set_title("Lyapunov Spectrum (Sano-Sawada)", fontsize=12,
                    fontweight='bold')
    ax_ly.grid(True, linestyle=':', alpha=0.5, axis='y')

    # ═══ (1,1): Rosenstein divergence ═══
    ax_ros = fig.add_subplot(gs[1, 1])
    n_plot = min(len(t_div), len(t_div) // 2 + 50)
    ax_ros.plot(t_div[:n_plot], avg_div[:n_plot], color='tab:green',
               linewidth=1.5, label=r'$\langle \ln d(k) \rangle$')
    fit_end = max(10, n_plot // 5)
    ax_ros.plot(t_div[:fit_end],
               lambda_1 * t_div[:fit_end] + avg_div[0],
               '--', color='tab:red', linewidth=2,
               label=f"Fit: $\\lambda_1 = {lambda_1:.4f}$")
    ax_ros.set_xlabel("Time lag $k$ (months)", fontsize=11)
    ax_ros.set_ylabel(r"$\langle \ln\, d(k) \rangle$", fontsize=11)
    ax_ros.set_title("Rosenstein Divergence", fontsize=12, fontweight='bold')
    ax_ros.legend(fontsize=10)
    ax_ros.grid(True, linestyle=':', alpha=0.5)

    # ═══ (1,2): Autocorrelation (diagnostic) ═══
    ax_acf = fig.add_subplot(gs[1, 2])
    acf = autocorrelation(data_norm, max_lag=min(200, len(data) // 3))
    ax_acf.plot(np.arange(len(acf)) / 12.0, acf, color='tab:purple',
               linewidth=1)
    ax_acf.axhline(0, color='black', linewidth=0.5)
    ax_acf.axvline(tau / 12.0, color='red', linewidth=1.5, linestyle='--',
                   label=f'$\\tau = {tau}$ months')
    ax_acf.set_xlabel("Lag (years)", fontsize=11)
    ax_acf.set_ylabel("Autocorrelation", fontsize=11)
    ax_acf.set_title("Autocorrelation Function", fontsize=12, fontweight='bold')
    ax_acf.legend(fontsize=10)
    ax_acf.grid(True, linestyle=':', alpha=0.5)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fname = "../../figures/enso_nino34/enso_nino34_dky_analysis.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"  Saved: {fname}")


# ══════════════════════════════════════════════════════════════════════════
# 7. MAIN
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print(f"  {DATA_NAME} — D_KY from Experimental SST Anomaly Data")
    print("=" * 60)

    # ── Load ──
    data = np.loadtxt(DATA_FILE)
    print(f"  Loaded {len(data)} monthly values (~{len(data)/12:.0f} years)")

    # ── Normalize ──
    data_norm = (data - np.mean(data)) / np.std(data)

    # ── Embedding delay ──
    tau = find_embedding_delay(data_norm)
    print(f"  Embedding delay τ = {tau} months")

    # ── Embedding dimension ──
    print(f"  Running FNN analysis...", end="", flush=True)
    m_opt, fnn_fracs = false_nearest_neighbors(data_norm, tau, max_dim=12)
    m_opt = max(m_opt, 3)
    print(f" m = {m_opt}")
    print(f"  FNN fractions: {[f'{f:.3f}' for f in fnn_fracs]}")

    # ── Largest Lyapunov exponent ──
    print(f"  Rosenstein algorithm...", end="", flush=True)
    lambda_1, t_div, avg_div, valid_div = rosenstein_lyapunov(
        data_norm, m_opt, tau
    )
    print(f" λ_1 = {lambda_1:.5f}")

    # ── Full spectrum ──
    print(f"  Sano-Sawada spectrum (m={m_opt})...", end="", flush=True)
    lyap = sano_sawada_spectrum(data_norm, m_opt, tau)
    print(f" done")
    print(f"  Spectrum: {[f'{l:.4f}' for l in lyap]}")

    # ── D_KY ──
    d_ky = kaplan_yorke_dimension(lyap)
    print(f"  D_KY = {d_ky:.3f}")

    # ── Export ──
    results = {
        "dataset": DATA_NAME,
        "n_points": len(data),
        "tau": int(tau),
        "m": int(m_opt),
        "lambda_1_rosenstein": round(float(lambda_1), 5),
        "full_spectrum": [round(float(l), 5) for l in lyap],
        "D_KY": round(float(d_ky), 3),
        "fnn_fractions": [round(float(f), 4) for f in fnn_fracs],
    }
    with open("../../data/enso_nino34/enso_nino34_dky_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Exported: ../data/enso_nino34/enso_nino34_dky_results.json")

    # ── Plot ──
    plot_results(data, data_norm, tau, m_opt, d_ky, lyap, lambda_1,
                 t_div, avg_div, valid_div)

    print("\nDone!")
    plt.show()
