#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════╗
║   SANTA FE LASER DATA — Chaotic Intensity Pulsations                  ║
║   D_KY Estimation from Experimental Time Series                       ║
╚══════════════════════════════════════════════════════════════════════════╝

DATASET
───────
The SantaFe competition dataset A consists of 4000 samples of intensity
measurements from a far-infrared NH3 laser in a chaotic regime. This is
a classic benchmark in nonlinear time series analysis.

The laser dynamics arise from a set of Maxwell-Bloch equations governing
the interaction between the electromagnetic field and the molecular medium.
The chaotic pulsations reflect deterministic low-dimensional chaos with
D_KY typically estimated around 2-4 (depending on analysis parameters).

ANALYSIS PIPELINE
─────────────────
Since we only have a scalar time series (not the governing equations),
we must:
  1. Reconstruct the attractor via TAKENS DELAY EMBEDDING
  2. Estimate the LYAPUNOV SPECTRUM from the embedded dynamics
  3. Compute D_KY from the estimated spectrum

See the detailed mathematical derivations in the docstring comments of
estimate_D_KY_from_data.py for:
  - Takens' theorem (delay embedding theory)
  - Fraser-Swinney mutual information for embedding delay τ
  - False Nearest Neighbors (FNN) for embedding dimension m
  - Rosenstein's algorithm for λ_1
  - Sano-Sawada method for the full Lyapunov spectrum
  - Relationship between D_KY and Hausdorff dimension

RELATIONSHIP: D_KY AND HAUSDORFF DIMENSION
───────────────────────────────────────────
The hierarchy of fractal dimensions for a typical strange attractor is:

    D_H ≥ D_1 ≥ D_2 ≥ D_KY

  D_H  = Hausdorff dimension  (purely geometric: how the set fills space)
  D_1  = Information dimension (how probability distributes across scales)
  D_2  = Correlation dimension (Grassberger-Procaccia, from pairwise distances)
  D_KY = Kaplan-Yorke dimension (from Lyapunov spectrum)

The Kaplan-Yorke conjecture states D_KY = D_1 for typical attractors.
When this holds, D_KY also provides a lower bound for D_H:

  D_KY ≤ D_H

Intuitively: higher D_KY means the attractor "fills" more of phase space,
corresponding to more complex, higher-dimensional chaos.
"""

import numpy as np
import os
import json
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.spatial import KDTree
from scipy.signal import welch


# ══════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE  = os.path.join(SCRIPT_DIR, "../../data/santafe_laser", "santafe.txt")
DATA_NAME  = "SantaFe Laser"


# ══════════════════════════════════════════════════════════════════════════
# 2. EMBEDDING PARAMETER SELECTION
# ══════════════════════════════════════════════════════════════════════════
def autocorrelation(x, max_lag=None):
    """
    Normalized autocorrelation C(τ) = <x(t)·x(t+τ)> / <x²>.
    Used to find the embedding delay: τ = first zero crossing.
    """
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
    """
    FNN algorithm (Kennel et al., 1992) to find embedding dimension m.
    Returns optimal m and the fraction of false neighbors at each dimension.
    """
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
    """
    Construct delay vectors: X_i = [x(i), x(i+τ), ..., x(i+(m-1)τ)]
    By Takens' theorem, this reconstructs the attractor topology.
    """
    n_embed = len(x) - (m - 1) * tau
    return np.array([x[i:i + m * tau:tau] for i in range(n_embed)])


# ══════════════════════════════════════════════════════════════════════════
# 4. LYAPUNOV EXPONENT ESTIMATION
# ══════════════════════════════════════════════════════════════════════════
def rosenstein_lyapunov(x, m, tau, dt=1.0, max_iter=None):
    """
    Rosenstein's algorithm for the largest Lyapunov exponent λ_1.
    Tracks average divergence of nearest-neighbor pairs in embedding space.

    Returns: λ_1, time array, average divergence, validity mask.
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
    Sano-Sawada method: estimate FULL Lyapunov spectrum from data.
    Uses local linear maps (data-driven Jacobians) + QR renormalization.
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
    """D_KY = j + S_j / |λ_{j+1}|  where S_j = Σ_{i=1}^j λ_i ≥ 0."""
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
    2×3 comprehensive predictability characterization figure:
      Row 1: (0) Time series  (1) Raw delay plot  (2) Reconstructed attractor
      Row 2: (3) Lyapunov spectrum  (4) Rosenstein divergence  (5) ACF + PSD
    """
    # ── Predictability horizon ──
    T_pred = 1.0 / lambda_1 if lambda_1 > 0 else float('inf')

    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle(
        f"{DATA_NAME}  —  Predictability Characterization\n"
        f"$D_{{KY}} = {d_ky:.2f}$,  $m = {m}$,  "
        f"$\\tau = {tau}$,  $\\lambda_1 = {lyap[0]:.4f}$,  "
        f"$T_{{pred}} \\approx {T_pred:.0f}$ samples",
        fontsize=14, fontweight='bold'
    )

    # ═══ (0,0): Time series ═══
    ax_ts = fig.add_subplot(gs[0, 0])
    ax_ts.plot(data, color='tab:blue', linewidth=0.5)
    ax_ts.set_xlabel("Sample index", fontsize=11)
    ax_ts.set_ylabel("Intensity", fontsize=11)
    ax_ts.set_title("Time Series", fontsize=12, fontweight='bold')
    ax_ts.grid(True, linestyle=':', alpha=0.5)

    # ═══ (0,1): Raw delay plot — x(t) vs x(t+τ) ═══
    ax_raw = fig.add_subplot(gs[0, 1])
    x_t = data_norm[:-tau]
    x_t_tau = data_norm[tau:]
    ax_raw.plot(x_t, x_t_tau, color='black', linewidth=0.3, alpha=0.6)
    ax_raw.set_xlabel(f"$x(t)$", fontsize=11)
    ax_raw.set_ylabel(f"$x(t + {tau})$", fontsize=11)
    ax_raw.set_title(f"Raw Delay Plot ($\\tau = {tau}$)",
                     fontsize=12, fontweight='bold')
    ax_raw.grid(True, linestyle=':', alpha=0.5)

    # ═══ (0,2): Reconstructed attractor (higher-dim projection) ═══
    ax_att = fig.add_subplot(gs[0, 2])
    X_embed = delay_embed(data_norm, m, tau)
    if m >= 3:
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
        fontsize=12, fontweight='bold')
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
    ax_ros.set_xlabel("Time lag $k$", fontsize=11)
    ax_ros.set_ylabel(r"$\langle \ln\, d(k) \rangle$", fontsize=11)
    ax_ros.set_title("Rosenstein Divergence", fontsize=12, fontweight='bold')
    ax_ros.legend(fontsize=10)
    ax_ros.grid(True, linestyle=':', alpha=0.5)

    # ═══ (1,2): Autocorrelation + Power Spectrum (twin axes) ═══
    ax_acf = fig.add_subplot(gs[1, 2])
    acf = autocorrelation(data_norm, max_lag=min(500, len(data) // 4))
    ax_acf.plot(np.arange(len(acf)), acf, color='tab:purple', linewidth=1)
    ax_acf.axhline(0, color='black', linewidth=0.5)
    ax_acf.axvline(tau, color='red', linewidth=1.5, linestyle='--',
                   label=f'$\\tau = {tau}$')
    ax_acf.set_xlabel("Lag (samples)", fontsize=11)
    ax_acf.set_ylabel("Autocorrelation", fontsize=11)
    ax_acf.set_title("Autocorrelation Function", fontsize=12,
                     fontweight='bold')
    ax_acf.legend(fontsize=10)
    ax_acf.grid(True, linestyle=':', alpha=0.5)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fname = "../../figures/santafe_laser/santafe_laser_dky_analysis.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"  Saved: {fname}")

    # ═══ BONUS: Separate Power Spectrum figure ═══
    fig2, ax_psd = plt.subplots(figsize=(8, 4))
    freqs, psd = welch(data_norm, fs=1.0, nperseg=min(256, len(data)//4))
    ax_psd.semilogy(freqs, psd, color='tab:orange', linewidth=1)
    ax_psd.set_xlabel("Frequency (cycles/sample)", fontsize=11)
    ax_psd.set_ylabel("PSD", fontsize=11)
    ax_psd.set_title(f"{DATA_NAME} — Power Spectral Density",
                     fontsize=13, fontweight='bold')
    ax_psd.grid(True, linestyle=':', alpha=0.5)
    fig2.tight_layout()
    plt.savefig("../../figures/santafe_laser/santafe_laser_psd.png", dpi=300, bbox_inches='tight')
    print(f"  Saved: ../figures/santafe_laser/santafe_laser_psd.png")


# ══════════════════════════════════════════════════════════════════════════
# 7. MAIN
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print(f"  {DATA_NAME} — D_KY from Experimental Time Series")
    print("=" * 60)

    # ── Load data ──
    data = np.loadtxt(DATA_FILE)
    print(f"  Loaded {len(data)} data points")

    # ── Normalize ──
    data_norm = (data - np.mean(data)) / np.std(data)

    # ── Embedding delay ──
    tau = find_embedding_delay(data_norm)
    print(f"  Embedding delay τ = {tau}")

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
    with open("../../data/santafe_laser/santafe_laser_dky_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Exported: ../data/santafe_laser/santafe_laser_dky_results.json")

    # ── Plot ──
    plot_results(data, data_norm, tau, m_opt, d_ky, lyap, lambda_1,
                 t_div, avg_div, valid_div)

    print("\nDone!")
    plt.show()
