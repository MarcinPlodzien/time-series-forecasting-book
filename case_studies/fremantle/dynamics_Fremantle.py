#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════╗
║   FREMANTLE SEA LEVEL — Monthly Mean Sea Level (RLR)                  ║
║   Time Series Characterization & D_KY Estimation                      ║
╚══════════════════════════════════════════════════════════════════════════╝

DATASET
───────
Fremantle is a port city on the coast of Western Australia, near Perth.
Its tide gauge (PSMSL station 111) provides one of the longest continuous
sea-level records in the Southern Hemisphere, with monthly RLR (Revised
Local Reference) data from 1897 to present (~1500 monthly values).

The record shows:
  • A dominant annual cycle (~20 cm amplitude) driven by seasonal
    thermal expansion and monsoon-related wind patterns
  • An upward trend of ~1.5 mm/year reflecting global sea-level rise
  • Interannual variability linked to ENSO (El Niño suppresses sea
    level in the eastern Indian Ocean / western Pacific)

Source: PSMSL (Permanent Service for Mean Sea Level)
  https://psmsl.org/data/obtaining/stations/111.php
  Monthly RLR data: https://psmsl.org/data/obtaining/rlr.monthly.data/111.rlrdata

ANALYSIS PIPELINE
─────────────────
Same Takens embedding pipeline as the other scripts:
  1. Download data from PSMSL
  2. Compute ACF, PSD, FNN
  3. Delay embedding (Takens)
  4. Rosenstein → λ₁
  5. Sano-Sawada → full spectrum → D_KY
  6. Comprehensive 2×3 figure
"""

import numpy as np
import os
import json
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.spatial import KDTree
from scipy.signal import welch
import urllib.request

# ══════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "..", "..", "data", "fremantle")
FIG_DIR    = os.path.join(SCRIPT_DIR, "..", "..", "figures", "fremantle")
DATA_NAME  = "Fremantle Sea Level (Monthly RLR)"

PSMSL_URL = "https://psmsl.org/data/obtaining/rlr.monthly.data/111.rlrdata"


# ══════════════════════════════════════════════════════════════════════════
# 2. DATA DOWNLOAD & PARSING
# ══════════════════════════════════════════════════════════════════════════
def download_fremantle():
    """Download Fremantle monthly RLR data from PSMSL."""
    raw_file = os.path.join(DATA_DIR, "fremantle_monthly_rlr.txt")

    if not os.path.exists(raw_file):
        print(f"  Downloading Fremantle sea-level data from PSMSL...")
        os.makedirs(DATA_DIR, exist_ok=True)
        urllib.request.urlretrieve(PSMSL_URL, raw_file)
        print(f"  Saved: {raw_file}")
    else:
        print(f"  Fremantle data already downloaded: {raw_file}")

    return raw_file


def parse_fremantle(raw_file):
    """Parse the PSMSL RLR monthly format.

    Format: semicolon-delimited
      year_fraction; level_mm; interpolation_flag; attention_flag
    Missing values are coded as -99999.
    Level is in mm relative to the Revised Local Reference.
    """
    years = []
    levels = []

    with open(raw_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = [p.strip() for p in line.split(';')]
            if len(parts) < 2:
                continue
            try:
                year = float(parts[0])
                level = int(parts[1])
                if level == -99999:
                    continue  # skip missing data
                years.append(year)
                levels.append(level)
            except (ValueError, IndexError):
                continue

    return np.array(years), np.array(levels, dtype=float)


# ══════════════════════════════════════════════════════════════════════════
# 3. EMBEDDING PARAMETER SELECTION
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


def mutual_information(x, max_lag=50, n_bins=32):
    """Average mutual information I(τ) using histogram binning."""
    n = len(x)
    mi = np.zeros(max_lag)

    edges = np.linspace(np.min(x), np.max(x) + 1e-10, n_bins + 1)

    for lag in range(max_lag):
        x1 = x[:n - lag]
        x2 = x[lag:n]

        h_joint, _, _ = np.histogram2d(x1, x2, bins=edges)
        h1, _ = np.histogram(x1, bins=edges)
        h2, _ = np.histogram(x2, bins=edges)

        p_joint = h_joint / np.sum(h_joint)
        p1 = h1 / np.sum(h1)
        p2 = h2 / np.sum(h2)

        for i in range(n_bins):
            for j in range(n_bins):
                if p_joint[i, j] > 0 and p1[i] > 0 and p2[j] > 0:
                    mi[lag] += p_joint[i, j] * np.log(p_joint[i, j] / (p1[i] * p2[j]))

    return mi


def find_embedding_delay(x):
    """Embedding delay τ = first minimum of mutual information."""
    mi = mutual_information(x, max_lag=min(60, len(x) // 10))

    for i in range(1, len(mi) - 1):
        if mi[i] < mi[i-1] and mi[i] < mi[i+1]:
            return i

    acf = autocorrelation(x, max_lag=len(x) // 4)
    for i in range(1, len(acf)):
        if acf[i] <= 0:
            return i

    return len(mi) // 3


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
# 4. DELAY EMBEDDING
# ══════════════════════════════════════════════════════════════════════════
def delay_embed(x, m, tau):
    """Construct delay vectors: X_i = [x(i), x(i+τ), ..., x(i+(m-1)τ)]"""
    n_embed = len(x) - (m - 1) * tau
    return np.array([x[i:i + m * tau:tau] for i in range(n_embed)])


# ══════════════════════════════════════════════════════════════════════════
# 5. LYAPUNOV EXPONENT ESTIMATION
# ══════════════════════════════════════════════════════════════════════════
def rosenstein_lyapunov(x, m, tau, dt=1.0, max_iter=None):
    """Rosenstein's algorithm for the largest Lyapunov exponent λ_1."""
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
    """Sano-Sawada method: full Lyapunov spectrum from data-driven Jacobians."""
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
# 6. KAPLAN-YORKE DIMENSION
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
# 7. PLOTTING
# ══════════════════════════════════════════════════════════════════════════
def plot_results(years, levels, data_norm, tau, m, d_ky, lyap,
                 lambda_1, t_div, avg_div, valid_div, fnn_fracs):
    """
    2×3 comprehensive predictability characterization figure:
      Row 1: (0) Sea level series  (1) Delay embedding  (2) ACF
      Row 2: (3) PSD  (4) Rosenstein divergence  (5) FNN
    """
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle(
        f"{DATA_NAME}  —  Predictability Characterization\n"
        f"$D_{{KY}} = {d_ky:.2f}$,  $m = {m}$,  "
        f"$\\tau = {tau}$ months,  $\\lambda_1 = {lyap[0]:.5f}$/month",
        fontsize=14, fontweight='bold'
    )

    # ═══ (0,0): Sea level time series ═══
    ax_ts = fig.add_subplot(gs[0, 0])
    ax_ts.plot(years, levels, color='teal', linewidth=0.4, alpha=0.5)
    # 12-month running mean
    if len(levels) > 12:
        kernel = np.ones(12) / 12.0
        smooth = np.convolve(levels, kernel, mode='valid')
        years_smooth = years[5:5+len(smooth)]
        ax_ts.plot(years_smooth, smooth, color='darkblue', linewidth=1.2,
                   label='12-mo mean')
    ax_ts.set_xlabel("Year", fontsize=11)
    ax_ts.set_ylabel("Sea level (mm, RLR)", fontsize=11)
    ax_ts.set_title("Fremantle Sea Level", fontsize=12, fontweight='bold')
    ax_ts.legend(fontsize=9)
    ax_ts.grid(True, linestyle=':', alpha=0.5)

    # ═══ (0,1): Delay embedding ═══
    ax_emb = fig.add_subplot(gs[0, 1])
    X_embed = delay_embed(data_norm, m, tau)
    if m >= 3:
        ax_emb.scatter(X_embed[::2, 0], X_embed[::2, 2], s=0.5, alpha=0.3,
                      c=np.arange(0, len(X_embed), 2) % (12*tau), cmap='hsv')
        ax_emb.set_ylabel(f"$h(t + {2*tau})$", fontsize=11)
    else:
        ax_emb.scatter(X_embed[::2, 0], X_embed[::2, 1], s=0.5, alpha=0.3,
                      color='teal')
        ax_emb.set_ylabel(f"$h(t + {tau})$", fontsize=11)
    ax_emb.set_xlabel(f"$h(t)$", fontsize=11)
    ax_emb.set_title(
        f"Takens Embedding (m={m}),  $D_{{KY}}={d_ky:.2f}$",
        fontsize=12, fontweight='bold')
    ax_emb.grid(True, linestyle=':', alpha=0.5)

    # ═══ (0,2): Autocorrelation ═══
    ax_acf = fig.add_subplot(gs[0, 2])
    acf = autocorrelation(data_norm, max_lag=min(120, len(data_norm) // 4))
    ax_acf.plot(np.arange(len(acf)), acf, color='tab:purple', linewidth=1)
    ax_acf.axhline(0, color='black', linewidth=0.5)
    ax_acf.axvline(tau, color='red', linewidth=1.5, linestyle='--',
                   label=f'$\\tau = {tau}$ mo')
    ax_acf.axvline(12, color='orange', linewidth=1, linestyle=':',
                   label='12-mo period')
    ax_acf.set_xlabel("Lag (months)", fontsize=11)
    ax_acf.set_ylabel("Autocorrelation", fontsize=11)
    ax_acf.set_title("Autocorrelation Function", fontsize=12, fontweight='bold')
    ax_acf.legend(fontsize=9)
    ax_acf.grid(True, linestyle=':', alpha=0.5)

    # ═══ (1,0): Power Spectrum ═══
    ax_psd = fig.add_subplot(gs[1, 0])
    freqs, psd = welch(data_norm, fs=12.0, nperseg=min(512, len(data_norm) // 4))
    ax_psd.loglog(freqs, psd, color='tab:orange', linewidth=1)
    ax_psd.axvline(1.0, color='red', linewidth=1, linestyle='--',
                   label='$f = 1$ yr$^{-1}$ (annual)')
    ax_psd.set_xlabel("Frequency (cycles/year)", fontsize=11)
    ax_psd.set_ylabel("PSD", fontsize=11)
    ax_psd.set_title("Power Spectral Density", fontsize=12, fontweight='bold')
    ax_psd.legend(fontsize=9)
    ax_psd.grid(True, linestyle=':', alpha=0.5)

    # ═══ (1,1): Rosenstein divergence ═══
    ax_ros = fig.add_subplot(gs[1, 1])
    n_plot = min(len(t_div), len(t_div) // 2 + 50)
    ax_ros.plot(t_div[:n_plot], avg_div[:n_plot], color='tab:green',
               linewidth=1.5, label=r'$\langle \ln d(k) \rangle$')
    fit_end = max(10, n_plot // 5)
    ax_ros.plot(t_div[:fit_end],
               lambda_1 * t_div[:fit_end] + avg_div[0],
               '--', color='tab:red', linewidth=2,
               label=f"Fit: $\\lambda_1 = {lambda_1:.5f}$/mo")
    ax_ros.set_xlabel("Time lag $k$ (months)", fontsize=11)
    ax_ros.set_ylabel(r"$\langle \ln\, d(k) \rangle$", fontsize=11)
    ax_ros.set_title("Rosenstein Divergence", fontsize=12, fontweight='bold')
    ax_ros.legend(fontsize=10)
    ax_ros.grid(True, linestyle=':', alpha=0.5)

    # ═══ (1,2): FNN ═══
    ax_fnn = fig.add_subplot(gs[1, 2])
    dims = np.arange(1, len(fnn_fracs) + 1)
    ax_fnn.plot(dims, fnn_fracs, 'o-', color='tab:blue', linewidth=2, markersize=6)
    ax_fnn.axhline(0.01, color='red', linewidth=1, linestyle='--', alpha=0.5,
                   label='1% threshold')
    ax_fnn.set_xlabel("Embedding dimension $m$", fontsize=11)
    ax_fnn.set_ylabel("FNN fraction", fontsize=11)
    ax_fnn.set_title("False Nearest Neighbors", fontsize=12, fontweight='bold')
    ax_fnn.legend(fontsize=10)
    ax_fnn.grid(True, linestyle=':', alpha=0.5)
    ax_fnn.set_ylim(-0.02, max(0.5, max(fnn_fracs) * 1.1))

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    os.makedirs(FIG_DIR, exist_ok=True)
    fname = os.path.join(FIG_DIR, "fremantle_dky_analysis.png")
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"  Saved: {fname}")


# ══════════════════════════════════════════════════════════════════════════
# 8. MAIN
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print(f"  {DATA_NAME} — Predictability Characterization")
    print("=" * 60)

    # ── Download & parse ──
    raw_file = download_fremantle()
    years, levels = parse_fremantle(raw_file)
    print(f"  Loaded {len(levels)} monthly values ({years[0]:.0f}--{years[-1]:.0f})")
    print(f"  Mean: {np.mean(levels):.0f} mm (RLR), Std: {np.std(levels):.0f} mm")

    # ── Handle gaps: the PSMSL data may have missing months ──
    # Check for large gaps and warn
    dt = np.diff(years)
    gaps = np.where(dt > 0.15)[0]  # > ~2 months gap
    if len(gaps) > 0:
        print(f"  WARNING: {len(gaps)} gaps > 2 months detected in record")
        # For records with many gaps, interpolate onto regular grid
        if len(gaps) > 10:
            print(f"  Interpolating to fill gaps...")
            year_reg = np.arange(years[0], years[-1], 1.0/12.0)
            levels_interp = np.interp(year_reg, years, levels)
            years = year_reg
            levels = levels_interp
            print(f"  After interpolation: {len(levels)} monthly values")

    # ── Save processed data ──
    os.makedirs(DATA_DIR, exist_ok=True)
    np.savetxt(os.path.join(DATA_DIR, "fremantle_monthly_level.txt"), levels)
    np.savetxt(os.path.join(DATA_DIR, "fremantle_monthly_years.txt"), years)

    # ── Work with anomalies (remove seasonal cycle + linear trend) ──
    # Remove monthly climatology
    month_indices = np.round((years % 1) * 12).astype(int) % 12
    climatology = np.zeros(12)
    for m_idx in range(12):
        mask = month_indices == m_idx
        if np.sum(mask) > 0:
            climatology[m_idx] = np.mean(levels[mask])
    anomalies = levels - climatology[month_indices]

    # Remove linear trend (sea-level rise)
    t_centered = years - np.mean(years)
    trend_coeffs = np.polyfit(t_centered, anomalies, 1)
    trend = np.polyval(trend_coeffs, t_centered)
    anomalies_detrended = anomalies - trend
    print(f"  Seasonal cycle amplitude: {climatology.max() - climatology.min():.0f} mm")
    print(f"  Linear trend: {trend_coeffs[0]*10:.1f} mm/decade")
    print(f"  Anomaly std (detrended): {np.std(anomalies_detrended):.1f} mm")

    # Normalize
    data_norm = (anomalies_detrended - np.mean(anomalies_detrended)) / np.std(anomalies_detrended)

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
    print(f" λ_1 = {lambda_1:.5f}/month")

    # ── Full spectrum ──
    print(f"  Sano-Sawada spectrum (m={m_opt})...", end="", flush=True)
    lyap = sano_sawada_spectrum(data_norm, m_opt, tau)
    print(f" done")
    print(f"  Spectrum: {[f'{l:.4f}' for l in lyap]}")

    # ── D_KY ──
    d_ky = kaplan_yorke_dimension(lyap)
    print(f"  D_KY = {d_ky:.3f}")

    # ── Lyapunov time ──
    if lambda_1 > 0.001:
        T_lambda = 1.0 / lambda_1
        print(f"  T_λ ≈ {T_lambda:.1f} months ({T_lambda/12:.1f} years)")
    else:
        print(f"  T_λ: λ₁ ≈ 0, quasi-periodic / decaying dynamics")

    # ── Export ──
    results = {
        "dataset": DATA_NAME,
        "n_points": len(levels),
        "year_range": f"{years[0]:.0f}-{years[-1]:.0f}",
        "tau": int(tau),
        "m": int(m_opt),
        "lambda_1_rosenstein": round(float(lambda_1), 6),
        "full_spectrum": [round(float(l), 5) for l in lyap],
        "D_KY": round(float(d_ky), 3),
        "fnn_fractions": [round(float(f), 4) for f in fnn_fracs],
        "seasonal_amplitude_mm": round(float(climatology.max() - climatology.min()), 1),
        "trend_mm_per_decade": round(float(trend_coeffs[0] * 10), 1),
    }
    results_file = os.path.join(DATA_DIR, "fremantle_dky_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Exported: {results_file}")

    # ── Plot ──
    plot_results(years, levels, data_norm, tau, m_opt, d_ky, lyap,
                 lambda_1, t_div, avg_div, valid_div, fnn_fracs)

    print("\nDone!")
    plt.show()
