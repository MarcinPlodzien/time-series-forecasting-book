#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════╗
║   ATLANTIC MULTIDECADAL OSCILLATION (AMO) — SST Anomaly Index         ║
║   Time Series Characterization & D_KY Estimation                      ║
╚══════════════════════════════════════════════════════════════════════════╝

DATASET
───────
The Atlantic Multidecadal Oscillation (AMO) is a mode of variability in
North Atlantic sea surface temperatures (SST) with a period of roughly
60–80 years.  The AMO index is computed as the area-weighted average SST
anomaly over the North Atlantic (0°–70°N), after removal of the global
warming trend.

Monthly values are available from 1856 to present (~2000 points),
maintained by NOAA's Physical Sciences Laboratory.

PHYSICS PERSPECTIVE
───────────────────
The AMO represents a fundamentally different dynamical regime from the
ENSO oscillation already discussed:

  ENSO:  2–7 year period,  fast coupled ocean-atmosphere instability,
         predictability horizon ~ 6–12 months for individual events.

  AMO:   60–80 year period, driven by variations in the Atlantic
         Meridional Overturning Circulation (AMOC) — the "conveyor belt"
         of warm surface water flowing northward and cold deep water
         returning south.

The AMO's long period means that even with ~170 years of data, we observe
at most 2–3 complete cycles.  This poses a fundamental challenge: can our
diagnostic toolkit reliably characterize dynamics when the observation
window barely exceeds the fundamental period?  The answer reveals the
practical limits of Takens embedding and dimension estimation for long-
period oscillations.

The AMO has enormous practical importance:
  - Positive (warm) AMO phases are associated with increased Atlantic
    hurricane activity, Sahel drought, and warming of the Arctic.
  - The AMO phase shift of the mid-1990s contributed to the recent
    increase in major hurricanes.
  - Understanding AMO predictability is critical for decadal climate
    prediction.

SOURCE
──────
NOAA Physical Sciences Laboratory:
  https://psl.noaa.gov/data/timeseries/AMO/
Enfield et al. (2001), "The Atlantic Multidecadal Oscillation and its
  relation to rainfall and river flows in the continental U.S."

ANALYSIS PIPELINE
─────────────────
Same Takens embedding pipeline:
  1. Download AMO index from NOAA
  2. Compute ACF, PSD
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
DATA_DIR   = os.path.join(SCRIPT_DIR, "..", "..", "data", "amo")
FIG_DIR    = os.path.join(SCRIPT_DIR, "..", "..", "figures", "amo")
DATA_NAME  = "Atlantic Multidecadal Oscillation (AMO)"

AMO_URL = "https://psl.noaa.gov/data/correlation/amon.us.long.data"


# ══════════════════════════════════════════════════════════════════════════
# 2. DATA DOWNLOAD & PARSING
# ══════════════════════════════════════════════════════════════════════════
def download_amo():
    """Download AMO monthly index from NOAA PSL."""
    os.makedirs(DATA_DIR, exist_ok=True)
    processed = os.path.join(DATA_DIR, "amo_monthly.txt")
    
    if os.path.exists(processed):
        print(f"  AMO data already processed: {processed}")
        return np.loadtxt(processed)
    
    raw_file = os.path.join(DATA_DIR, "amo_raw.txt")
    print(f"  Downloading AMO data from NOAA...")
    
    try:
        urllib.request.urlretrieve(AMO_URL, raw_file)
    except Exception as e:
        print(f"  Direct download failed ({e}), trying alternative URL...")
        alt_url = "https://psl.noaa.gov/data/correlation/amon.us.data"
        urllib.request.urlretrieve(alt_url, raw_file)
    
    # Parse NOAA format: "year val1 val2 ... val12"
    # Missing values are typically -99.990 or -99.99
    values = []
    years_list = []
    
    with open(raw_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                year = int(float(parts[0]))
                if year < 1850 or year > 2030:
                    continue
                for i, val_str in enumerate(parts[1:13]):
                    val = float(val_str)
                    if val > -90:  # valid
                        values.append(val)
                        years_list.append(year + (i + 0.5) / 12.0)
            except (ValueError, IndexError):
                continue
    
    data = np.array(values)
    years = np.array(years_list)
    print(f"  Loaded {len(data)} monthly AMO values ({years[0]:.0f}--{years[-1]:.0f})")
    
    np.savetxt(processed, data)
    np.savetxt(os.path.join(DATA_DIR, "amo_years.txt"), years)
    
    return data


# ══════════════════════════════════════════════════════════════════════════
# 3-6. ANALYSIS FUNCTIONS (same pipeline)
# ══════════════════════════════════════════════════════════════════════════
def autocorrelation(x, max_lag=None):
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


def delay_embed(x, m, tau):
    n_embed = len(x) - (m - 1) * tau
    return np.array([x[i:i + m * tau:tau] for i in range(n_embed)])


def rosenstein_lyapunov(x, m, tau, dt=1.0, max_iter=None):
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


def kaplan_yorke_dimension(lyap_sorted):
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
def plot_results(years, data_raw, data_norm, tau, m, d_ky, lyap,
                 lambda_1, t_div, avg_div, valid_div, fnn_fracs):
    T_pred = 1.0 / lambda_1 if lambda_1 > 0.001 else float('inf')

    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle(
        f"{DATA_NAME}  —  Predictability Characterization\n"
        f"$D_{{KY}} = {d_ky:.2f}$,  $m = {m}$,  "
        f"$\\tau = {tau}$ months,  $\\lambda_1 = {lyap[0]:.5f}$/month",
        fontsize=14, fontweight='bold'
    )

    # ═══ (0,0): AMO time series ═══
    ax_ts = fig.add_subplot(gs[0, 0])
    ax_ts.plot(years, data_raw, color='tab:blue', linewidth=0.3, alpha=0.5)
    # 10-year running mean
    kernel = np.ones(120) / 120.0  # 120 months = 10 years
    if len(data_raw) > 120:
        smooth = np.convolve(data_raw, kernel, mode='valid')
        y_smooth = years[59:59+len(smooth)]
        ax_ts.plot(y_smooth, smooth, color='darkred', linewidth=2, label='10-yr mean')
    ax_ts.axhline(0, color='black', linewidth=0.5)
    ax_ts.fill_between(years, data_raw, 0, where=data_raw > 0,
                       alpha=0.15, color='red', interpolate=True)
    ax_ts.fill_between(years, data_raw, 0, where=data_raw < 0,
                       alpha=0.15, color='blue', interpolate=True)
    ax_ts.set_xlabel("Year", fontsize=11)
    ax_ts.set_ylabel("AMO Index (°C)", fontsize=11)
    ax_ts.set_title("Atlantic Multidecadal Oscillation", fontsize=12, fontweight='bold')
    ax_ts.legend(fontsize=9)
    ax_ts.grid(True, linestyle=':', alpha=0.5)

    # ═══ (0,1): Delay embedding ═══
    ax_emb = fig.add_subplot(gs[0, 1])
    X_embed = delay_embed(data_norm, m, tau)
    if m >= 3:
        ax_emb.scatter(X_embed[::2, 0], X_embed[::2, 2], s=0.5, alpha=0.3,
                      color='darkgreen')
        ax_emb.set_ylabel(f"$x(t + {2*tau})$", fontsize=11)
    else:
        ax_emb.scatter(X_embed[::2, 0], X_embed[::2, 1], s=0.5, alpha=0.3,
                      color='darkgreen')
        ax_emb.set_ylabel(f"$x(t + {tau})$", fontsize=11)
    ax_emb.set_xlabel(f"$x(t)$", fontsize=11)
    ax_emb.set_title(
        f"Takens Embedding (m={m}),  $D_{{KY}}={d_ky:.2f}$",
        fontsize=12, fontweight='bold')
    ax_emb.grid(True, linestyle=':', alpha=0.5)

    # ═══ (0,2): ACF ═══
    ax_acf = fig.add_subplot(gs[0, 2])
    acf = autocorrelation(data_norm, max_lag=min(500, len(data_norm) // 4))
    lags_months = np.arange(len(acf))
    ax_acf.plot(lags_months / 12.0, acf, color='tab:purple', linewidth=1)
    ax_acf.axhline(0, color='black', linewidth=0.5)
    ax_acf.axvline(tau / 12.0, color='red', linewidth=1.5, linestyle='--',
                   label=f'$\\tau = {tau}$ mo')
    ax_acf.set_xlabel("Lag (years)", fontsize=11)
    ax_acf.set_ylabel("Autocorrelation", fontsize=11)
    ax_acf.set_title("ACF — Multidecadal Oscillation", fontsize=12, fontweight='bold')
    ax_acf.legend(fontsize=9)
    ax_acf.grid(True, linestyle=':', alpha=0.5)

    # ═══ (1,0): PSD ═══
    ax_psd = fig.add_subplot(gs[1, 0])
    freqs, psd = welch(data_norm, fs=12.0, nperseg=min(512, len(data_norm) // 4))
    ax_psd.loglog(freqs[1:], psd[1:], color='tab:orange', linewidth=1.5)
    # Mark ~70-year period
    ax_psd.axvline(1.0/70.0, color='red', linewidth=1, linestyle='--',
                   label='$f = 1/70$ yr$^{-1}$')
    ax_psd.set_xlabel("Frequency (cycles/year)", fontsize=11)
    ax_psd.set_ylabel("PSD", fontsize=11)
    ax_psd.set_title("Power Spectral Density", fontsize=12, fontweight='bold')
    ax_psd.legend(fontsize=9)
    ax_psd.grid(True, linestyle=':', alpha=0.5)

    # ═══ (1,1): Rosenstein divergence ═══
    ax_ros = fig.add_subplot(gs[1, 1])
    n_plot = min(len(t_div), len(t_div) // 2 + 50)
    ax_ros.plot(t_div[:n_plot] / 12.0, avg_div[:n_plot], color='tab:green',
               linewidth=1.5, label=r'$\langle \ln d(k) \rangle$')
    fit_end = max(10, n_plot // 5)
    ax_ros.plot(t_div[:fit_end] / 12.0,
               lambda_1 * t_div[:fit_end] + avg_div[0],
               '--', color='tab:red', linewidth=2,
               label=f"Fit: $\\lambda_1 = {lambda_1:.5f}$/mo")
    ax_ros.set_xlabel("Time lag (years)", fontsize=11)
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

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    os.makedirs(FIG_DIR, exist_ok=True)
    fname = os.path.join(FIG_DIR, "amo_dky_analysis.png")
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"  Saved: {fname}")


# ══════════════════════════════════════════════════════════════════════════
# 8. MAIN
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print(f"  {DATA_NAME} — Predictability Characterization")
    print("=" * 60)

    # ── Download & load ──
    data_raw = download_amo()
    years_file = os.path.join(DATA_DIR, "amo_years.txt")
    if os.path.exists(years_file):
        years = np.loadtxt(years_file)
    else:
        years = np.arange(len(data_raw)) / 12.0 + 1856.0
    
    print(f"  {len(data_raw)} monthly values ({years[0]:.0f}--{years[-1]:.0f})")
    print(f"  Mean: {np.mean(data_raw):.3f}°C, Std: {np.std(data_raw):.3f}°C")

    # ── Normalize ──
    data_norm = (data_raw - np.mean(data_raw)) / np.std(data_raw)

    # ── Embedding delay ──
    tau = find_embedding_delay(data_norm)
    print(f"  Embedding delay τ = {tau} months")

    # ── Embedding dimension ──
    print(f"  Running FNN analysis...", end="", flush=True)
    m_opt, fnn_fracs = false_nearest_neighbors(data_norm, tau, max_dim=10)
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

    # ── Predictability horizon ──
    if lambda_1 > 0.001:
        T_pred = 1.0 / lambda_1
        print(f"  T_pred ≈ {T_pred:.1f} months ({T_pred/12:.1f} years)")
    else:
        print(f"  T_pred: λ₁ ≈ 0, quasi-periodic oscillation")

    # ── Export ──
    results = {
        "dataset": DATA_NAME,
        "n_points": len(data_raw),
        "year_range": f"{years[0]:.0f}-{years[-1]:.0f}",
        "tau": int(tau),
        "m": int(m_opt),
        "lambda_1_rosenstein": round(float(lambda_1), 6),
        "full_spectrum": [round(float(l), 5) for l in lyap],
        "D_KY": round(float(d_ky), 3),
        "fnn_fractions": [round(float(f), 4) for f in fnn_fracs],
    }
    results_file = os.path.join(DATA_DIR, "amo_dky_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Exported: {results_file}")

    # ── Plot ──
    plot_results(years, data_raw, data_norm, tau, m_opt, d_ky, lyap,
                 lambda_1, t_div, avg_div, valid_div, fnn_fracs)

    print("\nDone!")
    plt.show()
