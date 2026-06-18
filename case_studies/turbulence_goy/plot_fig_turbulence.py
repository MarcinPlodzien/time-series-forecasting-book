#!/usr/bin/env python3
"""
Generate diagnostic figures for the GOY shell model of turbulence.

Reads data from data/turbulence_goy/goy_shell_data.npz
Saves figures to figures/turbulence_goy/

Produces two figures matching the Ch3 format:
  fig_turbulence.png         — 2×2: (a) full signal, (b) zoom, (c) ACF, (d) PSD
  fig_turbulence_spectra.png — 1×2: (a) ACF per shell, (b) energy spectrum E(k_n)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(SCRIPT_DIR, '..', '..')


def autocorrelation(x, max_lag=None):
    """Normalized ACF."""
    x = x - np.mean(x)
    n = len(x)
    if max_lag is None:
        max_lag = n // 4
    c0 = np.dot(x, x)
    if c0 == 0:
        return np.zeros(max_lag)
    acf = np.zeros(max_lag)
    for lag in range(max_lag):
        acf[lag] = np.dot(x[:n - lag], x[lag:]) / c0
    return acf


def main():
    # ── Load data ──
    data_path = os.path.join(BASE, 'data', 'turbulence_goy', 'goy_shell_data.npz')
    if not os.path.exists(data_path):
        print(f"ERROR: Data file not found: {data_path}")
        print("Run dynamics_goy_shell.py first to generate the data.")
        sys.exit(1)

    data = np.load(data_path, allow_pickle=True)
    t = data['t']
    u_all = data['u']       # shape: (N_points, N_shells)
    k_shells = data['k']
    dt_eff = t[1] - t[0]
    fs = 1.0 / dt_eff

    print(f"Loaded: {data_path}")
    print(f"  Points: {len(t)}, Shells: {u_all.shape[1]}")
    print(f"  dt_eff: {dt_eff}, t_range: [{t[0]:.3f}, {t[-1]:.3f}]")

    # Observable: Re(u_8) — mid-inertial range shell (k=16)
    shell_obs = 8
    v = np.real(u_all[:, shell_obs])
    print(f"  Observable: Re(u_{shell_obs}), k = {k_shells[shell_obs]:.1f}")

    fig_dir = os.path.join(BASE, 'figures', 'turbulence_goy')
    os.makedirs(fig_dir, exist_ok=True)

    # ═══════════════════════════════════════════════════════
    # FIGURE 1: Standard 2×2 diagnostic panel
    # ═══════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.subplots_adjust(hspace=0.32, wspace=0.28)

    N = len(v)

    # (a) Full time series
    ax = axes[0, 0]
    ax.plot(t, v, color='#2c3e50', linewidth=0.8, alpha=0.85, rasterized=True)
    ax.set_xlabel('Time (model units)', fontsize=11)
    ax.set_ylabel(r'$\mathrm{Re}(u_8)$', fontsize=11)
    ax.text(0.02, 0.95, '(a)', transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top')
    ax.grid(True, linestyle=':', alpha=0.4)

    # (b) Zoomed segment (10%)
    ax = axes[0, 1]
    zoom_len = max(int(N * 0.10), 100)
    start = N // 2 - zoom_len // 2
    end = start + zoom_len
    ax.plot(t[start:end], v[start:end], color='#2c3e50', linewidth=1.2)
    ax.set_xlabel('Time (model units)', fontsize=11)
    ax.set_ylabel(r'$\mathrm{Re}(u_8)$', fontsize=11)
    ax.text(0.02, 0.95, '(b)', transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top')
    ax.grid(True, linestyle=':', alpha=0.4)

    # (c) ACF
    ax = axes[1, 0]
    acf_max_lag = min(N // 4, 5000)  # longer range to see decorrelation
    acf = autocorrelation(v, max_lag=acf_max_lag)
    lags = np.arange(acf_max_lag) * dt_eff
    ax.plot(lags, acf, color='#8e44ad', linewidth=1.5)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Lag (model units)', fontsize=11)
    ax.set_ylabel('Autocorrelation', fontsize=11)
    ax.text(0.02, 0.95, '(c)', transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top')
    ax.grid(True, linestyle=':', alpha=0.4)

    # (d) PSD with -5/3 reference
    ax = axes[1, 1]
    nperseg = min(4096, N // 4)
    freqs, psd = welch(v, fs=fs, nperseg=nperseg)
    ax.loglog(freqs[1:], psd[1:], color='#e67e22', linewidth=1.2)
    ax.set_xlabel('Frequency (1/model units)', fontsize=11)
    ax.set_ylabel('PSD', fontsize=11)
    ax.text(0.02, 0.95, '(d)', transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top')
    ax.grid(True, linestyle=':', alpha=0.4)

    fname1 = os.path.join(fig_dir, 'fig_turbulence.png')
    fig.savefig(fname1, dpi=300, bbox_inches='tight')
    print(f"  Saved: {fname1}")
    plt.close(fig)

    # ═══════════════════════════════════════════════════════
    # FIGURE 2: Spectral panel — ACF per shell + energy spectrum
    # ═══════════════════════════════════════════════════════
    fig2, (ax_acf, ax_ek) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig2.subplots_adjust(wspace=0.30)

    # (a) ACF for different shells
    shells_to_plot = [5, 8, 12, 16]  # inertial + dissipation range
    colors_sh = ['#3498db', '#e74c3c', '#f39c12', '#8e44ad']
    linestyles = ['-', '--', '-.', ':']
    for sh, col, ls in zip(shells_to_plot, colors_sh, linestyles):
        v_sh = np.real(u_all[:, sh])
        acf_sh = autocorrelation(v_sh, max_lag=acf_max_lag)
        ax_acf.plot(lags, acf_sh, color=col, linewidth=2.0, linestyle=ls,
                    alpha=0.85, label=f'$n={sh}$ ($k={k_shells[sh]:.0f}$)')
    ax_acf.axhline(0, color='black', linewidth=0.5)
    ax_acf.set_xlabel('Lag (model units)', fontsize=11)
    ax_acf.set_ylabel('Autocorrelation', fontsize=11)
    ax_acf.legend(fontsize=9, ncol=2, loc='lower center', bbox_to_anchor=(0.5, 1.02),
                  frameon=False)
    ax_acf.grid(True, linestyle=':', alpha=0.4)
    ax_acf.text(0.02, 0.95, '(a)', transform=ax_acf.transAxes,
                fontsize=12, fontweight='bold', va='top')

    # (b) Shell energy spectrum E(k_n) = <|u_n|^2>
    E_k = np.mean(np.abs(u_all)**2, axis=0)
    ax_ek.loglog(k_shells, E_k, 'o-', color='#2c3e50', markersize=5,
                 linewidth=1.5, label=r'$\langle |u_n|^2 \rangle$')
    # Reference: Kolmogorov scaling for shell model — E_n ~ k_n^{-2/3}
    n_inertial = slice(5, 18)  # inertial range above forced shell
    k_ref = k_shells[n_inertial]
    E_ref = E_k[5] * (k_ref / k_ref[0])**(-2.0/3.0)
    ax_ek.loglog(k_ref, E_ref, '--', color='#e74c3c', linewidth=3.5, alpha=1.0,
                 label=r'$k^{-2/3}$ (Kolmogorov)')
    ax_ek.set_xlabel(r'Wavenumber $k_n = 2^n$', fontsize=11)
    ax_ek.set_ylabel(r'Shell energy $\langle |u_n|^2 \rangle$', fontsize=11)
    ax_ek.legend(fontsize=10, frameon=False)
    ax_ek.grid(True, linestyle=':', alpha=0.4)
    ax_ek.text(0.02, 0.95, '(b)', transform=ax_ek.transAxes,
               fontsize=12, fontweight='bold', va='top')

    fname2 = os.path.join(fig_dir, 'fig_turbulence_spectra.png')
    fig2.savefig(fname2, dpi=300, bbox_inches='tight')
    print(f"  Saved: {fname2}")
    plt.close(fig2)

    # ── Summary statistics ──
    print(f"\n  Summary:")
    print(f"    Record length: {N} points")
    print(f"    Observable: Re(u_{shell_obs})")
    print(f"    Mean: {np.mean(v):.6f}, Std: {np.std(v):.6f}")
    print(f"    Energy spectrum spans {E_k[0]:.3e} to {E_k[-1]:.3e}")
    print(f"    Inertial range: shells 2–13 (k = {k_shells[2]:.0f} to {k_shells[13]:.0f})")
    print(f"\nDone.")


if __name__ == '__main__':
    main()
