#!/usr/bin/env python3
"""
Shared 2×2 figure generator for all datasets.
Format: (0,0) Full signal, (0,1) 10% zoom, (1,0) ACF, (1,1) PSD.
No title — details go in LaTeX caption. Named fig_{dataset}.png.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
import os


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


def plot_signal_2x2(time, signal, dataset_key, fig_dir,
                    time_label="Time", signal_label="Value",
                    fs=1.0, freq_label="Frequency",
                    acf_max_lag=None, zoom_frac=0.10):
    """
    Create the standard 2×2 publication figure.

    Parameters
    ----------
    time : array — time axis for the signal
    signal : array — the signal values
    dataset_key : str — used for filename: fig_{dataset_key}.png
    fig_dir : str — directory to save the figure
    time_label : str — x-axis label for time panels
    signal_label : str — y-axis label for signal panels
    fs : float — sampling frequency for PSD (cycles per unit)
    freq_label : str — x-axis label for PSD
    acf_max_lag : int — max lag for ACF (default: N//4)
    zoom_frac : float — fraction of signal to zoom into (default: 0.10)
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.subplots_adjust(hspace=0.32, wspace=0.28)

    N = len(signal)

    # ── (0,0): Full signal ──
    ax = axes[0, 0]
    ax.plot(time, signal, color='#2c3e50', linewidth=0.4, alpha=0.8)
    ax.set_xlabel(time_label, fontsize=11)
    ax.set_ylabel(signal_label, fontsize=11)
    ax.text(0.02, 0.95, '(a)', transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top')
    ax.grid(True, linestyle=':', alpha=0.4)

    # ── (0,1): 10% zoom ──
    ax = axes[0, 1]
    zoom_len = max(int(N * zoom_frac), 50)
    # Pick a representative region (middle of the series)
    start = max(0, N // 2 - zoom_len // 2)
    end = min(N, start + zoom_len)
    ax.plot(time[start:end], signal[start:end],
            color='#2c3e50', linewidth=0.8)
    ax.set_xlabel(time_label, fontsize=11)
    ax.set_ylabel(signal_label, fontsize=11)
    ax.text(0.02, 0.95, '(b)', transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top')
    ax.grid(True, linestyle=':', alpha=0.4)

    # ── (1,0): ACF ──
    ax = axes[1, 0]
    if acf_max_lag is None:
        acf_max_lag = min(N // 4, 300)
    acf = autocorrelation(signal, max_lag=acf_max_lag)
    lags = np.arange(len(acf))
    ax.plot(lags, acf, color='#8e44ad', linewidth=1.0)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel("Lag", fontsize=11)
    ax.set_ylabel("Autocorrelation", fontsize=11)
    ax.text(0.02, 0.95, '(c)', transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top')
    ax.grid(True, linestyle=':', alpha=0.4)

    # ── (1,1): PSD ──
    ax = axes[1, 1]
    nperseg = min(512, N // 4)
    if nperseg < 16:
        nperseg = N
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    ax.loglog(freqs[1:], psd[1:], color='#e67e22', linewidth=1.0)
    ax.set_xlabel(freq_label, fontsize=11)
    ax.set_ylabel("PSD", fontsize=11)
    ax.text(0.02, 0.95, '(d)', transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top')
    ax.grid(True, linestyle=':', alpha=0.4)

    os.makedirs(fig_dir, exist_ok=True)
    fname = os.path.join(fig_dir, f"fig_{dataset_key}.png")
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"  Saved: {fname}")
    plt.close(fig)
    return fname


def plot_ode_2x2(time, signal, dataset_key, fig_dir,
                 attractor_coords=None, attractor_labels=None,
                 time_label="Time", signal_label="Value",
                 fs=1.0, freq_label="Frequency",
                 acf_max_lag=None, zoom_frac=0.10):
    """
    Create 2×2 figure for ODE systems: attractor + zoom + ACF + PSD.

    Parameters
    ----------
    attractor_coords : tuple of (x, y) or (x, y, z) for attractor plot
    attractor_labels : tuple of axis labels for attractor
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.subplots_adjust(hspace=0.32, wspace=0.28)

    N = len(signal)

    # ── (0,0): Attractor ──
    if attractor_coords is not None:
        ax = axes[0, 0]
        if len(attractor_coords) == 3:
            ax.remove()
            ax = fig.add_subplot(2, 2, 1, projection='3d')
            x, y, z = attractor_coords
            ax.plot(x, y, z, linewidth=0.15, alpha=0.6, color='#2c3e50')
            if attractor_labels:
                ax.set_xlabel(attractor_labels[0], fontsize=10)
                ax.set_ylabel(attractor_labels[1], fontsize=10)
                ax.set_zlabel(attractor_labels[2], fontsize=10)
            ax.tick_params(labelsize=8)
            ax.text2D(0.02, 0.95, '(a)', transform=ax.transAxes,
                      fontsize=12, fontweight='bold', va='top')
        else:
            x, y = attractor_coords[:2]
            ax.plot(x, y, linewidth=0.2, alpha=0.5, color='#2c3e50')
            if attractor_labels:
                ax.set_xlabel(attractor_labels[0], fontsize=11)
                ax.set_ylabel(attractor_labels[1], fontsize=11)
            ax.text(0.02, 0.95, '(a)', transform=ax.transAxes,
                    fontsize=12, fontweight='bold', va='top')
            ax.grid(True, linestyle=':', alpha=0.4)
    else:
        ax = axes[0, 0]
        ax.plot(time, signal, color='#2c3e50', linewidth=0.4, alpha=0.8)
        ax.set_xlabel(time_label, fontsize=11)
        ax.set_ylabel(signal_label, fontsize=11)
        ax.text(0.02, 0.95, '(a)', transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='top')
        ax.grid(True, linestyle=':', alpha=0.4)

    # ── (0,1): Zoom to 10% of signal ──
    ax = axes[0, 1]
    zoom_len = max(int(N * zoom_frac), 50)
    start = max(0, N // 2 - zoom_len // 2)
    end = min(N, start + zoom_len)
    ax.plot(time[start:end], signal[start:end],
            color='#2c3e50', linewidth=0.8)
    ax.set_xlabel(time_label, fontsize=11)
    ax.set_ylabel(signal_label, fontsize=11)
    ax.text(0.02, 0.95, '(b)', transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top')
    ax.grid(True, linestyle=':', alpha=0.4)

    # ── (1,0): ACF ──
    ax = axes[1, 0]
    if acf_max_lag is None:
        acf_max_lag = min(N // 4, 300)
    acf = autocorrelation(signal, max_lag=acf_max_lag)
    lags = np.arange(len(acf))
    ax.plot(lags, acf, color='#8e44ad', linewidth=1.0)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel("Lag", fontsize=11)
    ax.set_ylabel("Autocorrelation", fontsize=11)
    ax.text(0.02, 0.95, '(c)', transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top')
    ax.grid(True, linestyle=':', alpha=0.4)

    # ── (1,1): PSD ──
    ax = axes[1, 1]
    nperseg = min(512, N // 4)
    if nperseg < 16:
        nperseg = N
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    ax.loglog(freqs[1:], psd[1:], color='#e67e22', linewidth=1.0)
    ax.set_xlabel(freq_label, fontsize=11)
    ax.set_ylabel("PSD", fontsize=11)
    ax.text(0.02, 0.95, '(d)', transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top')
    ax.grid(True, linestyle=':', alpha=0.4)

    os.makedirs(fig_dir, exist_ok=True)
    fname = os.path.join(fig_dir, f"fig_{dataset_key}.png")
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"  Saved: {fname}")
    plt.close(fig)
    return fname
