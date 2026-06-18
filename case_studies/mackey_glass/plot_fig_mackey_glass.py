#!/usr/bin/env python3
"""
Generate two Mackey-Glass multi-τ figures:
  fig_mackey_glass.png         — Top: signals, Bottom: attractors (3 columns, τ=15,25,35)
  fig_mackey_glass_spectra.png — Left: ACF, Right: PSD (all τ overlaid)
τ values match mg_educational_attractors.png
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(SCRIPT_DIR, '..', '..')

# ── Parameters ──
beta_mg, gamma_mg, n_hill = 0.2, 0.1, 10
dt = 0.1
T_total = 15000.0
T_transient = 5000.0
tau_values = [15, 25, 35]
colors = ['#3498db', '#e74c3c', '#9b59b6']
linestyles = ['-', '--', ':']

def integrate_mg(tau_mg):
    N_steps = int(T_total / dt)
    N_hist = int(tau_mg / dt)
    x = np.zeros(N_steps)
    np.random.seed(42)
    x[:N_hist] = 1.5 + 0.01 * np.random.randn(N_hist)
    for i in range(N_hist, N_steps - 1):
        x_delayed = x[i - N_hist]
        dxdt = beta_mg * x_delayed / (1 + x_delayed**n_hill) - gamma_mg * x[i]
        x[i + 1] = x[i] + dt * dxdt
    i_start = int(T_transient / dt)
    return x[i_start::10]

def autocorrelation(x, max_lag=None):
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

# ── Integrate all ──
data = {}
for tau_mg in tau_values:
    print(f"  Integrating τ_MG = {tau_mg}...")
    data[tau_mg] = integrate_mg(tau_mg)

fig_dir = os.path.join(BASE, 'figures/mackiney_glass')
os.makedirs(fig_dir, exist_ok=True)

# ═══════════════════════════════════════════════════════
# FIGURE 1: Signals (top) + Attractors (bottom), 2×3
# ═══════════════════════════════════════════════════════
fig1, axes1 = plt.subplots(2, 3, figsize=(14, 7))
fig1.subplots_adjust(hspace=0.35, wspace=0.35)

for col, (tau_mg, color) in enumerate(zip(tau_values, colors)):
    x = data[tau_mg]
    N = len(x)
    t = np.arange(N)

    # ── Top: Signal (zoom to ~500 points in middle) ──
    ax = axes1[0, col]
    zoom_len = 500
    start = N // 2 - zoom_len // 2
    end = start + zoom_len
    ax.plot(t[start:end], x[start:end], color=color, linewidth=1.2)
    ax.set_xlabel('Time (t.u.)', fontsize=9)
    if col == 0:
        ax.set_ylabel('$x(t)$', fontsize=11)
    ax.set_title(f'$\\tau_{{MG}}={tau_mg}$', fontsize=12, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.3)
    ax.tick_params(labelsize=8)
    label = chr(ord('a') + col)
    ax.text(0.04, 0.96, f'({label})', transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='top')

    # ── Bottom: Attractor ──
    ax = axes1[1, col]
    tau_embed = tau_mg
    x_now = x[:-tau_embed]
    x_del = x[tau_embed:]
    ax.plot(x_now, x_del, linewidth=0.4, alpha=0.7, color=color)
    ax.set_xlabel('$x(t)$', fontsize=9)
    if col == 0:
        ax.set_ylabel(f'$x(t-\\tau_{{MG}})$', fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.3)
    ax.tick_params(labelsize=8)
    label2 = chr(ord('d') + col)
    ax.text(0.04, 0.96, f'({label2})', transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='top')

fname1 = os.path.join(fig_dir, 'fig_mackey_glass.png')
fig1.savefig(fname1, dpi=300, bbox_inches='tight')
print(f"  Saved: {fname1}")
plt.close(fig1)

# ═══════════════════════════════════════════════════════
# FIGURE 2: ACF (left) + PSD (right), 1×2
# ═══════════════════════════════════════════════════════
fig2, (ax_acf, ax_psd) = plt.subplots(1, 2, figsize=(12, 4.5))
fig2.subplots_adjust(wspace=0.30)

acf_max = 500
for tau_mg, color, ls in zip(tau_values, colors, linestyles):
    x = data[tau_mg]
    acf = autocorrelation(x, max_lag=acf_max)
    ax_acf.plot(np.arange(len(acf)), acf, color=color, linewidth=1.4,
                linestyle=ls, alpha=0.85, label=f'$\\tau_{{MG}}={tau_mg}$')
ax_acf.axhline(0, color='black', linewidth=0.5)
ax_acf.set_xlabel('Lag', fontsize=11)
ax_acf.set_ylabel('Autocorrelation', fontsize=11)
ax_acf.legend(fontsize=10, loc='upper right')
ax_acf.grid(True, linestyle=':', alpha=0.4)
ax_acf.text(0.02, 0.95, '(a)', transform=ax_acf.transAxes,
            fontsize=12, fontweight='bold', va='top')

for tau_mg, color, ls in zip(tau_values, colors, linestyles):
    x = data[tau_mg]
    N = len(x)
    nperseg = min(512, N // 4)
    freqs, psd = welch(x, fs=1.0, nperseg=nperseg)
    ax_psd.loglog(freqs[1:], psd[1:], color=color, linewidth=1.4,
                  linestyle=ls, alpha=0.85, label=f'$\\tau_{{MG}}={tau_mg}$')
ax_psd.set_xlabel('Frequency (1/t.u.)', fontsize=11)
ax_psd.set_ylabel('PSD', fontsize=11)
ax_psd.legend(fontsize=10, loc='lower left')
ax_psd.grid(True, linestyle=':', alpha=0.4)
ax_psd.text(0.02, 0.95, '(b)', transform=ax_psd.transAxes,
            fontsize=12, fontweight='bold', va='top')

fname2 = os.path.join(fig_dir, 'fig_mackey_glass_spectra.png')
fig2.savefig(fname2, dpi=300, bbox_inches='tight')
print(f"  Saved: {fname2}")
plt.close(fig2)
