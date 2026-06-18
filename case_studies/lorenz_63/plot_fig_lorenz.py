#!/usr/bin/env python3
"""
Generate two Lorenz multi-ρ figures:
  fig_lorenz.png         — Top: x(t) signals, Bottom: (x,z) attractors (4 columns)
  fig_lorenz_spectra.png — Left: ACF, Right: PSD (all ρ overlaid)
ρ = 20 (stable fixed point), 60 (chaos), 100 (periodic window), 200 (vigorous chaos)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import welch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(SCRIPT_DIR, '..', '..')

sigma, beta_l = 10.0, 8.0/3.0
rho_values = [20, 60, 100, 200]
colors = ['#3498db', '#e67e22', '#e74c3c', '#9b59b6']
linestyles = ['-', '--', '-.', ':']
# For ρ=20 (sub-critical), keep transient to show the damped spiral
rho_critical = 24.74

def lorenz(t, state, rho):
    x, y, z = state
    return [sigma*(y - x), x*(rho - z) - y, x*y - beta_l*z]

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
for rho in rho_values:
    print(f"  Integrating ρ = {rho}...")
    # Use a perturbation from the fixed point so transient is visible
    x0 = [10.0, 10.0, rho - 1]  # near one fixed point but perturbed
    sol = solve_ivp(lorenz, [0, 200], x0,
                    args=(rho,), max_step=0.01, dense_output=True)
    if rho < rho_critical:
        # Non-chaotic: keep transient to see damped oscillations
        t_arr = np.linspace(0, 50, 5000)
    else:
        # Chaotic / periodic: discard transient
        t_arr = np.linspace(50, 200, 15000)
    xyz = sol.sol(t_arr)
    data[rho] = {'t': t_arr, 'x': xyz[0], 'y': xyz[1], 'z': xyz[2]}

fig_dir = os.path.join(BASE, 'figures/lorenz_63')
os.makedirs(fig_dir, exist_ok=True)

# ═══════════════════════════════════════════════════════
# FIGURE 1: Signals (top) + Attractors (bottom), 2×4
# ═══════════════════════════════════════════════════════
fig1, axes1 = plt.subplots(2, 4, figsize=(16, 7))
fig1.subplots_adjust(hspace=0.35, wspace=0.35)

for col, (rho, color) in enumerate(zip(rho_values, colors)):
    d = data[rho]
    t, x, z = d['t'], d['x'], d['z']
    N = len(x)

    # ── Top: Signal ──
    ax = axes1[0, col]
    if rho < rho_critical:
        # Show the full transient for non-chaotic
        ax.plot(t, x, color=color, linewidth=1.2)
    else:
        # Zoom ~2000 points (~20 t.u.) for chaotic/periodic
        zoom_len = min(2000, N)
        start = N // 2 - zoom_len // 2
        end = start + zoom_len
        ax.plot(t[start:end], x[start:end], color=color, linewidth=1.2)
    ax.set_xlabel('Time (t.u.)', fontsize=9)
    if col == 0:
        ax.set_ylabel('$x(t)$', fontsize=11)
    ax.set_title(f'$\\rho={rho}$', fontsize=12, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.3)
    ax.tick_params(labelsize=8)
    label = chr(ord('a') + col)
    ax.text(0.04, 0.96, f'({label})', transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='top')

    # ── Bottom: Attractor (x-z projection) ──
    ax = axes1[1, col]
    ax.plot(x, z, linewidth=0.4, alpha=0.7, color=color)
    ax.set_xlabel('$x$', fontsize=9)
    if col == 0:
        ax.set_ylabel('$z$', fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.3)
    ax.tick_params(labelsize=8)
    label2 = chr(ord('e') + col)
    ax.text(0.04, 0.96, f'({label2})', transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='top')

fname1 = os.path.join(fig_dir, 'fig_lorenz.png')
fig1.savefig(fname1, dpi=300, bbox_inches='tight')
print(f"  Saved: {fname1}")
plt.close(fig1)

# ═══════════════════════════════════════════════════════
# FIGURE 2: ACF (left) + PSD (right), 1×2
# Only for chaotic/periodic ρ values (skip ρ < critical)
# ═══════════════════════════════════════════════════════
fig2, (ax_acf, ax_psd) = plt.subplots(1, 2, figsize=(12, 4.5))
fig2.subplots_adjust(wspace=0.30)

acf_max = 2000
for rho, color, ls in zip(rho_values, colors, linestyles):
    x = data[rho]['x']
    acf = autocorrelation(x, max_lag=min(acf_max, len(x)//4))
    ax_acf.plot(np.arange(len(acf)), acf, color=color, linewidth=1.4,
                linestyle=ls, alpha=0.85, label=f'$\\rho={rho}$')
ax_acf.axhline(0, color='black', linewidth=0.5)
ax_acf.set_xlabel('Lag (samples)', fontsize=11)
ax_acf.set_ylabel('Autocorrelation', fontsize=11)
ax_acf.legend(fontsize=9, ncol=2, loc='upper right')
ax_acf.grid(True, linestyle=':', alpha=0.4)
ax_acf.text(0.02, 0.95, '(a)', transform=ax_acf.transAxes,
            fontsize=12, fontweight='bold', va='top')

for rho, color, ls in zip(rho_values, colors, linestyles):
    t_arr = data[rho]['t']
    dt_s = t_arr[1] - t_arr[0]
    x = data[rho]['x']
    N = len(x)
    nperseg = min(1024, N // 4)
    freqs, psd = welch(x, fs=1.0/dt_s, nperseg=nperseg)
    ax_psd.loglog(freqs[1:], psd[1:], color=color, linewidth=1.4,
                  linestyle=ls, alpha=0.85, label=f'$\\rho={rho}$')
ax_psd.set_xlabel('Frequency (1/t.u.)', fontsize=11)
ax_psd.set_ylabel('PSD', fontsize=11)
ax_psd.legend(fontsize=9, ncol=2, loc='lower left')
ax_psd.grid(True, linestyle=':', alpha=0.4)
ax_psd.text(0.02, 0.95, '(b)', transform=ax_psd.transAxes,
            fontsize=12, fontweight='bold', va='top')

fname2 = os.path.join(fig_dir, 'fig_lorenz_spectra.png')
fig2.savefig(fname2, dpi=300, bbox_inches='tight')
print(f"  Saved: {fname2}")
plt.close(fig2)
