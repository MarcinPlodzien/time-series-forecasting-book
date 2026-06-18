#!/usr/bin/env python3
"""
Generate two Lorenz '96 figures:
  fig_lorenz96.png         — Top: Hovmöller diagrams, Bottom: x_0(t) signals (4 columns, F=2,4,8,16)
  fig_lorenz96_spectra.png — Left: ACF, Right: PSD (all F overlaid)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import welch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(SCRIPT_DIR, '..', '..')

# ── Parameters ──
N_DIM = 40
F_values = [2, 4, 8, 16]
colors = ['#3498db', '#e67e22', '#e74c3c', '#9b59b6']
linestyles = ['-', '--', '-.', ':']

T_TRANS = 100.0
T_SIM = 50.0
DT = 0.01

def lorenz96(t, x, F):
    x = np.asarray(x)
    return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F

def integrate_l96(F, N=N_DIM):
    """Integrate Lorenz '96 and return x_0 signal after transient."""
    np.random.seed(42)
    x0 = F * np.ones(N) + 0.01 * np.random.randn(N)
    # Discard transient
    sol_trans = solve_ivp(lorenz96, [0, T_TRANS], x0, args=(F,),
                          method='RK45', rtol=1e-8, atol=1e-10)
    x0 = sol_trans.y[:, -1]
    # Main integration
    t_eval = np.arange(0, T_SIM, DT)
    sol = solve_ivp(lorenz96, [0, T_SIM], x0, args=(F,),
                    method='RK45', t_eval=t_eval, rtol=1e-8, atol=1e-10)
    return sol.t, sol.y  # sol.y is (N, n_times)

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
for F in F_values:
    print(f"  Integrating F = {F}...")
    t_arr, X = integrate_l96(F)
    data[F] = (t_arr, X)

fig_dir = os.path.join(BASE, 'figures/lorenz_96')
os.makedirs(fig_dir, exist_ok=True)

# ═══════════════════════════════════════════════════════
# FIGURE 1: Hovmöller (top) + x_0(t) signal (bottom), 2×4
# ═══════════════════════════════════════════════════════
fig1, axes1 = plt.subplots(2, 4, figsize=(20, 7))
fig1.subplots_adjust(hspace=0.40, wspace=0.30)

regime_labels = {2: "Fixed Point", 4: "Periodic", 8: "Chaotic", 16: "Turbulent"}

for col, (F, color) in enumerate(zip(F_values, colors)):
    t_arr, X = data[F]

    # ── Top: Hovmöller diagram ──
    ax = axes1[0, col]
    stride = max(1, len(t_arr) // 400)
    # Use diverging colormap for atmospheric-style visualization
    vmax = max(abs(X.min()), abs(X.max()))
    if F <= 3:
        vmax = max(vmax, F + 1)  # Ensure some contrast for fixed point
    im = ax.pcolormesh(t_arr[::stride], np.arange(N_DIM),
                       X[:, ::stride], cmap='RdBu_r',
                       shading='auto', vmin=-vmax, vmax=vmax)
    regime = regime_labels.get(F, "")
    ax.set_title(f'$F={F}$ ({regime})', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time (t.u.)', fontsize=9)
    if col == 0:
        ax.set_ylabel('Grid point $i$', fontsize=11)
    ax.tick_params(labelsize=8)
    label = chr(ord('a') + col)
    ax.text(0.04, 0.96, f'({label})', transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='top', color='white',
            bbox=dict(boxstyle='round,pad=0.15', fc='black', alpha=0.5))

    # ── Bottom: x_0(t) signal ──
    ax = axes1[1, col]
    # Show first 20 t.u. for clarity
    t_show = 20.0
    mask = t_arr <= t_show
    ax.plot(t_arr[mask], X[0, mask], color=color, linewidth=0.8)
    ax.set_xlabel('Time (t.u.)', fontsize=9)
    if col == 0:
        ax.set_ylabel('$x_0(t)$', fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.3)
    ax.tick_params(labelsize=8)
    label2 = chr(ord('e') + col)
    ax.text(0.04, 0.96, f'({label2})', transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='top')

fname1 = os.path.join(fig_dir, 'fig_lorenz96.png')
fig1.savefig(fname1, dpi=300, bbox_inches='tight')
print(f"  Saved: {fname1}")
plt.close(fig1)

# ═══════════════════════════════════════════════════════
# FIGURE 2: ACF (left) + PSD (right), 1×2
# ═══════════════════════════════════════════════════════
fig2, (ax_acf, ax_psd) = plt.subplots(1, 2, figsize=(12, 4.5))
fig2.subplots_adjust(wspace=0.30)

acf_max = 500
for F, color, ls in zip(F_values, colors, linestyles):
    t_arr, X = data[F]
    x0 = X[0, :]  # First grid point
    acf = autocorrelation(x0, max_lag=acf_max)
    lags_time = np.arange(len(acf)) * DT  # Convert to time units
    ax_acf.plot(lags_time, acf, color=color, linewidth=1.4,
                linestyle=ls, alpha=0.85, label=f'$F={F}$')
ax_acf.axhline(0, color='black', linewidth=0.5)
ax_acf.set_xlabel('Lag (t.u.)', fontsize=11)
ax_acf.set_ylabel('Autocorrelation', fontsize=11)
ax_acf.legend(fontsize=10, loc='upper right')
ax_acf.grid(True, linestyle=':', alpha=0.4)
ax_acf.text(0.02, 0.95, '(a)', transform=ax_acf.transAxes,
            fontsize=12, fontweight='bold', va='top')

for F, color, ls in zip(F_values, colors, linestyles):
    t_arr, X = data[F]
    x0 = X[0, :]
    N = len(x0)
    nperseg = min(512, N // 4)
    freqs, psd = welch(x0, fs=1.0 / DT, nperseg=nperseg)
    ax_psd.loglog(freqs[1:], psd[1:], color=color, linewidth=1.4,
                  linestyle=ls, alpha=0.85, label=f'$F={F}$')
ax_psd.set_xlabel('Frequency (1/t.u.)', fontsize=11)
ax_psd.set_ylabel('PSD', fontsize=11)
ax_psd.legend(fontsize=10, loc='lower left')
ax_psd.grid(True, linestyle=':', alpha=0.4)
ax_psd.text(0.02, 0.95, '(b)', transform=ax_psd.transAxes,
            fontsize=12, fontweight='bold', va='top')

fname2 = os.path.join(fig_dir, 'fig_lorenz96_spectra.png')
fig2.savefig(fname2, dpi=300, bbox_inches='tight')
print(f"  Saved: {fname2}")
plt.close(fig2)
