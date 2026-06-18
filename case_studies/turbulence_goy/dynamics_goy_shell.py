#!/usr/bin/env python3
"""
GOY (Gledzer–Ohkitani–Yamada) shell model of turbulence.

Uses scipy.integrate.solve_ivp with a stiff-aware solver (Radau)
to handle the extreme scale separation between forcing and dissipation.

Standard parameters following Pisarenko et al. (1993) and Biferale (2003).

References
----------
  Yamada & Ohkitani (1987), J. Phys. Soc. Jpn. 56, 4210.
  L'vov et al. (1998), Phys. Rev. E 58, 1811.
  Biferale (2003), Annu. Rev. Fluid Mech. 35, 441.
  Pisarenko et al. (1993), Phys. Fluids A 5, 2533.
"""

import os
import numpy as np
from scipy.integrate import solve_ivp

# ──────────────────────────────────────────────────────────────────────────
# Parameters
# ──────────────────────────────────────────────────────────────────────────
N_SHELLS = 22
LAMBDA = 2.0
K0 = 2**(-4)            # k_n = k_0 * 2^n, so k_4 = 1
NU = 1e-7
EPSILON = 0.5           # energy + helicity conservation

# Forcing on shell n=4 (k=1, the "integral scale")
F_SHELL = 4
F_AMP = (1.0 + 1.0j) * 5e-3


def goy_rhs_real(t, y):
    """
    RHS of GOY shell model, written as a 2*N real system for solve_ivp.
    y = [Re(u_0), Im(u_0), Re(u_1), Im(u_1), ..., Re(u_{N-1}), Im(u_{N-1})]
    """
    N = N_SHELLS
    # Reconstruct complex amplitudes
    u = y[0::2] + 1j * y[1::2]
    k = K0 * LAMBDA ** np.arange(N, dtype=float)
    
    dudt = np.zeros(N, dtype=complex)
    uc = np.conj(u)
    
    for n in range(N):
        # Nonlinear: triadic coupling
        nl = 0.0 + 0.0j
        if n + 2 < N:
            nl += k[n] * uc[n+1] * uc[n+2]
        if n >= 1 and n + 1 < N:
            nl -= EPSILON * k[n-1] * uc[n-1] * uc[n+1]
        if n >= 2:
            nl -= (1 - EPSILON) * k[n-2] * uc[n-2] * uc[n-1]
        
        # Full RHS: nonlinear + dissipation + forcing
        dudt[n] = 1j * nl - NU * k[n]**2 * u[n]
    
    # Forcing
    dudt[F_SHELL] += F_AMP
    
    # Convert to real
    dydt = np.empty(2*N)
    dydt[0::2] = dudt.real
    dydt[1::2] = dudt.imag
    return dydt


def integrate():
    """Integrate GOY model using scipy's stiff ODE solver."""
    N = N_SHELLS
    k = K0 * LAMBDA ** np.arange(N, dtype=float)
    
    # Dissipation scale estimate
    # For Kolmogorov: k_d = (eps / nu^3)^(1/4)
    # With f ~ 5e-3: eps ~ f^(3/2) ~ 3.5e-4
    eps_est = np.abs(F_AMP)**1.5
    k_d = (eps_est / NU**3)**0.25
    n_d = np.log2(k_d / K0)
    
    print(f"GOY shell model — scipy Radau integration")
    print(f"  Shells: {N}, lambda={LAMBDA}, k0={K0}")
    print(f"  k range: [{k[0]:.4f}, {k[-1]:.1e}]")
    print(f"  nu = {NU:.1e}")
    print(f"  Forced shell: n={F_SHELL}, k={k[F_SHELL]:.4f}")
    print(f"  Estimated dissipation: k_d ~ {k_d:.1f} (shell ~ {n_d:.1f})")
    print(f"  gamma_max = nu*k_max^2 = {NU * k[-1]**2:.1e}")
    print()
    
    # Initial condition: small perturbation near forcing shell
    rng = np.random.default_rng(seed=42)
    u0 = np.zeros(N, dtype=complex)
    for n in range(F_SHELL - 2, min(F_SHELL + 6, N)):
        if n >= 0:
            u0[n] = 1e-4 * k[n]**(-1.0/3.0) * (
                rng.standard_normal() + 1j * rng.standard_normal())
    
    y0 = np.empty(2*N)
    y0[0::2] = u0.real
    y0[1::2] = u0.imag
    
    # Integration in two phases:
    # Phase 1: Spin-up (let the cascade establish) — 0 to T_trans
    # Phase 2: Data collection — T_trans to T_final
    T_trans = 50.0
    T_final = 250.0
    dt_save = 1e-3
    
    print(f"  Phase 1: Spin-up [0, {T_trans}]...")
    sol1 = solve_ivp(goy_rhs_real, [0, T_trans], y0,
                     method='Radau', rtol=1e-8, atol=1e-10,
                     max_step=0.01)
    
    if not sol1.success:
        print(f"  WARNING: spin-up failed: {sol1.message}")
        return None, None, None
    
    print(f"    Steps: {sol1.t.size}, final t: {sol1.t[-1]:.2f}")
    u_end = sol1.y[0::2, -1] + 1j * sol1.y[1::2, -1]
    E_spinup = np.sum(np.abs(u_end)**2)
    print(f"    E = {E_spinup:.6e}, max|u| = {np.max(np.abs(u_end)):.4e}")
    
    # Phase 2: Data collection with dense output
    y1 = sol1.y[:, -1]
    t_eval = np.arange(T_trans, T_final, dt_save)
    
    print(f"\n  Phase 2: Data collection [{T_trans}, {T_final}]...")
    print(f"    Saving {len(t_eval)} points at dt = {dt_save}")
    
    sol2 = solve_ivp(goy_rhs_real, [T_trans, T_final], y1,
                     method='Radau', rtol=1e-8, atol=1e-10,
                     t_eval=t_eval, max_step=0.01)
    
    if not sol2.success:
        print(f"  WARNING: data collection failed: {sol2.message}")
        return None, None, None
    
    print(f"    Steps: {sol2.t.size}")
    
    # Extract complex amplitudes
    u_all = sol2.y[0::2, :].T + 1j * sol2.y[1::2, :].T  # (N_times, N_shells)
    t = sol2.t
    
    E_final = np.sum(np.abs(u_all[-1])**2)
    print(f"    E_final = {E_final:.6e}")
    print(f"    Record: {len(t)} points, t in [{t[0]:.2f}, {t[-1]:.2f}]")
    
    # Energy spectrum diagnostic
    E_shells = np.mean(np.abs(u_all)**2, axis=0)
    print(f"\n  Time-averaged energy spectrum:")
    for n in range(N):
        if E_shells[n] > 0:
            bar = '#' * max(1, int(np.log10(E_shells[n]) + 12))
        else:
            bar = '.'
        print(f"    n={n:2d}  k={k[n]:12.4f}  E={E_shells[n]:.3e}  {bar}")
    
    # ── Save ──
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base = os.path.join(script_dir, '..', '..')
    data_dir = os.path.join(base, 'data', 'turbulence_goy')
    os.makedirs(data_dir, exist_ok=True)

    out_path = os.path.join(data_dir, 'goy_shell_data.npz')
    np.savez_compressed(out_path, t=t, u=u_all, k=k)
    print(f"\n  Data saved to: {out_path}")
    print(f"  Shape: t={t.shape}, u={u_all.shape}")

    return t, u_all, k


if __name__ == '__main__':
    integrate()
