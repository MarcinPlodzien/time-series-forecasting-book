"""
experiment 00: build and verify the benchmark datasets
======================================================

The appendix benchmarks four signals chosen to span the predictability spectrum,
from a fully known synthetic system to an essentially unpredictable market. All
four are shipped in ``benchmarks/data/`` so the companion repository is
self-contained: a reader can clone it and reproduce every number with no network
access and no external downloads.

    1. Mackey-Glass (synthetic) -- ``mackeyglass_tau17.txt``. The one dataset we
       generate, by integrating its delay differential equation. Because the
       generator is deterministic, this script reproduces the file bit-for-bit,
       and the true delay, dimension and Lyapunov exponent are known so the Part I
       diagnostics can be checked against ground truth. It is also the canonical
       reservoir-computing benchmark, which anchors our numbers to the literature.
    2. Santa Fe laser (empirical) -- ``santafe_laser_full.txt``. The full 10093
       point competition record (set A), raw integers; the same signal whose
       standardised first 4000 samples appear in the Chapter 3 case study.
    3. ENSO Nino 3.4 (empirical) -- ``enso_nino34.txt``. The monthly SST-anomaly
       index used in the Chapter 3 case study (NOAA Nino 3.4).
    4. S&P 500 (empirical) -- ``sp500_daily_close.txt``. Daily closing level of
       the index, 1927-2026, as used in the Chapter 3 case study. The loader
       derives the forecasting object (log-returns or squared returns) from it.

Running this script regenerates Mackey-Glass and prints the size, range and
sha256 of all four files so ``data/README.md`` can record what is shipped.
The three empirical files are not re-downloaded here (they are versioned with the
repo to keep it reproducible and consistent with Chapter 3); the helper functions
that fetched them from their public sources are kept at the bottom for provenance.

Run from the benchmarks/ directory:
    python experiments/00_make_datasets.py
"""

from __future__ import annotations

import hashlib
import os

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

SHIPPED = [
    "mackeyglass_tau17.txt",
    "santafe_laser_full.txt",
    "enso_nino34.txt",
    "sp500_daily_close.txt",
]


# ---------------------------------------------------------------------------
# Mackey-Glass: integrate the delay differential equation.
# ---------------------------------------------------------------------------
def make_mackey_glass(
    n_samples: int = 11000,
    tau_d: int = 17,
    beta: float = 0.2,
    gamma: float = 0.1,
    n_exp: int = 10,
    h: float = 0.1,
    sample_every: int = 10,
    transient: int = 3000,
    x0: float = 1.2,
) -> np.ndarray:
    """Integrate Mackey-Glass and return a scalar series sampled at unit time.

        dx/dt = beta * x(t - tau_d) / (1 + x(t - tau_d)^n) - gamma * x(t).

    Parameters reproduce the canonical chaotic regime used throughout the
    reservoir-computing literature: beta=0.2, gamma=0.1, n=10, tau_d=17 (the
    onset-of-chaos value, Kaplan-Yorke dimension ~2.1). We integrate with
    classical RK4 at a fine step h=0.1 and then keep every tenth point, so the
    returned series is sampled at unit time (Delta t = 1), matching the convention
    in which Mackey-Glass benchmark numbers are usually quoted.

    The delayed term sits 17 time units (170 fine steps) in the past, so it is
    always already computed; the half-step value an RK4 stage needs is obtained by
    linear interpolation between the two bracketing past samples. A long transient
    is discarded so the trajectory has settled onto the attractor.
    """
    lag = int(round(tau_d / h))  # delay in fine steps
    total_fine = transient + n_samples * sample_every + lag + 10
    x = np.empty(total_fine + 1, dtype=float)
    x[: lag + 1] = x0  # constant history on [-tau_d, 0], the usual initial state

    def f(xt: float, x_delay: float) -> float:
        return beta * x_delay / (1.0 + x_delay ** n_exp) - gamma * xt

    for i in range(lag, total_fine):
        xd0 = x[i - lag]                                # x(t - tau_d)
        xd_half = 0.5 * (x[i - lag] + x[i - lag + 1])  # x(t + h/2 - tau_d)
        xd1 = x[i - lag + 1]                            # x(t + h - tau_d)
        k1 = f(x[i], xd0)
        k2 = f(x[i] + 0.5 * h * k1, xd_half)
        k3 = f(x[i] + 0.5 * h * k2, xd_half)
        k4 = f(x[i] + h * k3, xd1)
        x[i + 1] = x[i] + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    settled = x[transient:]
    return settled[::sample_every][:n_samples]


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------
def _report(path: str) -> None:
    series = np.loadtxt(path)
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    print(f"  {os.path.basename(path)}")
    print(f"    points : {len(series)}")
    print(f"    min/max: {np.nanmin(series):.4f} / {np.nanmax(series):.4f}")
    print(f"    mean   : {np.nanmean(series):.4f}")
    print(f"    sha256 : {digest}")


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Generating Mackey-Glass (deterministic; reproduces the shipped file)")
    mg = make_mackey_glass()
    mg_path = os.path.join(DATA_DIR, "mackeyglass_tau17.txt")
    np.savetxt(mg_path, mg, fmt="%.10f")

    print("\nDataset files in data/ (record these in data/README.md):")
    missing = []
    for fname in SHIPPED:
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            _report(path)
        else:
            missing.append(fname)
    if missing:
        print("\n  MISSING (expected to be shipped with the repo): "
              + ", ".join(missing))
    else:
        print("\nAll four datasets present. The repository is self-contained.")


# ---------------------------------------------------------------------------
# Provenance: how the three empirical files were originally obtained. These are
# kept for the record and are not run by main(); the files they produced are
# versioned with the repository so it stays reproducible offline and consistent
# with the Chapter 3 case studies.
#
#   ENSO Nino 3.4 monthly anomalies:
#     https://psl.noaa.gov/data/correlation/nina34.anom.data   (NOAA PSL)
#   Santa Fe laser (competition set A), full 10093-point record:
#     CHARC mirror of the original competition file 'laser.txt'.
#   S&P 500 daily close, 1927-2026:
#     daily closing level of the S&P 500 index, as used in the Chapter 3 study.
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    main()
