"""
07_qrc_nv_scan.py
================

How the quantum reservoir scales with the number of virtual nodes nv (temporal
multiplexing). For one input step the single evolution of duration t is split
into nv equal sub-steps and the observables are read out after each, so the
readout feature count is nv x (one- and two-body Pauli set) without adding any
qubits. This sweep runs the QRC at several nv on each signal and records how the
forecasting capacity responds, so the chapter can show the scaling rather than
assert it.

Outputs are tagged by nv so nothing is overwritten and every run is kept:
    results/qrc_nv{nv}_{key}.csv        per (nv, signal): C_tot etc.
    results/qrc_nv{nv}_curves.npz       per nv: capacity curve C(tau) per signal
    results/qrc_nv_scan.csv             combined long-format table (all nv)
    results/qrc_nv_scan.png             C_tot vs nv per signal + laser C(tau) curves

    python experiments/07_qrc_nv_scan.py
    python experiments/07_qrc_nv_scan.py quick     # 1 seed, few origins

Reproducibility note: the nv=5 point here uses the same QRC settings as the
deployed full comparison benchmark (experiment 03), so the two QRC numbers cross-check.
"""

from __future__ import annotations

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from tsforecast import evaluation  # noqa: E402
from tsforecast.datasets import (  # noqa: E402
    load_enso, load_mackey_glass, load_santafe_laser, load_sp500,
)
from tsforecast.models import QuantumReservoir  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

QUICK = "quick" in sys.argv[1:]
# "oneseed" keeps the full origins/horizon but uses a single seed, for a fast
# trend curve (the headline nv=5 number with error bars comes from experiment 03).
ONESEED = "oneseed" in sys.argv[1:]
HORIZON = 60                          # same lead-time horizon as the full comparison
NV_VALUES = [1, 2, 3, 4, 5, 10]       # virtual-node counts to compare
SEEDS = [0] if (QUICK or ONESEED) else [0, 1, 2]
N_QUBITS = 6                          # deployed register size (kept fixed)

# All five full comparison signals, with the same rolling-origin settings as experiment
# 03 so the scan is directly consistent with the benchmark (the nv=5 points must
# reproduce the deployed QRC rows). The stochastic S&P signals are kept in to show
# that extra virtual nodes do not manufacture skill where there is no structure.
#   key, loader, n_origins, spacing
DATASETS = [
    ("mackeyglass",  lambda: load_mackey_glass(),        50, 50),
    ("laser",        lambda: load_santafe_laser(),       50, 40),
    ("enso",         lambda: load_enso(),                50,  5),
    ("sp500_logret", lambda: load_sp500("logreturns"),  100, 20),
    ("sp500_sqret",  lambda: load_sp500("sqreturns"),   100, 20),
]
if QUICK:
    DATASETS = [(k, ld, 10, sp) for (k, ld, _n, sp) in DATASETS]

DISPLAY = {
    "mackeyglass": "Mackey-Glass", "laser": "Santa Fe laser", "enso": "ENSO",
    "sp500_logret": "S&P 500 log-ret.", "sp500_sqret": "S&P 500 sq. ret.",
}


def _n_features(n_qubits: int, nv: int) -> int:
    """Readout feature count: nv x (3n single-body + 9*C(n,2) two-body)."""
    per_readout = 3 * n_qubits + 9 * (n_qubits * (n_qubits - 1) // 2)
    return nv * per_readout


def run() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    combined = []  # rows for the combined long-format csv

    for nv in NV_VALUES:
        nfeat = _n_features(N_QUBITS, nv)
        print(f"\n########## nv = {nv}  ({nfeat} readout features) ##########",
              flush=True)
        curves = {}
        for key, loader, n_org, spacing in DATASETS:
            split = loader()
            ctots, curve_seeds, nrmses, horizons = [], [], [], []
            for s in SEEDS:
                model = QuantumReservoir(n_qubits=N_QUBITS, n_virtual=nv, seed=s)
                model.fit(split.train)
                r = evaluation.evaluate(model, split.series, split.train_len,
                                        HORIZON, n_org, spacing)
                ctots.append(r["C_tot"])
                curve_seeds.append(r["capacity_curve"])
                nrmses.append(r["nrmse_short"])
                horizons.append(r["skill_horizon"])
            c_tot, c_std = float(np.mean(ctots)), float(np.std(ctots))
            curve = np.nanmean(curve_seeds, axis=0)
            curves[key] = curve
            row = {
                "nv": nv, "n_features": nfeat, "signal": key,
                "signal_name": DISPLAY[key], "C_tot": c_tot, "C_tot_std": c_std,
                "skill_horizon": float(np.mean(horizons)),
                "nrmse50": float(np.mean(nrmses)), "n_seeds": len(SEEDS),
            }
            combined.append(row)
            print(f"  {DISPLAY[key]:16s} C_tot={c_tot:6.2f} +-{c_std:5.2f}"
                  f"  skill_h={np.mean(horizons):5.1f}", flush=True)

            # Per-(nv, signal) csv, nv-tagged so nothing is overwritten.
            with open(os.path.join(RESULTS_DIR, f"qrc_nv{nv}_{key}.csv"), "w",
                      newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(row.keys()))
                w.writeheader()
                w.writerow(row)

        # Per-nv capacity curves, nv-tagged.
        np.savez(os.path.join(RESULTS_DIR, f"qrc_nv{nv}_curves.npz"), **curves)

    # Combined long-format table across all nv.
    path = os.path.join(RESULTS_DIR, "qrc_nv_scan.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(combined[0].keys()))
        w.writeheader()
        w.writerows(combined)
    print(f"\nwrote {path}")
    _figure(combined)


def _figure(combined: list) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"(skipping figure: {exc})")
        return

    keys = [k for k, *_ in DATASETS]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))

    # Left: C_tot vs nv, one line per signal.
    for key in keys:
        rows = sorted((r for r in combined if r["signal"] == key),
                      key=lambda r: r["nv"])
        xs = [r["nv"] for r in rows]
        ys = [r["C_tot"] for r in rows]
        es = [r["C_tot_std"] for r in rows]
        axL.errorbar(xs, ys, yerr=es, marker="o", lw=1.4, capsize=3,
                     label=DISPLAY[key])
    axL.set_xlabel("virtual nodes $n_v$", fontsize=11)
    axL.set_ylabel(r"total forecasting capacity $C_{\mathrm{tot}}$", fontsize=11)
    axL.set_xticks(NV_VALUES)
    axL.grid(True, linestyle=":", alpha=0.4)
    axL.legend(fontsize=9, frameon=False)
    axL.text(0.025, 0.96, "(a)", transform=axL.transAxes, fontsize=12,
             fontweight="bold", va="top")

    # Right: laser capacity curves C(tau) at each nv (the cleanest case).
    tau = np.arange(1, HORIZON + 1)
    for nv in NV_VALUES:
        npz = np.load(os.path.join(RESULTS_DIR, f"qrc_nv{nv}_curves.npz"))
        if "laser" in npz.files:
            axR.plot(tau, npz["laser"], lw=1.4, label=rf"$n_v={nv}$")
    axR.set_xlabel(r"forecast lead time $\tau$ (steps)", fontsize=11)
    axR.set_ylabel(r"$C(\tau)=\rho^2(\tau)$", fontsize=11)
    axR.set_xlim(1, HORIZON)
    axR.set_ylim(-0.02, 1.02)
    axR.grid(True, linestyle=":", alpha=0.4)
    axR.legend(fontsize=9, frameon=False, title="Santa Fe laser")
    axR.text(0.025, 0.96, "(b)", transform=axR.transAxes, fontsize=12,
             fontweight="bold", va="top")

    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "qrc_nv_scan.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    run()
