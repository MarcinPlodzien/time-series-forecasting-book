"""
08_mamba_depth_scan.py
=====================

How the selective state-space model (Mamba) scales with depth, at a fixed
parameter budget. For each number of stacked selective blocks (1..4) the channel
width is shrunk so the model always has about 3000 trainable parameters, matching
the budget every other network in the benchmark is held to. This isolates the
effect of depth alone -- not of spending more parameters -- and keeps the result
comparable to the full comparison numbers. It tests whether adding layers helps
Mamba on these signals (it largely does not on smooth chaos, where selectivity
has little to select; see Chapter 7).

Outputs are tagged by layer count so nothing is overwritten:
    results/mamba_L{nl}_{key}.csv       per (n_layers, signal)
    results/mamba_L{nl}_curves.npz      per n_layers: C(tau) per signal
    results/mamba_depth_scan.csv        combined long-format table
    results/mamba_depth_scan.png        C_tot vs n_layers per signal + laser C(tau)

    python experiments/08_mamba_depth_scan.py
    python experiments/08_mamba_depth_scan.py quick

The n_layers=2 point uses the same architecture as the deployed full comparison Mamba
(experiment 03), so the two numbers cross-check.
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
from tsforecast.models import neural  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

QUICK = "quick" in sys.argv[1:]
HORIZON = 60
LAYERS = [1, 2, 3, 4]
SEEDS = [0] if QUICK else [0, 1, 2]
N_STATE = 8                 # same per-block state size as the deployed Mamba
PARAM_BUDGET = 3000         # shared trainable-parameter budget
WINDOW = 30

#   key, loader, n_origins, spacing  (identical to experiment 03)
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


def _param_count(channels: int, n_layers: int) -> int:
    net = neural._Mamba(channels=channels, n_state=N_STATE, n_layers=n_layers)
    return sum(p.numel() for p in net.parameters())


def _fit_channels(n_layers: int) -> tuple[int, int]:
    """Channel width whose param count is closest to PARAM_BUDGET for this depth."""
    best_c, best_p, best_gap = None, None, None
    for c in range(4, 64):
        p = _param_count(c, n_layers)
        gap = abs(p - PARAM_BUDGET)
        if best_gap is None or gap < best_gap:
            best_c, best_p, best_gap = c, p, gap
    return best_c, best_p


def _build_forecaster(channels: int, n_layers: int, seed: int):
    arch = (f"Mamba, {n_layers} selective blocks x ({channels} channels, "
            f"{N_STATE} modes), causal conv k=4")
    return neural.SequenceForecaster(
        lambda: neural._Mamba(channels=channels, n_state=N_STATE, n_layers=n_layers),
        WINDOW, "Mamba", epochs=800, patience=120, seed=seed, arch=arch,
    )


def run() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    combined = []

    for nl in LAYERS:
        channels, nparams = _fit_channels(nl)
        print(f"\n########## n_layers = {nl}  ({channels} channels, "
              f"{nparams} params) ##########", flush=True)
        curves = {}
        for key, loader, n_org, spacing in DATASETS:
            split = loader()
            ctots, curve_seeds, nrmses, horizons = [], [], [], []
            for s in SEEDS:
                model = _build_forecaster(channels, nl, s)
                model.fit(split.train)
                r = evaluation.evaluate(model, split.series, split.train_len,
                                        HORIZON, n_org, spacing)
                ctots.append(r["C_tot"])
                curve_seeds.append(r["capacity_curve"])
                nrmses.append(r["nrmse_short"])
                horizons.append(r["skill_horizon"])
            c_tot, c_std = float(np.mean(ctots)), float(np.std(ctots))
            curves[key] = np.nanmean(curve_seeds, axis=0)
            row = {
                "n_layers": nl, "channels": channels, "n_params": nparams,
                "signal": key, "signal_name": DISPLAY[key],
                "C_tot": c_tot, "C_tot_std": c_std,
                "skill_horizon": float(np.mean(horizons)),
                "nrmse50": float(np.mean(nrmses)), "n_seeds": len(SEEDS),
            }
            combined.append(row)
            print(f"  {DISPLAY[key]:16s} C_tot={c_tot:6.2f} +-{c_std:5.2f}"
                  f"  skill_h={np.mean(horizons):5.1f}", flush=True)
            with open(os.path.join(RESULTS_DIR, f"mamba_L{nl}_{key}.csv"), "w",
                      newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(row.keys()))
                w.writeheader()
                w.writerow(row)
        np.savez(os.path.join(RESULTS_DIR, f"mamba_L{nl}_curves.npz"), **curves)

    path = os.path.join(RESULTS_DIR, "mamba_depth_scan.csv")
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

    for key in keys:
        rows = sorted((r for r in combined if r["signal"] == key),
                      key=lambda r: r["n_layers"])
        xs = [r["n_layers"] for r in rows]
        ys = [r["C_tot"] for r in rows]
        es = [r["C_tot_std"] for r in rows]
        axL.errorbar(xs, ys, yerr=es, marker="o", lw=1.4, capsize=3,
                     label=DISPLAY[key])
    axL.set_xlabel("Mamba depth (selective blocks)", fontsize=11)
    axL.set_ylabel(r"total forecasting capacity $C_{\mathrm{tot}}$", fontsize=11)
    axL.set_xticks(LAYERS)
    axL.grid(True, linestyle=":", alpha=0.4)
    axL.legend(fontsize=9, frameon=False)
    axL.text(0.025, 0.96, "(a)", transform=axL.transAxes, fontsize=12,
             fontweight="bold", va="top")

    tau = np.arange(1, HORIZON + 1)
    for nl in LAYERS:
        npz = np.load(os.path.join(RESULTS_DIR, f"mamba_L{nl}_curves.npz"))
        if "laser" in npz.files:
            axR.plot(tau, npz["laser"], lw=1.4, label=f"{nl} layer(s)")
    axR.set_xlabel(r"forecast lead time $\tau$ (steps)", fontsize=11)
    axR.set_ylabel(r"$C(\tau)=\rho^2(\tau)$", fontsize=11)
    axR.set_xlim(1, HORIZON)
    axR.set_ylim(-0.02, 1.02)
    axR.grid(True, linestyle=":", alpha=0.4)
    axR.legend(fontsize=9, frameon=False, title="Santa Fe laser")
    axR.text(0.025, 0.96, "(b)", transform=axR.transAxes, fontsize=12,
             fontweight="bold", va="top")

    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "mamba_depth_scan.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    run()
