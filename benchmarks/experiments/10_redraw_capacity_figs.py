"""
10_redraw_capacity_figs.py
========================

Redraw the per-signal capacity-curve figures (comparison_<key>.png) from the saved
capacity curves and the results table. This is needed after the readout-ridge patch
(experiment 03b), which updated the curve data but not the figures drawn during the
original run. It reproduces the 2x2 family-grouped layout of experiment 03 without
re-fitting anything.

    python experiments/10_redraw_capacity_figs.py
    -> results/comparison_<key>.png  (x5, overwritten with corrected curves)
"""

from __future__ import annotations

import csv
import os

import numpy as np

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
HORIZON = 60
KEYS = ["mackeyglass", "laser", "enso", "sp500_logret", "sp500_sqret"]
NAMES = {"mackeyglass": "Mackey-Glass", "laser": "Santa Fe laser",
         "enso": "ENSO", "sp500_logret": "S&P 500 log-returns",
         "sp500_sqret": "S&P 500 squared returns"}
FAMILIES = ["Linear & feature (Ch5)", "Reservoir & recurrent (Ch6)",
            "Modern sequence (Ch7)", "Quantum (Ch8)"]
PANEL = ["(a)", "(b)", "(c)", "(d)"]
EXCLUDE = {"S4D (random)"}  # ablation kept in data, not shown in the chapter figures


def redraw(key: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(os.path.join(RESULTS, f"comparison_{key}.csv")) as fh:
        meta = {r["model"]: r for r in csv.DictReader(fh)}
    curves = dict(np.load(os.path.join(RESULTS, f"comparison_{key}_curves.npz")))

    tau = np.arange(1, HORIZON + 1)
    base = curves.get("Mean")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for ax, fam, lab in zip(axes.ravel(), FAMILIES, PANEL):
        for model, row in meta.items():
            if row["family"] == fam and model in curves and model not in EXCLUDE:
                ax.plot(tau, curves[model], lw=1.4,
                        label=f"{model} ($C$={float(row['C_tot']):.1f})")
        if base is not None:
            ax.plot(tau, base, lw=1.0, ls="--", color="0.5", label="Mean")
        ax.set_title(fam, fontsize=11)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(1, HORIZON)
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.text(0.025, 0.96, lab, transform=ax.transAxes, fontsize=12,
                fontweight="bold", va="top")
        ax.legend(fontsize=7, loc="upper right", frameon=False)
    for ax in axes[-1]:
        ax.set_xlabel(r"lead time $\tau$", fontsize=11)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$C(\tau)$", fontsize=12)
    fig.tight_layout()
    out = os.path.join(RESULTS, f"comparison_{key}.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"redrew {out}")


def main() -> None:
    for key in KEYS:
        redraw(key)


if __name__ == "__main__":
    main()
