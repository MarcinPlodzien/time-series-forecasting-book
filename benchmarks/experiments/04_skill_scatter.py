"""
experiment 04: predicted-vs-true skill scatter
==============================================

For each method and signal, scatter the true value s_{k+tau} (x-axis) against the
model's prediction of it (y-axis), pooled over rolling-origin forecasts. The
least-squares R^2 of this scatter equals the forecasting capacity C(tau), so the
plot is the visual form of the headline metric: points hugging the identity line
mean skill at lead tau, a round blob means none.

Layout: methods (rows) x signals (columns), one figure per lead time tau. Each
panel shows the identity line (dashed), a least-squares fit (solid), and the
annotated rho^2 (= C(tau)) and slope.

This is a pure renderer: it reads the first-seed rolling-origin forecasts that
experiment 03 already persisted (results/comparison_<key>_forecasts.npz) and does
no fitting of its own, so the scatter's rho^2 is the C(tau) of the same forecast
pass that produced the tables. Run experiment 03 first.

    python experiments/04_skill_scatter.py

Outputs ../results/skill_scatter_tau{1,2,5,10}.png.
"""

from __future__ import annotations

import os

import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

TAUS = [1, 2, 5, 10]

# Signals (columns): (display label, results key used by experiment 03).
DATASETS = [
    ("Mackey-Glass", "mackeyglass"),
    ("Santa Fe laser", "laser"),
    ("ENSO", "enso"),
    ("S&P 500 returns", "sp500_logret"),
]

# Methods (rows): six methods spanning the families, ordered to read as a gradient
# from no skill to tight skill. (display label, model name as stored by experiment 03.)
METHODS = [
    ("Linear AR", "Linear AR"),
    ("NVAR", "NVAR"),
    ("LSTM", "LSTM"),
    ("Transformer", "Transformer"),
    ("Neural ODE", "Neural ODE"),
    ("QRC", "Quantum reservoir"),
]


def collect() -> dict:
    """Load the first-seed forecasts experiment 03 persisted per signal.

    Returns {(method_label, dataset_label) -> (preds, truths)} with each array of
    shape (n_origins, horizon). No fitting: the forecasts are those that produced
    the capacity tables, so the scatter and the tables cannot disagree.
    """
    out = {}
    for dname, key in DATASETS:
        path = os.path.join(RESULTS_DIR, f"comparison_{key}_forecasts.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{os.path.basename(path)} not found; run experiment 03 first "
                "(it now persists the forecasts this script renders)."
            )
        npz = np.load(path)
        truths = npz["__truths__"]
        for mlabel, store_name in METHODS:
            pkey = f"{store_name}|p"
            if pkey not in npz.files:
                continue  # method absent from this run; panel left out below
            out[(mlabel, dname)] = (npz[pkey], truths)
    return out


def make_figure(data: dict, tau: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dnames = [d[0] for d in DATASETS]
    mlabels = [m[0] for m in METHODS]
    nr, nc = len(mlabels), len(dnames)

    # Shared square limits per signal column (each column is one signal, so its
    # panels share an axis range; the identity line is then the same 45 degrees
    # everywhere in that column). Ranges differ across columns, so limits are not
    # shared across columns.
    col_lim = {}
    for dname in dnames:
        vals = []
        for mlabel in mlabels:
            preds, truths = data[(mlabel, dname)]
            vals.append(truths[:, tau - 1]); vals.append(preds[:, tau - 1])
        v = np.concatenate(vals)
        lo, hi = np.percentile(v, 0.5), np.percentile(v, 99.5)  # robust to rollout outliers
        pad = 0.05 * (hi - lo)
        col_lim[dname] = (lo - pad, hi + pad)

    fig, axes = plt.subplots(nr, nc, figsize=(2.3 * nc, 1.9 * nr))
    for i, mlabel in enumerate(mlabels):
        for j, dname in enumerate(dnames):
            ax = axes[i, j]
            preds, truths = data[(mlabel, dname)]
            x = truths[:, tau - 1]              # true s_{k+tau}
            y = preds[:, tau - 1]               # predicted
            lo, hi = col_lim[dname]
            ax.scatter(x, y, s=5, alpha=0.4, color="#2c3e50", edgecolors="none")
            ax.plot([lo, hi], [lo, hi], ls="--", lw=0.8, color="0.6")  # identity
            if np.std(x) > 1e-12 and np.std(y) > 1e-12:
                slope, intercept = np.polyfit(x, y, 1)
                rho = np.corrcoef(x, y)[0, 1]
                xs = np.array([lo, hi])
                ax.plot(xs, slope * xs + intercept, lw=1.1, color="#c0392b")
                ax.text(0.05, 0.93, rf"$\rho^2\!=\!{rho**2:.2f}$", transform=ax.transAxes,
                        fontsize=7, va="top")
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal", "box")
            # Sparse numeric ticks: marks on every panel, labels only on the
            # outer panels (bottom row for x, left column for y).
            ax.xaxis.set_major_locator(plt.MaxNLocator(3, prune="both"))
            ax.yaxis.set_major_locator(plt.MaxNLocator(3, prune="both"))
            ax.tick_params(labelsize=6, length=2.5, pad=1.5)
            if i != nr - 1:
                ax.tick_params(labelbottom=False)
            if j != 0:
                ax.tick_params(labelleft=False)
            ax.grid(True, linestyle=":", alpha=0.35)
            if j == 0:
                ax.set_ylabel(mlabel, fontsize=8.5, rotation=90, va="center", labelpad=22)
            if i == 0:
                ax.set_title(dname, fontsize=9.5)
    fig.supxlabel(rf"true value $s_{{k+{tau}}}$", fontsize=11)
    fig.supylabel("predicted value", fontsize=11, x=0.015)
    fig.subplots_adjust(wspace=0.07, hspace=0.07, left=0.14, right=0.985,
                        top=0.94, bottom=0.08)
    path = os.path.join(RESULTS_DIR, f"skill_scatter_tau{tau}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("loading persisted rolling-origin forecasts from experiment 03...")
    data = collect()
    for tau in TAUS:
        make_figure(data, tau)


if __name__ == "__main__":
    main()
