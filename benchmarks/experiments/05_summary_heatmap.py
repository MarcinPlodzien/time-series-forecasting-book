"""
05_summary_heatmap.py
=====================

Cross-signal summary figure for the closing chapter: a heatmap of the total
forecasting capacity C_tot for every method (rows) against every signal
(columns), read straight from the combined results table. It compresses the
five per-signal tables into one picture of the predictability spectrum -- bright
columns where the dynamics are forecastable (Mackey-Glass, laser), dark columns
where they are not (ENSO, S&P returns), with the volatility column (S&P squared
returns) lighting up again at short lead.

    python experiments/05_summary_heatmap.py
    -> results/benchmark_summary_heatmap.png

Reads results/comparison_all.csv (written by 03_full_comparison.py).
"""

from __future__ import annotations

import csv
import os

import numpy as np

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

# Signals in narrative (predictability-spectrum) order: most deterministic first.
SIGNALS = [
    ("mackeyglass", "Mackey-\nGlass"),
    ("laser", "Santa Fe\nlaser"),
    ("enso", "ENSO"),
    ("sp500_sqret", "S&P 500\nsq. ret."),
    ("sp500_logret", "S&P 500\nlog-ret."),
]

# Method display order, grouped by family (baseline -> linear -> ... -> quantum).
METHOD_ORDER = [
    "Persistence", "Mean",
    "AR", "ARIMA", "DLinear", "Volterra", "NVAR", "Koopman",
    "ESN", "RNN", "LSTM", "GRU",
    "MLP", "TCN", "S4D", "Mamba", "Transformer", "Neural ODE",
    "QRC",
]


def load_ctot() -> dict:
    """(method, signal_key) -> C_tot from the combined long-format table."""
    table = {}
    with open(os.path.join(RESULTS, "comparison_all.csv")) as fh:
        for r in csv.DictReader(fh):
            table[(r["model"], r["dataset"])] = float(r["C_tot"])
    return table


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = load_ctot()
    # Sort method rows best to worst by mean C_tot across the signals.
    present = [m for m in METHOD_ORDER
               if any((m, k) in table for k, _ in SIGNALS)]

    def _mean_ctot(m):
        vals = [table[(m, k)] for k, _ in SIGNALS if (m, k) in table]
        return sum(vals) / len(vals) if vals else -1.0

    methods = [m for m in sorted(present, key=_mean_ctot, reverse=True)
               if m != "Mean"]  # the Mean baseline is all-zero; omit the row
    keys = [k for k, _ in SIGNALS]
    labels = [lab for _, lab in SIGNALS]

    M = np.full((len(methods), len(keys)), np.nan)
    for i, m in enumerate(methods):
        for j, k in enumerate(keys):
            if (m, k) in table:
                M[i, j] = table[(m, k)]

    fig, ax = plt.subplots(figsize=(6.6, 8.4))
    im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=60)

    # White gridlines separating the cells.
    ax.set_xticks(np.arange(len(keys) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(methods) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)

    # Signal labels on top, method labels on the left, no tick marks or frame.
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=9)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Annotate each cell with the C_tot value, in a contrasting colour.
    for i in range(len(methods)):
        for j in range(len(keys)):
            if np.isnan(M[i, j]):
                continue
            val = M[i, j]
            ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                    fontsize=8, color="white" if val < 33 else "black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$C_{\mathrm{tot}}$", fontsize=12)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0)
    fig.tight_layout()
    out = os.path.join(RESULTS, "benchmark_summary_heatmap.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
