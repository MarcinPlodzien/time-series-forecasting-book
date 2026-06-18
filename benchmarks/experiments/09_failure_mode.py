"""
09_failure_mode.py
=================

Make the predicted failure mode visible. On the Santa Fe laser, a model whose
inductive bias matches a smooth low-dimensional flow (the neural ODE) should hold
the build-up-and-collapse pulsations in closed loop, while a linear recursion
should relax to a fixed low-amplitude oscillation and miss the collapses entirely.
This draws a single long closed-loop rollout from one origin for both, against the
truth, so the qualitative failure the diagnosis predicts is shown rather than only
stated.

    python experiments/09_failure_mode.py
    -> results/failure_mode_laser.png

Run after the data exist; uses one matched model (neural ODE) and one linear model
(autoregression).
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from tsforecast.datasets import load_santafe_laser  # noqa: E402
from tsforecast.models import LinearAR, make_neural_ode  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
HORIZON = 200          # long enough to show several pulse cycles and collapses
ORIGIN_OFFSET = 200    # start the rollout a little into the continuation


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    split = load_santafe_laser()
    series = split.series
    origin = split.train_len + ORIGIN_OFFSET
    warmup = series[:origin]
    truth = series[origin:origin + HORIZON]

    matched = make_neural_ode(seed=0)
    matched.fit(split.train)
    pred_matched = matched.forecast(HORIZON, warmup=warmup)

    linear = LinearAR(order=25)
    linear.fit(split.train)
    pred_linear = linear.forecast(HORIZON, warmup=warmup)

    t = np.arange(1, HORIZON + 1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5.4), sharex=True, sharey=True)

    ax1.plot(t, truth, color="0.2", lw=1.4, label="truth")
    ax1.plot(t, pred_matched, color="#c0392b", lw=1.4, label="neural ODE (matched)")
    ax1.set_ylabel("laser intensity", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.4)
    ax1.legend(fontsize=9, frameon=False, loc="upper right")
    ax1.text(0.012, 0.95, "(a)", transform=ax1.transAxes, fontsize=12,
             fontweight="bold", va="top")

    ax2.plot(t, truth, color="0.2", lw=1.4, label="truth")
    ax2.plot(t, pred_linear, color="#2c7fb8", lw=1.4, label="linear AR (mismatched)")
    ax2.set_xlabel(r"forecast lead time $\tau$ (steps)", fontsize=11)
    ax2.set_ylabel("laser intensity", fontsize=11)
    ax2.set_xlim(1, HORIZON)
    ax2.grid(True, linestyle=":", alpha=0.4)
    ax2.legend(fontsize=9, frameon=False, loc="upper right")
    ax2.text(0.012, 0.95, "(b)", transform=ax2.transAxes, fontsize=12,
             fontweight="bold", va="top")

    fig.tight_layout()
    out = os.path.join(RESULTS, "failure_mode_laser.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
