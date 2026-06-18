"""
experiment 01: diagnose every benchmark signal
==============================================

Compute the dynamical fingerprint of each signal from its training window alone,
before any model is chosen:

    * autocorrelation time      (a crude memory scale),
    * embedding delay tau       (first minimum of the mutual information),
    * embedding dimension m     (false-nearest-neighbour collapse),
    * largest Lyapunov exponent (Rosenstein) and the horizon ~ 1/lambda,
    * surrogate-data test        (AAFT: is the structure genuine nonlinear
                                  determinism, or a linear stochastic process?).

The fingerprint justifies the model choice and the expected outcome for each
signal. The four signals span the predictability spectrum: Mackey-Glass and the
laser are low-dimensional deterministic chaos; ENSO is a noisy oscillator; S&P
500 returns should look stochastic and fail the determinism test.

Run from the benchmarks/ directory:
    python experiments/01_diagnose.py
"""

from __future__ import annotations

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from tsforecast import embedding  # noqa: E402
from tsforecast.datasets import (  # noqa: E402
    load_enso, load_mackey_glass, load_santafe_laser, load_sp500,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

# The expensive estimators (FNN, Lyapunov, surrogates) are O(N^2); cap the window
# so every signal is diagnosed in comparable time. 1000 points is plenty for
# these summaries and matches the laser/Mackey-Glass training length.
MAX_DIAG = 1000


def diagnose(series: np.ndarray, name: str) -> dict:
    """Compute the dynamical fingerprint of one training window."""
    x = np.asarray(series, float)
    if len(x) > MAX_DIAG:
        x = x[:MAX_DIAG]

    ac = embedding.autocorrelation_time(x)
    tau = embedding.mutual_information_delay(x, max_lag=40)
    fnn = embedding.false_nearest_neighbours(x, tau=tau, max_dim=8)
    m = next((d + 1 for d, f in enumerate(fnn) if f < 0.1), len(fnn))
    lyap, _ = embedding.largest_lyapunov_rosenstein(x, tau=tau, dim=max(m, 3))
    horizon = (1.0 / lyap) if lyap > 0 else float("inf")
    surr = embedding.surrogate_determinism_test(x, tau=tau, dim=max(m, 3))

    return {
        "name": name, "n": len(x), "ac": ac, "tau": tau, "fnn": fnn, "m": m,
        "lyap": lyap, "horizon": horizon, "surr": surr,
    }


def verdict(d: dict) -> str:
    """A one-line reading derived from the numbers, not hard-coded per signal."""
    s = d["surr"]
    if s["deterministic"] and d["m"] <= 5 and d["lyap"] > 0:
        return ("low-dimensional deterministic chaos (surrogate null rejected): "
                "nonlinear state/feature models are matched, linear models are not.")
    if s["deterministic"]:
        return ("nonlinear structure present but not cleanly low-dimensional: "
                "a noisy or higher-dimensional deterministic component.")
    return ("consistent with a linear stochastic process (surrogate null NOT "
            "rejected): deterministic point forecasting is the wrong tool.")


def format_report(d: dict) -> str:
    s = d["surr"]
    hz = "inf" if not np.isfinite(d["horizon"]) else f"{d['horizon']:.1f}"
    return "\n".join([
        f"{d['name']} -- dynamical fingerprint (training window, {d['n']} samples)",
        "=" * 72,
        f"autocorrelation time (1/e)        : {d['ac']} samples",
        f"embedding delay tau (MI minimum)  : {d['tau']} samples",
        f"false-neighbour fractions by dim  : " + ", ".join(f"{f:.2f}" for f in d["fnn"]),
        f"embedding dimension m (<10% FNN)  : {d['m']}",
        f"largest Lyapunov exponent lambda1 : {d['lyap']:.4f} per sample",
        f"predictability horizon ~ 1/lambda : {hz} samples",
        f"surrogate test (AAFT, nonlin err) : data={s['stat_data']:.3f}  "
        f"surrogates={s['surr_mean']:.3f}+-{s['surr_std']:.3f}  z={s['z']:.1f}",
        f"  -> determinism null rejected    : {s['deterministic']}",
        "",
        "Reading: " + verdict(d),
    ])


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    splits = [
        (load_mackey_glass(), "mackeyglass"),
        (load_santafe_laser(), "laser"),
        (load_enso(), "enso"),
        (load_sp500("logreturns"), "sp500"),
    ]
    reports = []
    diags = []
    for split, key in splits:
        print(f"diagnosing {split.name} ...", flush=True)
        d = diagnose(split.train, split.name)
        diags.append(d)
        report = format_report(d)
        reports.append(report)
        with open(os.path.join(RESULTS_DIR, f"diagnose_{key}.txt"), "w") as fh:
            fh.write(report + "\n")
        print(report + "\n")

    combined = ("\n\n".join(reports)
                + "\n\nThe four fingerprints span the predictability spectrum the "
                  "benchmark is built to test.\n")
    with open(os.path.join(RESULTS_DIR, "diagnose_all.txt"), "w") as fh:
        fh.write(combined)
    print(f"wrote {os.path.join(RESULTS_DIR, 'diagnose_all.txt')}")

    # Structured table for the appendix (make_latex_tables reads this).
    with open(os.path.join(RESULTS_DIR, "diagnose_all.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["key", "name", "n", "ac", "tau", "m", "lyap", "horizon",
                    "surr_z", "deterministic", "verdict"])
        for (split, key), d in zip(splits, diags):
            s = d["surr"]
            hz = "" if not np.isfinite(d["horizon"]) else f"{d['horizon']:.1f}"
            short = ("chaos" if (s["deterministic"] and d["m"] <= 5 and d["lyap"] > 0)
                     else "nonlinear" if s["deterministic"] else "stochastic")
            w.writerow([key, d["name"], d["n"], d["ac"], d["tau"], d["m"],
                        f"{d['lyap']:.4f}", hz, f"{s['z']:.1f}",
                        s["deterministic"], short])
    print(f"wrote {os.path.join(RESULTS_DIR, 'diagnose_all.csv')}")


if __name__ == "__main__":
    main()
