"""
06_perlead_table.py
==================

Per-lead forecasting capacity C(tau) at tau = 1, 2, 5, 10, read from the saved
capacity curves. C_tot collapses the whole C(tau) curve to one number; this
unpacks it at a few representative leads so the decay of skill with horizon is
visible -- the short-lead capacity is high for matched models on the chaotic
signals and decays with tau, while it is flat-and-low on the stochastic signals.

Prints a console table (row = method, columns = the four leads) for each signal,
and writes a LaTeX table for the two deterministic signals where the lead
structure is informative.

    python experiments/06_perlead_table.py
    -> stdout tables for all signals
    -> results/table_perlead_chaos.tex  (Mackey-Glass + laser)

Reads results/comparison_<key>_curves.npz (written by 03_full_comparison.py).
"""

from __future__ import annotations

import os

import numpy as np

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

LEADS = [1, 2, 5, 10]  # one-indexed lead times; curve[tau-1] is C(tau)

SIGNALS = [
    ("mackeyglass", "Mackey-Glass"),
    ("laser", "Santa Fe laser"),
    ("enso", "ENSO"),
    ("sp500_sqret", "S&P 500 squared returns"),
    ("sp500_logret", "S&P 500 log-returns"),
]

# LaTeX table covers the two signals whose lead-decay is informative.
TEX_SIGNALS = [("mackeyglass", "Mackey--Glass"), ("laser", "Santa Fe laser")]

METHOD_ORDER = [
    "Persistence", "AR", "ARIMA", "DLinear", "Volterra", "NVAR", "Koopman",
    "ESN", "RNN", "LSTM", "GRU",
    "MLP", "TCN", "S4D", "Mamba", "Transformer", "Neural ODE", "QRC",
]


def _esc(s: str) -> str:
    return s.replace("&", "\\&").replace("_", "\\_")


def load_curves(key: str) -> dict:
    path = os.path.join(RESULTS, f"comparison_{key}_curves.npz")
    if not os.path.exists(path):
        return {}
    data = np.load(path)
    return {m: data[m] for m in data.files}


_EXCLUDE = {"Mean", "S4D (random)"}  # baseline + ablation kept in data, not tabled


def _ordered(curves: dict):
    present = [m for m in METHOD_ORDER if m in curves]
    present += [m for m in curves if m not in METHOD_ORDER and m not in _EXCLUDE]
    return present


def print_console() -> None:
    for key, name in SIGNALS:
        curves = load_curves(key)
        if not curves:
            print(f"\n[{name}] no curves npz found")
            continue
        print(f"\n=== {name} : C(tau) at tau = {LEADS} ===")
        hdr = "  method            " + "".join(f"  C({t}) " for t in LEADS) + "  C_tot"
        print(hdr)
        for m in _ordered(curves):
            c = curves[m]
            vals = "".join(f"  {c[t - 1]:5.2f}" for t in LEADS)
            print(f"  {m:18s}{vals}   {float(np.sum(c)):6.1f}")


def tex_table() -> str:
    lines = [
        r"\begin{table}[ht]",
        r"\centering\small",
        r"\caption{Per-lead forecasting capacity $C(\tau)=\rho^2(\tau)$ at lead "
        r"times $\tau=1,2,5,10$ on the two deterministic signals, unpacking the "
        r"$C_{\mathrm{tot}}$ of Tables~\ref{tab:appendix_results_mackeyglass} "
        r"and~\ref{tab:appendix_results_laser}. The matched nonlinear models hold "
        r"high capacity at short lead and decay gracefully; the linear models lose "
        r"it within a few steps.}",
        r"\label{tab:appendix_perlead}",
        r"\begin{tabular}{l" + "rrrr" * len(TEX_SIGNALS) + "}",
        r"\toprule",
        " & " + " & ".join(rf"\multicolumn{{4}}{{c}}{{{name}}}" for _, name in TEX_SIGNALS)
        + r" \\",
        "Method & " + " & ".join(" & ".join(rf"$C({t})$" for t in LEADS)
                                 for _ in TEX_SIGNALS) + r" \\",
        r"\midrule",
    ]
    curves_by_key = {k: load_curves(k) for k, _ in TEX_SIGNALS}
    methods = _ordered(curves_by_key[TEX_SIGNALS[0][0]])
    for m in methods:
        cells = []
        for k, _ in TEX_SIGNALS:
            c = curves_by_key[k].get(m)
            if c is None:
                cells += ["--"] * len(LEADS)
            else:
                cells += [f"{c[t - 1]:.2f}" for t in LEADS]
        lines.append(rf"{_esc(m)} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main() -> None:
    print_console()
    if all(load_curves(k) for k, _ in TEX_SIGNALS):
        out = os.path.join(RESULTS, "table_perlead_chaos.tex")
        with open(out, "w") as fh:
            fh.write(tex_table() + "\n")
        print(f"\nwrote {out}")
    else:
        print("\n(skipping LaTeX table: chaotic-signal curves not both present)")


if __name__ == "__main__":
    main()
