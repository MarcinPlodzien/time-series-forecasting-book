"""
make_latex_tables.py
===================

Generate the LaTeX tables for the appendix from the experiment outputs, so the
appendix is regenerated from data rather than transcribed by hand:

    table_diagnostics.tex      <- results/diagnose_all.csv     (characterize first)
    table_results_<key>.tex    <- results/comparison_<key>.csv   (one per signal)
    table_hparams.tex          <- results/comparison_config.txt  (+ live param counts)

    python experiments/make_latex_tables.py
"""

from __future__ import annotations

import csv
import os

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

# Signals, in narrative order, with display names and whether the results table
# should carry a directional-accuracy column.
DATASETS = [
    ("mackeyglass", "Mackey-Glass", False),
    ("laser", "Santa Fe laser", False),
    ("enso", "ENSO (Ni\\~no 3.4)", False),
    ("sp500_logret", "S\\&P 500 log-returns", True),
    ("sp500_sqret", "S\\&P 500 squared returns", False),
]

# Ablation variants kept in the benchmark data but not shown in the chapter tables.
CHAPTER_EXCLUDE = {"S4D (random)"}

FAMILY_ORDER = [
    "Baseline",
    "Linear & feature (Ch5)",
    "Reservoir & recurrent (Ch6)",
    "Modern sequence (Ch7)",
    "Quantum (Ch8)",
]

ROLE_LABEL = {
    "baseline": "baseline", "linear": "linear", "matched": "matched",
    "weak prior": "weak prior", "generic": "generic",
}

VERDICT_LABEL = {
    "chaos": "low-dim.\\ deterministic chaos",
    "nonlinear": "nonlinear, not low-dim.",
    "stochastic": "stochastic (null not rejected)",
}


def _esc(s: str) -> str:
    return s.replace("&", "\\&").replace("_", "\\_")


# ---------------------------------------------------------------------------
# Characterization table (the "characterize first" diagnostics).
# ---------------------------------------------------------------------------
def diagnostics_table() -> str:
    with open(os.path.join(RESULTS, "diagnose_all.csv")) as fh:
        rows = list(csv.DictReader(fh))
    lines = [
        r"\begin{table}[ht]",
        r"\centering\small",
        r"\caption{Dynamical fingerprint of each signal, computed on the training "
        r"window before any model is chosen: autocorrelation time, embedding delay "
        r"$\tau$ (first mutual-information minimum), embedding dimension $m$ "
        r"(false-nearest-neighbour collapse), largest Lyapunov exponent "
        r"$\lambda_1$, and the AAFT surrogate-data test (the $z$-score of a "
        r"nonlinear-prediction statistic against $39$ amplitude-adjusted Fourier "
        r"surrogates; a large negative $z$ rejects the linear-stochastic null). "
        r"The verdict sets the forecasting expectation for each signal.}",
        r"\label{tab:appendix_diagnostics}",
        r"\begin{tabular}{lrrrrrl}",
        r"\toprule",
        r"Signal & $t_{\mathrm{ac}}$ & $\tau$ & $m$ & $\lambda_1$ & $z_{\mathrm{surr}}$ & Verdict \\",
        r"\midrule",
    ]
    # Display names keyed by the diagnostics CSV keys (already LaTeX-escaped).
    diag_names = {
        "mackeyglass": "Mackey-Glass", "laser": "Santa Fe laser",
        "enso": "ENSO (Ni\\~no 3.4)", "sp500": "S\\&P 500 log-ret.",
    }
    order = ["mackeyglass", "laser", "enso", "sp500"]
    rows_by_key = {r["key"]: r for r in rows}
    for key in order:
        r = rows_by_key.get(key)
        if r is None:
            continue
        name = diag_names.get(key, _esc(r["name"]))
        lines.append(
            rf"{name} & {r['ac']} & {r['tau']} & {r['m']} & {float(r['lyap']):.4f} "
            rf"& {float(r['surr_z']):.1f} & {VERDICT_LABEL.get(r['verdict'], r['verdict'])} \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Per-signal results table.
# ---------------------------------------------------------------------------
def results_table(key: str, name: str, with_diracc: bool) -> str:
    with open(os.path.join(RESULTS, f"comparison_{key}.csv")) as fh:
        rows = list(csv.DictReader(fh))
    colspec = "llrr" + ("r" if with_diracc else "")
    header = (r"Method & Role & $C_{\mathrm{tot}}$ & $\tau_{1/2}$"
              + (r" & Dir.\ acc." if with_diracc else "") + r" \\")
    ncol = 5 if with_diracc else 4
    cap = (
        rf"Forecasting capacity on {name}, closed-loop rolling-origin evaluation. "
        r"$C_{\mathrm{tot}}=\sum_{\tau=1}^{60}C(\tau)$ with $C(\tau)=\rho^2(\tau)$ is "
        r"the total forecasting capacity in effective predictable steps, and "
        r"$\tau_{1/2}$ is the skill horizon, the first lead at which $C(\tau)$ falls "
        r"below $0.5$. Random models are averaged over three seeds ($\pm$ one s.d.\ "
        r"on $C_{\mathrm{tot}}$)."
        + (r" Directional accuracy ($0.5$ is chance) is the informative metric for "
           r"returns." if with_diracc else "")
    )
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{{cap}}}",
        rf"\label{{tab:appendix_results_{key}}}",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    by_fam = {f: [] for f in FAMILY_ORDER}
    for r in rows:
        if r["model"] in CHAPTER_EXCLUDE:   # ablation rows kept in data, not shown
            continue
        by_fam.setdefault(r["family"], []).append(r)
    for fam in FAMILY_ORDER:
        frows = sorted(by_fam.get(fam, []), key=lambda r: -float(r["C_tot"]))
        if not frows:
            continue
        lines.append(rf"\multicolumn{{{ncol}}}{{l}}{{\emph{{{_esc(fam.split(' (')[0])}}}}} \\")
        for r in frows:
            ctot = float(r["C_tot"])
            std = float(r["C_tot_std"])
            ctot_s = f"{ctot:.1f}" + (rf"$\pm${std:.1f}" if std > 0 else "")
            row = (rf"\quad {_esc(r['model'])} & {ROLE_LABEL.get(r['role'], r['role'])} "
                   rf"& {ctot_s} & {float(r['skill_horizon']):.0f}")
            if with_diracc:
                row += rf" & {float(r['dir_acc']):.2f}"
            lines.append(row + r" \\")
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Hyperparameter table.
# ---------------------------------------------------------------------------
def _neural_param_counts() -> dict:
    try:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
        from tsforecast.models import neural as N
        from tsforecast.models.neural_ode import make_neural_ode
    except Exception:
        return {}
    facs = {
        "RNN": N.make_rnn, "LSTM": N.make_lstm, "GRU": N.make_gru, "MLP": N.make_mlp,
        "TCN": N.make_tcn, "S4D": N.make_s4d, "Mamba": N.make_mamba,
        "Transformer": N.make_transformer,
        "DLinear": N.make_dlinear, "Neural ODE": make_neural_ode,
    }
    counts = {}
    for name, f in facs.items():
        net = f(seed=0).module_factory()
        counts[name] = sum(p.numel() for p in net.parameters())
    return counts


def hparams_table() -> str:
    params = _neural_param_counts()
    rows = []
    with open(os.path.join(RESULTS, "comparison_config.txt")) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("[") and "]" in line:
                rest = line[line.index("]") + 1:].strip()
                if ":" in rest:
                    name, hp = rest.split(":", 1)
                    rows.append((name.strip(), hp.strip()))
    lines = [
        r"\begin{table}[ht]",
        r"\centering\small",
        r"\caption{Hyperparameters of every method. Neural models share the "
        r"training protocol (Adam, learning rate $3\times10^{-3}$, early stopping "
        r"on a $15\%$ future-validation split, input window $30$); each is sized to "
        r"a shared budget of about $3000$ trainable parameters so the comparison "
        r"reflects inductive structure, not scale. The reservoir and feature "
        r"methods (ESN, NVAR, Koopman, QRC) train only a linear readout and are "
        r"sized by their substrate instead.}",
        r"\label{tab:appendix_hparams}",
        r"\begin{tabular}{lrp{0.55\textwidth}}",
        r"\toprule",
        r"Method & Params & Settings \\",
        r"\midrule",
    ]
    for name, hp in rows:
        if not hp:
            hp = "(parameter-free)"
        for boiler in [", weight_decay=1e-05", ", patience=50", ", patience=120",
                       ", batch_size=64", ", val_frac=0.15", ", seed=0", ", lr=0.003",
                       ", epochs=500", ", epochs=250", ", epochs=800", ", window=30"]:
            hp = hp.replace(boiler, "")
        pcount = params.get(name)
        pstr = f"{pcount}" if pcount is not None else "--"
        lines.append(rf"{_esc(name)} & {pstr} & {_esc(hp)} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main() -> None:
    with open(os.path.join(RESULTS, "table_diagnostics.tex"), "w") as fh:
        fh.write(diagnostics_table() + "\n")
    for key, name, with_diracc in DATASETS:
        with open(os.path.join(RESULTS, f"table_results_{key}.tex"), "w") as fh:
            fh.write(results_table(key, name, with_diracc) + "\n")
    with open(os.path.join(RESULTS, "table_hparams.tex"), "w") as fh:
        fh.write(hparams_table() + "\n")
    print("wrote table_diagnostics.tex, table_results_<key>.tex (x5), table_hparams.tex")


if __name__ == "__main__":
    main()
