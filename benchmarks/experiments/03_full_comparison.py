"""
experiment 03: the full architecture comparison across all signals
==========================================================

Every architecture forecasts every signal under the same closed-loop protocol,
scored by the same forecasting-capacity curve. This is the source of the appendix
tables and figures.

Signals (see datasets.py):
    Mackey-Glass            synthetic deterministic chaos (known ground truth)
    Santa Fe laser          empirical deterministic chaos
    ENSO (Nino 3.4)         noisy quasi-periodic climate index
    S&P 500 log-returns     near-efficient financial returns
    S&P 500 squared returns volatility proxy (structure in the second moment)

Scoring (metrics.py, evaluation.py):
    rolling origins -> C(tau) = squared correlation at lead tau
    -> C_tot = sum_{tau=1..HORIZON} C(tau).
C_tot is the headline scalar (total forecasting capacity, effective predictable
steps); the C(tau) curve shows how skill decays with the forecast lead tau. For
the financial objects we also report directional accuracy.

Random models (reservoirs, neural nets, the quantum reservoir, EDMD centres) are
averaged over several seeds. Deterministic models are run once.

Run (long; use `quick` for a fast 1-seed sanity pass):
    python experiments/03_full_comparison.py
    python experiments/03_full_comparison.py quick

Outputs (../results/): comparison_<key>.csv/.png/.txt per signal, comparison_all.csv,
comparison_config.txt.
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
from tsforecast.models import (  # noqa: E402
    ARIMA, ESN, KoopmanEDMD, LinearAR, MeanForecast, NVAR, Persistence,
    QuantumReservoir, Volterra, make_neural_ode, neural,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

QUICK = "quick" in sys.argv[1:]
# Lead-time horizon for the capacity sum C_tot = sum_{tau=1..HORIZON} C(tau).
HORIZON = 60
SEEDS = [0] if QUICK else [0, 1, 2]

# Per-signal rolling-origin settings. The continuation lengths differ, so the
# number of origins and their spacing are set per signal; the lead-time horizon
# is common so C_tot means the same thing everywhere. `report_dir` flags the
# signals for which directional accuracy is the informative extra metric.
#   key, loader, n_origins, spacing, report_dir
DATASETS = [
    ("mackeyglass", lambda: load_mackey_glass(),         50, 50, False),
    ("laser",       lambda: load_santafe_laser(),        50, 40, False),
    ("enso",        lambda: load_enso(),                 50,  5, False),
    ("sp500_logret", lambda: load_sp500("logreturns"),  100, 20, True),
    ("sp500_sqret",  lambda: load_sp500("sqreturns"),   100, 20, False),
]
if QUICK:
    DATASETS = [(k, ld, 10, sp, rd) for (k, ld, _n, sp, rd) in DATASETS]


# Each entry: (family, role, label, builder(seed) -> Forecaster, randomised?)
def model_specs() -> list:
    specs = [
        ("Baseline", "baseline", "Persistence", lambda s: Persistence(), False),
        ("Baseline", "baseline", "Mean", lambda s: MeanForecast(), False),
        ("Linear & feature (Ch5)", "linear", "ARIMA", lambda s: ARIMA(order=(12, 0, 1)), False),
        ("Linear & feature (Ch5)", "linear", "Linear AR", lambda s: LinearAR(order=25), False),
        ("Linear & feature (Ch5)", "matched", "Volterra", lambda s: Volterra(memory=12), False),
        ("Linear & feature (Ch5)", "matched", "NVAR", lambda s: NVAR(n_delays=10, degree=3), False),
        ("Linear & feature (Ch5)", "matched", "Koopman/EDMD", lambda s: KoopmanEDMD(delay=4, n_rbf=120, seed=s), True),
        ("Reservoir & recurrent (Ch6)", "matched", "ESN", lambda s: ESN(n_reservoir=500, spectral_radius=0.9, leak=0.6, seed=s), True),
        ("Reservoir & recurrent (Ch6)", "matched", "RNN", lambda s: neural.make_rnn(seed=s), True),
        ("Reservoir & recurrent (Ch6)", "matched", "LSTM", lambda s: neural.make_lstm(seed=s), True),
        ("Reservoir & recurrent (Ch6)", "matched", "GRU", lambda s: neural.make_gru(seed=s), True),
        ("Modern sequence (Ch7)", "generic", "MLP", lambda s: neural.make_mlp(seed=s), True),
        ("Modern sequence (Ch7)", "matched", "TCN", lambda s: neural.make_tcn(seed=s), True),
        ("Modern sequence (Ch7)", "matched", "S4D", lambda s: neural.make_s4d(seed=s), True),
        ("Modern sequence (Ch7)", "matched", "S4D (random)", lambda s: neural.make_s4d_random(seed=s), True),
        ("Modern sequence (Ch7)", "matched", "Mamba", lambda s: neural.make_mamba(seed=s), True),
        ("Modern sequence (Ch7)", "weak prior", "Transformer", lambda s: neural.make_transformer(seed=s), True),
        ("Modern sequence (Ch7)", "linear", "DLinear", lambda s: neural.make_dlinear(seed=s), True),
        ("Modern sequence (Ch7)", "matched", "Neural ODE", lambda s: make_neural_ode(seed=s), True),
        ("Quantum (Ch8)", "matched", "Quantum reservoir", lambda s: QuantumReservoir(seed=s), True),
    ]
    return [s for s in specs if s[3](0) is not None or s[2] in ("Persistence", "Mean")]


def evaluate_spec(spec, split, n_origins, spacing) -> dict:
    family, role, label, build, randomised = spec
    seeds = SEEDS if randomised else [0]
    curves, ctots, nrmses, horizons, diracc = [], [], [], [], []
    preds0 = truths0 = None
    for s in seeds:
        model = build(s)
        if model is None:
            return None
        model.fit(split.train)
        r = evaluation.evaluate(model, split.series, split.train_len,
                                HORIZON, n_origins, spacing)
        curves.append(r["capacity_curve"])
        ctots.append(r["C_tot"])
        nrmses.append(r["nrmse_short"])
        horizons.append(r["skill_horizon"])
        diracc.append(r["dir_acc"])
        if s == seeds[0]:  # keep the first-seed forecasts for the scatter store
            preds0, truths0 = r["preds"], r["truths"]
    return {
        "family": family, "role": role, "model": label,
        "C_tot": float(np.mean(ctots)), "C_tot_std": float(np.std(ctots)),
        "nrmse50": float(np.mean(nrmses)),
        "skill_horizon": float(np.mean(horizons)),
        "dir_acc": float(np.mean(diracc)),
        "curve": np.nanmean(curves, axis=0), "n_seeds": len(seeds),
        "preds0": preds0, "truths0": truths0,
    }


def _hparams(model) -> str:
    parts = []
    if getattr(model, "arch", ""):
        parts.append(model.arch)
    skip = {"name", "arch", "module_factory"}
    for k, v in vars(model).items():
        if k.startswith("_") or k in skip:
            continue
        if isinstance(v, (int, float, str, tuple, bool)):
            parts.append(f"{k}={v}")
    return ", ".join(parts)


def _save_config() -> None:
    lines = [
        "Full comparison configuration (one representative instance per method, seed 0)",
        f"Rolling-origin eval: lead-time horizon(max tau)={HORIZON}; seeds={SEEDS}.",
        "Training (neural): Adam, early stopping on a 15% future-validation split.",
        "=" * 72,
    ]
    for family, role, label, build, _rand in model_specs():
        model = build(0)
        if model is None:
            continue
        lines.append(f"[{family}] {label}: {_hparams(model)}")
    with open(os.path.join(RESULTS_DIR, "comparison_config.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {os.path.join(RESULTS_DIR, 'comparison_config.txt')}")


def _save_table(key, rows) -> None:
    path = os.path.join(RESULTS_DIR, f"comparison_{key}.csv")
    fields = ["family", "model", "role", "C_tot", "C_tot_std", "nrmse50",
              "skill_horizon", "dir_acc", "n_seeds"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: -r["C_tot"]):
            w.writerow(r)


def _save_figure(key, title, rows) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"(skipping figure: {exc})")
        return
    families = ["Linear & feature (Ch5)", "Reservoir & recurrent (Ch6)",
                "Modern sequence (Ch7)", "Quantum (Ch8)"]
    panel = ["(a)", "(b)", "(c)", "(d)"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    tau = np.arange(1, HORIZON + 1)
    base = next((r for r in rows if r["model"] == "Mean"), None)
    for ax, fam, lab in zip(axes.ravel(), families, panel):
        for r in rows:
            if r["family"] == fam:
                ax.plot(tau, r["curve"], lw=1.4,
                        label=f"{r['model']} ($C$={r['C_tot']:.1f})")
        if base is not None:
            ax.plot(tau, base["curve"], lw=1.0, ls="--", color="0.5", label="Mean")
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
    fig.savefig(os.path.join(RESULTS_DIR, f"comparison_{key}.png"), dpi=300,
                bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    _save_config()
    all_rows = []
    for key, loader, n_origins, spacing, report_dir in DATASETS:
        split = loader()
        print(f"\n=== {split.name}  (origins={n_origins}, spacing={spacing}) ===",
              flush=True)
        rows = []
        for spec in model_specs():
            res = evaluate_spec(spec, split, n_origins, spacing)
            if res is None:
                print(f"  skipped {spec[2]} (missing dependency)")
                continue
            rows.append(res)
            extra = f"  dir_acc={res['dir_acc']:.3f}" if report_dir else ""
            print(f"  {res['model']:18s} C_tot={res['C_tot']:6.2f}+-{res['C_tot_std']:5.2f}"
                  f"  skill_h={res['skill_horizon']:5.1f}  nrmse50={res['nrmse50']:.3f}{extra}",
                  flush=True)
        _save_table(key, rows)
        _save_figure(key, split.name, rows)
        # Persist the capacity curves so figures can be restyled without re-running.
        np.savez(os.path.join(RESULTS_DIR, f"comparison_{key}_curves.npz"),
                 **{r["model"]: r["curve"] for r in rows})
        # Persist the first-seed rolling-origin forecasts so the predicted-vs-true
        # scatter (experiment 04) reuses these exact forecasts instead of refitting.
        # The truths are identical across models (same origins), so store them once.
        fc = {f"{r['model']}|p": r["preds0"] for r in rows if r.get("preds0") is not None}
        truths0 = next((r["truths0"] for r in rows if r.get("truths0") is not None), None)
        if fc and truths0 is not None:
            fc["__truths__"] = truths0
            np.savez(os.path.join(RESULTS_DIR, f"comparison_{key}_forecasts.npz"), **fc)
        for r in rows:
            row = {k: v for k, v in r.items()
                   if k not in ("curve", "preds0", "truths0")}
            row["dataset"] = key
            row["dataset_name"] = split.name
            all_rows.append(row)

    # Combined long-format table across all signals.
    path = os.path.join(RESULTS_DIR, "comparison_all.csv")
    fields = ["dataset", "dataset_name", "family", "model", "role", "C_tot",
              "C_tot_std", "nrmse50", "skill_horizon", "dir_acc", "n_seeds"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
