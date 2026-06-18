"""
03b_patch_readout.py
===================

Re-run only the readout/feature methods (ESN, NVAR, Volterra, Koopman, QRC) with
the corrected closed-loop ridge selection, and patch their entries into the
existing full comparison outputs. The neural networks and baselines do not depend on the
ridge, so their (expensive) results from experiment 03 are left untouched; this
avoids retraining them just to update five rows.

Updates, per signal:
    results/comparison_<key>.csv            (replaces the 5 readout rows)
    results/comparison_<key>_curves.npz     (replaces their capacity curves)
    results/comparison_<key>_forecasts.npz  (replaces their seed-0 forecasts)
then rebuilds results/comparison_all.csv from the per-signal CSVs.

    python experiments/03b_patch_readout.py
    python experiments/03b_patch_readout.py quick

Run experiment 03 first (this patches its outputs in place).
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
    ESN, KoopmanEDMD, NVAR, QuantumReservoir, Volterra,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
QUICK = "quick" in sys.argv[1:]
HORIZON = 60
SEEDS = [0] if QUICK else [0, 1, 2]

# The five readout methods, as specified in experiment 03 (same labels,
# families, roles, and randomised flag), but now with ridge=None -> closed-loop
# selection.
READOUT_SPECS = [
    ("Linear & feature (Ch5)", "matched", "Volterra", lambda s: Volterra(memory=12), False),
    ("Linear & feature (Ch5)", "matched", "NVAR", lambda s: NVAR(n_delays=10, degree=3), False),
    ("Linear & feature (Ch5)", "matched", "Koopman/EDMD", lambda s: KoopmanEDMD(delay=4, n_rbf=120, seed=s), True),
    ("Reservoir & recurrent (Ch6)", "matched", "ESN", lambda s: ESN(n_reservoir=500, spectral_radius=0.9, leak=0.6, seed=s), True),
    ("Quantum (Ch8)", "matched", "Quantum reservoir", lambda s: QuantumReservoir(seed=s), True),
]

#   key, loader, n_origins, spacing, report_dir   (identical to experiment 03)
DATASETS = [
    ("mackeyglass",  lambda: load_mackey_glass(),        50, 50, False),
    ("laser",        lambda: load_santafe_laser(),       50, 40, False),
    ("enso",         lambda: load_enso(),                50,  5, False),
    ("sp500_logret", lambda: load_sp500("logreturns"),  100, 20, True),
    ("sp500_sqret",  lambda: load_sp500("sqreturns"),   100, 20, False),
]
if QUICK:
    DATASETS = [(k, ld, 10, sp, rd) for (k, ld, _n, sp, rd) in DATASETS]

CSV_FIELDS = ["family", "model", "role", "C_tot", "C_tot_std", "nrmse50",
              "skill_horizon", "dir_acc", "n_seeds"]
ALL_FIELDS = ["dataset", "dataset_name", "family", "model", "role", "C_tot",
              "C_tot_std", "nrmse50", "skill_horizon", "dir_acc", "n_seeds"]


def evaluate_spec(spec, split, n_origins, spacing) -> dict:
    family, role, label, build, randomised = spec
    seeds = SEEDS if randomised else [0]
    curves, ctots, nrmses, horizons, diracc = [], [], [], [], []
    preds0 = truths0 = None
    for s in seeds:
        model = build(s)
        model.fit(split.train)
        r = evaluation.evaluate(model, split.series, split.train_len,
                                HORIZON, n_origins, spacing)
        curves.append(r["capacity_curve"]); ctots.append(r["C_tot"])
        nrmses.append(r["nrmse_short"]); horizons.append(r["skill_horizon"])
        diracc.append(r["dir_acc"])
        if s == seeds[0]:
            preds0, truths0 = r["preds"], r["truths"]
    return {
        "family": family, "role": role, "model": label,
        "C_tot": float(np.mean(ctots)), "C_tot_std": float(np.std(ctots)),
        "nrmse50": float(np.mean(nrmses)),
        "skill_horizon": float(np.mean(horizons)),
        "dir_acc": float(np.mean(diracc)),
        "n_seeds": len(seeds),
        "curve": np.nanmean(curves, axis=0), "preds0": preds0, "truths0": truths0,
    }


def patch_signal(key, loader, n_origins, spacing) -> None:
    split = loader()
    print(f"\n=== {split.name} ===", flush=True)
    new = {}
    for spec in READOUT_SPECS:
        res = evaluate_spec(spec, split, n_origins, spacing)
        new[res["model"]] = res
        print(f"  {res['model']:18s} C_tot={res['C_tot']:6.2f}+-{res['C_tot_std']:5.2f}"
              f"  skill_h={res['skill_horizon']:5.1f}  nrmse50={res['nrmse50']:.3f}",
              flush=True)

    # --- patch the per-signal CSV: replace the 5 readout rows, keep the rest ---
    csv_path = os.path.join(RESULTS_DIR, f"comparison_{key}.csv")
    with open(csv_path) as fh:
        rows = {r["model"]: r for r in csv.DictReader(fh)}
    for label, res in new.items():
        rows[label] = {k: res[k] for k in CSV_FIELDS}
    ordered = sorted(rows.values(), key=lambda r: -float(r["C_tot"]))
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in ordered:
            w.writerow(r)

    # --- patch curves npz (replace the 5 readout curves) ---
    cpath = os.path.join(RESULTS_DIR, f"comparison_{key}_curves.npz")
    curves = {k: v for k, v in np.load(cpath).items()}
    for label, res in new.items():
        curves[label] = res["curve"]
    np.savez(cpath, **curves)

    # --- patch forecast store (replace the 5 readout seed-0 forecasts) ---
    fpath = os.path.join(RESULTS_DIR, f"comparison_{key}_forecasts.npz")
    fc = {k: v for k, v in np.load(fpath).items()}
    for label, res in new.items():
        if res["preds0"] is not None:
            fc[f"{label}|p"] = res["preds0"]
    np.savez(fpath, **fc)


def rebuild_all() -> None:
    out = []
    for key, loader, *_ in DATASETS:
        split_name = {"mackeyglass": "Mackey-Glass", "laser": "Santa Fe laser",
                      "enso": "ENSO (Nino 3.4)", "sp500_logret": "S&P 500 (log-returns)",
                      "sp500_sqret": "S&P 500 (squared returns)"}.get(key, key)
        with open(os.path.join(RESULTS_DIR, f"comparison_{key}.csv")) as fh:
            for r in csv.DictReader(fh):
                r["dataset"] = key
                r["dataset_name"] = split_name
                out.append(r)
    with open(os.path.join(RESULTS_DIR, "comparison_all.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=ALL_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in out:
            w.writerow(r)
    print(f"\nrebuilt comparison_all.csv ({len(out)} rows)")


def main() -> None:
    for key, loader, n_origins, spacing, _rd in DATASETS:
        patch_signal(key, loader, n_origins, spacing)
    rebuild_all()
    print("readout patch complete")


if __name__ == "__main__":
    main()
