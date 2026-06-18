"""
03c_patch_s4d.py
===============

Re-run only the S4D model, after switching it to the HiPPO-LegS (S4D-LegS)
initialisation, and patch its entries into the existing full comparison outputs. S4D is
the only model affected by the initialisation change, so the rest of the benchmark
is left untouched. Mirrors 03b_patch_readout.py.

    python experiments/03c_patch_s4d.py
    python experiments/03c_patch_s4d.py quick
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
VARIANT = "random" if "random" in sys.argv[1:] else "hippo"
HORIZON = 60
SEEDS = [0] if QUICK else [0, 1, 2]

FAMILY, ROLE = "Modern sequence (Ch7)", "matched"
MODEL = "S4D (random)" if VARIANT == "random" else "S4D"
_BUILD = neural.make_s4d_random if VARIANT == "random" else neural.make_s4d

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

CSV_FIELDS = ["family", "model", "role", "C_tot", "C_tot_std", "nrmse50",
              "skill_horizon", "dir_acc", "n_seeds"]
ALL_FIELDS = ["dataset", "dataset_name", "family", "model", "role", "C_tot",
              "C_tot_std", "nrmse50", "skill_horizon", "dir_acc", "n_seeds"]
NAMES = {"mackeyglass": "Mackey-Glass", "laser": "Santa Fe laser",
         "enso": "ENSO (Nino 3.4)", "sp500_logret": "S&P 500 (log-returns)",
         "sp500_sqret": "S&P 500 (squared returns)"}


def run_s4d(split, n_origins, spacing) -> dict:
    curves, ctots, nrmses, horizons, diracc = [], [], [], [], []
    preds0 = truths0 = None
    for s in SEEDS:
        model = _BUILD(seed=s)
        model.fit(split.train)
        r = evaluation.evaluate(model, split.series, split.train_len,
                                HORIZON, n_origins, spacing)
        curves.append(r["capacity_curve"]); ctots.append(r["C_tot"])
        nrmses.append(r["nrmse_short"]); horizons.append(r["skill_horizon"])
        diracc.append(r["dir_acc"])
        if s == SEEDS[0]:
            preds0, truths0 = r["preds"], r["truths"]
    return {
        "family": FAMILY, "model": MODEL, "role": ROLE,
        "C_tot": float(np.mean(ctots)), "C_tot_std": float(np.std(ctots)),
        "nrmse50": float(np.mean(nrmses)),
        "skill_horizon": float(np.mean(horizons)),
        "dir_acc": float(np.mean(diracc)), "n_seeds": len(SEEDS),
        "curve": np.nanmean(curves, axis=0), "preds0": preds0,
    }


def main() -> None:
    for key, loader, n_origins, spacing in DATASETS:
        split = loader()
        res = run_s4d(split, n_origins, spacing)
        print(f"  {NAMES[key]:24s} S4D  C_tot={res['C_tot']:6.2f}+-{res['C_tot_std']:5.2f}"
              f"  skill_h={res['skill_horizon']:5.1f}", flush=True)

        # patch the per-signal CSV row
        csv_path = os.path.join(RESULTS_DIR, f"comparison_{key}.csv")
        with open(csv_path) as fh:
            rows = {r["model"]: r for r in csv.DictReader(fh)}
        rows[MODEL] = {k: res[k] for k in CSV_FIELDS}
        ordered = sorted(rows.values(), key=lambda r: -float(r["C_tot"]))
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
            w.writeheader()
            for r in ordered:
                w.writerow(r)

        # patch the capacity curve
        cpath = os.path.join(RESULTS_DIR, f"comparison_{key}_curves.npz")
        curves = {k: v for k, v in np.load(cpath).items()}
        curves[MODEL] = res["curve"]
        np.savez(cpath, **curves)

        # patch the seed-0 forecast in the store
        fpath = os.path.join(RESULTS_DIR, f"comparison_{key}_forecasts.npz")
        fc = {k: v for k, v in np.load(fpath).items()}
        if res["preds0"] is not None:
            fc[f"{MODEL}|p"] = res["preds0"]
        np.savez(fpath, **fc)

    # rebuild the combined table
    out = []
    for key, *_ in DATASETS:
        with open(os.path.join(RESULTS_DIR, f"comparison_{key}.csv")) as fh:
            for r in csv.DictReader(fh):
                r["dataset"] = key
                r["dataset_name"] = NAMES[key]
                out.append(r)
    with open(os.path.join(RESULTS_DIR, "comparison_all.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=ALL_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(out)
    print("S4D patch complete")


if __name__ == "__main__":
    main()
