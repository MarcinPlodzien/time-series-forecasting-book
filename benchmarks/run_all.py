"""
run_all.py
==========

Run the full benchmark suite end to end and regenerate everything in results/.
This is the single entry point referenced by the appendix.

    python run_all.py

Each experiment is a self-contained, importable module; we call them in order so
the diagnosis is printed before the forecasts that it justifies.
"""

from __future__ import annotations

import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS = [
    "experiments/00_make_datasets.py",   # build/verify the datasets
    "experiments/01_diagnose.py",        # characterize first (diagnostics + surrogate)
    "experiments/03_full_comparison.py",        # full architecture comparison on every signal
    "experiments/04_skill_scatter.py",   # predicted-vs-true scatter (reads 03's forecast store)
    "experiments/05_summary_heatmap.py", # cross-signal C_tot heatmap
    "experiments/06_perlead_table.py",   # C(tau) at tau = 1, 2, 5, 10
    "experiments/make_latex_tables.py",  # emit the chapter tables
    # The two scaling studies are long (especially 07 at nv=10 and 08 across
    # depths); they run last so the core benchmark completes first.
    "experiments/07_qrc_nv_scan.py",     # QRC virtual-node scaling
    "experiments/08_mamba_depth_scan.py",# Mamba depth scaling at fixed budget
]


def main() -> None:
    for rel in EXPERIMENTS:
        path = os.path.join(HERE, rel)
        print("\n" + "#" * 72)
        print(f"# running {rel}")
        print("#" * 72)
        # run_path executes the script as __main__, so its `if __name__` block fires.
        runpy.run_path(path, run_name="__main__")


if __name__ == "__main__":
    sys.exit(main())
