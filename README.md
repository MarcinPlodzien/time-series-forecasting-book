# Time Series Forecasting: A Dynamical-Systems Approach

*Companion code — dynamical-systems diagnostics and an equal-budget forecasting benchmark.*

This repository reproduces every computational figure and number in the book. It
has two parts, mirroring the book's two halves:

```
codes/
  data/                  all input series, in one place (see data/README.md)
  case_studies/          Part I — the diagnostic case studies (Chapters 1–3)
    common/plotting.py   shared 2×2 signal figure (full | zoom | ACF | PSD)
    <system>/            one folder per system: dynamics_*.py + plot_fig_*.py
  benchmarks/            Part II — the forecasting benchmark (Chapter 9)
    tsforecast/          the library (datasets, metrics, evaluation, models)
    experiments/         the runnable experiments (00–08, make_latex_tables)
  requirements.txt
  LICENSE
```

Everything runs offline on a CPU; no GPU is required.

## Quick start

```bash
pip install -r requirements.txt

# Part I: regenerate a case-study figure (Chapter 3)
cd case_studies/santafe_laser && python plot_fig_santafe.py
#   -> ../../figures/santafe_laser/fig_santafe.png

# Part II: run the forecasting benchmark (Chapter 9)
cd benchmarks && python run_all.py        # ~1–1.5 h on CPU for the core run
```

## What maps to what

| Book part | Code | Output |
|---|---|---|
| Ch 1–3 case studies | `case_studies/<system>/dynamics_*.py` (D_KY, Lyapunov, embedding) and `plot_fig_*.py` (the signal figure) | `figures/<system>/` |
| Ch 9 benchmark | `benchmarks/experiments/03_full_zoo.py` | per-signal capacity tables + figures |
| Ch 9 diagnostics | `benchmarks/experiments/01_diagnose.py` | `tab:appendix_diagnostics` |
| Ch 9 scatter | `benchmarks/experiments/04_skill_scatter.py` | predicted-vs-true panels |
| Ch 9 QRC scaling | `benchmarks/experiments/07_qrc_nv_scan.py` | virtual-node scaling |
| Ch 9 Mamba depth | `benchmarks/experiments/08_mamba_depth_scan.py` | depth-at-fixed-budget scaling |

The case studies (Part I) characterize a signal; the benchmark (Part II) tests
whether that characterization predicts which forecasting architecture wins and how
the mismatched ones fail. See `benchmarks/README.md` for the benchmark details.

## Data

All input series live under `data/` with provenance, checksums, and licensing in
`data/README.md`. Public-domain and freely redistributable series are bundled;
the S&P 500 daily close is fetched by a small script (`data/fetch_sp500.py`)
rather than committed, because its redistribution terms are unclear.
