# Companion benchmarks

Reproducible code for the empirical appendix of *Time Series Forecasting: A
Dynamical-Systems Approach*. It puts the book's central claim to a concrete test
across **four signals spanning the predictability spectrum**: **characterize the
signal first, match an architecture to that characterization, and expect the
outcome the characterization predicts.**

The four signals are chosen so the framework predicts a *different* result for
each:

| signal | regime | expectation |
|---|---|---|
| Mackey-Glass (synthetic) | known-truth deterministic chaos | nonlinear models win; diagnostics check against ground truth |
| Santa Fe laser | empirical deterministic chaos | nonlinear win, linear models fail at the collapse |
| ENSO Nino 3.4 | noisy quasi-periodic oscillator | modest skill, short horizon, no clear winner |
| S&P 500 returns | near-efficient market | all models ~ baseline; structure only in volatility |

The code is written to be read. Every module maps to a step of the argument.

## Layout

Datasets live in the shared repository data folder, `../data/benchmarks/`
(provenance and checksums in `../data/README.md`); the loader reads them from
there, so the suite needs no data copy of its own.

```
benchmarks/
  tsforecast/               the library
    datasets.py             the forecasting objects (signal + split)
    embedding.py            diagnostics: MI delay, FNN dimension, Lyapunov, AAFT surrogate test
    metrics.py              capacity C(tau), C_tot, skill horizon, NRMSE, directional accuracy
    evaluation.py           rolling-origin closed-loop scoring
    base.py                 one Forecaster interface; standardise + bounded rollout
    models/                 one file per method (baselines, linear_ar, arima, volterra,
                            nvar, koopman_edmd, esn, neural.py, neural_ode, quantum_reservoir)
  experiments/
    00_make_datasets.py     build/verify the datasets (generates Mackey-Glass)
    01_diagnose.py          characterize first: the dynamical fingerprint of each signal
    03_full_comparison.py          every architecture on every signal (persists a forecast store)
    04_skill_scatter.py     predicted-vs-true scatter figures (rho^2 = C(tau)); renders 03's store
    05_summary_heatmap.py   cross-signal C_tot heatmap (methods x signals)
    06_perlead_table.py     per-lead capacity C(tau) at tau = 1, 2, 5, 10
    make_latex_tables.py    emit the chapter tables
    07_qrc_nv_scan.py       QRC virtual-node scaling (nv = 1,2,3,4,5,10), nv-tagged outputs
    08_mamba_depth_scan.py  Mamba depth scaling (1-4 blocks) at fixed ~3k-param budget
  results/                  generated tables and figures
  run_all.py                run everything end to end
```

## Run

```bash
pip install -r requirements.txt
python run_all.py        # offline; data is in ../data/benchmarks/
```

## What it shows

The headline metric is the total forecasting capacity `C_tot = sum_tau C(tau)`,
where `C(tau)` is the squared correlation between the closed-loop prediction and
the truth at lead time `tau`, over many rolling origins. It is the forecasting
analogue of the reservoir-computing memory capacity. Every trainable neural
network is sized to the same ~3000-parameter budget, so the comparison is about
inductive structure, not model size; the reservoir and feature methods (ESN, NVAR,
Koopman, QRC) train only a linear readout and are sized by their substrate.

The characterization (`01_diagnose.py`) splits the signals first: an AAFT
surrogate-data test rejects the linear-stochastic null decisively for Mackey-Glass
and the laser, but not for ENSO or the S&P returns. The forecasts then follow that
split: a clean linear-versus-nonlinear separation on the two chaotic signals, and
near-baseline capacity with coin-flip directional accuracy on the market returns,
whose only forecastable structure is in the squared returns (volatility
clustering).

## Architectures

Every in-regime architecture of Part II: linear AR, ARIMA, Volterra, NVAR,
Koopman/EDMD (Ch5); ESN, RNN, LSTM, GRU (Ch6); MLP, TCN, S4D, **Mamba**,
Transformer, DLinear, latent neural ODE (Ch7); and a small classically-simulated
**6-qubit quantum reservoir** with the full one- and two-body Pauli readout (Ch8).
The quantum reservoir makes no claim of advantage.

Two Part II families are deliberately omitted as out-of-regime for univariate
dissipative chaos: symplectic/Hamiltonian networks (conservative-dynamics prior)
and neural CDEs (multivariate/irregular-input). Benchmarking them here would test
the implementation rather than the principle.

## Hyperparameters

The trainable neural models share one training protocol: Adam at learning rate
3e-3, early stopping on a 15% future-validation split, input window 30. Each is
sized to a shared budget of about 3000 trainable parameters, so the comparison
reflects inductive structure rather than scale. The reservoir and feature methods
(ESN, NVAR, Koopman, QRC) train only a linear readout and are sized by their
substrate.

| Method | Params | Settings |
|---|---|---|
| Persistence | — | parameter-free |
| Mean | — | parameter-free |
| ARIMA | — | order=(12, 0, 1) |
| Linear AR | — | order=25 |
| Volterra | — | memory=12 |
| NVAR | — | n_delays=10, stride=1, degree=3 |
| Koopman/EDMD | — | delay=4, n_rbf=120 |
| ESN | — | n_reservoir=500, spectral_radius=0.9, input_scaling=1.0, leak=0.6, density=0.1 |
| RNN | 2913 | Elman RNN, hidden=52, 1 layer |
| LSTM | 3043 | LSTM, hidden=26, 1 layer |
| GRU | 3001 | GRU, hidden=30, 1 layer |
| MLP | 2921 | 2 hidden layers x40, tanh |
| TCN | 3059 | 22 channels, 3 dilated levels, kernel 3 |
| S4D | 2913 | 16 channels x 16 modes, 2 SSM layers |
| Mamba | 2974 | 2 selective blocks x (17 channels, 8 modes), causal conv k=4 |
| Transformer | 3481 | encoder, d_model=20, 4 heads, 1 layer |
| DLinear | 62 | moving-avg kernel 5 (intrinsically linear) |
| Neural ODE | 2905 | latent ODE, dim=16, RK4 x4 steps, MLP field h=72 |
| Quantum reservoir | — | n=6, t=4.0, n_virtual=5, observables=full, J_scale=1.0, h_x=1.0, h_z=0.5, washout=100, warmup_cap=120 |
