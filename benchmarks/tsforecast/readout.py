"""
readout.py
=========

Shared linear-readout solver for the feature/reservoir methods (ESN, NVAR,
Volterra, Koopman/EDMD, quantum reservoir). All of them end in the same step: a
ridge regression from a feature matrix to a target. The ridge strength is a real
hyperparameter, so rather than hand-set a fixed constant per method we select it
on a held-out validation tail of the training data over a logarithmic grid, then
refit on the whole training set with the chosen value. The validation tail is the
most recent slice, mirroring the forecasting task (predict the near future from
the past), and is never the test continuation.

This keeps the comparison fair: every linear-readout method gets its best
regularisation by the same rule, so a difference in capacity reflects the
substrate's features, not the choice of ridge constant.
"""

from __future__ import annotations

import numpy as np

# Logarithmic ridge grid spanning the regimes the different feature matrices need
# (ESN/QRC like tiny ridge; Volterra likes a large one). Three points per decade
# over 1e-8..1e3 so the selected value is resolved to about a third of a decade.
RIDGE_GRID = np.logspace(-8, 3, 34)

# Coarser grid for the (more expensive) closed-loop selection, one point per decade.
# Spans 1e-8..1e2: the feature methods can want a large ridge (rollout stability),
# while the quantum reservoir wants a very small one.
RIDGE_GRID_COARSE = np.logspace(-8, 2, 11)


def ridge_solve(Phi: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    """Closed-form ridge regression W = (Phi^T Phi + ridge I)^-1 Phi^T y.

    `y` may be 1-D (a scalar readout) or 2-D (e.g. the Koopman operator fit).
    """
    G = Phi.T @ Phi + ridge * np.eye(Phi.shape[1])
    return np.linalg.solve(G, Phi.T @ y)


def fit_ridge_cv(
    Phi: np.ndarray,
    y: np.ndarray,
    grid: np.ndarray | None = None,
    val_frac: float = 0.2,
    min_val: int = 20,
) -> tuple[np.ndarray, float]:
    """Select ridge by one-step MSE on a chronological validation tail.

    Splits (Phi, y) so the last `val_frac` (at least `min_val` rows) is the
    validation set, scores each ridge on it, picks the best, and refits on all of
    (Phi, y). Returns (W, chosen_ridge). Falls back to a small fixed ridge when
    there is too little data to split.
    """
    grid = RIDGE_GRID if grid is None else grid
    n = len(y)
    n_val = max(min_val, int(round(val_frac * n)))
    if n_val >= n - 1:  # not enough to split; use a small, safe ridge
        r = 1e-6
        return ridge_solve(Phi, y, r), r
    Phi_fit, y_fit = Phi[:-n_val], y[:-n_val]
    Phi_val, y_val = Phi[-n_val:], y[-n_val:]
    best_r, best_mse = None, None
    for r in grid:
        W = ridge_solve(Phi_fit, y_fit, r)
        mse = float(np.mean((Phi_val @ W - y_val) ** 2))
        if best_mse is None or mse < best_mse:
            best_mse, best_r = mse, float(r)
    return ridge_solve(Phi, y, best_r), best_r


def solve_readout(Phi: np.ndarray, y: np.ndarray, ridge) -> tuple[np.ndarray, float]:
    """Dispatch: ridge is None -> validation-select it; a float -> use it fixed.

    Returns (W, ridge_used) so the caller can report the value actually used.
    """
    if ridge is None:
        return fit_ridge_cv(Phi, y)
    return ridge_solve(Phi, y, ridge), float(ridge)


def select_ridge_closedloop(
    make_model,
    train: np.ndarray,
    grid: np.ndarray | None = None,
    val_frac: float = 0.2,
    n_origins: int = 8,
    horizon: int = 30,
    spacing: int = 8,
) -> float:
    """Choose ridge by closed-loop capacity on a validation tail of the training set.

    The benchmark scores closed-loop forecasting capacity C_tot, so the readout
    ridge must be chosen for rollout stability, not one-step fit: a one-step-optimal
    (tiny) ridge can destabilise the free-running rollout of the feature methods.
    For each ridge we fit a fresh model (via `make_model(ridge)`) on the first
    1-val_frac of the training series, run rolling-origin closed-loop forecasts over
    the held-out tail, and keep the ridge with the highest validation C_tot. The
    caller then refits on the full training set with the chosen ridge.

    `make_model(ridge)` must return an unfitted model with that fixed ridge (a float,
    so its own fit does not re-enter selection).
    """
    from . import evaluation, metrics  # local import avoids an import cycle

    grid = RIDGE_GRID_COARSE if grid is None else grid
    train = np.asarray(train, float)
    n = len(train)
    cut = int(round(n * (1.0 - val_frac)))
    # Need enough validation tail for at least one scored origin; else fall back.
    if cut < 50 or (n - cut) < horizon + 5:
        return 1e-2
    best_r, best_c = float(grid[0]), -np.inf
    for r in grid:
        try:
            m = make_model(float(r))
            m.fit(train[:cut])
            preds, truths = evaluation.rolling_origin_forecasts(
                m, train, cut, horizon, n_origins, spacing
            )
            c = float(np.nansum(metrics.capacity_by_lead(preds, truths)))
        except Exception:
            c = -np.inf  # an unstable ridge that blows up loses
        if np.isfinite(c) and c > best_c:
            best_c, best_r = c, float(r)
    return best_r
