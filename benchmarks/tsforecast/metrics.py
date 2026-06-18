"""
metrics.py
==========

Scoring of forecasts.

The metric is part of the forecasting problem, not an afterthought (Chapter 4).
For a chaotic signal there are two qualitatively different questions:

    1. "How close is the forecast, sample by sample?"  -> NRMSE.
    2. "For how long does the forecast stay on the true trajectory before
        exponential divergence pulls it away?"          -> valid prediction time.

The second is usually the more informative summary for chaos, because beyond the
Lyapunov horizon *no* model can stay close (Chapter 1), so a small per-sample
error averaged over a long window says more about the window than the model.

All functions take 1-D arrays of equal length and assume they are aligned in
time (same sampling, same start).
"""

from __future__ import annotations

import numpy as np


def rmse(truth: np.ndarray, pred: np.ndarray) -> float:
    """Plain root-mean-square error, in the signal's physical units."""
    truth, pred = np.asarray(truth, float), np.asarray(pred, float)
    return float(np.sqrt(np.mean((truth - pred) ** 2)))


def nrmse(truth: np.ndarray, pred: np.ndarray) -> float:
    """Normalised RMSE: RMSE divided by the standard deviation of the truth.

    The normalisation makes the number scale-free and interpretable:

        nrmse = 0   -> perfect,
        nrmse = 1   -> no better than predicting the constant mean of the truth,
        nrmse > 1   -> actively worse than the mean (common when a free-running
                       model diverges and amplifies).

    This is the standard reservoir-computing figure of merit, which is why we
    use it as the headline per-sample score.
    """
    truth, pred = np.asarray(truth, float), np.asarray(pred, float)
    denom = np.std(truth)
    if denom == 0:
        # A constant target has no scale to normalise by; fall back to RMSE.
        return rmse(truth, pred)
    return rmse(truth, pred) / denom


def valid_prediction_time(
    truth: np.ndarray,
    pred: np.ndarray,
    threshold: float = 0.4,
) -> int:
    """Number of leading steps for which the forecast stays "valid".

    We walk forward and stop at the first step whose *running* normalised error
    exceeds `threshold`. This estimates the practical prediction horizon, the
    analogue of the Lyapunov time that bounds predictability (Chapters 1--2).

    The error is normalised by the standard deviation of the whole truth window,
    so the threshold is a fraction of the signal's natural amplitude. The default
    0.4 is a common choice in the chaos-forecasting literature; it is a knob, not
    a law, and the experiments report it explicitly.

    Returns the count of valid steps (0 means the very first step already broke
    the threshold; len(truth) means the forecast never broke it).
    """
    truth, pred = np.asarray(truth, float), np.asarray(pred, float)
    scale = np.std(truth)
    if scale == 0:
        scale = 1.0
    # Pointwise normalised absolute deviation.
    err = np.abs(truth - pred) / scale
    # First index where the threshold is exceeded.
    breached = np.where(err > threshold)[0]
    return int(breached[0]) if breached.size else int(len(truth))


def directional_accuracy(truth: np.ndarray, pred: np.ndarray) -> float:
    """Fraction of steps where the forecast gets the sign of the move right.

    A useful companion to the magnitude metrics on near-efficient signals such as
    daily returns: 0.5 is a coin flip, and anything reliably above 0.5 would be
    exploitable. Steps where either value is zero (a flat predict-the-mean
    forecast, or an exactly zero move) are excluded, so a model that always
    predicts ~0 earns no directional credit.
    """
    truth, pred = np.asarray(truth, float), np.asarray(pred, float)
    nonzero = (truth != 0) & (pred != 0)
    if not np.any(nonzero):
        return 0.5
    hits = np.sign(truth[nonzero]) == np.sign(pred[nonzero])
    return float(np.mean(hits))


def summary(truth: np.ndarray, pred: np.ndarray, threshold: float = 0.4) -> dict:
    """Bundle the headline scores for a single forecast into a dict."""
    return {
        "nrmse": nrmse(truth, pred),
        "rmse": rmse(truth, pred),
        "valid_steps": valid_prediction_time(truth, pred, threshold),
    }


# ---------------------------------------------------------------------------
# Forecasting capacity: the reservoir-computing way of scoring skill vs horizon.
#
# A single closed-loop trajectory is one sample of a chaotic process, so its
# pointwise error is noisy. The cleaner question, borrowed from the reservoir
# memory-capacity literature (Chapters 6 and 8), is statistical: across many
# forecast origins, how strongly does the prediction at lead time tau correlate
# with the truth at lead time tau?
#
#     rho(tau) : Pearson correlation between predicted and true values at lead tau,
#                taken over an ensemble of forecast origins.
#     C(tau)   : rho(tau)^2, the coefficient of determination. This is the
#                per-delay capacity of the reservoir-computing memory-capacity task,
#                bounded in [0, 1]: 1 = the lead-tau value is fully predictable,
#                0 = no linear predictability remains.
#     C_tot    : sum_tau C(tau), the total forecasting capacity, a single scalar in
#                units of "effective predictable steps". It is the direct analogue
#                of the reservoir memory capacity MC = sum_tau C(tau).
#
# The curve C(tau) shows the shape of a method's skill decay (and where its failure
# sets in); the scalar C_tot ranks methods by total predictive skill.
# ---------------------------------------------------------------------------
def pearson_by_lead(preds: np.ndarray, truths: np.ndarray) -> np.ndarray:
    """Pearson correlation rho(tau) at each lead time, over forecast origins.

    `preds` and `truths` are 2-D arrays of shape (n_origins, horizon): row i is one
    closed-loop forecast launched from origin i and the matching true continuation.
    Column tau is the ensemble of (prediction, truth) pairs at lead time tau+1.
    """
    preds = np.asarray(preds, float)
    truths = np.asarray(truths, float)
    horizon = preds.shape[1]
    rho = np.full(horizon, np.nan)
    for tau in range(horizon):
        p, y = preds[:, tau], truths[:, tau]
        # Correlation is undefined if either column is constant across origins.
        if np.std(p) < 1e-12 or np.std(y) < 1e-12:
            rho[tau] = 0.0
            continue
        rho[tau] = float(np.corrcoef(p, y)[0, 1])
    return rho


def capacity_by_lead(preds: np.ndarray, truths: np.ndarray) -> np.ndarray:
    """Per-lead forecasting capacity C(tau) = rho(tau)^2, in [0, 1]."""
    rho = pearson_by_lead(preds, truths)
    return rho ** 2


def total_forecasting_capacity(preds: np.ndarray, truths: np.ndarray) -> float:
    """Total forecasting capacity C_tot = sum_tau C(tau) (effective steps)."""
    return float(np.nansum(capacity_by_lead(preds, truths)))


def skill_horizon(preds: np.ndarray, truths: np.ndarray, level: float = 0.5) -> int:
    """First lead time at which capacity C(tau) falls below `level`.

    A robust, ensemble-based predictability horizon: the analogue of the Lyapunov
    time read straight off the forecast-skill curve. Default level 0.5 marks where
    half the variance at that lead is still predictable.
    """
    C = capacity_by_lead(preds, truths)
    below = np.where(C < level)[0]
    return int(below[0]) if below.size else int(len(C))
