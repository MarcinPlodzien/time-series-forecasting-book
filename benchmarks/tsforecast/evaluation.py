"""
evaluation.py
=============

Rolling-origin evaluation, the standard way to score a forecaster on
one long record.

A single free-running forecast is one realisation. To measure skill we fit each
model once on the training set, then launch many closed-loop forecasts from
different origins spread across the held-out continuation. Each origin uses the
true data up to that point as warm-up (so the model's state is seeded from real
history), then free-runs for `horizon` steps with no further truth. Collecting
the forecasts into a matrix lets us compute the capacity curve C(tau) and the
total capacity C_tot defined in metrics.py.

This assumes the signal is stationary, which the Santa Fe laser is (a single
attractor sampled throughout), so a model trained on the first 1000 points is a
fair predictor at every later origin.
"""

from __future__ import annotations

import numpy as np

from .base import Forecaster
from . import metrics


def rolling_origin_forecasts(
    model: Forecaster,
    series: np.ndarray,
    train_len: int,
    horizon: int,
    n_origins: int = 80,
    spacing: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """Launch closed-loop forecasts from many origins in the continuation.

    The model must already be fitted. Origins start at `train_len` and step by
    `spacing`, staying far enough from the end that a full `horizon` of truth
    exists for comparison.

    Returns (preds, truths), each of shape (n_used_origins, horizon).
    """
    series = np.asarray(series, float)
    preds, truths = [], []
    origin = train_len
    for _ in range(n_origins):
        if origin + horizon > len(series):
            break  # not enough future truth left to score this origin
        warmup = series[:origin]  # all true history up to the origin
        pred = model.forecast(horizon, warmup=warmup)
        preds.append(pred)
        truths.append(series[origin : origin + horizon])
        origin += spacing
    return np.array(preds), np.array(truths)


def evaluate(
    model: Forecaster,
    series: np.ndarray,
    train_len: int,
    horizon: int,
    n_origins: int = 80,
    spacing: int = 30,
) -> dict:
    """Full rolling-origin score set for one fitted model.

    Returns a dict with the capacity curve and its scalar summaries, plus a
    short-lead NRMSE for readers who want a familiar per-sample number.
    """
    preds, truths = rolling_origin_forecasts(
        model, series, train_len, horizon, n_origins, spacing
    )
    C = metrics.capacity_by_lead(preds, truths)
    rho = metrics.pearson_by_lead(preds, truths)
    # Short-lead NRMSE, averaged over origins, over the first min(50, horizon) steps.
    lead = min(50, horizon)
    nrmse_short = float(
        np.mean([metrics.nrmse(t[:lead], p[:lead]) for p, t in zip(preds, truths)])
    )
    # Directional accuracy, averaged over origins. Meaningful for signed objects
    # such as returns; harmless to report elsewhere.
    dir_acc = float(
        np.mean([metrics.directional_accuracy(t, p) for p, t in zip(preds, truths)])
    )
    return {
        "n_origins": len(preds),
        "capacity_curve": C,
        "rho_curve": rho,
        "C_tot": float(np.nansum(C)),
        "skill_horizon": metrics.skill_horizon(preds, truths),
        "nrmse_short": nrmse_short,
        "dir_acc": dir_acc,
        # Raw rolling-origin forecasts, so a single fit can feed both the capacity
        # curve and the predicted-vs-true scatter (experiment 04) without refitting.
        "preds": preds,
        "truths": truths,
    }
