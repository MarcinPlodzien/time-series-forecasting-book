"""
models/baselines.py
===================

The baselines every comparison needs.

A forecasting result is only meaningful relative to the trivial alternatives. If
an elaborate model cannot beat "tomorrow equals today", the elaboration bought
nothing. The metric and the baseline are part of the problem statement
(Chapter 4). Two baselines:

    Persistence : predict the last observed value, forever.
    MeanForecast: predict the training mean, forever.

Both ignore the dynamics entirely, so on a chaotic oscillation both score around
nrmse = 1 (no better than the mean). They are the floor every other method must
clear.
"""

from __future__ import annotations

import numpy as np

from ..base import Forecaster


class Persistence(Forecaster):
    """Predict the last seen value for the entire horizon (random-walk forecast)."""

    name = "Persistence"

    def fit(self, train: np.ndarray) -> "Persistence":
        self._last = float(np.asarray(train, float)[-1])
        return self

    def forecast(self, horizon: int, warmup: np.ndarray | None = None) -> np.ndarray:
        last = float(np.asarray(warmup, float)[-1]) if warmup is not None else self._last
        return np.full(horizon, last)


class MeanForecast(Forecaster):
    """Predict the training mean for the entire horizon."""

    name = "Mean"

    def fit(self, train: np.ndarray) -> "MeanForecast":
        self._mean = float(np.mean(train))
        return self

    def forecast(self, horizon: int, warmup: np.ndarray | None = None) -> np.ndarray:
        return np.full(horizon, self._mean)
