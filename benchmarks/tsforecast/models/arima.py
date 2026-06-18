"""
models/arima.py
==============

ARIMA, the classical linear forecaster (Box--Jenkins), included because it is the
default a statistician reaches for and serves as a linear reference point
(Chapter 5). On the laser it is mismatched for the same reason the plain AR model
is: ARIMA is a linear stochastic model (autoregression, optional
differencing, moving-average noise), so it has no mechanism to fold a deterministic
nonlinear attractor. In closed loop it relaxes to a damped linear oscillation and
cannot track the growing-amplitude chaotic build-up.

We fit once on the training series and reuse the fitted coefficients at each
rolling-origin warm-up via statsmodels' `apply`, which re-applies the same model
to new data without re-estimating, so the evaluation stays fast and the model
fixed.
"""

from __future__ import annotations

import warnings

import numpy as np

from ..base import Forecaster

try:
    from statsmodels.tsa.arima.model import ARIMA as _SMARIMA

    SM_OK = True
except Exception:  # pragma: no cover
    SM_OK = False


class ARIMA(Forecaster):
    """Thin wrapper over statsmodels ARIMA with the common Forecaster API."""

    def __init__(self, order: tuple[int, int, int] = (12, 0, 1)) -> None:
        """`order` is (p, d, q): AR lags, differencing, MA lags. d=0 because the
        laser is stationary; a moderate p lets the linear model chase the
        oscillation as well as a linear model can."""
        if not SM_OK:
            raise RuntimeError("statsmodels is required for ARIMA")
        self.order = order
        self.name = f"ARIMA{order}"

    def fit(self, train: np.ndarray) -> "ARIMA":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # silence convergence chatter
            self._res = _SMARIMA(np.asarray(train, float), order=self.order).fit()
        return self

    def forecast(self, horizon: int, warmup: np.ndarray | None = None) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if warmup is None:
                fc = self._res.forecast(horizon)
            else:
                # Re-apply the fixed fitted parameters to the new history, then
                # forecast. This is ARIMA's native multi-step prediction, which is
                # already a closed-loop linear recursion under the hood.
                fc = self._res.apply(np.asarray(warmup, float)).forecast(horizon)
        return np.asarray(fc, float)
