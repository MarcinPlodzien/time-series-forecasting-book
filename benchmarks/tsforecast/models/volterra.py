"""
models/volterra.py
=================

Second-order Volterra filter, the classical nonlinear-systems route to forecasting
(Chapter 5). The next sample is a quadratic functional of a finite past window,

    x_hat_{t+1} = w_0
                  + sum_i        h1_i  x_{t-i}                 (linear kernel)
                  + sum_{i<=j}   h2_ij x_{t-i} x_{t-j}         (quadratic kernel)

with the kernels h1, h2 fit by ridge regression. This is the same "nonlinear
representation of the past, linear readout" template as NVAR (Chapter 4); indeed
the degree-2 Volterra map and the degree-2 NVAR feature map are the same object.
We include it separately and write it in explicit kernel form because it is a
historically and conceptually distinct method, and because seeing the quadratic
kernel written out shows why the model can represent the folding a linear filter
cannot. It is therefore a matched model for low-dimensional chaos, limited only by
the combinatorial growth of the quadratic kernel with window size.
"""

from __future__ import annotations

import itertools

import numpy as np

from ..base import Forecaster, Standardiser
from ..readout import ridge_solve, select_ridge_closedloop


class Volterra(Forecaster):
    """Ridge-fit second-order Volterra filter with bounded closed-loop rollout."""

    def __init__(self, memory: int = 8, ridge: float | None = None) -> None:
        """`memory` is the window length (number of past samples in each kernel);
        `ridge` regularises the readout. None (default) selects it on a
        validation split; a float fixes it."""
        self.memory = memory
        self.ridge = ridge
        self.name = f"Volterra (M={memory})"
        self._scaler = Standardiser()
        self._W: np.ndarray | None = None
        # Precompute the index pairs (i<=j) of the quadratic kernel once.
        self._pairs = list(itertools.combinations_with_replacement(range(memory), 2))

    def _features(self, window: np.ndarray) -> np.ndarray:
        """Map a window [x_{t-M+1..t}] to [1, linear terms, quadratic terms]."""
        lin = window
        quad = np.array([window[i] * window[j] for (i, j) in self._pairs])
        return np.concatenate([[1.0], lin, quad])

    def fit(self, train: np.ndarray) -> "Volterra":
        train = np.asarray(train, float)
        if self.ridge is None:  # pick ridge by closed-loop capacity on a val tail
            self._eff_ridge = select_ridge_closedloop(
                lambda r: Volterra(memory=self.memory, ridge=r), train)
        else:
            self._eff_ridge = float(self.ridge)
        z = self._scaler.fit(train).transform(train)
        feats, targets = [], []
        # Predict z[t] from the `memory` samples immediately before it. The window
        # must end at z[t-1] with the target z[t]: no gap, or the model would learn
        # a two-step map and diverge in the one-step closed loop.
        for t in range(self.memory, len(z)):
            feats.append(self._features(z[t - self.memory : t]))
            targets.append(z[t])
        Phi, y = np.array(feats), np.array(targets)
        self._W = ridge_solve(Phi, y, self._eff_ridge)
        self._ridge_used = self._eff_ridge
        return self

    def forecast(self, horizon: int, warmup: np.ndarray | None = None) -> np.ndarray:
        if self._W is None:
            raise RuntimeError("call fit first")
        warmup = warmup if warmup is not None else np.zeros(self.memory)
        hist = list(self._scaler.transform(np.asarray(warmup, float)))
        preds = []
        for _ in range(horizon):
            window = np.array(hist[-self.memory :])
            nxt = self._scaler.clamp(float(self._features(window) @ self._W))
            preds.append(nxt)
            hist.append(nxt)
        return self._scaler.inverse(np.array(preds))
