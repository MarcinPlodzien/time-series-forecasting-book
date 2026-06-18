"""
models/linear_ar.py
===================

Linear autoregression, AR(p): the mismatched model.

    x_hat_t = c + sum_{k=1..p} a_k x_{t-k}

This is the linear-past, linear-future corner of the taxonomy (Chapter 4). It is
the natural model for a signal whose dynamics are (nearly) linear: damped or
sustained oscillations, narrow-band processes, ARMA-type stochastic structure.

On the Santa Fe laser it is mismatched by construction, and the diagnosis
predicts how it must fail. The laser's defining event is the intensity collapse,
which is produced by the nonlinear folding of the Lorenz--Haken attractor. A
linear recursion has only fixed points and exponential/oscillatory eigenmodes
(the roots of its characteristic polynomial); it has no mechanism to fold the
trajectory back. Free-running, it therefore settles into a sustained linear
oscillation and sails straight through the collapse it cannot represent. That is
the diagnosable failure mode, and the experiment is designed to show it.

We fit by ordinary least squares (the Yule--Walker / normal-equations solution),
which is transparent and dependency-light.
"""

from __future__ import annotations

import numpy as np

from ..base import Forecaster, Standardiser
from ..embedding import delay_embed


class LinearAR(Forecaster):
    """Ordinary-least-squares AR(p) with closed-loop rollout."""

    def __init__(self, order: int = 20) -> None:
        """`order` is p, the number of past samples in the linear recursion."""
        self.order = order
        self.name = f"Linear AR(p={order})"
        self._scaler = Standardiser()
        self._coef: np.ndarray | None = None  # [bias, a_1, ..., a_p]

    def fit(self, train: np.ndarray) -> "LinearAR":
        z = self._scaler.fit(train).transform(train)
        # Design matrix of lagged values. delay_embed gives rows
        # [z_{t}, z_{t-1}, ..., z_{t-(p-1)}]; we use them to predict z_{t+1}.
        X = delay_embed(z, dim=self.order, tau=1)
        # Target is the sample immediately after each embedded window.
        y = z[self.order :]
        X = X[: len(y)]
        # Prepend a bias column, then solve the normal equations by lstsq.
        Xb = np.column_stack([np.ones(len(X)), X])
        self._coef, *_ = np.linalg.lstsq(Xb, y, rcond=None)
        return self

    def _step(self, window: np.ndarray) -> float:
        """One-step prediction from the most recent `order` (scaled) values.

        `window` is ordered oldest..newest; the design row is newest..oldest to
        match the embedding convention used at fit time.
        """
        row = np.concatenate([[1.0], window[::-1]])
        return float(row @ self._coef)

    def forecast(self, horizon: int, warmup: np.ndarray | None = None) -> np.ndarray:
        if self._coef is None:
            raise RuntimeError("call fit first")
        warmup = warmup if warmup is not None else self._scaler.inverse(np.zeros(self.order))
        # Seed the rolling window with the last `order` scaled observations.
        window = list(self._scaler.transform(np.asarray(warmup, float))[-self.order :])
        preds = []
        for _ in range(horizon):
            nxt = self._scaler.clamp(self._step(np.array(window)))
            preds.append(nxt)
            # Closed loop: the prediction becomes the next input.
            window.append(nxt)
            window.pop(0)
        return self._scaler.inverse(np.array(preds))
