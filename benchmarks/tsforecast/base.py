"""
base.py
=======

The common interface every forecaster in this repo implements.

Keeping one small interface is what makes the cross-method comparison fair: the
experiment code calls `fit` then `forecast` and never needs to know whether the
model is a linear filter, a polynomial, a reservoir, or a neural network. The
differences between methods then live entirely in how they represent the past
and roll themselves forward, the axis along which Chapter 4 organises models
("nonlinear representation of the past, linear readout of the future", and the
architectures that depart from it).

Two prediction modes appear throughout:

    one-step   : given the true recent past, predict the next sample. This
                 mostly measures interpolation.
    free-running (closed-loop, autonomous): seed the model with the end of the
                 training data, then feed its own predictions back in to roll
                 forward with no further truth. This is the demanding test,
                 because errors compound and only a model that captured the
                 dynamics (the vector field, not just a correlation) survives.
                 It is the setting in which a mismatched architecture fails in
                 the diagnosable way Chapter 4 describes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Forecaster(ABC):
    """Abstract base class. Subclasses implement `fit` and `forecast`."""

    #: Short label used in tables and plots. Override per subclass.
    name: str = "forecaster"

    @abstractmethod
    def fit(self, train: np.ndarray) -> "Forecaster":
        """Learn from the 1-D training series. Returns self for chaining."""

    @abstractmethod
    def forecast(self, horizon: int, warmup: np.ndarray | None = None) -> np.ndarray:
        """Autonomously predict `horizon` future samples (closed loop).

        `warmup` is the recent history used to initialise the model's state; it
        defaults to the training series. The returned array has length `horizon`
        and is expressed in the original physical units of the signal.
        """


class Standardiser:
    """Zero-mean, unit-variance scaling, fit on the training data.

    Most of these methods (ridge readouts, reservoirs, neural nets) behave far
    better on standardised inputs, and the regularisation strengths become
    comparable across models. We keep the transform explicit rather than hidden
    inside scikit-learn so each model's inputs are visible.
    """

    def __init__(self, clip_margin: float = 0.25) -> None:
        self.mean_ = 0.0
        self.std_ = 1.0
        # Scaled training range, recorded at fit time. Used to keep closed-loop
        # rollouts physically bounded (see clamp): the laser intensity lives in a
        # finite range, so a free-running prediction that leaves it by more than a
        # small margin is unphysical and is almost always the first sign of the
        # generic closed-loop instability, not a real excursion. Clamping turns a
        # blow-up into a saturated (still wrong) forecast instead of inf/nan.
        self.clip_margin = clip_margin
        self.zmin_ = -np.inf
        self.zmax_ = np.inf

    def fit(self, x: np.ndarray) -> "Standardiser":
        x = np.asarray(x, float)
        self.mean_ = float(np.mean(x))
        self.std_ = float(np.std(x)) or 1.0
        z = (x - self.mean_) / self.std_
        span = float(z.max() - z.min())
        self.zmin_ = float(z.min()) - self.clip_margin * span
        self.zmax_ = float(z.max()) + self.clip_margin * span
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, float) - self.mean_) / self.std_

    def inverse(self, z: np.ndarray) -> np.ndarray:
        return np.asarray(z, float) * self.std_ + self.mean_

    def clamp(self, z: float) -> float:
        """Clip a scaled value to the (padded) training range; tame blow-ups."""
        if not np.isfinite(z):
            # A non-finite step means the loop has already diverged; pin it to
            # the nearest bound so the rest of the rollout stays finite.
            return self.zmax_
        return float(min(max(z, self.zmin_), self.zmax_))
