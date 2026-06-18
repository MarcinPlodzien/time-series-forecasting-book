"""
models/esn.py
=============

Echo State Network (ESN), the canonical reservoir computer: the second matched
model, and the subject of the reservoir chapter (Chapter 6).

The idea (Jaeger 2001; Maass et al. 2002):

    * A large, fixed, random recurrent network (the "reservoir") is driven by
      the input. Its high-dimensional transient state is a rich nonlinear,
      fading-memory encoding of the recent input history.
    * Only a linear readout from reservoir state to output is trained.

So the reservoir supplies the nonlinear representation of the past and the
trained part is again just a linear readout (Chapter 4). The crucial design
condition is the Echo State Property: the reservoir must forget its initial
state, which is arranged by scaling the recurrent weight matrix so its spectral
radius is below (or near) 1. Tuning the reservoir "just below the edge of chaos"
gives long-but-fading memory, which is what a low-dimensional chaotic signal
with short autocorrelation needs.

Because the reservoir state carries memory, free-running prediction is natural:
we drive the reservoir with the warm-up history, then close the loop by feeding
the readout's own output back as the next input.

The random reservoir means results depend on the seed; we expose `seed` so the
experiment can report an average over several reservoirs rather than a single
draw.
"""

from __future__ import annotations

import numpy as np

from ..base import Forecaster, Standardiser
from ..readout import ridge_solve, select_ridge_closedloop


class ESN(Forecaster):
    """Leaky-integrator echo state network with a ridge readout."""

    def __init__(
        self,
        n_reservoir: int = 400,
        spectral_radius: float = 0.9,
        input_scaling: float = 1.0,
        leak: float = 1.0,
        ridge: float | None = None,
        density: float = 0.1,
        seed: int = 0,
    ) -> None:
        """
        n_reservoir     : number of reservoir units (state dimension).
        spectral_radius : largest |eigenvalue| of the recurrent matrix; the
                          memory/stability knob. < 1 guarantees the echo state
                          property for these inputs; ~0.9 is "below the edge".
        input_scaling   : gain on the input weights (how hard the input drives).
        leak            : leaky-integration rate in (0,1]; 1 = no leak. Lower
                          values slow the reservoir for smoother signals.
        ridge           : Tikhonov regularisation of the linear readout. None
                          (default) selects it on a validation split (see
                          readout.solve_readout); a float fixes it.
        density         : fraction of nonzero recurrent connections (sparse).
        seed            : RNG seed for the (fixed) random reservoir.
        """
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        self.leak = leak
        self.ridge = ridge
        self.density = density
        self.seed = seed
        self.name = f"ESN (N={n_reservoir}, rho={spectral_radius})"
        self._scaler = Standardiser()

    # ---- reservoir construction -------------------------------------------
    def _build_reservoir(self) -> None:
        rng = np.random.default_rng(self.seed)
        n = self.n_reservoir
        # Sparse random recurrent matrix.
        W = rng.standard_normal((n, n)) * (rng.random((n, n)) < self.density)
        # Rescale so the spectral radius (largest |eigenvalue|) hits the target.
        eigs = np.linalg.eigvals(W)
        radius = np.max(np.abs(eigs))
        if radius > 0:
            W *= self.spectral_radius / radius
        self._W = W
        # Input weights (scalar input -> reservoir) and a bias.
        self._Win = self.input_scaling * rng.uniform(-1, 1, size=(n, 1))
        self._bias = self.input_scaling * rng.uniform(-1, 1, size=(n,))

    def _update(self, state: np.ndarray, u: float) -> np.ndarray:
        """One leaky-integrator reservoir step driven by scalar input `u`."""
        pre = self._W @ state + (self._Win[:, 0] * u) + self._bias
        target = np.tanh(pre)
        # Leaky integration blends old state with the new activation.
        return (1.0 - self.leak) * state + self.leak * target

    # ---- training ----------------------------------------------------------
    def fit(self, train: np.ndarray, washout: int = 100) -> "ESN":
        train = np.asarray(train, float)
        if self.ridge is None:  # pick ridge by closed-loop capacity on a val tail
            eff = select_ridge_closedloop(
                lambda r: ESN(n_reservoir=self.n_reservoir,
                              spectral_radius=self.spectral_radius,
                              input_scaling=self.input_scaling, leak=self.leak,
                              ridge=r, density=self.density, seed=self.seed),
                train)
        else:
            eff = float(self.ridge)
        self._build_reservoir()
        z = self._scaler.fit(train).transform(train)
        state = np.zeros(self.n_reservoir)
        states, targets = [], []
        # Drive the reservoir with the true series; collect (state -> next) pairs.
        # The first `washout` steps are discarded so the readout only sees states
        # that have forgotten the arbitrary zero initial condition (echo state).
        for t in range(len(z) - 1):
            state = self._update(state, z[t])
            if t >= washout:
                # Augment with a constant and the current input (standard trick
                # that lets the readout reproduce affine and direct terms).
                states.append(np.concatenate([[1.0, z[t]], state]))
                targets.append(z[t + 1])
        S = np.array(states)
        y = np.array(targets)
        # Ridge readout, with the ridge chosen above.
        self._Wout = ridge_solve(S, y, eff)
        self._ridge_used = eff
        # Remember the final driven state and last input to seed free-running.
        self._final_state = state
        self._last_input = z[-1]
        return self

    # ---- closed-loop rollout ----------------------------------------------
    def forecast(self, horizon: int, warmup: np.ndarray | None = None) -> np.ndarray:
        state = self._final_state.copy()
        u = self._last_input
        if warmup is not None:
            # Re-drive the reservoir along the supplied history to set its state.
            zw = self._scaler.transform(np.asarray(warmup, float))
            state = np.zeros(self.n_reservoir)
            for val in zw:
                state = self._update(state, val)
            u = zw[-1]
        preds = []
        for _ in range(horizon):
            feat = np.concatenate([[1.0, u], state])
            u = self._scaler.clamp(float(feat @ self._Wout))  # readout prediction
            preds.append(u)
            # Closed loop: drive the reservoir with its own (bounded) prediction.
            state = self._update(state, u)
        return self._scaler.inverse(np.array(preds))
