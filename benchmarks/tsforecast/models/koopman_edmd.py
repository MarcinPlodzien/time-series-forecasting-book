"""
models/koopman_edmd.py
=====================

Extended Dynamic Mode Decomposition (EDMD), the data-driven Koopman method behind
the "nonlinear lift, linear readout" template (Chapters 4 and 5).

Koopman's theorem says that a nonlinear dynamical system becomes linear when
viewed through a rich enough set of observable functions. EDMD makes this
practical:

    1. Embed the scalar series into delay vectors v_t (Takens, Chapter 1).
    2. Lift each v_t through a dictionary of observables Psi(v) = [1, v, RBFs(v)].
    3. Fit a single matrix K (the approximate Koopman operator) so that
       Psi(v_{t+1}) ~ K Psi(v_t), by least squares over all training pairs.

Forecasting is then linear evolution in observable space. Because the dictionary
contains the identity coordinates, the predicted next delay vector is read
straight out of the linear block of K Psi(v_t); we reproject (rebuild the delay
vector from the predicted scalar and re-lift) at each step for a stable closed
loop. The nonlinear dictionary is what lets this linear operator capture the
folding of the laser attractor, so EDMD is a matched model here. Its accuracy is
set by how well the dictionary spans the relevant observables, the practical
closure problem (Chapter 5).
"""

from __future__ import annotations

import numpy as np

from ..base import Forecaster, Standardiser
from ..readout import ridge_solve, select_ridge_closedloop


class KoopmanEDMD(Forecaster):
    """EDMD with a radial-basis-function observable dictionary."""

    def __init__(
        self,
        delay: int = 4,
        n_rbf: int = 120,
        ridge: float | None = None,
        seed: int = 0,
    ) -> None:
        """
        delay : embedding dimension of the delay vector v_t.
        n_rbf : number of Gaussian observables (dictionary richness).
        ridge : regularisation of the least-squares Koopman fit. None (default)
                selects it on a validation split; a float fixes it.
        seed  : RNG seed for choosing RBF centres from the data.
        """
        self.delay = delay
        self.n_rbf = n_rbf
        self.ridge = ridge
        self.seed = seed
        self.name = f"Koopman/EDMD (d={delay}, rbf={n_rbf})"
        self._scaler = Standardiser()

    # ---- dictionary --------------------------------------------------------
    def _lift(self, V: np.ndarray) -> np.ndarray:
        """Lift delay vectors V (n, delay) to observables (n, 1+delay+n_rbf).

        Block layout [constant | linear coords | RBFs]. The linear block is what
        we reproject from, so we never need a separate readout matrix.
        """
        n = V.shape[0]
        # Squared distances to the RBF centres -> Gaussian features.
        d2 = np.sum(V**2, axis=1, keepdims=True) - 2 * V @ self._centres.T \
            + np.sum(self._centres**2, axis=1)[None, :]
        rbf = np.exp(-self._gamma * np.maximum(d2, 0.0))
        return np.column_stack([np.ones(n), V, rbf])

    def fit(self, train: np.ndarray) -> "KoopmanEDMD":
        train = np.asarray(train, float)
        if self.ridge is None:  # pick ridge by closed-loop capacity on a val tail
            self._eff_ridge = select_ridge_closedloop(
                lambda r: KoopmanEDMD(delay=self.delay, n_rbf=self.n_rbf,
                                      ridge=r, seed=self.seed),
                train)
        else:
            self._eff_ridge = float(self.ridge)
        rng = np.random.default_rng(self.seed)
        z = self._scaler.fit(train).transform(train)
        # Delay vectors v_t = [z_t, z_{t-1}, ..., z_{t-delay+1}], newest first.
        V = np.column_stack([z[self.delay - 1 - k : len(z) - 1 - k]
                             for k in range(self.delay)])
        Vnext = np.column_stack([z[self.delay - k : len(z) - k]
                                 for k in range(self.delay)])

        # RBF centres: a random subset of training delay vectors. Bandwidth from
        # the median pairwise distance (a standard, parameter-free heuristic).
        idx = rng.choice(len(V), size=min(self.n_rbf, len(V)), replace=False)
        self._centres = V[idx]
        sample = V[rng.choice(len(V), size=min(500, len(V)), replace=False)]
        med = np.median(np.linalg.norm(sample[:, None] - sample[None, :], axis=-1))
        self._gamma = 1.0 / (2.0 * (med**2 + 1e-12))

        Psi = self._lift(V)
        Psi_next = self._lift(Vnext)
        # Approximate Koopman operator: Psi_next ~ Psi K^T, ridge chosen above.
        self._K = ridge_solve(Psi, Psi_next, self._eff_ridge)
        self._ridge_used = self._eff_ridge
        return self

    # ---- closed-loop rollout (evolve + reproject) -------------------------
    def forecast(self, horizon: int, warmup: np.ndarray | None = None) -> np.ndarray:
        warmup = warmup if warmup is not None else np.zeros(self.delay)
        hist = list(self._scaler.transform(np.asarray(warmup, float)))
        preds = []
        for _ in range(horizon):
            v = np.array(hist[-self.delay :][::-1])      # newest first
            psi_next = self._lift(v[None, :]) @ self._K  # evolve in obs space
            # The linear block (indices 1..delay) is the predicted delay vector;
            # its first entry is the predicted next sample.
            nxt = self._scaler.clamp(float(psi_next[0, 1]))
            preds.append(nxt)
            hist.append(nxt)                              # reproject next step
        return self._scaler.inverse(np.array(preds))
