"""
models/quantum_reservoir.py
==========================

A small, classically simulated quantum reservoir computer (QRC) in the
ensemble-averaged (expectation-value) regime, following Fujii and Nakajima.

The reservoir is a fixed, untrained quantum many-body system; only a linear
readout is trained. Per input step the procedure is:

    encode  : inject the scalar input s_k by resetting qubit 0 to the state
              |psi> = sqrt(1-s_k)|0> + sqrt(s_k)|1> (s_k min-max scaled to [0,1]),
              keeping the rest of the register and its correlations,
    evolve  : apply the propagator U = exp(-i H tau) once,
    collect : read the expectation values <O> = Tr(O rho) of a fixed set of
              Pauli observables.

The collected feature vectors are stacked over the training series and a single
ridge regression maps them to the next sample. Forecasting runs closed-loop by
feeding each prediction back as the next input.

Hamiltonian (transverse-field Ising, fully connected, quenched-random couplings):

    H = sum_{i<j} J_{ij} Z_i Z_j + sum_i (h_x X_i + h_z Z_i),

with J_{ij} ~ U(-J_s, J_s) drawn once and frozen, and uniform fields h_x, h_z.
The transverse field makes H non-commuting, so the dynamics scramble the input
across the register through entanglement.

Readout observables: the full one- and two-body Pauli set, i.e. the single-qubit
<X_i>, <Y_i>, <Z_i> and the two-body <A_i B_j> for all A, B in {X, Y, Z} and all
pairs i<j. The two-body correlators are the entangled features a classical linear
model of the raw input cannot form on its own.

The simulation is small and exact (n=6 qubits, a 64-dimensional Hilbert space,
full density matrix). It makes no quantum-advantage claim; it is a small-scale
demonstration of the mechanism. The state must be a density matrix because the
reset input is a non-unitary channel that leaves the register mixed.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from ..base import Forecaster
from ..readout import solve_readout


# Single-qubit Pauli matrices.
_I2 = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def _kron_op(op: np.ndarray, site: int, n: int) -> np.ndarray:
    """Embed a single-qubit operator on `site` into the n-qubit Hilbert space."""
    mats = [op if k == site else _I2 for k in range(n)]
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


class QuantumReservoir(Forecaster):
    """Ensemble-averaged quantum reservoir computer with a ridge readout."""

    def __init__(
        self,
        n_qubits: int = 6,
        t: float = 4.0,
        n_virtual: int = 5,
        observables: str = "full",
        J_scale: float = 1.0,
        h_x: float = 1.0,
        h_z: float = 0.5,
        ridge: float | None = None,
        washout: int = 100,
        warmup_cap: int = 120,
        seed: int = 0,
    ) -> None:
        """
        n_qubits   : register size (Hilbert dimension 2**n_qubits).
        t          : evolution time per input step under H.
        n_virtual  : virtual nodes (temporal multiplexing, Fujii-Nakajima). The
                     single evolution of duration t is split into n_virtual equal
                     sub-steps and the observables are read out after each, so one
                     input step yields n_virtual successive snapshots of the
                     reservoir's transient relaxation rather than only the final
                     one. This multiplies the readout feature count by n_virtual
                     without adding qubits and injects the fading-memory structure
                     the single snapshot lacks. The intermediate readouts are
                     non-destructive here because we take expectation values of the
                     density matrix (no measurement back-action); on hardware they
                     would cost extra shots. Default 5; set 1 for a single readout.
        observables: "full" uses the complete one- and two-body Pauli set
                     (<A_i B_j>, A,B in {X,Y,Z}); "diag" uses only same-axis
                     correlators (<X_iX_j>, <Y_iY_j>, <Z_iZ_j>).
        J_scale    : half-width of the coupling distribution J_{ij} ~ U(-J_s, J_s).
        h_x, h_z   : transverse and longitudinal field strengths (uniform).
        ridge      : regularisation of the linear readout. None (default) selects
                     it on a validation split; a float fixes it.
        washout    : steps discarded at the start of training (transient).
        warmup_cap : history length used to re-seed rho at each forecast origin;
                     the reservoir has fading memory, so recent history suffices
                     and this keeps rolling-origin evaluation fast.
        seed       : RNG seed for the fixed random couplings.
        """
        self.n = n_qubits
        self.t = t
        self.n_virtual = n_virtual
        self.observables = observables
        self.J_scale = J_scale
        self.h_x = h_x
        self.h_z = h_z
        self.ridge = ridge
        self.washout = washout
        self.warmup_cap = warmup_cap
        self.seed = seed
        self.name = f"Quantum reservoir (n={n_qubits})"
        self._u_min = 0.0
        self._u_max = 1.0

    # ---- reservoir construction -------------------------------------------
    def _build(self) -> None:
        rng = np.random.default_rng(self.seed)
        n = self.n
        D = 2 ** n
        self._D = D

        Xs = [_kron_op(_X, i, n) for i in range(n)]
        Ys = [_kron_op(_Y, i, n) for i in range(n)]
        Zs = [_kron_op(_Z, i, n) for i in range(n)]

        # H = sum_{i<j} J_ij Z_i Z_j + sum_i (h_x X_i + h_z Z_i).
        H = np.zeros((D, D), dtype=complex)
        for i in range(n):
            H += self.h_x * Xs[i] + self.h_z * Zs[i]
            for j in range(i + 1, n):
                H += rng.uniform(-self.J_scale, self.J_scale) * (Zs[i] @ Zs[j])
        # Propagator for one virtual node (= one full step when n_virtual == 1).
        self._Usub = expm(-1j * H * (self.t / self.n_virtual))
        self._Usubdag = self._Usub.conj().T

        # Readout observable set, stacked into one (n_obs, D, D) array so all
        # expectation values come from a single einsum per readout.
        one_body = Xs + Ys + Zs
        two_body = []
        axes = [Xs, Ys, Zs]
        for i in range(n):
            for j in range(i + 1, n):
                if self.observables == "full":
                    two_body += [A[i] @ B[j] for A in axes for B in axes]
                else:  # "diag": same-axis correlators only
                    two_body += [A[i] @ A[j] for A in axes]
        self._Obs = np.stack(one_body + two_body)  # (n_obs, D, D)

    def _expect(self, rho: np.ndarray) -> np.ndarray:
        """All observable expectations Tr(O rho) in one pass.

        Tr(O rho) = sum_ij O_ij rho_ji, batched over the observable axis k.
        """
        return np.einsum("kij,ji->k", self._Obs, rho).real

    def _inject(self, rho: np.ndarray, u: float) -> np.ndarray:
        """Reset qubit 0 to |psi(u)> while keeping the rest of the register."""
        D = self._D
        rest = D // 2
        r = rho.reshape(2, rest, 2, rest)          # partial trace over qubit 0
        rho_rest = r[0, :, 0, :] + r[1, :, 1, :]
        u = min(max(u, 0.0), 1.0)
        psi = np.array([np.sqrt(1.0 - u), np.sqrt(u)], dtype=complex)
        rho_q0 = np.outer(psi, psi.conj())
        return np.kron(rho_q0, rho_rest)

    def _step(self, rho: np.ndarray, u: float) -> tuple[np.ndarray, np.ndarray]:
        """Encode the input, evolve, collect observables.

        With n_virtual == 1 this is a single evolution and a single readout;
        larger n_virtual reads out after each sub-step and concatenates.
        """
        rho = self._inject(rho, u)
        feats = []
        for _ in range(self.n_virtual):
            rho = self._Usub @ rho @ self._Usubdag
            feats.append(self._expect(rho))
        return rho, np.concatenate(feats)

    # ---- input scaling -----------------------------------------------------
    def _to_unit(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, float) - self._u_min) / (self._u_max - self._u_min)

    def _from_unit(self, u: np.ndarray) -> np.ndarray:
        return np.asarray(u, float) * (self._u_max - self._u_min) + self._u_min

    # ---- training ----------------------------------------------------------
    def _features_targets(self, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Evolve the reservoir along scaled inputs `u` and collect (bias+observables,
        next-sample) pairs after the washout."""
        rho = np.eye(self._D, dtype=complex) / self._D  # maximally mixed start
        feats, targets = [], []
        for t in range(len(u) - 1):
            rho, f = self._step(rho, u[t])
            if t >= self.washout:
                feats.append(np.concatenate([[1.0], f]))  # bias + observables
                targets.append(u[t + 1])
        return np.array(feats), np.array(targets)

    def _select_ridge_closedloop(self, train: np.ndarray, u: np.ndarray) -> float:
        """Choose the readout ridge by closed-loop capacity on a validation tail.

        The reservoir features depend only on the inputs, not on the ridge, so we
        evolve once over the fit part and then, for each candidate ridge, solve the
        readout and run a short rolling-origin rollout over the held-out tail.
        This keeps the (otherwise costly) selection to one reservoir pass plus light
        rollouts, and aligns the ridge with the closed-loop score we report.
        """
        from ..readout import RIDGE_GRID_COARSE, ridge_solve
        from .. import evaluation, metrics

        n = len(u)
        cut = int(round(0.8 * n))
        horizon, n_org, spacing = 30, 8, 8
        if cut < self.washout + 50 or (n - cut) < horizon + 5:
            return 1e-6
        Ff, yf = self._features_targets(u[:cut])  # one reservoir pass on the fit part
        if len(yf) < 10:
            return 1e-6
        best_r, best_c = float(RIDGE_GRID_COARSE[0]), -np.inf
        for r in RIDGE_GRID_COARSE:
            self._W = ridge_solve(Ff, yf, float(r))   # reuses Ff
            try:
                preds, truths = evaluation.rolling_origin_forecasts(
                    self, train, cut, horizon, n_org, spacing
                )
                c = float(np.nansum(metrics.capacity_by_lead(preds, truths)))
            except Exception:
                c = -np.inf
            if np.isfinite(c) and c > best_c:
                best_c, best_r = c, float(r)
        return best_r

    def fit(self, train: np.ndarray) -> "QuantumReservoir":
        from ..readout import ridge_solve

        self._build()
        train = np.asarray(train, float)
        self._u_min, self._u_max = float(train.min()), float(train.max())
        u = self._to_unit(train)
        # Ridge readout. With many virtual nodes the feature count is large, so the
        # regularisation matters; ridge=None selects it by closed-loop capacity on a
        # validation tail (the objective we report), else a fixed float is used.
        eff = self._select_ridge_closedloop(train, u) if self.ridge is None \
            else float(self.ridge)
        F, y = self._features_targets(u)          # features on the full training set
        self._W = ridge_solve(F, y, eff)
        self._ridge_used = eff
        return self

    # ---- closed-loop rollout ----------------------------------------------
    def forecast(self, horizon: int, warmup: np.ndarray | None = None) -> np.ndarray:
        if warmup is None:
            raise RuntimeError("quantum reservoir needs warm-up history")
        hist = np.asarray(warmup, float)[-self.warmup_cap:]
        u_hist = self._to_unit(hist)
        rho = np.eye(self._D, dtype=complex) / self._D
        # Re-drive the reservoir along the recent history to set its state.
        for val in u_hist:
            rho, f = self._step(rho, val)
        u = float(np.concatenate([[1.0], f]) @ self._W)  # first prediction
        preds = []
        for _ in range(horizon):
            u = min(max(u, 0.0), 1.0)  # physical encoding range
            preds.append(u)
            rho, f = self._step(rho, u)               # feed prediction back
            u = float(np.concatenate([[1.0], f]) @ self._W)
        return self._from_unit(np.array(preds))
