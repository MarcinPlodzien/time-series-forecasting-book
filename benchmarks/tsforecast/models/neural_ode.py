"""
models/neural_ode.py
===================

Latent Neural ODE (Chen et al. 2018), the continuous-time entry of Chapter 7.

Instead of a discrete recurrence, a Neural ODE learns a continuous vector field
and integrates it. Here we use the standard latent form: encode the recent window
into a latent state h0, integrate dh/dt = f(h) for a fixed interval with a
classical Runge--Kutta (RK4) solver, then decode the latent state to the next
sample. The learned vector field f is a small MLP.

Why this belongs in the matched group for the laser: the laser obeys a smooth
three-variable flow (laser-Lorenz), so a learned smooth vector field is a natural
hypothesis class. Its known difficulty is that adjoint training through long
integrations of a chaotic field is delicate; we sidestep that here by integrating
over a short fixed horizon per step and training one step at a time, which is a
stable way to use a neural ODE as a one-step map.

We integrate with plain backprop through the RK4 steps (the "discretise-then-
optimise" route) rather than the adjoint, since the per-step horizon is short.
The module maps a window to a scalar, so it plugs straight into the shared
SequenceForecaster harness and inherits the bounded closed-loop rollout.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .torch_base import SequenceForecaster


class _VectorField(nn.Module):
    """The learned dh/dt = f(h): a small MLP on the latent state."""

    def __init__(self, latent: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent, hidden), nn.Tanh(),
            nn.Linear(hidden, latent),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class _NeuralODE(nn.Module):
    """Encode the window, integrate a latent ODE with RK4, decode the next value."""

    def __init__(self, window: int, latent: int = 16, hidden: int = 72,
                 n_steps: int = 4, dt: float = 0.25) -> None:
        super().__init__()
        self.encoder = nn.Linear(window, latent)
        self.field = _VectorField(latent, hidden=hidden)
        self.decoder = nn.Linear(latent, 1)
        self.n_steps = n_steps
        self.dt = dt

    def _rk4(self, h: torch.Tensor) -> torch.Tensor:
        """One classical fourth-order Runge--Kutta step of dh/dt = f(h)."""
        dt = self.dt
        k1 = self.field(h)
        k2 = self.field(h + 0.5 * dt * k1)
        k3 = self.field(h + 0.5 * dt * k2)
        k4 = self.field(h + dt * k3)
        return h + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        for _ in range(self.n_steps):  # integrate the latent trajectory
            h = self._rk4(h)
        return self.decoder(h).squeeze(-1)


def make_neural_ode(window: int = 30, seed: int = 0) -> SequenceForecaster:
    # latent=16, field hidden=72 lands the model at the shared ~3k budget.
    return SequenceForecaster(lambda: _NeuralODE(window, latent=16, hidden=72, n_steps=4),
                              window, "Neural ODE", seed=seed,
                              arch="latent ODE, dim=16, RK4 x4 steps, MLP field h=72")
