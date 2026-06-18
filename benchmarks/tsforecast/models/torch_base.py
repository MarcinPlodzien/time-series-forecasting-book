"""
models/torch_base.py
====================

A shared training harness for every neural forecaster in this suite.

Almost all of these architectures share one template (Chapter 4): represent a
window of the recent past nonlinearly, then read out the next value. The
differences live entirely in how the window is processed, an MLP, a recurrent
cell, a dilated convolution, a state-space scan, or self-attention.
This file factors out everything those models have in common so each new
architecture is just a small `nn.Module` that maps a window of length L to a
single next-step prediction. That keeps the comparison fair: identical data,
identical standardisation, identical training loop, identical closed-loop
rollout. Only the inductive structure changes.

Closed-loop (free-running) forecasting is again the demanding test: we seed the
model with the last L observed samples, predict one step, append the prediction,
slide the window, and repeat with no further ground truth.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn

    TORCH_OK = True
except Exception:  # pragma: no cover - torch should be present
    TORCH_OK = False

from ..base import Forecaster, Standardiser


def set_seed(seed: int) -> None:
    """Make a run reproducible (CPU)."""
    np.random.seed(seed)
    if TORCH_OK:
        torch.manual_seed(seed)


def make_windows(z: np.ndarray, L: int):
    """Turn a 1-D scaled series into (windows, next-value) supervised pairs.

    windows[i] = z[i : i+L]   ->   target[i] = z[i+L].
    Every windowed model (MLP, RNN, TCN, transformer, ...) trains on these pairs,
    so the only thing that differs between them is the function that maps a window
    to its prediction.
    """
    X, y = [], []
    for i in range(len(z) - L):
        X.append(z[i : i + L])
        y.append(z[i + L])
    return np.asarray(X, np.float32), np.asarray(y, np.float32)


class SequenceForecaster(Forecaster):
    """Wrap any window -> scalar `nn.Module` as a closed-loop forecaster.

    Parameters
    ----------
    module_factory : callable() -> nn.Module
        Builds a fresh network mapping a tensor of shape (batch, L) to (batch,).
        A factory (not an instance) so we can re-seed and rebuild cleanly.
    window : int
        Input length L, the number of past samples the model sees. This is the
        explicit "memory horizon" of the architecture; for a model with a finite
        window it is the bound discussed in Chapter 7.
    name : str
        Label for tables and plots.
    epochs, lr, weight_decay, patience, batch_size : training hyperparameters.
    val_frac : float
        Fraction of the (end of the) training series held out for early stopping,
        so we do not overfit, which is what makes a closed-loop rollout blow up.
    seed : int
        RNG seed (these models are randomly initialised).
    """

    def __init__(
        self,
        module_factory,
        window: int,
        name: str,
        epochs: int = 500,
        lr: float = 3e-3,
        weight_decay: float = 1e-5,
        patience: int = 50,
        batch_size: int = 64,
        val_frac: float = 0.15,
        seed: int = 0,
        arch: str = "",
    ) -> None:
        if not TORCH_OK:
            raise RuntimeError("PyTorch is required for the neural models")
        self.module_factory = module_factory
        self.window = window
        self.name = name
        # Short human-readable description of the network's architecture, recorded
        # so the experiment can dump a reproducibility table of hyperparameters.
        self.arch = arch
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.batch_size = batch_size
        self.val_frac = val_frac
        self.seed = seed
        self._scaler = Standardiser()

    # ---- training ----------------------------------------------------------
    def fit(self, train: np.ndarray) -> "SequenceForecaster":
        set_seed(self.seed)
        z = self._scaler.fit(train).transform(train).astype(np.float32)
        X, y = make_windows(z, self.window)

        # Chronological train / validation split (no shuffling across the cut, so
        # validation always lies in the future of what the model trained on).
        n_val = max(1, int(len(X) * self.val_frac))
        Xtr, ytr = X[:-n_val], y[:-n_val]
        Xva, yva = X[-n_val:], y[-n_val:]
        Xtr_t = torch.from_numpy(Xtr)
        ytr_t = torch.from_numpy(ytr)
        Xva_t = torch.from_numpy(Xva)
        yva_t = torch.from_numpy(yva)

        self.net = self.module_factory()
        opt = torch.optim.Adam(
            self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        loss_fn = nn.MSELoss()

        best_val = float("inf")
        best_state = None
        bad = 0
        n = len(Xtr_t)
        for _epoch in range(self.epochs):
            self.net.train()
            # Mini-batch SGD over a fresh shuffling of the training windows.
            perm = torch.randperm(n)
            for s in range(0, n, self.batch_size):
                idx = perm[s : s + self.batch_size]
                opt.zero_grad()
                pred = self.net(Xtr_t[idx])
                loss = loss_fn(pred, ytr_t[idx])
                loss.backward()
                opt.step()
            # Early stopping on the held-out future window.
            self.net.eval()
            with torch.no_grad():
                vloss = float(loss_fn(self.net(Xva_t), yva_t))
            if vloss < best_val - 1e-6:
                best_val, bad = vloss, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.eval()
        return self

    # ---- closed-loop rollout ----------------------------------------------
    def forecast(self, horizon: int, warmup: np.ndarray | None = None) -> np.ndarray:
        warmup = warmup if warmup is not None else np.zeros(self.window)
        z = list(self._scaler.transform(np.asarray(warmup, float))[-self.window :])
        preds = []
        with torch.no_grad():
            for _ in range(horizon):
                win = torch.tensor(z[-self.window :], dtype=torch.float32).reshape(1, -1)
                nxt = self._scaler.clamp(float(self.net(win)[0]))
                preds.append(nxt)
                z.append(nxt)  # feed the (bounded) prediction back in
        return self._scaler.inverse(np.array(preds))
