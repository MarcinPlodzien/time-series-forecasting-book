"""
models/neural.py
===============

Neural forecasting architectures, each a window -> scalar map used through the
shared SequenceForecaster harness (see torch_base.py). The models are grouped by
the chapter that introduces them:

    Chapter 6 (reservoirs and recurrent networks)
        RNN   -- Elman recurrent network
        LSTM  -- long short-term memory
        GRU   -- gated recurrent unit
    Chapter 7 (modern sequence models)
        MLP         -- feedforward baseline over the flattened window
        TCN         -- temporal convolutional network (finite causal receptive field)
        S4D         -- diagonal structured state-space model (HiPPO-LegS init)
        Mamba       -- selective state-space model
        Transformer -- self-attention encoder
        DLinear     -- linear decomposition forecaster

Every trainable network is sized to a common budget of about 3000 parameters by
adjusting its hidden width, so a difference in forecasting capacity reflects
inductive structure rather than parameter count. Each architecture's inductive
bias and expected failure mode are discussed in the corresponding chapter; the
docstrings below describe the construction only.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .torch_base import SequenceForecaster


# ===========================================================================
#  Chapter 6: recurrent networks
# ===========================================================================
class _RecurrentNet(nn.Module):
    """Shared body for RNN / LSTM / GRU: run the cell over the window, read out
    the final hidden state with a linear layer. The only thing that varies is the
    recurrent cell."""

    def __init__(self, cell: str, hidden: int = 64, layers: int = 1) -> None:
        super().__init__()
        rnn_cls = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[cell]
        # input_size=1: the series is scalar, fed one value per time step.
        self.rnn = rnn_cls(input_size=1, hidden_size=hidden, num_layers=layers,
                           batch_first=True)
        self.readout = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, L) -> (batch, L, 1)
        out, _ = self.rnn(x.unsqueeze(-1))
        last = out[:, -1, :]  # final-step hidden state summarises the window
        return self.readout(last).squeeze(-1)


# ===========================================================================
#  Chapter 7: feedforward, convolutional, state-space, attention, linear
# ===========================================================================
class _MLP(nn.Module):
    """Plain multilayer perceptron over the flattened window. No temporal prior
    at all beyond the window length, which makes it the reference point for
    the structured models."""

    def __init__(self, window: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(window, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class _Chomp1d(nn.Module):
    """Trim the right padding so a convolution stays strictly causal (no peek
    into the future). This enforces causality discretely for the TCN (Chapter 7)."""

    def __init__(self, chomp: int) -> None:
        super().__init__()
        self.chomp = chomp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.chomp] if self.chomp > 0 else x


class _TCN(nn.Module):
    """Temporal convolutional network: a stack of dilated causal convolutions.
    The dilations grow geometrically so the receptive field covers the whole
    window with few layers, but the receptive field remains finite."""

    def __init__(self, channels: int = 32, levels: int = 3, kernel: int = 3) -> None:
        super().__init__()
        layers = []
        in_ch = 1
        for i in range(levels):
            dilation = 2 ** i
            pad = (kernel - 1) * dilation
            layers += [
                nn.Conv1d(in_ch, channels, kernel, padding=pad, dilation=dilation),
                _Chomp1d(pad),
                nn.ReLU(),
            ]
            in_ch = channels
        self.tcn = nn.Sequential(*layers)
        self.readout = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.tcn(x.unsqueeze(1))  # (batch, channels, L)
        return self.readout(h[:, :, -1]).squeeze(-1)  # last (causal) step


def _hippo_legs_diag(n_state: int) -> torch.Tensor:
    """Diagonal HiPPO-LegS modes: the S4D-LegS initialisation.

    Build the HiPPO-LegS state matrix, form its normal part (a constant -1/2 on
    the diagonal plus a skew-symmetric remainder), and return its eigenvalues
    lambda_n = -1/2 + i*w_n. These complex poles are the long-range-memory
    initialisation of S4 reduced to the diagonal case (the structured
    state-space section of Chapter 7), as opposed to a generic random diagonal SSM.
    """
    idx = torch.arange(n_state, dtype=torch.float64)
    p = torch.sqrt(1.0 + 2.0 * idx)
    A = -(torch.tril(p.unsqueeze(1) * p.unsqueeze(0)) - torch.diag(idx))  # HiPPO-LegS
    rank = torch.sqrt(idx + 0.5)
    S = A + rank.unsqueeze(1) * rank.unsqueeze(0)          # normal part: -1/2 I + skew
    return torch.linalg.eigvals(S.to(torch.complex128))   # (N,) complex, Re ~ -1/2


class _S4DLayer(nn.Module):
    """One diagonal structured-state-space layer (S4D), acting per channel.

    Each of the H channels carries its own bank of N independent complex modes
    s_t = a * s_{t-1} + b * x_t, read out linearly to a real channel output. The
    continuous pole is lambda = -softplus(decay) + i*freq and the discrete pole is

        a = exp(dt * lambda),    |a| < 1,

    so the state has stable, fading memory (the Chapter 7 prior) while the
    imaginary part lets each mode oscillate.

    The modes are initialised from the HiPPO-LegS operator (the S4D-LegS scheme of
    Gu et al., the structured state-space section of Chapter 7): the diagonal poles
    are the eigenvalues of the normal part of the HiPPO-LegS state matrix,
    lambda_n = -1/2 + i*w_n. This long-range-memory initialisation is what
    distinguishes S4 from a generic random diagonal SSM. We keep the recurrence
    explicit (a scan over the short window) for readability rather than the fast
    FFT form.
    """

    def __init__(self, channels: int, n_state: int = 32, init: str = "hippo") -> None:
        super().__init__()
        H, N = channels, n_state
        if init == "hippo":
            # HiPPO-LegS diagonal (S4D-LegS) init: lambda_n from the HiPPO-LegS
            # eigenvalues (Re ~ 1/2), the long-range-memory initialisation.
            lam = _hippo_legs_diag(N)
            re = (-lam.real).clamp(min=1e-4).to(torch.float32)
            im = lam.imag.abs().to(torch.float32)
            self.log_dt = nn.Parameter(torch.rand(H, N) * 4.6 - 6.9)
            self.decay = nn.Parameter(torch.log(torch.expm1(re)).expand(H, N).clone())
            self.freq = nn.Parameter(im.expand(H, N).clone())
            self.b = nn.Parameter(torch.ones(H, N))
        elif init == "random":
            # Generic random diagonal SSM, no HiPPO structure. Kept as an ablation
            # that shows what the HiPPO initialisation actually buys.
            self.log_dt = nn.Parameter(torch.rand(H, N) * 0.5 - 3.0)
            self.decay = nn.Parameter(torch.rand(H, N) * 0.5 + 0.1)
            self.freq = nn.Parameter(torch.rand(H, N) * math.pi)
            self.b = nn.Parameter(torch.randn(H, N) * 0.5)
        else:
            raise ValueError(f"unknown S4D init: {init!r}")
        self.c = nn.Parameter(torch.randn(H, N) * (1.0 / math.sqrt(N)))
        self.d = nn.Parameter(torch.zeros(H))  # direct skip per channel

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        # u: (B, L, H) -> output (B, L, H)
        B, L, H = u.shape
        dt = torch.exp(self.log_dt)
        a = torch.exp(dt * (-torch.nn.functional.softplus(self.decay)
                            + 1j * self.freq))            # (H, N)
        b = self.b.to(torch.cfloat)
        s = torch.zeros(B, H, a.shape[1], dtype=torch.cfloat)
        outs = []
        for t in range(L):
            s = s * a + b * u[:, t, :].unsqueeze(-1)       # (B, H, N)
            y_t = (s.real * self.c).sum(-1) + self.d * u[:, t, :]
            outs.append(y_t)
        return torch.stack(outs, dim=1)                    # (B, L, H)


class _S4D(nn.Module):
    """A small but genuinely nonlinear S4 model.

    A single linear SSM with a linear readout is a
    linear time-invariant system overall, and like any linear model it cannot
    fold a chaotic attractor, so on closed-loop laser forecasting it fails as the
    linear AR model does. Real S4/Mamba networks interleave SSM layers with
    pointwise nonlinearities and channel mixing, and that is what makes them
    nonlinear sequence models. We therefore stack: embed the scalar into H
    channels, apply an S4D layer, a GELU and a linear channel-mixing, a second S4D
    layer, then read out the last time step. This is a minimal version of the
    architecture rather than a single linear scan.
    """

    def __init__(self, channels: int = 24, n_state: int = 32, init: str = "hippo") -> None:
        super().__init__()
        self.embed = nn.Linear(1, channels)
        self.ssm1 = _S4DLayer(channels, n_state, init)
        self.mix = nn.Linear(channels, channels)
        self.ssm2 = _S4DLayer(channels, n_state, init)
        self.readout = nn.Linear(channels, 1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x.unsqueeze(-1))          # (B, L, H)
        h = self.act(self.ssm1(h))
        h = self.act(self.mix(h))                # pointwise nonlinear channel mix
        h = self.act(self.ssm2(h))
        return self.readout(h[:, -1, :]).squeeze(-1)


class _MambaBlock(nn.Module):
    """One selective state-space block, H channels in and out.

    The state-space parameters are input-dependent (the "selective" mechanism):
    per step the input produces B(t), C(t) and a step size Delta(t) = softplus(.),
    so the diagonal recurrence

        h_t = exp(Delta_t * A) * h_{t-1} + (Delta_t * B_t) * u_t,
        y_t = sum_n C_t * h_t + D * u_t,

    is linear time-varying. A large Delta_t writes the current input into the
    state; a small Delta_t lets it coast, a content-based gate analogous to an LSTM
    forget gate. A short causal depthwise conv mixes local context before the scan,
    and a SiLU gate branch follows it, both standard Mamba components. The scan is
    sequential, which is fine for the short window here. Returns the full sequence
    so blocks can be stacked with residual connections.
    """

    def __init__(self, channels: int, n_state: int, conv_k: int = 4) -> None:
        super().__init__()
        H, N = channels, n_state
        self.H, self.N = H, N
        self.conv = nn.Conv1d(H, H, conv_k, groups=H, padding=conv_k - 1)
        self.A_log = nn.Parameter(torch.log(torch.rand(H, N) * 0.5 + 0.5))  # A=-exp
        self.x_proj = nn.Linear(H, 2 * N + H)    # -> B(t)[N], C(t)[N], dt_raw[H]
        self.dt_bias = nn.Parameter(torch.zeros(H))
        self.D = nn.Parameter(torch.ones(H))
        self.gate = nn.Linear(H, H)
        self.out_proj = nn.Linear(H, H)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        B, L, H = u.shape
        uc = self.conv(u.transpose(1, 2))[:, :, :L].transpose(1, 2)
        uc = torch.nn.functional.silu(uc)        # causal local mixing
        A = -torch.exp(self.A_log)               # (H, N), stable (negative real)
        h = torch.zeros(B, H, self.N, device=u.device)
        ys = []
        for t in range(L):
            ut = uc[:, t, :]                      # (B, H)
            proj = self.x_proj(ut)               # (B, 2N+H)
            Bt = proj[:, : self.N]
            Ct = proj[:, self.N : 2 * self.N]
            dt = torch.nn.functional.softplus(proj[:, 2 * self.N :] + self.dt_bias)
            dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))   # (B, H, N)
            dB = dt.unsqueeze(-1) * Bt.unsqueeze(1)             # (B, H, N)
            h = dA * h + dB * ut.unsqueeze(-1)
            ys.append((h * Ct.unsqueeze(1)).sum(-1) + self.D * ut)
        y = torch.stack(ys, dim=1)               # (B, L, H)
        y = y * torch.nn.functional.silu(self.gate(u))  # gate from block input
        return self.out_proj(y)


class _Mamba(nn.Module):
    """Stacked selective state-space (Mamba) model as a window -> scalar map.

    Embeds the scalar window into H channels, applies `n_layers` selective blocks
    with residual connections (matching the depth of the S4D model so the two
    state-space architectures are compared at equal depth as well as equal
    parameter budget), and reads out the last step. Sized to the shared ~3k budget.
    """

    def __init__(self, channels: int = 17, n_state: int = 8, n_layers: int = 2) -> None:
        super().__init__()
        self.embed = nn.Linear(1, channels)
        self.blocks = nn.ModuleList(
            [_MambaBlock(channels, n_state) for _ in range(n_layers)]
        )
        self.readout = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x.unsqueeze(-1))          # (B, L, H)
        for blk in self.blocks:
            h = h + blk(h)                       # residual stack
        return self.readout(h[:, -1, :]).squeeze(-1)


class _PositionalEncoding(nn.Module):
    """Standard fixed sinusoidal positions, so attention can use order. A
    Transformer is permutation-invariant without this, which is why it has no
    built-in notion of time and must be told about it (Chapter 7)."""

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1)].unsqueeze(0)


class _Transformer(nn.Module):
    """Self-attention encoder over the window, read out from the last token.
    Attention is a weakly-structured prior for smooth low-dimensional dynamics; the
    model is kept small to limit overfitting on the short training set."""

    def __init__(self, window: int, d_model: int = 32, heads: int = 4,
                 layers: int = 2) -> None:
        super().__init__()
        self.embed = nn.Linear(1, d_model)
        self.pos = _PositionalEncoding(d_model, max_len=window + 1)
        enc = nn.TransformerEncoderLayer(d_model, heads, dim_feedforward=2 * d_model,
                                         batch_first=True, dropout=0.0)
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.readout = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x.unsqueeze(-1))  # (batch, L, d_model)
        h = self.pos(h)
        h = self.encoder(h)
        return self.readout(h[:, -1, :]).squeeze(-1)


class _DLinear(nn.Module):
    """Decomposition-linear forecaster (Zeng et al. 2023): split the window into
    a moving-average trend and a remainder, apply one linear map to each, and add.
    An almost parameter-free linear baseline for the comparison."""

    def __init__(self, window: int, kernel: int = 5) -> None:
        super().__init__()
        self.kernel = kernel
        self.trend = nn.Linear(window, 1)
        self.season = nn.Linear(window, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Moving-average trend via average pooling with refl
        pad = self.kernel // 2
        xp = torch.nn.functional.pad(x.unsqueeze(1), (pad, pad), mode="replicate")
        trend = torch.nn.functional.avg_pool1d(xp, self.kernel, stride=1).squeeze(1)
        trend = trend[:, : x.shape[1]]
        season = x - trend
        return (self.trend(trend) + self.season(season)).squeeze(-1)


# ===========================================================================
#  Public constructors: each returns a ready-to-use SequenceForecaster.
#
#  Two choices keep the comparison about inductive structure rather than scale:
#    * Every model uses the same input window (30 samples), chosen from the
#      diagnosis (autocorrelation ~2; a few oscillation periods span ~30 samples).
#    * Every trainable network is sized to the same parameter budget, about 3000
#      trainable parameters, by picking the appropriate hidden width per
#      architecture. With only 1000 training points this small, common budget
#      makes a difference in capacity reflect structure rather than one model
#      having more parameters. The one exception is
#      DLinear, which is an intrinsically linear reference (two linear maps over
#      the window, a few dozen parameters) and is not meant to be inflated.
#
#  The per-architecture widths below were chosen so that the parameter counts all
#  land near 3000; experiments/03_full_comparison.py prints the exact counts.
# ===========================================================================
_PARAM_BUDGET = 3000  # target trainable parameters for every sized network


def make_rnn(window: int = 30, hidden: int = 52, seed: int = 0) -> SequenceForecaster:
    return SequenceForecaster(lambda: _RecurrentNet("rnn", hidden=hidden),
                              window, "RNN", seed=seed,
                              arch=f"Elman RNN, hidden={hidden}, 1 layer")


def make_lstm(window: int = 30, hidden: int = 26, seed: int = 0) -> SequenceForecaster:
    return SequenceForecaster(lambda: _RecurrentNet("lstm", hidden=hidden),
                              window, "LSTM", seed=seed,
                              arch=f"LSTM, hidden={hidden}, 1 layer")


def make_gru(window: int = 30, hidden: int = 30, seed: int = 0) -> SequenceForecaster:
    return SequenceForecaster(lambda: _RecurrentNet("gru", hidden=hidden),
                              window, "GRU", seed=seed,
                              arch=f"GRU, hidden={hidden}, 1 layer")


def make_mlp(window: int = 30, hidden: int = 40, seed: int = 0) -> SequenceForecaster:
    return SequenceForecaster(lambda: _MLP(window, hidden=hidden),
                              window, "MLP", seed=seed,
                              arch=f"MLP, 2 hidden layers x{hidden}, tanh")


def make_tcn(window: int = 30, channels: int = 22, seed: int = 0) -> SequenceForecaster:
    return SequenceForecaster(lambda: _TCN(channels=channels, levels=3),
                              window, "TCN", seed=seed,
                              arch=f"TCN, {channels} channels, 3 dilated levels, kernel 3")


def make_s4d(window: int = 30, seed: int = 0) -> SequenceForecaster:
    # 16 channels x 16 modes lands the SSM at the shared ~3k budget; the explicit
    # complex scan is the slowest module, so we also cap epochs (early stopping
    # reaches a good fit well before the cap).
    return SequenceForecaster(lambda: _S4D(channels=16, n_state=16, init="hippo"),
                              window, "S4D", epochs=250, seed=seed,
                              arch="S4D, 16 channels x 16 modes, 2 layers, HiPPO-LegS init")


def make_s4d_random(window: int = 30, seed: int = 0) -> SequenceForecaster:
    # Same architecture as make_s4d but with a generic random diagonal init instead
    # of HiPPO-LegS, kept as an ablation to show what the HiPPO initialisation buys.
    return SequenceForecaster(lambda: _S4D(channels=16, n_state=16, init="random"),
                              window, "S4D (random)", epochs=250, seed=seed,
                              arch="S4D, 16 channels x 16 modes, 2 layers, random diagonal init")


def make_mamba(window: int = 30, seed: int = 0) -> SequenceForecaster:
    # Two stacked selective blocks (17 channels x 8 modes) match the S4D depth at
    # the shared ~3k budget. The selective SSM fits the one-step map quickly but
    # needs longer training for a stable free-running rollout (the one-step
    # validation loss plateaus while the closed-loop error keeps falling), so we
    # raise the epoch cap and patience.
    return SequenceForecaster(lambda: _Mamba(channels=17, n_state=8, n_layers=2),
                              window, "Mamba", epochs=800, patience=120, seed=seed,
                              arch="Mamba, 2 selective blocks x (17 channels, 8 modes), causal conv k=4")


def make_transformer(window: int = 30, d_model: int = 20, layers: int = 1,
                     seed: int = 0) -> SequenceForecaster:
    return SequenceForecaster(lambda: _Transformer(window, d_model=d_model, heads=4,
                                                   layers=layers),
                              window, "Transformer", seed=seed,
                              arch=f"Transformer enc, d_model={d_model}, 4 heads, {layers} layer")


def make_dlinear(window: int = 30, seed: int = 0) -> SequenceForecaster:
    return SequenceForecaster(lambda: _DLinear(window, kernel=5),
                              window, "DLinear", seed=seed,
                              arch="DLinear, moving-avg kernel 5 (intrinsically linear)")
