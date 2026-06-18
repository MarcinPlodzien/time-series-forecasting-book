"""
embedding.py
============

Diagnostic tools for the signal (Part I).

Before choosing a model we diagnose the signal. Two questions dominate for a
deterministic record:

    * How much of the past must we carry?   -> a delay/embedding dimension,
                                                estimated from the first minimum
                                                of the mutual information (delay)
                                                and false-nearest-neighbours
                                                (dimension).
    * How fast do nearby trajectories separate? -> the largest Lyapunov exponent,
                                                    which sets the prediction
                                                    horizon (Chapter 1).

These are compact, dependency-light implementations. They are not the most
statistically refined estimators in the literature; on measured data these are
estimator-dependent summaries to be sanity-checked, not invariants to be trusted
blindly.
"""

from __future__ import annotations

import numpy as np


def delay_embed(x: np.ndarray, dim: int, tau: int = 1) -> np.ndarray:
    """Takens delay-coordinate embedding of a scalar series.

    Each row is the vector  [x_t, x_{t-tau}, ..., x_{t-(dim-1)tau}].
    This turns a scalar observable into points in a reconstructed phase space
    whose geometry is (generically) diffeomorphic to the true attractor
    (Takens' theorem, Chapter 1). Every delay-based forecaster in this repo
    builds its inputs with this function, which is why the embedding dimension
    matters.

    Returns an array of shape (N - (dim-1)*tau, dim), most recent lag first.
    """
    x = np.asarray(x, float)
    span = (dim - 1) * tau
    n = x.size - span
    if n <= 0:
        raise ValueError("series too short for this (dim, tau)")
    # Column j holds the series lagged by j*tau, trimmed to a common length.
    cols = [x[span - j * tau : span - j * tau + n] for j in range(dim)]
    return np.column_stack(cols)


def autocorrelation_time(x: np.ndarray) -> int:
    """First lag at which the autocorrelation drops below 1/e.

    A crude but robust memory scale. We use it as a default delay `tau` and as a
    sanity check on the more careful mutual-information estimate below.
    """
    x = np.asarray(x, float) - np.mean(x)
    var = np.dot(x, x)
    if var == 0:
        return 1
    for lag in range(1, len(x) // 2):
        ac = np.dot(x[:-lag], x[lag:]) / var
        if ac < np.exp(-1.0):
            return lag
    return 1


def mutual_information_delay(x: np.ndarray, max_lag: int = 50, bins: int = 16) -> int:
    """Delay from the first minimum of the time-delayed mutual information.

    Mutual information captures *nonlinear* dependence between x_t and x_{t+lag},
    where autocorrelation only sees linear dependence. The first local minimum is
    the standard heuristic (Fraser & Swinney) for the embedding delay: it is the
    lag at which the delayed coordinate is "maximally new" while still dynamically
    related. Returns 1 if no clear minimum is found within max_lag.
    """
    x = np.asarray(x, float)
    # Discretise once; reuse the bin edges for every lag so the histograms align.
    edges = np.histogram_bin_edges(x, bins=bins)
    sym = np.clip(np.digitize(x, edges[1:-1]), 0, bins - 1)

    def mi(lag: int) -> float:
        a, b = sym[:-lag], sym[lag:]
        joint = np.zeros((bins, bins))
        np.add.at(joint, (a, b), 1.0)
        joint /= joint.sum()
        pa = joint.sum(axis=1, keepdims=True)
        pb = joint.sum(axis=0, keepdims=True)
        nz = joint > 0
        return float(np.sum(joint[nz] * np.log(joint[nz] / (pa * pb)[nz])))

    mis = [mi(lag) for lag in range(1, max_lag + 1)]
    # First lag whose MI is below both neighbours -> first local minimum.
    for k in range(1, len(mis) - 1):
        if mis[k] < mis[k - 1] and mis[k] < mis[k + 1]:
            return k + 1
    return 1


def false_nearest_neighbours(
    x: np.ndarray,
    tau: int,
    max_dim: int = 10,
    rtol: float = 15.0,
    atol: float = 2.0,
) -> list[float]:
    """Fraction of false neighbours as a function of embedding dimension.

    Idea (Kennel et al.): in too low a dimension the attractor is squashed and
    distant points are projected on top of one another. Increasing the dimension
    "unfolds" these false crossings. The smallest dimension at which the false
    fraction collapses toward zero is a good embedding dimension.

    Returns a list `frac[d]` for d = 1..max_dim. O(N^2) per dimension, fine for
    the 1000-point training window used here; not meant for long records.
    """
    x = np.asarray(x, float)
    attractor_size = np.std(x)  # global scale for Kennel's absolute criterion
    fractions = []
    for d in range(1, max_dim + 1):
        emb = delay_embed(x, dim=d, tau=tau)
        emb_next = delay_embed(x, dim=d + 1, tau=tau)
        # The two embeddings start at different times: the (d+1)-D embedding needs
        # one extra (older) lag, so its first row is `tau` samples later than the
        # d-D embedding's first row. Drop the leading `tau` rows of the d-D
        # embedding so emb[i] and emb_next[i] refer to the same time point. This
        # alignment is the step that, if skipped, makes the dimension collapse to 1.
        emb = emb[tau:]
        m = min(len(emb), len(emb_next))
        emb, emb_next = emb[:m], emb_next[:m]

        false = 0
        for i in range(m):
            # Nearest neighbour of point i in the d-D embedding (exclude itself).
            d2 = np.sum((emb - emb[i]) ** 2, axis=1)
            d2[i] = np.inf
            j = int(np.argmin(d2))
            r_d = np.sqrt(d2[j])
            # Extra distance contributed by the new (deepest) coordinate.
            extra = abs(emb_next[i, -1] - emb_next[j, -1])
            r_d1 = np.sqrt(r_d**2 + extra**2)  # distance in dimension d+1
            # Kennel's two tests: a neighbour is false if adding the coordinate
            # blows up its distance relative to the d-D distance (rtol), OR if the
            # resulting (d+1)-D distance is large on the scale of the attractor
            # (atol). The second test catches the r_d ~ 0 ties the first cannot.
            test1 = (extra / r_d > rtol) if r_d > 0 else True
            test2 = r_d1 / attractor_size > atol
            if test1 or test2:
                false += 1
        fractions.append(false / m)
    return fractions


def largest_lyapunov_rosenstein(
    x: np.ndarray,
    tau: int,
    dim: int,
    mean_period: int = 1,
    max_t: int = 40,
) -> tuple[float, np.ndarray]:
    """Largest Lyapunov exponent via Rosenstein's algorithm.

    Track how the distance between initially-close embedded points grows. For a
    chaotic system the average log-distance rises linearly with time at a slope
    equal to the largest Lyapunov exponent lambda_1 (per sample). The prediction
    horizon then scales like 1/lambda_1 (Chapter 1).

    Returns (slope, divergence_curve). The curve is mean log-divergence vs. time,
    and the slope is a least-squares fit over its initial linear stretch. We fit
    the first third of the curve, which is the usual "scaling region" heuristic.
    """
    x = np.asarray(x, float)
    emb = delay_embed(x, dim=dim, tau=tau)
    n = len(emb)
    # Nearest neighbour of each point, excluding temporally close ones
    # (Theiler window) so we measure dynamical, not trivial, proximity.
    neighbour = np.full(n, -1, dtype=int)
    for i in range(n):
        d2 = np.sum((emb - emb[i]) ** 2, axis=1)
        d2[max(0, i - mean_period) : i + mean_period + 1] = np.inf
        neighbour[i] = int(np.argmin(d2))

    # Average log-distance of each pair after t steps.
    divergence = []
    for t in range(max_t):
        logs = []
        for i in range(n - t):
            j = neighbour[i]
            if j + t < n:
                dist = np.linalg.norm(emb[i + t] - emb[j + t])
                if dist > 0:
                    logs.append(np.log(dist))
        divergence.append(np.mean(logs) if logs else np.nan)
    divergence = np.array(divergence)

    # Least-squares slope over the initial (linear) scaling region.
    fit_end = max(2, max_t // 3)
    t_axis = np.arange(fit_end)
    good = np.isfinite(divergence[:fit_end])
    slope = np.polyfit(t_axis[good], divergence[:fit_end][good], 1)[0]
    return float(slope), divergence


# ---------------------------------------------------------------------------
# Surrogate-data test: is the structure genuine nonlinear determinism, or just a
# linear stochastic process dressed up by a fat-tailed amplitude distribution?
#
# A positive Lyapunov estimate or a finite embedding dimension on their own do
# not prove determinism: linearly correlated noise with a skewed/heavy-tailed
# histogram can fake both. The standard guard is to compare a nonlinearity
# statistic on the data against an ensemble of surrogates that share the data's
# power spectrum (linear correlations) and its amplitude distribution, but have
# any nonlinear structure randomised away. If the data statistic sits far outside
# the surrogate spread, the linear-stochastic null is rejected.
# ---------------------------------------------------------------------------
def nonlinear_prediction_error(x: np.ndarray, dim: int, tau: int,
                               n_neighbours: int = 4) -> float:
    """NRMSE of a locally-constant one-step predictor in delay space.

    For each embedded point, predict its next value as the mean next value of its
    nearest neighbours (excluding immediate temporal neighbours). Deterministic
    structure makes this error small; for noise it stays near 1. Used as the
    discriminating statistic for the surrogate test below.
    """
    x = np.asarray(x, float)
    emb = delay_embed(x, dim=dim, tau=tau)
    span = (dim - 1) * tau
    # Target is the sample one step after the most-recent coordinate of each row.
    n = len(emb) - 1
    emb = emb[:n]
    target = x[span + 1 : span + 1 + n]
    preds = np.empty(n)
    theiler = tau + 1  # exclude near-in-time neighbours
    for i in range(n):
        d2 = np.sum((emb - emb[i]) ** 2, axis=1)
        d2[max(0, i - theiler): i + theiler + 1] = np.inf
        nn = np.argsort(d2)[:n_neighbours]
        preds[i] = np.mean(target[nn])
    denom = np.std(target) or 1.0
    return float(np.sqrt(np.mean((target - preds) ** 2)) / denom)


def aaft_surrogate(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One amplitude-adjusted Fourier-transform (AAFT) surrogate.

    Preserves the amplitude distribution (histogram) and approximately the power
    spectrum (linear autocorrelations) of `x`, while randomising the Fourier
    phases so any nonlinear, deterministic structure is destroyed.
    """
    x = np.asarray(x, float)
    n = len(x)
    ranks = np.argsort(np.argsort(x))
    # Map the data onto Gaussian values of matching rank.
    gauss = np.sort(rng.standard_normal(n))[ranks]
    # Phase-randomise the Gaussian copy.
    spec = np.fft.rfft(gauss)
    phases = rng.uniform(0, 2 * np.pi, len(spec))
    phases[0] = 0.0
    if n % 2 == 0:
        phases[-1] = 0.0  # Nyquist term stays real
    surr_gauss = np.fft.irfft(np.abs(spec) * np.exp(1j * phases), n)
    # Re-impose the original amplitude distribution by rank.
    return np.sort(x)[np.argsort(np.argsort(surr_gauss))]


def surrogate_determinism_test(x: np.ndarray, tau: int, dim: int,
                               n_surrogates: int = 39, max_points: int = 1500,
                               seed: int = 0) -> dict:
    """Test the linear-stochastic null with AAFT surrogates.

    Compares the nonlinear prediction error of the data to that of `n_surrogates`
    AAFT surrogates. A z-score far below zero (data error much smaller than the
    surrogates') rejects the null: the predictability is genuine nonlinear
    determinism, not a linear process with a skewed histogram. 39 surrogates give
    a one-sided test at the 0.025 level by rank.

    Returns a dict with the data statistic, the surrogate mean/std, the z-score
    and a boolean `deterministic` (data error below every surrogate).
    """
    x = np.asarray(x, float)
    if len(x) > max_points:
        x = x[:max_points]  # cap the O(N^2) neighbour search
    rng = np.random.default_rng(seed)
    stat_data = nonlinear_prediction_error(x, dim=dim, tau=tau)
    stat_surr = np.array([
        nonlinear_prediction_error(aaft_surrogate(x, rng), dim=dim, tau=tau)
        for _ in range(n_surrogates)
    ])
    mu, sd = float(np.mean(stat_surr)), float(np.std(stat_surr))
    z = (stat_data - mu) / sd if sd > 0 else 0.0
    return {
        "stat_data": stat_data,
        "surr_mean": mu,
        "surr_std": sd,
        "z": float(z),
        "deterministic": bool(stat_data < stat_surr.min()),
    }
