"""
tsforecast
==========

A small, heavily commented forecasting library written as the companion code to
the book *Time Series Forecasting: A Dynamical-Systems Approach*. The modules are:

    datasets   -- fix the forecasting problem (which signal, which split),
    embedding  -- diagnose the signal (Part I: delay, dimension, Lyapunov),
    metrics    -- score (per-sample error and prediction horizon),
    base       -- one common Forecaster interface,
    models/    -- the architectures, organised by their inductive structure.

The suite tests, on the Santa Fe laser, whether a model whose structure matches
the signal's dynamics forecasts well, and whether a mismatched one fails in a way
the diagnosis anticipates.
"""

from . import datasets, embedding, metrics, models
from .base import Forecaster, Standardiser

__all__ = ["datasets", "embedding", "metrics", "models", "Forecaster", "Standardiser"]
