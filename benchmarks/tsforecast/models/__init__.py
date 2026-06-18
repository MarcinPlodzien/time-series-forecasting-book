"""Forecasting models, one file per method, all sharing the `Forecaster` API."""

from .baselines import MeanForecast, Persistence
from .esn import ESN
from .linear_ar import LinearAR
from .nvar import NVAR
from .volterra import Volterra
from .koopman_edmd import KoopmanEDMD
from .quantum_reservoir import QuantumReservoir

# ARIMA needs statsmodels; neural models need torch. Import defensively so the
# classical suite still loads if an optional dependency is missing.
try:
    from .arima import ARIMA
except Exception:  # pragma: no cover
    ARIMA = None

try:
    from . import neural
    from .neural_ode import make_neural_ode
except Exception:  # pragma: no cover
    neural = None
    make_neural_ode = None

__all__ = [
    "Persistence", "MeanForecast", "LinearAR", "NVAR", "ESN", "Volterra",
    "KoopmanEDMD", "QuantumReservoir", "ARIMA", "neural", "make_neural_ode",
]
