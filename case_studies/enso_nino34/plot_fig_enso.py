#!/usr/bin/env python3
"""Generate fig_enso.png — ENSO Niño 3.4 2×2 figure."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))
import numpy as np
from plotting import plot_signal_2x2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(SCRIPT_DIR, '..', '..')
data = np.loadtxt(os.path.join(BASE, 'data/enso_nino34/enso_nino34.txt'))
N = len(data)
# Monthly from ~1950
years = 1950.0 + np.arange(N) / 12.0

plot_signal_2x2(
    years, data, 'enso',
    fig_dir=os.path.join(BASE, 'figures/enso_nino34'),
    time_label='Year', signal_label='Niño 3.4 SST anomaly (°C)',
    fs=12.0, freq_label='Frequency (cycles/year)',
    acf_max_lag=200
)
