#!/usr/bin/env python3
"""Generate fig_sunspot.png — Sunspot 2×2 figure."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))
import numpy as np
from plotting import plot_signal_2x2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(SCRIPT_DIR, '..', '..')
data = np.loadtxt(os.path.join(BASE, 'data/sunspot/sunspot_monthly.txt'))
N = len(data)
# Monthly data starting ~1749
years = 1749.0 + np.arange(N) / 12.0

plot_signal_2x2(
    years, data, 'sunspot',
    fig_dir=os.path.join(BASE, 'figures/sunspot'),
    time_label='Year', signal_label='Monthly sunspot number',
    fs=12.0, freq_label='Frequency (cycles/year)',
    acf_max_lag=300
)
