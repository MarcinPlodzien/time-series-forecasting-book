#!/usr/bin/env python3
"""Generate fig_fremantle.png — Fremantle sea level 2×2 figure."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))
import numpy as np
from plotting import plot_signal_2x2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(SCRIPT_DIR, '..', '..')
data = np.loadtxt(os.path.join(BASE, 'data/fremantle/fremantle_monthly_level.txt'))
years_arr = np.loadtxt(os.path.join(BASE, 'data/fremantle/fremantle_monthly_years.txt'))
N = len(data)
if len(years_arr) == N:
    time = years_arr
else:
    time = 1897.0 + np.arange(N) / 12.0

plot_signal_2x2(
    time, data, 'fremantle',
    fig_dir=os.path.join(BASE, 'figures/fremantle'),
    time_label='Year', signal_label='Sea level (mm)',
    fs=12.0, freq_label='Frequency (cycles/year)',
    acf_max_lag=200
)
