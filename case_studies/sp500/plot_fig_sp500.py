#!/usr/bin/env python3
"""Generate fig_sp500.png — S&P 500 log-returns 2×2 figure."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))
import numpy as np
from plotting import plot_signal_2x2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(SCRIPT_DIR, '..', '..')
prices = np.loadtxt(os.path.join(BASE, 'data/sp500/sp500_daily_close.txt'))
# Compute log-returns
returns = np.diff(np.log(prices))
N = len(returns)
# Daily from ~1928
years = 1928.0 + np.arange(N) / 252.0

plot_signal_2x2(
    years, returns, 'sp500',
    fig_dir=os.path.join(BASE, 'figures/sp500'),
    time_label='Year', signal_label='Daily log-return',
    fs=252.0, freq_label='Frequency (cycles/year)',
    acf_max_lag=200
)
