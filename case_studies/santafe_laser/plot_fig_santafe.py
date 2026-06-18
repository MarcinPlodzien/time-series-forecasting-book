#!/usr/bin/env python3
"""Generate fig_santafe.png — SantaFe laser 2×2 figure."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))
import numpy as np
from plotting import plot_signal_2x2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(SCRIPT_DIR, '..', '..')
data = np.loadtxt(os.path.join(BASE, 'data/santafe_laser/santafe.txt'))
time = np.arange(len(data))

plot_signal_2x2(
    time, data, 'santafe',
    fig_dir=os.path.join(BASE, 'figures/santafe_laser'),
    time_label='Sample index', signal_label='Intensity (a.u.)',
    fs=1.0, freq_label='Frequency (cycles/sample)',
    acf_max_lag=300
)
