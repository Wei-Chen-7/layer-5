"""Magnetometer response.

The OPM in this group's recent ZULF work has a 3 dB bandwidth of 150 Hz and a
hardware low-pass at 500 Hz, while the J lines sit at 140 to 300 Hz.  For a
methyl group the true 2J : J intensity ratio is exactly 5/4; through a single
pole at 150 Hz it becomes about 0.81, so the taller line looks shorter, with a
large phase shift on top.  Any parameter the fit is free to move will absorb
that as bias.

On a line list the response is exact and free: multiply each complex weight by
the transfer function at that line's frequency.
"""

import numpy as np


def single_pole(freq, f_cut):
    return 1.0 / (1.0 + 1j * freq / f_cut)


def response(freq, f_cut=150.0, f_lowpass=500.0, order_lowpass=1, gain=1.0, phase0=0.0):
    """Cascade of the sensor pole and the hardware anti-alias filter."""
    h = single_pole(freq, f_cut)
    if f_lowpass is not None:
        h = h * single_pole(freq, f_lowpass) ** order_lowpass
    return gain * np.exp(1j * phase0) * h


def apply(freq, amp, **kwargs):
    """Fold the detector response into the line weights."""
    return amp * response(freq, **kwargs)
