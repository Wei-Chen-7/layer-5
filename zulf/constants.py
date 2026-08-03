"""Gyromagnetic ratios and unit conventions.

Convention used everywhere in this package:

    J          in Hz
    gamma      in rad s^-1 T^-1
    B          in T
    H          in rad s^-1   (i.e. energy / hbar, hbar = 1)
    frequency  in Hz         (omega / 2 pi)

The sensor axis is the laboratory z axis.  The residual field is
B = |B| (sin(theta), 0, cos(theta)); by symmetry about the sensor axis
the azimuthal angle is unobservable, so the field is two parameters.
"""

import numpy as np

# gamma / 2pi in Hz/T
GAMMA_OVER_2PI = {
    "1H": 42.577478e6,
    "2H": 6.536e6,
    "13C": 10.7084e6,
    "15N": -4.3173e6,
    "19F": 40.078e6,
    "31P": 17.235e6,
}

GAMMA = {k: 2.0 * np.pi * v for k, v in GAMMA_OVER_2PI.items()}


def gammas(labels):
    """Gyromagnetic ratios in rad s^-1 T^-1 for a list of nucleus labels."""
    return np.array([GAMMA[l] for l in labels], dtype=float)
