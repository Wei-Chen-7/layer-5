"""The benchmark molecules, plus a single-entry forward model."""

import numpy as np

from .constants import gammas
from .detector import apply as apply_detector
from .hamiltonian import coupling_matrix, field_vector, hamiltonian
from .operators import magnetization_z
from .preparation import adiabatic_then_pulse, singlet_order, thermal_sudden
from .signal import line_list


class SpinSystem:
    def __init__(self, name, labels, equivalent=()):
        self.name = name
        self.labels = list(labels)
        self.gam = gammas(labels)
        self.n = len(labels)
        # tuples of indices that are magnetically equivalent; used to build the
        # symmetry-distinct parameterization and to fix the permutation gauge
        self.equivalent = tuple(tuple(g) for g in equivalent)

    def __repr__(self):
        return f"SpinSystem({self.name!r}, {self.n} spins)"


FORMIC_ACID = SpinSystem("[13C]formic acid", ["13C", "1H"])
METHANOL = SpinSystem("[13C]methanol methyl (XA3)", ["13C", "1H", "1H", "1H"],
                      equivalent=[(1, 2, 3)])
ACETONITRILE = SpinSystem("[1,2-13C2]acetonitrile", ["13C", "13C", "1H", "1H", "1H"],
                          equivalent=[(2, 3, 4)])


def forward(system, jpairs, b_mag=0.0, b_theta=0.0, prep="thermal", proton_angle=np.pi / 2,
            rate=0.05, detector=None):
    """Couplings and nuisance parameters in, line list out.

    Parameters
    ----------
    jpairs : {(i, j): J_Hz}
    b_mag, b_theta : residual field magnitude (T) and angle to the sensor axis
    prep : 'thermal' | 'pulse' | 'singlet'
    rate : scalar or per-line decay rate, s^-1
    detector : None, or a dict of keyword arguments for detector.response
    """
    jmat = coupling_matrix(system.n, jpairs)
    h = hamiltonian(jmat, system.gam, field_vector(b_mag, b_theta))
    mz = magnetization_z(system.gam)

    if prep == "thermal":
        rho0 = thermal_sudden(system.gam)
    elif prep == "pulse":
        rho0 = adiabatic_then_pulse(jmat, system.gam, proton_angle)
    elif prep == "singlet":
        from .preparation import dc_pulse
        u = dc_pulse(system.gam, proton_angle)
        rho0 = u @ singlet_order(system.n) @ u.conj().T
    else:
        raise ValueError(f"unknown preparation {prep!r}")

    freq, amp = line_list(h, rho0, mz)
    if detector is not None:
        amp = apply_detector(freq, amp, **detector)
    rates = np.full(freq.shape, rate, dtype=float) if np.isscalar(rate) else np.asarray(rate)
    return freq, amp, rates
