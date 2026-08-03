"""Simulation-based inference of J-couplings from ZULF NMR spectra.

The package is layered and each layer talks only to the one below it:
operators and Hamiltonian, preparation, line list, detector and relaxation,
spectrum and peak list, parameterization and priors, inference.  Only the
top layer is allowed to depend on torch; everything here is numpy and scipy.
"""

from .systems import ACETONITRILE, FORMIC_ACID, METHANOL, SpinSystem, forward

__all__ = ["ACETONITRILE", "FORMIC_ACID", "METHANOL", "SpinSystem", "forward"]
