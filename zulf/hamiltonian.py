"""The ZULF Hamiltonian

    H = sum_{i<j} 2 pi J_ij  I_i . I_j  -  sum_i gamma_i B . I_i        [rad/s]

Sign convention: with J > 0 the singlet of a coupled pair lies lowest, and
a spin with gamma > 0 lies lowest along +B.  Both are the standard choices
and both are asserted in the test suite, because every later stage inherits
them.

BLOCKING WARNING
----------------
Total F_z is conserved only when the field is along the sensor axis.  A
transverse component mixes F_z sectors, which is exactly why it splits
lines and why it has to be kept.  So the 2^3 speedup from F_z blocking is
NOT available in the general case.  What is available in the general case
is blocking by the permutation symmetry of magnetically equivalent nuclei
(the methyl group of methanol, for instance).  Do not write code that
assumes F_z blocking works and then hand it a tilted field.
"""

import numpy as np

from .operators import dot_products, spin_operators


def coupling_matrix(n, pairs):
    """Build a symmetric n x n J matrix (Hz) from a {(i, j): J} mapping."""
    jmat = np.zeros((n, n), dtype=float)
    for (i, j), value in pairs.items():
        jmat[i, j] = value
        jmat[j, i] = value
    return jmat


def hamiltonian(jmat, gam, bfield=(0.0, 0.0, 0.0)):
    """H in rad/s.

    Parameters
    ----------
    jmat : (n, n) symmetric array of scalar couplings in Hz
    gam  : (n,) gyromagnetic ratios in rad s^-1 T^-1
    bfield : 3-vector of the residual field in T, sensor axis = z
    """
    jmat = np.asarray(jmat, dtype=float)
    gam = np.asarray(gam, dtype=float)
    n = len(gam)

    dots = dot_products(n)
    h = np.zeros((2 ** n, 2 ** n), dtype=complex)
    for i in range(n):
        for j in range(i + 1, n):
            if jmat[i, j] != 0.0:
                h += 2.0 * np.pi * jmat[i, j] * dots[i, j]

    bfield = np.asarray(bfield, dtype=float)
    if np.any(bfield):
        ix, iy, iz = spin_operators(n)
        for i in range(n):
            h -= gam[i] * (
                bfield[0] * ix[i] + bfield[1] * iy[i] + bfield[2] * iz[i]
            )
    return h


def field_vector(magnitude, theta):
    """Residual field from magnitude (T) and polar angle (rad) to sensor axis."""
    return np.array(
        [magnitude * np.sin(theta), 0.0, magnitude * np.cos(theta)], dtype=float
    )


def eigen(h):
    """Eigen-decomposition with eigenvalues returned in Hz, not rad/s."""
    evals, evecs = np.linalg.eigh(h)
    return evals / (2.0 * np.pi), evecs
