"""Spin-1/2 operators for an N-spin system.

Everything is dense.  For N <= 10 (dim 1024) dense eigendecomposition is
still the cheapest thing to write and fast enough; sparsity does not pay
until well past the range where 8^N has already killed you.

Operators are cached per N because building them is pure overhead inside
a simulation loop.
"""

from functools import lru_cache

import numpy as np

SX = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
SY = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
ID = np.eye(2, dtype=complex)


def _embed(op, i, n):
    """Place a single-spin operator on site i of an n-spin chain."""
    out = np.array([[1.0 + 0j]])
    for k in range(n):
        out = np.kron(out, op if k == i else ID)
    return out


@lru_cache(maxsize=16)
def spin_operators(n):
    """Return (Ix, Iy, Iz), each of shape (n, 2**n, 2**n)."""
    ix = np.stack([_embed(SX, i, n) for i in range(n)])
    iy = np.stack([_embed(SY, i, n) for i in range(n)])
    iz = np.stack([_embed(SZ, i, n) for i in range(n)])
    ix.flags.writeable = False
    iy.flags.writeable = False
    iz.flags.writeable = False
    return ix, iy, iz


@lru_cache(maxsize=16)
def dot_products(n):
    """Return the n x n array of operators I_i . I_j (only i<j is filled)."""
    ix, iy, iz = spin_operators(n)
    dim = 2 ** n
    out = np.zeros((n, n, dim, dim), dtype=complex)
    for i in range(n):
        for j in range(i + 1, n):
            out[i, j] = ix[i] @ ix[j] + iy[i] @ iy[j] + iz[i] @ iz[j]
    out.flags.writeable = False
    return out


@lru_cache(maxsize=16)
def fz_blocks(n):
    """Index lists grouped by total m_z.

    Only usable when the field is along the sensor axis (or zero), because
    a transverse field does not commute with F_z.  See hamiltonian.py.
    """
    m = np.array([bin(k).count("1") for k in range(2 ** n)])
    mz = 0.5 * n - m  # basis state k has m ones = spins down
    blocks = []
    for value in np.unique(mz)[::-1]:
        blocks.append(np.flatnonzero(mz == value))
    return tuple(blocks)


def magnetization_z(gam):
    """Observable M_z = sum_i gamma_i I_iz along the sensor axis."""
    n = len(gam)
    _, _, iz = spin_operators(n)
    return np.tensordot(np.asarray(gam, dtype=complex), iz, axes=(0, 0))
