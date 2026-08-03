"""Initial states rho(0) for the ZULF preparation sequences.

The line POSITIONS never depend on rho(0); only the weights do.  That is why
the preparation can be carried as a handful of nuisance parameters instead of
being assumed.  All states here are the traceless part only, since the
identity contributes nothing to Tr[rho M_z].
"""

import numpy as np
from scipy.linalg import expm

from .hamiltonian import hamiltonian
from .operators import fz_blocks, magnetization_z, spin_operators


def thermal_sudden(gam):
    """Prepolarize along the sensor axis, then drop the field suddenly.

    rho(0) propto M_z exactly.  This is the only protocol for which the
    weights reduce to |<n|M_z|m>|^2 and are therefore non-negative.
    """
    return magnetization_z(gam)


def _adiabatic_populations(jmat, gam, b_high=1.0):
    """Populations after an adiabatic drop from high field to zero field.

    F_z is conserved along a ramp with the field on the sensor axis, so the
    adiabatic theorem maps the k-th lowest state of a given F_z block at high
    field onto the k-th lowest state of the same block at zero field.
    """
    n = len(gam)
    h_hi = hamiltonian(jmat, gam, (0.0, 0.0, b_high))
    h_lo = hamiltonian(jmat, gam, (0.0, 0.0, 0.0))

    rho = np.zeros((2 ** n, 2 ** n), dtype=complex)
    mz = magnetization_z(gam)

    for idx in fz_blocks(n):
        sub_hi = h_hi[np.ix_(idx, idx)]
        sub_lo = h_lo[np.ix_(idx, idx)]
        e_hi, v_hi = np.linalg.eigh(sub_hi)
        _, v_lo = np.linalg.eigh(sub_lo)
        # high-field thermal weight of each high-field eigenstate
        mz_sub = mz[np.ix_(idx, idx)]
        pops = np.real(np.einsum("kn,kl,ln->n", v_hi.conj(), mz_sub, v_hi))
        # rank-preserving transfer onto the zero-field eigenstates
        order = np.argsort(e_hi)
        pops = pops[order]
        block = (v_lo * pops) @ v_lo.conj().T
        rho[np.ix_(idx, idx)] = block
    return rho


def dc_pulse(gam, proton_angle, axis="x", reference="1H"):
    """Propagator for a hard DC pulse.

    One field pulse rotates every nucleus, by an angle proportional to its
    gyromagnetic ratio.  The free parameter is therefore a single angle; the
    others follow.
    """
    from .constants import GAMMA

    n = len(gam)
    ix, iy, _ = spin_operators(n)
    op = ix if axis == "x" else iy
    scale = proton_angle / GAMMA[reference]
    generator = np.tensordot(np.asarray(gam, dtype=complex) * scale, op, axes=(0, 0))
    return expm(-1j * generator)


def adiabatic_then_pulse(jmat, gam, proton_angle, axis="x"):
    """The usual pulse-acquire sequence.

    Without the pulse this state commutes with H and gives no signal at all.
    With the pulse the weights depend on the angle and are not sign-definite.
    """
    rho = _adiabatic_populations(jmat, gam)
    u = dc_pulse(gam, proton_angle, axis=axis)
    return u @ rho @ u.conj().T


def singlet_order(n, pair=(0, 1)):
    """Singlet order on one pair, the hyperpolarization starting point.

    Commutes with an isolated pair Hamiltonian, so it too needs a pulse.
    """
    ix, iy, iz = spin_operators(n)
    i, j = pair
    dot = ix[i] @ ix[j] + iy[i] @ iy[j] + iz[i] @ iz[j]
    return -dot  # traceless part of |S><S|, up to a positive factor
