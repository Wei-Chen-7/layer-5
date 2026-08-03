"""Line lists.

The central design decision of this codebase: the forward model returns a
LIST OF LINES (frequency, complex amplitude), not a binned spectrum.

Why it matters.  A 10 mHz linewidth over a 500 Hz band needs ~2e5 bins.
Carrying 2e5 floats through 1e5 training simulations is 160 GB.  Carrying
~40 lines through the same 1e5 simulations is 30 MB.  Detector response,
relaxation and phase are all exactly representable on the line list and cost
nothing there.  Binning happens once, at the very end, only to reproduce
whatever the real instrument did to the real data.

    S(t) = sum_{n,m} rho_nm(0) <m|M_z|n> exp(-i omega_nm t)

Lines with omega > 0 are kept; the negative-frequency partners are their
complex conjugates and are reinstated when the FID is built.
"""

import numpy as np

from .hamiltonian import eigen


def line_list(h, rho0, obs, amp_tol=1e-12, merge_tol=1e-9, fmin=1e-6):
    """Transitions of H with their complex weights.

    Returns
    -------
    freq : (K,) line frequencies in Hz, ascending
    amp  : (K,) complex weights; the real FID is
           sum_k 2 * Re[amp_k * exp(-2i pi freq_k t)]
    """
    energies, vecs = eigen(h)  # energies in Hz
    rho_e = vecs.conj().T @ rho0 @ vecs
    obs_e = vecs.conj().T @ obs @ vecs

    # amp[n, m] = rho_e[n, m] * obs_e[m, n]
    amp = rho_e * obs_e.T
    freq = energies[:, None] - energies[None, :]

    keep = freq > fmin
    freq = freq[keep]
    amp = amp[keep]

    strong = np.abs(amp) > amp_tol
    freq, amp = freq[strong], amp[strong]
    if freq.size == 0:
        return np.zeros(0), np.zeros(0, dtype=complex)

    order = np.argsort(freq)
    freq, amp = freq[order], amp[order]

    # merge degenerate transitions
    cut = np.flatnonzero(np.diff(freq) > merge_tol) + 1
    groups = np.split(np.arange(freq.size), cut)
    fout = np.array([np.average(freq[g], weights=np.abs(amp[g]) + 1e-300) for g in groups])
    aout = np.array([amp[g].sum() for g in groups])

    strong = np.abs(aout) > amp_tol
    return fout[strong], aout[strong]


def intensities(amp):
    """Absorption-mode peak intensity, i.e. the in-phase part of each weight."""
    return np.real(amp)
