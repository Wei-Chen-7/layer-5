"""Every claim in the proposal, as a test.

If one of these breaks, something downstream is silently wrong, so they run
first and they run on every change.
"""

import numpy as np
import pytest

from zulf import ACETONITRILE, FORMIC_ACID, METHANOL, SpinSystem, forward
from zulf.constants import gammas
from zulf.hamiltonian import coupling_matrix, field_vector, hamiltonian
from zulf.operators import magnetization_z
from zulf.preparation import _adiabatic_populations, adiabatic_then_pulse, thermal_sudden
from zulf.signal import line_list

METHYL = {(0, 1): 140.6, (0, 2): 140.6, (0, 3): 140.6,
          (1, 2): -12.4, (1, 3): -12.4, (2, 3): -12.4}


def spectrum(system, jpairs, **kw):
    f, a, _ = forward(system, jpairs, **kw)
    keep = np.abs(np.real(a)) > 1e-6 * np.max(np.abs(np.real(a)))
    return f[keep], np.real(a)[keep]


# ---------------------------------------------------------------- known cases

def test_two_spin_gives_one_line_at_J():
    f, a = spectrum(FORMIC_ACID, {(0, 1): 221.0})
    assert f.size == 1
    assert abs(f[0] - 221.0) < 1e-9


def test_xa3_gives_J_and_2J_with_ratio_five_fourths():
    f, a = spectrum(METHANOL, METHYL)
    assert f.size == 2
    assert abs(f[0] - 140.6) < 1e-9
    assert abs(f[1] - 281.2) < 1e-9
    assert abs(a[1] / a[0] - 1.25) < 1e-9


def test_methyl_HH_coupling_is_invisible():
    ref = spectrum(METHANOL, METHYL)
    for jhh in (0.0, 5.0, -12.4, 100.0):
        pairs = dict(METHYL)
        pairs.update({(1, 2): jhh, (1, 3): jhh, (2, 3): jhh})
        f, a = spectrum(METHANOL, pairs)
        assert np.allclose(f, ref[0], atol=1e-9)
        assert np.allclose(a / a[0], ref[1] / ref[1][0], atol=1e-9)


def test_equal_gamma_gives_no_spectrum():
    sys4 = SpinSystem("four protons", ["1H"] * 4)
    rng = np.random.default_rng(0)
    pairs = {(i, j): rng.uniform(-20, 20) for i in range(4) for j in range(i + 1, 4)}
    f, a, _ = forward(sys4, pairs)
    _, a_ref, _ = forward(METHANOL, METHYL)
    assert np.max(np.abs(a), initial=0.0) < 1e-12 * np.max(np.abs(a_ref))


def test_line_count_matches_proposal_table():
    assert spectrum(FORMIC_ACID, {(0, 1): 221.0})[0].size == 1
    assert spectrum(METHANOL, METHYL)[0].size == 2
    pairs = {(0, 1): 56.6, (0, 2): -10.0, (0, 3): -10.0, (0, 4): -10.0,
             (1, 2): 136.0, (1, 3): 136.0, (1, 4): 136.0,
             (2, 3): -16.9, (2, 4): -16.9, (3, 4): -16.9}
    f, a = spectrum(ACETONITRILE, pairs)
    assert f.size <= 9


# ---------------------------------------------------------------- symmetries

def test_global_sign_flip_at_zero_field():
    pairs = {(0, 1): 140.6, (0, 2): 140.6, (0, 3): 140.6,
             (1, 2): -12.4, (1, 3): -12.4, (2, 3): -12.4}
    f1, a1 = spectrum(METHANOL, pairs)
    f2, a2 = spectrum(METHANOL, {k: -v for k, v in pairs.items()})
    assert np.allclose(f1, f2, atol=1e-9)
    assert np.allclose(a1, a2, rtol=1e-9)


@pytest.mark.parametrize("theta", [0.0, np.pi / 3, np.pi / 2])
def test_global_sign_flip_with_residual_field(theta):
    """The proposal asserts this holds with a field too.  It is checked, not assumed."""
    pairs = {(0, 1): 221.0}
    f1, a1 = spectrum(FORMIC_ACID, pairs, b_mag=1e-9, b_theta=theta)
    f2, a2 = spectrum(FORMIC_ACID, {k: -v for k, v in pairs.items()},
                      b_mag=1e-9, b_theta=theta)
    assert np.allclose(np.sort(f1), np.sort(f2), atol=1e-9)


def test_permutation_of_equal_gamma_nuclei():
    base = {(0, 1): 140.0, (0, 2): 120.0, (0, 3): 100.0,
            (1, 2): -12.0, (1, 3): -9.0, (2, 3): -5.0}
    perm = {0: 0, 1: 2, 2: 1, 3: 3}
    swapped = {}
    for (i, j), v in base.items():
        a, b = sorted((perm[i], perm[j]))
        swapped[(a, b)] = v
    f1, a1 = spectrum(METHANOL, base)
    f2, a2 = spectrum(METHANOL, swapped)
    assert np.allclose(np.sort(f1), np.sort(f2), atol=1e-9)


# ---------------------------------------------------------------- the field

def test_longitudinal_field_does_almost_nothing():
    f0, _ = spectrum(FORMIC_ACID, {(0, 1): 221.0})
    f1, _ = spectrum(FORMIC_ACID, {(0, 1): 221.0}, b_mag=1e-9, b_theta=0.0)
    assert f1.size == 1
    assert abs(f1[0] - f0[0]) < 1e-4  # ~2e-6 Hz, far under any linewidth


def test_transverse_field_splits_the_line():
    """Three lines around J, split by the SUM of the Larmor frequencies, plus a
    low-frequency line at their MEAN.  Use a general angle: at exactly theta =
    pi/2 the central component vanishes by symmetry and only two are left."""
    f, a = spectrum(FORMIC_ACID, {(0, 1): 221.0}, b_mag=1e-9, b_theta=np.pi / 3)
    gam = gammas(["13C", "1H"])
    larmor_sum = (gam[0] + gam[1]) * 1e-9 / (2 * np.pi)

    high = np.unique(np.round(f[f > 100.0], 9))
    assert high.size == 3
    gap = np.diff(high)
    assert np.allclose(gap, larmor_sum / 2, rtol=0.05)
    assert 0.02 < (high[-1] - high[0]) < 0.08  # ~53 mHz, 5x the best linewidth

    low = np.unique(np.round(f[f < 1.0], 9))
    assert np.allclose(low, larmor_sum / 2, rtol=0.05)


def test_exactly_transverse_field_kills_the_central_component():
    f, _ = spectrum(FORMIC_ACID, {(0, 1): 221.0}, b_mag=1e-9, b_theta=np.pi / 2)
    assert np.unique(np.round(f[f > 100.0], 9)).size == 2


# ---------------------------------------------------------------- preparation

def test_adiabatic_without_pulse_gives_no_signal():
    jmat = coupling_matrix(2, {(0, 1): 221.0})
    rho = _adiabatic_populations(jmat, FORMIC_ACID.gam)
    h = hamiltonian(jmat, FORMIC_ACID.gam)
    f, a = line_list(h, rho, magnetization_z(FORMIC_ACID.gam))
    _, a_ref, _ = forward(FORMIC_ACID, {(0, 1): 221.0}, prep="thermal")
    assert np.max(np.abs(a), initial=0.0) < 1e-12 * np.max(np.abs(a_ref))


@pytest.mark.parametrize("angle", [np.pi / 2, np.pi])
def test_pulse_acquire_weights_are_not_sign_definite(angle):
    f, a, _ = forward(FORMIC_ACID, {(0, 1): 221.0}, prep="pulse", proton_angle=angle)
    assert f.size >= 1
    scale = np.max(np.abs(np.real(a)))
    assert np.min(np.real(a)) < -1e-6 * scale or np.isclose(scale, 0.0, atol=1e-9)


def test_thermal_weights_are_non_negative():
    f, a, _ = forward(METHANOL, METHYL, prep="thermal")
    assert np.all(np.real(a) > -1e-9 * np.max(np.real(a)))


# ---------------------------------------------------------------- detector

def test_detector_distorts_the_five_fourths_ratio():
    f, a, _ = forward(METHANOL, METHYL, detector={"f_cut": 150.0, "f_lowpass": None})
    ratio = np.abs(a[1]) / np.abs(a[0])
    assert 0.80 < ratio < 0.82  # 5/4 pushed below 1: the taller line looks shorter
    # the ~62 deg in the proposal is the ABSOLUTE phase of the 2J line;
    # the phase difference the fit actually sees between the two lines is ~19 deg
    assert 60.0 < abs(np.degrees(np.angle(a[1]))) < 64.0
    rel = abs(np.degrees(np.angle(a[1]) - np.angle(a[0])))
    assert 17.0 < rel < 21.0


# ---------------------------------------------------------------- data pipeline

def test_analytic_spectrum_matches_brute_force_fft():
    from zulf.fast_spectrum import spectrum_at_bins
    from zulf.spectrum import acquire

    f, a, r = forward(METHANOL, METHYL, rate=0.05)
    t_acq, fs = 40.0, 1200.0
    nu, s_fft = acquire(f, a, r, t_acq=t_acq, rate_sample=fs)
    n = int(round(t_acq * fs))
    s_exact = spectrum_at_bins(f, a, r, np.arange(nu.size), n, fs)
    assert np.abs(s_exact - s_fft).max() < 1e-10 * np.abs(s_fft).max()


def test_windowed_spectrum_recovers_the_peaks():
    from zulf.fast_spectrum import windowed_spectrum
    from zulf.spectrum import pick_peaks

    f, a, r = forward(METHANOL, METHYL, rate=0.05)
    nu, s, bins = windowed_spectrum(f, a, r, t_acq=400.0, rate_sample=1200.0)
    assert bins.size < 2000  # not 240000
    pk = pick_peaks(nu, s, threshold=0.05 * np.abs(s).max())
    got = np.sort(pk[:, 0])
    assert abs(got[0] - 140.6) < 1e-3 and abs(got[1] - 281.2) < 1e-3
    ratio = pk[np.argmax(pk[:, 0]), 1] / pk[np.argmin(pk[:, 0]), 1]
    assert abs(ratio - 1.25) < 1e-3
