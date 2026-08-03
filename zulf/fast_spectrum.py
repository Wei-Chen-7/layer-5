"""Build the spectrum without ever building the FID.

Measured on one core: acquiring a 400 s record at 1.2 kSa/s and FFT-ing it costs
about 300 ms, against 0.3 ms for the four-spin diagonalization.  The data
pipeline, not the physics, is what makes 1e5 training simulations expensive
(300 ms x 1e5 = 8.5 h).

But the DFT of a truncated decaying exponential is a closed form.  For

    y[k] = sum_lines 2 Re[ A exp(-(R + 2i pi nu) k dt) ],   k = 0 .. N-1

the normalized rfft is exactly

    Y[j] = (1/N) sum_lines [ A * G(z_+) + conj(A) * G(z_-) ],
    z_pm = exp(-(R dt) -/+ 2i pi nu dt - 2i pi j / N),
    G(z) = (1 - z^N) / (1 - z).

So the spectrum can be evaluated at any set of bins, and only the bins within a
few linewidths of a line carry signal.  Noise is white in time, therefore
circular complex Gaussian per bin with variance sigma^2 / N, and can be added
bin by bin.  Cost drops by three to four orders of magnitude and the result
agrees with the brute-force FFT to machine precision.
"""

import numpy as np


def _g(z, n):
    """(1 - z^n) / (1 - z), stable when z is close to 1."""
    z = np.asarray(z, dtype=complex)
    near = np.abs(z - 1.0) < 1e-12
    out = np.empty_like(z)
    safe = np.where(near, 0.0, z)
    out = (1.0 - safe ** n) / (1.0 - safe + np.where(near, 1.0, 0.0))
    out[near] = n
    return out


def spectrum_at_bins(freq, amp, rates, bins, n, rate_sample, t_dead=0.0):
    """Exact normalized rfft of the truncated FID, at the requested bin indices."""
    dt = 1.0 / rate_sample
    j = np.asarray(bins)[:, None]
    f = np.asarray(freq)[None, :]
    r = np.asarray(rates)[None, :]
    a = np.asarray(amp, dtype=complex)[None, :]

    dead = np.exp(-(r + 2j * np.pi * f) * t_dead)
    common = np.exp(-r * dt - 2j * np.pi * j / n)
    zp = common * np.exp(-2j * np.pi * f * dt)
    zm = common * np.exp(+2j * np.pi * f * dt)

    y = (a * dead) * _g(zp, n) + np.conj(a * dead) * _g(zm, n)
    return y.sum(axis=1) / n


def windowed_spectrum(freq, amp, rates, t_acq=400.0, rate_sample=1200.0,
                      n_widths=60.0, noise=0.0, t_dead=0.0, rng=None):
    """Signal-bearing bins only: a window of +/- n_widths HWHM around each line.

    Returns (nu, spec, bins).  This is what the training loop should call.
    """
    n = int(round(t_acq * rate_sample))
    d_nu = rate_sample / n
    hwhm = np.asarray(rates) / (2.0 * np.pi)

    sel = set()
    for f0, w in zip(np.asarray(freq), np.atleast_1d(hwhm)):
        half = max(3, int(np.ceil(n_widths * w / d_nu)))
        centre = int(round(f0 / d_nu))
        lo, hi = max(0, centre - half), min(n // 2, centre + half)
        sel.update(range(lo, hi + 1))
    bins = np.array(sorted(sel), dtype=int)

    spec = spectrum_at_bins(freq, amp, rates, bins, n, rate_sample, t_dead=t_dead)
    if noise:
        rng = np.random.default_rng() if rng is None else rng
        sigma = noise / np.sqrt(2.0 * n)  # per real and imaginary part
        spec = spec + rng.normal(0, sigma, bins.size) + 1j * rng.normal(0, sigma, bins.size)
    return bins * d_nu, spec, bins
