"""Line list -> FID -> spectrum -> peak list.

The peak list is what the network sees.  Critical rule: the SAME peak picker
must run on simulated and on real data.  If the network is trained on the exact
line list and then shown peaks found by a picker, it has been trained on a
different measurement than the one it is asked about, and no amount of prior
width will repair that.
"""

import numpy as np


def fid(freq, amp, rates, t, t_dead=0.0):
    """Real time-domain signal.

    rates : (K,) decay rate per line in s^-1.  One rate per resolved multiplet
            is the sensible starting model; one rate for the whole spectrum is
            not right, because different ZULF coherences decay differently.
    """
    tt = np.asarray(t)[:, None] + t_dead
    phase = np.exp(-2j * np.pi * np.asarray(freq)[None, :] * tt)
    decay = np.exp(-np.asarray(rates)[None, :] * tt)
    return 2.0 * np.real((np.asarray(amp)[None, :] * phase * decay).sum(axis=1))


def acquire(freq, amp, rates, t_acq=400.0, rate_sample=2000.0, noise=0.0,
            t_dead=0.0, rng=None):
    """Simulate one recording.  Returns (nu, complex spectrum)."""
    n = int(round(t_acq * rate_sample))
    t = np.arange(n) / rate_sample
    y = fid(freq, amp, rates, t, t_dead=t_dead)
    if noise:
        rng = np.random.default_rng() if rng is None else rng
        y = y + rng.normal(0.0, noise, size=n)
    spec = np.fft.rfft(y) / n
    nu = np.fft.rfftfreq(n, d=1.0 / rate_sample)
    return nu, spec


def pick_peaks(nu, spec, threshold, max_peaks=64):
    """Local maxima of the magnitude spectrum, with sub-bin refinement.

    Returns an array of (frequency, amplitude, width) triples, the input
    format the closest published NMR work uses.
    """
    mag = np.abs(spec)
    interior = np.arange(1, mag.size - 1)
    is_max = (mag[interior] > mag[interior - 1]) & (mag[interior] > mag[interior + 1])
    cand = interior[is_max & (mag[interior] > threshold)]
    if cand.size == 0:
        return np.zeros((0, 3))

    cand = cand[np.argsort(mag[cand])[::-1][:max_peaks]]
    d_nu = nu[1] - nu[0]

    y0, y1, y2 = mag[cand - 1], mag[cand], mag[cand + 1]
    denom = y0 - 2 * y1 + y2
    shift = np.where(np.abs(denom) > 0, 0.5 * (y0 - y2) / np.where(denom == 0, 1, denom), 0.0)
    shift = np.clip(shift, -0.5, 0.5)
    f_hat = nu[cand] + shift * d_nu
    a_hat = y1 - 0.25 * (y0 - y2) * shift

    # half width at half maximum, read off the same parabola
    curv = np.abs(denom)
    width = np.where(curv > 0, d_nu * np.sqrt(np.maximum(a_hat, 0) / np.maximum(curv, 1e-300)), d_nu)

    out = np.column_stack([f_hat, a_hat, width])
    return out[np.argsort(out[:, 0])]


def pad_peaks(peaks, k=32):
    """Fixed-size padded view, for a network that wants a rectangular input.

    Prefer a permutation-invariant set encoder (DeepSets / set transformer)
    over this, and keep the mask.
    """
    out = np.zeros((k, 4))
    m = min(k, len(peaks))
    out[:m, :3] = peaks[:m]
    out[:m, 3] = 1.0
    return out
