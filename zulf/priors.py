"""Layer 5: priors on the symmetry-distinct parameters, and the movement
diagnostic that says whether the data moved them.

WHY THE COUPLINGS DO NOT GET A FLAT PRIOR
-----------------------------------------
DFT and ML predictors give 1J_CH to about 1 Hz.  That is useless as a
measurement, since the whole point of the experiment is a number good to
better than a mHz, and it is excellent as a prior, since it narrows the
search by two orders of magnitude without assuming the answer.  So the
coupling prior is normal, centred on a supplied prediction, with a supplied
width.  The width is an argument rather than a constant precisely so that the
sensitivity check -- widen it and see whether the posterior moves -- is one
keyword away.  Where no prediction exists the fallback is a wide normal
chosen by coupling type, and those are wide enough to be honest about the
fact that nothing is known.

WHY THE NUISANCE PRIORS ARE DELIBERATELY WIDE
---------------------------------------------
A network trained on a world messier than the real one transfers, and one
trained on a tidy world does not.  Every nuisance range below is wider than
the instrument is believed to be, and that is the intent, not sloppiness.
The cost is training samples; the alternative is a simulation-to-real gap
that shows up as a confidently wrong coupling.

THE GAUGE ENTERS THE PRIOR
--------------------------
The global sign flip of `parameters.canonical_sign` is a two-to-one map on
the coupling block.  A prior written on the unquotiented space therefore has
to be *folded* before it can be compared with a posterior reported on the
quotient: the density at a canonical theta is the sum of the densities at
theta and at its sign-flipped partner.  `SignGaugedPrior` does that, and it
matters whenever the coupling prior has appreciable mass on both sides of
zero, which is exactly the case for the wide fallbacks.  Comparing an
unfolded prior with a folded posterior is the sort of mistake that shows up
as a spurious factor of two in a KL and nowhere else.
"""

import numpy as np

from .parameters import (canonical_sign, is_canonical, orbit_index, orbit_names,
                         orbits)

__all__ = [
    "Normal",
    "Uniform",
    "LogUniform",
    "SineAngle",
    "Prior",
    "SignGaugedPrior",
    "FALLBACK_COUPLING",
    "coupling_components",
    "nuisance_components",
    "default_prior",
    "movement",
    "Movement",
]


# ------------------------------------------------------------- 1-D components

class _Component:
    """One scalar parameter.  Subclasses give `sample` and `log_prob`."""

    lo = -np.inf
    hi = np.inf

    def sample(self, n, rng):
        raise NotImplementedError

    def log_prob(self, x):
        raise NotImplementedError

    def __repr__(self):
        args = ", ".join(f"{k}={v!r}" for k, v in vars(self).items())
        return f"{type(self).__name__}({args})"


class Normal(_Component):
    """The prediction-centred coupling prior, and nothing else so far."""

    def __init__(self, mu, sigma):
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self.mu = float(mu)
        self.sigma = float(sigma)

    def sample(self, n, rng):
        return rng.normal(self.mu, self.sigma, size=n)

    def log_prob(self, x):
        z = (np.asarray(x, dtype=float) - self.mu) / self.sigma
        return -0.5 * z * z - np.log(self.sigma) - 0.5 * np.log(2.0 * np.pi)


class Uniform(_Component):
    def __init__(self, lo, hi):
        if not hi > lo:
            raise ValueError("need hi > lo")
        self.lo = float(lo)
        self.hi = float(hi)

    def sample(self, n, rng):
        return rng.uniform(self.lo, self.hi, size=n)

    def log_prob(self, x):
        x = np.asarray(x, dtype=float)
        inside = (x >= self.lo) & (x <= self.hi)
        return np.where(inside, -np.log(self.hi - self.lo), -np.inf)


class LogUniform(_Component):
    """Scale parameters get this, because their uncertainty is in orders of
    magnitude and a flat prior on a positive scale is really a prior that the
    answer is large."""

    def __init__(self, lo, hi):
        if not 0 < lo < hi:
            raise ValueError("need 0 < lo < hi")
        self.lo = float(lo)
        self.hi = float(hi)

    def sample(self, n, rng):
        return np.exp(rng.uniform(np.log(self.lo), np.log(self.hi), size=n))

    def log_prob(self, x):
        x = np.asarray(x, dtype=float)
        inside = (x >= self.lo) & (x <= self.hi)
        safe = np.where(inside, x, 1.0)
        return np.where(
            inside, -np.log(safe) - np.log(np.log(self.hi / self.lo)), -np.inf
        )


class SineAngle(_Component):
    """Polar angle of an isotropically oriented residual field.

    p(theta) = sin(theta)/2 on [0, pi].  Uniform in theta would pile the
    field up along the sensor axis, which is the one direction where it does
    almost nothing (a 1 nT longitudinal field moves the line by 2e-6 Hz,
    against a 53 mHz splitting when it is transverse).  A prior that avoids
    the informative geometry is a prior that trains the network on the easy
    case.
    """

    lo = 0.0
    hi = np.pi

    def sample(self, n, rng):
        return np.arccos(1.0 - 2.0 * rng.uniform(0.0, 1.0, size=n))

    def log_prob(self, x):
        x = np.asarray(x, dtype=float)
        inside = (x >= 0.0) & (x <= np.pi)
        safe = np.where(inside, x, 0.5 * np.pi)
        # sin vanishes at both ends, where the density really is zero, so the
        # divide-by-zero is the right answer rather than a problem to patch
        with np.errstate(divide="ignore"):
            return np.where(inside, np.log(np.sin(safe)) - np.log(2.0), -np.inf)


# ------------------------------------------------------------------ the prior

class Prior:
    """A product of independent named scalar priors.

    Layer 6 sees only `names`, `sample` and `log_prob`, so the whole object
    can be swapped for a correlated prior, or for the posterior of a previous
    round, without anything downstream noticing.
    """

    def __init__(self, components):
        components = list(components)
        names = [name for name, _ in components]
        if len(set(names)) != len(names):
            raise ValueError("parameter names must be unique")
        self.names = names
        self.components = [dist for _, dist in components]

    @property
    def ndim(self):
        return len(self.names)

    def index(self, name):
        return self.names.index(name)

    def support(self):
        """(lo, hi) arrays, for samplers that want a bounding box."""
        return (
            np.array([c.lo for c in self.components]),
            np.array([c.hi for c in self.components]),
        )

    def sample(self, n, rng=None):
        """Draw n parameter vectors, shape (n, ndim)."""
        rng = np.random.default_rng() if rng is None else rng
        cols = [c.sample(n, rng) for c in self.components]
        return np.stack(cols, axis=-1) if cols else np.zeros((n, 0))

    def log_prob(self, theta):
        """Log density.  Scalar for one vector, (n,) for a stack of them."""
        theta = np.asarray(theta, dtype=float)
        single = theta.ndim == 1
        theta = np.atleast_2d(theta)
        if theta.shape[1] != self.ndim:
            raise ValueError(
                f"prior has {self.ndim} parameters, got {theta.shape[1]}"
            )
        out = np.zeros(theta.shape[0])
        for k, comp in enumerate(self.components):
            out = out + comp.log_prob(theta[:, k])
        return float(out[0]) if single else out

    def as_dict(self, theta):
        """One parameter vector as {name: value}, for printing and for tests."""
        theta = np.asarray(theta, dtype=float).ravel()
        return dict(zip(self.names, theta))

    def __repr__(self):
        # the class name, not a literal, so a gauged prior says so: whether
        # the fold is on changes what log_prob means
        return (f"{type(self).__name__}({self.ndim} parameters: "
                f"{', '.join(self.names)})")


class SignGaugedPrior(Prior):
    """A Prior folded onto the quotient by the global coupling sign flip.

    Samples come out obeying the convention in `parameters.canonical_sign`,
    and the density is the sum over the two preimages, so it is still
    normalized on the canonical half-space.  Outside that half-space the
    density is zero, because those points are not parameters, they are the
    same parameters written the other way round.
    """

    def __init__(self, components, n_couplings):
        super().__init__(components)
        if not 0 <= n_couplings <= self.ndim:
            raise ValueError("n_couplings out of range")
        self.n_couplings = int(n_couplings)

    def _flip(self, theta):
        out = theta.copy()
        out[:, : self.n_couplings] *= -1.0
        return out

    def sample(self, n, rng=None):
        theta = super().sample(n, rng)
        theta[:, : self.n_couplings] = canonical_sign(theta[:, : self.n_couplings])
        return theta

    def log_prob(self, theta):
        theta = np.asarray(theta, dtype=float)
        single = theta.ndim == 1
        theta = np.atleast_2d(theta)
        here = np.atleast_1d(super().log_prob(theta))
        there = np.atleast_1d(super().log_prob(self._flip(theta)))
        out = np.logaddexp(here, there)
        ok = is_canonical(theta[:, : self.n_couplings])
        out = np.where(np.atleast_1d(ok), out, -np.inf)
        return float(out[0]) if single else out


# --------------------------------------------------------- coupling structure

# Wide normals used where no predicted coupling was supplied, keyed by the
# number of bonds between the two nuclei.  All are centred on zero: the sign
# of a coupling of a given type varies between molecules, and a zero-centred
# prior is also the only one that survives the global-sign gauge without
# quietly preferring one branch of it.  The widths cover the tabulated ranges
# with room to spare -- |1J_CH| runs 120 to 250 Hz, |1J_CC| 35 to 60 Hz,
# 2J and 3J proton couplings are single digits to low tens.
FALLBACK_COUPLING = {
    "one-bond": (0.0, 90.0),
    "two-bond": (0.0, 20.0),
    "long-range": (0.0, 8.0),
    "unknown": (0.0, 100.0),
}


def _bond_type(nbonds):
    if nbonds is None:
        return "unknown"
    nbonds = int(nbonds)
    if nbonds <= 0:
        raise ValueError("a coupling spans at least one bond")
    return {1: "one-bond", 2: "two-bond"}.get(nbonds, "long-range")


def _by_orbit(system, mapping, what):
    """Accept a mapping keyed by orbit index or by any pair in the orbit.

    Callers hold couplings as {(i, j): value} because that is what the
    forward model takes, so requiring them to convert to orbit indices first
    is an invitation to convert wrongly.  A mapping that names two members of
    one orbit with different values is an error for the same reason
    `from_pairs` raises: the caller and the symmetry declaration disagree.
    """
    n = len(orbits(system))
    out = [None] * n
    if not mapping:
        return out
    where = orbit_index(system)
    for key, value in mapping.items():
        if isinstance(key, (int, np.integer)):
            k = int(key)
            if not 0 <= k < n:
                raise ValueError(f"{what}: orbit {k} does not exist")
        else:
            i, j = (key[0], key[1]) if key[0] < key[1] else (key[1], key[0])
            if (i, j) not in where:
                raise ValueError(f"{what}: {(i, j)} is not a pair of {system.name}")
            k = where[(i, j)]
        value = float(value)
        if out[k] is not None and not np.isclose(out[k], value, rtol=0, atol=1e-12):
            raise ValueError(
                f"{what}: orbit {k} of {system.name} was given two different "
                f"values, {out[k]:.12g} and {value:.12g}"
            )
        out[k] = value
    return out


def coupling_components(system, predicted=None, width=1.0, nbonds=None):
    """One normal per orbit, centred on the prediction where there is one.

    Parameters
    ----------
    predicted : mapping to predicted J in Hz, keyed by orbit index or by any
        pair in the orbit.  Orbits left out fall back on `nbonds`.
    width : the prior standard deviation in Hz for the predicted orbits,
        either a scalar for all of them or a mapping keyed the same way as
        `predicted`.  One Hz is the accuracy claimed by current predictors;
        pass something larger to check that the posterior does not depend on
        having believed them.
    nbonds : mapping to the number of bonds spanned, keyed the same way, used
        only for the orbits with no prediction.
    """
    names = orbit_names(system)
    mu = _by_orbit(system, predicted, "predicted")
    sd = _by_orbit(system, width, "width") if isinstance(width, dict) else None
    bonds = _by_orbit(system, nbonds, "nbonds")

    out = []
    for k, name in enumerate(names):
        if mu[k] is None:
            centre, sigma = FALLBACK_COUPLING[_bond_type(bonds[k])]
            out.append((name, Normal(centre, sigma)))
            continue
        sigma = float(width) if sd is None else sd[k]
        if sigma is None:
            raise ValueError(
                f"orbit {k} ({name}) of {system.name} has a predicted coupling "
                "but no width.  A prediction without a stated uncertainty is "
                "not a prior."
            )
        out.append((name, Normal(mu[k], sigma)))
    return out


# ----------------------------------------------------------------- nuisances

def nuisance_components(n_rates=1, overrides=None):
    """The nuisance block, one entry per parameter, deliberately wide.

    n_rates is the number of resolved multiplets: one decay rate each is the
    starting model, because different ZULF coherences demonstrably do not
    decay at the same rate, and one rate for the whole spectrum is therefore
    already a misspecification.  It cannot be derived from the spin system
    alone -- it depends on which lines the acquisition resolves -- so it is an
    argument.

    Ranges, and why each one is wider than the instrument:

    b_mag      0.1 pT to 10 nT.  A shielded ZULF setup should be at the pT
               end; 1 nT transverse already splits a line by 53 mHz, so this
               spans "invisible" to "obviously broken".
    b_theta    isotropic, see SineAngle.
    rate_k     0.005 to 5 s^-1, i.e. HWHM 0.8 mHz to 0.8 Hz, around the ~10
               mHz linewidths these spectra actually show.
    noise      time-domain sigma relative to the largest line amplitude,
               1e-4 to 1, i.e. from far better than any real record to no
               visible signal at all.
    scale      overall amplitude, a decade either side of nominal, since the
               absolute magnetization is not what is being measured.
    prep_angle the DC pulse angle referred to protons.  Full range, because
               which sequence produced the archived spectra is a named risk
               and may have to be inferred rather than assumed.
    f_cut      sensor pole, 50 to 1000 Hz around a nominal 150 Hz.  An
               uncalibrated detector response goes straight into J as bias,
               so this is inferred, not fixed.
    gain       0.3 to 3.
    t_dead     0 to 50 ms of dead time before the first sample.
    """
    comps = [
        ("b_mag", LogUniform(1e-13, 1e-8)),
        ("b_theta", SineAngle()),
    ]
    for k in range(int(n_rates)):
        comps.append((f"rate{k}", LogUniform(5e-3, 5.0)))
    comps += [
        ("noise", LogUniform(1e-4, 1.0)),
        ("scale", LogUniform(0.1, 10.0)),
        ("prep_angle", Uniform(0.0, np.pi)),
        ("f_cut", LogUniform(50.0, 1000.0)),
        ("gain", LogUniform(0.3, 3.0)),
        ("t_dead", Uniform(0.0, 0.05)),
    ]
    if overrides:
        lookup = dict(comps)
        unknown = set(overrides) - set(lookup)
        if unknown:
            raise ValueError(f"unknown nuisance parameters: {sorted(unknown)}")
        comps = [(name, overrides.get(name, dist)) for name, dist in comps]
    return comps


def default_prior(system, predicted=None, width=1.0, nbonds=None, n_rates=1,
                  gauge=True, nuisance=True, overrides=None):
    """The prior layer 6 trains against: couplings first, then nuisances.

    The couplings come first so that the sign gauge acts on a contiguous
    leading block, which is what `SignGaugedPrior` assumes.
    """
    comps = coupling_components(system, predicted=predicted, width=width,
                                nbonds=nbonds)
    n_couplings = len(comps)
    if nuisance:
        comps = comps + nuisance_components(n_rates=n_rates, overrides=overrides)
    if gauge:
        return SignGaugedPrior(comps, n_couplings)
    return Prior(comps)


# ----------------------------------------------- prior-to-posterior movement

class Movement:
    """Per-parameter record of how far the data moved the prior.

    `sd_ratio` is posterior sd over prior sd, `kl` is an estimate of
    KL(posterior || prior) for the one-dimensional marginal, and `flat` marks
    the parameters the data did not constrain.  Both numbers are reported
    because they fail in different places: the ratio is blind to a posterior
    that moved without narrowing, and the KL of a marginal is blind to a
    direction that is only constrained jointly.
    """

    def __init__(self, names, sd_ratio, kl, flat, threshold):
        self.names = list(names)
        self.sd_ratio = np.asarray(sd_ratio, dtype=float)
        self.kl = np.asarray(kl, dtype=float)
        self.flat = np.asarray(flat, dtype=bool)
        self.threshold = float(threshold)

    @property
    def flat_names(self):
        return [n for n, f in zip(self.names, self.flat) if f]

    def table(self):
        width = max([4] + [len(n) for n in self.names])
        lines = [f"{'name'.ljust(width)}  sd_post/sd_prior       KL  flat"]
        for name, r, k, f in zip(self.names, self.sd_ratio, self.kl, self.flat):
            lines.append(
                f"{name.ljust(width)}  {r:16.4f} {k:8.3f}  {'YES' if f else ''}"
            )
        if self.flat_names:
            lines.append(
                f"# {len(self.flat_names)} of {len(self.names)} parameters have "
                f"sd ratio above {self.threshold:g}: the prior was returned "
                f"there, so those numbers are not measurements."
            )
        return "\n".join(lines)

    def __repr__(self):
        return f"Movement({len(self.names)} parameters, {len(self.flat_names)} flat)"


def _kl_gaussian(post, prior):
    """KL(posterior || prior) with both replaced by their moments.

    Cheap, stable, and wrong in exactly one interesting way: it cannot see
    multimodality, which is what an unfixed gauge produces.  Use it as the
    default and the nearest-neighbour estimator as the check.
    """
    mq, sq = post.mean(), post.std(ddof=1)
    mp, sp = prior.mean(), prior.std(ddof=1)
    if sp <= 0 or sq <= 0:
        return np.nan
    return np.log(sp / sq) + (sq ** 2 + (mq - mp) ** 2) / (2.0 * sp ** 2) - 0.5


def _kl_knn(post, prior):
    """Nearest-neighbour estimate of KL(posterior || prior) in one dimension.

    The Perez-Cruz (2008) estimator, which for d = 1 is

        KL = mean(log(s_i / r_i)) + log(m / (n - 1)),

    with r_i the distance from posterior sample i to its nearest other
    posterior sample and s_i the distance to the nearest prior sample.  It
    makes no shape assumption, so it survives the multimodal posteriors this
    problem produces, at the price of being noisy and occasionally slightly
    negative for distributions that are genuinely identical.

    Measured against the closed-form KL of two Gaussians, over 20 seeds: worst
    error 0.032 nats at 2e4 samples, 0.020 at 5e4, 0.016 at 1e5, with a bias
    of order 0.002.  So it resolves a real move but should not be quoted to
    better than a few hundredths.
    """
    q = np.sort(np.asarray(post, dtype=float))
    p = np.sort(np.asarray(prior, dtype=float))
    n, m = q.size, p.size
    if n < 2 or m < 1:
        return np.nan

    gaps = np.diff(q)
    left = np.concatenate(([np.inf], gaps))
    right = np.concatenate((gaps, [np.inf]))
    r = np.minimum(left, right)

    idx = np.searchsorted(p, q)
    lo = np.clip(idx - 1, 0, m - 1)
    hi = np.clip(idx, 0, m - 1)
    s = np.minimum(np.abs(q - p[lo]), np.abs(q - p[hi]))

    scale = max(np.ptp(q), np.ptp(p), 1.0)
    floor = 1e-12 * scale
    r = np.maximum(r, floor)
    s = np.maximum(s, floor)
    return float(np.mean(np.log(s / r)) + np.log(m / (n - 1.0)))


def movement(prior_samples, posterior_samples, names=None, method="gaussian",
             threshold=0.9):
    """How far the data moved each parameter away from its prior.

    This is the diagnostic that makes a flat direction visible instead of
    hidden.  Without it a posterior over six methanol couplings looks like
    six results, four of which are the prior with a different label.

    Parameters
    ----------
    prior_samples, posterior_samples : (n, d) and (m, d) arrays, same d.
    method : 'gaussian' for the moment-matched KL of the marginal, 'knn' for
        the nearest-neighbour estimator, which makes no shape assumption and
        is the one to trust when the posterior is multimodal.
    threshold : sd ratios above this are flagged as flat directions.  0.9 is
        the project convention: a parameter whose posterior is more than 90%
        as wide as its prior has not been measured.
    """
    # a bare 1-D array is n draws of one parameter, not one draw of n of them
    prior_samples = np.asarray(prior_samples, dtype=float).reshape(
        -1, 1 if np.ndim(prior_samples) == 1 else np.shape(prior_samples)[-1])
    posterior_samples = np.asarray(posterior_samples, dtype=float).reshape(
        -1, 1 if np.ndim(posterior_samples) == 1 else np.shape(posterior_samples)[-1])
    if prior_samples.shape[1] != posterior_samples.shape[1]:
        raise ValueError(
            f"prior has {prior_samples.shape[1]} parameters, posterior has "
            f"{posterior_samples.shape[1]}"
        )
    if prior_samples.shape[0] < 2 or posterior_samples.shape[0] < 2:
        raise ValueError("need at least two samples of each to estimate a width")

    d = prior_samples.shape[1]
    if names is None:
        names = [f"theta{k}" for k in range(d)]
    elif hasattr(names, "names"):  # a Prior was passed
        names = list(names.names)
    else:
        names = list(names)
    if len(names) != d:
        raise ValueError(f"got {len(names)} names for {d} parameters")

    estimator = {"gaussian": _kl_gaussian, "knn": _kl_knn}.get(method)
    if estimator is None:
        raise ValueError(f"unknown method {method!r}")

    sd_ratio = np.empty(d)
    kl = np.empty(d)
    for k in range(d):
        sp = prior_samples[:, k].std(ddof=1)
        sq = posterior_samples[:, k].std(ddof=1)
        sd_ratio[k] = np.inf if sp == 0 else sq / sp
        kl[k] = estimator(posterior_samples[:, k], prior_samples[:, k])

    return Movement(names, sd_ratio, kl, sd_ratio > threshold, threshold)
