"""Layer 5 as a contract: the parameterization, the gauge, the priors, and the
diagnostic that says whether a posterior is a measurement or a restatement of
the prior.

The claim this whole layer exists to support is at the bottom, in
`test_methyl_orbit_is_a_flat_direction_end_to_end`.  tests/test_physics.py
already proves that the methyl HH coupling moves nothing in the spectrum.
Here that fact is carried through the parameterization and shown to come out
of the movement diagnostic as a flat direction, which is what stops it being
reported as a measured number.
"""

import numpy as np
import pytest

from zulf import ACETONITRILE, FORMIC_ACID, METHANOL, SpinSystem, forward
from zulf.parameters import (
    Ordering,
    canonical_order,
    canonical_sign,
    from_pairs,
    is_canonical,
    orbit_index,
    orbit_names,
    orbits,
    relabel,
    report_gauge,
    to_pairs,
    validate_equivalence,
)
from zulf.priors import (
    FALLBACK_COUPLING,
    LogUniform,
    Normal,
    Prior,
    SignGaugedPrior,
    SineAngle,
    Uniform,
    coupling_components,
    default_prior,
    movement,
    nuisance_components,
)

METHYL = {(0, 1): 140.6, (0, 2): 140.6, (0, 3): 140.6,
          (1, 2): -12.4, (1, 3): -12.4, (2, 3): -12.4}

NITRILE = {(0, 1): 56.6, (0, 2): -10.0, (0, 3): -10.0, (0, 4): -10.0,
           (1, 2): 136.0, (1, 3): 136.0, (1, 4): 136.0,
           (2, 3): -16.9, (2, 4): -16.9, (3, 4): -16.9}


def lines(system, jpairs, **kw):
    f, a, _ = forward(system, jpairs, **kw)
    keep = np.abs(np.real(a)) > 1e-6 * np.max(np.abs(np.real(a)))
    return f[keep], np.real(a)[keep]


# ------------------------------------------------------------------- orbits

def test_orbits_formic_acid():
    assert orbits(FORMIC_ACID) == [[(0, 1)]]


def test_orbits_methanol():
    assert orbits(METHANOL) == [[(0, 1), (0, 2), (0, 3)],
                                [(1, 2), (1, 3), (2, 3)]]


def test_orbits_acetonitrile():
    assert orbits(ACETONITRILE) == [[(0, 1)],
                                    [(0, 2), (0, 3), (0, 4)],
                                    [(1, 2), (1, 3), (1, 4)],
                                    [(2, 3), (2, 4), (3, 4)]]


@pytest.mark.parametrize("system", [FORMIC_ACID, METHANOL, ACETONITRILE])
def test_orbits_partition_every_pair_exactly_once(system):
    orbs = orbits(system)
    flat = [p for orb in orbs for p in orb]
    assert len(flat) == len(set(flat)) == system.n * (system.n - 1) // 2


def test_orbit_count_is_the_number_of_sampled_parameters():
    """Six pairwise couplings in methanol, two numbers the spectrum can hold."""
    assert len(orbits(FORMIC_ACID)) == 1
    assert len(orbits(METHANOL)) == 2
    assert len(orbits(ACETONITRILE)) == 4


def test_orbit_names_use_the_representative_pair():
    assert orbit_names(METHANOL) == ["J(0,1)", "J(1,2)"]
    assert orbit_names(ACETONITRILE) == ["J(0,1)", "J(0,2)", "J(1,2)", "J(2,3)"]


def test_orbit_index_maps_every_pair():
    where = orbit_index(METHANOL)
    assert where[(0, 2)] == 0 and where[(2, 3)] == 1
    assert len(where) == 6


# ------------------------------------------------------------ the two maps

def test_from_pairs_reads_the_two_methanol_numbers():
    theta = from_pairs(METHANOL, METHYL)
    assert np.allclose(theta, [140.6, -12.4])


def test_from_pairs_reads_the_four_acetonitrile_numbers():
    theta = from_pairs(ACETONITRILE, NITRILE)
    assert np.allclose(theta, [56.6, -10.0, 136.0, -16.9])


@pytest.mark.parametrize(
    "system, theta",
    [(FORMIC_ACID, [221.0]),
     (METHANOL, [140.6, -12.4]),
     (ACETONITRILE, [56.6, -10.0, 136.0, -16.9])],
)
def test_to_pairs_and_from_pairs_round_trip(system, theta):
    jpairs = to_pairs(system, theta)
    assert len(jpairs) == system.n * (system.n - 1) // 2
    assert np.allclose(from_pairs(system, jpairs), theta)


def test_to_pairs_rejects_the_wrong_number_of_orbits():
    with pytest.raises(ValueError, match="orbits"):
        to_pairs(METHANOL, [140.6])


def test_from_pairs_raises_when_an_orbit_is_not_constant():
    """A caller who disagrees with SpinSystem.equivalent has to hear about it,
    because averaging the disagreement away produces a plausible number."""
    bad = dict(METHYL)
    bad[(0, 2)] = 139.0
    with pytest.raises(ValueError, match="not constant"):
        from_pairs(METHANOL, bad)


def test_from_pairs_tolerates_float_noise_but_not_physics():
    ok = dict(METHYL)
    ok[(0, 2)] = 140.6 + 1e-12
    assert np.allclose(from_pairs(METHANOL, ok), [140.6, -12.4])
    bad = dict(METHYL)
    bad[(0, 2)] = 140.6 + 1e-6  # a microhertz is a thousand times the target
    with pytest.raises(ValueError, match="not constant"):
        from_pairs(METHANOL, bad)


def test_from_pairs_reads_a_missing_pair_as_zero():
    partial = {(1, 2): 0.0, (1, 3): 0.0, (2, 3): 0.0,
               (0, 1): 140.6, (0, 2): 140.6, (0, 3): 140.6}
    assert np.allclose(from_pairs(METHANOL, partial), [140.6, 0.0])
    # but a pair dropped from an orbit whose other members are non-zero is a
    # disagreement, not a shorthand
    with pytest.raises(ValueError, match="not constant"):
        from_pairs(METHANOL, {k: v for k, v in METHYL.items() if k != (0, 3)})


def test_from_pairs_accepts_either_index_order():
    flipped = {(j, i): v for (i, j), v in METHYL.items()}
    assert np.allclose(from_pairs(METHANOL, flipped), [140.6, -12.4])


def test_from_pairs_rejects_a_pair_outside_the_system():
    with pytest.raises(ValueError, match="outside"):
        from_pairs(FORMIC_ACID, {(0, 1): 221.0, (0, 5): 1.0})


def test_orbit_parameterization_reproduces_the_forward_model():
    f_ref, a_ref = lines(METHANOL, METHYL)
    f, a = lines(METHANOL, to_pairs(METHANOL, from_pairs(METHANOL, METHYL)))
    assert np.allclose(f, f_ref, atol=1e-9)
    assert np.allclose(a, a_ref, rtol=1e-12)


# ---------------------------------------------------------- equivalence check

def test_validate_rejects_a_class_that_mixes_species():
    bad = SpinSystem("nonsense", ["13C", "1H"], equivalent=[(0, 1)])
    with pytest.raises(ValueError, match="magnetically equivalent"):
        orbits(bad)


def test_validate_rejects_overlapping_classes():
    bad = SpinSystem("nonsense", ["1H"] * 4, equivalent=[(0, 1), (1, 2)])
    with pytest.raises(ValueError, match="disjoint"):
        validate_equivalence(bad)


def test_validate_rejects_an_index_off_the_end():
    bad = SpinSystem("nonsense", ["1H"] * 2, equivalent=[(0, 1, 2)])
    with pytest.raises(ValueError, match="spins"):
        validate_equivalence(bad)


# ----------------------------------------------------------------- sign gauge

def test_canonical_sign_makes_the_largest_orbit_positive():
    assert np.allclose(canonical_sign([140.6, -12.4]), [140.6, -12.4])
    assert np.allclose(canonical_sign([-140.6, 12.4]), [140.6, -12.4])


def test_canonical_sign_collapses_the_two_branches_onto_one():
    theta = np.array([56.6, -10.0, 136.0, -16.9])
    assert np.allclose(canonical_sign(theta), canonical_sign(-theta))


def test_canonical_sign_is_idempotent():
    rng = np.random.default_rng(3)
    theta = rng.normal(0, 50, size=(200, 4))
    once = canonical_sign(theta)
    assert np.allclose(canonical_sign(once), once)
    assert np.all(is_canonical(once))


def test_canonical_sign_breaks_a_magnitude_tie_on_the_lowest_orbit():
    assert np.allclose(canonical_sign([-5.0, 5.0]), [5.0, -5.0])


def test_canonical_sign_leaves_the_zero_vector_alone():
    assert np.allclose(canonical_sign([0.0, 0.0]), [0.0, 0.0])


def test_canonical_sign_agrees_with_the_forward_model_degeneracy():
    """The convention is only allowed because the spectrum really is blind to
    the flip.  tests/test_physics.py asserts that; this checks that the
    representative we keep gives the same spectrum as the one we discard."""
    theta = from_pairs(METHANOL, METHYL)
    f1, a1 = lines(METHANOL, to_pairs(METHANOL, canonical_sign(theta)))
    f2, a2 = lines(METHANOL, to_pairs(METHANOL, canonical_sign(-theta)))
    assert np.allclose(f1, f2, atol=1e-9)
    assert np.allclose(a1, a2, rtol=1e-12)


def test_report_gauge_states_both_conventions():
    text = report_gauge()
    assert "largest-magnitude" in text
    assert "positive" in text
    assert "relabel" in text.lower()
    assert "quotient" in text


# ---------------------------------------------------------- permutation gauge

def test_canonical_order_is_the_identity_on_a_symmetric_molecule():
    order = canonical_order(METHANOL, METHYL)
    assert isinstance(order, Ordering)
    assert order.perm == (0, 1, 2, 3)
    assert order.jpairs == METHYL


def test_canonical_order_undoes_a_relabelling_of_two_carbons():
    """Acetonitrile's two carbons have the same gamma and are not
    magnetically equivalent, so which one is called 0 is a gauge choice."""
    swap = {0: 1, 1: 0, 2: 2, 3: 3, 4: 4}
    swapped = relabel(NITRILE, swap)
    assert swapped != NITRILE

    a = canonical_order(ACETONITRILE, NITRILE)
    b = canonical_order(ACETONITRILE, swapped)
    assert a.jpairs == b.jpairs
    assert b.perm != (0, 1, 2, 3, 4)


def test_canonical_order_undoes_the_permutation_from_test_physics():
    """The same relabelling that tests/test_physics.py shows the spectrum
    cannot see, mapped back onto one representative."""
    base = {(0, 1): 140.0, (0, 2): 120.0, (0, 3): 100.0,
            (1, 2): -12.0, (1, 3): -9.0, (2, 3): -5.0}
    swapped = relabel(base, {0: 0, 1: 2, 2: 1, 3: 3})
    assert swapped != base

    f1, a1 = lines(METHANOL, base)
    f2, a2 = lines(METHANOL, swapped)
    assert np.allclose(np.sort(f1), np.sort(f2), atol=1e-9)  # the degeneracy

    assert canonical_order(METHANOL, base).jpairs == \
        canonical_order(METHANOL, swapped).jpairs


def test_canonical_order_never_mixes_species():
    swapped = relabel(NITRILE, {0: 1, 1: 0, 2: 2, 3: 3, 4: 4})
    perm = canonical_order(ACETONITRILE, swapped).perm
    assert {perm[0], perm[1]} == {0, 1}       # the carbons stay carbons
    assert {perm[2], perm[3], perm[4]} == {2, 3, 4}


def test_canonical_order_preserves_the_orbit_structure():
    swapped = relabel(NITRILE, {0: 1, 1: 0, 2: 2, 3: 3, 4: 4})
    out = canonical_order(ACETONITRILE, swapped).jpairs
    assert np.allclose(from_pairs(ACETONITRILE, out),
                       [56.6, -10.0, 136.0, -16.9])


def test_canonical_order_does_not_change_the_spectrum():
    swapped = relabel(NITRILE, {0: 1, 1: 0, 2: 2, 3: 3, 4: 4})
    f1, _ = lines(ACETONITRILE, swapped)
    f2, _ = lines(ACETONITRILE, canonical_order(ACETONITRILE, swapped).jpairs)
    assert np.allclose(np.sort(f1), np.sort(f2), atol=1e-9)


def test_canonical_order_ties_break_on_the_index():
    """Three equivalent protons have identical coupling vectors, so nothing
    should move and the answer must not depend on dictionary order."""
    assert canonical_order(ACETONITRILE, NITRILE).perm == (0, 1, 2, 3, 4)


# -------------------------------------------------------------- prior pieces

@pytest.mark.parametrize(
    "dist, lo, hi",
    [(Normal(140.0, 1.0), 130.0, 150.0),   # +/- 10 sigma, tails are 1e-23
     (Uniform(0.0, np.pi), -0.5, np.pi + 0.5),
     (LogUniform(1e-3, 1e2), 1e-3, 1e2),
     (SineAngle(), 0.0, np.pi)],
)
def test_components_are_normalized(dist, lo, hi):
    """Adaptive quadrature, not a linear grid: LogUniform is 1/x over five
    decades and a uniform grid mis-integrates it by parts in a thousand while
    the density is exactly right."""
    from scipy.integrate import quad

    mass, err = quad(lambda t: float(np.exp(dist.log_prob(t))), lo, hi, limit=400)
    assert abs(mass - 1.0) < 1e-6


@pytest.mark.parametrize(
    "dist", [Normal(0.0, 3.0), Uniform(-1.0, 2.0), LogUniform(0.1, 10.0), SineAngle()]
)
def test_components_sample_inside_their_support(dist):
    rng = np.random.default_rng(1)
    x = dist.sample(5000, rng)
    assert np.all(np.isfinite(dist.log_prob(x)))
    assert x.shape == (5000,)


def test_sine_angle_prefers_the_transverse_direction():
    """A field on the sensor axis is nearly invisible; one across it splits the
    line by 53 mHz.  A prior that avoided the informative geometry would train
    the network on the easy case."""
    rng = np.random.default_rng(2)
    theta = SineAngle().sample(100000, rng)
    assert abs(np.mean(theta) - np.pi / 2) < 0.02
    assert np.mean(np.abs(theta - np.pi / 2) < 0.5) > np.mean(theta < 0.5) * 5


# ------------------------------------------------------------ coupling priors

def test_coupling_prior_centres_on_the_prediction():
    comps = dict(coupling_components(METHANOL, predicted={(0, 1): 140.3},
                                     width=1.0, nbonds={(1, 2): 3}))
    assert comps["J(0,1)"].mu == 140.3
    assert comps["J(0,1)"].sigma == 1.0


def test_coupling_prior_width_is_an_argument_for_the_sensitivity_check():
    narrow = dict(coupling_components(METHANOL, predicted={(0, 1): 140.3}, width=1.0))
    wide = dict(coupling_components(METHANOL, predicted={(0, 1): 140.3}, width=10.0))
    assert wide["J(0,1)"].sigma == 10.0 * narrow["J(0,1)"].sigma
    assert wide["J(0,1)"].mu == narrow["J(0,1)"].mu


def test_coupling_prior_accepts_a_per_orbit_width():
    comps = dict(coupling_components(
        ACETONITRILE,
        predicted={(0, 1): 56.6, (1, 2): 136.0},
        width={(0, 1): 2.0, (1, 2): 0.5},
    ))
    assert comps["J(0,1)"].sigma == 2.0
    assert comps["J(1,2)"].sigma == 0.5


def test_coupling_prior_falls_back_by_bond_type():
    comps = dict(coupling_components(
        ACETONITRILE, nbonds={(0, 1): 1, (1, 2): 1, (0, 2): 2, (2, 3): 2}))
    assert comps["J(0,1)"].sigma == FALLBACK_COUPLING["one-bond"][1]
    assert comps["J(0,2)"].sigma == FALLBACK_COUPLING["two-bond"][1]


def test_fallback_widths_are_ordered_by_bond_count():
    one = FALLBACK_COUPLING["one-bond"][1]
    two = FALLBACK_COUPLING["two-bond"][1]
    long = FALLBACK_COUPLING["long-range"][1]
    assert one > two > long
    assert one >= 60.0          # must cover |1J_CH| out to 250 Hz
    assert FALLBACK_COUPLING["unknown"][1] >= one


def test_a_prediction_narrows_the_search_by_two_orders_of_magnitude():
    """The reason predictions are used as priors and not as answers."""
    predicted = dict(coupling_components(METHANOL, predicted={(0, 1): 140.3},
                                         width=1.0))["J(0,1)"]
    blind = dict(coupling_components(METHANOL))["J(0,1)"]
    assert blind.sigma / predicted.sigma > 50.0


def test_coupling_prior_rejects_two_values_for_one_orbit():
    with pytest.raises(ValueError, match="two different values"):
        coupling_components(METHANOL, predicted={(0, 1): 140.0, (0, 2): 138.0},
                            width=1.0)


def test_coupling_prior_rejects_a_pair_outside_the_system():
    with pytest.raises(ValueError, match="not a pair"):
        coupling_components(FORMIC_ACID, predicted={(0, 4): 221.0}, width=1.0)


def test_coupling_prior_rejects_a_prediction_with_no_width():
    """A predicted coupling with no stated uncertainty is not a prior, and
    silently supplying one would be inventing the very number the sensitivity
    check is supposed to vary."""
    with pytest.raises(ValueError, match="no width"):
        coupling_components(ACETONITRILE, predicted={(0, 1): 56.6, (1, 2): 136.0},
                            width={(0, 1): 2.0})


# ------------------------------------------------------------- nuisance block

def test_nuisance_block_has_one_rate_per_multiplet():
    names = [n for n, _ in nuisance_components(n_rates=3)]
    assert names.count("rate0") == names.count("rate1") == names.count("rate2") == 1
    assert "rate3" not in names


def test_nuisance_block_covers_everything_the_instrument_can_do():
    names = {n for n, _ in nuisance_components()}
    assert {"b_mag", "b_theta", "rate0", "noise", "scale", "prep_angle",
            "f_cut", "gain", "t_dead"} <= names


def test_nuisance_priors_are_deliberately_wide():
    """A network trained on a tidy world does not transfer."""
    comps = dict(nuisance_components())
    assert comps["b_mag"].hi / comps["b_mag"].lo >= 1e4
    assert comps["rate0"].hi / comps["rate0"].lo >= 100.0
    assert comps["f_cut"].lo < 150.0 < comps["f_cut"].hi   # nominal sensor pole
    assert comps["prep_angle"].hi >= np.pi


def test_nuisance_overrides_replace_only_the_named_component():
    comps = dict(nuisance_components(overrides={"gain": Uniform(0.9, 1.1)}))
    assert isinstance(comps["gain"], Uniform)
    assert isinstance(comps["scale"], LogUniform)
    with pytest.raises(ValueError, match="unknown nuisance"):
        nuisance_components(overrides={"nonsense": Uniform(0, 1)})


# ----------------------------------------------------------------- the Prior

def test_prior_names_couplings_first_then_nuisances():
    prior = default_prior(METHANOL, predicted={(0, 1): 140.3}, width=1.0, n_rates=2)
    assert prior.names[:2] == ["J(0,1)", "J(1,2)"]
    assert prior.names[2:4] == ["b_mag", "b_theta"]
    assert prior.ndim == len(prior.names)


def test_prior_sample_and_log_prob_shapes():
    prior = default_prior(METHANOL, predicted={(0, 1): 140.3}, width=1.0)
    rng = np.random.default_rng(0)
    theta = prior.sample(64, rng)
    assert theta.shape == (64, prior.ndim)
    assert prior.log_prob(theta).shape == (64,)
    assert np.isscalar(prior.log_prob(theta[0]))
    assert np.all(np.isfinite(prior.log_prob(theta)))


def test_prior_log_prob_is_the_sum_of_independent_components():
    prior = default_prior(METHANOL, predicted={(0, 1): 140.3}, width=1.0,
                          nuisance=False, gauge=False)
    theta = np.array([140.5, -12.0])
    expected = sum(c.log_prob(v) for c, v in zip(prior.components, theta))
    assert abs(prior.log_prob(theta) - expected) < 1e-12


def test_prior_as_dict_round_trips_the_names():
    prior = default_prior(FORMIC_ACID, predicted={(0, 1): 221.0}, width=1.0)
    d = prior.as_dict(prior.sample(1, np.random.default_rng(0))[0])
    assert set(d) == set(prior.names)


def test_prior_rejects_duplicate_names():
    with pytest.raises(ValueError, match="unique"):
        Prior([("J", Normal(0, 1)), ("J", Normal(0, 1))])


def test_prior_repr_says_whether_the_gauge_is_on():
    """Whether the fold is applied changes what log_prob means, so it has to
    be visible without reading the constructor call."""
    assert repr(default_prior(METHANOL, gauge=True)).startswith("SignGaugedPrior")
    assert repr(default_prior(METHANOL, gauge=False)).startswith("Prior(")


# --------------------------------------------------- the gauge inside the prior

def test_gauged_prior_only_ever_samples_canonical_couplings():
    prior = default_prior(METHANOL, nuisance=False)  # wide, straddles zero
    theta = prior.sample(5000, np.random.default_rng(4))
    assert np.all(is_canonical(theta))


def test_gauged_prior_density_is_the_sum_over_the_two_preimages():
    base = default_prior(METHANOL, nuisance=False, gauge=False)
    gauged = SignGaugedPrior(list(zip(base.names, base.components)), 2)
    theta = np.array([80.0, -10.0])
    expected = np.logaddexp(base.log_prob(theta), base.log_prob(-theta))
    assert abs(gauged.log_prob(theta) - expected) < 1e-12


def test_gauged_prior_gives_no_density_outside_the_canonical_region():
    prior = default_prior(METHANOL, nuisance=False)
    assert prior.log_prob(np.array([-80.0, 10.0])) == -np.inf
    assert np.isfinite(prior.log_prob(np.array([80.0, -10.0])))


def test_gauged_prior_is_still_normalized():
    """One coupling, so the fold can be integrated exactly.  A prior that is
    not folded would integrate to 1/2 on the quotient and quietly halve every
    density ratio computed against it."""
    prior = default_prior(FORMIC_ACID, nuisance=False)
    x = np.linspace(0.0, 1000.0, 400001)
    p = np.exp(prior.log_prob(x[:, None]))
    assert abs(np.trapezoid(p, x) - 1.0) < 1e-4


def test_the_fold_is_invisible_when_the_prediction_is_sharp():
    """With a 1 Hz prior on a 140 Hz coupling the other branch has no mass, so
    folding must cost nothing.  It matters only for the wide fallbacks."""
    base = default_prior(METHANOL, predicted={(0, 1): 140.3}, width=1.0,
                         nuisance=False, gauge=False)
    gauged = default_prior(METHANOL, predicted={(0, 1): 140.3}, width=1.0,
                           nuisance=False, gauge=True)
    theta = np.array([140.4, -12.0])
    assert abs(gauged.log_prob(theta) - base.log_prob(theta)) < 1e-12


def test_gauged_samples_are_usable_by_the_forward_model():
    prior = default_prior(METHANOL, predicted={(0, 1): 140.3}, width=1.0,
                          nuisance=False)
    theta = prior.sample(5, np.random.default_rng(5))
    for row in theta:
        f, a = lines(METHANOL, to_pairs(METHANOL, row))
        assert f.size == 2 and np.all(f > 0)


# ------------------------------------------------- prior-to-posterior movement

def test_movement_flags_the_untouched_parameter_and_not_the_measured_one():
    rng = np.random.default_rng(7)
    n = 20000
    prior_samples = np.column_stack([rng.normal(0, 1, n), rng.normal(0, 1, n)])
    post = np.column_stack([rng.normal(0.5, 0.05, n), rng.normal(0, 1, n)])

    m = movement(prior_samples, post, names=["measured", "flat"])
    assert m.sd_ratio[0] < 0.1
    assert 0.9 < m.sd_ratio[1] < 1.1
    assert m.flat_names == ["flat"]
    assert m.kl[0] > 2.0
    assert abs(m.kl[1]) < 0.05


def test_movement_threshold_is_the_stated_nine_tenths():
    rng = np.random.default_rng(8)
    n = 40000
    prior_samples = rng.normal(0, 1, (n, 2))
    post = np.column_stack([rng.normal(0, 0.85, n), rng.normal(0, 0.95, n)])
    m = movement(prior_samples, post)
    assert not m.flat[0] and m.flat[1]
    assert m.threshold == 0.9


def test_movement_gaussian_kl_matches_the_closed_form():
    rng = np.random.default_rng(9)
    n = 400000
    prior_samples = rng.normal(0.0, 1.0, (n, 1))
    post = rng.normal(0.5, 0.2, (n, 1))
    exact = np.log(1.0 / 0.2) + (0.2 ** 2 + 0.5 ** 2) / 2.0 - 0.5
    assert abs(movement(prior_samples, post).kl[0] - exact) < 0.01


def test_movement_knn_kl_agrees_with_the_gaussian_one_on_gaussians():
    """The nearest-neighbour estimator makes no shape assumption, which is why
    it is available; on Gaussians the two must agree or one of them is wrong.

    Measured over 20 seeds at this sample size, the estimator's worst error
    against the closed form is 0.020 nats and its bias is -0.001, so the 0.05
    tolerance is the estimator's noise and not a fudge.
    """
    rng = np.random.default_rng(10)
    n = 50000
    prior_samples = rng.normal(0.0, 1.0, (n, 1))
    post = rng.normal(0.5, 0.2, (n, 1))
    exact = np.log(1.0 / 0.2) + (0.2 ** 2 + 0.5 ** 2) / 2.0 - 0.5
    knn = movement(prior_samples, post, method="knn").kl[0]
    assert abs(knn - exact) < 0.05


def test_movement_knn_sees_a_move_that_the_moments_miss():
    """A posterior that split into two modes of the same total width has the
    same standard deviation as its prior.  The moment-matched KL calls that
    unmoved; the nearest-neighbour estimate does not.  This is the failure
    mode an unfixed gauge produces, which is why both numbers are reported."""
    rng = np.random.default_rng(11)
    n = 20000
    prior_samples = rng.normal(0.0, 1.0, (n, 1))
    sign = rng.choice([-1.0, 1.0], size=n)
    post = (sign * (1.0 + rng.normal(0.0, 0.05, n))).reshape(n, 1)

    assert abs(movement(prior_samples, post).kl[0]) < 0.2      # moments blind
    assert movement(prior_samples, post, method="knn").kl[0] > 1.0


def test_movement_takes_names_from_a_prior_object():
    prior = default_prior(METHANOL, predicted={(0, 1): 140.3}, width=1.0)
    rng = np.random.default_rng(12)
    m = movement(prior.sample(2000, rng), prior.sample(2000, rng), names=prior)
    assert m.names == prior.names
    assert "J(0,1)" in m.table()


def test_movement_reads_a_bare_1d_array_as_one_parameter():
    rng = np.random.default_rng(14)
    m = movement(rng.normal(0, 1, 5000), rng.normal(0, 0.1, 5000))
    assert m.sd_ratio.shape == (1,)
    assert m.sd_ratio[0] < 0.15 and not m.flat[0]


def test_movement_rejects_mismatched_dimensions():
    with pytest.raises(ValueError, match="parameters"):
        movement(np.zeros((10, 3)), np.zeros((10, 2)))


def test_movement_rejects_an_unknown_estimator():
    with pytest.raises(ValueError, match="unknown method"):
        movement(np.zeros((10, 1)), np.ones((10, 1)), method="magic")


# ------------------------------------------------------------ acceptance test

def test_the_methyl_orbit_moves_nothing_in_the_spectrum():
    """The physics half of the acceptance test, restated through the
    parameterization: sweep the methyl orbit over its whole prior range and
    the line list does not move."""
    ref_f, ref_a = lines(METHANOL, to_pairs(METHANOL, [140.6, -12.4]))
    for jhh in (-100.0, -12.4, 0.0, 5.0, 40.0):
        f, a = lines(METHANOL, to_pairs(METHANOL, [140.6, jhh]))
        assert np.allclose(f, ref_f, atol=1e-9)
        assert np.allclose(a / a[0], ref_a / ref_a[0], atol=1e-9)


def test_the_ch_orbit_moves_the_spectrum():
    """The other half: the parameter that is measurable must be measurable, or
    the flat direction below would be trivially true of everything."""
    f0, _ = lines(METHANOL, to_pairs(METHANOL, [140.6, -12.4]))
    f1, _ = lines(METHANOL, to_pairs(METHANOL, [141.6, -12.4]))
    assert np.abs(f1 - f0).max() > 0.9


def test_methyl_orbit_is_a_flat_direction_end_to_end():
    """The point of the whole layer.

    A posterior conditioned on a methanol spectrum cannot have learned
    anything about the methyl HH orbit, because the spectrum does not depend
    on it.  Layer 6 does not exist yet, so the posterior here is synthetic:
    the CH orbit is drawn tight around a measured value and everything else is
    redrawn from the prior, which is exactly the answer the physics demands.
    The diagnostic has to report that as one measured parameter and the rest
    returned unchanged, rather than as a fifteen-parameter result.
    """
    prior = default_prior(METHANOL, predicted={(0, 1): 140.3}, width=1.0,
                          nbonds={(1, 2): 3}, n_rates=2)
    rng = np.random.default_rng(13)
    n = 20000

    prior_samples = prior.sample(n, rng)
    posterior = prior.sample(n, rng)
    posterior[:, prior.index("J(0,1)")] = rng.normal(140.612, 0.002, n)

    m = movement(prior_samples, posterior, names=prior)

    ch, hh = prior.index("J(0,1)"), prior.index("J(1,2)")
    assert m.sd_ratio[ch] < 0.01          # measured to 2 mHz against a 1 Hz prior
    assert m.kl[ch] > 5.0
    assert not m.flat[ch]

    assert 0.9 < m.sd_ratio[hh] < 1.1     # the prior, handed back
    assert abs(m.kl[hh]) < 0.05
    assert m.flat[hh]

    assert "J(1,2)" in m.flat_names and "J(0,1)" not in m.flat_names
    assert "not measurements" in m.table()
