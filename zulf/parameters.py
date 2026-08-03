"""Layer 5: the symmetry-distinct coupling parameterization and the gauge.

Methanol has six pairwise couplings and two measurable numbers.  A posterior
over all six would return the prior in four directions while still looking
like a result, which is the failure this module exists to prevent.  So the
sampled parameters are not pairs, they are orbits.

THE GROUP
---------
The symmetry group is the direct product of the symmetric groups on each
declared equivalence class, taken from ``SpinSystem.equivalent``.  It acts on
the unordered pairs (i, j), and the sampled parameters are the orbits of that
action.  Nothing here knows any chemistry; it reads the declaration and takes
it at its word, except for one check: the members of a declared class must
share a gyromagnetic ratio, because magnetic equivalence implies the same
isotope, and a class that mixes species is a typo rather than a physical
claim.

Orbits are computed as connected components of the graph whose edges join a
pair to its image under a generating transposition.  Adjacent transpositions
inside a class generate the symmetric group on that class, and every group
element is invertible, so the components are exactly the orbits.  This avoids
enumerating the group, which is a product of factorials.

Verified against the three benchmark molecules:

    formic acid    1 pair    1 orbit    [(0,1)]
    methanol XA3   6 pairs   2 orbits   [(0,1),(0,2),(0,3)], [(1,2),(1,3),(2,3)]
    acetonitrile  10 pairs   4 orbits   [(0,1)],
                                        [(0,2),(0,3),(0,4)],
                                        [(1,2),(1,3),(1,4)],
                                        [(2,3),(2,4),(3,4)]

GAUGE CONVENTIONS
-----------------
Two exact degeneracies survive the orbit reduction.  Both are asserted in
tests/test_physics.py, so they are measured properties of the forward model
rather than assumptions about it, and both are fixed here by convention.  A
degeneracy left unfixed does not go away; it turns into a multimodal
posterior whose modes are the same physics reported twice.

**Global sign flip.**  Negating every coupling leaves the spectrum unchanged,
with or without a residual field.  CONVENTION USED IN THIS PACKAGE: *the
orbit carrying the largest-magnitude coupling is taken positive*; ties in
magnitude are resolved in favour of the lowest orbit index, and the all-zero
vector is left alone.  ``canonical_sign`` applies it and ``report_gauge``
states it in the words a paper needs.  Note that this is a statement about
one orbit only.  The signs of the remaining orbits are physical and are
inferred, not fixed.

**Permutation of equal-gamma nuclei.**  The orbit parameterization already
removes permutations inside a declared equivalence class.  What it does not
remove is the residual freedom to relabel same-gamma nuclei that are not
magnetically equivalent, such as the two carbons of acetonitrile.
``canonical_order`` fixes it by sorting those nuclei on their couplings to
everything else, ties broken by index.
"""

from collections import namedtuple
from functools import lru_cache

import numpy as np

__all__ = [
    "orbits",
    "orbit_index",
    "orbit_names",
    "to_pairs",
    "from_pairs",
    "canonical_sign",
    "is_canonical",
    "report_gauge",
    "canonical_order",
    "relabel",
    "validate_equivalence",
    "Ordering",
]


# --------------------------------------------------------------------- orbits

def validate_equivalence(system):
    """Check that the declared equivalence classes could be physical.

    Raises rather than warns, because a bad declaration silently changes the
    number of sampled parameters, and a posterior of the wrong dimension is
    not something you notice by looking at it.
    """
    seen = set()
    for cls in system.equivalent:
        if len(cls) != len(set(cls)):
            raise ValueError(f"equivalence class {cls} repeats an index")
        for i in cls:
            if not (0 <= i < system.n):
                raise ValueError(
                    f"equivalence class {cls} refers to spin {i}, "
                    f"but {system.name} has {system.n} spins"
                )
            if i in seen:
                raise ValueError(
                    f"spin {i} appears in more than one equivalence class; "
                    "the classes must be disjoint for the group to be a "
                    "direct product"
                )
            seen.add(i)
        gam = {system.gam[i] for i in cls}
        if len(gam) > 1:
            labels = [system.labels[i] for i in cls]
            raise ValueError(
                f"equivalence class {cls} mixes {sorted(set(labels))}; "
                "magnetically equivalent nuclei are the same isotope"
            )


def _generators(n, equivalent):
    """Adjacent transpositions inside each class, as index permutations.

    These generate the direct product of the symmetric groups on the classes,
    which is the whole symmetry group, so the orbits of the pairs under the
    generators are the orbits under the group.
    """
    gens = []
    for cls in equivalent:
        members = sorted(cls)
        for a, b in zip(members[:-1], members[1:]):
            perm = list(range(n))
            perm[a], perm[b] = perm[b], perm[a]
            gens.append(tuple(perm))
    return gens


@lru_cache(maxsize=64)
def _orbits(n, equivalent):
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    root = {p: p for p in pairs}

    def find(p):
        while root[p] != p:
            root[p] = root[root[p]]
            p = root[p]
        return p

    def union(p, q):
        rp, rq = find(p), find(q)
        if rp != rq:
            root[max(rp, rq)] = min(rp, rq)

    for perm in _generators(n, equivalent):
        for (i, j) in pairs:
            a, b = perm[i], perm[j]
            union((i, j), (min(a, b), max(a, b)))

    groups = {}
    for p in pairs:
        groups.setdefault(find(p), []).append(p)

    out = [sorted(g) for g in groups.values()]
    out.sort(key=lambda g: g[0])
    return tuple(tuple(g) for g in out)


def orbits(system):
    """The symmetry-distinct couplings, as a list of lists of pairs.

    Orbits are ordered by their smallest pair and each orbit is sorted, so the
    parameter vector has a stable meaning across runs.  Every pair belongs to
    exactly one orbit, so ``sum(len(o) for o in orbits(s))`` is n(n-1)/2.
    """
    validate_equivalence(system)
    return [list(g) for g in _orbits(system.n, system.equivalent)]


def orbit_index(system):
    """Map every unordered pair to the index of its orbit."""
    return {p: k for k, orb in enumerate(orbits(system)) for p in orb}


def orbit_names(system):
    """Readable names for the sampled couplings, one per orbit.

    Named after the representative pair, so a posterior table can be read
    without holding the orbit list in your head.
    """
    return [f"J({o[0][0]},{o[0][1]})" for o in orbits(system)]


def _key(i, j):
    if i == j:
        raise ValueError(f"({i}, {j}) is not a pair of distinct spins")
    return (i, j) if i < j else (j, i)


def to_pairs(system, theta):
    """Orbit values in, a full {(i, j): J} mapping out.

    This is the direction the forward model wants: it takes every pair, and
    the symmetry says what each one must be.
    """
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    orbs = orbits(system)
    if theta.shape != (len(orbs),):
        raise ValueError(
            f"{system.name} has {len(orbs)} orbits, got theta of shape {theta.shape}"
        )
    return {p: float(theta[k]) for k, orb in enumerate(orbs) for p in orb}


def from_pairs(system, jpairs, tol=1e-9):
    """A full {(i, j): J} mapping in, orbit values out.

    Raises if the couplings are not constant within an orbit.  That is not a
    numerical nicety: it means the caller and the equivalence declaration
    disagree about the molecule, and silently averaging them would hide the
    disagreement inside a number that still looks like a coupling.

    A pair absent from ``jpairs`` is read as zero, matching
    ``hamiltonian.coupling_matrix``.  The tolerance is absolute, in Hz, and is
    set six orders of magnitude below the sub-mHz precision this project is
    aiming at, so it only ever absorbs float noise.
    """
    orbs = orbits(system)
    values = {}
    for key, value in jpairs.items():
        i, j = _key(*key)
        if not (0 <= i < system.n and 0 <= j < system.n):
            raise ValueError(
                f"coupling {(i, j)} is outside {system.name}, which has "
                f"{system.n} spins"
            )
        values[(i, j)] = float(value)

    theta = np.zeros(len(orbs))
    for k, orb in enumerate(orbs):
        got = np.array([values.get(p, 0.0) for p in orb])
        if got.max() - got.min() > tol:
            detail = ", ".join(f"{p}={v:.12g}" for p, v in zip(orb, got))
            raise ValueError(
                f"couplings are not constant on orbit {k} of {system.name}: "
                f"{detail}.  Either the couplings are wrong or "
                f"SpinSystem.equivalent is."
            )
        theta[k] = got.mean()
    return theta


# ----------------------------------------------------------------- sign gauge

def canonical_sign(theta):
    """Return theta or -theta, whichever satisfies the sign convention.

    The convention, stated once more because it is the kind of thing that
    gets lost between the code and the paper: the orbit carrying the
    largest-magnitude coupling is positive.  Ties go to the lowest orbit
    index and an all-zero vector is returned unchanged.

    Accepts a single vector or a stack of them, one per row.
    """
    theta = np.asarray(theta, dtype=float)
    if theta.ndim == 1:
        return theta.copy() if is_canonical(theta) else -theta
    if theta.ndim == 2:
        flip = np.where(is_canonical(theta), 1.0, -1.0)[:, None]
        return theta * flip
    raise ValueError("theta must be a vector or a stack of vectors")


def is_canonical(theta):
    """True where the largest-magnitude entry is non-negative.

    Accepts a single coupling vector or a stack of them.  This is the test
    that says whether a point is the representative of its sign-flip pair or
    the other one; the folded prior in priors.py needs it to know where its
    density is allowed to be non-zero.
    """
    theta = np.asarray(theta, dtype=float)
    if theta.shape[-1] == 0:
        return np.ones(theta.shape[:-1], dtype=bool) if theta.ndim > 1 else True
    lead = np.argmax(np.abs(theta), axis=-1)
    if theta.ndim == 1:
        return bool(theta[lead] >= 0.0)
    return np.take_along_axis(theta, lead[:, None], axis=-1)[:, 0] >= 0.0


def report_gauge():
    """The gauge conventions in prose, for the methods section of the paper.

    Kept in the code rather than in a document so that the statement and the
    thing it describes cannot drift apart.
    """
    return (
        "Gauge conventions for the coupling parameterization.\n"
        "\n"
        "1. Symmetry-distinct couplings.  The sampled parameters are the "
        "orbits of the direct product of the symmetric groups on the declared "
        "magnetically equivalent nuclei, acting on the unordered pairs of "
        "spins.  Couplings related by that group are one parameter, not "
        "several, because the spectrum cannot distinguish them.\n"
        "\n"
        "2. Global sign.  Negating every scalar coupling leaves the ZULF "
        "spectrum unchanged, with or without a residual field.  We fix the "
        "resulting two-fold degeneracy by requiring the orbit that carries "
        "the largest-magnitude coupling to be positive; ties in magnitude are "
        "resolved in favour of the lowest orbit index.  The signs of the "
        "remaining orbits are physical and are inferred.\n"
        "\n"
        "3. Nuclear labelling.  Nuclei of the same species that are not "
        "magnetically equivalent may be relabelled without changing the "
        "spectrum.  We fix this by ordering them on the sorted vector of "
        "their couplings to all other nuclei, compared lexicographically, "
        "with ties broken by spin index.\n"
        "\n"
        "Posteriors are reported on the quotient defined by 1 to 3.  Prior "
        "densities are folded onto the same quotient, so prior and posterior "
        "are compared on the same space."
    )


# ---------------------------------------------------------- permutation gauge

Ordering = namedtuple("Ordering", ["perm", "jpairs"])


def _class_of(system):
    """Which declared equivalence class each spin belongs to, -1 for none."""
    out = [-1] * system.n
    for k, cls in enumerate(system.equivalent):
        for i in cls:
            out[i] = k
    return out


def relabel(jpairs, perm):
    """Rewrite a coupling mapping under a relabelling ``perm[old] = new``."""
    out = {}
    for key, value in jpairs.items():
        i, j = _key(*key)
        out[_key(perm[i], perm[j])] = value
    return out


def canonical_order(system, jpairs, tol=1e-9):
    """Fix the relabelling freedom among same-species nuclei.

    Returns ``Ordering(perm, jpairs)``, where ``perm[old] = new`` is the
    relabelling and the second field is ``jpairs`` rewritten under it.

    Each nucleus is keyed on the *sorted* vector of its couplings to every
    other nucleus, and the keys are compared lexicographically with ties
    broken by index.  The vector has to be sorted: the raw vector is indexed
    by the labels of the other nuclei, which is the very thing being fixed,
    so an unsorted key would make the answer depend on where you started.
    Sorting makes the key invariant under relabelling of everything else, and
    that is what lets the two labellings of a molecule agree on one answer.

    Nuclei are only permuted within groups that share both a gyromagnetic
    ratio and an equivalence-class membership.  Nuclei of different species
    are physically distinguishable and must not move.  Two *distinct*
    equivalence classes of the same species -- two chemically inequivalent
    methyl groups, say -- are also left in place; interchanging them whole is
    a further symmetry that none of the benchmark molecules has, and doing it
    by sorting individual nuclei would interleave the classes and invalidate
    the orbit structure.

    ``tol`` quantizes the key so that couplings agreeing to within it are
    treated as tied and fall back to the index.  Without it, float noise at
    the 1e-16 level would decide the labelling of a genuinely symmetric
    molecule.
    """
    validate_equivalence(system)

    jmat = np.zeros((system.n, system.n))
    for key, value in jpairs.items():
        i, j = _key(*key)
        if not (0 <= i < system.n and 0 <= j < system.n):
            raise ValueError(
                f"coupling {(i, j)} is outside {system.name}, which has "
                f"{system.n} spins"
            )
        jmat[i, j] = jmat[j, i] = float(value)

    cls = _class_of(system)
    groups = {}
    for i in range(system.n):
        groups.setdefault((system.gam[i], cls[i]), []).append(i)

    def key_of(i):
        others = np.array([jmat[i, j] for j in range(system.n) if j != i])
        others = np.sort(others)
        if tol > 0:
            others = np.round(others / tol) * tol
        return tuple(others)

    perm = list(range(system.n))
    for members in groups.values():
        if len(members) < 2:
            continue
        slots = sorted(members)
        ordered = sorted(members, key=lambda i: (key_of(i), i))
        for slot, spin in zip(slots, ordered):
            perm[spin] = slot

    perm = tuple(perm)
    return Ordering(perm, relabel(jpairs, perm))
