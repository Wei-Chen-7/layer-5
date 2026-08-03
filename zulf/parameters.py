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

import itertools
import math
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
    "canonical_theta",
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
        "their couplings to all other nuclei, compared lexicographically.  "
        "That vector is a permutation invariant but not a complete one, so "
        "where two nuclei share it we take the relabelling that makes the "
        "coupling matrix lexicographically smallest, which resolves the "
        "remaining cases exactly; the spin index is used only as a final "
        "tie-break between relabellings that give identical matrices.\n"
        "\n"
        "Posteriors are reported on the quotient defined by 1 to 3, with "
        "samples mapped to their representatives before summarizing.  The "
        "prior density is folded over the sign degeneracy of 2, so that prior "
        "and posterior are compared on the same space.  It is not folded over "
        "the labelling degeneracy of 3, which is immaterial whenever the "
        "predicted couplings distinguish the nuclei in question -- for "
        "acetonitrile the two carbons differ by more than a hundred prior "
        "widths -- and is handled by canonicalizing the samples when they do "
        "not."
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


def _quantize(values, tol):
    """Round to a grid so that float noise cannot decide an ordering."""
    values = np.asarray(values, dtype=float)
    return np.round(values / tol) * tol if tol > 0 else values


def _upper_triangle(jmat, perm, n, tol):
    """The relabelled coupling matrix as one comparable tuple."""
    inverse = [0] * n
    for old, new in enumerate(perm):
        inverse[new] = old
    vals = [jmat[inverse[a], inverse[b]] for a in range(n) for b in range(a + 1, n)]
    return tuple(_quantize(vals, tol))


def _is_fully_symmetric(jmat, block, n, tol):
    """True when every permutation of `block` leaves the couplings alone.

    The common case by far: the three protons of a methyl group tie on the
    key because they really are interchangeable, so there is nothing to
    resolve and the k! permutations need never be enumerated.
    """
    block = sorted(block)
    outside = [k for k in range(n) if k not in set(block)]
    first = block[0]
    for i in block[1:]:
        if any(_quantize(jmat[i, k], tol) != _quantize(jmat[first, k], tol)
               for k in outside):
            return False
    intra = {float(_quantize(jmat[i, j], tol))
             for x, i in enumerate(block) for j in block[x + 1:]}
    return len(intra) <= 1


def canonical_order(system, jpairs, tol=1e-9, max_permutations=5040):
    """Fix the relabelling freedom among same-species nuclei.

    Returns ``Ordering(perm, jpairs)``, where ``perm[old] = new`` is the
    relabelling and the second field is ``jpairs`` rewritten under it.

    Each nucleus is keyed on the *sorted* vector of its couplings to every
    other nucleus, and the keys are compared lexicographically.  The vector
    has to be sorted: the raw vector is indexed by the labels of the other
    nuclei, which is the very thing being fixed, so an unsorted key would make
    the answer depend on where you started.  Sorting makes the key invariant
    under relabelling of everything else, and that is what lets the two
    labellings of a molecule agree on one answer.

    Where the keys tie the ordering is not yet determined, and breaking the
    tie on the spin index is not enough.  The sorted key is a permutation
    invariant, not a complete one: two nuclei can share a key and still couple
    differently to the rest of the molecule, and then two labellings of the
    same molecule -- same spectrum, checked -- would keep two different
    representatives, which is the degeneracy this function exists to remove.
    So tied nuclei are resolved by taking the relabelling that makes the
    coupling matrix lexicographically smallest, with the permutation itself as
    the last tie-break.  When the keys are distinct this is exactly the sort,
    and when a tied block is genuinely interchangeable -- a methyl group -- the
    block is recognised and skipped, so the enumeration stays small.
    ``max_permutations`` caps it, and the cap raises rather than silently
    returning a half-fixed gauge.

    Nuclei are only permuted within groups that share both a gyromagnetic
    ratio and an equivalence-class membership.  Nuclei of different species
    are physically distinguishable and must not move.  Two *distinct*
    equivalence classes of the same species -- two chemically inequivalent
    methyl groups, say -- are also left in place; interchanging them whole is
    a further symmetry that none of the benchmark molecules has, and doing it
    by sorting individual nuclei would interleave the classes and invalidate
    the orbit structure.

    ``tol`` quantizes every comparison, in Hz, so that float noise at the
    1e-16 level cannot decide the labelling of a symmetric molecule.
    """
    validate_equivalence(system)
    n = system.n

    jmat = np.zeros((n, n))
    for key, value in jpairs.items():
        i, j = _key(*key)
        if not (0 <= i < n and 0 <= j < n):
            raise ValueError(
                f"coupling {(i, j)} is outside {system.name}, which has "
                f"{n} spins"
            )
        jmat[i, j] = jmat[j, i] = float(value)

    cls = _class_of(system)
    groups = {}
    for i in range(n):
        groups.setdefault((system.gam[i], cls[i]), []).append(i)

    def key_of(i):
        return tuple(_quantize(np.sort([jmat[i, j] for j in range(n) if j != i]),
                               tol))

    keys = {i: key_of(i) for i in range(n)}

    perm = list(range(n))
    tied = []
    for members in groups.values():
        if len(members) < 2:
            continue
        for slot, spin in zip(sorted(members),
                              sorted(members, key=lambda i: (keys[i], i))):
            perm[spin] = slot
        blocks = {}
        for i in members:
            blocks.setdefault(keys[i], []).append(i)
        tied += [sorted(b) for b in blocks.values()
                 if len(b) > 1 and not _is_fully_symmetric(jmat, b, n, tol)]

    if tied:
        cost = math.prod(math.factorial(len(b)) for b in tied)
        if cost > max_permutations:
            raise ValueError(
                f"{system.name} has {cost} labellings left undetermined by the "
                "coupling-vector key, above the cap of "
                f"{max_permutations}.  Raise max_permutations, or declare the "
                "interchangeable nuclei in SpinSystem.equivalent so they stop "
                "being a free relabelling."
            )
        best = None
        for combo in itertools.product(*(itertools.permutations(b) for b in tied)):
            trial = list(perm)
            for block, arrangement in zip(tied, combo):
                for slot, spin in zip(sorted(perm[i] for i in block), arrangement):
                    trial[spin] = slot
            stamp = (_upper_triangle(jmat, trial, n, tol), tuple(trial))
            if best is None or stamp < best[0]:
                best = (stamp, trial)
        perm = best[1]

    perm = tuple(perm)
    return Ordering(perm, relabel(jpairs, perm))


@lru_cache(maxsize=64)
def _orbit_action(n, equivalent, labels):
    """The permutations of the ORBITS induced by relabelling same-species nuclei.

    Empty when the labelling freedom never reaches the sampled parameters at
    all, and that is the common case: formic acid has no two nuclei of the
    same species, and methanol's three protons are declared equivalent, so the
    orbit parameterization has already absorbed every permutation of them.
    Acetonitrile is the molecule where it does reach them -- exchanging the two
    carbons exchanges the C0-H and C1-H orbits -- and it is the reason
    `canonical_theta` cannot simply be `canonical_sign`.

    Each allowed transposition maps whole orbits onto whole orbits.  Two nuclei
    of one declared class generate an element of the equivalence group itself,
    which fixes every orbit; two class-less nuclei have support disjoint from
    that group, so they commute with it and carry orbits to orbits.  Both cases
    are checked here rather than assumed.
    """
    orbs = _orbits(n, equivalent)
    where = {p: k for k, orb in enumerate(orbs) for p in orb}
    cls = {i: g for g, c in enumerate(equivalent) for i in c}

    groups = {}
    for i in range(n):
        groups.setdefault((labels[i], cls.get(i, -1)), []).append(i)

    out = set()
    for members in groups.values():
        for a, b in itertools.combinations(sorted(members), 2):
            swap = list(range(n))
            swap[a], swap[b] = b, a
            induced = []
            for orb in orbs:
                images = {where[_key(swap[i], swap[j])] for (i, j) in orb}
                if len(images) != 1:
                    induced = None
                    break
                induced.append(images.pop())
            if induced is not None and induced != list(range(len(orbs))):
                out.add(tuple(induced))
    return tuple(sorted(out))


def canonical_theta(system, theta, tol=1e-9, max_permutations=5040):
    """Both gauges at once, in the space the sampled parameters actually live in.

    `canonical_sign` works on theta and `canonical_order` works on a
    {(i, j): J} mapping, but layer 6 holds neither a molecule nor a dict, it
    holds a stack of parameter vectors.  This is the composition it needs:
    sign gauge, then labelling gauge through the pair representation and back,
    then sign again in case the relabelling moved which orbit carries the
    largest magnitude.

    Accepts one vector or a stack of them.  When the labelling freedom cannot
    reach the parameters -- formic acid, methanol -- this costs one sign flip
    and no dictionary round-trip at all.  Measured on 2e4 vectors: 0.04 us per
    sample on that path, against 139 us per sample for acetonitrile, which
    does need the round-trip.  That is a few seconds for a whole posterior and
    is meant to be paid once when reporting, not inside the training loop;
    simulation does not need a canonical theta, only the summary does.
    """
    theta = np.asarray(theta, dtype=float)
    single = theta.ndim == 1
    rows = canonical_sign(np.atleast_2d(theta))

    if _orbit_action(system.n, system.equivalent, tuple(system.labels)):
        fixed = np.empty_like(rows)
        for r, row in enumerate(rows):
            pairs = canonical_order(system, to_pairs(system, row), tol=tol,
                                    max_permutations=max_permutations).jpairs
            fixed[r] = from_pairs(system, pairs, tol=tol)
        rows = canonical_sign(fixed)

    return rows[0] if single else rows
