# CLAUDE.md

Simulation-based inference of J-couplings from zero- to ultralow-field (ZULF)
NMR spectra. Physics research code that ends in a paper, so correctness beats
speed and beats convenience.

Run `python -m pytest tests/ -q` before and after every change. 104 tests pass
today: 20 in `tests/test_physics.py` and 84 in `tests/test_parameters.py`.

---

## Conventions that break silently

These are the ones where wrong code still runs and still produces a plausible
number. Do not change any of them without saying so explicitly.

**Units.** J in Hz. Gyromagnetic ratio in rad s^-1 T^-1. B in T. The
Hamiltonian is in rad s^-1 with hbar = 1. Every frequency that leaves the
package is in Hz, not rad/s. `hamiltonian.eigen` divides by 2*pi for exactly
this reason.

**Signs.** With J > 0 the singlet of a coupled pair lies lowest. A spin with
gamma > 0 lies lowest along +B. Both are standard and both are asserted in
tests.

**Geometry.** The sensor axis is lab z. The residual field is
B = |B| (sin(theta), 0, cos(theta)). By symmetry about the sensor axis the
azimuthal angle is unobservable, so the field is two parameters, not three.

**Fz blocking is not generally valid.** Total Fz is conserved only when the
field lies along the sensor axis. A transverse component mixes Fz sectors,
which is precisely why it splits lines. Do not add Fz blocking to any code path
that can receive a tilted field. The blocking that does survive in general is
by the permutation symmetry of magnetically equivalent nuclei.

**rho(0) stays general.** The reduction of the weights to |<n|Mz|m>|^2 holds
only for thermal prepolarization followed by a sudden drop. For pulse-acquire
the weights are not sign-definite, and adiabatic-with-no-pulse gives no signal
at all. Never assume non-negative weights.

**A J-spectrum needs at least two distinct gamma.** If every spin has the same
gamma, Mz commutes with the coupling Hamiltonian and there is no spectrum.

**The gauge is a convention, and the convention is written down.** The
sampled couplings are orbits of the equivalence group, not pairs. Flipping
the sign of every coupling leaves the spectrum unchanged, so the orbit
carrying the largest-magnitude coupling is taken positive, ties going to the
lowest orbit index. Same-species nuclei that are not magnetically equivalent
are ordered by their sorted coupling vectors. `parameters.report_gauge()`
states all three in the words the paper needs; change the code and that
string changes with it. A prior written on the unquotiented space has to be
folded before it is compared with a posterior reported on the quotient, which
is what `priors.SignGaugedPrior` is for.

---

## Architecture

The forward model returns a **line list**: frequencies and complex amplitudes.
It never returns a binned spectrum. Relaxation, detector response and phase are
exact and free on a line list. Binning happens once, at the end, only to
reproduce what the real instrument did to the real data.

| Layer | Files | Status |
|---|---|---|
| 0 Operators, Hamiltonian, eigen | `operators.py`, `hamiltonian.py` | done |
| 1 Preparation, rho(0) | `preparation.py` | done |
| 2 Line list | `signal.py` | done |
| 3 Detector, relaxation | `detector.py` | done |
| 4 Spectrum and peak list | `fast_spectrum.py`, `spectrum.py` | done |
| 5 Priors, parameterization, gauge | `parameters.py`, `priors.py` | done |
| 6 NPE, calibration, reweighting | `inference/` | to write |

Each layer talks only to the one below it.

**Never build a full FID inside a loop.** `fast_spectrum.spectrum_at_bins`
evaluates the exact DFT of the truncated decaying signal at chosen bins.
Measured: 0.88 ms against 306 ms for the brute-force FFT, agreeing to 3e-12
relative. The brute-force path in `spectrum.acquire` exists to validate the
analytic one and for one-off plots, not for training data.

**The same peak picker runs on simulated and on real data.** Training the
network on exact line lists and then showing it picked peaks means it was
trained on a different measurement from the one it is asked about.

---

## Test discipline

`tests/test_physics.py` is the contract. Every assertion in it encodes a
physical claim that the project depends on.

- If a test fails after your change, the default assumption is that the new
  code is wrong, not that the test is wrong.
- Never weaken an assertion or widen a tolerance to make something pass. If a
  tolerance is genuinely wrong, say so in the message and explain the physics.
- New physics gets a new test in the same commit.
- Compare intensities against a reference amplitude, never against an absolute
  constant. The amplitudes carry gamma and are of order 1e16.

---

## Dependencies

Core is numpy and scipy only, and stays that way. `torch`, `sbi`, and
`ultranest` belong to layer 6 and must not be imported anywhere in layers 0 to 5.

---

## Style

Plain prose in docstrings. Explain why a thing is done, not what the line does.
Where a number was measured rather than assumed, write the measured value into
the docstring so the next reader does not have to re-derive it.
