# Building the ZULF SBI project

The proposal says what to do. This says how to put it together, in what order,
and what the working code already changed about the plan.

Everything below refers to the package in `zulf/` and the tests in `tests/`.
A hundred and four tests pass, in 1 s. Twenty of them encode every physics claim
the proposal makes, so if a later change breaks one, you find out immediately
instead of in a referee report. The rest are layer 5.

---

## 1. The shape of the thing

Seven layers. Each one only talks to the one below it. That is the whole point:
you can replace the network without touching the physics, and replace the
physics without touching the network.

| Layer | What it does | Files |
|---|---|---|
| 0 | Spin operators, Hamiltonian, eigen-decomposition | `operators.py`, `hamiltonian.py` |
| 1 | Preparation, i.e. $\rho(0)$ for each protocol | `preparation.py` |
| 2 | **Line list**: $(\nu_k, A_k)$ with complex $A_k$ | `signal.py` |
| 3 | Detector response and relaxation, applied on the line list | `detector.py` |
| 4 | Line list to spectrum to peak list | `fast_spectrum.py`, `spectrum.py` |
| 5 | Priors, parameterization, symmetry gauge | `parameters.py`, `priors.py` |
| 6 | NPE training, calibration, refinement, reweighting | *to write* |

**The one decision that everything else hangs off**: the forward model returns a
list of lines, not a binned spectrum. A 10 mHz linewidth across a 500 Hz band
needs about $2\times10^5$ bins. Carrying that through $10^5$ training
simulations is 160 GB. Carrying about 40 lines through the same $10^5$
simulations is 30 MB. Relaxation, detector response and phase are all exact and
free on the line list. Binning happens once, at the end, and only to reproduce
what the real instrument did to the real data.

---

## 2. Build order

Build in the order that lets the project fail early and cheaply. Steps 1 to 4
answer whether the idea works at all. Everything after that is scaling.

**Step 1. Forward model, checked against cases with known answers.** Done. The
tests that matter:

- two spins give one line at $J$;
- the methyl group XA3 gives lines at $J$ and $2J$ with intensity ratio exactly
  $5/4$, and $J_{HH}$ moves nothing;
- four spins with equal $\gamma$ give no spectrum at all;
- a 1 nT field along the sensor axis moves the line by $\sim 2\times10^{-6}$ Hz,
  a 1 nT transverse field splits it by 53 mHz;
- an adiabatic drop with no pulse gives zero signal, and a DC pulse gives
  weights that go negative.

**Step 2. Exact likelihood, then nested sampling on two spins.** Before any
network exists. This gives the reference posterior *and* proves the whole
inference stack. If nested sampling on formic acid does not reproduce the
sub-mHz number from the group's own fitting method, nothing later will.

**Step 3. Peak-list summary, run on simulated *data*, not on the true line
list.** This is the failure mode that is easy to walk into: train the network on
exact line lists, then show it peaks found by a peak picker, and it has been
trained on a different measurement from the one it is asked about. No prior
width repairs that. Run `pick_peaks` on both.

**Step 4. First NPE on two spins, compared against step 2.** This is the go/no-go.
One coupling, plus field, decay, scale, noise. If the flow cannot find the right
basin here, stop and rethink.

**Step 5. Scale to three, four, five spins.** Methanol, then acetonitrile.

**Step 6. Real data.** Only now.

The proposal's timeline maps onto this cleanly. The one change: move nested
sampling from mid September to *now*, because it is the reference everything
else is judged against, and it is the cheapest thing in the project.

---

## 3. Three things the code found that change the plan

### 3.1 The cost is in the data pipeline, not the physics

The proposal counts diagonalization. Measured on one core:

| N spins | dim | ms per spectrum |
|---|---|---|
| 2 | 4 | 0.10 |
| 4 | 16 | 0.29 |
| 6 | 64 | 3.0 |
| 8 | 256 | 51 |

But building a 400 s record at 1.2 kSa/s and FFT-ing it costs **306 ms**, which
is a thousand times more than the four-spin diagonalization. At $10^5$ training
simulations that is 8.5 hours on one core, and the Hamiltonian is 0.01% of it.

Fix, implemented in `fast_spectrum.py`. The DFT of a truncated decaying
exponential has a closed form. For

$$y[k] = \sum_k 2\,\mathrm{Re}\!\left[A_k e^{-(R_k + 2i\pi\nu_k)\,k\,\Delta t}\right],$$

the normalized rfft is exactly

$$Y[j] = \frac{1}{N}\sum_k \left[A_k\,G(z_k^+) + A_k^*\,G(z_k^-)\right],
\qquad G(z) = \frac{1-z^N}{1-z},$$

with $z_k^\pm = \exp(-R_k\Delta t \mp 2i\pi\nu_k\Delta t - 2i\pi j/N)$. So the
spectrum can be evaluated at any chosen bins, and only bins within a few
linewidths of a line carry signal. Noise is white in time, so it is circular
complex Gaussian per bin with variance $\sigma^2/N$ and gets added bin by bin.

Measured: agrees with the brute-force FFT to $3\times10^{-12}$ relative, keeps
766 bins out of 240,000, and costs **0.88 ms** instead of 306 ms.
$10^5$ training spectra go from 8.5 hours to **1.5 minutes**.

This is what makes "runs on a laptop" true rather than aspirational.

### 3.2 The importance-reweighting step, as written, will fail its own criterion

This is the serious one. The proposal reweights the *network* samples, and sets
a failure criterion of 1% sample efficiency. Take a target posterior of width
1 mHz on $J$ (which is what the group's fitting method achieves) and a flow
proposal of realistic width. Gaussian on Gaussian, ESS fraction:

| proposal width | $d=1$ | $d=3$ | $d=5$ |
|---|---|---|---|
| 10 mHz | 0.14 | 2.8e-3 | 8.5e-5 |
| 100 mHz | 1.4e-2 | 7.4e-6 | 2.5e-6 |
| 1000 mHz | 1.4e-3 | 2.5e-6 | 2.5e-6 |

A flow reading a peak list will not be 10 mHz wide on $J$. So with the raw flow
as the proposal, the efficiency is below 1% for anything past one parameter, and
the project trips its own stated failure criterion for a reason that has nothing
to do with ZULF NMR.

The fix is a change of proposal, not a change of ambition. Use the flow to find
the *modes*, run a local fit plus a Laplace approximation at each mode, and use
the **mixture of local Gaussians as the importance proposal**. Then the proposal
width matches the target width by construction and the efficiency is order 1.
The flow's job becomes mode-finding and multiplicity, which is what it is
actually good at, and the exact likelihood supplies the precision, which is what
it is actually good at.

Say this in the proposal explicitly. A referee who knows Dax et al. will ask.

### 3.3 Two small corrections to numbers in the proposal

- **The transverse-field pattern.** Three lines around $J$ split by the sum of
  the Larmor frequencies, plus a low-frequency line at their **mean**: this is
  right at a general field angle. At exactly $\theta = \pi/2$ the central
  component vanishes by symmetry and only two lines are left. Both cases are in
  the tests.
- **The detector phase.** Through a single pole at 150 Hz the $2J:J$ ratio goes
  from 1.250 to 0.805, matching the proposal. The 62 degrees is the *absolute*
  phase of the $2J$ line. The phase difference the fit actually sees between the
  two lines is 19 degrees. Quote the 19, it is the one that biases anything.

There is also a blocking warning worth writing down. Total $F_z$ is conserved
only when the field lies along the sensor axis. A transverse component mixes
$F_z$ sectors, which is exactly why it splits lines. So the factor $2^3$ from
$F_z$ blocking is **not** available in the general case. What is available is
blocking by the permutation symmetry of magnetically equivalent nuclei, which is
what buys back the methyl groups.

---

## 4. Decisions to lock in before writing layer 5

1. **Parameterize by symmetry-distinct couplings.** Methanol has six pairwise
   couplings and one measurable number. A posterior over six would return the
   prior in five directions while looking like a result. Compute the orbit of
   the equivalence group (already stored on `SpinSystem.equivalent`) and
   parameterize by orbit representatives.
2. **Fix the gauge, then report on the quotient.** Global sign flip fixed by
   convention, stated. Permutations of equal-$\gamma$ nuclei removed by canonical
   ordering. Both are tested.
3. **Report prior-to-posterior movement for every parameter.** One number per
   parameter, e.g. KL or the ratio of standard deviations. This is what makes a
   flat direction visible instead of hidden.
4. **Use a set encoder, not a padded matrix.** The peak list is a set of
   (frequency, amplitude, width) triples with variable length. DeepSets or a
   small set transformer is permutation invariant by construction.
   `pad_peaks` exists as a fallback, not as the plan.
5. **Consider marginalizing the linear parameters analytically.** With Gaussian
   noise and a model linear in the line amplitudes, the amplitudes integrate out
   in closed form. This is Bretthorst's trick and it is already reference [11].
   It cuts the nuisance dimension a lot. The cost is that you give up the
   physical link between $\rho(0)$ and the amplitudes, so treat it as a
   robustness variant rather than the main model.
6. **Peak-list compression is not the precision bottleneck.** Measured: with an
   isolated line at HWHM 8 mHz, the picked frequency scatters by about 1 µHz at
   moderate SNR. The real risks in the summary are overlapping lines and a wrong
   lineshape, not the loss of resolution.

---

## 5. Named risks, in the order they will actually bite

1. Lineshape misspecification on real data. One decay rate per resolved
   multiplet is the starting model, and it is still a model.
2. Detector response not calibrated. Get the coil calibration if it exists;
   anything unmodelled here goes straight into $J$ as bias.
3. Which preparation sequence was used for the archived spectra. Equation (2)
   says the weights depend on it, and the pulse-acquire weights are not even
   sign-definite. Ask for this in the same email as the data.
4. Simulation-to-real gap. Wide priors are the defence, and the importance
   efficiency on real data is the test of whether the defence worked.

---

## 6. What to ask Dima

Liuba has passed this to him and he likes it, so the meeting is about turning it
into a working arrangement, not about selling the idea. Three concrete asks:

1. The archived formic acid, methanol and acetonitrile spectra, **plus** the
   pulse sequence used for each. Request this first; it is the long pole.
2. Detector calibration data for the magnetometer, if it exists.
3. A named person in the group to ask instrument questions, given the remote
   setup. This is the real ask. Without it, every question about the hardware
   costs a week.

Bring one page: the failure criterion, the two-stage design, and the three data
requests. Not the whole proposal.

---

## Running it

```bash
pip install numpy scipy pytest
python -m pytest tests/ -q          # 104 passed
```

Next dependencies, when you reach layer 6: `sbi`, `torch`, and either
`ultranest` or `dynesty` for the reference posteriors.
