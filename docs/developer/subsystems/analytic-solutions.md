# Analytic solutions

Exact solutions are the code's source of truth. They are the only diagnostic that
can tell you a solver returned a *wrong* answer rather than an unconverged one —
the SolCx port caught a direct `ksponly + lu` solve silently mangling the singular
saddle on a free-slip problem, which no residual norm reported.

This document governs how they are implemented, how a new one earns trust, and
where each vendored reference kernel came from.

## Where they live

`underworld3.analytic` — one namespace, reached as `uw.analytic.<Name>(mesh, ...)`.

```{note}
`uw.function.analytic` is the historic location. It is a *compiled extension
module*, which is why the suite could not grow there: the name is owned by a
`.so`, so no submodules or pure-Python solutions can live under it.
```

## One implementation form: SymPy

**Every analytic solution is a pure SymPy expression on `mesh.X`.** No exceptions,
no tiers. A solution written this way:

- compiles through the normal JIT path when handed to a solver, so it is C where
  speed matters;
- carries its own analytic Jacobian, which a hand-written C kernel cannot supply;
- can be used as a Dirichlet boundary value, which a kernel-backed function cannot;
- evaluates through `uw.function.evaluate` like any other field, vectorised.

Point evaluation is not in a hot loop — it is tests and error norms — so the SymPy
form costs nothing where it is used, and wins where it matters.

## Transcribing a reference kernel

Most of the classical solutions (Velic's, and PETSc's copies of them) exist as
machine-generated C: straight-line single-assignment code, `t125 = 0.4e1*t81*t83 +
...`, several hundred lines per branch. `scripts/maple_c_to_sympy.py` converts
these mechanically.

**Preserve the grouping term for term. Never `simplify()`.** The Maple grouping is
what keeps `sinh(k)*exp(-k)`-style products numerically stable at large wavenumber
and large viscosity contrast. A re-derivation is a *different* grouping and can
lose eight digits in exactly the regime the benchmark exists to probe. Do not
re-derive what you cannot revalidate.

The reference kernel is **kept**, not deleted, and stays reachable:

```python
sol = uw.analytic.SolCx(mesh, ...)                    # SymPy (default)
ref = uw.analytic.SolCx(mesh, ..., reference=True)    # the Velic C, verbatim
```

so "is this the transcription or the model?" stays a one-line question.

## The validation protocol

Velic's kernels are obsessively careful. The risk sits entirely on our side of the
conversion, and pointwise agreement on a sample is a weak test — it can pass while
the transcription is wrong off-sample, or wrong in its derivatives, which is what
the solver actually consumes. **No transcription lands until all six gates pass.**

### Gate 1 — two independent oracles, not one

PETSc maintains its own copy of several Velic solutions, with different call
signatures from Underworld2's: `SolCxSolution` and `SolKxSolution` in
`src/snes/tutorials/ex69.c`. Agreement of UW2-C ↔ PETSc-C ↔ our SymPy is a far
stronger statement than agreement with either alone, and both trees are already
available. For SolKx there is a third: PETSc's `ex75.h` is a 41×41 table of
tabulated reference values at `B=100, kn=km=100π` — an independent fixture in the
hardest wavenumber regime.

### Gate 2 — adversarial sampling, max error not mean

Stratified so no region goes unsampled, and deliberately loaded with the hard
places: on and either side of a viscosity interface, on the boundaries, at
corners, and across parameter extremes (viscosity ratios spanning `1e-6` to `1e8`,
wavenumbers 1–8). Report the **maximum** relative error. Threshold `1e-10`. A
failing point gets investigated, not sampled around.

### Gate 3 — derivatives, against independently derived output

The solver consumes derivatives of these fields, and a transcription can be
pointwise right yet derivative-wrong if a dropped term happens to vanish on the
sample set.

The reference kernels return `vel`, `pressure`, `total_stress` and `strain_rate`
as *separately derived* quantities — they do not differentiate the velocity to get
them. So compare SymPy's **symbolic** derivative of the velocity against the
kernel's own strain rate, and the symbolic constitutive stress against its total
stress. That is a genuine independent check rather than a tautology.

### Gate 4 — the physics residual, which needs no oracle

Substitute the transcribed fields back into the equations they claim to solve and
confirm the residual vanishes:

$$\nabla\cdot\mathbf u = 0, \qquad
  \nabla\cdot\left(2\eta\,\varepsilon(\mathbf u)\right) - \nabla p + \mathbf f = 0$$

using the solution's *own* `fn_viscosity` and `fn_bodyforce`; plus traction
continuity across any interface, and the boundary conditions the solution claims.

This tests the transcription **as a solution** rather than as a table of numbers,
and it catches what a convergence test structurally cannot. If the transcription
and the solver share the same mistaken convention — a sign, a factor of two — the
solve converges beautifully to the wrong answer. That is not hypothetical: the
original SolCx port hit exactly this, with Underworld2's documentation quoting
$-\cos(\pi x)\sin(n\pi z)$ where UW3's momentum convention needs $+\cos$. Gate 4
is the guard, because it never consults the solver.

### Gate 5 — negative control

Flip the sign of a single coefficient in the transcription and confirm that gates
1–4 all **fail**. A gate that passes a deliberately broken input is measuring
nothing. Do this once per solution, in the test suite, with the perturbation
applied programmatically.

### Gate 6 — separate transcription error from conditioning

If SymPy and C disagree near threshold, we need to know whether that is our
transcription or the C's own double-precision cancellation. Evaluate the SymPy
form under `mpmath` at 50 digits and compare against both. This makes "the Maple
grouping is numerically stable" a measured claim rather than an assumption — and
if the high-precision form agrees with the kernel's intent while the
double-precision form does not, that is a concrete instruction to preserve
grouping harder.

### After the gates: pin it

Freeze a table of validated values as a test fixture — the pattern PETSc itself
uses with `ex75.h` — so later refactors cannot drift silently. Record the measured
maximum error, the sampling design, and which oracles were used in the solution's
docstring. Not just "validated".

## Status: SolCx transcribed and validated

The transcriber is in place and measured. On `solCx.c` — the largest kernel in
the family at 1500 lines — it reads both arrangements and both spatial branches
in **0.25 s**, and the resulting expressions are 2000–7000 operations, well
within what SymPy, `lambdify` and the JIT handle. That measurement is what makes
the all-SymPy form viable for the whole family; it was not obvious in advance.

Gate 1 and Gate 2 on the `_solCx_A` arrangement, against the published kernel, at
40 stratified points per case:

| eta_A | eta_B | x_c | n | max relative error |
|---|---|---|---|---|
| 1 | 10 | 0.5 | 1 | 4.3e-15 |
| 1 | 1e3 | 0.5 | 1 | 1.7e-14 |
| 1 | 1e6 | 0.5 | 1 | 1.1e-14 |
| 1 | 1e8 | 0.5 | 1 | 9.3e-15 |
| 1 | 1e6 | 0.5 | 3 | 4.9e-15 |
| 1 | 1e6 | 0.75 | 1 | 4.1e-15 |
| 1e6 | 1 | 0.5 | 1 | 1.5e-14 |
| 1e3 | 1 | 0.25 | 2 | 8.9e-16 |
| 1 | 1e-6 | 0.5 | 1 | 1.5e-14 |

### What the gates caught, and what turned out to be true

Three things looked like transcription failures and were not. All three are worth
knowing before transcribing the next kernel.

**`_solCx_B` is not a second conditioning — it is the mirror.** The published
source dispatches on $\eta_A > \eta_B$, which reads as two arrangements of one
formula chosen for numerical conditioning. Transcribing `_solCx_B` directly gave
an answer wrong by a factor of tens in every regime, while `_solCx_A` matched to
1e-14 — including in the regime where the dispatcher runs `_solCx_B`. Evaluated
exactly at 50 digits the relationship is clean: $B(x,z) = A(1-x, z)$. `_solCx_B`
solves the mirrored problem so the algebra derived for a stiff *left* column can
be reused when the stiff column is on the right, and reflects on the way out.

So only `_solCx_A` is transcribed and there is no dispatch. That is safe because
the stated reason for the dispatch was checked rather than assumed: the table
above spans ratios from 1e-6 to 1e8 in both directions and the error never leaves
1e-14. **Do not assume a kernel's internal dispatch means what its condition
suggests — evaluate both arms exactly and compare.**

**The isoviscous case is a genuine limitation, and it is now refused.** The closed
form carries $Z_R - 1$ in several denominators, so $\eta_A = \eta_B$ is a
removable singularity. SymPy cancels it when the expression is evaluated
symbolically at a point — verified to 40 digits against the kernel — but not in
the compiled form, where it survives as $0/0$. Rather than return nonsense,
`SolCx` raises for equal viscosities: it is a viscosity-jump benchmark, and
uniform viscosity is a different solution. Parameters are substituted as exact
`Rational`s regardless, since that is what lets the cancellation happen at all.

**The first gate run's 1e12 errors were the metric, not the transcription.** A
pointwise relative error divides by the true value, and these fields pass through
zero, so the ratio explodes wherever the solution is small — the values were
close all along. Normalise by the field's magnitude over the sample. Suspect the
metric before the result when every case fails alike.

### The harness

`underworld3.analytic._validation` holds the checks, so a new transcription gets
them by calling four functions rather than reinventing them:

| function | gate |
|---|---|
| `adversarial_points` | 2 — stratified, plus interface, walls and corners |
| `reference_agreement` | 1, 2 — against the kernel, worst case, normalised |
| `incompressibility_residual` | 4 — $\nabla\cdot\mathbf u$, no oracle |
| `momentum_residual` | 4 — $\nabla\cdot\sigma + \mathbf f$, no oracle |
| `strainrate_consistency` | 3 — derivatives, against separately derived output |
| `high_precision_value` | 6 — 50 digits, to separate our error from the kernel's |

Measured on SolCx at contrast 1e3: incompressibility 3.6e-17, momentum 2.3e-16,
strain rate 8.5e-16 — each in under half a second.

**Evaluate through `lambdify`, not `uw.function.evaluate`.** The checks
differentiate the fields, and a viscosity-jump solution puts a large `Piecewise`
inside a stress derivative. Routed through the JIT that combination takes so long
to generate and compile that the suite becomes unrunnable — the same blow-up seen
with `add_nitsche_bc` on SolCx. These expressions are pure SymPy in the mesh
coordinates, so `lambdify(..., cse=True)` is both correct and three orders of
magnitude faster. `_validation.sample` does this; note it must first swap the
mesh coordinates for plain symbols, which `lambdify` cannot bind directly.

Gate 5, the negative control, belongs in each solution's test rather than the
harness: perturb one coefficient and assert the other checks fail.
`test_the_checks_reject_a_broken_transcription` is the pattern — it perturbs the
velocity by a part in a thousand and requires both the comparison and the
oracle-free residual to report it.

## One interface, so a mistake cannot be local

Each solution used to assemble its own `fn_*` attributes, and each had its own
test file. That combination let a real error through: SolNL's kernel publishes
the deviatoric stress, it was stored as the total, and its momentum residual was
**1.06** rather than zero. Its test file checked agreement with the kernel and
incompressibility — both passed — and nothing checked the momentum balance.

Two changes make that class of mistake structural rather than a matter of
remembering.

**Assembly happens once.** A solution hands its components to
`AnalyticSolution.set_fields`, which applies the conventions. Whether the source
publishes $\sigma$ or $\tau$ is a class-level declaration,
`stress_is_deviatoric`, honoured in exactly one place. A solution can no longer
quietly disagree with its neighbours about what its own stress means.

**Conformance is checked for every registered solution.**
`tests/test_1024_analytic_conformance.py` iterates over `uw.analytic.available()`
and applies the same checks to all of them: the contract is fully populated, the
metadata is declared, the flow is incompressible, the momentum balance holds, the
stress and strain rate agree, and the boundary conditions configure a solver. A
solution added later is covered the moment it is registered.

Where a solution differs from the others it says so through the contract rather
than by being exempted:

| method | default | why a solution overrides it |
|---|---|---|
| `sample_points` | unit box or cube, plus faces and corners | the elliptical inclusion is not box-filling, and the conformal map is singular at its foci — a generic sampler lands on both |
| `boundaries` | box wall labels | a curved geometry has its own |
| `apply_boundary_conditions` | free slip or Dirichlet mixin | — |
| `stress_is_deviatoric` | `False` | the source publishes $\tau$ |

One consequence worth knowing: `_validation.sample` returns real values, and if
an expression evaluates complex it checks the imaginary part is round-off before
discarding it. The elliptical inclusion is built from complex potentials and is
real-valued without SymPy being able to prove it — but a *genuinely* complex
result would mean the construction is wrong, and silently taking the real part
would hide exactly that.

## The stress convention is not uniform across the family

**Check which stress a kernel publishes. Do not read it off the variable name.**

| solution | its stress output is |
|---|---|
| SolA, SolB, SolC, SolCx, SolDA, SolKx | total (Cauchy) $\sigma$ |
| SolKz, SolNL, SolDB2d, SolDB3d | deviatoric $\tau$ |
| SolM | published, but wrong — see below |

SolA is the clearest case to read: its source writes `u3 = 2*kn*ss_z - pp`, with the
pressure subtracted in plain sight. SolKz's writes the same quantity without it.

### The body force is minus the density

Most of these kernels negate internally — they write `rho = -sigma*sin*cos` and
force with `+sigma*sin*cos`. SolC does not: it accumulates the density itself, so
the transcription negates. Measured, as always: as summed the momentum residual
is 1.8; negated it is 1.6e-16.

Worth stating as a rule because the sign is invisible to everything except the
momentum balance. Incompressibility and the free-slip conditions hold either way.

### One published stress is simply wrong

SolM's kernel declares its viscosity as $(1 + \cos(r\pi x))\eta_0 + 1$ and then
computes its stress as $2(\eta - 1)\dot\varepsilon$ — the constant part is
missing. The difference from $2(\eta-1)\dot\varepsilon$ is *exactly* zero, so
this is a defect in the source rather than a transcription slip: using the
published stress leaves the momentum residual at **0.21**, while deriving it from
the kernel's own strain rate and viscosity gives **1.7e-16**.

Everything else SolM publishes — velocity, pressure, strain rate, viscosity, body
force — is mutually consistent. Only the stress output is defective, so the
transcription supplies the strain rate to `set_fields` and lets the stress be
derived.

This is the case for having a check that consults no reference. Comparing SolM
against its own kernel would have reproduced the error faithfully and reported
agreement.

SolKz is the trap: it writes into an array literally called `total_stress`, and
the contents are the deviator. Taking the name at face value leaves the momentum
residual at order $|\mathbf f|$ *and* manufactures a horizontal body force in a
benchmark that has none — a large, structured error that reads like a
transcription failure rather than a convention one.

Two cheap signatures tell them apart, and both are worth running on any new
kernel:

- a deviator is traceless, so its $xx$ and $zz$ entries are exact negatives;
- $\tau = 2\eta\dot\varepsilon$, and the strain rate follows from the velocity —
  a *different* output of the same kernel, so the comparison is independent.

On SolKz the shear component agreed with $2\eta\dot\varepsilon$ to machine
precision while the normal components did not agree with anything, which located
the problem immediately. `test_kernel_publishes_the_deviatoric_stress` keeps
both checks.

## A solution that is derived rather than transcribed

`EllipticalInclusion` (Schmid & Podladchikov 2003) is the one case so far where
the published source does **not** contain what we need. The authors' MATLAB gives
pressure, deviatoric stress and the rotation rate; it does not give velocity. So
the Muskhelishvili potentials had to be recovered from the fields, and the
velocity built from those.

That changes what validation means. There is no kernel to compare velocity
against, so the checks have to come from physics and from internal consistency:

| check | pairs against |
|---|---|
| $\eta\nabla^2\mathbf v = \nabla p$ | the *published* pressure |
| velocity continuity across the interface | the interior uniform-gradient field |
| far field | the imposed shear, computed independently |
| interior uniformity | the Eshelby property |

Measured: Stokes residual 1.4e-17, $\nabla\cdot\mathbf v$ 1.7e-16, far field
agreeing to a few parts in $10^6$ (the inclusion's own $1/r^2$ perturbation at
finite distance, not error).

Two things about this derivation are worth carrying to the next one.

**What the published data cannot constrain.** A purely imaginary constant in
$\varphi'$ contributes nothing to pressure ($-2\,\mathrm{Re}\,\varphi'$) or to
stress (which involves $\varphi''$). It is a far-field rigid rotation. Reading
the potentials off stress and pressure alone therefore loses the spin entirely,
and an imposed simple shear comes back as pure shear — with the correct strain
magnitude, which is what makes it easy to miss. Its value came from a different
published expression: taken to a circle, the rotation rate collapses to
$-\dot\gamma/2$ for every viscosity ratio. **When reading potentials back out of
fields, ask what the fields are blind to.**

**Branch cuts are not cosmetic.** Inverting $z = \zeta + 1/\zeta$ as
`sqrt(z**2 - 4)` cuts along a ray and selects the root *inside* the unit circle
for $x < 0$ — the wrong Riemann sheet. The far field then comes out asymmetric,
roughly three times too fast on one side. Written `sqrt(z-2)*sqrt(z+2)` the cut
lies on $[-2, 2]$, the slit the map already has, and $|\zeta| > 1$ everywhere
outside. Sampling only positive $x$ would have missed this, which is why the test
samples both.

A SymPy consequence: the correct branch defeats `re()`/`im()`, which then survive
into derivatives as an unprintable `Derivative(re(...))`. Building the components
as $(w + \bar w)/2$ and $(w - \bar w)/2i$ avoids them, with `conjugate` obtained
by flipping the sign of `I` — for an expression in real symbols that is exactly
conjugation, and unlike `sympy.conjugate` it distributes through a square root.
The result is real-valued but complex-typed; SymPy cannot prove the imaginary
part vanishes, so callers take `.real`.

## The family is complete

All twelve Velic solutions are transcribed, plus the Schmid & Podladchikov
inclusion. Every one is validated by the conformance suite.

Two lessons from the last three, which I had recorded as too large to attempt:

**Measure before estimating.** SolDA was written off on a guess about expression
size. Measuring took two minutes: one mode is 0.07 s and about four thousand
operations, putting twenty modes in the same range as SolKz. SolH was written off
on the source's own warning that it is "very expensive" — true of a *compiled*
kernel, which re-sums every mode at every evaluation point, and backwards for a
transcription, where the sum is built once and each mode is the smallest in the
family at ninety operations. Both transcribed and validated on the first attempt.

**The named obstacle is the cheap one.** For SolDA the blocker I could point at —
chained assignment, `del_rhoB = del_rhoA = del_rho` — was a ten-line fix, while
the one I could only estimate was not an obstacle at all. Same for SolH: the C
ternary and the index-conditioned guards were mechanical additions;
the cost was imagined.

### What the last three needed from the reader

| addition | why |
|---|---|
| `CSource.loop_body` | the series solutions accumulate over modes with `+=`, which is not an assignment; the caller evaluates per mode and sums in SymPy |
| chained assignment | `a = b = c;` read as one statement has `b = c` as its *value*, which is not an expression |
| C ternary and `&&`/`\|\|` | SolH guards its zero modes with `(n!=0 \|\| m!=0) ? … : …` |
| `resolve_branches` | guards on loop indices have an answer at transcription time. Left unresolved, `evaluate_block` reads every branch in order and each guarded variable keeps the *last* one — in SolH that silently zeroes two velocity components, which looks plausible rather than broken |

## The scalar transport family

The Stokes solutions solve for a velocity-and-pressure pair. The scalar solutions
solve for one field, so they declare `solves = "transport"` and carry a different
set of `fn_*`:

| | Stokes | transport |
|---|---|---|
| unknowns | `fn_velocity`, `fn_pressure` | `fn_solution` |
| material | `fn_viscosity` | `fn_coefficient` |
| forcing | `fn_bodyforce` | `fn_source` |
| set by | `set_fields(...)` | `set_scalar_field(...)` |
| residual | momentum + incompressibility | `transport_residual` / `diffusion_residual` |

These were already in the repository — written inline in the tests that used them.
Collecting them gains the residual check, which they never had, and a name to cite.

| solution | equation | previously |
|---|---|---|
| `Poisson1D` | $\nabla^2 u + f = 0$, three sources | `test_1000_poissonCart.py` |
| `TwoLayerDarcy` | $\nabla\cdot(k\nabla p) = 0$ across a permeability jump | `test_1004_DarcyCartesian.py` |
| `ErfcDiffusion` | $\partial_t u = D\nabla^2 u$ | `test_1005_TransientDarcyCartesian.py` |
| `AdvectedFront` | $\partial_t c + u\,\partial_x c = \kappa\,\partial_{xx} c$ | `test_1100_AdvDiffCartesian.py` |

The transient ones expose time as a symbol, so a solver is checked at whatever
time it actually reached:

```python
sol = uw.analytic.ErfcDiffusion(mesh, diffusivity=0.5)
exact = sol.fn_solution.subs(sol.t, t_end)
```

Starting a comparison from a smooth profile at $t > 0$ rather than from the step
itself is the point of using these: the step is not representable on the mesh,
which is what makes `test_1100`'s current comparison fragile.

They cannot reuse the Stokes boundary-condition mixins, which apply a *velocity*.
`_Transport` prescribes `fn_solution` on every wall instead.

### The residual has to be the equation the solution actually solves

`AdvectedFront` reported a residual of 1.44 against 0.00 for everything else. The
solution was right; the check was the wrong equation. `diffusion_residual` tested
$\partial_t u = \nabla\cdot(k\nabla u)$, and an advecting front does not satisfy
that — it satisfies advection-diffusion.

The failure is worth recording because of how it presents: an order-one residual
next to a column of zeros reads unambiguously as a broken solution, and the
tempting next step is to go looking for the transcription error. The fix was to
include the advection term, which is the general case — a purely diffusive
solution declares no advecting velocity and the term drops out, leaving the other
three at zero exactly as before. The lesson mirrors Gate 4's: a residual only
means something if it is the residual of the right equation, and a check narrower
than the family it is applied to will convict a correct solution.

## The Richards family

Unsaturated flow is the one nonlinear scalar equation in the suite:

$$C(\psi)\,\frac{\partial\psi}{\partial t}
    = \nabla\cdot\!\left[K(\psi)\left(\nabla\psi + \hat y\right)\right]$$

so it declares `solves = "richards"` and carries `fn_conductivity` and
`fn_capacity` rather than a single coefficient. Gardner's exponential model
$K = K_s e^{\alpha\psi}$ is the case that closes, because $u = e^{\alpha\psi}$
linearises it *exactly* — under that substitution Richards becomes linear
advection–diffusion in $u$, which is why `GardnerTransient` is an Ogata–Banks
form and shares its shape with `AdvectedFront`.

| solution | content |
|---|---|
| `GardnerSteady` | constant flux down a column; head is $\ln[(u_0-q^*)e^{-\alpha y} + q^*]/\alpha$ |
| `GardnerTransient` | a wetting front advancing at $V = K_s/\Delta\theta$ |

Both already existed as NumPy functions in `utilities/retention_curves.py`. Those
functions keep their signatures and now evaluate the same SymPy expression the
classes build, so there is one formula rather than two copies that can drift.

### Two things this family taught the harness

**A residual can be degenerate rather than wrong.** The first `richards_residual`
normalised by the flux divergence — which *is* the residual. It reported exactly
`1.00` for a solution that turned out to be exact to the last bit. An order-one
number from a normalised residual is not automatically a failing solution; it can
be a scale that divides the quantity by itself. The fix is to normalise by the
terms that have to *cancel*, kept separately.

**Not every perturbation is a negative control.** Gate 5 says a check that passes
a broken input is measuring nothing — but scaling $K$ by a constant leaves the
steady residual at zero, and that is correct: it is a genuine symmetry of
$\nabla\cdot[K(\nabla\psi+\hat y)]=0$, not a defect the gate missed. A control has
to break something the solution actually asserts. Three that do: a wrong $\alpha$
inside $K$ (0.64), a head scaled by 1% (0.0099, tracking the perturbation), and a
conductivity that ignores the head at all (1.0). Both facts are asserted in
`tests/test_1026_analytic_richards.py`, the symmetry included, so neither is left
as a claim in prose.

### `erfc` is not in NumPy

`lambdify(..., "numpy")` falls back to the scalar `math.erfc` without complaint;
the failure surfaces much later as `only 0-dimensional arrays can be converted to
Python scalars`, from generated code, at the first array. It went unseen for as
long as it did because differentiating an `erfc` removes it — every earlier
residual differentiated, and only the Richards head keeps one inside a logarithm.
`_validation.sample` now asks for `["scipy", "numpy"]`.

## Provenance

Each vendored reference kernel keeps its original copyright header.

| Source | Licence | Compatible with UW3 (LGPL-3-or-later) |
|---|---|---|
| Underworld2 `Velic_sol*` kernels | LGPL-3 | Yes — same licence |
| PETSc `ex69.c`, `ex13.c`, `ex24.c`, `ex45.c` | BSD-2-Clause | Yes — permissive; retain the notice |
| Schmid & Podladchikov MATLAB (`dwschmid/muskhelishvili`) | BSD-3-Clause | Yes — permissive; retain the notice |
| `assess` (Kramer et al. 2021) | External, optional dependency | Not vendored; wrapped lazily |

## Optional dependencies

A solution requiring a package Underworld3 does not depend on is wrapped lazily,
in the style SciPy uses for its optional backends: the import happens at
construction, and its absence raises with an install message rather than breaking
`import underworld3`. `uw.analytic.available()` lists such solutions and marks
them unavailable rather than omitting them — omitting would make the listing
truthful about what constructs and silent about what exists, leaving a user with
no way to discover that a benchmark is one `pip install` away.

There is one: `CylindricalStokes`, wrapping `assess` (Kramer et al. 2021) for
curved-geometry Stokes. It is declared as the `benchmarks` extra:

```bash
pip install "underworld3[benchmarks]"
```

Four scripts under `docs/examples/` already imported `assess` while nothing
declared it, so on a normal install they failed with a bare
`ModuleNotFoundError`.

### It is an oracle, not a member of the family

`assess` gives numeric callables, so there is nothing to differentiate — a
Kramer solution can be compared against a solver but **cannot be checked against
the equations it claims to solve**. None of the six gates reaches it.

That is a real gap, not a technicality, so it is declared rather than
described: `symbolic = False`, and the conformance sweep excludes on the
declaration. The sweep then asserts what it excluded and why, so an accidental
exclusion — a mistyped class attribute — fails the suite instead of quietly
shrinking it.

Two class attributes carry this:

| attribute | meaning |
|---|---|
| `symbolic` | fields are SymPy on `mesh.X`. False means the residual gates cannot be applied at all |
| `requires` | name of an optional package, or `None` |

## Adding a new solution

1. Subclass `AnalyticSolution` and one of the boundary-condition mixins
   (`FreeSlipWalls`, `FixedWalls`) — or `_Transport` for a scalar solution.
2. Build the exact fields on `mesh.X` in `__init__`; set `dim`, `reference`, and
   the `eqn_*` LaTeX strings that document the *problem*. Set the fields through
   `set_fields` (Stokes) or `set_scalar_field` (transport) rather than assigning
   `fn_*` directly — that is where the stress convention and the advection term
   are applied, and the conformance suite trusts the declaration.
3. Export it from `underworld3/analytic/__init__.py` and register it in
   `_SOLUTIONS` — a namespace entry and the registry entry land in the same PR.
4. If it was transcribed from a reference kernel, clear all six gates and pin the
   validated values.
5. Add a convergence test: the error must decrease under refinement *and* clear an
   absolute floor.
