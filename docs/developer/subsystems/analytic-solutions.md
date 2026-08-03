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
| SolA, SolB, SolC, SolCx, SolKx | total (Cauchy) $\sigma$ |
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

## What is not transcribed yet

Two of the Velic solutions remain. Both are reachable with the machinery here —
the mode loop is proven by SolC — but neither is a small addition, and the source
for each is already vendored.

**SolDA** (`solDA.c`, 974 lines) is the largest kernel in the family: a truncated
series with *both* a viscosity jump and a rectangular forcing, so it combines what
SolCx and SolC each test separately. Probed, not attempted — three things are in
the way:

- **chained assignment.** The loop opens with `del_rhoB = del_rhoA = del_rho;`,
  which the reader takes as one statement assigning `del_rhoA = del_rho` to
  `del_rhoB` — not valid as an expression, so it raises. `_STATEMENT` needs to
  split a chain into its individual targets.
- **two sequential spatial branches inside the mode loop**, both `if (z < zc)`,
  each about 790 lines. Each becomes a `Piecewise` *per mode*, so the expression
  structure compounds with the mode count in a way SolC's does not.
- **size**. SolC at 40 modes builds in 2.4 s from a 186-line kernel. SolDA's loop
  body is an order of magnitude larger, so measure the per-mode expression before
  choosing a default `modes` — and expect to justify a much smaller one.

**SolH** (`solH.c`) is 3D and needs three things at once:

- a **double** mode loop, `n` and `m` each to `nmodes`. At the published default
  of 30 that is 900 terms, and its own header warns that SolH "can become *very*
  expensive to compute". Expect to choose a smaller default and document the
  trade-off, as SolC does.
- nested `if`/`else` inside the loop selecting `del_rho` for the `n = 0` and
  `m = 0` modes. The conditions are on the loop indices, which are known integers
  at transcription time, so the right branch can be picked per mode rather than
  turned into a `Piecewise` — but `CSource.branches` handles one level, not three.
- six stress components rather than three, laid out `xx, yy, zz, xy, xz, yz`.

Its output mapping is transposed like SolKz's: `vel[0] = sum3`, `vel[1] = sum2`,
`vel[2] = sum1`. Read it from the source, do not assume.

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
them unavailable rather than omitting them.

## Adding a new solution

1. Subclass `AnalyticSolution` and one of the boundary-condition mixins
   (`FreeSlipWalls`, `FixedWalls`).
2. Build the exact fields on `mesh.X` in `__init__`; set `dim`, `reference`, and
   the `eqn_*` LaTeX strings that document the *problem*.
3. Export it from `underworld3/analytic/__init__.py` and register it in
   `_SOLUTIONS` — a namespace entry and the registry entry land in the same PR.
4. If it was transcribed from a reference kernel, clear all six gates and pin the
   validated values.
5. Add a convergence test: the error must decrease under refinement *and* clear an
   absolute floor.
