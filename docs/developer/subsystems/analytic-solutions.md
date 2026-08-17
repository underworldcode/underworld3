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
| `apply_boundary_conditions` | refuses, with the helpers named | every solution writes it — see below |
| `stress_is_deviatoric` | `False` | the source publishes $\tau$ |

One consequence worth knowing: `_validation.sample` returns real values, and if
an expression evaluates complex it checks the imaginary part is round-off before
discarding it. The elliptical inclusion is built from complex potentials and is
real-valued without SymPy being able to prove it — but a *genuinely* complex
result would mean the construction is wrong, and silently taking the real part
would hide exactly that.

## Boundary conditions: composed functions, not inherited wall types

Every solution writes its own `apply_boundary_conditions`. It composes three
module-level helpers, each of which takes **the boundaries it applies to**:

```python
from underworld3.analytic import free_slip, prescribed_velocity, prescribed_scalar

free_slip(solver, boundaries, normal=None)          # strong rotated u.n = 0
prescribed_velocity(solver, boundaries, velocity)   # Dirichlet velocity
prescribed_scalar(solver, boundaries, field)        # Dirichlet scalar
```

A typical Velic solution is then three lines:

```python
    def apply_boundary_conditions(self, solver):
        """Free slip on all four walls; the enclosed box has a pressure nullspace."""

        free_slip(solver, self.boundaries)
        solver.petsc_use_pressure_nullspace = True
```

**Why not mixins.** This started as `FreeSlipWalls` and `FixedWalls` — mixed in,
looping over every boundary, applying one condition. That encodes "every boundary
is a wall of the same kind", which is true of the classical box benchmarks and
false of most of what the suite has to serve next: a spherical shell or annulus
with different conditions on the two radii, a faulted disc, a channel driven at
one end. A mixin cannot express those without growing a parameter for each, and
inheritance advertises the choice as if it were part of what the solution *is*.
Functions taking an explicit boundary list say what is imposed *where*, and a
solution needing two kinds calls two of them:

```python
    def apply_boundary_conditions(self, solver):
        free_slip(solver, ["Upper"])
        prescribed_velocity(solver, ["Lower"], self.fn_velocity)
```

**The pressure nullspace is stated by the solution, not by a wall type.** It is a
property of the domain — enclosed, so the pressure is determined only up to a
constant — and not of any one boundary's condition. Both mixins used to set it,
which hid that. Leaving it out on an enclosed domain is the failure this whole
suite exists to catch: a direct solve on the singular saddle returns a quiet,
wrong answer that only an exact solution exposes.

**Curved boundaries.** `free_slip` takes an optional `normal=`. Leave it out —
the solver's geometric facet normal is measure-weighted to match the
straight-facet integral the assembler evaluates, and keeps the constant pressure
a null vector to machine precision. Pass an analytic normal such as `X/|X|` only
when the constraint must follow the *true* surface rather than the mesh; it is
exact for the geometry but keeps a consistency error that grows with facet
non-uniformity. `CylindricalStokes`, the one curved-geometry solution here, uses
the default deliberately. See
[rotated-freeslip.md](rotated-freeslip.md) ("Which normal to use").

## The stress convention is not uniform across the family

**Check which stress a kernel publishes. Do not read it off the variable name.**

| solution | what we transcribe is |
|---|---|
| SolA, SolB, SolC, SolCx, SolDA, SolKx | total (Cauchy) $\sigma$ |
| SolKz, SolNL, SolDB2d, SolDB3d | deviatoric $\tau$ (SolKz: see the cut-point note below) |
| SolM | published, but wrong — see below |
| SolA | total, but its $zz$ component is wrong — see below |

SolB is the clearest case to read: its source writes `u3 = 2.0*Z*kn*ss_z - pp`,
with the pressure subtracted in plain sight. (Do **not** use solA's version of
that line as the exemplar, as this document once did — it is the one that is
missing the viscosity.)

Two further traps that are not about deviatoric-vs-total:

- **Component order is not uniform.** Every kernel here writes `[xx, zz, xz]`
  except `solKx.c`, which writes **`[xx, xz, zz]`** (:481-491, with the legend
  at :456). It has a different provenance from the rest — it is vendored from
  PETSc's `ex69.c`, not the Underworld tree.
- **Several kernels label the vertical velocity `u1`** and the horizontal `u2`
  — solA (:151), solB (:135), solC (:142), solDA (:910), solKz (:493). solCx
  (:1467) and solKx (:456) do not. `solH.c` reverses all three (:173-182). The
  transcription maps components explicitly rather than positionally; a swap here
  is caught by the momentum residual only because the components have different
  functional forms, and would be invisible in a symmetric problem.

Both are uniform across the family in the two ways that matter for a solve, and
neither is stated in most of the files — both had to be measured by
finite-differencing the kernels' own outputs: the momentum sign is
$\nabla\cdot\sigma + \mathbf f = 0$ with $\mathbf f = -\rho\hat z$ under unit
gravity in $-z$, and pressure is positive in compression.

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

SolKz is the trap, though it needs stating more carefully than it was here
originally. `solKz.c` *does* convert to the total stress, and says so — `u6 -=
u5; /* get total stress */` (:490), restated at :527 as `/* sigma = tau - p */`.
The array the C function returns is the total. What is deviatoric is **what our
transcription captures**: the transcriber reads the per-mode straight-line block
and stops at the first accumulation, and `sum5 += ...` (:489) precedes the
conversion (:490), so the per-mode `u6` and `u3` we take are pre-conversion.
`stress_is_deviatoric = True` on `SolKz` is therefore correct, but it describes
the transcription's cut point rather than the kernel's output — and anyone
transcribing afresh from the *returned array* must not set it.

Getting that wrong leaves the momentum residual at order $|\mathbf f|$ *and*
manufactures a horizontal body force in a benchmark that has none — a large,
structured error that reads like a transcription failure rather than a
convention one. Measured on the transcribed fields: as the deviator (ours)
1.7e-16, as the total 6.0e-1.

### A published stress that is silently right at the default

`solA.c:156` computes the $zz$ stress as `u3 = 2.0*kn*ss_z - pp`. The matching
line in `solB.c:140` is `2.0*Z*kn*ss_z - pp`, and solA's own $xx$ stress two
lines later carries the `Z`. The published $\sigma_{zz}$ is short by
$\tau_{zz}(1-Z)/Z$.

That is **identically zero at $Z = 1$** — the only case the file's own disabled
driver exercised, and the default `eta` in our transcription. Every gate in the
conformance file passed. At `eta=3`, three of them fail: momentum 2.8e-1,
deviator trace 6.7e-1, strain-rate consistency 6.7e-1, where
$|1-3|/3 = 0.667$ exactly.

The transcription restores the factor, declared per solution as
`_zz_stress_lost_the_viscosity`; the vendored source stays verbatim.

Note the repair that was **rejected**. Tracelessness gives
$\sigma_{zz} = -\sigma_{xx} - 2p$ straight from the correct $xx$ component, and
it works — but it would make the deviator traceless *by construction* and so
retire one of the three gates that caught the defect. Restoring the missing
factor instead keeps tracelessness an independent statement about the repair.
`test_1028` asserts both halves: that solA's published deviator is not traceless
at $Z=3$, and that solB's is.

**The general lesson, and the reason `test_1028_analytic_parameter_sweep.py`
exists**: the conformance sweep builds every solution from a mesh alone. That is
right — the defaults are part of the interface — but a coefficient that is unity
by default multiplies a term nothing ever looks at. Any new solution needs an
entry in that file's `SWEEP` table, and the file asserts that every registered
Stokes solution has one.

## Two test tiers, and how to run the slow one

The residual gates are not cheap. Every one of them differentiates the
solution's expressions symbolically and runs common-subexpression elimination
over the result, and five solutions produce expressions with tens of thousands
of operations — SolC accumulates over forty modes, SolDA and SolH over several
more, and SolKx and SolKz carry an exponential viscosity.

Measured: those five cost **565s of the suite's 1010s**; the other eight
together cost about 17s. CI was already within five minutes of its 60-minute cap
before this suite existed, so they cannot ride on every PR.

| tier | what it covers | cost | where |
|---|---|---|---|
| per-PR | every gate, on every solution that is cheap to validate; one canonical parameter case for SolKx/SolKz | **4m07s** | `tests/test_101[5-9]_analytic_*`, `tests/test_102[0-9]_analytic_*` — matched by the CI batch globs |
| full family | every gate, on every solution, over the whole parameter table | **8m34s** | `tests/analytic_full/` — matched by nothing in CI |

```bash
# the full family — before a release, and after touching
# underworld3/analytic/ or _validation.py
pixi run -e amr-dev python -m pytest tests/analytic_full/ -v
```

**The split is by solutions per run, never by checks per solution.** The
momentum residual, incompressibility, tracelessness, strain-rate consistency and
the body-force negative control all run in both tiers, on every solution that
tier covers. Those are the gates that caught the four errata above, and none of
them is weakened by the split.

Solutions declare their own side of it — `expensive_to_validate` on the class —
so the two tiers partition the family from one source of truth. Two guards keep
that honest: `test_1024` asserts that every solution it skips is *named* in the
full-family file, and the full-family file asserts its own hand-written list
matches the declarations.

`tests/analytic_full/` is a **subdirectory** rather than a marked file because
`scripts/test.sh` batches by file glob (`tests/test_101*py`, ...) and not by
marker — a `slow` marker alone would need every batch line to remember to
deselect it, whereas the globs do not recurse. The `level_2` mark on the file
additionally keeps it out of `pytest -m "level_1 and tier_a" tests/`, which does.

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

## A solution with a fault in it

`FaultedMedium` (Barr & Houseman 1996, `analytic/barr_houseman_96.py`) is the linear
plane-strain case of their Appendix: a fault that terminates *inside* a viscous
medium, on a disc of radius $R_0$ with the tip at the centre and the fault
running out to the perimeter. It carries zero shear traction on the fault,
continuous normal velocity and continuous normal stress across it, and slip
$2U_0\sqrt{r/R_0}$.

The year remains in the module name to distinguish this closed-form 1996
solution from the related 1992 study, which does not provide the corresponding
full Cartesian analytical field.

It is here because it is the only absolute standard in the suite for a fault
calculation. Everything else a fault model can be measured against is another
discretisation.

The structure is the interesting part. In polar coordinates about the tip the
stream function separates into a Fourier series in $m = q/2$: whole-integer $m$
is continuous deformation, half-integer $m$ is the fault discontinuity, and
boundedness of the velocity at $r = 0$ admits only one negative index,
$m = -1/2$. That single mode carries the entire singularity, which is *why* slip
goes as $\sqrt r$ and stress as $1/\sqrt r$ — the exponents are a property of the
fault's own Fourier mode rather than an assumption. Drop the half-integer term
and the slip vanishes; there is a test that does exactly that.

### Three ways it does not fit the contract, and what was done about each

**It is not a function of position.** The field is multivalued about the tip —
that is what a fault is — so it is a function of $(r,\theta)$ with
$\theta\in[0,2\pi)$, and the branch cut has to lie *on the fault*, which is where
the field is genuinely discontinuous. A bare `atan2` puts the cut on the negative
$x$ axis and silently returns the wrong face.

The class therefore carries **two representations of one solution**: the polar
fault-frame expressions, in which the fault conditions can be stated exactly at
$\theta = 0$ and $2\pi$, and the contract's `fn_*` fields on `mesh.X`, which is
what the residual gates and a solver consume. A test pins the second to the
first.

Getting $\theta$ into the Cartesian form is where the care went. From
$\tan(\theta/2) = (r-x)/y$, the cut of $2\,\mathrm{atan2}(r-x,\,y)$ falls on the
positive $x$ axis — the fault — which is right. But near the fault $r - x$ is a
difference of two nearly equal numbers, and the relative accuracy of $\theta$
degrades as $1/\theta^2$. Since $(r-x)(r+x) = y^2$, scaling both arguments by the
positive quantity $(r+x)$ leaves the angle alone and removes the cancellation:
$2\,\mathrm{atan2}(y^2,\,y(r+x))$, used where $x > 0$. Measured, at $10^{-6}$
radians off the fault, the momentum residual is **3e-6** formed directly and
**3e-15** formed this way — so `sample_points` can put a point just off each face
and the gates still mean something there.

**It has no walls.** `apply_boundary_conditions` **refuses**, and that refusal is
deliberate. The fault is an internal boundary whose two faces must be separate
degrees of freedom at coincident coordinates — a property of the mesh, not of the
solver, and one Underworld cannot yet build for a fault that reaches the domain
boundary (#549). Its conditions are also per-component (fault-normal velocity
prescribed on both faces, tangential traction natural), so none of the
whole-velocity helpers fits. Applying the perimeter datum and quietly
leaving the fault unconstrained would solve a different problem and report a
plausible error, which is worse than refusing. The pieces a model needs —
`boundary_velocity()`, `fault_normal_velocity()`, `slip()` — are exposed instead.

**It has no body force.** Like `EllipticalInclusion` it is driven entirely by its
boundary, so the family's body-force negative control cannot fire and it is
excluded from that control by name in both sweeps. It is not left without one:
for this solution the momentum residual certifies the **pressure sign** instead,
and just as sharply — 3.6e-16 with UW3's compression-positive pressure, **1.06**
with the paper's extension-positive one, while $\mathrm{tr}\,\sigma + d\,p$ sits
at 5.7e-16 either way because `set_fields` cancels that term by construction.

### The erratum

Barr & Houseman's pressure is extension-positive; ours is not, so `fn_pressure`
is $-1$ times their (A9c). Their printed (A9b) also disagrees with their own
(A8b) in the sign of the half-integer sine terms, and carries $\cos(3\theta/2)$
where $\sin(3\theta/2)$ belongs. All three are recorded, with the measurements
that settle them, in the conventions-and-errata note (§3.8) rather than applied
silently.

### Credit

The implementation follows @gthyagi's, and he verified it independently three
ways in PR #550: against both BH papers (which is how the A9b function error was
found), against a UW3 Stokes model on a Gmsh slit disc (velocity error 1.92% →
0.74% → 0.40% under refinement, normal-velocity jump at machine zero, and the
pressure sign confirmed numerically at ~199% error with the paper's own sign),
and against his BH92 rectangular-fault-zone model, whose near-tip exponent fits
**-0.498** against the exact **-0.500** on the resolved singular interval.

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
| `TwoLayerDarcy` | $\nabla\cdot[k(\nabla p + S\hat z)] = 0$ across a permeability jump | `test_1004_DarcyCartesian.py` |
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

Their boundary condition prescribes the field itself, not a velocity, so it uses
`prescribed_scalar`. `_Transport` applies `fn_solution` on every wall, and there
is no pressure nullspace to remove.

### Transient solutions are singular at t = 0, and that changes how you use them

Every diffusive similarity solution here — `ErfcDiffusion`, `AdvectedFront`,
`GardnerTransient` — is a step with unbounded gradient at $t = 0$. **That state
cannot be represented on any mesh.** Two consequences, and both are enforced in
code rather than left to the reader:

**You cannot start at $t = 0$.** `sol.at(t)` refuses $t \le 0$ with an
explanation instead of returning the singular profile. Returning it would only
ever produce a benchmark measuring its own projection error.

**A transient benchmark is a pair of times, never one.** You initialise from the
solution at $t_0$ and compare at $t_1$, and the error depends on *both*. An
error quoted at a single time is not interpretable — a small $t_0$ means
starting from a profile the mesh cannot hold, and the error you then attribute
to the timestepper is mostly initial projection error.

**And $t_0$ has a floor set by the mesh, not by the solution.** The front has
width $\sim 2\sqrt{Dt}$; asking it to span $n$ elements of size $h$ gives

$$t_0 \ge \frac{1}{D}\left(\frac{n h}{2}\right)^2$$

which is `sol.earliest_resolvable_time(h, elements_across=4)`. It falls as
$h^2$, so refining buys an earlier start quickly.

```python
t0 = sol.earliest_resolvable_time(mesh.get_min_radius())
field.array = uw.function.evaluate(sol.at(t0), field.coords)
...                                        # step to t1
error = sol.error("solution", field)       # against sol.at(t1)
```

#### What this diagnoses

`tests/test_1100_AdvDiffCartesian.py` has carried an `xfail` describing itself
as "not a great test", with a note saying it needs "an error-function IC
starting at $t > 0$ with a meaningful transport distance". The floor says
exactly how badly, at its own parameters ($res=24$, $\kappa=1$, $u=1/24$,
$t_0=10^{-4}$, $t_1=2\times10^{-4}$):

| | |
|---|---|
| earliest resolvable $t_0$ | $3.5\times10^{-3}$ — the test starts **35× too early** |
| front width at its $t_0$ | 0.68 elements, i.e. narrower than one cell |
| transport over the whole run | $4.2\times10^{-6}$ = **0.0001 elements** |

So it initialises a profile the mesh cannot represent and then advects it by a
ten-thousandth of a cell. It measures neither advection nor diffusion, which is
why it has always been sensitive to which evaluation path `uw.function.evaluate`
happens to take. Reworking it needs a timestep convergence study, not just new
constants — a resolution-consistent setup at $res=24$ still shows 11% error in
five steps, dominated by time discretisation. That is left as follow-up work.

### The four tests now use them

The inline copies are gone; each test keeps its assertions and tolerances and
only swaps the expression for the class. Two details worth knowing:

`TwoLayerDarcy` was generalised to take the column extent and a gravity term
$S$, because the Darcy test is posed on $y \in (-1, 0)$ and runs the case twice,
with and without gravity. The profile is now **derived** from constant flux
rather than transcribed from the test's closed form — so agreement is a check on
both, not a copy of one. It matches to 1.1e-16 in both cases.

Its arguments are `k_lower` / `k_upper` rather than `k1` / `k2`, because
`test_1004`'s `k1` is the *upper* layer and the opposite reading is silent: it
produces a perfectly smooth wrong answer.

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
| Barr & Houseman 1996 (GJI 125, 473-490), Appendix | Nothing vendored — transcribed from the published equations | n/a |
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

`assess` is also a dev dependency in `pixi.toml`, so the tests actually run
rather than skipping themselves into a green tick.

### It is an oracle, not a member of the family

`assess` gives numeric callables, so there is nothing to differentiate — a
Kramer solution can be compared against a solver but **cannot be checked against
the equations it claims to solve**. None of the six gates reaches it.

What can still be asked is asked by finite differences, which is the same idea
as Gate 4 with a weaker instrument. All four cases give
$|\nabla\cdot\mathbf u|/|\mathbf u| \approx 10^{-9}$ — the difference floor at
$h=10^{-6}$, so machine-level. Free slip gives $\mathbf u\cdot\hat n \sim
10^{-17}$ on both arcs while $|\mathbf u| \sim 10^{-2}$ there, and zero slip
gives $|\mathbf u| \sim 10^{-17}$ on the walls with $10^{-5}$ inside. Each of
those carries its own control: the free-slip wall is demonstrably *slipping*, so
$\mathbf u\cdot\hat n = 0$ is not passing because everything is zero, and the
divergence probe is checked against $\mathbf u = (x, y)$ to confirm it reports 2
rather than reporting 0 for everything.

### Testing the absent path once the package is present

Installing `assess` silently removed the coverage that mattered most: the
missing-dependency path is what a normal install takes, and it was the thing the
previous arrangement got wrong. Skipping it whenever the package is present
means it never runs in CI.

So absence is **simulated** — `builtins.__import__` and `importlib.util.find_spec`
are monkeypatched — and the fixture has its own negative control asserting that
the simulation actually blocks. Without that, a Python change to import
resolution would let every one of those tests pass by importing the real
package while appearing to cover a path they never touch.

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

### The format to supply a new solution in

If you are deriving a solution rather than porting one, supply it in the form
below and it drops in with no translation step. This is the format to ask
contributors for.

**Fields as SymPy expressions in the mesh coordinates**, not as callables,
lambdas, or NumPy code. `mesh.X` gives the coordinate symbols; build everything
from those. Expressions route through the JIT, so an analytic viscosity or body
force compiles to C *and* supplies its own Jacobian — neither of which a numeric
callable can do. `CylindricalStokes` is the one exception in the suite and it
pays for it: none of the six gates can reach it.

**Do not simplify.** Preserve whatever grouping the derivation produced. The
Maple grouping in the Velic kernels is what keeps `sinh(k)*exp(-k)` products
stable at large wavenumber; a re-derived, "tidier" grouping can lose eight
digits. `cse=True` at evaluation time recovers the sharing anyway.

**State the stress convention explicitly** if the solution has one — deviatoric
or total. It is a declaration (`stress_is_deviatoric`), never inferred, because
the family is not consistent and one kernel writes its deviator into an array
named `total_stress`. Getting it wrong leaves the momentum residual at order
$|\mathbf f|$ and looks like a transcription failure rather than a convention
one.

**Give the equation the solution solves**, not just the answer. The oracle-free
residual is the strongest check in the suite — it caught four defects that
comparison against the source could not, including a published stress that is
simply wrong. It needs to know what to substitute back into.

**Say whether it is singular anywhere**, in time or in space. A $t = 0$
similarity singularity needs `singular_at_origin = True` and a `diffusivity`;
a spatial singularity (the inclusion's foci, SolCx's isoviscous limit) needs a
`sample_points` override so the validation harness does not land on it.

**Say what parameter ranges it is valid over.** SolKx is only free-slip for
integer wavenumbers; the Gardner solutions require $\psi < 0$. Ranges become
constructor validation, which is where they stop being folklore.

1. Subclass `AnalyticSolution` — or `_Transport` for a scalar solution — and
   write `apply_boundary_conditions`, composing the helpers below.
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
