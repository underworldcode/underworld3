# A yield-stress homotopy for hard-`Min` viscoplastic Stokes

**Status.** Method + results note. The homotopy and the consistent-tangent
infrastructure are implemented and merged-ready on `bugfix/yield-homotopy`
(source: `constitutive_models.py`, `petsc_generic_snes_solvers.pyx`; tests:
`tests/test_1053_yield_homotopy.py`, all green). All numbers below are from the
Spiegelman et al. (2016) notch benchmark at full Drucker–Prager, reproduced by the
harness in `~/+Simulations/spiegelman_hardcase/` (driver `drivers/convergence.py`,
live index `README.md`). Convergence figures and yield-law curves are in that
directory's `figures/`. Companion planning log:
`yield-homotopy-convergence-study.md`.

## TL;DR

Hard-`Min` Drucker–Prager (DP) viscoplastic Stokes is hard to *solve*, not because
the answer is exotic but because the constitutive law has a **non-differentiable
yield corner** and the consistent Newton tangent has a **small basin of
attraction** (Spiegelman, May & Wilson 2016). Instead of changing the physics, we
**sequence a single regularisation parameter to zero**:

1. Replace the hard `η_eff = Min(η_ve, η_pl)` with a one-parameter **soft-min**
   `η_eff = η_ve / g(f, δ)` that rounds the corner by an amount δ and is
   **identically `Min` at δ = 0** (so the converged answer is the exact yield
   surface — you can step onto the discontinuous law at the end).
2. **March δ from a generous value to 0**, fully solving at each δ and
   warm-starting the next, sharper problem. The smooth (δ > 0) problem is
   easy to converge from anywhere and walks the iterate into the basin of the
   exact-`Min` solution.

This is the **numerical mirror of the physical strategy** of sequencing a yield
strength downward — but with a *single, regime-independent* parameter that makes
the sequencing *smart* rather than an arbitrary change of physics. It **opens the
hard-`Min` problem to iterative (geometric multigrid, FMG) solvers** across the
band of regimes where the difficulty is the yield corner, and reaches the **exact**
hard-`Min` solution there.

For the most extreme regimes (very strong layers / fast compression), the
difficulty is no longer the corner but **extreme viscosity contrast** in a
deep-yielded material, which defeats the corner-homotopy. A **second axis** — a
small strain-rate dependence of the yield stress (a Perzyna overstress, physically
a damage/viscoplastic regularisation) — rescues those cases **under FMG**, where
the augmented-Lagrangian pressure penalty does not.

The warm-started progression solves DP problems well past the regime where cold
Newton fails — and in doing so it doubles as a **probe of the problem's
conditioning**: the sharp-residual diagnostic reveals a **flat-residual /
near-singular direction** (the shear-band-width mode the equations barely
constrain), which is *why* the cold problem is hard and why the homotopy's
incremental warm-starting is the right tool for it.

## The problem this solves

The DP effective viscosity is `η_eff = Min(η_ve, η_pl)` with the plastic branch
`η_pl = τ_y / (2 ε̇_II)` and yield stress `τ_y = C + sin φ · P`. Two distinct
difficulties make this hard for a Newton–Krylov solver:

- **The corner.** `Min(a, b)` is non-differentiable at `a = b` (the yield onset
  `f ≡ η_ve/η_pl = 1`). A cold Newton step lands on the kink and the line search
  fails (`DIVERGED_LINE_SEARCH` / stall). This is exactly Spiegelman et al.'s
  finding that consistent Newton has a *small basin of attraction* from the
  viscous guess.
- **The non-symmetry.** Because `τ_y` depends on the *solution pressure* `P`, the
  consistent tangent carries a `∂η_p/∂P` term — the **deviatoric** stress depends
  on pressure, with no mirror in the continuity equation. The Newton operator is
  **non-symmetric** (verified: `∂F1/∂P` has 4 non-zero entries for DP vs 2 for von
  Mises). This is why Fig. 5 of Spiegelman et al. shows von Mises converging where
  DP fails.

The homotopy attacks the *corner*; the non-symmetry is handled by a robust linear
solver (a Schur fieldsplit whose velocity block is FMG — the scalable deliverable —
or, as a reference-only diagnostic, an exact MUMPS solve).

## The method

### The δ-soft-min law

`ViscousFlowModel._combine_yield` implements one δ-parameterised soft-min, shared by
all viscoplastic / VEP subclasses:

$$\eta_\mathrm{eff} = \frac{\eta_\mathrm{ve}}{g(f,\delta)}, \qquad
g = 1 + \tfrac{1}{2}\!\left(f - 1 + \sqrt{(f-1)^2 + \delta^2}\right) - \mathrm{offset},
\qquad f = \frac{\eta_\mathrm{ve}}{\eta_\mathrm{pl}},$$

with the onset offset `(-1+√(1+δ²))/2` pinning `g(0)=1`. At **δ = 0**, `g = max(1,f)`
**exactly**, so this is *identically* `Min(η_ve, η_pl)` — validated to machine
precision (`test_1053`). δ and the offset are held as `constants[]` UWexpression
atoms, so δ can be **ramped at runtime via `PetscDSSetConstants` with no JIT
recompile**.

δ has a **regime-independent** meaning: in the normalised stress `τ/τ_y = f/g`, a
given δ is a fixed *percentage deviation* from the yield value, independent of the
actual η or τ_y — so one schedule transfers across problems. The sqrt soft-min
**overshoots** the yield surface in the transition (τ/τ_y up to ≈ 1.6 at δ = 64,
collapsing onto the exact corner as δ → 0; `figures/homotopy_yield_curves.png`).

### Why the sqrt soft-min is the homotrope (not the power-mean)

We evaluated a second family, the **power-mean** p-norm soft-min
`η_eff = (η_ve^{-s} + η_pl^{-s})^{-1/s}`, `s = 1/δ`, which *undershoots* the yield
surface (`τ/τ_y ≤ 1` always — physically attractive, no over-yield). Both families
reach `Min` in the limit, and they **bracket the exact corner from opposite sides**
(`figures/blend_yield_curves.png`). But as a *homotopy from cold* they are not
equivalent:

- **sqrt (from above): a safe homotrope.** Its smoothing *strengthens* the material
  in the transition, so a generous-δ start is a stiffer-but-similar problem in the
  *same basin* as the `Min` solution. Cold Newton converges there, and the march
  stays in the basin all the way to δ = 0.
- **power-mean (from below): not a cold homotrope.** Its smoothing *weakens* the
  load-bearing layer. A smooth start collapses into a **spurious degenerate basin**
  (η → 0, fully-yielded self-consistent fixed point), and a sharp start hits the
  kink — there is no safe cold entry.

So **sqrt is the chosen homotrope**: it mirrors the physical sequenced
yield-stress reduction, has a single regime-independent parameter, and steps onto
the *exact* discontinuous `Min` at the end. The power-mean is retained as the
instructive from-below contrast.

### The sequencer

`run_continuation` performs the march: fully converge at δ₀ (= 64), warm-start the
halved δ, repeat. Each step is a `constants[]` update only (no JIT/solver rebuild).
A tight per-step iteration budget makes a step that is *too hard* abort cheaply and
**back off to the smoothest feasible δ**, reporting it — so the scheme self-paces
and never grinds. Because plasticity **pins the stress** at `τ_y`, the converged
*solution* is δ-insensitive (verified: the smooth-δ solution satisfies the exact
δ = 0 residual to the same `‖F‖`, ×1.0) — the march's role is to make the *solver
path* tractable, not to change the destination.

## Results

### Phase 1 — what the sequencer solves, and what is beyond the corner

Regime map over the Spiegelman Fig. 5 plane, ordered by driving stress
`η_bg · V`, comparing **cold Newton at δ = 0**, a **single smooth cold solve at
δ = 0.5**, and the **δ-sequencer** (size-3 notch, φ = 30, FMG;
`logs/regime_summary.txt`):

| `η_bg·V` | `(η_bg, V)` | verdict | mechanism |
|----------|-------------|---------|-----------|
| ~2.5e24 | (1e24, 2.5) | **cold-OK** | sub-yield — no plasticity active |
| ~1e26 | (1e25, 10) | **smooth-cold-OK** | cold δ=0 (kink) fails; **one** smooth δ=0.5 solve = hard-`Min` (×1.0) |
| ~2.5e26 | (1e25, 25) | **sequencer-needed** | single solve below cold-basin threshold; the δ-march reaches hard-`Min` |
| ~3e26 | (3e25, 10) | **sequencer-needed** | the genuine homotopy win |
| ≳1e27 | (1e26, 10/25) | **beyond the corner-homotopy** | see Phase 2 |

The headline: across the **mid-band** (`η_bg·V ~ 2–3e26`) — the geodynamically
interesting hard cases — **cold Newton dies at the kink, a single smooth solve is
below its basin threshold, but the sequenced march reaches exact hard-`Min`**
(44 % / 34 % of the domain yielded; δ = 0 residual unchanged ×1.0). At the onset
band a single smooth-δ cold solve already suffices; the sequencer's value is that
it **removes the need to guess a workable δ** — it always starts safe and marches.

### Why the extreme corner is hard — and it is *not* the kink

At `(1e26, ·)` the non-dimensional `τ_y` is tiny, so `f = stress/τ_y ~ 1e4`
**everywhere** — the material is deep in the yielded branch, nowhere near the
corner. Diagnosis (single δ = 64 solve, exact MUMPS velocity block):

| | result |
|---|---|
| viscous pre-solve (no plasticity, same contrast) | `reason = 2`, `‖F‖ = 2.5e-13` (clean) |
| **consistent Newton** tangent | **`DIVERGED_LINEAR_SOLVE`** at iter 2 (even `KSP_MAXIT = 1000`) |
| **Picard** (frozen viscosity) | no linear-solve divergence — converging, but slow |

So three things compound, none of them the corner: (1) the corner-smoothing δ
**does not engage** (nothing operates near `f = 1`); (2) the ~4-order viscosity
contrast makes the Schur ill-conditioned, and the **consistent tangent's
non-symmetric `A` breaks the linear solve** (Picard's symmetric `A` does not); (3)
the augmented-Lagrangian pressure penalty **fixes the linear divergence for the
direct solver** but is **hostile to FMG** — the γ·grad-div augmentation wrecks the
MG smoother (FMG + AL ran 47 min on one δ = 64 step without converging).

### Phase 2 — a second homotopy axis: rate-strengthening yield (damage)

The δ axis regularises the **corner**; the extreme corner needs the **deep-yielded
branch** regularised. A small **strain-rate dependence of the yield stress** —
`τ_y → σ_y + ξ · η_reg · 2 ε̇_II`, a Perzyna overstress, so
`η_pl = σ_y/(2 ε̇_II) + ξ · η_reg` gains a regularising viscosity floor — does
exactly that. It is rate-*strengthening* (`∂τ_y/∂ε̇ > 0`): it bounds the viscosity
contrast (conditions the Schur), adds a positive-definite tangent contribution
(coercive), and **stays MG-friendly** — unlike AL. `ξ` is a `constants[]` atom;
`ξ → 0` recovers rate-independent DP.

**Result (sqrt δ-sequencer, FMG, at the `(1e26, 10)` corner that was
`TOO-HARD`):**

| ξ | sequencer | vrms | yielding |
|---|-----------|------|----------|
| **0** (control) | **FAIL** (δ=64 entry `DIVERGED_LINEAR_SOLVE`) | — | — |
| **0.01** | **converges** → δ=0.002, hard-`Min` ×1.0 | 0.747 | 87 % |
| **0.03** | **converges** (8 it) | 0.712 | 89 % |
| **0.1** | **converges** (6 it) | 0.682 | 79 % |

Two findings. **(i) Solver:** a *1 %* regularising floor (ξ = 0.01) **rescues the
FMG solve** where the δ-homotopy failed and AL + FMG could not converge — it is the
**MG-friendly fix** for the extreme-contrast corner. **(ii) Pattern selection:**
unlike the δ axis (where the solution is δ-*insensitive*, stress-pinned), the
**solution genuinely changes with ξ**. The converged field is the classic DP
localisation — conjugate shear bands (an X) from the notch plus a horizontal band
along the weak layer (`figures/strainrate_*_xi*.png`) — and its *sharpness* tracks
ξ: smaller ξ → wider strain-rate range → more concentrated bands. So the rate
dependence performs real *physical selection* of the localisation pattern, not just
conditioning.

#### ξ → 0 continuation (does the selected pattern converge?)

Starting cold at ξ = 0.1 and **warm-stepping ξ downward** toward the
rate-independent limit (each step warm-starts the δ-sequencer from the previous ξ;
common-colour band render at each ξ — `figures/ximarch_bands_*.png`,
`figures/ximarch_vrms_*.png`):

| ξ | 0.1 | 0.05 | 0.025 | 0.0125 | 0.006 | 0.003 | 0.0015 |
|---|-----|------|-------|--------|-------|-------|--------|
| vrms | 0.682 | 0.698 | 0.718 | 0.740 | 0.762 | 0.776 | floor (grinds) |
| `‖F(δ=0,ξ=0)‖` | 5.0e-8 | 3.4e-8 | 1.2e-8 | 2.7e-8 | 1.8e-8 | 2.4e-8 | — |

The two-axis homotopy (δ *and* ξ both small) **walks the (1e26, V10) corner — which
was `TOO-HARD` for the δ-homotopy + FMG — down to ξ = 0.003 under FMG**, i.e. to
within a 0.3 % regularising floor of rate-independent DP. Below that (ξ = 0.0015)
the unregularised extreme-contrast problem returns and the iterative linear solve
grinds — the practical ξ-floor.

The decisive diagnostic is the **true sharp-corner residual** `‖F(x_ξ, δ=0, ξ=0)‖`
— the residual of the *exact* unsmoothed, rate-independent DP problem at each
regularised solution. It sits at the **solver tolerance floor (1–5e-8) for every
ξ, with no trend**, while vrms changes **+14 %**. So *every* ξ-variant genuinely
satisfies the exact yield-surface problem to tolerance — they are **equivalent in
residual** — yet localise differently. The residual is **blind to the band-width
mode**: tightening a thin shear band barely changes the far-field flow or the
integrated residual, so the equations constrain that mode only weakly — a
**near-singular direction** along which ξ selects.

As ξ → 0 the shear bands **tighten** monotonically (`figures/ximarch_bands_*.png`),
toward the singular rate-independent limit (infinitely-thin slip lines, mesh-limited).
The variants share the same conjugate-X band *geometry* and tighten toward the limit
rather than switching to a rival pattern — so at this regime it is a graded
one-parameter family, not discretely-distinct slip-line fields.

**What ξ is — and is not.** ξ is a *rate* (Perzyna / viscoplastic) regulariser, and
its dual role must be kept separate:
- **As a solver conditioner it is genuinely useful** — it bounds the contrast,
  keeps the velocity block MG-friendly, and is what rescues the FMG solve at the
  extreme corner (the headline above).
- **As a physical regulariser it is the *wrong* tool.** It sets a band width only
  *indirectly* (emergent from the rate-strengthening vs. the local dynamics), and
  that width is **mesh-sensitive** — there is **no directly-settable physical length
  scale**. A genuine shear-band-limiting length (gradient / nonlocal plasticity) is
  the physically-correct regulariser; ξ does not replace it.

The physically-correct, *settable*-length regularisation is an **implicit-gradient
(nonlocal) term solved *inside* the SNES**: a Helmholtz-filtered strain rate
`ε̇̄ = (1 − ℓ²∇²)⁻¹ ε̇` used in the yield, with ℓ the band-width length, coupled as
an auxiliary field in the residual/Jacobian. An **outside-SNES** strain-rate
smoothing / block-ε̇ term (an operator split: ε̇̄ built from the previous iterate)
is **weak under strong viscosity contrast** — the regularisation lags the
localisation, which moves between iterates, so it decouples exactly where it is
needed (observed). The clean division of labour is **δ/ξ for the solver path
(homotope → 0), ℓ for the physics (kept finite, in-SNES)** — orthogonal roles, the
same rampable-`constants[]` machinery applies to all three.

## Honest limitations

- **The δ-homotopy addresses the corner only.** Where the difficulty is extreme
  contrast in deep yield (very strong layer / fast compression, `η_bg·V ≳ 1e27`),
  the corner-smoothing does not engage and the consistent tangent's non-symmetric
  operator breaks the iterative linear solve. The rate-strengthening axis (Phase 2)
  is required there; whether those regimes are physically interesting (the
  non-dimensional yield stress is ~1e-4 of the driving stress) is a separate
  question.
- **AL does not transfer to FMG.** The augmented-Lagrangian penalty that conditions
  the Schur for a direct velocity solve degrades the MG smoother. For the scalable
  FMG deliverable, the rate-strengthening regularisation (which keeps the velocity
  block MG-friendly) is the better Schur-conditioning route at extreme contrast.
- **The consistent tangent has two fragilities, not one.** Beyond Spiegelman's
  small *nonlinear* basin, it is *linear-solve* fragile at extreme contrast
  (non-symmetric `A` → divergent Schur Krylov). Picard is robust but slow; the
  homotopy should pair with Picard (or a Picard→Newton continuation) there.
- **Rate-independent DP is non-unique** in general (perfect plasticity admits
  multiple shear-band patterns). The ξ-regularisation *selects* one. At the regime
  tested here the `ξ → 0` variants are equivalent in residual (all at tolerance) and
  form a graded one-parameter family (tightening bands, same geometry) rather than
  rival patterns; whether different regimes / ξ-paths select genuinely distinct
  limits is open.
- **Band-width vs mesh resolution.** As ξ → 0 the bands tighten toward the element
  scale (the rate-independent localisation limit is mesh-limited). At the tightest ξ
  reached (0.003) part of the 14 % vrms swing may be the bands becoming
  *under-resolved* rather than purely physical selection. A ξ-march on a finer notch
  mesh is needed to separate physical band-width selection from mesh-limited
  tightening.

## Next steps

1. **ξ → 0 continuation across regimes.** At (1e26, V10) it converges to a single
   pattern down to ξ = 0.003 (done). Repeat across the Fig. 5 plane and test whether
   different ξ-paths (or δ/ξ orderings) select different limits — the direct probe
   of non-uniqueness.
2. **Pattern-selection study** as a deliberate physics investigation (damage-rate
   selection of localisation), framed as *uniqueness/selection*, not convergence.
   Include a **mesh-resolution ξ-march** to separate the physical band-width scale
   from mesh-limited tightening (the bands tighten toward the element size as ξ→0).
3. **Two-axis interaction** (δ for the corner, ξ for the branch) — when is each
   needed, and does a joint schedule cover the whole Fig. 5 plane.
4. **Pair the homotopy with Picard** (or Picard→Newton continuation) at extreme
   contrast, where the consistent tangent's linear-solve fragility dominates.
5. **Lagged / Picard gradient plasticity *inside* the homotopy harness.** The
   physically-correct nonlocal length-scale term (implicit-gradient
   `ε̇̄ = (1−ℓ²∇²)⁻¹ε̇`) is coupling that currently drops *outside* the MG loop;
   doing it MG-robustly needs SNES-FAS (a separate effort). But a **lagged/Picard**
   version may be usable *now* by hosting it in this harness: a lagged nonlocal
   coupling is binding only when the solution changes a lot between iterates (the
   strong-contrast regime where it goes weak), whereas each homotopy step is a small
   increment warm-started from the previous *lag-consistent* solution — so the
   carried-in ε̇̄ is already nearly correct and the fixed point has little to fix.
   The homotopy **amortizes the lag over small steps**; the ε̇̄ refresh rides as
   another `SNESSetUpdate`/`constants[]` callback, with ℓ a genuine physical
   band-width length. Not a substitute for SNES-FAS, but a credible way to make the
   lagged scheme work in the hard regimes without new solver machinery.
6. **Rate-WEAKENING (m) axis — parked, exploratory.** The mirror of ξ: a smooth δ
   for solvability + rate-weakening `τ_y → σ_y·(1 − m·ε̇/(ε̇+ε̇_ref))` to *restore*
   the fine bands δ washes out (L.M.). Observations so far: weakening is genuinely
   **anti-coercive** (cold-solving with it on grinds — it fights the solver, the
   opposite of ξ), so it needs a *warm* m-homotopy from m=0. **Flagged JIT bug**: a
   `constants[]` atom in an *exponent* did not ramp (residual byte-identical across
   m); it must be a linear/multiplicative coefficient to be repackable — a constant
   should be usable in exponent position, so this is a JIT/constants-handling bug,
   not a modelling constraint. Open question (the actual physics): does a δ↔m
   *balance* set a finite, mesh-independent band width, or do the bands go
   mesh-limited as with pure rate-weakening?

## References

- Spiegelman, M., May, D. A. & Wilson, C. R. (2016). On the solvability of
  incompressible Stokes with viscoplastic rheologies in geodynamics.
  *Geochem. Geophys. Geosyst.* 17, 2213–2238.
- Perzyna, P. (1966). Fundamental problems in viscoplasticity.
  *Adv. Appl. Mech.* 9, 243–377. (rate-dependent regularisation)
- Duretz, T., de Borst, R. & Le Pourhiet, L. (2019). Finite thickness of shear
  bands in frictional viscoplasticity. *J. Geophys. Res. Solid Earth*. (rate /
  viscosity regularisation of localisation)
- Fraters, M., Bangerth, W., Thieulot, C., et al. (ASPECT) — defect-correction
  Picard and Newton for viscoplastic Stokes.
