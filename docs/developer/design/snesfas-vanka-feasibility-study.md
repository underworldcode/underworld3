---
title: "Feasibility study: a scalable saddle-point smoother for Stokes SNESFAS"
status: "Investigation record (preserved via PR #245, 2026-06-18); prototype only — FAS wrap / UW3 API not landed"
---

# Feasibility study — scalable saddle smoother for Stokes FAS

**Bottom line: GO — demonstrated with a working, mesh-independent prototype.
[Revised 2026-06-15 after expert input + experiments — supersedes the earlier
NO-GO.]** Vanka works on UW3's simplex Taylor-Hood Stokes; the earlier failure was
PETSc's *stock* `-pc_patch_construct_type vanka`/`star` heuristic mis-constructing
patches for continuous-P1 simplices, **not** Vanka. With patches built as *the
support of each pressure basis function* (one pressure DOF + its B-coupled velocity
DOFs), driven by **`PCASM` with custom index sets** and exact local LU mini-Stokes
solves, and used as the smoother of a geometric multigrid on the full saddle, the
solver is **mesh-independent: 5 → 5 → 6 outer FGMRES iterations across a 16× growth
in DOFs (1.2k → 19k)**. The one non-obvious ingredient: the smoother must be a
**Krylov (GMRES) smoother** wrapping the Vanka PC — an undamped Richardson smoother
amplifies the additive-Schwarz spectrum and diverges. So the scalable-smoother piece
is **done in prototype**. **Caveat on the payoff:** the iteration count is
mesh-independent, but at moderate **2-D** sizes it is *not* faster in wall-clock than
a sparse direct solve / fieldsplit+FMG (2-D direct solves are cheap); Vanka's
wall-clock advantage is asymptotic in 2-D (~10⁵–10⁶ DOFs) and real in **3-D and
large-parallel** runs, where direct solves scale badly. So this is the tool for
big/3-D/parallel Stokes, not a speed-up for typical 2-D problems. What remains is
productionisation (wrap in FAS — already proven with an LU smoother; per-level
pressure nullspace; parallel patch construction; a UW3 API). Until that lands, FMG +
Newton/Picard/nested ships and LU-smoothed FAS is the moderate-size nonlinear option.

> The detailed reasoning below that concluded NO-GO was **correct about stock
> `PCPATCH` but wrong about Vanka in general**. Read it as "stock PCPATCH
> constructions are the wrong tool", then jump to *Correction* near the end for the
> working approach and the revised path forward.

## Objective

Decide — *before* a major rewrite — whether a production-scalable Stokes FAS is
achievable, by answering whether a cheap saddle-point smoother can be made to work
in UW3.

## Method

Staged, cheapest-fatal-risk-first, with the key move of **decoupling the smoother
from FAS**: test each smoother candidate as a plain *linear* preconditioner on a
constant-viscosity UW3 Stokes solve first (a `pc_type` swap), since FAS itself is
already proven. Option sets were lifted from PETSc's own CI-tested example
`src/snes/tutorials/ex62.c`.

## Findings

### Stage 0 — the known-good Vanka recipe exists (and my first attempt was mis-configured)

`ex62.c`'s `2d_q1_p0_gmg_vanka` test gives the authoritative incantation:
`-pc_type patch -pc_patch_partition_of_unity 0 -pc_patch_construct_codim 0
-pc_patch_construct_type vanka -..._sub_pc_type lu -mg_coarse_pc_type svd`. Note:
**every passing Vanka test in ex62 uses `dm_plex_simplex 0` (quads), Q1–P0
(discontinuous pressure)** — there is no CI-tested Vanka for simplex Taylor-Hood.

### Stage 1 — Vanka does not work on UW3's Stokes (the crux, R2)

PCPATCH Vanka as the outer linear PC on a constant-viscosity UW3 Stokes solve, with
the corrected recipe, fails **uniformly**:

| variant | result |
|---|---|
| P1-continuous, sub LU | **zero pivot** (singular patch) |
| P0/P1-discontinuous, sub LU | **zero pivot** |
| `+ pc_use_amat` (true saddle Jac, not UW3's Schur Pmat) | **zero pivot** |
| sub `svd` (tolerates singular patches), P1-cont | builds, but **no convergence** (400 its, no progress) |
| sub `svd`, P0-disc | builds, **crawls** to 0.00274 vs 0.00319 ref, 427 its, no convergence |
| **quad** mesh (ex62's Vanka element family), all of the above | **identical** failure pattern |

So the failure is **not** element-type (simplex vs quad), pressure continuity, or
sub-solver. The patches are genuinely singular (SVD removes the pivot but leaves an
ineffective preconditioner). The obstruction is in how `PCPATCH` constructs/assembles
the local patch operators (and their boundary conditions / local nullspaces) from
UW3's DM/DS — the same interface Firedrake feeds correctly when it runs DMPlex Stokes
Vanka. UW3 confirmed to build a **separate Schur-structured Pmat** (`1/μ` pressure
mass; `petsc_generic_snes_solvers.pyx:4381`), which is part of why the default path
hands Vanka the wrong operator, but `pc_use_amat` did not rescue it — the problem is
deeper than operator selection.

### Braess–Sarazin hedge — works as a PC, too weak as a smoother

A *correct* Braess–Sarazin (diagonal velocity block `fieldsplit_velocity_pc_type=jacobi`
+ `selfp` Schur on the true operator via `pc_use_amat`) **converges as an outer PC**
(reason 3, exact reference solution) — options-only, no patches, no zero pivot. But
as a **FAS level smoother it fails** (`DIVERGED_INNER`): it needs ~63 iterations to
converge as a PC, i.e. it is a *weak* relaxation, so a few sweeps per smooth don't
reduce error. A stronger variant (SOR velocity + full Schur) converged at ref2 (5
cycles) but was **22× slower than LU and failed at ref3** — unreliable and expensive.

### The smoother-candidate matrix

| smoother | cheap? | effective? | scalable? | status in UW3 |
|---|---|---|---|---|
| monolithic **LU** | no | yes | no | works; fastest at moderate size; O(N^1.5) wall |
| **Vanka** (PCPATCH) | yes | yes | yes | **blocked** — singular/ineffective patches |
| **Braess–Sarazin** (diag) | yes | no (weak) | yes | works as PC, fails as smoother |
| BS + SOR velocity | ~ | unreliable | ? | 22× slower, fails at ref3 |
| heavy fieldsplit-Schur | no | yes | ~ | 9–17× slower than LU |

No candidate is simultaneously cheap, effective, and configurable. The only effective
smoothers are expensive (LU, heavy fieldsplit); the only cheap one (BS) is too weak.

## De-risking step 1 — DONE: simplex Vanka fails in clean PETSc too

The proposed de-risk was to test Vanka *outside* UW3. Rather than re-write a Stokes
solver in petsc4py, the cleanest reference is PETSc's own `ex62.c` (DMPlex Stokes,
the source of the recipe). Compiled it (this build has **no 2-D simplex generator** —
`--download-triangle` absent — so a gmsh unit-square `.msh` was loaded via
`-dm_plex_filename ... -dm_plex_boundary_label marker`; UW3 itself only gets simplices
through gmsh for the same reason). Results, **pure PETSc, no UW3**:

| element / construction | result |
|---|---|
| **quad** Q1–P0, `vanka` (the CI-tested recipe) | ✅ converges, 59 KSP its |
| simplex P2–P1, full LU **and** fieldsplit-Schur (controls) | ✅ converge — mesh/BC are correct |
| simplex P2–P1, `vanka` patches (sub lu / svd) | ❌ KSP **stalls at the initial residual** (29.46 flat for 200 its) |
| simplex P2–P0, `vanka` patches | ❌ diverges |
| simplex P2–P1 / P2–P0, `star` (vertex) patches | ❌ diverges |

So Vanka works on quads and the same binary solves the simplex mesh fine with
standard solvers, but **every patch-smoother construction is ineffective on simplex
Taylor-Hood** — the KSP makes *zero* progress. This is a PETSc/numerical-methods
fact, independent of UW3. (The earlier UW3 zero-pivot was just the first symptom of
the same underlying problem; "fixing UW3's DM/DS↔PCPATCH interface" would **not** have
helped, since clean PETSc fails identically.)

**Why:** stock `PCPATCH` Vanka/star patches are built for low-order, quad/structured,
discontinuous-pressure elements. Effective patch smoothing for simplex Stokes needs
specialist constructions from the literature — typically a patch-smoothable stable
element (e.g. **Scott–Vogelius on barycentrically-refined/Alfeld meshes**) plus
**macro-element patches** (the Farrell–Mitchell–Wechsung programme). That is a change
of *element and mesh*, not a smoother option.

## Correction (2026-06-15): custom-IS PCASM Vanka works — the NO-GO was wrong

External expert input reframed Vanka as *"a specialised overlapping-Schwarz
preconditioner whose patches are defined by the pressure space, not by geometry"*:
loop over pressure DOFs, use the FE sparsity of the divergence block **B** to gather
the coupled velocity DOFs, extract the local mixed matrix, factor it, and run
additive/multiplicative Schwarz — i.e. **`PCASM`/`PCGASM` with custom index sets**,
the patches being the *support of each pressure basis function*. This is standard for
Taylor-Hood, MINI, Scott–Vogelius, variable-viscosity convection, etc. The stock
`PCPATCH` `vanka`/`star` constructs I tested are *not* this — they apply a fixed
topological heuristic that misfires for continuous-P1 simplices.

Tested directly on UW3's true saddle Jacobian (open-top, no pressure nullspace):

```
saddle 1110×1110, fields [velocity, pressure], n_vel=968 n_pres=142
patches = 142 (one per pressure DOF), size 11–45 (mean 32.5)
PCASM-Vanka (RAS and additive): CONVERGED, 312 fgmres its, rel_err 2.2e-6
```

One-level iteration count vs mesh size — the smoother signature:

| cellSize | ndof | n_pres | 1-level Vanka its |
|---|---|---|---|
| 0.15 | 555 | 75 | 147 |
| 0.10 | 1110 | 142 | 312 |
| 0.07 | 2479 | 303 | 688 |

Iterations grow ~with 1/h (no coarse correction) — high-frequency error is removed,
low-frequency is left for the MG coarse grid. That is precisely a multigrid smoother.
So **the failure was the patch *construction*, not Vanka**, and the smoother exists
for UW3's element.

## The working recipe (prototype: `vanka_mg_WORKING.py`)

A geometric multigrid on the **full** Stokes saddle, custom-IS Vanka smoother:

1. **Hierarchy + Galerkin.** PCMG over UW3's `dm_hierarchy` (built by `refinement=N`),
   interpolation per level from `DMCreateInterpolation`, coarse operators by Galerkin
   `pc_mg_galerkin both` (the FMG path — UW3 has no coarse-DM callbacks). Block-diagonal
   velocity/pressure interpolation keeps the coarse operators saddle-structured.
2. **Per-level Vanka smoother.** For each level: from the level's field decomposition
   + the operator's row sparsity, build one patch per pressure DOF = {pressure DOF} ∪
   {B-coupled velocity DOFs}; install a `PCASM` (RESTRICT) with those index sets and
   exact `sub_pc_type lu` mini-Stokes solves. (Set subdomains after a `PC.reset()` +
   re-attach operator, since they must precede `PCSetUp`.)
3. **Krylov smoother — the key.** Wrap the Vanka PC in **`ksp_type gmres`, ~6 its**
   per level. Richardson (even damped) diverges because additive Schwarz has spectral
   radius > 1; GMRES self-stabilises it. Outer solver is FGMRES (flexible, since the
   smoother is now a Krylov method).
4. **Coarse solve:** LU.

Measured: **5 / 5 / 6 / 5 outer iterations at ndof 1.2k / 4.8k / 19k / 76k** —
mesh-independent.

### Timing — and the right competitor is FMG, not LU

A direct (LU) solve is a *strawman* competitor: if a full-scale factorization is
affordable you would just use it directly, not as a smoother — and in 3-D / parallel
that option disappears (the sparse direct solver, MUMPS, scales ~O(N²) in 3-D and is
unreliable at large core counts). The honest competitor is **FMG** (velocity-block
geometric MG inside the Schur fieldsplit, #231).

Linear 2-D Stokes, same problem (FMG numbers include assembly/JIT — not a clean
linear-solve isolation; Vanka is linear-solve only):

| | 1.2k | 4.8k | 19k | 76k | iters |
|---|---|---|---|---|---|
| **FMG** total solve | (JIT) | ~1.7s | ~1.8s | ~4.7s | outer Schur = 1 |
| **Vanka-MG** linear solve | 0.02s | 0.13s | 1.1s | 5.7s | 5–6 |
| LU-smoother MG (≈ direct) | 0.01s | 0.07s | 0.54s | ~5s | 1 |

For **linear 2-D Stokes the three are in the same ballpark** — FMG is mature and
slightly ahead, the LU/direct path is cheap because 2-D fill-in is modest, and
Vanka-MG is competitive but not a winner. So Vanka does **not** earn its keep on easy
(linear, moderate-2-D) problems.

**Where Vanka-MG wins — and the reason to build it:**

- **3-D and large-parallel**, where the direct solves that FMG's coarse grid and the
  LU paths lean on scale badly / become unreliable, while Vanka's local patch solves
  scale ~O(N) and parallelise naturally. (3-D and parallel timing is the obvious next
  measurement.)
- **Strongly nonlinear / plastic rheology** — the decisive case. Fieldsplit-Schur+FMG
  assumes a good Schur (pressure) approximation, which degrades when the viscosity
  varies wildly (viscoplastic yield, thermal runaway). A **full-saddle Vanka-FAS**
  smooths the coupled nonlinear system directly, and — the key point — the **coarse
  problems are better conditioned** (plasticity localises; on coarse grids the yield
  is smeared/milder), so the nonlinear coarse correction is especially effective.
  This is exactly the regime where Newton+FMG needs many iterations or stalls
  (cf. the viscoplastic results in `snesfas-feasibility.md`, where Newton, Picard
  *and* LU-FAS all struggled).

**Takeaway:** the value proposition is *not* "faster on linear 2-D" — there FMG is
fine. It is **robust, scalable nonlinear Stokes in 3-D / parallel / plasticity**,
where the fieldsplit-Schur assumptions and direct solves break down.

### First data point — FMG vs FAS on hard benchmarks (`benchmark_fmg_vs_fas.py`)

FMG (Newton + fieldsplit-Schur + velocity FMG, `saddle_preconditioner=1/η`) vs FAS
(snes_type=fas, LU smoother — *not yet Vanka*), open-top so no nullspace,
refinement=2:

*SolCx viscosity step (linear), Δη = 1 → 10⁶:* both **converge at every contrast**.
FMG is faster (1–2 outer iterations, 7–40 s) but its time *grows* with the jump
(12 s → 40 s from 10⁴ → 10⁶ as the velocity-block MG degrades); FAS-LU is flat
(~62 s, 2 cycles, robust but slow). So with the right Schur preconditioner FMG
handles the discontinuous-viscosity benchmark well; FAS's edge only shows as the
contrast becomes extreme.

*Viscoplastic yield (nonlinear), τ_y = 10 → 0.25:* the nonlinear iteration count
favours FAS — at τ_y = 1, **FAS converges in 3 nonlinear cycles where FMG+Newton
needs 9 iterations** (the nonlinear coarse correction at work, as predicted). Below
τ_y ≈ 0.5 *both* fail — the hard-yielding regime is continuation/regularisation
territory for every solver (consistent with `snesfas-feasibility.md`). FAS is slower
in wall-clock here only because it uses the LU smoother; the Vanka smoother is what
would make the 3-vs-9 nonlinear-iteration advantage also a wall-clock win at scale.

**Reading:** FMG is the right default for linear / moderate problems; FAS's
robustness advantage is real but currently shows as *fewer nonlinear iterations* in
the plastic regime, not yet as wall-clock (LU smoother). Wiring in the Vanka smoother
+ pushing to 3-D / extreme contrast is where the combination should pull clearly
ahead. The very-hard yield regime needs continuation regardless of solver.

### Three turnkey choices — FMG vs GAMG vs FAS-Vanka (`benchmark_3way.py`)

With the Vanka smoother actually wired into FAS (custom-IS injection, GMRES Krylov
smoother), the three production options compared on the same problems (open top,
res 16 / refinement 1):

*SolCx viscosity step (linear):*

| Δη | FMG | GAMG | FAS-Vanka |
|---|---|---|---|
| 1 | 1 it, 3.9s | 2 it, 4.9s | 1 it, 3.0s |
| 10³ | 1 it, 2.8s | 3 it, 14s | 1 it (gmres-15 smoother) |
| 10⁶ | 2 it, 5.8s | 8 it, **99s** | **FAIL** |

*Viscoplastic yield (nonlinear):*

| τ_y | FMG | GAMG | FAS-Vanka |
|---|---|---|---|
| 10 | 1 it | 2 it | 1 it |
| 1 | 10 it | 10 it | **2 it** |
| 0.3 | FAIL | FAIL | FAIL |

**What this says — and the honest "how much tuning":**

- **FMG** is the **robust all-rounder and the right default.** Its
  `saddle_preconditioner = 1/η` makes the Schur (pressure) approximation
  viscosity-robust, so it sails through the 10⁶ contrast (5.8 s) where the others
  struggle, and it is fastest almost everywhere. Needs a refinement hierarchy.
- **GAMG** is the **no-hierarchy fallback** — robust but it *cliffs in cost* at high
  contrast (99 s at 10⁶, 8 outer iterations). Use when there is no geometric
  hierarchy.
- **FAS-Vanka** is the **nonlinear / plasticity specialist.** It crushes the
  viscoplastic case (**2 nonlinear iterations vs 10** for the others — the coarse
  problem is better conditioned, exactly as expected) and handles moderate viscosity
  contrast (10³) once the Krylov smoother is bumped to ~15 inner iterations. But with
  *additive* PCASM Vanka it **fails at extreme contrast (10⁶)** no matter how hard you
  smooth — that needs a **multiplicative** Vanka (PCGASM / a coloured patch sweep)
  and/or viscosity-aware coarsening, which is genuine development, not a flag.

**Bottom line for "2–3 good choices, not much tuning":** FMG (default) and GAMG
(fallback) are the two robust, low-tuning options today. FAS-Vanka is a strong third
**specifically for strongly nonlinear / plastic problems** (where it beats both on
iteration count) and for 3-D / large-parallel; making it a *robust* turnkey peer of
FMG across all regimes needs the multiplicative-Vanka smoother (the extreme-contrast
gap) plus the productionisation items (per-level nullspace, parallel patches, UW3
API). The plasticity win, though, is real and in hand now.

## Path forward (GO — productionise the prototype)

1. **Wrap in FAS for the nonlinear case.** The smoother is the hard part and it is now
   solved; FAS adds the nonlinear coarse correction, already proven to work with an LU
   smoother (`snesfas-feasibility.md`). Swap LU → the Vanka GMRES smoother above.
2. **Per-level pressure nullspace** for enclosed / free-slip problems (the prototype
   used an open top to avoid it). Register it at the DM/DS level so each level inherits
   it.
3. **Parallel patch construction** (PCASM is parallel; build the pressure-support index
   sets rank-locally) and a **UW3 API** (a `smoother="vanka"` path that builds the IS
   per level and installs the smoother — ≈ the prototype's ~40 lines, generalised).
4. **Open optimisation:** whether `PCPATCH` can be configured to build these exact
   pressure-support patches (would make it options-only, no custom-IS code).

The earlier "research only / different element" conclusion is fully withdrawn: a
mesh-independent Vanka multigrid for UW3's existing simplex Taylor-Hood Stokes is
demonstrated; what remains is engineering.

## Meanwhile

`FMG + Picard / nested iteration` remains the production workhorse for nonlinear
Stokes; LU-smoothed FAS is a correct, moderate-size-faster option where nonlinear
robustness matters more than asymptotic scaling. Neither is blocked; both ship today.

## Reproduction

`~/+Simulations/snesfas_spike/`: `vanka_stage1.py` (UW3 Stage 1 matrix), `/tmp`-staged
probes `vanka_quad.py`, `braess.py`, `bs_fas.py`, `bs_sor.py`. Clean-PETSc reference:
`petsc-custom/petsc/src/snes/tutorials/ex62.c` (compiled in the `amr-dev` env; gmsh
mesh `/tmp/square.msh` from `/tmp/mksquare.py`; driven with the `vanka`/`star` patch
options above). PETSc `/*TEST*/` block `*_vanka` suffixes are the recipe source.
