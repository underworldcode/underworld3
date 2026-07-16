---
title: "SNESFAS (nonlinear multigrid) feasibility in Underworld3"
status: "Investigation record (preserved via PR #245, 2026-06-18); the production geometric-MG path is custom prolongation (PR #290)"
---

# SNESFAS feasibility spike

**Verdict: GO for scalar nonlinear, and GO for nonlinear Stokes with adaptation
(with two engineering follow-ons).** PETSc's `SNESFAS` (Full Approximation Scheme —
nonlinear geometric multigrid) runs through Underworld3's existing DMPlex-FE solver
stack **with no source changes** — it is activated entirely through `petsc_options`.
On a strongly nonlinear scalar problem it is mesh-independent, more robust than
Newton, and faster, in serial and parallel. On **nonlinear (power-law) Stokes** it
holds a nonlinearity-independent 2–3 multigrid cycles where Newton climbs to 30+
iterations, **including on an adapted (coordinate-deformed) mesh** — the actual
production target. The two remaining pieces before it is production-ready are
scalable saddle-point smoothers (the spike used direct LU per level) and a
per-level pressure nullspace for enclosed/free-slip problems.

This note records the spike (scalar `SNES_Scalar` / `Poisson`, static meshes) that
established feasibility, and what remains before FAS could be offered as a
production feature or extended to Stokes.

## Background: why FAS, and the architectural question

The geometric **FMG** preconditioner landed in #231 gives *linear* multigrid that
is robust to mesh anisotropy. But nonlinear robustness still rests on Newton
(`newtonls`) / Picard, whose convergence basin shrinks as nonlinearity sharpens
(temperature-dependent viscosity, viscoplastic yield, Richards fronts). `SNESFAS`
performs coarse-grid correction on the **nonlinear residual itself**, which is the
classic lever for global robustness.

The open question was whether UW3 could feed FAS a genuine nonlinear residual on
every coarse level. The FMG note (`docs/advanced/multigrid-preconditioning.md`)
states that UW3 "does not install residual/Jacobian callbacks on the coarse DMs,
so the coarse operators must be formed by Galerkin projection (RAP)". That is true
for the **linear `PCMG`** path — but it turns out **not** to block FAS.

## Key finding: FAS works with zero code changes

In `petsc_generic_snes_solvers.pyx` each solver `_build`:

1. copies the `PetscDS` (which holds the JIT pointwise residual/Jacobian function
   pointers) to **every** coarse DM — `self.dm.copyDS(coarse_dm)` in a loop over
   `self.dm_hierarchy`; and
2. installs the SNES local-FEM callback on the **fine** DM only —
   `UW_DMPlexSetSNESLocalFEM(cdm.dm, ...)`.

When `snes_type = fas`, PETSc's `SNESSetUp_FAS` walks the coarse-DM chain (already
linked by `setCoarseDM` when the mesh is built with `refinement=N`) and
**propagates the fine DM's `DMSNES` local-FEM callback down to each level** via
`DMCopyDMSNES`. Combined with the already-copied DS, every level can evaluate its
own nonlinear residual and Jacobian. `-snes_view` confirms it:

```
type: fas
  type is FULL, levels=3, cycles=1
  Not using Galerkin computed coarse grid function evaluation   <-- genuine re-discretisation
  Coarse grid solver -- level 0:  newtonls + LU,   rows=445
  Down/Up smoother on level 1:    newtonls,        rows=1857
  Down/Up smoother on level 2:    newtonls (fine)
```

So the spike's anticipated "install callbacks on coarse DMs" change was
**unnecessary**. FAS is a `petsc_options`-only capability today.

## Results (scalar, static mesh)

Two testbeds on the unit square, P2 elements, mesh built with `refinement`:

- **Bratu** `-Δu = λ e^u`, `u=0` on `∂Ω` (lower branch exists for `λ < λ_c ≈ 6.808`).
- **Exponential nonlinear diffusion** `-∇·(e^{βu}∇u) = f`, manufactured solution
  `u = sin πx · sin πy` so a solution provably exists at every β (a stiff Newton
  stressor as β grows).

Configurations: `newton` (`newtonls`), `fas` (`snes_type=fas`, FULL cycle,
`newtonls` smoothers, LU coarse), `ngmres+fas` (NGMRES outer with FAS as the
nonlinear preconditioner, `-npc_snes_type fas`).

### Correctness and mesh-independence (Bratu, λ=5)

| refinement | levels | Newton its | FAS its (F-cycles) | ‖u_FAS − u_Newton‖ |
|---|---|---|---|---|
| 1 | 2 | 4 | **1** | 7e-15 |
| 2 | 3 | 4 | **1** | 2e-15 |
| 3 | 4 | 4 | **1** | 2e-15 |

FAS converges in a single F-cycle independent of refinement, to the same solution
as Newton. Across `λ = 3…6.8`, `fas` and `ngmres+fas` agree with Newton to
1e-9…1e-15. Both Newton and FAS hit the **same** envelope at the fold
(`λ ≳ 6.85`, where no solution exists) — Bratu's lower branch is reachable by
Newton from a cold start, so Bratu shows FAS *works*, not that it is *more robust*.

### Robustness advantage (exponential diffusion, MMS)

| β | Newton | FAS | NGMRES+FAS | notes |
|---|---|---|---|---|
| 1 | 16 its ✓ | **1** ✓ | 1 ✓ | L2 ≈ 2e-7 all |
| 2 | 26 its ✓ | **1** ✓ | 1 ✓ | |
| 3 | **diverges** (line search) | **1 ✓** | **1 ✓** | solution exists — FAS finds it (L2 2.2e-7), Newton cannot |
| 4 | diverges | diverges | diverges | default smoothers insufficient |
| 5 | diverges | 80 (stalls) | 27 (fails) | |

This is the headline: at β=3 a solution provably exists and **FAS converges in one
cycle where Newton diverges from the cold start**. FAS is not a silver bullet —
β ≥ 4 needs stronger smoothers/continuation than the spike's defaults.

### Wall-clock (exponential diffusion)

| β | refinement | Newton | FAS | speed-up |
|---|---|---|---|---|
| 1 | 2 | 1.86 s (16) | 1.10 s (1) | 1.7× |
| 2 | 2 | 2.59 s (26) | 1.14 s (1) | 2.3× |
| 1 | 3 | 6.45 s (16) | 5.35 s (1) | 1.2× |
| 2 | 3 | 10.6 s (26) | 5.56 s (1) | 1.9× |

FAS is faster as well as more robust when the nonlinearity is strong (Newton needs
many steps). On mild problems where Newton converges in 3–4 steps, FAS's
per-cycle cost makes it merely competitive, not faster.

### Parallel

np=2 with a parallel-safe coarse solve (`fas_coarse_pc_type=redundant`,
`fas_coarse_redundant_pc_type=lu`) converges in 1 cycle, `L2=2.06e-7` vs np=1's
`2.07e-7`. The scalar parallel gate is clear.

### Adaptation — FAS on a coordinate-deformed mesh

The headline risk in the first cut of this note was that, because the movers
update only the fine DM's coordinates, FAS's coarse(undeformed) → fine(deformed)
inter-grid transfers might be "too weak" to converge. **Tested and refuted for the
scalar case.** A refined mesh (3 levels) was deformed with `follow_metric`
(tanh-front metric) at increasing strength, then the exponential-diffusion MMS was
solved on the deformed mesh:

| follow R | q_min | coarse moved | fine moved | Newton | FAS | FAS L2 |
|---|---|---|---|---|---|---|
| — (static) | 0.89 | — | — | 26 ✓ | **1** ✓ | 2.1e-7 |
| 1.5 | 0.37 | **0** | 0.21 | 24 ✓ | **2** ✓ | 4.2e-7 |
| 2.0 | 0.32 | **0** | 0.35 | 24 ✓ | **2** ✓ | 6.3e-7 |
| 2.5 | 0.34 | **0** | 0.41 | 25 ✓ | **2** ✓ | 5.7e-7 |

The mismatch is real ("coarse moved" = 0 confirms coarse DMs keep original
geometry while the fine DM deforms), yet FAS degrades only from 1 to **2 F-cycles**
and never loses correctness (L2 tracks Newton exactly). Pushed harder — into the
β=3 regime where Newton **diverges** on a uniform mesh — FAS on the *deformed* mesh
still converges in 2 cycles and stays correct:

| β | follow R | q_min | Newton | FAS | FAS L2 |
|---|---|---|---|---|---|
| 3 | 1.5 | 0.37 | **diverges** | **2 ✓** | 4.3e-7 |
| 3 | 2.0 | 0.32 | **diverges** | **2 ✓** | 6.7e-7 |
| 2 | 3.0 | 0.32 | 25 ✓ | 2 ✓ | 5.3e-7 |
| 3 | 3.0 | 0.32 | **diverges** | **2 ✓** | 5.9e-7 |

Why it works despite the mismatch: PETSc builds the transfers from the parent→child
*refinement* relationship (reference-element shape functions), not from a match of
physical coarse/fine geometry, and FAS only needs the coarse step to be a useful
*correction* — the fine smoother removes whatever the geometrically-imperfect
coarse correction leaves behind. Coarse-coordinate propagation would likely recover
the single-cycle count, but is **not required for convergence** on the scalar case.

## How to use it today (no code required)

```python
mesh = uw.meshing.UnstructuredSimplexBox(..., refinement=2)   # builds the hierarchy
poisson = uw.systems.Poisson(mesh, u_Field=u)
# ... constitutive model, nonlinear f, BCs ...

po = poisson.petsc_options
po["snes_type"] = "fas"
po["snes_fas_type"] = "full"            # FMG-style F-cycle
po["fas_levels_snes_type"] = "newtonls" # per-level nonlinear smoother
po["fas_levels_snes_max_it"] = 4
po["fas_levels_snes_linesearch_type"] = "basic"
po["fas_coarse_snes_type"] = "newtonls" # coarse nonlinear solve
po["fas_coarse_ksp_type"] = "preonly"
po["fas_coarse_pc_type"] = "lu"         # parallel: "redundant" + redundant_pc_type "lu"
poisson.solve(zero_init_guess=True)

# Or FAS as a nonlinear preconditioner to a robust outer accelerator:
#   po["snes_type"] = "ngmres";  po["npc_snes_type"] = "fas";  po["npc_fas_*"] = ...
```

## Stokes (saddle-point) — the production target

Everything above is scalar. The real goal is **nonlinear Stokes with adaptation**.
The spike carried the result all the way there.

**Plumbing (linear Stokes).** FAS runs on the velocity–pressure saddle-point system
with no code changes. Using a monolithic LU smoother on each level (newtonls +
`preonly`/`lu`) on a constant-viscosity box with an **open top** (traction-free, so
*no* constant-pressure nullspace — the LU level solves stay non-singular), FAS
reaches the same velocity field as the default fieldsplit+FMG solve (rel.diff 4e-5,
1 cycle). So coarse residual, inter-grid transfers, and coarse correction all work
on a saddle system. The Stokes `solve()` reads `snes_type` from
`self.snes.getType()` (set by `petsc_options` during `_build`) and re-applies it, so
`snes_type=fas` survives on the standard path (`picard=0`, `zero_init_guess=True`).

**Nonlinear Stokes — the win.** A *smooth* shear-thinning power-law viscosity
`η = η₀ (ε_II/ε_ref)^(1/n − 1)` (n=1 linear, larger n more nonlinear), open top:

| n | Newton its | FAS cycles | same soln |
|---|---|---|---|
| 1 | 1 | 1 | 4e-6 |
| 2 | 14 | **2** | 2e-4 |
| 3 | 22 | **2** | 3e-4 |
| 4 | 28 | **3** | 3e-4 |
| 5 | 30 | **3** | 2e-4 |

FAS holds **2–3 cycles while Newton climbs 14 → 30** as the nonlinearity sharpens —
the same nonlinearity-independence seen in the scalar exponential-diffusion case,
now on the saddle-point system, converging to the same solution.

**Wall-clock — is the complexity worth it?** Yes, even with the un-optimised LU
smoother. FAS (monolithic LU per level) vs Newton (default fieldsplit + FMG):

| n | refine | levels | Newton | FAS | speed-up |
|---|---|---|---|---|---|
| 3 | 2 | 3 | 15.0 s | 9.1 s | 1.66× |
| 5 | 2 | 3 | 19.9 s | 13.9 s | 1.44× |
| 3 | 3 | 4 | 72.5 s | 38.6 s | 1.88× |
| 5 | 3 | 4 | 95.3 s | 59.9 s | 1.59× |

FAS is **1.4–1.9× faster** despite paying for a direct solve on every level, and the
gap *widens* with mesh size (1.66→1.88× from ref2→ref3 at n=3) because the cycle
count stays flat while Newton's iteration count does not.

**But the per-level smoother is the crux, and it is not a `petsc_options` swap.**
`-snes_view` confirms the smoother on *every* level (including the fine, 17505×17505)
is `newtonls`+`preonly`+`lu` — a full direct factorization, ~6 of them per cycle on
the fine grid. That does not scale (2-D sparse LU is ≈ O(N^1.5)). The obvious "fix"
— a fieldsplit-Schur smoother on the levels, LU only on the coarse grid — was tried
and is **9–17× slower**, not faster:

| n | refine | LU smoother | fieldsplit-Schur smoother |
|---|---|---|---|
| 3 | 2 | 9.0 s | 81 s |
| 3 | 3 | 37.6 s | 657 s |

Both converge in 2–3 cycles to the same solution, but a Schur fieldsplit is itself
an approximate Stokes *solve*, and as a smoother it is invoked many times (≈ 2 Newton
steps × down+up × per level × per cycle) — so each relaxation does a near-complete
Stokes solve. At these sizes a single direct LU is cheaper. The genuine requirement
is an **inexpensive saddle-point smoother** — Vanka (element/patch block), Braess–
Sarazin, or a distributive/Uzawa relaxation — none of which is a stock PETSc
`petsc_options` choice for a DMPlex FE Stokes operator. **This, not the nullspace,
is the real research/engineering cost of production Stokes FAS.** Until it exists,
LU-smoothed FAS is a correct and (at moderate size) faster method whose cost is
dominated by the fine-grid direct solve.

**Nonlinear Stokes ON AN ADAPTED MESH — the target.** Power-law Stokes (n=3) with
the mesh deformed by `follow_metric` toward the buoyancy feature (coarse DMs keep
their original geometry — the same mismatch the scalar case shrugged off):

| follow R | q_min | Newton its | FAS cycles | same soln |
|---|---|---|---|---|
| — (static) | 0.89 | 22 | 2 | 0.0931≈0.0932 |
| 1.5 | 0.31 | 22 | **2** | 2.8e-4 |
| 2.0 | 0.32 | 22 | **3** | 2.7e-4 |
| 2.5 | 0.36 | 23 | **3** | 1.8e-4 |

FAS keeps its 2–3 cycle convergence on the deformed saddle-point mesh, same answer
every time. **Feasibility for nonlinear-Stokes-with-adaptation is established.**

**Decomposing the benefit: globalization vs convergence rate.** FAS delivers two
distinct things — (a) it drops the fine grid into Newton's quadratic basin (the
coarse→fine ramp-up / *globalization*), and (b) a genuine nonlinear-multigrid
*convergence rate* via repeated coarse correction. The globalization piece can be
had *cheaply*, reusing the existing Newton + linear-FMG machinery with no saddle
smoother, by **nested iteration / grid sequencing**: solve coarse, interpolate,
warm-start fine. Two ways to get it, and how much it buys:

- PETSc `-snes_grid_sequence` **does not work** on UW3 meshes via `petsc_options`:
  it triggers `DMPlexComputeInterpolatorGeneral` (PETSc err 56) rather than the
  pre-chained nested hierarchy that FAS reuses — so, unlike FAS, it is *not*
  plug-and-play.
- **Manual nested iteration at the UW3 level** (coarse solve → `uw.function.evaluate`
  onto the fine field → warm-started fine solve) *does* work, and drops the fine
  Newton count from 22→14 (n=3) and 29→19 (n=5) — but only ~1.1–1.2× wall-clock.

So globalization alone is a *modest* win: it fixes the initial guess but not the
per-level convergence rate, and the coarse solve still pays the full nonlinear cost.
FAS's larger 1.5–1.9× comes from (b), the multigrid convergence rate — which is
exactly the part that needs the inexpensive saddle smoother. Practical reading: for
*robustness* on hard nonlinear Stokes, Picard warm-up (already in UW3) or manual
nested iteration is cheap and reuses FMG; for *speed*, FAS is the lever but only
once a real saddle smoother exists.

**Where Stokes FAS does *not* help: non-smooth yielding.** A von Mises viscoplastic
viscosity `η = min(η₀, τ_y/2ε_II)` is hard for *every* solver: once the domain
yields (τ_y ≲ 1 here) Newton, Picard **and** FAS all fail. The `min()` kink is
non-smooth and, worse for FAS, the coarse level yields on a different pattern than
the fine, so the coarse correction is inconsistent. This is the known
regularization/continuation regime, not a FAS-specific failure — the same shape as
the scalar β≥4 ceiling.

**Two engineering follow-ons before production Stokes FAS:**

1. **An inexpensive saddle-point smoother (the hard part — feasible, but real work).**
   The spike's per-level smoother is a monolithic direct LU — fine for proving the
   algorithm (the 2–3 cycle count is the real result), but it does not scale.
   Replacements tried, all on power-law n=3 at ref2/ref3:

   | smoother | result |
   |---|---|
   | monolithic LU (baseline) | converges, 2 cycles, 11 s / 38 s — but LU does not scale |
   | fieldsplit-Schur, heavy (gmres 20) | converges but **9–17× slower** (a Schur solve per relaxation) |
   | fieldsplit-Schur, light (gmres 1–2) | **zero pivot** → `DIVERGED_INNER` (−7) |
   | PCPATCH **Vanka** | **zero pivot** on setup (singular local patches) |

   Every cheap attempt dies on the same rock: **"Zero pivot in LU factorization."**
   The pressure block has a zero diagonal, so naive relaxations (SOR / ILU / a plain
   sub-LU) divide by zero. That is the defining saddle-point difficulty, and exactly
   what Vanka (solve each local velocity+pressure patch as a coupled block) and
   Braess–Sarazin (a specific approximate-Schur relaxation) are constructed to avoid.
   So the smoother is **feasible — Vanka via PETSc `PCPATCH` is the standard tool and
   is used in production for Stokes geometric MG (e.g. Firedrake)** — but wiring it
   into UW3 is a real task: correct `PCPATCH` patch construction so the local saddles
   are non-singular, plus UW3-side exposure of the DM fields/topology to the patch PC
   (analogous to `_setup_block_fieldsplit_options`). It is **not** a `petsc_options`
   one-liner, which is what makes it the genuine engineering cost of scalable Stokes
   FAS.
2. **Per-level pressure nullspace.** The spike used an open-top problem to avoid the
   constant-pressure nullspace. `_attach_stokes_nullspace()` sets the nullspace on
   the *outer* matrix only; FAS builds each level's matrix internally. Enclosed /
   free-slip problems will need the nullspace registered at the DM/DS level (e.g.
   `DMSetNullSpaceConstructor` on the pressure field) so every FAS level inherits
   it. This is the one place a small UW3 code change is likely required.

## Limitations and open questions

- **Smoother/coarse tuning matters.** Default `newtonls` smoothers handle moderate
  nonlinearity; very stiff regimes need stronger smoothers, more pre/post sweeps,
  W-cycles, or continuation. This is normal FAS practice, not a UW3 limitation.
- **Coarse geometry under movers — TESTED, benign for scalar.** The movers update
  only the fine DM's coordinates; coarse DMs keep their original geometry (verified:
  "coarse moved" = 0). This was the headline risk, but on the scalar case FAS only
  slows from 1 to 2 F-cycles under deformation strong enough to drop q_min to 0.32,
  stays correct, and still converges where Newton diverges (see the adaptation
  tables above). Coarse-coordinate propagation down the hierarchy would likely
  restore the single-cycle count but is **not a prerequisite for convergence**. Whether
  this still holds for the much larger anisotropy of an adapted Stokes problem is
  the next thing to confirm.
- **Stokes saddle-point FAS — separate, harder.** FAS smoothers on the
  velocity-pressure system need a fieldsplit smoother and a constant-pressure
  nullspace on every level. Out of scope here; revisit only after the scalar path
  and the coarse-geometry question are settled.

## Recommendation

1. **Offer FAS as an opt-in for scalar (and likely vector) nonlinear solves.** It
   needs no new code — only a documented option bundle, or a thin
   `nonlinear_preconditioner` / `snes_type='fas'` convenience knob mirroring the
   FMG `preconditioner` property (default off; FMG/Newton defaults untouched).
   The exact deliverable shape is deferred pending these results.
2. **Validate on Richards** (`docs/beginner/tutorials/16/17`), the strongest
   real-world scalar FAS target, before committing to an API.
3. **Nonlinear-Stokes-with-adaptation is demonstrated** (2–3 cycles vs Newton's
   ~22, on a deformed mesh, same solution; 1.4–1.9× faster with LU smoothers).
   Production scalability hinges on an **inexpensive saddle-point smoother** (Vanka /
   Braess–Sarazin / distributive) — *not* a stock fieldsplit, which is 9–17× slower
   as a smoother. A **per-level pressure nullspace** is the smaller second piece
   (enclosed/free-slip). Coarse-coordinate propagation stays an optimisation; the
   non-smooth viscoplastic regime is out of scope (hard for all solvers).

## Reproduction

Spike scripts (not part of the repo; under `~/+Simulations/snesfas_spike/`):
`bratu_baseline.py`, `fas_probe.py`, `fas_measure.py`, `expdiff_mms.py`,
`adapt_fas.py` (FAS on a `follow_metric`-deformed mesh), and the Stokes set:
`stokes_fas.py` (linear plumbing), `stokes_fas_nl.py` (viscoplastic — hard for
all), `stokes_fas_powerlaw.py` (smooth nonlinear win), `stokes_fas_adapt.py`
(nonlinear Stokes on an adapted mesh).
Worktree `feature/snesfas-spike` off `origin/development` — **no source files
changed** (`git status` on `src/` is clean); the only artifact in-repo is this
note. Regression sanity: `tests/test_1014_stokes_multigrid.py` +
`tests/test_1000_poissonCart.py` → 18 passed.
