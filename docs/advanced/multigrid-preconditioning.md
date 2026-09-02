---
title: "Multigrid Preconditioning (FMG vs GAMG)"
---

# Multigrid Preconditioning: FMG vs GAMG

Underworld3 solvers are preconditioned with multigrid. There are two flavours,
and choosing the right one — or letting the solver choose for you — makes a large
difference to robustness and cost, especially on adapted or anisotropic meshes.

- **GAMG** (*algebraic* multigrid) builds its coarse levels from the assembled
  operator's connection graph. It is general and needs no mesh hierarchy, but it
  is **sensitive to anisotropy**: stretched cells (exactly what mesh adaptation
  produces) degrade the aggregates and the iteration count can cliff.
- **FMG** (geometric *Full Multigrid*) builds its coarse levels from a genuine
  **mesh refinement hierarchy**. Because the hierarchy is geometric, it is
  **inherently robust to anisotropy** — the coarse spaces are correct regardless
  of how the operator is stretched.

## The one-liner: build a mesh with `refinement`

Geometric multigrid needs a refinement hierarchy. Build one by passing
`refinement=N` to any mesh constructor:

```python
import underworld3 as uw

# refinement=2 -> a 3-level geometric hierarchy (coarse -> medium -> fine)
mesh = uw.meshing.Annulus(radiusInner=0.5, radiusOuter=1.0,
                          cellSize=0.1, refinement=2)

stokes = uw.systems.Stokes(mesh)
# ... constitutive model, body force, boundary conditions ...
stokes.solve()        # velocity block is preconditioned with geometric FMG
```

That is the whole story for the common case. When the mesh carries a hierarchy,
the solver **automatically** uses geometric Full Multigrid on the velocity block
(for Stokes) or the top-level preconditioner (for scalar/vector solvers). When it
does not, the solver falls back to GAMG. You do not need to set any PETSc options.

## The `preconditioner` knob

Every Stokes / scalar / vector solver exposes a `preconditioner` property:

```python
stokes.preconditioner = "auto"   # default: FMG if the mesh has a hierarchy, else GAMG
stokes.preconditioner = "fmg"    # force geometric multigrid (warns + falls back if no hierarchy)
stokes.preconditioner = "gamg"   # force algebraic multigrid (the historical default)
```

`"mg"` is accepted as an alias for `"fmg"`.

```{note}
`"auto"` is conservative. It only ever *adds* geometric multigrid on top of an
untouched default — it never rewrites a preconditioner you configured yourself. If
you set `pc_type` directly through `solver.petsc_options[...]`, or a solver applies
its own tuned options internally, `"auto"` leaves those settings alone.
```

## What the FMG bundle actually sets

For reference (you should rarely need to set these by hand), selecting geometric
multigrid is equivalent to the following options on the relevant block
(`fieldsplit_velocity_` for Stokes, top-level for scalar/vector):

```python
pc_type                      = "mg"
pc_mg_type                   = "full"        # Full Multigrid (F-cycle)
pc_mg_galerkin               = "both"        # RAP coarse operators (see below)
mg_levels_ksp_type           = "chebyshev"
mg_levels_pc_type            = "sor"
mg_levels_ksp_max_it         = 4
mg_coarse_pc_type            = "redundant"   # direct coarse solve, on every rank
mg_coarse_redundant_pc_type  = "lu"
```

**Why Galerkin coarse operators?** Underworld3 does not install
residual/Jacobian callbacks on the coarse DMs, so the coarse operators must be
formed by Galerkin projection ($R A P$) from the fine operator rather than by
re-discretising on the coarse mesh. `pc_mg_galerkin = both` does this.

## Hierarchy and mesh adaptation

This is the key reason to prefer FMG when you adapt:

- The coordinate-deforming adaptation movers (the MMPDE mover /
  `follow_metric`) **preserve mesh topology**. The refinement
  hierarchy survives them, so geometric multigrid keeps working as the mesh
  deforms — precisely where GAMG struggles with the resulting anisotropy.
- A **true remesh** (a topology change) collapses the hierarchy to a single
  level. In `"auto"` mode the solver detects this and transparently reverts to
  GAMG on the next solve, so nothing breaks — you simply lose the geometric path
  until a hierarchy is available again.

## The coarse level must be small, or the bottom solve stops being free

Multigrid is cheap because each level costs a fraction of the one above it, and
the bottom of the hierarchy is solved **exactly** at negligible cost. The word
doing the work in that sentence is *negligible*. A direct (or SVD) coarse solve
is only free if the coarsest level is genuinely tiny; nothing in the method makes
it so, and nothing warns you when it is not.

### What the default does, and why

The bundle sets a redundant direct solve:

```python
mg_coarse_pc_type            = "redundant"
mg_coarse_redundant_pc_type  = "lu"
```

`redundant` copies the whole coarse system to **every rank** and factors it
there. For a genuinely small system that is the right trade: communicating
across ranks to solve a few thousand unknowns collectively costs more than
solving the same thing everywhere. It is also np-safe, which a bare `lu` is not
— a serial LU cannot factor a distributed matrix and fails at `np > 1` with
`DIVERGED_LINEAR_SOLVE` after zero iterations. In serial, `redundant` + `lu` is
identical to `lu`.

This is not an Underworld idiosyncrasy, and it is worth being clear about that
before blaming the choice for a memory problem. **PETSc makes the same choice on
its own.** With nothing set under `mg_coarse_*`, `PCSetUp_MG` selects
`PCREDUNDANT` when the communicator has more than one rank and `PCLU` when it
does not — its own source comment reads "coarse solve is (redundant) LU by
default" (`src/ksp/pc/impls/mg/mg.c`). Confirmed by inspection at `np` = 1, 2
and 4 on PETSc 3.25.

Underworld sets it explicitly rather than inheriting it, so a bundle applied
after another cannot pick up a sibling's leftover options (#468) — but the value
is the one PETSc would have picked anyway. Deleting these keys and letting PETSc
decide would change nothing here; the leverage is in the hierarchy, not the
bottom solve.

### The failure mode: replicated memory

Because the factorization is replicated, its memory is paid **per rank**, and
the total scales with the coarse system size *times* the core count. That is
harmless for a small bottom level and fatal for a large one.

Measured on Setonix (issue #644), a production model at `np = 30` under a
900 MB/rank cgroup cap:

| Base mesh | Refinement | Coarsest velocity DOFs | Peak RSS/rank | Linear solve |
|---|---|---|---|---|
| 10 km | 2 | 14,195 | **683 MB** — OOM killed | ~100 s |
| 20 km | 3 | ~1,900 | **416 MB** | 111 s (+11%) |

Both configurations have essentially the same *finest* grid (233k vs 227k
velocity DOFs) and converge identically — the same SNES iteration sequence. The
only real difference is how far down the hierarchy goes. Shrinking the coarsest
level alone removed 261 MB/rank, which matches the gather-plus-fill arithmetic
(~45 MB for the gathered copy, ~232 MB for the factor at a 5.16x fill ratio).

### The fix is the hierarchy, not the coarse solver

The instinct on an OOM is to reach for a different bottom solve. That treats the
symptom. A 14,000-DOF coarsest level means the hierarchy was built from too fine
a base with too few levels — the bottom of the ladder never got small, so the
"solve the coarse problem exactly, for free" premise never held.

Keep the finest resolution you need and add levels underneath it:

```python
# OOM: fine base, few levels -> the bottom is still large
mesh = uw.meshing.UnstructuredSimplexBox(..., cellSize=0.10, refinement=2)

# same finest grid, one more level, a much smaller bottom
mesh = uw.meshing.UnstructuredSimplexBox(..., cellSize=0.20, refinement=3)
```

Each refinement level divides cell size by two, so the level above has 4x the
cells in 2-D and 8x in 3-D — measured exactly, 66 / 264 / 1056 and
184 / 1472 / 11776 on a `refinement=2` box. Doubling the base cell size and
adding one level therefore lands on the same finest grid while making every
coarser level — the bottom one included — about 4x (or 8x) smaller.

"The same finest grid" is approximate, because the base mesh is unstructured and
gmsh does not produce exactly 4x the cells for half the target size. Measured on
a unit box, `cellSize=0.10, refinement=2` gives 3,872 finest cells against
`cellSize=0.20, refinement=3` at 4,224 — 9% more, while the coarsest level drops
to 27% of its former size. The production model in #644 landed the other way:
227k against 233k. Expect a few percent either side, and check rather than
assume.

Check what you actually built rather than assuming; `dm_hierarchy[0]` is the
coarsest level and `[-1]` is the mesh you solve on:

```python
for i, dm in enumerate(mesh.dm_hierarchy):
    cells = uw.mpi.comm.allreduce(dm.getStratumSize("depth", mesh.dim))
    uw.pprint(f"level {i}: {cells} cells")
```

There is no single DOF number that is "small enough", because the quantity that
has to fit is the replicated factor **per rank**, against your per-rank memory
budget — the core count is in the arithmetic, not just the coarse size. The #644
measurement calibrates it: ~14k velocity DOFs cost ~300 MB/rank at a 5.16x LU
fill, so a 900 MB/rank cap is already lost before the rest of the solver is
counted, while ~1.9k DOFs cost little enough to disappear.

As a working rule, keep the coarsest level in the low thousands of DOFs, and
treat five figures as a bug in the hierarchy rather than a tuning question. If
you are near the line, estimate it rather than guessing: the factor grows faster
than the DOF count, and you pay it on every rank at once.

### When the coarse operator has a null space: SVD

One case does require a different bottom solve, and the solver handles it for
you. A **rotated** velocity block whose domain leaves rigid rotations free — a
closed circle (one mode), a spherical shell (three) — hands those modes to the
Galerkin-coarsened coarse operator, because the rotations survive the
constraint. `redundant`/LU hits a zero pivot on a singular coarse operator and
fails with `SUBPC_ERROR` (outer reason `-11`), so the rotated path selects SVD
instead.

It is deliberately **not** a blanket choice for every rotated problem. It keys
on the count of *verified rotation* modes, because a rotated problem with
Dirichlet walls, or a split-fault contact box, has none — and there an SVD is
pure cost. A 3-D P2 coarse level is a **dense** factorization: measured, it
accounted for most of ~0.8 s per V-cycle application at healthy iteration counts
(#622).

```{warning}
There is no `petsc_options` override for the coarse solve **on the rotated
path**. That route writes its multigrid bundle under the velocity sub-PC's own
prefix, applies it, and deletes the keys again; anything you set through
`solver.petsc_options[...]` lives under the solver's SNES prefix and never
reaches it. The override below works on the standard path only.
```

Because SVD is dense, it is affordable only on a genuinely small coarse level —
which makes the sizing guidance above a hard requirement on this path rather
than a preference.

### Overriding the coarse solve yourself

On the standard (non-rotated) path, `"auto"` does not overwrite options you set
yourself, so you can keep `preconditioner = "auto"` and change just the bottom
solve:

```python
stokes.preconditioner = "auto"
stokes.petsc_options["fieldsplit_velocity_mg_coarse_pc_type"] = "svd"
```

```{note}
`PCTELESCOPE` — the usual PETSc answer to "replicating onto every rank is too
many copies", which gathers the coarse system onto a sub-communicator instead —
**is not available here**. It has no DMPlex support, and every Underworld mesh
is a DMPlex, so requesting it aborts the solve with `Support for DMPLEX is
currently not available` (PETSc error 56). There is no bottom-solve escape hatch
from an over-large coarse level: the hierarchy is the lever.
```

## Benchmark: FMG vs GAMG on a deforming adaptive mesh

The payoff is clearest on an aggressively adapted mesh. The figure below is a
Stokes convection run (annulus, Ra = 10⁷, Δη = 10³, res 32, resolution-ratio
R = 8, mode-1, `np = 5`) whose mesh is continuously deformed by the MMPDE
coordinate mover, adapted every timestep. The geometric hierarchy (3 levels)
survives every step — topology is preserved — and both engines converge cleanly
(`snes` reason 3) throughout. FMG (`PCVEL=gmg MG_TYPE=full`) and GAMG
(`PCVEL=amg`) ran the *identical* adapted-mesh sequence over 50 steps.

```{figure} figures/bench_fmg_vs_gamg_velocity_ksp.svg
:alt: Inner velocity-block KSP iterations and Stokes-solve wall time versus adaptation step, for FMG and GAMG.

**Velocity-block solver scaling under adaptive remeshing.** Inner velocity-block
KSP iterations (top) and Stokes-solve wall time (bottom) versus adaptation step.
Geometric full multigrid (FMG) keeps a mesh-independent ≈ 5 inner iterations as
the cells stretch, where algebraic multigrid (GAMG) runs a volatile ≈ 64–131
(≈ 23×) without cliffing at this anisotropy. The wall-clock gap is only ≈ 1.8×:
each GAMG V-cycle is far cheaper than an FMG F-cycle, and the cold-start Stokes
solve after each adapt (common to both) dominates the time. The outer Schur KSP
converges in one iteration for both — the difference is entirely in the inner
velocity block.
```

The metric that matters is the **inner velocity-block KSP iteration count** — the
outer Schur KSP is one iteration for both engines, so the entire difference lives
in this inner block:

- **FMG** holds a **mesh-independent ≈ 5** iterations (3–6) as the anisotropy
  sharpens — the geometric-MG signature.
- **GAMG** runs a **volatile ≈ 64–131** (median ≈ 114, so ≈ 23×) as the algebraic
  aggregates cope with the stretched cells. It does *not* cliff at R = 8 over
  these 50 steps, but it is unpredictable from step to step.

The wall-clock gap is only **≈ 1.8×**, not 23×: a single GAMG V-cycle is far
cheaper than an FMG F-cycle, and the cold-start Stokes solve after each adapt
(common to both engines) dominates the per-step time. So the value of geometric
FMG here is **predictability and mesh-independence** — properties that widen with
problem size and multigrid depth — rather than a large raw speed-up. The iteration
gap is also where GAMG eventually loses robustness on harder problems (higher
anisotropy, more levels).

The per-step data is in
[`figures/bench_fmg_vs_gamg_velocity_ksp.csv`](figures/bench_fmg_vs_gamg_velocity_ksp.csv);
the reproduction command and full provenance are in the companion note
`figures/bench_fmg_vs_gamg_velocity_ksp.md`.

## When to use which

| Situation | Recommended |
|-----------|-------------|
| Adapted / deformed / anisotropic meshes | **FMG** (build with `refinement`) |
| Uniform mesh, want mesh-independent iteration counts | **FMG** |
| No refinement hierarchy available / quick prototype | GAMG (automatic fallback) |
| Reproducing historical results exactly | `preconditioner = "gamg"` |

## See also

- {doc}`mesh-adaptation` — the movers whose anisotropy FMG handles gracefully
- {doc}`performance` — profiling and scaling
- `docs/developer/design/solver-strategies-catalogue.md` — the solver-strategy
  design notes (developer-facing)
