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
mg_coarse_pc_type            = "lu"          # direct coarse solve
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

## Parallel coarse solve

The default coarse solver is a serial direct solve (`mg_coarse_pc_type = "lu"`),
which is the fast, simple choice in serial and for modest core counts. For large
parallel partitions, replicate or gather the (small) coarse grid instead:

```python
# redundant: copy the coarse grid to every rank and solve it there
stokes.petsc_options["fieldsplit_velocity_mg_coarse_pc_type"] = "redundant"
stokes.petsc_options["fieldsplit_velocity_mg_coarse_redundant_pc_type"] = "lu"

# or telescope onto a sub-communicator for very large runs
# stokes.petsc_options["fieldsplit_velocity_mg_coarse_pc_type"] = "telescope"
```

Because `"auto"` does not overwrite options you set yourself, you can keep
`preconditioner = "auto"` and just override the coarse solver as above.

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
