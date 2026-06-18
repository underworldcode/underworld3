# SNESFAS / Vanka / grid-sequencing — research prototypes

> **Status: research prototypes, not production code.** These scripts back the design
> notes in `docs/developer/design/` (`snesfas-feasibility.md`,
> `snesfas-vanka-feasibility-study.md`, `multilevel-nonlinear-stokes-strategy.md`).
> They are preserved so the *working* multilevel-Stokes implementations are not lost
> when the investigation is parked. They are **not tested, not tier-classified, and
> make no `src/` changes** — everything here runs through `petsc_options` and
> petsc4py on top of stock UW3. Wall-clock numbers in the docs are soft (JIT / noise);
> trust the iteration counts.

Run any of them with the worktree's environment, e.g.
`pixi run -e <env> python docs/examples/snesfas_investigation/<script>.py`.
All build their own meshes (with `refinement=` so a `dm_hierarchy` exists).

## The working pieces

| script | what it demonstrates |
|---|---|
| `vanka_asm.py` | **The core recipe.** Vanka as overlapping Schwarz with patches = *support of each pressure basis function* (one pressure DOF + its B-coupled velocity DOFs), built as custom `PCASM` index sets with exact local LU. Works on UW3's simplex P2–P1 where stock `PCPATCH` fails. |
| `vanka_mg.py` | **Mesh-independent Vanka multigrid.** Geometric MG on the *full* Stokes saddle over UW3's `dm_hierarchy` (Galerkin coarse operators) with the custom-IS Vanka smoother + a **GMRES Krylov smoother** (additive Schwarz alone diverges). ~5–6 outer iterations independent of refinement. Also times vs FMG. |
| `fas_vanka.py` | **Working SNESFAS-Vanka for nonlinear Stokes** (modest contrast). Drives `snes_type=fas`, injects the custom-IS Vanka smoother into each FAS level (warm-up solve to assemble level operators → inject → re-solve). This is the piece that gives **2 nonlinear iterations vs 10** for Newton+FMG on viscoplastic τ_y=1. |
| `benchmark_3way.py` | **FMG vs GAMG vs FAS-Vanka** on SolCx viscosity contrast (linear) and viscoplastic yield (nonlinear). The head-to-head behind the "three choices" table. |
| `notched_beam_cascade.py` | **Grid sequencing (nested iteration)** prototype: solve coarse → interpolate up (`uw.function.evaluate`, robust to non-nested meshes) → solve fine, FMG per level. Robust where PETSc's `-snes_grid_sequence` fails on UW3 meshes. |

## Key recipes (so they aren't lost in the code)

**Custom-IS Vanka patch** (per pressure DOF `pg`, from the assembled saddle Jacobian `J`):
patch = `{pg}` ∪ `{velocity DOFs in J.getRow(pg)}`, fed to `PCASM` (RESTRICT) with
`sub_ksp_type=preonly`, `sub_pc_type=lu`.

**Vanka geometric MG** (linear): `pc_type=mg`, `pc_mg_galerkin=both`, per-level
`ksp_type=gmres` (the Krylov smoother — *essential*; Richardson diverges) wrapping the
custom-IS `PCASM`, `mg_coarse_pc_type=lu`/`svd`.

**FAS-Vanka** (nonlinear): `snes_type=fas`, `snes_fas_type=full`; level smoothers are
`newtonls` whose KSP is `gmres` (max_it ~15 for moderate contrast) + the custom-IS
`PCASM`; coarse solve LU. The patch *structure* is sparsity-only, so it stays valid as
the operator values change across nonlinear iterations.

## Known limits (see the design notes)

- Additive Vanka **fails at extreme viscosity contrast (≳10⁶)** — needs *multiplicative*
  Vanka + interface-aware coarsening.
- Per-level **pressure nullspace** for enclosed problems, **parallel** patch
  construction, and a clean UW3 API are all unimplemented (the prototypes use open-top
  problems to avoid the nullspace).
- The cascade only helps when the per-level solve is iteration-limited; it is overhead
  otherwise, so it should be opt-in.
