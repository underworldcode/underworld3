# SNES update callbacks — follow-up: boundary-correct field scatter

> **Status:** the per-iteration callback feature is **landed and pushed** on
> `feature/snes-update-callbacks` (commit "Add per-iteration SNES update callbacks").
> Use case 1 (pressure gauge) works to machine precision. This brief is the **one
> remaining piece** for use case 2 (Helmholtz / shear-band smoother). Continue on this
> same branch.

## What works now

- `solver.add_update_callback(fn)` — `fn(solver, iteration)` at the start of every SNES
  iteration; dispatcher in `petsc_generic_snes_solvers.pyx` (`_dispatch_snes_update`,
  `_scatter_global_to_fields`, `_gather_fields_to_global`, `_refresh_auxiliary_vec`),
  plus a final post-solve application.
- `SNES_Stokes.set_pressure_gauge(boundary)` — validated machine-zero.
- Tests: `tests/test_1016_snes_update_callbacks.py` (all pass).
- Docs: `docs/advanced/solver-iteration-callbacks.md` (has a `{warning}` to remove once
  this lands).

## The remaining bug

`_scatter_global_to_fields` uses `subdm.globalToLocal(sgvec, var.vec)`. That fills a
field's **interior** and **zero-Dirichlet** DOFs correctly, but **not non-zero Dirichlet
(driven) boundary DOFs** — those are not in the global vector; they are imposed on the
*local* vector by `DMPlexSNESComputeBoundaryFEM`. So a callback that reads the velocity
on a driven boundary (e.g. a lid) sees stale values there. Measured: Helmholtz `ē`
self-consistency ~35 % error, concentrated at the driven boundary. The pressure gauge is
unaffected (pressure carries no Dirichlet DOFs).

## The fix

Make the dispatcher's scatter mirror the **post-solve copy-back** (same file, in the
Stokes `solve()`, ~lines 6852–6910): `globalToLocal` into a parent-DM local vec →
`DMPlexSNESComputeBoundaryFEM` → per-field extraction via the **local** index sets
(`_velocity_is`, `_pressure_is`, `_multiplier_is`), instead of the `_subdict`/`globalToLocal`
shortcut.

Concretely:
1. Factor the copy-back's inline `get_local_field_is(...)` + the building of
   `_velocity_is/_pressure_is/_multiplier_is` into a method, e.g.
   `_ensure_local_field_index_sets()`, callable both from the copy-back and mid-solve
   (build on first use; they are state-independent given the section).
2. Add a Stokes override of `_scatter_global_to_fields(gvec)` that does:
   `clvec = dm.getLocalVec(); dm.globalToLocal(gvec, clvec);
    DMPlexSNESComputeBoundaryFEM(dm, clvec, NULL); _ensure_local_field_index_sets();`
   then for each field copy `clvec.getSubVector(is_).array -> var.vec.array`
   (exactly the copy-back loop), restore, then the existing cache-invalidation +
   `_refresh_auxiliary_vec`.
3. Keep the base-class `_subdict` scatter for scalar/vector solvers (no Dirichlet-DOF
   issue there in the current use cases), or generalise similarly if needed.
4. Leave the existing post-solve copy-back **unchanged** apart from calling the factored
   IS-builder (lowest risk).

## Validation

- Re-run the Helmholtz check (lid-driven, viscosity depends on a projected `ē`,
  `add_update_callback(lambda s,i: smoother.solve())`): `ē` self-consistency rel change on
  re-projection at convergence should drop from ~0.35 to `< 1e-4`. Add this as a
  `tier_a` case in `tests/test_1016_snes_update_callbacks.py`.
- `test_1016` (gauge, fires, no-op) must still pass; `test_1010` bit-identical.
- Remove the `{warning}` from `docs/advanced/solver-iteration-callbacks.md` and finalise
  the Helmholtz example/test.

## Reproducer

`~/+Simulations/snes-callbacks/validate.py` — `[gauge] PASS`, `[helmholtz] FAIL`
(rel ~0.35). After the fix the helmholtz line should PASS.
