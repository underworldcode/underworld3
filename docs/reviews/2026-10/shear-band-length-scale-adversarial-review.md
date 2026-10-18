# Shear-band length-scale branch: adversarial review record

Branch `feature/shear-band-length-scale` at `acb3b642`, against `origin/development`
(18 files). Four independent reviewers, each against the installed branch build, with a
base build (development plus the unrelated #827 fix) for comparison: semantic
correctness, numerics against the base, parallel safety, and tests / style / contracts
/ docs. Fixes in the commit that carries this record.

## What the branch does

- Coupled volume fields in Stokes (`add_coupled_field`), grouped with velocity in split 0
  of the Schur fieldsplit; `remove_coupled_field`.
- Multi-field custom multigrid transfers (`CustomMGHierarchy._build_multifield`).
- Rotated free-slip and fault contact carrying coupled fields, with each rigid-rotation
  null mode completed on the coupled fields from the operator.
- The `Micromorphic` yield-law regularisation as a constitutive option on ViscoPlastic,
  VEP, TI-VEP and MultiMaterial.
- The automatic cold Picard warm-up reaching the rotated / fault-contact loop.

Design: `docs/developer/design/micromorphic-plasticity.md`; reference:
`docs/developer/subsystems/saddle-point-field-groups.md`.

## Findings and their state

Blocking:
- **The adjoint of a coupled solve was wrong and recorded as supported.** The kernel
  route transposes only the solver's own volume kernels; a two-way chi problem gave
  relative error 0.20 against an explicit J^T solve (residual 15.6). Fixed: coupled
  fields and multipliers take the matrix route (`_adjoint_by_kernels`). Test fails on
  the old rule (residual ratio 443/28), passes at < 1e-8.
- **The rotated warm-up drove a working VP solve (two rotated walls) into a linear
  solve that did not end.** Newton converged to rel |F| 5.3e-8 against 1e-8; the next
  increment's linear solve, asked for a 1e-8 relative reduction of a tiny right-hand
  side, multiplied its nested caps. Pre-existing fragility, reached by the new path.
  Fixed: each increment's linear solve has an absolute floor of half the nonlinear
  target (inexact Newton). The case converges in 13 iterations (19 s).

High / medium:
- Rotated constraints with two or more coupled fields raised on every rank
  (`IS.concatenate` is not in petsc4py). Fixed: one index set from the DM; test serial
  and np 2 / 3.
- `Micromorphic.chi` was created lazily (a collective) and named after the model, so a
  replacement reused the old field (wrong degree, shared data). Fixed: created when the
  regularisation is attached, on every rank, uniquely named; tested.
- The chi equation was frozen at registration; a changed mask (normalised level sets,
  the model used on its own) left the source and the shift disagreeing. Fixed: the
  model's request is the equation at every solve (`_set_coupled_equation`, normalised
  like `add_coupled_field`, so an unchanged equation costs no rewire); the mask is
  passed by MultiMaterial and cleared on a standalone model; tested.
- `describe()` omitted coupled fields, their equations and their conditions. Fixed;
  `coupled_essential` joins `_constraint_mechanisms`; tested.
- The rotated warm-up was not recorded in the solve report. Fixed
  (`picard_warmup_its`); tested.
- A test of the per-family shift checked the helper, not the model's flux. Rewritten to
  check each family's flux (VEP and TI-VEP with elastic parameters).

Low:
- `picard_steps` over-counted on a Picard-phase exit; `_velocity_split_is` rebuilt a
  sub-DM per Newton iteration and leaked it (cached against the solver's `_subdict`,
  which every rank rebuilds together); rank-local raises before a collective in
  `_split_index_maps` (reduced first); a rank with no cell on an independent coarse
  level divided by zero, then could not triangulate (components reduced across ranks,
  cross-partition build chosen collectively; pre-existing); chi registered when the
  yield is disabled; the TI-VEP split model's lagged path dropped the shift; stale
  guide, design-doc and docstring text; `.data` in new tests; docstring sections;
  `Micromorphic` added to the API docs; the snapshot test skips in parallel (per-rank
  `tmp_path`).
- Pre-existing, found on the way: `test_0016` / `test_0017` failed whenever
  `test_1057` (which defines solver subclasses) ran first in the process. The contract
  tests now enumerate library classes only.

## Attacks that failed

- Taylor test of residual against Jacobian on the option path: O(h^2) for VP (slopes
  2.01), VEP with history (1.99 / 2.00), TI-VEP (2.01 / 2.00).
- Without a regularisation the generated residual and Jacobian, and v, p, are bit for
  bit the base's: VP, Drucker-Prager, VEP, TI-VEP, constrained Stokes (multipliers
  included), single-field custom FMG (transfer CSR arrays identical at np 1 and 2).
- Units: a units model reproduces the non-dimensional chi to every digit.
- Parallel: results partition-independent to solver tolerance at np 1-3 (Cosserat
  annulus 2e-10, custom FMG with a coupled field 1.2e-10, rotated cold micromorphic
  2e-14); the coupled-mode decision uses a global norm; the warm-up decision is
  rank-uniform.
- The regression tests fail on their bugs (velocity-only split, velocity-only rotation
  modes, old warm-up planning, old adjoint rule).

## Benchmark (rotated and fault-contact, cold solves, branch against base)

26 cases from the numerics review, rerun on the fixed build. Newton iterations never
above the base; Krylov counts equal or lower; time equal or lower but for noise. Where
the base converges the answers agree to solver tolerance (<= 1.6e-7).

| Case | Base nl / Krylov / s | Branch nl / Krylov / s |
|---|---|---|
| power-law box, picard=3 | 17 / 17 / 5.3 | 8 / 8 / 2.1 |
| VP shear box | 9 / 45 / 150 | 7 / 26 / 68 |
| VP shear box, picard=3 | 24 / 93 / 131 | 8 / 30 / 75 |
| VP shear box, continuation | 24 / 93 / 186 | 24 / 67 / 129 |
| DP shear box (unconverged in both) | 6 / 191 / 794 | 4 / 90 / 202 |
| DP annulus (unconverged in both) | 9 / 33 / 204 | 9 / 27 / 136 |
| fault + power-law, picard=2 | refused | runs (unconverged, as without picard) |

The unconverged plastic cases (line-search stall) are the same in both builds and are
outside this branch.

## Suites

Every warm-up, rotated and fault-contact test file plus the coupled, multigrid,
constitutive, adjoint and description suites: 535 passed before the ordering fix, the
2 failures being that pre-existing ordering; the parallel wrappers (`test_1070`
rotated, `test_1075`, `test_1078`) at np 2 and 3.
