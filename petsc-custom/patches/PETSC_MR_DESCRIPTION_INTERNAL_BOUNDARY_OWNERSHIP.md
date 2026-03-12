# PETSc Patch: Internal Boundary Ownership Consistency

## Patch file

- `plexfem-internal-boundary-ownership-fix.patch`

## Scope

File touched in PETSc:

- `src/dm/impls/plex/plexfem.c`

## PETSc Version Details

- Upstream baseline used for patch generation:
  - PETSc repository branch: `release`
  - Local baseline commit: `5a03b95372f` (before applying this patch)
- Runtime/tested configuration:
  - PETSc: `3.24.5` (release build)
  - petsc4py: `3.24.5`
  - `PETSC_ARCH`: `petsc-4-uw-openmpi`
  - `PETSC_DIR`: `.../underworld3/petsc-custom/petsc`

## Problem

Internal-boundary contributions in parallel were rank-dependent for natural BC / boundary integral workflows.
Symptoms included:

- Different `v_l2`/`p_l2` across MPI sizes for the same problem setup.
- Different boundary callback totals between ranks/partitions.

Root causes:

1. Boundary facet sets were not consistently filtered to owned facets in all boundary assembly paths.
2. `DMPlexComputeBdResidualSingleByKey` always used `support[0]` during closure gather/scatter, which is not part-consistent when `key.part != 0`.
3. Part filtering for boundary points needed to guard against insufficient support for `part > 0`.

## Fix Summary

The patch makes boundary assembly ownership-consistent and part-consistent:

1. Filter SF leaf (ghost) facets from `facetIS` using `ISDifference(...)` in:
   - `DMPlexComputeBdIntegral`
   - `DMPlexComputeBdResidual_Internal`
   - `DMPlexComputeBdJacobian_Internal`

2. In `DMPlexComputeBdResidualSingleByKey`:
   - For `key.part > 0`, filter points with `supportSize <= key.part`.
   - Use `support[key.part]` (not hardcoded `support[0]`) for:
     - local closure reads (`locX`, `locX_t`)
     - auxiliary closure lookup
     - local residual insertion (`DMPlexVecSetClosure`)
   - Add `PetscCheck(supportSize > key.part, ...)` guards before use.

## Why this resolves the bug

Each physical boundary facet is assembled once (owned rank only), and each part uses the correct adjacent cell side. This removes over/under-assembly and residual/Jacobian side mismatches that previously depended on partition topology.

## Validation

Validated with Underworld3 annulus internal-boundary cases:

- `Ex_Stokes_Kramer_latest.py` with natural BC, `np=1,2,4,8`:
  - stable velocity L2 around `2.32345982e-03`
  - no branch split at `np=8`

- Boundary callback totals (`bdres`/`bdjac`) match between `np=7` and `np=8` after patch.

- A UW MPI regression test was added:
  - `tests/parallel/test_0765_internal_boundary_integral_mpi.py`
  - checks annulus internal-boundary circumference in parallel against analytic value.

## PETSc-Only Reproducer (For Upstream MR)

Goal: provide a PETSc-native (no UW dependency) reproducer under
`src/dm/impls/plex/tests` that demonstrates rank-dependent internal-boundary
assembly before this patch and rank-invariant behavior after.

### Proposed test shape

- New PETSc test source:
  - `src/dm/impls/plex/tests/ex_internalbd_ownership.c` (name illustrative)
- Setup:
  - create a distributed 2D DMPlex box mesh
  - define a facet `DMLabel` for an interior line (internal boundary)
  - add a simple natural-boundary contribution over that internal label
  - compute a deterministic scalar diagnostic (boundary integral or derived norm)
- Runs:
  - `nsize: 1` and `nsize: 4` (or `8`)
  - compare scalar outputs with strict tolerance

### Why C test (not petsc4py)

`petsc4py` does not currently expose the needed internal DMPlex boundary
assembly entry points directly enough for a clean reproducer of this path.
Upstream PETSc C tests are the correct home for long-term regression coverage.

## Exact Before / After Numbers

The following were measured on the same mesh/problem configuration
(`internal_bc_rank_reproducer.py`, cellsize `0.03125`):

### Before fix (rank-dependent branch split)

- `np=7`: `v_l2_int=2.3234597202763355e-03`
- `np=8`: `v_l2_int=6.2729732160772709e-03`

### After fix (rank-invariant)

- `np=7`: `v_l2_int=2.3234597202763251e-03`
- `np=8`: `v_l2_int=2.3234597202740370e-03`

### Tolerance criteria

- Rank-invariance target for this case:
  - relative difference between MPI sizes `< 1e-10` for `v_l2_int` in this benchmark family
- Practical acceptance in PETSc test harness:
  - choose tolerance robust to platform/compiler variation (e.g. `1e-9` to `1e-8`)

## Changed-Function Map

Patch updates are limited to:

- `DMPlexComputeBdIntegral`
- `DMPlexComputeBdResidual_Internal`
- `DMPlexComputeBdJacobian_Internal`
- `DMPlexComputeBdResidualSingleByKey`

No public API signatures were changed.

## Rationale: `support[key.part]` vs `support[0]`

Boundary weak-form callbacks can be registered with a nonzero `part` to target
a specific side of a facet. Using `support[0]` unconditionally can pull
closures from the wrong adjacent cell and produce side-inconsistent assembly.
Therefore `support[key.part]` must be used, with explicit support-size guards,
and points invalid for `part > 0` must be filtered.

## Ownership Model Note (SF leaves)

In parallel DMPlex, ghost facets appear as SF leaves on non-owning ranks.
If these are assembled in boundary paths, shared internal facets can
contribute multiple times and then be summed by MPI reductions / ADD_VALUES.
Filtering SF leaves ensures each physical facet contributes exactly once
(owner rank only), removing duplicate assembly.

## PETSc Regression Test Plan (Upstream)

Add a regression test under `src/dm/impls/plex/tests` with MPI coverage:

- `nsize: 1`
- `nsize: 4` (or `8`)
- check scalar diagnostic equality within tolerance
- include in PETSc test harness (`TEST` block) so CI exercises the path

Suggested command pattern once merged in PETSc tree:

```bash
make -C $PETSC_DIR check TESTDIR=src/dm/impls/plex/tests
```

Or direct:

```bash
mpirun -np 4 ./ex_internalbd_ownership [args]
```

## Version Scope

- Baseline for patch generation:
  - PETSc `release` branch, local baseline commit `5a03b95372f`
- Runtime validated in this workflow:
  - PETSc `3.24.5`
  - petsc4py `3.24.5`
  - OpenMPI `5.0.10`
- Affected code sections are structurally stable across recent PETSc releases,
  so backport applicability is expected to be broad with minimal adaptation.

## Risk / Performance

- No API changes.
- Correctness-only fix in boundary assembly ownership/part handling.
- Additional operations are lightweight (`ISDifference`, support checks) and
  expected overhead is negligible relative to FE assembly costs.

## Backport Recommendation

- Recommended target: PETSc `release` branch.
- Also suitable for older maintained branches where the same boundary assembly
  code is present (confirm with `git apply --check` and test run).

## Benchmark Script And Case

- Benchmark script (GitHub):
  - https://github.com/gthyagi/UW3_Annulus_Spherical_Benchmarks/blob/main/benchmarks/annulus/Ex_Stokes_Kramer_latest.py

- Recommended case for reproducing/validating this fix:
  - `-uw_case case1 -uw_bc_type natural`

- Example runs:

```bash
# from UW3_Annulus_Spherical_Benchmarks/benchmarks/annulus
PY=/path/to/uw/.pixi/envs/amr-dev/bin/python
SCRIPT=Ex_Stokes_Kramer_latest.py

for n in 1 2 4 8; do
  mpirun -np "$n" "$PY" "$SCRIPT" -uw_case case1 -uw_bc_type natural
done
```

## Patch Application Matrix

Use `git apply --check` before applying:

```bash
git apply --check /path/to/patch.patch
```

### Case A: old patch is not applied

- Recommended and simplest path.
- Apply only:
  - `plexfem-internal-boundary-ownership-fix.patch`

### Case B: old patch is already applied

- Do not stack both full patches.
- `plexfem-internal-boundary-ownership-fix.patch` usually overlaps with the old patch and will fail `--check`.
- Recommended options:
  1. Revert/restore `plexfem.c` to clean PETSc source, then apply only the new patch.
  2. Or create/use an incremental patch from old -> new (if you need sequential layering).

## Verification Commands

### 1) UW MPI regression test

```bash
mpirun -np 4 python -m pytest --with-mpi tests/parallel/test_0765_internal_boundary_integral_mpi.py -q
```

Expected:
- `2 passed`

### 2) Benchmark validation

```bash
# from UW3_Annulus_Spherical_Benchmarks/benchmarks/annulus
for n in 1 2 4 8; do
  mpirun -np "$n" "$PY" "$SCRIPT" -uw_case case1 -uw_bc_type natural
done
```

Expected:
- `Relative velocity L2 error` should be stable across `np=1,2,4,8`
- Typical value around `2.32345982e-03` for this case

## Notes

- This patch supersedes the narrower `plexfem-ghost-facet-fix.patch` by also addressing part-consistent residual assembly and Jacobian ownership filtering.
- Apply **only one** of these two patches.
  - Preferred: `plexfem-internal-boundary-ownership-fix.patch`
  - Legacy fallback: `plexfem-ghost-facet-fix.patch`
- Do **not** apply both patches together (overlapping hunks / duplicate logic).
