# PETSc Patch: Internal Boundary Ownership Consistency

## Patch file

- `plexfem-internal-boundary-ownership-fix.patch`

## Scope

File touched in PETSc:

- `src/dm/impls/plex/plexfem.c`

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
