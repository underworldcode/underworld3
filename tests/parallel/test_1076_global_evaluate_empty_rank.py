#!/usr/bin/env python3
"""global_evaluate must not deadlock when a rank receives no query points (#611).

The cell-location policy (`mesh._hint_is_authoritative`) decides whether the
barycentric hint may bypass PETSc's `DMLocatePoints`. `DMLocatePoints` is
COLLECTIVE on the mesh DM communicator, so every rank has to reach the same
verdict — and the policy is a mesh capability, so they do.

It was then thrown away in transit. `CachedDMInterpolationInfo.create_structure`
had no hint array to pass when a rank held zero points, and passed
`hintAuthoritative = 0` hardcoded along with the NULL. That rank alone took the
`DMLocatePoints` branch and blocked inside `DMGetBoundingBox -> MPI_Allreduce`,
while its peers bypassed and ran on to `DMSwarmMigrate -> MPI_Comm_dup`.

Measured before the fix, at np=4: three ranks in `MPI_Comm_dup`, one in
`MPI_Allreduce`, no progress in 900 s on a 300-point query.

The trigger is a query set that leaves some rank empty. Biasing every point into
x > 0.5 does it at np=4 — one rank owns only x < 0.5 — but not at np=2, where
the coarser partition leaves both ranks straddling the split. That is exactly
why this hid at np=2 and appeared at np=4.
"""
import numpy as np
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


def _mesh_and_field(tag):
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0))
    field = uw.discretisation.MeshVariable(tag, mesh, mesh.dim, degree=2)
    return mesh, field


def test_global_evaluate_with_points_only_on_some_ranks():
    """The regression. Every point in x > 0.5, so some rank owns none of them."""
    _mesh, field = _mesh_and_field("u1076a")
    rng = np.random.default_rng(42)
    coords = rng.random((300, 2))
    coords[:, 0] = 0.5 + 0.5 * coords[:, 0]

    result = uw.function.global_evaluate(field.sym, coords)
    assert result.shape[0] == 300, (
        f"rank {uw.mpi.rank}: expected 300 results, got {result.shape[0]}")


def test_global_evaluate_with_points_everywhere_still_works():
    """Negative control.

    The unbiased query already passed before the fix, so if this ever fails the
    change has broken the ordinary path rather than repaired the empty-rank one.
    """
    _mesh, field = _mesh_and_field("u1076b")
    rng = np.random.default_rng(42)
    coords = rng.random((300, 2))

    result = uw.function.global_evaluate(field.sym, coords)
    assert result.shape[0] == 300, (
        f"rank {uw.mpi.rank}: expected 300 results, got {result.shape[0]}")
