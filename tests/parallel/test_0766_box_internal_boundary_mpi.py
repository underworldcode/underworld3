"""
MPI regression test for BoxInternalBoundary construction (BF-13).

BoxInternalBoundary used to bind its ``boundaries``/``boundary_normals``
Enums only inside the rank-0 gmsh block, so every rank > 0 raised
``UnboundLocalError`` before the mesh existed and the job hung in the
subsequent collective (2026-07 audit, ``docs/reviews/2026-07/
REMEDIATION-WORKLIST.md`` BF-13). This test constructs the mesh on
2+ ranks and checks the internal-boundary integral is rank-consistent.
"""

import numpy as np
import pytest
import underworld3 as uw


pytestmark = [
    pytest.mark.level_2,
    pytest.mark.tier_a,
    pytest.mark.mpi(min_size=2),
    pytest.mark.timeout(120),
]


@pytest.mark.mpi(min_size=2)
def test_box_internal_boundary_constructs_on_all_ranks():
    """Mesh construction must succeed on every rank (regression: BF-13)."""
    mesh = uw.meshing.BoxInternalBoundary(
        elementRes=(8, 8),
        zelementRes=(4, 4),
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        zintCoord=0.5,
    )
    # The boundary Enum must be bound (and identical) on all ranks.
    names = [b.name for b in mesh.boundaries]
    gathered = uw.mpi.comm.allgather(names)
    assert all(g == gathered[0] for g in gathered), f"Rank mismatch in boundaries: {gathered}"
    assert "Internal" in names

    # PETSc integration path requires at least one variable on the mesh.
    uw.discretisation.MeshVariable("T_boxIB_mpi", mesh, 1, degree=1)

    # Internal boundary at y=0.5 across a unit box has length 1.
    value = float(uw.maths.BdIntegral(mesh=mesh, fn=1.0, boundary="Internal").evaluate())
    assert abs(value - 1.0) < 1.0e-3, f"Internal boundary length {value} != 1.0"

    gathered_vals = uw.mpi.comm.allgather(value)
    assert max(gathered_vals) - min(gathered_vals) < 1.0e-12, (
        f"Rank mismatch in integral values: {gathered_vals}"
    )
