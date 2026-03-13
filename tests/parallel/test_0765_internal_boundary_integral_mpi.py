"""
MPI regression test for internal-boundary boundary-integral ownership.

This guards against rank-dependent over/under-assembly caused by
ghost/internal facet handling in PETSc boundary assembly paths.
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


def _annulus_mesh():
    return uw.meshing.AnnulusInternalBoundary(
        radiusOuter=2.22,
        radiusInternal=2.0,
        radiusInner=1.22,
        cellSize_Inner=1.0 / 32.0,
        cellSize_Internal=(1.0 / 32.0) / 2.0,
        cellSize_Outer=1.0 / 32.0,
    )


mesh_annulus = _annulus_mesh()
# PETSc integration path requires at least one variable on the mesh.
_dummy_var = uw.discretisation.MeshVariable("T_annulus_mpi_bd", mesh_annulus, 1, degree=1)


@pytest.mark.mpi(min_size=2)
def test_internal_boundary_circumference_parallel():
    """
    Internal boundary circumference should match 2*pi*R in parallel.
    """
    value = float(uw.maths.BdIntegral(mesh=mesh_annulus, fn=1.0, boundary="Internal").evaluate())
    expected = 2.0 * np.pi * 2.0

    rel_err = abs(value - expected) / expected
    assert rel_err < 2.0e-2, f"Internal circumference rel_err={rel_err:.3e}, value={value}, expected={expected}"

    gathered = uw.mpi.comm.allgather(value)
    assert max(gathered) - min(gathered) < 1.0e-12, f"Rank mismatch in integral values: {gathered}"


@pytest.mark.mpi(min_size=2)
def test_outer_boundary_circumference_parallel():
    """
    External boundary circumference remains correct in parallel.
    """
    value = float(uw.maths.BdIntegral(mesh=mesh_annulus, fn=1.0, boundary="Upper").evaluate())
    expected = 2.0 * np.pi * 2.22

    rel_err = abs(value - expected) / expected
    assert rel_err < 2.0e-2, f"Outer circumference rel_err={rel_err:.3e}, value={value}, expected={expected}"
