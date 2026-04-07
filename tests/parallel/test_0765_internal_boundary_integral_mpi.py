"""
MPI regression test for internal-boundary boundary-integral ownership.

This guards against rank-dependent over/under-assembly caused by
ghost/internal facet handling in PETSc boundary assembly paths.
"""

import math
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


@pytest.mark.mpi(min_size=2)
def test_deformed_spherical_shell_boundary_area_parallel():
    """
    Boundary integrals must remain valid after coordinate deformation in MPI.

    This specifically guards the BdIntegral path when some ranks have no local
    entities for the requested boundary after mesh deformation.
    """

    mesh = uw.meshing.SphericalShell(
        radiusOuter=1.0,
        radiusInner=0.5,
        cellSize=1.0 / 4.0,
        degree=1,
        qdegree=2,
    )
    uw.discretisation.MeshVariable("T_spherical_deformed_bd", mesh, 1, degree=1, continuous=True)

    coords = np.asarray(mesh.X.coords, dtype=np.float64).copy()
    radii = np.linalg.norm(coords, axis=1)
    thickness = 0.5
    t = (radii - 0.5) / thickness
    a = math.log(2.0)
    mapped = (np.exp(a * t) - 1.0) / (math.exp(a) - 1.0)
    new_radii = 0.5 + thickness * mapped
    mesh._deform_mesh(coords * (new_radii / radii)[:, None])

    lower = float(uw.maths.BdIntegral(mesh=mesh, fn=1.0, boundary="Lower").evaluate())
    upper = float(uw.maths.BdIntegral(mesh=mesh, fn=1.0, boundary="Upper").evaluate())

    expected_lower = 4.0 * math.pi * 0.5**2
    expected_upper = 4.0 * math.pi * 1.0**2

    rel_err_lower = abs(lower - expected_lower) / expected_lower
    rel_err_upper = abs(upper - expected_upper) / expected_upper

    assert rel_err_lower < 5.0e-2, (
        f"Deformed lower area rel_err={rel_err_lower:.3e}, value={lower}, expected={expected_lower}"
    )
    assert rel_err_upper < 5.0e-2, (
        f"Deformed upper area rel_err={rel_err_upper:.3e}, value={upper}, expected={expected_upper}"
    )

    gathered_lower = uw.mpi.comm.allgather(lower)
    gathered_upper = uw.mpi.comm.allgather(upper)
    assert max(gathered_lower) - min(gathered_lower) < 1.0e-12, (
        f"Rank mismatch in lower-boundary integral values: {gathered_lower}"
    )
    assert max(gathered_upper) - min(gathered_upper) < 1.0e-12, (
        f"Rank mismatch in upper-boundary integral values: {gathered_upper}"
    )
