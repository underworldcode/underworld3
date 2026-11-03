"""Opt-in spherical-gap hydrostatic and geometric checks; no stability claim."""

import os
from enum import Enum

import numpy as np
import pytest
import sympy as sp
from petsc4py import PETSc
import underworld3 as uw

from underworld3.utilities._spherical_stokes_gap import _create_matrix, _source_values

pytestmark = pytest.mark.level_2


@pytest.fixture(scope="module")
def shell():
    if "UW_SPHERICAL_GAP_TEST_MESH" in os.environ:
        class Boundaries(Enum):
            Lower = 11
            Upper = 12
            Centre = 1
        return uw.discretisation.Mesh(
            os.environ["UW_SPHERICAL_GAP_TEST_MESH"], degree=1, qdegree=2,
            coordinate_system_type=uw.coordinates.CoordinateSystemType.SPHERICAL,
            useMultipleTags=True, useRegions=True, markVertices=True,
            boundaries=Boundaries, boundary_normals=None,
        )
    return uw.meshing.SphericalShell(
        radiusInner=0.55, radiusOuter=1, cellSize=1/8, degree=1, qdegree=2,
    )


@pytest.mark.tier_b
@pytest.mark.parametrize("zero_slip", [False, True])
def test_constant_lift_hydrostatic_gauge_and_physical_reactions(shell, zero_slip):
    velocity = uw.discretisation.MeshVariable("GapHydroVelocity", shell, 3, degree=2)
    pressure = uw.discretisation.MeshVariable("GapHydroPressure", shell, 1, degree=1)
    stokes = uw.systems.Stokes(shell, velocityField=velocity, pressureField=pressure)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.viscosity = 1
    stokes.bodyforce = sp.zeros(1, 3)
    polyhedral_volume = uw.maths.Integral(shell, sp.Integer(1)).evaluate()
    solve = stokes.solve_spherical_gap_zero_slip if zero_slip else stokes.solve_spherical_gap
    # MUMPS' default workspace allowance exhausted INFOG(1)=-9 in MPI8;
    # reserve 100% extra factor workspace explicitly without changing equations.
    result = solve(
        inner_boundary="Lower", outer_boundary="Upper",
        radius_inner=0.55, radius_outer=1, viscosity=1,
        pressure_jump=sp.Integer(1), pressure_jump_integral=polyhedral_volume,
        gap_bodyforce=sp.zeros(1, 3), volume_force=None, quadrature=6, radial_quadrature=7,
        linear_solver_options={"ksp_type": "preonly", "pc_type": "lu",
                               "pc_factor_mat_solver_type": "mumps",
                               "mat_mumps_icntl_14": 100},
        batch_size=128,
    )
    physical_volume = 4*np.pi*(1-0.55**3)/3
    np.testing.assert_allclose(result.checks["physical_volume"], physical_volume, rtol=1e-9)
    assert max(result.checks["null_relative_residuals"]) < 1e-10
    assert len(result.checks["null_relative_residuals"]) == (1 if zero_slip else 4)
    assert abs(result.checks["physical_pressure_integral"]) < physical_volume*1e-11
    velocity_rms = np.sqrt(abs(uw.maths.Integral(shell, velocity.sym.dot(velocity.sym)).evaluate())/physical_volume)
    pressure_rms = np.sqrt(abs(uw.maths.Integral(shell, (pressure.sym[0]+1)**2).evaluate())/physical_volume)
    assert velocity_rms < 1e-8
    assert pressure_rms < 1e-7
    assert result.checks["relative_residual"] < 1e-8
    normal_rms = np.sqrt(abs(uw.maths.Integral(shell, result.normal_reaction.sym[0]**2).evaluate())/physical_volume)
    tangent_rms = np.sqrt(abs(uw.maths.Integral(shell, result.tangential_reaction.sym.dot(result.tangential_reaction.sym)).evaluate())/physical_volume)
    assert normal_rms < 1e-7
    assert tangent_rms < 1e-7
    checks = result.checks
    assert checks["surface_quadrature_order"] == 6
    assert checks["radial_quadrature_order"] == 7
    assert checks["curtain_radial_quadrature_order"] == 4
    assert checks["matrix_dynamic_allocations"] == 0
    assert checks["matrix_mallocs_after_assembly"] == checks["matrix_mallocs_preallocation"]
    assert checks["matrix_nz_used"] == checks["matrix_nz_allocated"] >= checks["unknowns"]
    assert np.isfinite(checks["matrix_memory_bytes"]) and checks["matrix_memory_bytes"] >= 0
    for stage in ("geometry", "graph", "numeric_assembly", "algebraic_preparation", "setup", "solve"):
        assert np.isfinite(checks[f"{stage}_seconds"]) and checks[f"{stage}_seconds"] >= 0


@pytest.mark.tier_b
def test_exact_distributed_graph_rejects_an_unplanned_owned_connection():
    matrix = _create_matrix(
        2*uw.mpi.size, np.empty((0, 30), dtype=PETSc.IntType),
        np.empty((0, 4), dtype=PETSc.IntType), [], [], PETSc.COMM_WORLD,
    )
    try:
        before = matrix.getInfo(PETSc.Mat.InfoType.LOCAL)
        row, _ = matrix.getOwnershipRange()
        matrix.setValue(row, row, 1)
        matrix.assemble()
        after = matrix.getInfo(PETSc.Mat.InfoType.LOCAL)
        assert after["mallocs"] == before["mallocs"]
        assert after["nz_used"] == after["nz_allocated"] == 2
        with pytest.raises(PETSc.Error) as error:
            matrix.setValue(row, row+1, 1)
        # PETSC_ERR_ARG_OUTOFRANGE: both indices are valid, but the connection
        # lies outside the declared graph. UW's handler prints details separately.
        assert error.value.ierr == 63
    finally:
        matrix.destroy()


@pytest.mark.tier_b
def test_dg0_prescribed_jump_force_uses_each_cell_constant(shell):
    force = uw.discretisation.MeshVariable("GapDG0Force", shell, 3, degree=0, continuous=False)
    coordinates = np.asarray(force.coords)
    force.array[:, 0, :] = (1+coordinates[:, :1])*np.array([1., 2., -3.])
    rows = shell._cell_node_indices(1, True)
    vertices = np.asarray(shell._get_coords_for_basis(1, True))[rows]
    cells = np.flatnonzero(shell._get_owned_cells_mask())
    points = np.repeat(vertices[cells].mean(axis=1)[:, None], 5, axis=1)
    values = _source_values(force.sym, vertices, points, cells, shell)
    force_rows = shell._cell_node_indices(0, False)[cells, 0]
    expected = np.broadcast_to(np.asarray(force.array)[force_rows, 0, :][:, None], points.shape)
    np.testing.assert_array_equal(values, expected)
