import numpy as np
import pytest
import sympy
import underworld3 as uw


pytestmark = pytest.mark.level_2


def test_rotated_spherical_shell_geoid_matches_serial_reference():
    if uw.mpi.size == 1:
        pytest.skip("Run this regression with at least two MPI ranks.")

    radius_inner = 0.55
    radius_outer = 1.0
    rint = 0.775
    mesh = uw.meshing.SphericalShellInternalBoundary(
        radiusOuter=radius_outer,
        radiusInternal=rint,
        radiusInner=radius_inner,
        cellSize=0.25,
        qdegree=2,
        degree=1,
    )
    velocity = uw.discretisation.MeshVariable("U_geoid_mpi", mesh, mesh.dim, degree=2)
    pressure = uw.discretisation.MeshVariable("P_geoid_mpi", mesh, 1, degree=1, continuous=True)
    stokes = uw.systems.Stokes(
        mesh,
        velocityField=velocity,
        pressureField=pressure,
    )
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0

    theta = mesh.CoordinateSystem.xR[1]
    unit_r = mesh.CoordinateSystem.unit_e_0
    load = sympy.assoc_legendre(2, 0, sympy.cos(theta)) * unit_r
    stokes.add_natural_bc(load, "Internal")
    stokes.add_rotated_freeslip_bc(0, "Upper", normal=unit_r)
    stokes.add_rotated_freeslip_bc(0, "Lower", normal=-unit_r)
    stokes.petsc_use_nullspace = True
    stokes.petsc_options["snes_type"] = "ksponly"
    stokes.tolerance = 1.0e-5
    stokes.solve()

    response = uw.postprocessing.geoid.spherical_shell_response_from_rotated_stokes(
        stokes=stokes,
        radius_inner=radius_inner,
        radius_outer=radius_outer,
        harmonic_degree=2,
        internal_load_radius=rint,
        internal_load_coefficient=1.0,
        include_self_gravity=True,
        surface_density_contrast=3300.0,
        cmb_density_contrast=5400.0,
        planet_radius=6370000.0,
        gravity=9.8,
        gravitational_constant=6.67e-11,
    )
    values = np.array(
        [
            response.surface_topography,
            response.cmb_topography,
            response.surface_geoid,
            response.cmb_geoid,
            response.self_gravity.surface_topography,
            response.self_gravity.cmb_topography,
            response.self_gravity.surface_geoid,
            response.self_gravity.cmb_geoid,
        ]
    )

    for rank_values in uw.mpi.comm.allgather(values):
        assert np.array_equal(rank_values, values)

    zhong_table_2 = np.array(
        [0.41920, 0.77060, 0.02579, 0.03206, 0.49980, 0.93130, 0.04486, 0.05461]
    )
    assert np.allclose(values, zhong_table_2, rtol=0.10, atol=1.0e-12)
