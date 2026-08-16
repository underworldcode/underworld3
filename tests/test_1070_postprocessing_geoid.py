import math

import numpy as np
import pytest
import sympy
import underworld3 as uw

pytestmark = pytest.mark.level_2


def test_postprocessing_module_is_public():
    assert hasattr(uw, "postprocessing")
    assert hasattr(uw.postprocessing, "topography")
    assert hasattr(uw.postprocessing, "geoid")
    assert hasattr(uw.postprocessing, "spherical_shell_dynamic_response")
    assert hasattr(uw.systems.Stokes, "geoid")


def test_spherical_shell_geoid_response_matches_direct_formula():
    response = uw.postprocessing.geoid.spherical_shell_geoid_response(
        radius_inner=0.55,
        radius_outer=1.0,
        radius_internal=0.775,
        harmonic_degree=2,
        surface_topography=0.3918264884592169,
        cmb_topography=0.2653834107104151,
    )

    surface_expected = (
        0.55 * (0.55 / 1.0) ** 3 * 0.2653834107104151
        + 1.0 * 0.3918264884592169
        - 0.775 * (0.775 / 1.0) ** 3
    ) / 5.0
    cmb_expected = (
        0.55 * 0.2653834107104151
        + 1.0 * (0.55 / 1.0) ** 2 * 0.3918264884592169
        - 0.775 * (0.55 / 0.775) ** 2
    ) / 5.0

    assert math.isclose(response.surface_geoid, surface_expected)
    assert math.isclose(response.cmb_geoid, cmb_expected)


def test_spherical_shell_self_gravity_response_matches_direct_solve():
    response = uw.postprocessing.geoid.spherical_shell_self_gravity_response(
        radius_inner=0.55,
        radius_outer=1.0,
        radius_internal=0.775,
        harmonic_degree=2,
        surface_topography=0.3918264884592169,
        cmb_topography=0.2653834107104151,
    )

    q_surface = 4.0 * np.pi * 6.67e-11 * 6370000.0 * 3300.0 / 9.8
    q_cmb = 4.0 * np.pi * 6.67e-11 * 6370000.0 * 5400.0 / 9.8
    denominator = 5.0
    surface_b = 0.55 * (0.55 / 1.0) ** 3 / denominator
    surface_s = 1.0 / denominator
    surface_load = -0.775 * (0.775 / 1.0) ** 3 / denominator
    cmb_b = 0.55 / denominator
    cmb_s = 1.0 * (0.55 / 1.0) ** 2 / denominator
    cmb_load = -0.775 * (0.55 / 0.775) ** 2 / denominator

    matrix = np.array(
        [
            [1.0 - q_surface * surface_s, -q_surface * surface_b],
            [-q_cmb * cmb_s, 1.0 - q_cmb * cmb_b],
        ]
    )
    rhs = np.array(
        [
            0.3918264884592169 + q_surface * surface_load,
            0.2653834107104151 + q_cmb * cmb_load,
        ]
    )
    surface_topography, cmb_topography = np.linalg.solve(matrix, rhs)
    surface_geoid = (
        surface_b * cmb_topography + surface_s * surface_topography + surface_load
    )
    cmb_geoid = cmb_b * cmb_topography + cmb_s * surface_topography + cmb_load

    assert math.isclose(response.q_surface, q_surface)
    assert math.isclose(response.q_cmb, q_cmb)
    assert math.isclose(response.surface_topography, surface_topography)
    assert math.isclose(response.cmb_topography, cmb_topography)
    assert math.isclose(response.surface_geoid, surface_geoid)
    assert math.isclose(response.cmb_geoid, cmb_geoid)


def test_stokes_geoid_rejects_non_spherical_geometry():
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0, 0.0),
        maxCoords=(1.0, 1.0, 1.0),
        cellSize=0.5,
    )
    stokes = uw.systems.Stokes(mesh, degree=2)

    with pytest.raises(ValueError, match="spherical-coordinate"):
        stokes.geoid(
            radius_inner=0.55,
            radius_outer=1.0,
            radius_internal=0.775,
            harmonic_degree=2,
        )


def test_stokes_geoid_matches_dynamic_response_for_rotated_reaction():
    """The Stokes facade exactly matches the rotated-reaction implementation."""

    radius_inner = 0.55
    radius_outer = 1.0
    radius_internal = 0.775
    mesh = uw.meshing.SphericalShellInternalBoundary(
        radiusOuter=radius_outer,
        radiusInternal=radius_internal,
        radiusInner=radius_inner,
        cellSize=0.25,
        qdegree=2,
        degree=1,
    )
    velocity = uw.discretisation.MeshVariable(
        "U_rotated_geoid", mesh, mesh.dim, degree=2
    )
    pressure = uw.discretisation.MeshVariable(
        "P_rotated_geoid", mesh, 1, degree=1, continuous=True
    )
    stokes = uw.systems.Stokes(mesh, velocityField=velocity, pressureField=pressure)
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

    with pytest.raises(ValueError, match="ordered"):
        stokes.geoid(
            radius_inner=radius_outer,
            radius_outer=radius_inner,
            radius_internal=radius_internal,
            harmonic_degree=2,
        )
    with pytest.raises(TypeError, match="harmonic_degree"):
        stokes.geoid(
            radius_inner=radius_inner,
            radius_outer=radius_outer,
            radius_internal=radius_internal,
            harmonic_degree=2.0,
        )
    with pytest.raises(TypeError, match="self_gravity"):
        stokes.geoid(
            radius_inner=radius_inner,
            radius_outer=radius_outer,
            radius_internal=radius_internal,
            harmonic_degree=2,
            self_gravity="on",
        )
    with pytest.raises(RuntimeError, match="completed, converged"):
        stokes.geoid(
            radius_inner=radius_inner,
            radius_outer=radius_outer,
            radius_internal=radius_internal,
            harmonic_degree=2,
        )

    stokes.solve()

    direct_response = uw.postprocessing.spherical_shell_dynamic_response(
        stokes=stokes,
        radius_inner=radius_inner,
        radius_outer=radius_outer,
        radius_internal=radius_internal,
        harmonic_degree=2,
        include_self_gravity=True,
        topography_source="auto",
    )
    response = stokes.geoid(
        radius_inner=radius_inner,
        radius_outer=radius_outer,
        radius_internal=radius_internal,
        harmonic_degree=2,
        self_gravity=True,
    )
    response_without_self_gravity = stokes.geoid(
        radius_inner=radius_inner,
        radius_outer=radius_outer,
        radius_internal=radius_internal,
        harmonic_degree=2,
        self_gravity=False,
    )

    assert response.topography_source == "rotated_reaction"
    assert response == direct_response
    assert response_without_self_gravity.self_gravity is None
    assert (
        response_without_self_gravity.surface_topography == response.surface_topography
    )
    assert response_without_self_gravity.cmb_topography == response.cmb_topography
    assert response_without_self_gravity.surface_geoid == response.surface_geoid
    assert response_without_self_gravity.cmb_geoid == response.cmb_geoid
    assert np.isclose(response.surface_topography, 0.41920, rtol=0.10)
    assert np.isclose(response.cmb_topography, 0.77060, rtol=0.10)
    assert np.isclose(response.surface_geoid, 0.02579, rtol=0.10)
    assert np.isclose(response.cmb_geoid, 0.03206, rtol=0.10)
    assert np.isclose(response.self_gravity.surface_topography, 0.49980, rtol=0.10)
    assert np.isclose(response.self_gravity.cmb_topography, 0.93130, rtol=0.10)
    assert np.isclose(response.self_gravity.surface_geoid, 0.04486, rtol=0.10)
    assert np.isclose(response.self_gravity.cmb_geoid, 0.05461, rtol=0.10)
