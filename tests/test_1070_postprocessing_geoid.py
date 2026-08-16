import math

import numpy as np
import pytest
import sympy
import underworld3 as uw


pytestmark = pytest.mark.level_2


def test_zhong2008_postprocessing_is_public():
    assert hasattr(uw, "postprocessing")
    assert hasattr(uw.postprocessing, "zhong2008_geoid_response")
    assert hasattr(uw.postprocessing, "zhong2008_self_gravity_response")
    assert hasattr(uw.postprocessing, "zhong2008_response_from_rotated_stokes")


def test_zhong2008_geoid_response_matches_direct_formula_and_load_scale():
    radius_inner = 0.55
    radius_outer = 1.0
    radius_internal = 0.775
    degree = 2
    load_scale = 1.75
    surface_topography = 0.3918264884592169
    cmb_topography = 0.2653834107104151

    response = uw.postprocessing.zhong2008_geoid_response(
        radius_inner=radius_inner,
        radius_outer=radius_outer,
        radius_internal=radius_internal,
        harmonic_degree=degree,
        surface_topography_coefficient=surface_topography,
        cmb_topography_coefficient=cmb_topography,
        load_scale=load_scale,
    )

    denominator = 2 * degree + 1
    surface_expected = (
        radius_inner * (radius_inner / radius_outer) ** (degree + 1) * cmb_topography
        + radius_outer * surface_topography
        - load_scale
        * radius_internal
        * (radius_internal / radius_outer) ** (degree + 1)
    ) / denominator
    cmb_expected = (
        radius_inner * cmb_topography
        + radius_outer * (radius_inner / radius_outer) ** degree * surface_topography
        - load_scale * radius_internal * (radius_inner / radius_internal) ** degree
    ) / denominator

    assert math.isclose(response.surface_geoid, surface_expected)
    assert math.isclose(response.cmb_geoid, cmb_expected)


def test_zhong2008_self_gravity_response_matches_direct_solve():
    surface_topography = 0.3918264884592169
    cmb_topography = 0.2653834107104151
    response = uw.postprocessing.zhong2008_self_gravity_response(
        radius_inner=0.55,
        radius_outer=1.0,
        radius_internal=0.775,
        harmonic_degree=2,
        surface_topography_coefficient=surface_topography,
        cmb_topography_coefficient=cmb_topography,
        load_scale=1.0,
        surface_density_contrast=3300.0,
        cmb_density_contrast=5400.0,
    )

    q_surface = 4.0 * np.pi * 6.67e-11 * 6370000.0 * 3300.0 / 9.8
    q_cmb = 4.0 * np.pi * 6.67e-11 * 6370000.0 * 5400.0 / 9.8
    denominator = 5.0
    operator = np.array(
        [
            [1.0 / denominator, 0.55 * 0.55**3 / denominator],
            [0.55**2 / denominator, 0.55 / denominator],
        ]
    )
    load = np.array(
        [
            -0.775 * 0.775**3 / denominator,
            -0.775 * (0.55 / 0.775) ** 2 / denominator,
        ]
    )
    q_matrix = np.diag([q_surface, q_cmb])
    expected_topography = np.linalg.solve(
        np.eye(2) - q_matrix @ operator,
        np.array([surface_topography, cmb_topography]) + q_matrix @ load,
    )
    expected_geoid = operator @ expected_topography + load

    assert math.isclose(response.q_surface, q_surface)
    assert math.isclose(response.q_cmb, q_cmb)
    assert np.allclose(
        [response.surface_topography, response.cmb_topography],
        expected_topography,
    )
    assert np.allclose(
        [response.surface_geoid, response.cmb_geoid],
        expected_geoid,
    )


@pytest.mark.parametrize("harmonic_degree", [0, -1, 2.0, True])
def test_zhong2008_geoid_rejects_invalid_harmonic_degree(harmonic_degree):
    error = TypeError if isinstance(harmonic_degree, (float, bool)) else ValueError
    with pytest.raises(error):
        uw.postprocessing.zhong2008_geoid_response(
            radius_inner=0.55,
            radius_outer=1.0,
            radius_internal=0.775,
            harmonic_degree=harmonic_degree,
            surface_topography_coefficient=0.4,
            cmb_topography_coefficient=0.7,
        )


def test_zhong2008_rotated_stokes_adapter_matches_table_2():
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
    velocity = uw.discretisation.MeshVariable("U_geoid", mesh, mesh.dim, degree=2)
    pressure = uw.discretisation.MeshVariable(
        "P_geoid", mesh, 1, degree=1, continuous=True
    )
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

    response = uw.postprocessing.zhong2008_response_from_rotated_stokes(
        stokes=stokes,
        radius_inner=radius_inner,
        radius_outer=radius_outer,
        radius_internal=radius_internal,
        harmonic_degree=2,
        load_scale=1.0,
        include_self_gravity=True,
    )

    assert np.isclose(response.surface_topography, 0.41920, rtol=0.10)
    assert np.isclose(response.cmb_topography, 0.77060, rtol=0.10)
    assert np.isclose(response.surface_geoid, 0.02579, rtol=0.10)
    assert np.isclose(response.cmb_geoid, 0.03206, rtol=0.10)
    assert np.isclose(response.self_gravity.surface_topography, 0.49980, rtol=0.10)
    assert np.isclose(response.self_gravity.cmb_topography, 0.93130, rtol=0.10)
    assert np.isclose(response.self_gravity.surface_geoid, 0.04486, rtol=0.10)
    assert np.isclose(response.self_gravity.cmb_geoid, 0.05461, rtol=0.10)
