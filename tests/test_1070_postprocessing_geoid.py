import math

import numpy as np
import pytest
import sympy
import underworld3 as uw


pytestmark = pytest.mark.level_2


def test_spherical_shell_postprocessing_is_public():
    assert hasattr(uw, "postprocessing")
    assert uw.postprocessing.__all__ == ["geoid"]
    assert hasattr(uw.postprocessing.geoid, "spherical_shell_geoid_response")
    assert hasattr(uw.postprocessing.geoid, "spherical_shell_self_gravity_response")
    assert hasattr(uw.postprocessing.geoid, "spherical_shell_response_from_rotated_stokes")


def test_spherical_shell_geoid_response_matches_direct_formula_and_load():
    radius_inner = 0.55
    radius_outer = 1.0
    rint = 0.775
    degree = 2
    internal_load_coefficient = 1.75
    surface_topography = 0.3918264884592169
    cmb_topography = 0.2653834107104151

    response = uw.postprocessing.geoid.spherical_shell_geoid_response(
        radius_inner=radius_inner,
        radius_outer=radius_outer,
        harmonic_degree=degree,
        surface_topography_coefficient=surface_topography,
        cmb_topography_coefficient=cmb_topography,
        internal_load_radius=rint,
        internal_load_coefficient=internal_load_coefficient,
    )

    denominator = 2 * degree + 1
    surface_expected = (
        radius_inner * (radius_inner / radius_outer) ** (degree + 1) * cmb_topography
        + radius_outer * surface_topography
        - internal_load_coefficient * rint * (rint / radius_outer) ** (degree + 1)
    ) / denominator
    cmb_expected = (
        radius_inner * cmb_topography
        + radius_outer * (radius_inner / radius_outer) ** degree * surface_topography
        - internal_load_coefficient * rint * (radius_inner / rint) ** degree
    ) / denominator

    assert math.isclose(response.surface_geoid, surface_expected)
    assert math.isclose(response.cmb_geoid, cmb_expected)


def test_spherical_shell_geoid_response_without_internal_load():
    response = uw.postprocessing.geoid.spherical_shell_geoid_response(
        radius_inner=0.55,
        radius_outer=1.0,
        harmonic_degree=3,
        surface_topography_coefficient=0.4,
        cmb_topography_coefficient=-0.2,
    )

    denominator = 7.0
    expected = np.array(
        [
            (0.4 + 0.55 * 0.55**4 * -0.2) / denominator,
            (0.55**3 * 0.4 + 0.55 * -0.2) / denominator,
        ]
    )
    assert np.allclose([response.surface_geoid, response.cmb_geoid], expected)


def test_spherical_shell_self_gravity_response_matches_direct_solve():
    surface_topography = 0.3918264884592169
    cmb_topography = 0.2653834107104151
    response = uw.postprocessing.geoid.spherical_shell_self_gravity_response(
        radius_inner=0.55,
        radius_outer=1.0,
        harmonic_degree=2,
        surface_topography_coefficient=surface_topography,
        cmb_topography_coefficient=cmb_topography,
        internal_load_radius=0.775,
        internal_load_coefficient=1.0,
        surface_density_contrast=3300.0,
        cmb_density_contrast=5400.0,
        planet_radius=6370000.0,
        gravity=9.8,
        gravitational_constant=6.67e-11,
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


def test_spherical_shell_geoid_supports_degree_zero():
    response = uw.postprocessing.geoid.spherical_shell_geoid_response(
        radius_inner=0.55,
        radius_outer=1.0,
        harmonic_degree=0,
        surface_topography_coefficient=0.4,
        cmb_topography_coefficient=-0.2,
    )

    assert math.isclose(response.surface_geoid, 0.4 - 0.55**2 * 0.2)
    assert math.isclose(response.cmb_geoid, 0.4 - 0.55 * 0.2)


@pytest.mark.parametrize("harmonic_degree", [-1, 2.0, True])
def test_spherical_shell_geoid_rejects_invalid_harmonic_degree(harmonic_degree):
    error = TypeError if isinstance(harmonic_degree, (float, bool)) else ValueError
    with pytest.raises(error):
        uw.postprocessing.geoid.spherical_shell_geoid_response(
            radius_inner=0.55,
            radius_outer=1.0,
            harmonic_degree=harmonic_degree,
            surface_topography_coefficient=0.4,
            cmb_topography_coefficient=0.7,
        )


def test_rotated_adapter_rejects_degree_zero():
    with pytest.raises(ValueError, match="requires harmonic_degree >= 1"):
        uw.postprocessing.geoid.spherical_shell_response_from_rotated_stokes(
            stokes=object(),
            radius_inner=0.55,
            radius_outer=1.0,
            harmonic_degree=0,
        )


def test_internal_load_requires_a_radius():
    with pytest.raises(ValueError, match="internal_load_radius is required"):
        uw.postprocessing.geoid.spherical_shell_geoid_response(
            radius_inner=0.55,
            radius_outer=1.0,
            harmonic_degree=2,
            surface_topography_coefficient=0.4,
            cmb_topography_coefficient=0.7,
            internal_load_coefficient=1.0,
        )


def test_rotated_adapter_requires_explicit_self_gravity_parameters():
    with pytest.raises(ValueError, match="Self-gravity requires explicit values"):
        uw.postprocessing.geoid.spherical_shell_response_from_rotated_stokes(
            stokes=object(),
            radius_inner=0.55,
            radius_outer=1.0,
            harmonic_degree=2,
            include_self_gravity=True,
        )


def test_rotated_adapter_rejects_unknown_projection():
    with pytest.raises(ValueError, match="projection must be"):
        uw.postprocessing.geoid.spherical_shell_response_from_rotated_stokes(
            stokes=object(),
            radius_inner=0.55,
            radius_outer=1.0,
            harmonic_degree=2,
            projection="nodal",
        )


def test_rotated_stokes_adapter_matches_zhong_table_2():
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
    velocity = uw.discretisation.MeshVariable("U_geoid", mesh, mesh.dim, degree=2)
    pressure = uw.discretisation.MeshVariable("P_geoid", mesh, 1, degree=1, continuous=True)
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
    reaction_response = uw.postprocessing.geoid.spherical_shell_response_from_rotated_stokes(
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
        projection="reaction",
    )

    assert np.isclose(response.surface_topography, 0.41920, rtol=0.10)
    assert np.isclose(response.cmb_topography, 0.77060, rtol=0.10)
    assert np.isclose(response.surface_geoid, 0.02579, rtol=0.10)
    assert np.isclose(response.cmb_geoid, 0.03206, rtol=0.10)
    assert np.isclose(response.self_gravity.surface_topography, 0.49980, rtol=0.10)
    assert np.isclose(response.self_gravity.cmb_topography, 0.93130, rtol=0.10)
    assert np.isclose(response.self_gravity.surface_geoid, 0.04486, rtol=0.10)
    assert np.isclose(response.self_gravity.cmb_geoid, 0.05461, rtol=0.10)
    assert np.isclose(reaction_response.surface_topography, 0.41920, rtol=0.03)
    assert np.isclose(reaction_response.cmb_topography, 0.77060, rtol=0.03)
    assert np.isclose(reaction_response.surface_geoid, 0.02579, rtol=0.03)
    assert np.isclose(reaction_response.cmb_geoid, 0.03206, rtol=0.03)
