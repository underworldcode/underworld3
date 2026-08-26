from types import SimpleNamespace

from mpi4py import MPI
import numpy as np
import pytest
import underworld3 as uw


pytestmark = [pytest.mark.level_2, pytest.mark.tier_c]

RADIUS_INNER = 1.22
RADIUS_INTERNAL = 2.0
RADIUS_OUTER = 2.22
GRAVITATIONAL_CONSTANT = 3.7


def _circle_samples(radius, coefficient, mean, wavenumber, count=256):
    theta = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    coords = radius * np.column_stack((np.cos(theta), np.sin(theta)))
    values = coefficient * np.cos(wavenumber * theta) + mean
    return coords, values


class PartitionedFakeRotatedStokes:
    """Return a disjoint subset of each synthetic boundary on every rank."""

    def __init__(self, boundary_data):
        self.mesh = SimpleNamespace(dim=2)
        self.boundary_data = boundary_data

    def boundary_normal_traction(self, boundary, mass="auto"):
        assert mass == "auto"
        coords, values = self.boundary_data[boundary]
        indices = np.arange(coords.shape[0])[MPI.COMM_WORLD.rank :: MPI.COMM_WORLD.size]
        return coords[indices], values[indices]


def _adapter_kwargs(wavenumber=2):
    return {
        "radius_inner": RADIUS_INNER,
        "radius_outer": RADIUS_OUTER,
        "wavenumber": wavenumber,
        "outer_density_contrast": 0.06,
        "inner_density_contrast": 0.09,
        "outer_reference_gravity": 1.7,
        "inner_reference_gravity": 2.4,
        "internal_load_radius": RADIUS_INTERNAL,
        "internal_surface_density_coefficient": 0.027,
        "outer_buoyancy_scale": 1.0,
        "inner_buoyancy_scale": -1.0,
        "gravitational_constant": 0.1,
    }


def _pure_gravity_kwargs(adapter_kwargs):
    return {
        key: value
        for key, value in adapter_kwargs.items()
        if key not in ("outer_buoyancy_scale", "inner_buoyancy_scale")
    }


def test_cylindrical_geoid_api_is_public():
    expected = (
        "CylindricalGravityResponse",
        "CylindricalSelfGravityResponse",
        "CylindricalAnnulusResponse",
        "cylindrical_sheet_potential_coefficient",
        "cylindrical_sheet_radial_derivative_coefficient",
        "cylindrical_annulus_potential_operator",
        "cylindrical_annulus_geoid_response",
        "cylindrical_annulus_self_gravity_response",
        "cylindrical_cosine_boundary_coefficient",
        "cylindrical_annulus_response_from_rotated_stokes",
    )
    for name in expected:
        assert name in uw.postprocessing.geoid.__all__
        assert hasattr(uw.postprocessing.geoid, name)


@pytest.mark.parametrize("wavenumber", [1, 2, 3, 4, 8])
def test_sheet_kernel_supports_positive_fourier_modes(wavenumber):
    density = -0.73
    source = uw.postprocessing.geoid.cylindrical_sheet_potential_coefficient(
        source_radius=RADIUS_INTERNAL,
        target_radius=RADIUS_INTERNAL,
        wavenumber=wavenumber,
        surface_density_coefficient=density,
        gravitational_constant=GRAVITATIONAL_CONSTANT,
    )
    expected = (
        2.0 * np.pi * GRAVITATIONAL_CONSTANT * RADIUS_INTERNAL * density / wavenumber
    )
    assert source == pytest.approx(expected)

    inner_radius = 0.4 * RADIUS_INTERNAL
    outer_radius = 3.0 * RADIUS_INTERNAL
    inner = uw.postprocessing.geoid.cylindrical_sheet_potential_coefficient(
        source_radius=RADIUS_INTERNAL,
        target_radius=inner_radius,
        wavenumber=wavenumber,
        surface_density_coefficient=density,
        gravitational_constant=GRAVITATIONAL_CONSTANT,
    )
    outer = uw.postprocessing.geoid.cylindrical_sheet_potential_coefficient(
        source_radius=RADIUS_INTERNAL,
        target_radius=outer_radius,
        wavenumber=wavenumber,
        surface_density_coefficient=density,
        gravitational_constant=GRAVITATIONAL_CONSTANT,
    )
    assert inner == pytest.approx(
        source * (inner_radius / RADIUS_INTERNAL) ** wavenumber
    )
    assert outer == pytest.approx(
        source * (RADIUS_INTERNAL / outer_radius) ** wavenumber
    )


@pytest.mark.parametrize("wavenumber", [1, 2, 3, 4, 8])
def test_sheet_derivative_has_poisson_jump(wavenumber):
    density = 0.41
    common = {
        "source_radius": RADIUS_INTERNAL,
        "target_radius": RADIUS_INTERNAL,
        "wavenumber": wavenumber,
        "surface_density_coefficient": density,
        "gravitational_constant": GRAVITATIONAL_CONSTANT,
    }
    derivative_inside = (
        uw.postprocessing.geoid.cylindrical_sheet_radial_derivative_coefficient(
            source_side="inside",
            **common,
        )
    )
    derivative_outside = (
        uw.postprocessing.geoid.cylindrical_sheet_radial_derivative_coefficient(
            source_side="outside",
            **common,
        )
    )
    expected_jump = -4.0 * np.pi * GRAVITATIONAL_CONSTANT * density
    assert derivative_outside - derivative_inside == pytest.approx(expected_jump)


def test_axisymmetric_mode_requires_separate_logarithmic_solution():
    with pytest.raises(ValueError, match="logarithmic radial solution"):
        uw.postprocessing.geoid.cylindrical_sheet_potential_coefficient(
            source_radius=2.0,
            target_radius=2.2,
            wavenumber=0,
            surface_density_coefficient=1.0,
        )


def test_negative_mode_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        uw.postprocessing.geoid.cylindrical_sheet_potential_coefficient(
            source_radius=2.0,
            target_radius=2.2,
            wavenumber=-1,
            surface_density_coefficient=1.0,
        )


@pytest.mark.parametrize("wavenumber", [True, 2.0])
def test_noninteger_modes_are_rejected(wavenumber):
    with pytest.raises(TypeError, match="must be an integer"):
        uw.postprocessing.geoid.cylindrical_sheet_potential_coefficient(
            source_radius=2.0,
            target_radius=2.2,
            wavenumber=wavenumber,
            surface_density_coefficient=1.0,
        )


def test_annulus_geoid_response_superposes_boundaries_and_internal_load():
    mode = 3
    outer_topography = -0.77
    inner_topography = -0.32
    outer_density = 0.6
    inner_density = -0.9
    internal_density = 0.27
    outer_gravity = 1.7
    inner_gravity = 2.4
    operator = uw.postprocessing.geoid.cylindrical_annulus_potential_operator(
        radius_inner=RADIUS_INNER,
        radius_outer=RADIUS_OUTER,
        wavenumber=mode,
        outer_density_contrast=outer_density,
        inner_density_contrast=inner_density,
        gravitational_constant=GRAVITATIONAL_CONSTANT,
    )
    expected_potential = operator @ np.array([outer_topography, inner_topography])
    for index, target_radius in enumerate((RADIUS_OUTER, RADIUS_INNER)):
        expected_potential[
            index
        ] += uw.postprocessing.geoid.cylindrical_sheet_potential_coefficient(
            source_radius=RADIUS_INTERNAL,
            target_radius=target_radius,
            wavenumber=mode,
            surface_density_coefficient=internal_density,
            gravitational_constant=GRAVITATIONAL_CONSTANT,
        )

    response = uw.postprocessing.geoid.cylindrical_annulus_geoid_response(
        radius_inner=RADIUS_INNER,
        radius_outer=RADIUS_OUTER,
        wavenumber=mode,
        outer_topography_coefficient=outer_topography,
        inner_topography_coefficient=inner_topography,
        outer_density_contrast=outer_density,
        inner_density_contrast=inner_density,
        outer_reference_gravity=outer_gravity,
        inner_reference_gravity=inner_gravity,
        internal_load_radius=RADIUS_INTERNAL,
        internal_surface_density_coefficient=internal_density,
        gravitational_constant=GRAVITATIONAL_CONSTANT,
    )
    np.testing.assert_allclose(
        [response.outer_potential, response.inner_potential],
        expected_potential,
    )
    np.testing.assert_allclose(
        [response.outer_geoid, response.inner_geoid],
        expected_potential / np.array([outer_gravity, inner_gravity]),
    )


def test_cylindrical_self_gravity_satisfies_matrix_equation():
    kwargs = _adapter_kwargs(wavenumber=4)
    pure_kwargs = _pure_gravity_kwargs(kwargs)
    original_topography = np.array([-0.77, -0.32])
    response = uw.postprocessing.geoid.cylindrical_annulus_self_gravity_response(
        **pure_kwargs,
        outer_topography_coefficient=original_topography[0],
        inner_topography_coefficient=original_topography[1],
    )
    operator = uw.postprocessing.geoid.cylindrical_annulus_potential_operator(
        radius_inner=kwargs["radius_inner"],
        radius_outer=kwargs["radius_outer"],
        wavenumber=kwargs["wavenumber"],
        outer_density_contrast=kwargs["outer_density_contrast"],
        inner_density_contrast=kwargs["inner_density_contrast"],
        gravitational_constant=kwargs["gravitational_constant"],
    )
    load = np.array(
        [
            uw.postprocessing.geoid.cylindrical_sheet_potential_coefficient(
                source_radius=kwargs["internal_load_radius"],
                target_radius=target_radius,
                wavenumber=kwargs["wavenumber"],
                surface_density_coefficient=kwargs[
                    "internal_surface_density_coefficient"
                ],
                gravitational_constant=kwargs["gravitational_constant"],
            )
            for target_radius in (RADIUS_OUTER, RADIUS_INNER)
        ]
    )
    q_matrix = np.diag([response.q_outer, response.q_inner])
    corrected_topography = np.array(
        [response.outer_topography, response.inner_topography]
    )
    residual = (
        (np.eye(2) - q_matrix @ operator) @ corrected_topography
        - original_topography
        - q_matrix @ load
    )
    np.testing.assert_allclose(residual, 0.0, atol=2.0e-16)
    assert response.matrix_residual_norm < 2.0e-16


def test_cylindrical_projection_recovers_mode_and_mean():
    coords, values = _circle_samples(2.22, -0.73, 0.12, 8)
    coefficient, mean = uw.postprocessing.geoid.cylindrical_cosine_boundary_coefficient(
        coords,
        values,
        8,
    )
    assert coefficient == pytest.approx(-0.73, abs=2.0e-15)
    assert mean == pytest.approx(0.12, abs=2.0e-15)


def test_partitioned_rotated_adapter_matches_pure_response():
    kwargs = _adapter_kwargs(wavenumber=3)
    outer_reaction = 0.8
    inner_reaction = -0.4
    stokes = PartitionedFakeRotatedStokes(
        {
            "Upper": _circle_samples(
                RADIUS_OUTER,
                outer_reaction,
                0.03,
                kwargs["wavenumber"],
            ),
            "Lower": _circle_samples(
                RADIUS_INNER,
                inner_reaction,
                -0.02,
                kwargs["wavenumber"],
            ),
        }
    )
    response = uw.postprocessing.geoid.cylindrical_annulus_response_from_rotated_stokes(
        stokes=stokes,
        include_self_gravity=True,
        **kwargs,
    )
    expected_outer_topography = -outer_reaction / kwargs["outer_buoyancy_scale"]
    expected_inner_topography = -inner_reaction / kwargs["inner_buoyancy_scale"]
    pure_kwargs = _pure_gravity_kwargs(kwargs)
    expected = uw.postprocessing.geoid.cylindrical_annulus_geoid_response(
        **pure_kwargs,
        outer_topography_coefficient=expected_outer_topography,
        inner_topography_coefficient=expected_inner_topography,
    )

    assert response.outer_reaction == pytest.approx(outer_reaction)
    assert response.inner_reaction == pytest.approx(inner_reaction)
    np.testing.assert_allclose(
        [response.outer_topography, response.inner_topography],
        [expected_outer_topography, expected_inner_topography],
    )
    np.testing.assert_allclose(
        [
            response.outer_potential,
            response.inner_potential,
            response.outer_geoid,
            response.inner_geoid,
        ],
        [
            expected.outer_potential,
            expected.inner_potential,
            expected.outer_geoid,
            expected.inner_geoid,
        ],
    )
    assert response.self_gravity is not None
