from types import SimpleNamespace

import numpy as np
import pytest
import sympy
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


class IntegralFakeRotatedStokes:
    """Expose fitted reaction functionals while rejecting pointwise recovery."""

    def __init__(self, boundary_data):
        radius, theta = sympy.symbols("r theta", real=True)
        self.mesh = SimpleNamespace(
            dim=2,
            CoordinateSystem=SimpleNamespace(xR=(radius, theta)),
            boundary_data=boundary_data,
        )
        self.boundary_data = boundary_data
        self.integral_calls = []

    def boundary_normal_traction(self, boundary, mass="auto"):
        raise AssertionError("The finite-element adapter must not recover point values.")

    def boundary_normal_traction_integral(self, boundary, fn, remove_mean=True):
        radius, coefficient, mean = self.boundary_data[boundary]
        self.integral_calls.append((boundary, bool(remove_mean)))
        if remove_mean:
            return coefficient * np.pi * radius
        assert float(sympy.sympify(fn)) == 1.0
        return mean * 2.0 * np.pi * radius


class FakeBoundaryIntegral:
    """Exact circular measure used to isolate the adapter's projection route."""

    def __init__(self, mesh, fn, boundary):
        self.mesh = mesh
        self.fn = sympy.sympify(fn)
        self.boundary = boundary

    def evaluate(self):
        radius = self.mesh.boundary_data[self.boundary][0]
        if self.fn.is_number and float(self.fn) == 1.0:
            return 2.0 * np.pi * radius
        return np.pi * radius


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
    assert hasattr(uw.systems.Stokes, "boundary_normal_traction_integral")


def test_reaction_integral_requires_boolean_mean_removal():
    from underworld3.utilities.rotated_bc import boundary_normal_traction_integral

    with pytest.raises(TypeError, match="remove_mean must be True or False"):
        boundary_normal_traction_integral(
            solver=None,
            boundary="Upper",
            solve_result=None,
            fn=1.0,
            remove_mean="yes",
        )


def test_distributed_reaction_integral_on_finite_element_annulus():
    """Exercise the complete adapter on an assembled rotated-free-slip reaction."""

    radius_inner = 0.5
    radius_outer = 1.0
    mode = 4
    mesh = uw.meshing.Annulus(
        radiusInner=radius_inner,
        radiusOuter=radius_outer,
        cellSize=0.15,
        qdegree=3,
    )
    x, y = mesh.X
    radius = sympy.sqrt(x**2 + y**2)
    theta = sympy.atan2(y, x)
    velocity = uw.discretisation.MeshVariable(
        "V_geoid_integral", mesh, mesh.dim, degree=2, continuous=True
    )
    pressure = uw.discretisation.MeshVariable(
        "P_geoid_integral", mesh, 1, degree=1, continuous=True
    )
    stokes = uw.systems.Stokes(
        mesh,
        velocityField=velocity,
        pressureField=pressure,
    )
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    radial_force = (
        sympy.cos(mode * theta)
        * (radius - radius_inner)
        * (radius_outer - radius)
        * 40.0
    )
    stokes.bodyforce = sympy.Matrix(
        [[x * radial_force / radius, y * radial_force / radius]]
    )
    normal = sympy.Matrix([[x / radius, y / radius]])
    stokes.add_rotated_freeslip_bc(0, "Lower", normal=normal)
    stokes.add_rotated_freeslip_bc(0, "Upper", normal=normal)
    stokes.petsc_use_pressure_nullspace = True
    stokes.petsc_options["snes_type"] = "ksponly"
    stokes.solve()

    response = uw.postprocessing.geoid.cylindrical_annulus_response_from_rotated_stokes(
        stokes=stokes,
        radius_inner=radius_inner,
        radius_outer=radius_outer,
        wavenumber=mode,
        outer_density_contrast=1.0,
        inner_density_contrast=1.0,
        outer_reference_gravity=1.0,
        inner_reference_gravity=1.0,
    )

    assert response.outer_reaction == pytest.approx(0.4044747, rel=2.0e-4)
    assert response.inner_reaction == pytest.approx(0.5158471, rel=2.0e-4)
    assert np.all(
        np.isfinite(
            [
                response.outer_reaction_mean,
                response.inner_reaction_mean,
                response.outer_geoid,
                response.inner_geoid,
            ]
        )
    )


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


def test_rotated_adapter_uses_distributed_reaction_integral(monkeypatch):
    kwargs = _adapter_kwargs(wavenumber=3)
    outer_reaction = 0.8
    inner_reaction = -0.4
    stokes = IntegralFakeRotatedStokes(
        {
            "Upper": (RADIUS_OUTER, outer_reaction, 0.03),
            "Lower": (RADIUS_INNER, inner_reaction, -0.02),
        }
    )
    monkeypatch.setattr(uw.maths, "BdIntegral", FakeBoundaryIntegral)
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
    assert response.outer_reaction_mean == pytest.approx(0.03)
    assert response.inner_reaction_mean == pytest.approx(-0.02)
    assert stokes.integral_calls == [
        ("Upper", True),
        ("Upper", False),
        ("Lower", True),
        ("Lower", False),
    ]
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
