"""Independent Newton-shell and radial Green-integral oracles for gravity kernels."""

import numpy as np
import pytest
from scipy.integrate import quad

from underworld3.analytic.spherical import gravity

pytestmark = pytest.mark.level_1


@pytest.mark.tier_b
@pytest.mark.parametrize("source_radius", [0.4, 2.7])
@pytest.mark.parametrize("density", [-0.7, 1.3])
def test_sheet_monopole_obeys_newton_shell_theorem(source_radius, density):
    gravitational_constant = 0.37
    mass = 4 * np.pi * source_radius**2 * density
    for relative_radius in (0.25, 1, 3):
        radius = relative_radius * source_radius
        expected = gravitational_constant * mass / max(radius, source_radius)
        measured = gravity.spherical_sheet_potential_coefficient(
            source_radius=source_radius, target_radius=radius, harmonic_degree=0,
            surface_density_coefficient=density,
            gravitational_constant=gravitational_constant,
        )
        np.testing.assert_allclose(measured, expected, rtol=3e-15)


@pytest.mark.tier_b
@pytest.mark.parametrize("degree", [0, 1, 2, 8])
def test_sheet_derivative_jump_is_the_prescribed_poisson_source(degree):
    radius, density, gravitational_constant = 1.7, -1.23, 0.37
    derivatives = [gravity.spherical_sheet_radial_derivative_coefficient(
        source_radius=radius, target_radius=radius, harmonic_degree=degree,
        surface_density_coefficient=density, gravitational_constant=gravitational_constant,
        source_side=side,
    ) for side in ("inside", "outside")]
    np.testing.assert_allclose(derivatives[1] - derivatives[0],
                               -4 * np.pi * gravitational_constant * density, rtol=3e-15)


@pytest.mark.tier_b
@pytest.mark.parametrize("radius", [0.7, 1.22, 1.7, 2.22, 3.4])
def test_uniform_volume_shell_has_the_newtonian_monopole(radius):
    inner, outer, density, gravitational_constant = 1.22, 2.22, 1.3, 0.37
    if radius <= inner:
        expected = 2 * np.pi * gravitational_constant * density * (outer**2 - inner**2)
    elif radius >= outer:
        expected = 4 * np.pi * gravitational_constant * density * (outer**3 - inner**3) / (3 * radius)
    else:
        expected = 4 * np.pi * gravitational_constant * density * (
            (radius**3 - inner**3) / (3 * radius) + (outer**2 - radius**2) / 2)
    measured = gravity.spherical_volume_density_potential_coefficient(
        radius_inner=inner, radius_outer=outer, target_radius=radius, harmonic_degree=0,
        density_power=0, density_amplitude=density,
        gravitational_constant=gravitational_constant,
    )
    np.testing.assert_allclose(measured, expected, rtol=3e-15)


@pytest.mark.tier_b
@pytest.mark.parametrize("degree", [0, 2, 5])
@pytest.mark.parametrize("power", [-3, -2, 0, 2])
def test_power_law_volume_kernel_matches_independent_radial_quadrature(degree, power):
    inner, radius, outer, density, gravitational_constant = 1.22, 1.7, 2.22, -0.8, 0.37
    interior = quad(lambda source: density * (source / outer)**power * source**(degree + 2),
                    inner, radius, epsabs=1e-12, epsrel=1e-12)[0]
    exterior = quad(lambda source: density * (source / outer)**power * source**(1 - degree),
                    radius, outer, epsabs=1e-12, epsrel=1e-12)[0]
    expected = 4 * np.pi * gravitational_constant / (2 * degree + 1) * (
        interior / radius**(degree + 1) + radius**degree * exterior)
    measured = gravity.spherical_volume_density_potential_coefficient(
        radius_inner=inner, radius_outer=outer, target_radius=radius, harmonic_degree=degree,
        density_power=power, density_amplitude=density,
        gravitational_constant=gravitational_constant,
    )
    np.testing.assert_allclose(measured, expected, rtol=2e-13)


@pytest.mark.tier_b
@pytest.mark.parametrize("degree", [0, 2, 8])
def test_finite_dirichlet_kernels_vanish_at_the_exterior_radius(degree):
    inner, source, outer, exterior = 1.22, 1.7, 2.22, 4.3
    sheet = gravity.spherical_sheet_finite_dirichlet_potential_coefficient(
        source_radius=source, exterior_radius=exterior, target_radius=exterior,
        harmonic_degree=degree, surface_density_coefficient=1.3, gravitational_constant=0.37,
    )
    volume = gravity.spherical_volume_density_finite_dirichlet_potential_coefficient(
        radius_inner=inner, radius_outer=outer, exterior_radius=exterior,
        target_radius=exterior, harmonic_degree=degree, density_power=-2,
        density_amplitude=1.3, gravitational_constant=0.37,
    )
    scale = 4 * np.pi * 0.37 * 1.3 * outer**2
    np.testing.assert_allclose([sheet, volume], 0, atol=32 * np.finfo(float).eps * scale)
