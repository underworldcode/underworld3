r"""Analytical spherical-harmonic sheet and volume gravity kernels.

This module deliberately has no Underworld dependency.  It derives the radial
potential of one spherical density sheet from

    laplacian(Phi) = -4*pi*G*sigma_lm*delta(r-r_s)*Y_lm,

using the regular interior branch ``r**ell`` and decaying exterior branch
``r**(-(ell+1))``.  The derivative jump fixes the source-radius amplitude to

    Phi_lm(r_s) = 4*pi*G*sigma_lm*r_s/(2*ell+1).

The normalized two-boundary operator factors out ``4*pi*G/g`` and therefore
matches the coefficient convention used by the Zhong et al. (2008) response
benchmark.  A positive outward internal-load coefficient is represented by a
negative density-sheet coefficient and consequently enters the load vector
with a negative sign.

Reference and provenance
------------------------
Zhong, S., McNamara, A., Tan, E., Moresi, L. and Gurnis, M. (2008), A benchmark
study on mantle convection in a 3-D spherical shell using CitcomS, Geochemistry,
Geophysics, Geosystems 9, Q10017, Appendix A.
https://doi.org/10.1029/2008GC002048
The physical sheet/volume kernels are derived here from the displayed Poisson
equation. General powers, finite-boundary kernels and multi-mode support are
UW3 extensions; the citation identifies the normalized response convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np


__all__ = [
    "GeoidResponse",
    "SelfGravityResponse",
    "spherical_shell_geoid_response",
    "spherical_shell_self_gravity_response",
    "CONVENTIONS",
    "axisymmetric_harmonic_norm",
    "recover_axisymmetric_coefficient",
    "spherical_sheet_source_amplitude",
    "spherical_sheet_radial_factor",
    "spherical_sheet_potential_coefficient",
    "spherical_sheet_finite_dirichlet_potential_coefficient",
    "spherical_sheet_radial_derivative_coefficient",
    "spherical_volume_density_potential_coefficient",
    "spherical_volume_density_finite_dirichlet_potential_coefficient",
    "spherical_volume_density_potential_coefficient_quadrature",
    "spherical_volume_density_radial_derivative_coefficient",
    "spherical_shell_potential_operator",
    "spherical_physical_shell_response",
]


@dataclass(frozen=True)
class GeoidResponse:
    r"""Normalized spherical geoid response without interface feedback.

    Attributes
    ----------
    surface_geoid, cmb_geoid : float
        Signed geoid coefficients at the outer surface and core-mantle boundary,
        in the common input harmonic basis. The normalized kernel factors out
        ``4*pi*G/g``; these coefficients are not automatically SI heights."""

    surface_geoid: float
    cmb_geoid: float


@dataclass(frozen=True)
class SelfGravityResponse:
    r"""Normalized spherical topography and geoid after two-interface feedback.

    Attributes
    ----------
    surface_topography, cmb_topography : float
        Corrected nondimensional interface-displacement coefficients.
    surface_geoid, cmb_geoid : float
        Corrected normalized geoid coefficients in the input harmonic basis.
    q_surface, q_cmb : float
        Signed dimensionless feedback factors ``4*pi*G*planet_radius*Delta_rho/g``.
    matrix_condition_number : float, default=1
        Two-norm condition number of the two-by-two feedback system.
    matrix_residual_norm : float, default=0
        Euclidean norm of the corrected-topography system residual, not a spatial
        field error or an accuracy acceptance test."""

    surface_topography: float
    cmb_topography: float
    surface_geoid: float
    cmb_geoid: float
    q_surface: float
    q_cmb: float
    matrix_condition_number: float = 1.0
    matrix_residual_norm: float = 0.0


CONVENTIONS = {
    "harmonic_basis": ("Y_ell0=P_ell(cos(theta)); integral(Y_ell0^2 dOmega)=" "4*pi/(2*ell+1)"),
    "potential_convention": "laplacian(Phi)=-4*pi*G*delta_rho",
    "poisson_source_sign": "UW3 -div(grad(Phi))=+4*pi*G*delta_rho",
    "surface_normal": "+e_r",
    "cmb_normal": "-e_r for the mantle domain",
    "topography_convention": "h_surface=-sigma_rr_surface; h_cmb=+sigma_rr_cmb",
    "internal_load_convention": (
        "positive outward load enters the density-sheet load vector with a " "negative sign"
    ),
    "geoid_convention": "N=Phi/g; normalized operator factors out 4*pi*G/g",
    "coefficient_dimensions": (
        "stored response coefficients are nondimensional; planet_radius, "
        "gravity, density contrasts, and G retain the SI units named by "
        "their metadata keys"
    ),
}


def _finite_float(value: Real, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real number.") from error
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _positive_float(value: Real, name: str) -> float:
    result = _finite_float(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return result


def _validate_degree(harmonic_degree: int) -> int:
    if isinstance(harmonic_degree, bool) or not isinstance(harmonic_degree, Integral):
        raise TypeError("harmonic_degree must be an integer.")
    degree = int(harmonic_degree)
    if degree < 0:
        raise ValueError("harmonic_degree must be non-negative.")
    return degree


def _validate_geometry(
    radius_inner: Real,
    radius_outer: Real,
    harmonic_degree: int,
) -> tuple[float, float, int]:
    radius_inner = _positive_float(radius_inner, "radius_inner")
    radius_outer = _positive_float(radius_outer, "radius_outer")
    if radius_inner >= radius_outer:
        raise ValueError("Expected radius_inner < radius_outer.")
    return radius_inner, radius_outer, _validate_degree(harmonic_degree)


def axisymmetric_harmonic_norm(harmonic_degree: int) -> float:
    r"""Return ``integral(P_ell(cos(theta))**2 dOmega)`` on the unit sphere.

    Parameters
    ----------
    harmonic_degree : int
        Nonnegative spherical-harmonic degree; the pure radial kernels include degree zero.

    Returns
    -------
    float
        Dimensionless unit-sphere angular norm 4*pi/(2*harmonic_degree+1).

    Raises
    ------
    TypeError
        harmonic_degree is not an integer or is boolean.
    ValueError
        harmonic_degree is negative.

    Notes
    -----
    The angular basis is the unnormalized axisymmetric Legendre polynomial ``P_l(cos(theta))``. No
    mesh, MPI operation or PDE solve is performed.
    """

    degree = _validate_degree(harmonic_degree)
    return float(4.0 * np.pi / (2 * degree + 1))


def recover_axisymmetric_coefficient(
    *,
    weighted_surface_integral: Real,
    sphere_radius: Real,
    harmonic_degree: int,
) -> float:
    r"""Recover ``f_ell0`` from ``integral(f P_ell dS)`` on a sphere.

    Parameters
    ----------
    weighted_surface_integral : real
        Finite integral of the scalar field times P_l over the spherical surface; units are
        field times area.
    sphere_radius : real
        Finite positive sphere radius in the surface integral's length units.
    harmonic_degree : int
        Nonnegative spherical-harmonic degree; the pure radial kernels include degree zero.

    Returns
    -------
    float
        Integral divided by sphere_radius**2 and the unit-sphere angular norm, in the
        original field units.

    Raises
    ------
    TypeError
        The harmonic degree/mode or density power has an invalid type, or a real input
        cannot be converted.
    ValueError
        A numeric input is nonfinite, a required positive value is nonpositive, or the
        harmonic degree/mode, radius ordering or source support is invalid.

    Notes
    -----
    The angular basis is the unnormalized axisymmetric Legendre polynomial ``P_l(cos(theta))``. No
    mesh, MPI operation or PDE solve is performed.
    """

    integral = _finite_float(weighted_surface_integral, "weighted_surface_integral")
    radius = _positive_float(sphere_radius, "sphere_radius")
    return float(integral / (radius**2 * axisymmetric_harmonic_norm(harmonic_degree)))


def spherical_sheet_source_amplitude(
    *,
    source_radius: Real,
    harmonic_degree: int,
    surface_density_coefficient: Real,
    gravitational_constant: Real = 1.0,
) -> float:
    r"""Return the potential coefficient at a spherical density sheet.

    Parameters
    ----------
    source_radius : real
        Finite positive sheet radius in the chosen length units.
    harmonic_degree : int
        Nonnegative spherical-harmonic degree; the pure radial kernels include degree zero.
    surface_density_coefficient : real
        Finite signed surface-mass density coefficient multiplying the chosen harmonic;
        physical units are mass/area.
    gravitational_constant : real, default=1.0
        Finite positive G in units consistent with radii and densities. No unit conversion
        is performed.

    Returns
    -------
    float
        Potential coefficient at the target radius in length^2/time^2 for physical inputs.

    Raises
    ------
    TypeError
        The harmonic degree/mode or density power has an invalid type, or a real input
        cannot be converted.
    ValueError
        A numeric input is nonfinite, a required positive value is nonpositive, or the
        harmonic degree/mode, radius ordering or source support is invalid.

    Notes
    -----
    The convention is laplacian(Phi)=-4*pi*G*rho. All density coefficients and returned coefficients
    share the selected angular basis/normalization. No mesh, MPI operation or PDE solve is
    performed.
    """

    source_radius = _positive_float(source_radius, "source_radius")
    degree = _validate_degree(harmonic_degree)
    density = _finite_float(surface_density_coefficient, "surface_density_coefficient")
    gravity_constant = _positive_float(gravitational_constant, "gravitational_constant")
    return float(4.0 * np.pi * gravity_constant * density * source_radius / (2 * degree + 1))


def spherical_sheet_radial_factor(
    *,
    source_radius: Real,
    target_radius: Real,
    harmonic_degree: int,
) -> float:
    r"""Return the regular/decaying radial continuation factor.

    Parameters
    ----------
    source_radius : real
        Finite positive sheet radius in the chosen length units.
    target_radius : real
        Finite positive evaluation radius, inside or outside the source support.
    harmonic_degree : int
        Nonnegative spherical-harmonic degree; the pure radial kernels include degree zero.

    Returns
    -------
    float
        Dimensionless factor (target/source)**l inside or (source/target)**(l+1) outside; it
        is 1 on the sheet.

    Raises
    ------
    TypeError
        The harmonic degree/mode or density power has an invalid type, or a real input
        cannot be converted.
    ValueError
        A numeric input is nonfinite, a required positive value is nonpositive, or the
        harmonic degree/mode, radius ordering or source support is invalid.

    Notes
    -----
    The factor is independent of source amplitude and angular normalization. No mesh, MPI operation
    or PDE solve is performed.
    """

    source_radius = _positive_float(source_radius, "source_radius")
    target_radius = _positive_float(target_radius, "target_radius")
    degree = _validate_degree(harmonic_degree)
    if target_radius <= source_radius:
        return float((target_radius / source_radius) ** degree)
    return float((source_radius / target_radius) ** (degree + 1))


def spherical_sheet_potential_coefficient(
    *,
    source_radius: Real,
    target_radius: Real,
    harmonic_degree: int,
    surface_density_coefficient: Real,
    gravitational_constant: Real = 1.0,
) -> float:
    r"""Return ``Phi_lm`` at one target radius for a spherical density sheet.

    Parameters
    ----------
    source_radius : real
        Finite positive sheet radius in the chosen length units.
    target_radius : real
        Finite positive evaluation radius, inside or outside the source support.
    harmonic_degree : int
        Nonnegative spherical-harmonic degree; the pure radial kernels include degree zero.
    surface_density_coefficient : real
        Finite signed surface-mass density coefficient multiplying the chosen harmonic;
        physical units are mass/area.
    gravitational_constant : real, default=1.0
        Finite positive G in units consistent with radii and densities. No unit conversion
        is performed.

    Returns
    -------
    float
        Potential coefficient at the target radius in length^2/time^2 for physical inputs.

    Raises
    ------
    TypeError
        The harmonic degree/mode or density power has an invalid type, or a real input
        cannot be converted.
    ValueError
        A numeric input is nonfinite, a required positive value is nonpositive, or the
        harmonic degree/mode, radius ordering or source support is invalid.

    Notes
    -----
    The convention is laplacian(Phi)=-4*pi*G*rho. All density coefficients and returned coefficients
    share the selected angular basis/normalization. No mesh, MPI operation or PDE solve is
    performed.
    """

    amplitude = spherical_sheet_source_amplitude(
        source_radius=source_radius,
        harmonic_degree=harmonic_degree,
        surface_density_coefficient=surface_density_coefficient,
        gravitational_constant=gravitational_constant,
    )
    factor = spherical_sheet_radial_factor(
        source_radius=source_radius,
        target_radius=target_radius,
        harmonic_degree=harmonic_degree,
    )
    return float(amplitude * factor)


def spherical_sheet_radial_derivative_coefficient(
    *,
    source_radius: Real,
    target_radius: Real,
    harmonic_degree: int,
    surface_density_coefficient: Real,
    gravitational_constant: Real = 1.0,
    source_side: str | None = None,
) -> float:
    r"""Return ``dPhi_lm/dr`` on an interior or exterior radial branch.

    ``source_side`` is required only when ``target_radius == source_radius``;
    use ``"inside"`` for the regular branch and ``"outside"`` for the
    decaying branch.

    Parameters
    ----------
    source_radius : real
        Finite positive sheet radius in the chosen length units.
    target_radius : real
        Finite positive evaluation radius, inside or outside the source support.
    harmonic_degree : int
        Nonnegative spherical-harmonic degree; the pure radial kernels include degree zero.
    surface_density_coefficient : real
        Finite signed surface-mass density coefficient multiplying the chosen harmonic;
        physical units are mass/area.
    gravitational_constant : real, default=1.0
        Finite positive G in units consistent with radii and densities. No unit conversion
        is performed.
    source_side : {"inside", "outside"} or None, default=None
        Required exactly on a density sheet to select the one-sided derivative; must be None
        away from the sheet.

    Returns
    -------
    float
        Radial potential derivative coefficient in potential/length units. Compact
        volume-source derivatives are continuous at the source-support boundaries.

    Raises
    ------
    TypeError
        The harmonic degree/mode or density power has an invalid type, or a real input
        cannot be converted.
    ValueError
        A numeric input is nonfinite, a required positive value is nonpositive, or the
        harmonic degree/mode, radius ordering or source support is invalid.

    Notes
    -----
    The convention is laplacian(Phi)=-4*pi*G*rho. All density coefficients and returned coefficients
    share the selected angular basis/normalization. No mesh, MPI operation or PDE solve is
    performed. The sheet derivative jump outside minus inside is -4*pi*G*sigma; select a side
    explicitly on the sheet.
    """

    source_radius = _positive_float(source_radius, "source_radius")
    target_radius = _positive_float(target_radius, "target_radius")
    degree = _validate_degree(harmonic_degree)
    potential = spherical_sheet_potential_coefficient(
        source_radius=source_radius,
        target_radius=target_radius,
        harmonic_degree=degree,
        surface_density_coefficient=surface_density_coefficient,
        gravitational_constant=gravitational_constant,
    )

    if np.isclose(target_radius, source_radius, rtol=0.0, atol=0.0):
        if source_side not in {"inside", "outside"}:
            raise ValueError("source_side must be 'inside' or 'outside' at the source radius.")
        inside = source_side == "inside"
    else:
        if source_side is not None:
            raise ValueError("source_side is valid only at the source radius.")
        inside = target_radius < source_radius

    radial_power = degree if inside else -(degree + 1)
    return float(radial_power * potential / target_radius)


def _power_integral(lower: float, upper: float, exponent: int) -> float:
    """Return the exact integral of ``s**exponent`` on positive bounds."""

    if upper <= lower:
        return 0.0
    if exponent == -1:
        return float(np.log(upper / lower))
    power = exponent + 1
    return float((upper**power - lower**power) / power)


def _validate_volume_density_parameters(
    *,
    radius_inner: Real,
    radius_outer: Real,
    target_radius: Real,
    harmonic_degree: int,
    density_power: int,
    density_amplitude: Real,
    gravitational_constant: Real,
) -> tuple[float, float, float, int, int, float, float]:
    radius_inner, radius_outer, degree = _validate_geometry(
        radius_inner, radius_outer, harmonic_degree
    )
    target = _positive_float(target_radius, "target_radius")
    if isinstance(density_power, bool) or not isinstance(density_power, Integral):
        raise TypeError("density_power must be an integer.")
    power = int(density_power)
    amplitude = _finite_float(density_amplitude, "density_amplitude")
    gravity_constant = _positive_float(gravitational_constant, "gravitational_constant")
    return (
        radius_inner,
        radius_outer,
        target,
        degree,
        power,
        amplitude,
        gravity_constant,
    )


def spherical_volume_density_potential_coefficient(
    *,
    radius_inner: Real,
    radius_outer: Real,
    target_radius: Real,
    harmonic_degree: int,
    density_power: int = 2,
    density_amplitude: Real = 1.0,
    gravitational_constant: Real = 1.0,
) -> float:
    r"""Return the potential coefficient for a compact radial volume source.

    The source coefficient is

    ``rho_ell(r) = density_amplitude * (r/radius_outer)**density_power``

    between ``radius_inner`` and ``radius_outer`` and is zero elsewhere.  The
    result follows directly from the regular/decaying spherical Green kernel
    and uses the convention ``laplacian(Phi) = -4*pi*G*rho``.

    Parameters
    ----------
    radius_inner : real
        Finite positive inner source/shell radius in consistent length units.
    radius_outer : real
        Finite positive outer source/shell radius, strictly larger than radius_inner.
    target_radius : real
        Finite positive evaluation radius, inside or outside the source support.
    harmonic_degree : int
        Nonnegative spherical-harmonic degree; the pure radial kernels include degree zero.
    density_power : int, default=2
        Integer k in the compact source rho(r)=density_amplitude*(r/radius_outer)**k;
        default 2. Signed powers are supported on positive source radii.
    density_amplitude : real, default=1.0
        Finite signed volume-density coefficient at radius_outer; default 1. Physical units
        are mass/volume; support is the source annulus/shell.
    gravitational_constant : real, default=1.0
        Finite positive G in units consistent with radii and densities. No unit conversion
        is performed.

    Returns
    -------
    float
        Potential coefficient at the target radius in length^2/time^2 for physical inputs.

    Raises
    ------
    TypeError
        The harmonic degree/mode or density power has an invalid type, or a real input
        cannot be converted.
    ValueError
        A numeric input is nonfinite, a required positive value is nonpositive, or the
        harmonic degree/mode, radius ordering or source support is invalid.

    Notes
    -----
    The convention is laplacian(Phi)=-4*pi*G*rho. All density coefficients and returned coefficients
    share the selected angular basis/normalization. No mesh, MPI operation or PDE solve is
    performed.
    """

    ri, ro, radius, degree, power, amplitude, gravity_constant = (
        _validate_volume_density_parameters(
            radius_inner=radius_inner,
            radius_outer=radius_outer,
            target_radius=target_radius,
            harmonic_degree=harmonic_degree,
            density_power=density_power,
            density_amplitude=density_amplitude,
            gravitational_constant=gravitational_constant,
        )
    )
    scale = amplitude / ro**power
    inside_integral = scale * _power_integral(
        ri,
        min(radius, ro),
        degree + 2 + power,
    )
    outside_integral = scale * _power_integral(
        max(radius, ri),
        ro,
        1 - degree + power,
    )
    prefactor = 4.0 * np.pi * gravity_constant / (2 * degree + 1)
    return float(
        prefactor
        * (radius ** (-(degree + 1)) * inside_integral + radius**degree * outside_integral)
    )


def spherical_sheet_finite_dirichlet_potential_coefficient(
    *,
    source_radius: Real,
    exterior_radius: Real,
    target_radius: Real,
    harmonic_degree: int,
    surface_density_coefficient: Real,
    gravitational_constant: Real = 1.0,
) -> float:
    r"""Return one sheet's regular full-ball solution with ``Phi(R_ext)=0``.

    Parameters
    ----------
    source_radius : real
        Finite positive sheet radius in the chosen length units.
    exterior_radius : real
        Finite positive zero-Dirichlet truncation radius, strictly beyond the sheet or outer
        volume-source radius.
    target_radius : real
        Finite positive evaluation radius, inside or outside the source support. Must not
        exceed ``exterior_radius``.
    harmonic_degree : int
        Nonnegative spherical-harmonic degree; the pure radial kernels include degree zero.
    surface_density_coefficient : real
        Finite signed surface-mass density coefficient multiplying the chosen harmonic;
        physical units are mass/area.
    gravitational_constant : real, default=1.0
        Finite positive G in units consistent with radii and densities. No unit conversion
        is performed.

    Returns
    -------
    float
        Potential coefficient at the target radius in length^2/time^2 for physical inputs.

    Raises
    ------
    TypeError
        The harmonic degree/mode or density power has an invalid type, or a real input
        cannot be converted.
    ValueError
        A numeric input is nonfinite, a required positive value is nonpositive, or the
        harmonic degree/mode, radius ordering or source support is invalid.

    Notes
    -----
    The convention is laplacian(Phi)=-4*pi*G*rho. All density coefficients and returned coefficients
    share the selected angular basis/normalization. No mesh, MPI operation or PDE solve is
    performed. A regular homogeneous r**l correction enforces Phi(exterior_radius)=0; targets
    outside that boundary are rejected.
    """

    source = _positive_float(source_radius, "source_radius")
    exterior = _positive_float(exterior_radius, "exterior_radius")
    target = _positive_float(target_radius, "target_radius")
    degree = _validate_degree(harmonic_degree)
    if exterior <= source:
        raise ValueError("exterior_radius must be greater than source_radius.")
    if target > exterior:
        raise ValueError("target_radius must not exceed exterior_radius.")
    infinite_target = spherical_sheet_potential_coefficient(
        source_radius=source,
        target_radius=target,
        harmonic_degree=degree,
        surface_density_coefficient=surface_density_coefficient,
        gravitational_constant=gravitational_constant,
    )
    infinite_exterior = spherical_sheet_potential_coefficient(
        source_radius=source,
        target_radius=exterior,
        harmonic_degree=degree,
        surface_density_coefficient=surface_density_coefficient,
        gravitational_constant=gravitational_constant,
    )
    return float(infinite_target - infinite_exterior * (target / exterior) ** degree)


def spherical_volume_density_radial_derivative_coefficient(
    *,
    radius_inner: Real,
    radius_outer: Real,
    target_radius: Real,
    harmonic_degree: int,
    density_power: int = 2,
    density_amplitude: Real = 1.0,
    gravitational_constant: Real = 1.0,
) -> float:
    r"""Return ``dPhi_ell/dr`` for the compact radial volume source.

    Parameters
    ----------
    radius_inner : real
        Finite positive inner source/shell radius in consistent length units.
    radius_outer : real
        Finite positive outer source/shell radius, strictly larger than radius_inner.
    target_radius : real
        Finite positive evaluation radius, inside or outside the source support.
    harmonic_degree : int
        Nonnegative spherical-harmonic degree; the pure radial kernels include degree zero.
    density_power : int, default=2
        Integer k in the compact source rho(r)=density_amplitude*(r/radius_outer)**k;
        default 2. Signed powers are supported on positive source radii.
    density_amplitude : real, default=1.0
        Finite signed volume-density coefficient at radius_outer; default 1. Physical units
        are mass/volume; support is the source annulus/shell.
    gravitational_constant : real, default=1.0
        Finite positive G in units consistent with radii and densities. No unit conversion
        is performed.

    Returns
    -------
    float
        Radial potential derivative coefficient in potential/length units. Compact
        volume-source derivatives are continuous at the source-support boundaries.

    Raises
    ------
    TypeError
        The harmonic degree/mode or density power has an invalid type, or a real input
        cannot be converted.
    ValueError
        A numeric input is nonfinite, a required positive value is nonpositive, or the
        harmonic degree/mode, radius ordering or source support is invalid.

    Notes
    -----
    The convention is laplacian(Phi)=-4*pi*G*rho. All density coefficients and returned coefficients
    share the selected angular basis/normalization. No mesh, MPI operation or PDE solve is
    performed.
    """

    ri, ro, radius, degree, power, amplitude, gravity_constant = (
        _validate_volume_density_parameters(
            radius_inner=radius_inner,
            radius_outer=radius_outer,
            target_radius=target_radius,
            harmonic_degree=harmonic_degree,
            density_power=density_power,
            density_amplitude=density_amplitude,
            gravitational_constant=gravitational_constant,
        )
    )
    scale = amplitude / ro**power
    inside_integral = scale * _power_integral(
        ri,
        min(radius, ro),
        degree + 2 + power,
    )
    outside_integral = scale * _power_integral(
        max(radius, ri),
        ro,
        1 - degree + power,
    )
    prefactor = 4.0 * np.pi * gravity_constant / (2 * degree + 1)
    return float(
        prefactor
        * (
            -(degree + 1) * radius ** (-(degree + 2)) * inside_integral
            + degree * radius ** (degree - 1) * outside_integral
        )
    )


def spherical_volume_density_finite_dirichlet_potential_coefficient(
    *,
    radius_inner: Real,
    radius_outer: Real,
    exterior_radius: Real,
    target_radius: Real,
    harmonic_degree: int,
    density_power: int = 2,
    density_amplitude: Real = 1.0,
    gravitational_constant: Real = 1.0,
) -> float:
    r"""Return the exact full-ball solution with ``Phi(R_ext)=0``.

    The infinite-domain solution is corrected by the regular homogeneous
    branch ``C*r**ell``. This preserves regularity at the centre and enforces
    the finite exterior Dirichlet condition exactly.

    Parameters
    ----------
    radius_inner : real
        Finite positive inner source/shell radius in consistent length units.
    radius_outer : real
        Finite positive outer source/shell radius, strictly larger than radius_inner.
    exterior_radius : real
        Finite positive zero-Dirichlet truncation radius, strictly beyond the sheet or outer
        volume-source radius.
    target_radius : real
        Finite positive evaluation radius, inside or outside the source support. Must not
        exceed ``exterior_radius``.
    harmonic_degree : int
        Nonnegative spherical-harmonic degree; the pure radial kernels include degree zero.
    density_power : int, default=2
        Integer k in the compact source rho(r)=density_amplitude*(r/radius_outer)**k;
        default 2. Signed powers are supported on positive source radii.
    density_amplitude : real, default=1.0
        Finite signed volume-density coefficient at radius_outer; default 1. Physical units
        are mass/volume; support is the source annulus/shell.
    gravitational_constant : real, default=1.0
        Finite positive G in units consistent with radii and densities. No unit conversion
        is performed.

    Returns
    -------
    float
        Potential coefficient at the target radius in length^2/time^2 for physical inputs.

    Raises
    ------
    TypeError
        The harmonic degree/mode or density power has an invalid type, or a real input
        cannot be converted.
    ValueError
        A numeric input is nonfinite, a required positive value is nonpositive, or the
        harmonic degree/mode, radius ordering or source support is invalid.

    Notes
    -----
    The convention is laplacian(Phi)=-4*pi*G*rho. All density coefficients and returned coefficients
    share the selected angular basis/normalization. No mesh, MPI operation or PDE solve is
    performed. A regular homogeneous r**l correction enforces Phi(exterior_radius)=0; targets
    outside that boundary are rejected.
    """

    exterior = _positive_float(exterior_radius, "exterior_radius")
    target = _positive_float(target_radius, "target_radius")
    degree = _validate_degree(harmonic_degree)
    radius_outer_value = _positive_float(radius_outer, "radius_outer")
    if exterior <= radius_outer_value:
        raise ValueError("exterior_radius must be greater than radius_outer.")
    if target > exterior:
        raise ValueError("target_radius must not exceed exterior_radius.")
    infinite_target = spherical_volume_density_potential_coefficient(
        radius_inner=radius_inner,
        radius_outer=radius_outer,
        target_radius=target,
        harmonic_degree=degree,
        density_power=density_power,
        density_amplitude=density_amplitude,
        gravitational_constant=gravitational_constant,
    )
    infinite_exterior = spherical_volume_density_potential_coefficient(
        radius_inner=radius_inner,
        radius_outer=radius_outer,
        target_radius=exterior,
        harmonic_degree=degree,
        density_power=density_power,
        density_amplitude=density_amplitude,
        gravitational_constant=gravitational_constant,
    )
    return float(infinite_target - infinite_exterior * (target / exterior) ** degree)


def spherical_volume_density_potential_coefficient_quadrature(
    *,
    radius_inner: Real,
    radius_outer: Real,
    target_radius: Real,
    harmonic_degree: int,
    density_power: int = 2,
    density_amplitude: Real = 1.0,
    gravitational_constant: Real = 1.0,
    quadrature_points: int = 128,
) -> float:
    r"""Evaluate the same volume-source Green integral by Gauss--Legendre.

    Parameters
    ----------
    radius_inner : real
        Finite positive inner source/shell radius in consistent length units.
    radius_outer : real
        Finite positive outer source/shell radius, strictly larger than radius_inner.
    target_radius : real
        Finite positive evaluation radius, inside or outside the source support.
    harmonic_degree : int
        Nonnegative spherical-harmonic degree; the pure radial kernels include degree zero.
    density_power : int, default=2
        Integer k in the compact source rho(r)=density_amplitude*(r/radius_outer)**k;
        default 2. Signed powers are supported on positive source radii.
    density_amplitude : real, default=1.0
        Finite signed volume-density coefficient at radius_outer; default 1. Physical units
        are mass/volume; support is the source annulus/shell.
    gravitational_constant : real, default=1.0
        Finite positive G in units consistent with radii and densities. No unit conversion
        is performed.
    quadrature_points : int, default=128
        At least two Gauss-Legendre nodes per radial subinterval; default 128. This is
        integration resolution, not a mesh size.

    Returns
    -------
    float
        Potential coefficient at the target radius in length^2/time^2 for physical inputs.

    Raises
    ------
    TypeError
        The harmonic degree/mode or density power has an invalid type, or a real input
        cannot be converted.
    ValueError
        A numeric input is nonfinite, a required positive value is nonpositive, or the
        harmonic degree/mode, radius ordering or source support is invalid.

    Notes
    -----
    The convention is laplacian(Phi)=-4*pi*G*rho. All density coefficients and returned coefficients
    share the selected angular basis/normalization. No mesh, MPI operation or PDE solve is
    performed.
    """

    ri, ro, radius, degree, power, amplitude, gravity_constant = (
        _validate_volume_density_parameters(
            radius_inner=radius_inner,
            radius_outer=radius_outer,
            target_radius=target_radius,
            harmonic_degree=harmonic_degree,
            density_power=density_power,
            density_amplitude=density_amplitude,
            gravitational_constant=gravitational_constant,
        )
    )
    if (
        isinstance(quadrature_points, bool)
        or not isinstance(quadrature_points, Integral)
        or quadrature_points < 2
    ):
        raise ValueError("quadrature_points must be an integer of at least 2.")
    nodes, weights = np.polynomial.legendre.leggauss(int(quadrature_points))

    def integrate(lower: float, upper: float, exponent: int) -> float:
        if upper <= lower:
            return 0.0
        points = 0.5 * (upper - lower) * nodes + 0.5 * (upper + lower)
        return float(0.5 * (upper - lower) * np.dot(weights, points**exponent))

    scale = amplitude / ro**power
    inside_integral = scale * integrate(
        ri,
        min(radius, ro),
        degree + 2 + power,
    )
    outside_integral = scale * integrate(
        max(radius, ri),
        ro,
        1 - degree + power,
    )
    prefactor = 4.0 * np.pi * gravity_constant / (2 * degree + 1)
    return float(
        prefactor
        * (radius ** (-(degree + 1)) * inside_integral + radius**degree * outside_integral)
    )


def spherical_shell_potential_operator(
    *,
    radius_inner: Real,
    radius_outer: Real,
    harmonic_degree: int,
    internal_load_radius: Real | None = None,
    internal_load_coefficient: Real = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Return the normalized surface/CMB sheet operator and load vector.

    Rows target the surface then CMB. Columns source unit surface then CMB
    density/topography coefficients. The normalization has ``4*pi*G/g = 1``.

    Parameters
    ----------
    radius_inner : real
        Finite positive inner source/shell radius in consistent length units.
    radius_outer : real
        Finite positive outer source/shell radius, strictly larger than radius_inner.
    harmonic_degree : int
        Nonnegative spherical-harmonic degree; the pure radial kernels include degree zero.
    internal_load_radius : real or None, default=None
        Internal sheet radius, strictly inside the shell when supplied. Required for a
        nonzero internal load; default None.
    internal_load_coefficient : real, default=0.0
        Finite signed normalized outward radial-load coefficient; default 0. It enters the
        normalized potential with negative sheet-density sign.

    Returns
    -------
    operator : numpy.ndarray, shape (2, 2)
        Normalized unit boundary-source operator; rows/columns are surface then CMB.
    load : numpy.ndarray, shape (2,)
        Fixed normalized internal-sheet potential coefficients in the same ordering.

    Raises
    ------
    TypeError
        The harmonic degree/mode or density power has an invalid type, or a real input
        cannot be converted.
    ValueError
        A numeric input is nonfinite, a required positive value is nonpositive, or the
        harmonic degree/mode, radius ordering or source support is invalid.

    Notes
    -----
    Radii and inputs are nondimensional. This kernel factors out 4*pi*G/g and does not take density
    contrasts or return SI potential. Boundary source coefficients must share a harmonic
    normalization.
    """

    radius_inner, radius_outer, degree = _validate_geometry(
        radius_inner, radius_outer, harmonic_degree
    )
    load_coefficient = _finite_float(internal_load_coefficient, "internal_load_coefficient")
    denominator = float(2 * degree + 1)

    def normalized_sheet(source_radius: float, target_radius: float) -> float:
        return float(
            source_radius
            * spherical_sheet_radial_factor(
                source_radius=source_radius,
                target_radius=target_radius,
                harmonic_degree=degree,
            )
            / denominator
        )

    operator = np.array(
        [
            [
                normalized_sheet(radius_outer, radius_outer),
                normalized_sheet(radius_inner, radius_outer),
            ],
            [
                normalized_sheet(radius_outer, radius_inner),
                normalized_sheet(radius_inner, radius_inner),
            ],
        ],
        dtype=float,
    )

    load = np.zeros(2, dtype=float)
    if internal_load_radius is None:
        if load_coefficient != 0.0:
            raise ValueError("internal_load_radius is required for a nonzero internal load.")
    else:
        load_radius = _positive_float(internal_load_radius, "internal_load_radius")
        if not radius_inner < load_radius < radius_outer:
            raise ValueError("internal_load_radius must lie strictly between the shell radii.")
        load = -load_coefficient * np.array(
            [
                normalized_sheet(load_radius, radius_outer),
                normalized_sheet(load_radius, radius_inner),
            ],
            dtype=float,
        )
    return operator, load


def spherical_physical_shell_response(
    *, radius_inner, radius_outer, harmonic_degree,
    outer_topography, inner_topography, outer_density_contrast,
    inner_density_contrast, outer_gravity, inner_gravity,
    gravitational_constant, internal_load_radius=None,
    internal_surface_density=0.0, volume_density=0.0, density_power=2,
    include_self_gravity=False,
):
    r"""Physical potential/geoid from radial-outward displacement coefficients.

    Radii and densities have caller-selected consistent physical units.
    Interface contrasts are rho_smaller_radius-rho_larger_radius. All sources
    and displacements use one real harmonic. Unlike the normalized Zhong API,
    this returns Phi and Phi/g with G and both gravity values retained.
    Linear interface feedback solves (I-diag(1/g)*K)h=h_dynamic+Phi_load/g.
    The initial Stokes stress must not already contain this gravity feedback.

    Parameters
    ----------
    radius_inner : real
        Finite positive inner source/shell radius in consistent length units.
    radius_outer : real
        Finite positive outer source/shell radius, strictly larger than radius_inner.
    harmonic_degree : int
        Nonnegative spherical-harmonic degree; the pure radial kernels include degree zero.
    outer_topography : real
        Finite outward-positive hydrodynamic displacement coefficient at the outer boundary,
        in length units.
    inner_topography : real
        Finite outward-positive hydrodynamic displacement coefficient at the inner boundary,
        in the same length units.
    outer_density_contrast : real
        Finite signed outer density contrast rho_smaller_radius-rho_larger_radius, in
        mass/volume units.
    inner_density_contrast : real
        Corresponding finite signed inner density contrast, in the same units.
    outer_gravity : real
        Finite positive outer reference gravity magnitude, in length/time^2.
    inner_gravity : real
        Finite positive inner reference gravity magnitude, in length/time^2.
    gravitational_constant : real
        Finite positive G in units consistent with radii and densities. No unit conversion
        is performed.
    internal_load_radius : real or None, default=None
        Internal sheet radius, strictly inside the shell when supplied. Required for a
        nonzero internal load; default None.
    internal_surface_density : real, default=0.0
        Finite signed internal sheet-density coefficient in mass/area units; default 0.
    volume_density : real, default=0.0
        Finite signed compact volume-density amplitude at radius_outer, in mass/volume
        units; default 0 disables it.
    density_power : int, default=2
        Integer k in the compact source rho(r)=density_amplitude*(r/radius_outer)**k;
        default 2. Signed powers are supported on positive source radii.
    include_self_gravity : bool, default=False
        Enable linear two-interface potential/displacement feedback; default False. The
        original Stokes stresses must not already include that feedback.

    Returns
    -------
    dict
        outer_topography/inner_topography, outer_potential/inner_potential and
        outer_geoid/inner_geoid in physical input units. Feedback adds corresponding
        ``*_self_gravity`` keys plus matrix_condition_number and matrix_residual_norm.

    Raises
    ------
    TypeError
        The harmonic degree/mode or density power has an invalid type, or a real input
        cannot be converted.
    ValueError
        A numeric input is nonfinite, a required positive value is nonpositive, or the
        harmonic degree/mode, radius ordering or source support is invalid.
    numpy.linalg.LinAlgError
        The feedback matrix is singular or numerically ill-conditioned.

    Notes
    -----
    The convention is laplacian(Phi)=-4*pi*G*rho. All density coefficients and returned coefficients
    share the selected angular basis/normalization. No mesh, MPI operation or PDE solve is
    performed. Geoid is Phi/g at each boundary, with length units. Feedback solves
    (I-diag(1/g)*K)h=h_dynamic+Phi_load/g. Unlike the normalized Zhong API, G and both gravities are
    retained. density_power is used only for a nonzero volume source.
    """
    ri, ro, ell = _validate_geometry(radius_inner,radius_outer,harmonic_degree)
    G = _positive_float(gravitational_constant,'gravitational_constant')
    contrasts = np.array([_finite_float(outer_density_contrast,'outer_density_contrast'),
                          _finite_float(inner_density_contrast,'inner_density_contrast')])
    gravity = np.array([_positive_float(outer_gravity,'outer_gravity'),
                        _positive_float(inner_gravity,'inner_gravity')])
    h = np.array([_finite_float(outer_topography,'outer_topography'),
                  _finite_float(inner_topography,'inner_topography')])
    sigma = _finite_float(internal_surface_density,'internal_surface_density')
    rho = _finite_float(volume_density,'volume_density')
    if not isinstance(include_self_gravity,bool):
        raise TypeError('include_self_gravity must be bool.')
    if internal_load_radius is None and sigma:
        raise ValueError('internal_load_radius is required for a nonzero sheet.')
    if internal_load_radius is not None:
        source = _positive_float(internal_load_radius,'internal_load_radius')
        if not ri < source < ro:raise ValueError('Internal sheet must lie strictly inside the shell.')
    targets = (ro,ri)
    operator = np.array([[spherical_sheet_potential_coefficient(
        source_radius=source_radius,target_radius=target,harmonic_degree=ell,
        surface_density_coefficient=contrast,gravitational_constant=G)
        for source_radius,contrast in zip(targets,contrasts)] for target in targets])
    load=np.zeros(2)
    for row,target in enumerate(targets):
        if sigma:
            load[row]+=spherical_sheet_potential_coefficient(source_radius=source,
                target_radius=target,harmonic_degree=ell,surface_density_coefficient=sigma,
                gravitational_constant=G)
        if rho:
            load[row]+=spherical_volume_density_potential_coefficient(radius_inner=ri,
                radius_outer=ro,harmonic_degree=ell,density_power=density_power,
                density_amplitude=rho,target_radius=target,gravitational_constant=G)
    potential=operator@h+load
    result=dict(outer_topography=float(h[0]),inner_topography=float(h[1]),
                outer_potential=float(potential[0]),inner_potential=float(potential[1]),
                outer_geoid=float(potential[0]/gravity[0]),inner_geoid=float(potential[1]/gravity[1]))
    if include_self_gravity:
        system=np.eye(2)-operator/gravity[:,None]
        rhs=h+load/gravity
        condition=float(np.linalg.cond(system))
        if not np.isfinite(condition) or condition>1/np.finfo(float).eps:
            raise np.linalg.LinAlgError('Singular spherical interface feedback system.')
        corrected=np.linalg.solve(system,rhs)
        corrected_potential=operator@corrected+load
        result.update(outer_topography_self_gravity=float(corrected[0]),
                      inner_topography_self_gravity=float(corrected[1]),
                      outer_potential_self_gravity=float(corrected_potential[0]),
                      inner_potential_self_gravity=float(corrected_potential[1]),
                      outer_geoid_self_gravity=float(corrected_potential[0]/gravity[0]),
                      inner_geoid_self_gravity=float(corrected_potential[1]/gravity[1]),
                      matrix_condition_number=condition,
                      matrix_residual_norm=float(np.linalg.norm(system@corrected-rhs)))
    return result


def _spherical_shell_geoid_operator(
    radius_inner: float,
    radius_outer: float,
    harmonic_degree: int,
    internal_load_radius: float | None,
    internal_load_coefficient: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the canonical analytical sheet operator and fixed-load vector."""
    return spherical_shell_potential_operator(
        radius_inner=radius_inner,
        radius_outer=radius_outer,
        harmonic_degree=harmonic_degree,
        internal_load_radius=internal_load_radius,
        internal_load_coefficient=internal_load_coefficient,
    )


def spherical_shell_geoid_response(
    *,
    radius_inner: float,
    radius_outer: float,
    harmonic_degree: int,
    surface_topography_coefficient: float,
    cmb_topography_coefficient: float,
    internal_load_radius: float | None = None,
    internal_load_coefficient: float = 0.0,
) -> GeoidResponse:
    r"""Return normalized spherical-shell geoid coefficients without feedback.

    Parameters
    ----------
    radius_inner, radius_outer : float
        Positive nondimensional shell radii, ordered inner < outer. They must
        use the same planetary reference-radius scaling as the source coefficients.
    harmonic_degree : int
        Nonnegative degree, including zero for these coefficient-only kernels.
    surface_topography_coefficient, cmb_topography_coefficient : float
        Finite signed nondimensional topography coefficients in a common harmonic
        basis, ordered surface then core-mantle boundary.
    internal_load_radius : float or None, default=None
        Nondimensional sheet radius strictly inside the shell when provided.
        Required for nonzero ``internal_load_coefficient``.
    internal_load_coefficient : float, default=0
        Finite normalized outward radial-load coefficient. It enters the fixed
        potential load with a negative sheet-density sign.

    Returns
    -------
    GeoidResponse
        Surface/CMB geoid coefficients with ``4*pi*G/g`` factored out. No physical
        units are converted and no Stokes or MPI operation is performed.

    Raises
    ------
    TypeError
        Degree is not an integer or radii/load data cannot be converted to numbers.
    ValueError
        Degree, radius ordering, finite coefficients or internal sheet radius is invalid.

    Notes
    -----
    All source coefficients must share one harmonic normalization. The radial
    kernel depends on degree rather than order, so consistently normalized
    coefficients of any real harmonic order may be used. This normalized response
    is distinct from the physical ``Phi/g`` sheet kernels."""

    operator, load = _spherical_shell_geoid_operator(
        radius_inner,
        radius_outer,
        harmonic_degree,
        internal_load_radius,
        internal_load_coefficient,
    )
    topography = np.array(
        [surface_topography_coefficient, cmb_topography_coefficient],
        dtype=float,
    )
    if not np.all(np.isfinite(topography)):
        raise ValueError("The topography coefficients must be finite.")
    geoid = operator @ topography + load
    return GeoidResponse(float(geoid[0]), float(geoid[1]))


def spherical_shell_self_gravity_response(
    *,
    radius_inner: float,
    radius_outer: float,
    harmonic_degree: int,
    surface_topography_coefficient: float,
    cmb_topography_coefficient: float,
    surface_density_contrast: float,
    cmb_density_contrast: float,
    planet_radius: float,
    gravity: float,
    internal_load_radius: float | None = None,
    internal_load_coefficient: float = 0.0,
    gravitational_constant: float = 6.67430e-11,
) -> SelfGravityResponse:
    r"""Return normalized spherical response after two-interface self-gravity feedback.

    Parameters
    ----------
    radius_inner, radius_outer : float
        Positive nondimensional shell radii, ordered inner < outer. They must
        use the same planetary reference-radius scaling as the source coefficients.
    harmonic_degree : int
        Nonnegative degree, including zero for these coefficient-only kernels.
    surface_topography_coefficient, cmb_topography_coefficient : float
        Finite signed nondimensional topography coefficients in a common harmonic
        basis, ordered surface then core-mantle boundary.
    internal_load_radius : float or None, default=None
        Nondimensional sheet radius strictly inside the shell when provided.
        Required for nonzero ``internal_load_coefficient``.
    internal_load_coefficient : float, default=0
        Finite normalized outward radial-load coefficient. It enters the fixed
        potential load with a negative sheet-density sign.
    surface_density_contrast, cmb_density_contrast : float
        Finite signed density contrasts in kg/m^3, supplying the signs of the
        surface/CMB diagonal feedback factors.
    planet_radius : float
        Finite positive reference planetary radius in metres.
    gravity : float
        Finite positive reference gravity in m/s^2.
    gravitational_constant : float, default=6.67430e-11
        Finite positive SI gravitational constant in m^3/(kg*s^2).

    Returns
    -------
    SelfGravityResponse
        Corrected nondimensional topography and normalized geoid coefficients,
        feedback factors and matrix diagnostics. Physical constants set the
        dimensionless feedback; returned coefficients remain normalized.

    Raises
    ------
    TypeError
        Degree has an invalid type or radius/load data cannot be converted to numbers.
    ValueError
        Geometry, coefficient, contrast or positive physical-constant requirements fail.
    numpy.linalg.LinAlgError
        The two-interface feedback matrix is singular.

    Notes
    -----
    With the normalized kernel ``K`` and fixed load ``b``, solve
    ``(I-diag(q)*K)*h_corrected=h+diag(q)*b``, where
    ``q=4*pi*G*planet_radius*Delta_rho/gravity``. All coefficients must share
    the input harmonic normalization. This pure coefficient operation changes no
    velocity and performs no MPI communication."""

    operator, load = _spherical_shell_geoid_operator(
        radius_inner,
        radius_outer,
        harmonic_degree,
        internal_load_radius,
        internal_load_coefficient,
    )
    topography = np.array(
        [surface_topography_coefficient, cmb_topography_coefficient],
        dtype=float,
    )
    density_contrasts = np.array([surface_density_contrast, cmb_density_contrast], dtype=float)
    physical_constants = np.array([planet_radius, gravity, gravitational_constant], dtype=float)
    if not np.all(np.isfinite(topography)):
        raise ValueError("The topography coefficients must be finite.")
    if not np.all(np.isfinite(density_contrasts)):
        raise ValueError("Density contrasts must be finite.")
    if not np.all(np.isfinite(physical_constants)) or np.any(physical_constants <= 0.0):
        raise ValueError("Physical constants must be finite and positive.")

    gravity_scale = 4.0 * np.pi * gravitational_constant * planet_radius / gravity
    q = gravity_scale * density_contrasts
    q_matrix = np.diag(q)
    system = np.eye(2) - q_matrix @ operator
    right_hand_side = topography + q_matrix @ load
    corrected_topography = np.linalg.solve(system, right_hand_side)
    corrected_geoid = operator @ corrected_topography + load

    return SelfGravityResponse(
        surface_topography=float(corrected_topography[0]),
        cmb_topography=float(corrected_topography[1]),
        surface_geoid=float(corrected_geoid[0]),
        cmb_geoid=float(corrected_geoid[1]),
        q_surface=float(q[0]),
        q_cmb=float(q[1]),
        matrix_condition_number=float(np.linalg.cond(system)),
        matrix_residual_norm=float(np.linalg.norm(system @ corrected_topography - right_hand_side)),
    )
