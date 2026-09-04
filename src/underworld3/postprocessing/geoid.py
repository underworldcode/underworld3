r"""Spherical and cylindrical geoid and self-gravity response functions.

The pure coefficient functions in this module are independent of a particular
Stokes discretisation. They combine boundary and optional internal-load
coefficients through the appropriate radial Green's function. Separate
convenience adapters obtain topography coefficients from completed
rotated-free-slip Stokes solves.

The cylindrical sheet kernel follows Simons (1996), Appendix B, with its
normalisation fixed directly by potential continuity and the radial-derivative
jump condition.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any

from mpi4py import MPI
import numpy as np


__all__ = [
    "GeoidResponse",
    "SelfGravityResponse",
    "SphericalShellResponse",
    "CylindricalGravityResponse",
    "CylindricalSelfGravityResponse",
    "CylindricalAnnulusResponse",
    "spherical_shell_geoid_response",
    "spherical_shell_self_gravity_response",
    "spherical_shell_response_from_rotated_stokes",
    "cylindrical_sheet_potential_coefficient",
    "cylindrical_sheet_radial_derivative_coefficient",
    "cylindrical_annulus_potential_operator",
    "cylindrical_annulus_geoid_response",
    "cylindrical_annulus_self_gravity_response",
    "cylindrical_cosine_boundary_coefficient",
    "cylindrical_annulus_response_from_rotated_stokes",
]


@dataclass(frozen=True)
class GeoidResponse:
    """Surface and CMB geoid coefficients without self-gravity feedback."""

    surface_geoid: float
    cmb_geoid: float


@dataclass(frozen=True)
class SelfGravityResponse:
    """Self-gravity-corrected topography and geoid coefficients."""

    surface_topography: float
    cmb_topography: float
    surface_geoid: float
    cmb_geoid: float
    q_surface: float
    q_cmb: float


@dataclass(frozen=True)
class SphericalShellResponse:
    """Spherical-shell topography and corresponding geoid coefficients."""

    surface_topography: float
    cmb_topography: float
    surface_geoid: float
    cmb_geoid: float
    self_gravity: SelfGravityResponse | None = None


@dataclass(frozen=True)
class CylindricalGravityResponse:
    """Potential and geoid coefficients at the outer and inner boundaries."""

    outer_potential: float
    inner_potential: float
    outer_geoid: float
    inner_geoid: float


@dataclass(frozen=True)
class CylindricalSelfGravityResponse:
    """Self-gravity-corrected cylindrical response coefficients."""

    outer_topography: float
    inner_topography: float
    outer_potential: float
    inner_potential: float
    outer_geoid: float
    inner_geoid: float
    q_outer: float
    q_inner: float
    matrix_residual_norm: float


@dataclass(frozen=True)
class CylindricalAnnulusResponse:
    """Rotated-Stokes topography and cylindrical gravity coefficients."""

    outer_reaction: float
    inner_reaction: float
    outer_reaction_mean: float
    inner_reaction_mean: float
    outer_topography: float
    inner_topography: float
    outer_potential: float
    inner_potential: float
    outer_geoid: float
    inner_geoid: float
    self_gravity: CylindricalSelfGravityResponse | None = None


def _validate_geometry(
    radius_inner: float,
    radius_outer: float,
    harmonic_degree: int,
) -> tuple[float, float, int]:
    try:
        ri = float(radius_inner)
        ro = float(radius_outer)
    except (TypeError, ValueError) as error:
        raise TypeError("The shell radii must be real numbers.") from error
    if not np.all(np.isfinite((ri, ro))) or not 0.0 < ri < ro:
        raise ValueError(
            "Expected finite radii ordered as 0 < radius_inner < radius_outer."
        )
    if isinstance(harmonic_degree, bool) or not isinstance(harmonic_degree, Integral):
        raise TypeError("harmonic_degree must be an integer.")
    degree = int(harmonic_degree)
    if degree < 0:
        raise ValueError("harmonic_degree must be non-negative.")
    return ri, ro, degree


def _spherical_shell_geoid_operator(
    radius_inner: float,
    radius_outer: float,
    harmonic_degree: int,
    internal_load_radius: float | None,
    internal_load_coefficient: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the boundary-topography operator and fixed-load vector."""

    ri, ro, degree = _validate_geometry(
        radius_inner,
        radius_outer,
        harmonic_degree,
    )
    internal_load_coefficient = float(internal_load_coefficient)
    if not np.isfinite(internal_load_coefficient):
        raise ValueError("internal_load_coefficient must be finite.")

    rint = None
    if internal_load_radius is not None:
        try:
            rint = float(internal_load_radius)
        except (TypeError, ValueError) as error:
            raise TypeError(
                "internal_load_radius must be a real number or None."
            ) from error
        if not np.isfinite(rint) or not ri < rint < ro:
            raise ValueError(
                "internal_load_radius must lie strictly between the shell radii."
            )
    elif internal_load_coefficient != 0.0:
        raise ValueError(
            "internal_load_radius is required when internal_load_coefficient is nonzero."
        )

    denominator = float(2 * degree + 1)
    operator = np.array(
        [
            [
                ro / denominator,
                ri * (ri / ro) ** (degree + 1) / denominator,
            ],
            [
                ro * (ri / ro) ** degree / denominator,
                ri / denominator,
            ],
        ],
        dtype=float,
    )
    load = np.zeros(2, dtype=float)
    if rint is not None and internal_load_coefficient != 0.0:
        load = -internal_load_coefficient * np.array(
            [
                rint * (rint / ro) ** (degree + 1) / denominator,
                rint * (ri / rint) ** degree / denominator,
            ],
            dtype=float,
        )
    return operator, load


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
    r"""Return spherical-shell geoid coefficients without self-gravity.

    All source coefficients must use one consistent spherical-harmonic
    normalisation.  The radial potential kernel depends on degree but not order,
    so coefficients for any order :math:`m` may be used.  A positive optional
    internal-load coefficient follows the outward radial-load convention and
    enters the potential with the opposite sign to boundary topography.
    """

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
    r"""Return self-gravity-corrected spherical-shell response coefficients.

    Density contrasts are signed and in kg/m³, ``planet_radius`` is in metres,
    ``gravity`` is in m/s², and ``gravitational_constant`` is in SI units.
    Shell radii and response coefficients remain nondimensional.
    """

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
    density_contrasts = np.array(
        [surface_density_contrast, cmb_density_contrast], dtype=float
    )
    physical_constants = np.array(
        [planet_radius, gravity, gravitational_constant], dtype=float
    )
    if not np.all(np.isfinite(topography)):
        raise ValueError("The topography coefficients must be finite.")
    if not np.all(np.isfinite(density_contrasts)):
        raise ValueError("Density contrasts must be finite.")
    if not np.all(np.isfinite(physical_constants)) or np.any(physical_constants <= 0.0):
        raise ValueError("Physical constants must be finite and positive.")

    gravity_scale = 4.0 * np.pi * gravitational_constant * planet_radius / gravity
    q = gravity_scale * density_contrasts
    q_matrix = np.diag(q)
    corrected_topography = np.linalg.solve(
        np.eye(2) - q_matrix @ operator,
        topography + q_matrix @ load,
    )
    corrected_geoid = operator @ corrected_topography + load

    return SelfGravityResponse(
        surface_topography=float(corrected_topography[0]),
        cmb_topography=float(corrected_topography[1]),
        surface_geoid=float(corrected_geoid[0]),
        cmb_geoid=float(corrected_geoid[1]),
        q_surface=float(q[0]),
        q_cmb=float(q[1]),
    )


def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real number.") from error
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _positive_float(value: Any, name: str) -> float:
    result = _finite_float(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return result


def _validate_cylindrical_mode(wavenumber: int) -> int:
    if isinstance(wavenumber, bool) or not isinstance(wavenumber, Integral):
        raise TypeError("wavenumber must be an integer.")
    mode = int(wavenumber)
    if mode < 0:
        raise ValueError("wavenumber must be non-negative.")
    if mode == 0:
        raise ValueError(
            "The n=0 cylindrical mode has a logarithmic radial solution and "
            "requires a potential gauge."
        )
    return mode


def _validate_cylindrical_annulus(
    radius_inner: float,
    radius_outer: float,
    wavenumber: int,
) -> tuple[float, float, int]:
    radius_inner = _positive_float(radius_inner, "radius_inner")
    radius_outer = _positive_float(radius_outer, "radius_outer")
    if radius_inner >= radius_outer:
        raise ValueError("Expected radius_inner < radius_outer.")
    return (
        radius_inner,
        radius_outer,
        _validate_cylindrical_mode(wavenumber),
    )


def cylindrical_sheet_potential_coefficient(
    *,
    source_radius: float,
    target_radius: float,
    wavenumber: int,
    surface_density_coefficient: float,
    gravitational_constant: float = 1.0,
) -> float:
    r"""Return one cylindrical mass sheet's potential coefficient.

    The result multiplies the unnormalised real Fourier basis
    :math:`\cos(n\theta)`. For :math:`n\geq 1`, a sheet at radius :math:`r_s`
    has

    .. math::

       \Phi_n(r_s) = \frac{2\pi G r_s\sigma_n}{n},

    with radial factors :math:`(r/r_s)^n` inside and :math:`(r_s/r)^n`
    outside. Positive density gives positive potential under the convention
    :math:`\nabla^2\Phi=-4\pi G\rho` and :math:`\mathbf{g}=\nabla\Phi`.

    Radii, density, and ``gravitational_constant`` may be dimensional or
    nondimensional, but they must use one mutually consistent unit system.
    The axisymmetric ``n=0`` mode is intentionally excluded because its
    logarithmic exterior branch requires a separate potential gauge.

    Parameters
    ----------
    source_radius, target_radius : real
        Positive source-sheet and evaluation radii.
    wavenumber : int
        Positive azimuthal Fourier wavenumber.
    surface_density_coefficient : real
        Sheet-density coefficient multiplying :math:`\cos(n\theta)`.
    gravitational_constant : real, default=1
        Positive gravitational constant in the selected unit system.

    Returns
    -------
    float
        Potential coefficient at ``target_radius``.
    """

    source_radius = _positive_float(source_radius, "source_radius")
    target_radius = _positive_float(target_radius, "target_radius")
    mode = _validate_cylindrical_mode(wavenumber)
    density = _finite_float(
        surface_density_coefficient,
        "surface_density_coefficient",
    )
    gravity_constant = _positive_float(
        gravitational_constant,
        "gravitational_constant",
    )

    radial_factor = (
        min(source_radius, target_radius) / max(source_radius, target_radius)
    ) ** mode
    source_amplitude = 2.0 * np.pi * gravity_constant * source_radius * density / mode
    return float(source_amplitude * radial_factor)


def cylindrical_sheet_radial_derivative_coefficient(
    *,
    source_radius: float,
    target_radius: float,
    wavenumber: int,
    surface_density_coefficient: float,
    gravitational_constant: float = 1.0,
    source_side: str | None = None,
) -> float:
    r"""Return :math:`d\Phi_n/dr` on either side of a cylindrical sheet.

    At ``target_radius == source_radius``, ``source_side`` must be
    ``"inside"`` or ``"outside"`` because the derivative is discontinuous.
    The returned branches satisfy
    :math:`[d\Phi_n/dr]_{outside-inside}=-4\pi G\sigma_n`.

    Parameters
    ----------
    source_radius, target_radius : real
        Positive source-sheet and evaluation radii.
    wavenumber : int
        Positive azimuthal Fourier wavenumber.
    surface_density_coefficient : real
        Sheet-density coefficient multiplying :math:`\cos(n\theta)`.
    gravitational_constant : real, default=1
        Positive gravitational constant in the selected unit system.
    source_side : {"inside", "outside"}, optional
        Radial branch when evaluating exactly on the sheet.

    Returns
    -------
    float
        Radial derivative coefficient at ``target_radius``.
    """

    source_radius = _positive_float(source_radius, "source_radius")
    target_radius = _positive_float(target_radius, "target_radius")
    mode = _validate_cylindrical_mode(wavenumber)
    potential = cylindrical_sheet_potential_coefficient(
        source_radius=source_radius,
        target_radius=target_radius,
        wavenumber=mode,
        surface_density_coefficient=surface_density_coefficient,
        gravitational_constant=gravitational_constant,
    )

    if target_radius == source_radius:
        if source_side not in ("inside", "outside"):
            raise ValueError(
                "source_side must be 'inside' or 'outside' at the sheet radius."
            )
        branch_sign = 1.0 if source_side == "inside" else -1.0
    else:
        if source_side is not None:
            raise ValueError(
                "source_side is only valid when target_radius equals source_radius."
            )
        branch_sign = 1.0 if target_radius < source_radius else -1.0
    return float(branch_sign * mode * potential / target_radius)


def cylindrical_annulus_potential_operator(
    *,
    radius_inner: float,
    radius_outer: float,
    wavenumber: int,
    outer_density_contrast: float,
    inner_density_contrast: float,
    gravitational_constant: float = 1.0,
) -> np.ndarray:
    r"""Return the two-boundary operator :math:`\Phi=G_n h`.

    Rows are target boundaries ``[outer, inner]`` and columns are topographic
    sheet sources ``[outer, inner]``. Density contrasts are signed as density
    on the smaller-radius side minus density on the larger-radius side. Thus
    positive outward topography creates sheet density
    :math:`\Delta\rho\,h`.

    Parameters
    ----------
    radius_inner, radius_outer : real
        Positive annulus radii ordered from inner to outer.
    wavenumber : int
        Positive azimuthal Fourier wavenumber.
    outer_density_contrast, inner_density_contrast : real
        Signed density contrasts at the two boundaries.
    gravitational_constant : real, default=1
        Positive gravitational constant in the selected unit system.

    Returns
    -------
    numpy.ndarray
        Two-by-two potential operator with targets in rows and sources in
        columns, both ordered ``[outer, inner]``.
    """

    radius_inner, radius_outer, mode = _validate_cylindrical_annulus(
        radius_inner,
        radius_outer,
        wavenumber,
    )
    density_contrasts = np.array(
        [
            _finite_float(
                outer_density_contrast,
                "outer_density_contrast",
            ),
            _finite_float(
                inner_density_contrast,
                "inner_density_contrast",
            ),
        ]
    )
    gravity_constant = _positive_float(
        gravitational_constant,
        "gravitational_constant",
    )
    source_radii = (radius_outer, radius_inner)
    target_radii = (radius_outer, radius_inner)

    operator = np.empty((2, 2), dtype=float)
    for row, target_radius in enumerate(target_radii):
        for column, (source_radius, density) in enumerate(
            zip(source_radii, density_contrasts)
        ):
            operator[row, column] = cylindrical_sheet_potential_coefficient(
                source_radius=source_radius,
                target_radius=target_radius,
                wavenumber=mode,
                surface_density_coefficient=density,
                gravitational_constant=gravity_constant,
            )
    return operator


def _cylindrical_internal_load_vector(
    *,
    radius_inner: float,
    radius_outer: float,
    wavenumber: int,
    internal_load_radius: float | None,
    internal_surface_density_coefficient: float,
    gravitational_constant: float,
) -> np.ndarray:
    """Return boundary potential coefficients from an internal mass sheet."""

    load_density = _finite_float(
        internal_surface_density_coefficient,
        "internal_surface_density_coefficient",
    )
    load = np.zeros(2, dtype=float)
    if internal_load_radius is None:
        if load_density != 0.0:
            raise ValueError(
                "internal_load_radius is required for a nonzero internal load."
            )
        return load

    internal_load_radius = _positive_float(
        internal_load_radius,
        "internal_load_radius",
    )
    if not radius_inner < internal_load_radius < radius_outer:
        raise ValueError("internal_load_radius must lie strictly inside the annulus.")
    for index, target_radius in enumerate((radius_outer, radius_inner)):
        load[index] = cylindrical_sheet_potential_coefficient(
            source_radius=internal_load_radius,
            target_radius=target_radius,
            wavenumber=wavenumber,
            surface_density_coefficient=load_density,
            gravitational_constant=gravitational_constant,
        )
    return load


def cylindrical_annulus_geoid_response(
    *,
    radius_inner: float,
    radius_outer: float,
    wavenumber: int,
    outer_topography_coefficient: float,
    inner_topography_coefficient: float,
    outer_density_contrast: float,
    inner_density_contrast: float,
    outer_reference_gravity: float,
    inner_reference_gravity: float,
    internal_load_radius: float | None = None,
    internal_surface_density_coefficient: float = 0.0,
    gravitational_constant: float = 1.0,
) -> CylindricalGravityResponse:
    r"""Assemble annulus potential and geoid coefficients for one mode.

    Potential and topography retain their physical signs. Geoid is defined as
    :math:`N_n=\Phi_n/g_{reference}` independently at the outer and inner
    boundaries. The optional internal source is a cylindrical sheet-density
    coefficient in the same Fourier normalisation and unit system.

    Parameters
    ----------
    radius_inner, radius_outer : real
        Positive annulus radii ordered from inner to outer.
    wavenumber : int
        Positive azimuthal Fourier wavenumber.
    outer_topography_coefficient, inner_topography_coefficient : real
        Signed boundary topography coefficients.
    outer_density_contrast, inner_density_contrast : real
        Signed density contrasts at the two boundaries.
    outer_reference_gravity, inner_reference_gravity : real
        Positive gravity magnitudes used to convert potential to geoid.
    internal_load_radius : real, optional
        Radius of an internal sheet, strictly inside the annulus.
    internal_surface_density_coefficient : real, default=0
        Density coefficient of the optional internal sheet.
    gravitational_constant : real, default=1
        Positive gravitational constant in the selected unit system.

    Returns
    -------
    CylindricalGravityResponse
        Outer and inner potential and geoid coefficients.
    """

    radius_inner, radius_outer, mode = _validate_cylindrical_annulus(
        radius_inner,
        radius_outer,
        wavenumber,
    )
    topography = np.array(
        [
            _finite_float(
                outer_topography_coefficient,
                "outer_topography_coefficient",
            ),
            _finite_float(
                inner_topography_coefficient,
                "inner_topography_coefficient",
            ),
        ],
        dtype=float,
    )
    reference_gravity = np.array(
        [
            _positive_float(
                outer_reference_gravity,
                "outer_reference_gravity",
            ),
            _positive_float(
                inner_reference_gravity,
                "inner_reference_gravity",
            ),
        ],
        dtype=float,
    )
    gravity_constant = _positive_float(
        gravitational_constant,
        "gravitational_constant",
    )
    operator = cylindrical_annulus_potential_operator(
        radius_inner=radius_inner,
        radius_outer=radius_outer,
        wavenumber=mode,
        outer_density_contrast=outer_density_contrast,
        inner_density_contrast=inner_density_contrast,
        gravitational_constant=gravity_constant,
    )
    load = _cylindrical_internal_load_vector(
        radius_inner=radius_inner,
        radius_outer=radius_outer,
        wavenumber=mode,
        internal_load_radius=internal_load_radius,
        internal_surface_density_coefficient=(internal_surface_density_coefficient),
        gravitational_constant=gravity_constant,
    )

    potential = operator @ topography + load
    geoid = potential / reference_gravity
    return CylindricalGravityResponse(
        outer_potential=float(potential[0]),
        inner_potential=float(potential[1]),
        outer_geoid=float(geoid[0]),
        inner_geoid=float(geoid[1]),
    )


def cylindrical_annulus_self_gravity_response(
    *,
    radius_inner: float,
    radius_outer: float,
    wavenumber: int,
    outer_topography_coefficient: float,
    inner_topography_coefficient: float,
    outer_density_contrast: float,
    inner_density_contrast: float,
    outer_reference_gravity: float,
    inner_reference_gravity: float,
    internal_load_radius: float | None = None,
    internal_surface_density_coefficient: float = 0.0,
    gravitational_constant: float = 1.0,
    outer_feedback_factor: float | None = None,
    inner_feedback_factor: float | None = None,
) -> CylindricalSelfGravityResponse:
    r"""Return the two-boundary cylindrical self-gravity correction.

    Holding the hydrodynamic traction fixed gives
    :math:`h_{sg}=h+Q\Phi_{sg}`. For positive reference-gravity magnitudes the
    default factors are :math:`Q=diag(1/g_o,1/g_i)`. Explicit factors may be
    supplied to test a signed convention or disable either feedback row. The
    solved equation is

    .. math::

       (I-QG_n)h_{sg}=h+Q\phi_{load}.

    Parameters
    ----------
    radius_inner, radius_outer : real
        Positive annulus radii ordered from inner to outer.
    wavenumber : int
        Positive azimuthal Fourier wavenumber.
    outer_topography_coefficient, inner_topography_coefficient : real
        Hydrodynamic topography coefficients before self-gravity feedback.
    outer_density_contrast, inner_density_contrast : real
        Signed density contrasts at the two boundaries.
    outer_reference_gravity, inner_reference_gravity : real
        Positive reference-gravity magnitudes.
    internal_load_radius : real, optional
        Radius of an internal sheet, strictly inside the annulus.
    internal_surface_density_coefficient : real, default=0
        Density coefficient of the optional internal sheet.
    gravitational_constant : real, default=1
        Positive gravitational constant in the selected unit system.
    outer_feedback_factor, inner_feedback_factor : real, optional
        Explicit diagonal entries of :math:`Q`; defaults are reciprocal
        reference-gravity magnitudes.

    Returns
    -------
    CylindricalSelfGravityResponse
        Corrected topography, potential, geoid, feedback, and residual values.
    """

    radius_inner, radius_outer, mode = _validate_cylindrical_annulus(
        radius_inner,
        radius_outer,
        wavenumber,
    )
    reference_gravity = np.array(
        [
            _positive_float(
                outer_reference_gravity,
                "outer_reference_gravity",
            ),
            _positive_float(
                inner_reference_gravity,
                "inner_reference_gravity",
            ),
        ],
        dtype=float,
    )
    topography = np.array(
        [
            _finite_float(
                outer_topography_coefficient,
                "outer_topography_coefficient",
            ),
            _finite_float(
                inner_topography_coefficient,
                "inner_topography_coefficient",
            ),
        ],
        dtype=float,
    )
    gravity_constant = _positive_float(
        gravitational_constant,
        "gravitational_constant",
    )
    operator = cylindrical_annulus_potential_operator(
        radius_inner=radius_inner,
        radius_outer=radius_outer,
        wavenumber=mode,
        outer_density_contrast=outer_density_contrast,
        inner_density_contrast=inner_density_contrast,
        gravitational_constant=gravity_constant,
    )
    load = _cylindrical_internal_load_vector(
        radius_inner=radius_inner,
        radius_outer=radius_outer,
        wavenumber=mode,
        internal_load_radius=internal_load_radius,
        internal_surface_density_coefficient=(internal_surface_density_coefficient),
        gravitational_constant=gravity_constant,
    )
    feedback = np.array(
        [
            (
                1.0 / reference_gravity[0]
                if outer_feedback_factor is None
                else _finite_float(
                    outer_feedback_factor,
                    "outer_feedback_factor",
                )
            ),
            (
                1.0 / reference_gravity[1]
                if inner_feedback_factor is None
                else _finite_float(
                    inner_feedback_factor,
                    "inner_feedback_factor",
                )
            ),
        ],
        dtype=float,
    )
    q_matrix = np.diag(feedback)
    system_matrix = np.eye(2) - q_matrix @ operator
    right_hand_side = topography + q_matrix @ load
    try:
        corrected_topography = np.linalg.solve(
            system_matrix,
            right_hand_side,
        )
    except np.linalg.LinAlgError as error:
        raise ValueError("The self-gravity feedback matrix is singular.") from error
    corrected_potential = operator @ corrected_topography + load
    corrected_geoid = corrected_potential / reference_gravity
    residual = system_matrix @ corrected_topography - right_hand_side

    return CylindricalSelfGravityResponse(
        outer_topography=float(corrected_topography[0]),
        inner_topography=float(corrected_topography[1]),
        outer_potential=float(corrected_potential[0]),
        inner_potential=float(corrected_potential[1]),
        outer_geoid=float(corrected_geoid[0]),
        inner_geoid=float(corrected_geoid[1]),
        q_outer=float(feedback[0]),
        q_inner=float(feedback[1]),
        matrix_residual_norm=float(np.linalg.norm(residual)),
    )


def _spherical_triangle_area(a, b, c, radius: float) -> float:
    determinant = abs(float(np.dot(a, np.cross(b, c))))
    denominator = float(1.0 + np.dot(a, b) + np.dot(b, c) + np.dot(c, a))
    return 2.0 * np.arctan2(determinant, denominator) * radius**2


def _project_spherical_harmonic_samples(
    coords: np.ndarray,
    values: np.ndarray,
    radius: float,
    harmonic_degree: int,
    response_sign: float,
) -> float:
    from scipy.spatial import ConvexHull

    unit_coords = coords / np.linalg.norm(coords, axis=1)[:, None]
    hull = ConvexHull(unit_coords, qhull_options="QJ")
    projected_integral = 0.0
    harmonic = np.polynomial.legendre.Legendre.basis(harmonic_degree)

    for simplex in hull.simplices:
        area = _spherical_triangle_area(
            unit_coords[simplex[0]],
            unit_coords[simplex[1]],
            unit_coords[simplex[2]],
            radius,
        )
        centroid = unit_coords[simplex].mean(axis=0)
        centroid /= np.linalg.norm(centroid)
        projected_integral += (
            area * float(values[simplex].mean()) * harmonic(np.clip(centroid[2], -1, 1))
        )

    harmonic_norm = 4.0 * np.pi / (2 * harmonic_degree + 1)
    return float(response_sign * projected_integral / (radius**2 * harmonic_norm))


def _rotated_topography_coefficient(
    stokes,
    boundary: str,
    radius: float,
    harmonic_degree: int,
    buoyancy_scale: float,
    response_sign: float,
) -> float:
    buoyancy_scale = float(buoyancy_scale)
    if not np.isfinite(buoyancy_scale) or buoyancy_scale == 0.0:
        raise ValueError("Boundary buoyancy scales must be finite and nonzero.")

    coords, sigma_nn = stokes.boundary_normal_traction(boundary, mass="auto")
    local_rows = np.column_stack(
        (
            np.asarray(coords, dtype=float),
            -np.asarray(sigma_nn, dtype=float) / buoyancy_scale,
        )
    )
    gathered_rows = MPI.COMM_WORLD.gather(local_rows, root=0)

    coefficient = None
    root_error = None
    if MPI.COMM_WORLD.rank == 0:
        try:
            if gathered_rows is None:
                raise RuntimeError("MPI gather returned no root payload.")
            nonempty_rows = [rows for rows in gathered_rows if rows.size]
            if not nonempty_rows:
                raise RuntimeError(f"No samples found on boundary {boundary!r}.")
            merged = {}
            for row in np.vstack(nonempty_rows):
                key = tuple(np.round(row[:3], 12))
                if key not in merged:
                    merged[key] = [row[:3].copy(), 0.0, 0]
                merged[key][1] += float(row[3])
                merged[key][2] += 1
            global_coords = np.array([item[0] for item in merged.values()])
            global_values = np.array([item[1] / item[2] for item in merged.values()])
            coefficient = _project_spherical_harmonic_samples(
                global_coords,
                global_values,
                radius,
                harmonic_degree,
                response_sign,
            )
        except Exception as error:
            root_error = f"{type(error).__name__}: {error}"

    root_error = MPI.COMM_WORLD.bcast(root_error, root=0)
    if root_error is not None:
        raise RuntimeError(f"Spherical-harmonic projection failed: {root_error}")
    return float(MPI.COMM_WORLD.bcast(coefficient, root=0))


def spherical_shell_response_from_rotated_stokes(
    *,
    stokes,
    radius_inner: float,
    radius_outer: float,
    harmonic_degree: int,
    internal_load_radius: float | None = None,
    internal_load_coefficient: float = 0.0,
    surface_boundary: str = "Upper",
    cmb_boundary: str = "Lower",
    surface_buoyancy_scale: float = 1.0,
    cmb_buoyancy_scale: float = 1.0,
    include_self_gravity: bool = False,
    surface_density_contrast: float | None = None,
    cmb_density_contrast: float | None = None,
    planet_radius: float | None = None,
    gravity: float | None = None,
    gravitational_constant: float = 6.67430e-11,
) -> SphericalShellResponse:
    r"""Compute spherical-shell response from a rotated-free-slip Stokes solve.

    Normal traction recovery is delegated to the existing
    :meth:`Stokes.boundary_normal_traction` implementation. This adapter only
    projects the two boundary responses onto the unnormalised
    axisymmetric :math:`P_l^0` harmonic.  Use the pure coefficient functions
    directly for other harmonic orders or topography-recovery methods. Density
    contrasts, planet radius, and gravity are required when
    ``include_self_gravity`` is true.
    """

    ri, ro, degree = _validate_geometry(
        radius_inner,
        radius_outer,
        harmonic_degree,
    )
    if not isinstance(include_self_gravity, bool):
        raise TypeError("include_self_gravity must be True or False.")
    if degree == 0:
        raise ValueError(
            "The rotated-Stokes adapter requires harmonic_degree >= 1 because "
            "boundary_normal_traction() removes the degree-zero mean."
        )
    if include_self_gravity:
        missing = [
            name
            for name, value in (
                ("surface_density_contrast", surface_density_contrast),
                ("cmb_density_contrast", cmb_density_contrast),
                ("planet_radius", planet_radius),
                ("gravity", gravity),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                "Self-gravity requires explicit values for " + ", ".join(missing) + "."
            )

    surface_topography = _rotated_topography_coefficient(
        stokes,
        surface_boundary,
        ro,
        degree,
        surface_buoyancy_scale,
        1.0,
    )
    cmb_topography = _rotated_topography_coefficient(
        stokes,
        cmb_boundary,
        ri,
        degree,
        cmb_buoyancy_scale,
        -1.0,
    )
    geoid = spherical_shell_geoid_response(
        radius_inner=ri,
        radius_outer=ro,
        harmonic_degree=degree,
        surface_topography_coefficient=surface_topography,
        cmb_topography_coefficient=cmb_topography,
        internal_load_radius=internal_load_radius,
        internal_load_coefficient=internal_load_coefficient,
    )
    self_gravity = None
    if include_self_gravity:
        assert surface_density_contrast is not None
        assert cmb_density_contrast is not None
        assert planet_radius is not None
        assert gravity is not None
        self_gravity = spherical_shell_self_gravity_response(
            radius_inner=ri,
            radius_outer=ro,
            harmonic_degree=degree,
            surface_topography_coefficient=surface_topography,
            cmb_topography_coefficient=cmb_topography,
            internal_load_radius=internal_load_radius,
            internal_load_coefficient=internal_load_coefficient,
            surface_density_contrast=float(surface_density_contrast),
            cmb_density_contrast=float(cmb_density_contrast),
            planet_radius=float(planet_radius),
            gravity=float(gravity),
            gravitational_constant=gravitational_constant,
        )

    return SphericalShellResponse(
        surface_topography=surface_topography,
        cmb_topography=cmb_topography,
        surface_geoid=geoid.surface_geoid,
        cmb_geoid=geoid.cmb_geoid,
        self_gravity=self_gravity,
    )


def _trapezoidal_integral(values: np.ndarray, coordinates: np.ndarray) -> float:
    """Integrate samples without requiring NumPy's version-specific helpers."""

    widths = np.diff(coordinates)
    averages = 0.5 * (values[:-1] + values[1:])
    return float(np.sum(widths * averages))


def cylindrical_cosine_boundary_coefficient(
    coords,
    values,
    wavenumber: int,
) -> tuple[float, float]:
    r"""Project sampled circular data onto :math:`\cos(n\theta)` and the mean.

    ``coords`` must contain at least three two-dimensional Cartesian boundary
    points. The samples may begin at any angle and need not include a duplicate
    endpoint; this function sorts and closes the periodic interval. This helper
    is intended for external sampled data. Finite-element reaction loads use
    :meth:`Stokes.boundary_normal_traction_integral` instead, so their projection
    follows the actual boundary facets without a global gather or reconstructed
    angular ordering.

    Parameters
    ----------
    coords : array-like, shape (n, 2)
        Cartesian circular-boundary coordinates.
    values : array-like, shape (n,)
        Scalar values at ``coords``.
    wavenumber : int
        Positive azimuthal Fourier wavenumber.

    Returns
    -------
    coefficient, mean : tuple of float
        Cosine-mode coefficient and degree-zero mean.
    """

    mode = _validate_cylindrical_mode(wavenumber)
    coords = np.asarray(coords, dtype=float)
    values = np.asarray(values, dtype=float).reshape(-1)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape (n, 2).")
    if coords.shape[0] != values.size:
        raise ValueError("coords and values must contain the same number of samples.")
    if coords.shape[0] < 3:
        raise ValueError("At least three circular-boundary samples are required.")
    if not np.all(np.isfinite(coords)) or not np.all(np.isfinite(values)):
        raise ValueError("Boundary coordinates and values must be finite.")

    theta = np.mod(np.arctan2(coords[:, 1], coords[:, 0]), 2.0 * np.pi)
    order = np.argsort(theta)
    theta = theta[order]
    values = values[order]
    theta = np.append(theta, theta[0] + 2.0 * np.pi)
    values = np.append(values, values[0])

    coefficient = (
        _trapezoidal_integral(
            values * np.cos(mode * theta),
            theta,
        )
        / np.pi
    )
    mean = _trapezoidal_integral(values, theta) / (2.0 * np.pi)
    return float(coefficient), float(mean)


def _rotated_cylindrical_boundary_response(
    *,
    stokes,
    boundary: str,
    wavenumber: int,
    buoyancy_scale: float,
) -> tuple[float, float, float]:
    """Project one boundary's assembled reaction and topography coefficients."""

    mode = _validate_cylindrical_mode(wavenumber)
    buoyancy_scale = _finite_float(buoyancy_scale, "buoyancy_scale")
    if buoyancy_scale == 0.0:
        raise ValueError("Boundary buoyancy scales must be nonzero.")
    if getattr(stokes.mesh, "dim", None) != 2:
        raise ValueError("The cylindrical adapter requires a two-dimensional mesh.")

    import sympy
    from underworld3.maths import BdIntegral

    theta = stokes.mesh.CoordinateSystem.xR[1]
    harmonic = sympy.cos(mode * theta)
    reaction_integral = float(
        stokes.boundary_normal_traction_integral(
            boundary,
            harmonic,
            remove_mean=True,
        )
    )
    reaction_total = float(
        stokes.boundary_normal_traction_integral(
            boundary,
            1.0,
            remove_mean=False,
        )
    )
    boundary_measure = float(
        BdIntegral(stokes.mesh, fn=1.0, boundary=boundary).evaluate()
    )
    harmonic_norm = float(
        BdIntegral(stokes.mesh, fn=harmonic**2, boundary=boundary).evaluate()
    )
    if not np.all(
        np.isfinite(
            (reaction_integral, reaction_total, boundary_measure, harmonic_norm)
        )
    ):
        raise RuntimeError("Cylindrical reaction projection produced non-finite data.")
    if boundary_measure <= 0.0 or harmonic_norm <= 0.0:
        raise RuntimeError(
            "Cylindrical reaction projection requires positive boundary integrals."
        )

    reaction_coefficient = reaction_integral / harmonic_norm
    reaction_mean = reaction_total / boundary_measure
    return (
        float(reaction_coefficient),
        float(reaction_mean),
        float(-reaction_coefficient / buoyancy_scale),
    )


def cylindrical_annulus_response_from_rotated_stokes(
    *,
    stokes,
    radius_inner: float,
    radius_outer: float,
    wavenumber: int,
    outer_density_contrast: float,
    inner_density_contrast: float,
    outer_reference_gravity: float,
    inner_reference_gravity: float,
    internal_load_radius: float | None = None,
    internal_surface_density_coefficient: float = 0.0,
    outer_boundary: str = "Upper",
    inner_boundary: str = "Lower",
    outer_buoyancy_scale: float = 1.0,
    inner_buoyancy_scale: float = -1.0,
    gravitational_constant: float = 1.0,
    include_self_gravity: bool = False,
    outer_feedback_factor: float | None = None,
    inner_feedback_factor: float | None = None,
) -> CylindricalAnnulusResponse:
    r"""Compute a cylindrical response from rotated-free-slip wall reactions.

    :meth:`Stokes.boundary_normal_traction_integral` contracts the assembled
    wall reaction directly with a boundary test function. This adapter uses

    .. math::

       h=-reaction_{nn}/signed\_buoyancy\_scale

    and projects each boundary onto the unnormalised real basis
    :math:`\cos(n\theta)`. The numerator and harmonic norm use the same finite-
    element boundary measure. Owned reaction degrees of freedom are reduced on
    the mesh communicator; no pointwise traction recovery, angular sorting, or
    rank-zero boundary gather is performed.

    Parameters
    ----------
    stokes : underworld3.systems.Stokes
        Completed two-dimensional rotated-free-slip Stokes solve.
    radius_inner, radius_outer : float
        Positive annulus radii ordered from inner to outer.
    wavenumber : int
        Positive azimuthal Fourier wavenumber.
    outer_density_contrast, inner_density_contrast : float
        Signed density contrasts at the two boundaries.
    outer_reference_gravity, inner_reference_gravity : float
        Positive gravity magnitudes used to convert potential to geoid.
    internal_load_radius : float, optional
        Radius of an internal sheet, strictly inside the annulus.
    internal_surface_density_coefficient : float, default=0
        Density coefficient of the optional internal sheet.
    outer_boundary, inner_boundary : str
        Mesh boundary labels used for reaction recovery.
    outer_buoyancy_scale, inner_buoyancy_scale : float
        Signed scales converting wall reaction to dynamic topography.
    gravitational_constant : float, default=1
        Positive gravitational constant in the selected unit system.
    include_self_gravity : bool, default=False
        Return the self-gravity-corrected response when true.
    outer_feedback_factor, inner_feedback_factor : float, optional
        Explicit self-gravity feedback factors.

    Returns
    -------
    CylindricalAnnulusResponse
        Recovered reactions, topography, gravity response, and optional
        self-gravity correction, identical on every MPI rank.
    """

    radius_inner, radius_outer, mode = _validate_cylindrical_annulus(
        radius_inner,
        radius_outer,
        wavenumber,
    )
    if not isinstance(include_self_gravity, bool):
        raise TypeError("include_self_gravity must be True or False.")

    outer_reaction, outer_mean, outer_topography = (
        _rotated_cylindrical_boundary_response(
            stokes=stokes,
            boundary=outer_boundary,
            wavenumber=mode,
            buoyancy_scale=outer_buoyancy_scale,
        )
    )
    inner_reaction, inner_mean, inner_topography = (
        _rotated_cylindrical_boundary_response(
            stokes=stokes,
            boundary=inner_boundary,
            wavenumber=mode,
            buoyancy_scale=inner_buoyancy_scale,
        )
    )
    gravity = cylindrical_annulus_geoid_response(
        radius_inner=radius_inner,
        radius_outer=radius_outer,
        wavenumber=mode,
        outer_topography_coefficient=outer_topography,
        inner_topography_coefficient=inner_topography,
        outer_density_contrast=outer_density_contrast,
        inner_density_contrast=inner_density_contrast,
        outer_reference_gravity=outer_reference_gravity,
        inner_reference_gravity=inner_reference_gravity,
        internal_load_radius=internal_load_radius,
        internal_surface_density_coefficient=internal_surface_density_coefficient,
        gravitational_constant=gravitational_constant,
    )
    self_gravity = None
    if include_self_gravity:
        self_gravity = cylindrical_annulus_self_gravity_response(
            radius_inner=radius_inner,
            radius_outer=radius_outer,
            wavenumber=mode,
            outer_topography_coefficient=outer_topography,
            inner_topography_coefficient=inner_topography,
            outer_density_contrast=outer_density_contrast,
            inner_density_contrast=inner_density_contrast,
            outer_reference_gravity=outer_reference_gravity,
            inner_reference_gravity=inner_reference_gravity,
            internal_load_radius=internal_load_radius,
            internal_surface_density_coefficient=internal_surface_density_coefficient,
            gravitational_constant=gravitational_constant,
            outer_feedback_factor=outer_feedback_factor,
            inner_feedback_factor=inner_feedback_factor,
        )

    return CylindricalAnnulusResponse(
        outer_reaction=outer_reaction,
        inner_reaction=inner_reaction,
        outer_reaction_mean=outer_mean,
        inner_reaction_mean=inner_mean,
        outer_topography=outer_topography,
        inner_topography=inner_topography,
        outer_potential=gravity.outer_potential,
        inner_potential=gravity.inner_potential,
        outer_geoid=gravity.outer_geoid,
        inner_geoid=gravity.inner_geoid,
        self_gravity=self_gravity,
    )
