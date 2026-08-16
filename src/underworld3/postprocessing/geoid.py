r"""Spherical-harmonic geoid and self-gravity response functions.

The pure coefficient functions in this module are independent of a particular
Stokes discretisation.  They combine surface, CMB, and optional internal-load
coefficients through the radial Green's function for one spherical-harmonic
degree.  A separate convenience adapter obtains the two topography coefficients
from a completed rotated-free-slip Stokes solve.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

from mpi4py import MPI
import numpy as np


__all__ = [
    "GeoidResponse",
    "SelfGravityResponse",
    "SphericalShellResponse",
    "spherical_shell_geoid_response",
    "spherical_shell_self_gravity_response",
    "spherical_shell_response_from_rotated_stokes",
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
        raise ValueError("Expected finite radii ordered as 0 < radius_inner < radius_outer.")
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
            raise TypeError("internal_load_radius must be a real number or None.") from error
        if not np.isfinite(rint) or not ri < rint < ro:
            raise ValueError("internal_load_radius must lie strictly between the shell radii.")
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
