r"""Zhong et al. (2008) spherical-shell geoid response functions."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

from mpi4py import MPI
import numpy as np


@dataclass(frozen=True)
class Zhong2008GeoidResponse:
    """No-self-gravity surface and CMB geoid coefficients."""

    surface_geoid: float
    cmb_geoid: float


@dataclass(frozen=True)
class Zhong2008SelfGravityResponse:
    """Self-gravity-corrected topography and geoid coefficients."""

    surface_topography: float
    cmb_topography: float
    surface_geoid: float
    cmb_geoid: float
    q_surface: float
    q_cmb: float


@dataclass(frozen=True)
class Zhong2008DynamicResponse:
    """Rotated-Stokes topography and corresponding geoid coefficients."""

    surface_topography: float
    cmb_topography: float
    surface_geoid: float
    cmb_geoid: float
    self_gravity: Zhong2008SelfGravityResponse | None = None


def _validate_geometry(
    radius_inner: float,
    radius_outer: float,
    radius_internal: float,
    harmonic_degree: int,
) -> tuple[float, float, float, int]:
    try:
        ri = float(radius_inner)
        ro = float(radius_outer)
        rint = float(radius_internal)
    except (TypeError, ValueError) as error:
        raise TypeError("The shell radii must be real numbers.") from error
    if not np.all(np.isfinite((ri, rint, ro))) or not 0.0 < ri < rint < ro:
        raise ValueError(
            "Expected finite radii ordered as "
            "0 < radius_inner < radius_internal < radius_outer."
        )
    if isinstance(harmonic_degree, bool) or not isinstance(harmonic_degree, Integral):
        raise TypeError("harmonic_degree must be an integer.")
    degree = int(harmonic_degree)
    if degree < 1:
        raise ValueError("harmonic_degree must be at least one.")
    return ri, ro, rint, degree


def _zhong2008_geoid_operator(
    radius_inner: float,
    radius_outer: float,
    radius_internal: float,
    harmonic_degree: int,
    load_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the Zhong Appendix A topography operator and load vector."""

    ri, ro, rint, degree = _validate_geometry(
        radius_inner,
        radius_outer,
        radius_internal,
        harmonic_degree,
    )
    load_scale = float(load_scale)
    if not np.isfinite(load_scale):
        raise ValueError("load_scale must be finite.")

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
    load = load_scale * np.array(
        [
            -rint * (rint / ro) ** (degree + 1) / denominator,
            -rint * (ri / rint) ** degree / denominator,
        ],
        dtype=float,
    )
    return operator, load


def zhong2008_geoid_response(
    *,
    radius_inner: float,
    radius_outer: float,
    radius_internal: float,
    harmonic_degree: int,
    surface_topography_coefficient: float,
    cmb_topography_coefficient: float,
    load_scale: float = 1.0,
) -> Zhong2008GeoidResponse:
    r"""Return the nondimensional Zhong no-self-gravity geoid coefficients.

    The topography coefficients and ``load_scale`` must use the same
    unnormalised :math:`P_l^0` forcing convention as the Stokes solve.
    """

    operator, load = _zhong2008_geoid_operator(
        radius_inner,
        radius_outer,
        radius_internal,
        harmonic_degree,
        load_scale,
    )
    topography = np.array(
        [surface_topography_coefficient, cmb_topography_coefficient],
        dtype=float,
    )
    if not np.all(np.isfinite(topography)):
        raise ValueError("The topography coefficients must be finite.")
    geoid = operator @ topography + load
    return Zhong2008GeoidResponse(float(geoid[0]), float(geoid[1]))


def zhong2008_self_gravity_response(
    *,
    radius_inner: float,
    radius_outer: float,
    radius_internal: float,
    harmonic_degree: int,
    surface_topography_coefficient: float,
    cmb_topography_coefficient: float,
    load_scale: float = 1.0,
    surface_density_contrast: float = 3300.0,
    cmb_density_contrast: float = 5400.0,
    planet_radius: float = 6370000.0,
    gravity: float = 9.8,
    gravitational_constant: float = 6.67e-11,
) -> Zhong2008SelfGravityResponse:
    r"""Return the Zhong self-gravity-corrected response coefficients.

    Density contrasts are in kg/m³, ``planet_radius`` in metres, ``gravity`` in
    m/s², and ``gravitational_constant`` in SI units. Shell radii and response
    coefficients remain nondimensional.
    """

    operator, load = _zhong2008_geoid_operator(
        radius_inner,
        radius_outer,
        radius_internal,
        harmonic_degree,
        load_scale,
    )
    topography = np.array(
        [surface_topography_coefficient, cmb_topography_coefficient],
        dtype=float,
    )
    physical_values = np.array(
        [
            surface_density_contrast,
            cmb_density_contrast,
            planet_radius,
            gravity,
            gravitational_constant,
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(topography)):
        raise ValueError("The topography coefficients must be finite.")
    if not np.all(np.isfinite(physical_values)) or np.any(physical_values <= 0.0):
        raise ValueError("Density contrasts and physical constants must be positive.")

    gravity_scale = 4.0 * np.pi * gravitational_constant * planet_radius / gravity
    q = gravity_scale * np.array(
        [surface_density_contrast, cmb_density_contrast], dtype=float
    )
    q_matrix = np.diag(q)
    corrected_topography = np.linalg.solve(
        np.eye(2) - q_matrix @ operator,
        topography + q_matrix @ load,
    )
    corrected_geoid = operator @ corrected_topography + load

    return Zhong2008SelfGravityResponse(
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


def zhong2008_response_from_rotated_stokes(
    *,
    stokes,
    radius_inner: float,
    radius_outer: float,
    radius_internal: float,
    harmonic_degree: int,
    load_scale: float = 1.0,
    surface_boundary: str = "Upper",
    cmb_boundary: str = "Lower",
    surface_buoyancy_scale: float = 1.0,
    cmb_buoyancy_scale: float = 1.0,
    include_self_gravity: bool = False,
    surface_density_contrast: float = 3300.0,
    cmb_density_contrast: float = 5400.0,
    planet_radius: float = 6370000.0,
    gravity: float = 9.8,
    gravitational_constant: float = 6.67e-11,
) -> Zhong2008DynamicResponse:
    r"""Compute Zhong response coefficients from a rotated-free-slip Stokes solve.

    Normal traction recovery is delegated to the existing
    :meth:`Stokes.boundary_normal_traction` implementation. This adapter only
    projects the two boundary responses onto the unnormalised
    :math:`P_l^0` harmonic and applies the Zhong Appendix A operator.
    """

    ri, ro, _, degree = _validate_geometry(
        radius_inner,
        radius_outer,
        radius_internal,
        harmonic_degree,
    )
    if not isinstance(include_self_gravity, bool):
        raise TypeError("include_self_gravity must be True or False.")

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
    geoid = zhong2008_geoid_response(
        radius_inner=ri,
        radius_outer=ro,
        radius_internal=radius_internal,
        harmonic_degree=degree,
        surface_topography_coefficient=surface_topography,
        cmb_topography_coefficient=cmb_topography,
        load_scale=load_scale,
    )
    self_gravity = None
    if include_self_gravity:
        self_gravity = zhong2008_self_gravity_response(
            radius_inner=ri,
            radius_outer=ro,
            radius_internal=radius_internal,
            harmonic_degree=degree,
            surface_topography_coefficient=surface_topography,
            cmb_topography_coefficient=cmb_topography,
            load_scale=load_scale,
            surface_density_contrast=surface_density_contrast,
            cmb_density_contrast=cmb_density_contrast,
            planet_radius=planet_radius,
            gravity=gravity,
            gravitational_constant=gravitational_constant,
        )

    return Zhong2008DynamicResponse(
        surface_topography=surface_topography,
        cmb_topography=cmb_topography,
        surface_geoid=geoid.surface_geoid,
        cmb_geoid=geoid.cmb_geoid,
        self_gravity=self_gravity,
    )
