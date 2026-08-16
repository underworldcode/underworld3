r"""Spherical-shell geoid and dynamic-response post-processing helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SphericalShellGeoidResponse:
    """No-self-gravity geoid response at the surface and CMB."""

    surface_geoid: float
    cmb_geoid: float

    def __iter__(self):
        yield self.surface_geoid
        yield self.cmb_geoid


@dataclass(frozen=True)
class SphericalShellSelfGravityResponse:
    """Self-gravity corrected topography and geoid response."""

    surface_topography: float
    cmb_topography: float
    surface_geoid: float
    cmb_geoid: float
    q_surface: float
    q_cmb: float


@dataclass(frozen=True)
class SphericalShellDynamicResponse:
    """Dynamic topography plus geoid response for a spherical shell."""

    surface_topography: float
    cmb_topography: float
    surface_geoid: float
    cmb_geoid: float
    topography_source: str
    self_gravity: SphericalShellSelfGravityResponse | None = None


def _spherical_shell_geoid_operator(
    radius_inner: float,
    radius_outer: float,
    radius_internal: float,
    harmonic_degree: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the Zhong Appendix A topography operator and load vector."""

    ri = float(radius_inner)
    ro = float(radius_outer)
    rint = float(radius_internal)
    degree = int(harmonic_degree)
    if not 0.0 < ri < rint < ro:
        raise ValueError("Expected 0 < radius_inner < radius_internal < radius_outer.")
    if degree < 0:
        raise ValueError("harmonic_degree must be non-negative.")

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
    load = np.array(
        [
            -rint * (rint / ro) ** (degree + 1) / denominator,
            -rint * (ri / rint) ** degree / denominator,
        ],
        dtype=float,
    )
    return operator, load


def spherical_shell_geoid_response(
    *,
    radius_inner: float,
    radius_outer: float,
    radius_internal: float,
    harmonic_degree: int,
    surface_topography: float,
    cmb_topography: float,
) -> SphericalShellGeoidResponse:
    """Return Zhong-style no-self-gravity surface and CMB geoid responses."""

    operator, load = _spherical_shell_geoid_operator(
        radius_inner,
        radius_outer,
        radius_internal,
        harmonic_degree,
    )
    geoid = (
        operator @ np.array([float(surface_topography), float(cmb_topography)]) + load
    )
    return SphericalShellGeoidResponse(float(geoid[0]), float(geoid[1]))


def spherical_shell_self_gravity_response(
    *,
    radius_inner: float,
    radius_outer: float,
    radius_internal: float,
    harmonic_degree: int,
    surface_topography: float,
    cmb_topography: float,
    density_mantle: float = 3300.0,
    density_cmb: float = 5400.0,
    planet_radius: float = 6370000.0,
    gravity: float = 9.8,
    gravitational_constant: float = 6.67e-11,
) -> SphericalShellSelfGravityResponse:
    """Return Zhong-style self-gravity corrected topography/geoid response."""

    operator, load = _spherical_shell_geoid_operator(
        radius_inner,
        radius_outer,
        radius_internal,
        harmonic_degree,
    )
    gravity_scale = float(
        4.0
        * np.pi
        * float(gravitational_constant)
        * float(planet_radius)
        / float(gravity)
    )
    q = gravity_scale * np.array(
        [float(density_mantle), float(density_cmb)], dtype=float
    )
    topography = np.array(
        [float(surface_topography), float(cmb_topography)], dtype=float
    )
    q_matrix = np.diag(q)
    corrected_topography = np.linalg.solve(
        np.eye(2) - q_matrix @ operator,
        topography + q_matrix @ load,
    )
    corrected_geoid = operator @ corrected_topography + load

    return SphericalShellSelfGravityResponse(
        surface_topography=float(corrected_topography[0]),
        cmb_topography=float(corrected_topography[1]),
        surface_geoid=float(corrected_geoid[0]),
        cmb_geoid=float(corrected_geoid[1]),
        q_surface=float(q[0]),
        q_cmb=float(q[1]),
    )


def spherical_shell_dynamic_response(
    *,
    stokes,
    velocity=None,
    radius_inner: float,
    radius_outer: float,
    radius_internal: float,
    harmonic_degree: int,
    include_self_gravity: bool = False,
    boundary_tolerance: float | None = None,
    surface_boundary: str = "Upper",
    cmb_boundary: str = "Lower",
    topography_source: str = "auto",
    constrained_reference=None,
    **self_gravity_kwargs,
) -> SphericalShellDynamicResponse:
    """Return topography and geoid response for a solved spherical-shell Stokes model.

    ``topography_source="auto"`` uses constrained multiplier topography when
    the solved Stokes system has boundary multipliers, rotated constraint
    reactions for rotated strong free slip, and CBF-lumped residual topography
    otherwise. Rotated and constrained recovery need only ``stokes``;
    ``velocity`` is optional and used only by the CBF fallback.
    """

    from .topography import spherical_shell_topography_coefficients

    topography = spherical_shell_topography_coefficients(
        stokes=stokes,
        velocity=velocity,
        radius_inner=radius_inner,
        radius_outer=radius_outer,
        harmonic_degree=harmonic_degree,
        source=topography_source,
        boundary_tolerance=boundary_tolerance,
        surface_boundary=surface_boundary,
        cmb_boundary=cmb_boundary,
        constrained_reference=constrained_reference,
    )
    surface_topography = topography.surface
    cmb_topography = topography.cmb

    geoid = spherical_shell_geoid_response(
        radius_inner=radius_inner,
        radius_outer=radius_outer,
        radius_internal=radius_internal,
        harmonic_degree=harmonic_degree,
        surface_topography=surface_topography,
        cmb_topography=cmb_topography,
    )

    self_gravity = None
    if include_self_gravity:
        self_gravity = spherical_shell_self_gravity_response(
            radius_inner=radius_inner,
            radius_outer=radius_outer,
            radius_internal=radius_internal,
            harmonic_degree=harmonic_degree,
            surface_topography=surface_topography,
            cmb_topography=cmb_topography,
            **self_gravity_kwargs,
        )

    return SphericalShellDynamicResponse(
        surface_topography=surface_topography,
        cmb_topography=cmb_topography,
        surface_geoid=geoid.surface_geoid,
        cmb_geoid=geoid.cmb_geoid,
        topography_source=topography.source,
        self_gravity=self_gravity,
    )
