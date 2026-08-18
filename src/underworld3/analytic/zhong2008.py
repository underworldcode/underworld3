r"""Semi-analytic spherical-shell Stokes responses from Zhong et al. (2008).

The benchmark applies one radial delta-function load in a spherical harmonic
and asks for the surface and CMB response.  Following Hager and O'Connell
(1981), the poloidal Stokes equations are written as a four-component first
order system in :math:`v=\ln r` and propagated exactly through each
constant-viscosity layer with a matrix exponential.

This is a numeric reference oracle, not a finite-element solve and not a SymPy
field on a mesh.  It returns the scalar response coefficients published in
Tables 2--4 of Zhong et al. (2008).  The no-self-gravity topography is recovered
from radial stress first; geoid and self-gravity feedback are then delegated to
the generic functions in :mod:`underworld3.postprocessing.geoid`.

References
----------
Hager, B.H. & O'Connell, R.J. (1981). A simple global model of plate dynamics
and mantle convection. *Journal of Geophysical Research* 86, 4843--4878.
doi:10.1029/JB086iB06p04843

Zhong, S. et al. (2008). A benchmark study on mantle convection in a 3-D
spherical shell using CitcomS. *Geochemistry, Geophysics, Geosystems* 9,
Q10017. doi:10.1029/2008GC002048
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from scipy.linalg import expm

from underworld3.postprocessing.geoid import (
    SelfGravityResponse,
    spherical_shell_geoid_response,
    spherical_shell_self_gravity_response,
)


__all__ = ["Zhong2008", "Zhong2008Response"]


@dataclass(frozen=True)
class Zhong2008Response:
    r"""Boundary response coefficients for one harmonic delta load.

    ``surface_topography``, ``cmb_topography``, and the two corresponding geoid
    coefficients exclude self-gravity.  The values including self-gravity are
    grouped in ``self_gravity``.  This mirrors the separation in Zhong et al.
    (2008): self-gravity changes the topography/geoid interpretation but not the
    Stokes velocity.
    """

    surface_topography: float
    cmb_topography: float
    surface_geoid: float
    cmb_geoid: float
    surface_characteristic_velocity: float
    cmb_characteristic_velocity: float
    surface_velocity_divergence: float
    cmb_velocity_divergence: float
    self_gravity: SelfGravityResponse


class Zhong2008:
    r"""Propagator-matrix reference solution for Zhong et al. (2008).

    Parameters
    ----------
    harmonic_degree : int
        Spherical-harmonic degree :math:`l`.  The radial response is independent
        of harmonic order :math:`m`.
    radius_inner, radius_outer : float
        Nondimensional CMB and surface radii.
    internal_load_radius : float
        Radius of the radial delta-function load.
    internal_load_coefficient : float
        Coefficient of the outward load.  The Zhong benchmark uses one.
    viscosity_interfaces : sequence of float
        Strictly increasing radii at which viscosity changes.  Omit for an
        isoviscous shell.
    viscosities : sequence of float
        Positive dimensionless viscosities from the innermost to outermost
        layer.  Its length must be one greater than ``viscosity_interfaces``.
    surface_density_contrast, cmb_density_contrast : float
        Signed density contrasts used only for self-gravity postprocessing, in
        kg/m^3.  Defaults reproduce Zhong et al. (2008).
    planet_radius : float
        Dimensional planet radius in metres, used only for self-gravity.
    gravity : float
        Surface gravitational acceleration in m/s^2.
    gravitational_constant : float
        Gravitational constant in SI units.  The default is the value used in
        Zhong et al. (2008), not the current CODATA value.

    Notes
    -----
    The propagated state is

    .. math::

       \mathbf u=(y_1,y_2,r\sigma_{rr}/\eta_0,r\sigma_{r\perp}/\eta_0)^T,

    where :math:`y_1` is radial velocity and :math:`y_2` is the characteristic
    horizontal velocity.  Impermeable free slip is therefore ``u[0] = u[3] =
    0`` at both boundaries.  Across the load at :math:`r=r_0`, only the radial
    traction state jumps:

    .. math::

       \mathbf u(r_0^+)-\mathbf u(r_0^-)
       =(0,0,-r_0 A,0)^T.

    Examples
    --------
    The isoviscous, degree-two, mid-mantle case from Table 2:

    >>> result = Zhong2008(harmonic_degree=2, internal_load_radius=0.775).response()
    >>> round(result.self_gravity.surface_topography, 4)
    0.4998

    The layered case from Table 3 has a :math:`10^4` viscosity lid:

    >>> result = Zhong2008(
    ...     harmonic_degree=2,
    ...     internal_load_radius=0.775,
    ...     viscosity_interfaces=(0.971875,),
    ...     viscosities=(1.0, 1.0e4),
    ... ).response()
    >>> round(result.self_gravity.surface_geoid, 5)
    0.05789
    """

    reference = (
        "Zhong et al. (2008), Geochem. Geophys. Geosyst. 9, Q10017, "
        "doi:10.1029/2008GC002048; propagator formulation after Hager and "
        "O'Connell (1981), doi:10.1029/JB086iB06p04843."
    )

    def __init__(
        self,
        *,
        harmonic_degree: int = 2,
        radius_inner: float = 0.55,
        radius_outer: float = 1.0,
        internal_load_radius: float = 0.775,
        internal_load_coefficient: float = 1.0,
        viscosity_interfaces=(),
        viscosities=(1.0,),
        surface_density_contrast: float = 3300.0,
        cmb_density_contrast: float = 5400.0,
        planet_radius: float = 6_370_000.0,
        gravity: float = 9.8,
        gravitational_constant: float = 6.67e-11,
    ):
        if isinstance(harmonic_degree, bool) or not isinstance(
            harmonic_degree, Integral
        ):
            raise TypeError("harmonic_degree must be an integer.")
        if harmonic_degree < 1:
            raise ValueError("harmonic_degree must be positive.")

        ri = float(radius_inner)
        ro = float(radius_outer)
        rint = float(internal_load_radius)
        if not np.all(np.isfinite((ri, ro, rint))) or not 0.0 < ri < rint < ro:
            raise ValueError(
                "Expected finite radii ordered as 0 < radius_inner < "
                "internal_load_radius < radius_outer."
            )

        interfaces = np.asarray(tuple(viscosity_interfaces), dtype=float)
        layer_viscosities = np.asarray(tuple(viscosities), dtype=float)
        if interfaces.ndim != 1 or layer_viscosities.ndim != 1:
            raise ValueError(
                "Viscosity interfaces and viscosities must be one-dimensional."
            )
        if len(layer_viscosities) != len(interfaces) + 1:
            raise ValueError(
                "viscosities must contain one value more than viscosity_interfaces."
            )
        if (
            not np.all(np.isfinite(interfaces))
            or np.any(interfaces <= ri)
            or np.any(interfaces >= ro)
            or np.any(np.diff(interfaces) <= 0.0)
        ):
            raise ValueError(
                "viscosity_interfaces must be finite, strictly increasing, and "
                "strictly inside the shell."
            )
        if (
            not len(layer_viscosities)
            or not np.all(np.isfinite(layer_viscosities))
            or np.any(layer_viscosities <= 0.0)
        ):
            raise ValueError("viscosities must be finite and positive.")

        load = float(internal_load_coefficient)
        physical = np.asarray(
            (
                surface_density_contrast,
                cmb_density_contrast,
                planet_radius,
                gravity,
                gravitational_constant,
            ),
            dtype=float,
        )
        if not np.isfinite(load):
            raise ValueError("internal_load_coefficient must be finite.")
        if not np.all(np.isfinite(physical[:2])):
            raise ValueError("Density contrasts must be finite.")
        if not np.all(np.isfinite(physical[2:])) or np.any(physical[2:] <= 0.0):
            raise ValueError("Physical constants must be finite and positive.")

        self.harmonic_degree = int(harmonic_degree)
        self.radius_inner = ri
        self.radius_outer = ro
        self.internal_load_radius = rint
        self.internal_load_coefficient = load
        self.viscosity_interfaces = tuple(float(value) for value in interfaces)
        self.viscosities = tuple(float(value) for value in layer_viscosities)
        self.surface_density_contrast = float(surface_density_contrast)
        self.cmb_density_contrast = float(cmb_density_contrast)
        self.planet_radius = float(planet_radius)
        self.gravity = float(gravity)
        self.gravitational_constant = float(gravitational_constant)

    def _system_matrix(self, viscosity: float) -> np.ndarray:
        r"""Hager--O'Connell four-state poloidal matrix for one layer."""

        degree = self.harmonic_degree
        angular_eigenvalue = degree * (degree + 1)
        eta = float(viscosity)
        return np.array(
            [
                [-2.0, angular_eigenvalue, 0.0, 0.0],
                [-1.0, 1.0, 0.0, 1.0 / eta],
                [12.0 * eta, -6.0 * angular_eigenvalue * eta, 1.0, angular_eigenvalue],
                [-6.0 * eta, 2.0 * (2.0 * angular_eigenvalue - 1.0) * eta, -1.0, -2.0],
            ],
            dtype=float,
        )

    def _propagate(
        self, state: np.ndarray, radius_a: float, radius_b: float
    ) -> np.ndarray:
        """Propagate one state upward, applying the load at its exact radius."""

        breakpoints = sorted(
            {
                self.radius_inner,
                self.radius_outer,
                self.internal_load_radius,
                *self.viscosity_interfaces,
                float(radius_a),
                float(radius_b),
            }
        )
        points = [point for point in breakpoints if radius_a <= point <= radius_b]
        propagated = np.asarray(state, dtype=float).copy()

        for lower, upper in zip(points[:-1], points[1:]):
            midpoint = 0.5 * (lower + upper)
            layer = int(np.searchsorted(self.viscosity_interfaces, midpoint))
            matrix = self._system_matrix(self.viscosities[layer])
            propagated = expm(matrix * np.log(upper / lower)) @ propagated
            if upper == self.internal_load_radius:
                propagated[2] -= (
                    self.internal_load_radius * self.internal_load_coefficient
                )

        return propagated

    def _boundary_states(self) -> tuple[np.ndarray, np.ndarray]:
        """Solve the two free-slip boundary constraints for the boundary states."""

        basis = np.zeros((4, 2), dtype=float)
        basis[1, 0] = 1.0
        basis[2, 1] = 1.0
        offset = self._propagate(
            np.zeros(4, dtype=float), self.radius_inner, self.radius_outer
        )
        transferred_basis = np.column_stack(
            [
                self._propagate(basis[:, column], self.radius_inner, self.radius_outer)
                - offset
                for column in range(2)
            ]
        )
        unknowns = np.linalg.solve(transferred_basis[[0, 3], :], -offset[[0, 3]])
        cmb_state = basis @ unknowns
        surface_state = self._propagate(cmb_state, self.radius_inner, self.radius_outer)
        return cmb_state, surface_state

    def response(self) -> Zhong2008Response:
        r"""Compute no-self-gravity and self-gravity response coefficients."""

        cmb_state, surface_state = self._boundary_states()
        degree = self.harmonic_degree
        angular_eigenvalue = degree * (degree + 1)

        surface_topography = -surface_state[2] / self.radius_outer
        cmb_topography = cmb_state[2] / self.radius_inner
        surface_velocity = surface_state[1]
        cmb_velocity = cmb_state[1]
        surface_divergence = -angular_eigenvalue * surface_velocity / self.radius_outer
        cmb_divergence = -angular_eigenvalue * cmb_velocity / self.radius_inner

        geoid = spherical_shell_geoid_response(
            radius_inner=self.radius_inner,
            radius_outer=self.radius_outer,
            harmonic_degree=degree,
            surface_topography_coefficient=surface_topography,
            cmb_topography_coefficient=cmb_topography,
            internal_load_radius=self.internal_load_radius,
            internal_load_coefficient=self.internal_load_coefficient,
        )
        self_gravity = spherical_shell_self_gravity_response(
            radius_inner=self.radius_inner,
            radius_outer=self.radius_outer,
            harmonic_degree=degree,
            surface_topography_coefficient=surface_topography,
            cmb_topography_coefficient=cmb_topography,
            internal_load_radius=self.internal_load_radius,
            internal_load_coefficient=self.internal_load_coefficient,
            surface_density_contrast=self.surface_density_contrast,
            cmb_density_contrast=self.cmb_density_contrast,
            planet_radius=self.planet_radius,
            gravity=self.gravity,
            gravitational_constant=self.gravitational_constant,
        )

        return Zhong2008Response(
            surface_topography=float(surface_topography),
            cmb_topography=float(cmb_topography),
            surface_geoid=geoid.surface_geoid,
            cmb_geoid=geoid.cmb_geoid,
            surface_characteristic_velocity=float(surface_velocity),
            cmb_characteristic_velocity=float(cmb_velocity),
            surface_velocity_divergence=float(surface_divergence),
            cmb_velocity_divergence=float(cmb_divergence),
            self_gravity=self_gravity,
        )
