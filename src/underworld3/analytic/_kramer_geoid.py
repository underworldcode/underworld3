"""Kramer stress-to-geoid adapter; all gravity kernels retain their owners.

Stokes reference: Kramer, Davies and Wilson (2021), Analytical solutions for
mantle flow in cylindrical and spherical shells.
https://doi.org/10.5194/gmd-14-1899-2021
Gravity and interface feedback are UW3 extensions using the documented kernels
in cylindrical.gravity/spherical.gravity, not published Kramer geoid results.
"""

from dataclasses import asdict

import numpy as np
import sympy as sp


class _KramerGeoidMixin:
    case = "kramer"

    def boundary_stress_coefficients(self):
        r"""Total sigma_rr coefficients in this solution's angular basis.

        Boundary ordering is outer/inner. Spherical coefficients multiply the
        selected real Y_lm, including its normalization; cylindrical ones
        multiply cos(n*theta). No surface sampling or harmonic division occurs.

        Returns
        -------
        dict
            outer_radial_normal_stress and inner_radial_normal_stress, in the
            reference stress units and angular normalization. These are fluid sigma_rr,
            not sign-reversed wall reactions.
        """
        r = sp.Symbol("_response_r", positive=True)
        if self.geometry == "cylindrical":
            from .cylindrical.kramer import analytical_radial_normal_stress_sympy

            theta = sp.Symbol("_response_theta", real=True)
            expression = lambda branch: analytical_radial_normal_stress_sympy(
                branch, r, theta
            ).subs(theta, 0)
        else:
            from .spherical.kramer import analytical_radial_normal_stress_sympy

            expression = lambda branch: analytical_radial_normal_stress_sympy(
                branch, r, sp.S.One
            )
        return dict(
            outer_radial_normal_stress=float(
                expression(self._above).subs(r, self.r_outer)
            ),
            inner_radial_normal_stress=float(
                expression(self._below).subs(r, self.r_inner)
            ),
        )

    def geoid_response(
        self,
        *,
        outer_density_contrast,
        inner_density_contrast,
        outer_reference_gravity,
        inner_reference_gravity,
        gravitational_constant,
        source_density_coefficient,
        include_self_gravity=False,
    ):
        r"""Analytical free-slip topography/potential/geoid coefficients.

        Required inputs use one consistent unit system with the Stokes stress
        and radii. Signed contrasts are density at smaller radius minus density
        at larger radius. h_outer=-sigma_rr/(Delta_rho*g),
        h_inner=+sigma_rr/(Delta_rho*g); both are radial-outward displacements.
        Delta_rho must be nonzero; reference gravity magnitudes must be positive.

        ``source_density_coefficient`` is signed sheet density for ``delta`` or
        volume-density amplitude multiplying (r/r_outer)**k for ``smooth``.
        It is explicit because a Stokes forcing coefficient is not a density
        until the forcing gravity and dimensional scales have been specified.
        With unit inward forcing gravity and consistent scales it equals g.
        N=Phi/g_reference, laplacian(Phi)=-4*pi*G*rho. Optional self-gravity is
        linear two-interface feedback after the Stokes solve, not a new solve.

        Parameters
        ----------
        outer_density_contrast, inner_density_contrast : float
            Finite nonzero signed contrasts, smaller-radius-side minus larger-radius-side
            density, in mass/volume units consistent with the Stokes stress scaling.
        outer_reference_gravity, inner_reference_gravity : float
            Finite positive gravity magnitudes in the corresponding length/time^2 units.
        gravitational_constant : float
            Finite positive G in the same physical or nondimensional unit system.
        source_density_coefficient : float
            Explicit finite signed surface density for a delta sheet (mass/area), or
            volume-density amplitude at r_outer for a smooth source (mass/volume).
            The Stokes forcing amplitude is not automatically treated as this density.
        include_self_gravity : bool, default=False
            Add separate linear two-interface feedback quantities when True.

        Returns
        -------
        dict
            Outer/inner hydrodynamic topography, physical potential and geoid coefficients;
            feedback adds corresponding ``*_self_gravity`` keys and matrix diagnostics.
            The spherical result also includes its feedback condition number.

        Raises
        ------
        TypeError
            include_self_gravity is not boolean.
        ValueError
            Wall choice, contrasts, reference gravity or source inputs are invalid.
        numpy.linalg.LinAlgError
            The spherical interface-feedback system is singular or ill-conditioned.

        Notes
        -----
        Call only for a free-slip Kramer solution. No FE solve, field sampling or unit conversion is
        performed. All sources use this solution's angular normalization; cylindrical gravity
        assumes axial invariance.
        """
        if self.boundary != "free":
            raise ValueError(
                "Dynamic geoid response requires the free-slip Kramer case."
            )
        if not isinstance(include_self_gravity, bool):
            raise TypeError("include_self_gravity must be bool.")
        contrasts = np.array(
            [outer_density_contrast, inner_density_contrast], dtype=float
        )
        gravity = np.array(
            [outer_reference_gravity, inner_reference_gravity], dtype=float
        )
        if not np.all(np.isfinite(contrasts)) or np.any(contrasts == 0):
            raise ValueError("Density contrasts must be finite and nonzero.")
        if not np.all(np.isfinite(gravity)) or np.any(gravity <= 0):
            raise ValueError(
                "Reference gravity magnitudes must be finite and positive."
            )
        stress = self.boundary_stress_coefficients()
        ho = -stress["outer_radial_normal_stress"] / (contrasts[0] * gravity[0])
        hi = stress["inner_radial_normal_stress"] / (contrasts[1] * gravity[1])
        source_radius = self.r_int if self.density == "delta" else None
        sheet = source_density_coefficient if self.density == "delta" else 0.0
        volume = source_density_coefficient if self.density == "smooth" else 0.0
        if self.geometry == "spherical":
            from .spherical.gravity import spherical_physical_shell_response

            return spherical_physical_shell_response(
                radius_inner=self.r_inner,
                radius_outer=self.r_outer,
                harmonic_degree=self.ell,
                outer_topography=ho,
                inner_topography=hi,
                outer_density_contrast=contrasts[0],
                inner_density_contrast=contrasts[1],
                outer_gravity=gravity[0],
                inner_gravity=gravity[1],
                gravitational_constant=gravitational_constant,
                internal_load_radius=source_radius,
                internal_surface_density=sheet,
                volume_density=volume,
                density_power=self.k,
                include_self_gravity=include_self_gravity,
            )
        from .cylindrical.gravity import (
            cylindrical_annulus_geoid_response,
            cylindrical_annulus_self_gravity_response,
        )

        parameters = dict(
            radius_inner=self.r_inner,
            radius_outer=self.r_outer,
            wavenumber=self.n,
            outer_topography_coefficient=ho,
            inner_topography_coefficient=hi,
            outer_density_contrast=contrasts[0],
            inner_density_contrast=contrasts[1],
            outer_reference_gravity=gravity[0],
            inner_reference_gravity=gravity[1],
            internal_load_radius=source_radius,
            internal_surface_density_coefficient=sheet,
            volume_density_radial_exponent=self.k if self.density == "smooth" else None,
            volume_density_coefficient=volume,
            gravitational_constant=gravitational_constant,
        )
        result = dict(
            outer_topography=float(ho),
            inner_topography=float(hi),
            **asdict(cylindrical_annulus_geoid_response(**parameters)),
        )
        if include_self_gravity:
            corrected = asdict(cylindrical_annulus_self_gravity_response(**parameters))
            result.update(
                {
                    f"{name}_self_gravity": value
                    for name, value in corrected.items()
                    if name.startswith(("outer_", "inner_"))
                }
            )
            result["matrix_residual_norm"] = corrected["matrix_residual_norm"]
        return result
