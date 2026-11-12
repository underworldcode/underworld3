r"""Native Kramer spherical Stokes coefficients, fields and batch evaluators.

Radial formulas follow Kramer, Davies & Wilson (2021), GMD 14, 1899–1919.
The Cartesian solid-harmonic representation avoids coordinate pole singularities.

Spherical Kramer reference
--------------------------
Kramer, S. C., Davies, D. R. and Wilson, C. R. (2021), Analytical solutions
for mantle flow in cylindrical and spherical shells, Geoscientific Model
Development 14, 1899-1919. https://doi.org/10.5194/gmd-14-1899-2021
"""


__all__ = [
    "coefficients_sphere_delta_fs",
    "coefficients_sphere_delta_ns",
    "coefficients_sphere_smooth_fs",
    "coefficients_sphere_smooth_ns",
    "build_delta_solution",
    "build_smooth_solution",
    "analytical_velocity_cartesian_sympy",
    "analytical_pressure_sympy",
    "analytical_Pl_sympy",
    "analytical_radial_normal_stress_sympy",
    "KramerSphericalStokes",
]

from types import SimpleNamespace

import sympy as sp


def coefficients_sphere_delta_fs(Rp, Rm, rp, l, g, nu, sign):
    r"""Construct the spherical Kramer free-slip radial coefficients.

    Parameters
    ----------
    Rp : real
        Positive outer radius in the chosen length scaling; ``Rm < Rp``.
    Rm : real
        Positive inner radius in the same scaling as ``Rp``.
    rp : real
        Delta-sheet radius strictly between ``Rm`` and ``Rp``.
    l : int
        Spherical-harmonic degree, at least 2.
    g : real
        Signed normal-stress jump coefficient ``g*Y``; the physical sheet force is ``-g*Y*e_r``.
    nu : real
        Positive constant dynamic viscosity in the same scaling as the forcing.
    sign : {-1, 1}
        Select the inner (-1) or outer (+1) branch relative to the delta sheet.

    Returns
    -------
    tuple of real or sympy expressions
        Radial coefficients ``(A, B, C, D)`` in the published radial-power ordering.

    Notes
    -----
    These low-level helpers assume validated radii, modes and nonsingular powers. Use the public
    Stokes class for input validation. No mesh, PDE solve or unit conversion is performed; the
    angular coefficient normalization is supplied by the caller.
    """
    alpha_pm, alpha_mp = [Rp / rp, Rm / rp][:: int(sign)]
    pm = sign

    A = -0.5 * (alpha_mp ** (2 * l - 1) - 1) * g * pm * rp ** (-l + 2) / (
        (alpha_mp ** (2 * l - 1) - alpha_pm ** (2 * l - 1)) * (2 * l + 1) * (2 * l - 1) * nu
    )
    B = -0.5 * (alpha_mp ** (-2 * l - 3) - 1) * g * pm * rp ** (l + 3) / (
        (alpha_mp ** (-2 * l - 3) - alpha_pm ** (-2 * l - 3)) * (2 * l + 3) * (2 * l + 1) * nu
    )
    C = 0.5 * (alpha_mp ** (2 * l + 3) - 1) * g * pm / (
        (alpha_mp ** (2 * l + 3) - alpha_pm ** (2 * l + 3)) * (2 * l + 3) * (2 * l + 1) * nu * rp**l
    )
    D = 0.5 * (alpha_mp ** (-2 * l + 1) - 1) * g * pm * rp ** (l + 1) / (
        (alpha_mp ** (-2 * l + 1) - alpha_pm ** (-2 * l + 1)) * (2 * l + 1) * (2 * l - 1) * nu
    )
    return A, B, C, D


def coefficients_sphere_delta_ns(Rp, Rm, rp, l, g, nu, sign):
    r"""Construct the spherical Kramer zero-slip radial coefficients.

    Parameters
    ----------
    Rp : real
        Positive outer radius in the chosen length scaling; ``Rm < Rp``.
    Rm : real
        Positive inner radius in the same scaling as ``Rp``.
    rp : real
        Delta-sheet radius strictly between ``Rm`` and ``Rp``.
    l : int
        Spherical-harmonic degree, at least 2.
    g : real
        Signed normal-stress jump coefficient ``g*Y``; the physical sheet force is ``-g*Y*e_r``.
    nu : real
        Positive constant dynamic viscosity in the same scaling as the forcing.
    sign : {-1, 1}
        Select the inner (-1) or outer (+1) branch relative to the delta sheet.

    Returns
    -------
    tuple of real or sympy expressions
        Radial coefficients ``(A, B, C, D)`` in the published radial-power ordering.

    Notes
    -----
    These low-level helpers assume validated radii, modes and nonsingular powers. Use the public
    Stokes class for input validation. No mesh, PDE solve or unit conversion is performed; the
    angular coefficient normalization is supplied by the caller.
    """
    alpha_p, alpha_m = [Rp / rp, Rm / rp]
    pm, mp = sign, -sign

    denom = (
        (2 * l + 1) ** 2 * (alpha_m**2 / alpha_p**2 + alpha_p**2 / alpha_m**2)
        - 2 * (2 * l + 3) * (2 * l - 1)
        - 4 * (alpha_m / alpha_p) ** (2 * l + 1)
        - 4 * (alpha_m / alpha_p) ** (-2 * l - 1)
    ) * nu

    A = -0.5 * (
        alpha_m**2
        - alpha_p**2
        - (2 * l + 1) * mp * (alpha_m / alpha_p) ** (2 * mp) / (2 * l - 1)
        - (2 * l + 3) * pm / (2 * l + 1)
        + 2 * (alpha_m ** (-2 * l - 1) - alpha_p ** (-2 * l - 1)) / (2 * l + 1)
        + 2 * (alpha_m**2 * alpha_p ** (-2 * l - 1) - alpha_m ** (-2 * l - 1) * alpha_p**2) / (2 * l - 1)
        - 4 * pm * (alpha_m / alpha_p) ** ((2 * l + 1) * pm) / ((2 * l + 1) * (2 * l - 1))
    ) * g * rp ** (-l + 2) / denom
    B = -0.5 * (
        alpha_m**2
        - alpha_p**2
        - (2 * l + 1) * mp * (alpha_m / alpha_p) ** (2 * mp) / (2 * l + 3)
        - (2 * l - 1) * pm / (2 * l + 1)
        - 2 * (alpha_m**2 * alpha_p ** (2 * l + 1) - alpha_m ** (2 * l + 1) * alpha_p**2) / (2 * l + 3)
        - 2 * (alpha_m ** (2 * l + 1) - alpha_p ** (2 * l + 1)) / (2 * l + 1)
        - 4 * pm * (alpha_m / alpha_p) ** ((2 * l + 1) * mp) / ((2 * l + 3) * (2 * l + 1))
    ) * g * rp ** (l + 3) / denom
    C = 0.5 * (
        (2 * l + 1) * pm * (alpha_m / alpha_p) ** (2 * pm) / (2 * l + 3)
        + (2 * l - 1) * mp / (2 * l + 1)
        - 2 * (alpha_m ** (-2 * l - 1) - alpha_p ** (-2 * l - 1)) / (2 * l + 1)
        + 4 * mp * (alpha_m / alpha_p) ** ((2 * l + 1) * pm) / ((2 * l + 3) * (2 * l + 1))
        + 2 * (alpha_m ** (-2 * l - 1) / alpha_p**2 - alpha_p ** (-2 * l - 1) / alpha_m**2) / (2 * l + 3)
        + 1 / alpha_m**2
        - 1 / alpha_p**2
    ) * g / (denom * rp**l)
    D = 0.5 * (
        (2 * l + 1) * pm * (alpha_m / alpha_p) ** (2 * pm) / (2 * l - 1)
        + (2 * l + 3) * mp / (2 * l + 1)
        + 2 * (alpha_m ** (2 * l + 1) - alpha_p ** (2 * l + 1)) / (2 * l + 1)
        + 4 * mp * (alpha_m / alpha_p) ** ((2 * l + 1) * mp) / ((2 * l + 1) * (2 * l - 1))
        - 2 * (alpha_m ** (2 * l + 1) / alpha_p**2 - alpha_p ** (2 * l + 1) / alpha_m**2) / (2 * l - 1)
        + 1 / alpha_m**2
        - 1 / alpha_p**2
    ) * g * rp ** (l + 1) / denom
    return A, B, C, D


def coefficients_sphere_smooth_fs(Rp, Rm, k, l, g, nu):
    r"""Construct the spherical Kramer free-slip radial coefficients.

    Parameters
    ----------
    Rp : real
        Positive outer radius in the chosen length scaling; ``Rm < Rp``.
    Rm : real
        Positive inner radius in the same scaling as ``Rp``.
    k : int
        Radial exponent of smooth density ``(r/Rp)**k``. Resonant powers require a
        logarithmic family and are excluded.
    l : int
        Spherical-harmonic degree, at least 2.
    g : real
        Signed coefficient of the smooth radial body force ``-g*(r/Rp)**k*Y*e_r``.
    nu : real
        Positive constant dynamic viscosity in the same scaling as the forcing.

    Returns
    -------
    tuple of real or sympy expressions
        Radial coefficients ``(A, B, C, D, E)`` in the published radial-power ordering.

    Notes
    -----
    These low-level helpers assume validated radii, modes and nonsingular powers. Use the public
    Stokes class for input validation. No mesh, PDE solve or unit conversion is performed; the
    angular coefficient normalization is supplied by the caller.
    """
    alpha = Rm / Rp
    A = 0.5 * Rp ** (-l + 3) * (alpha ** (k + 3) - alpha ** (-l + 1)) * g / (
        (alpha**l - alpha ** (-l + 1)) * (k + l + 2) * (k - l + 3) * (2 * l + 1) * nu
    )
    B = 0.5 * Rp ** (l + 4) * (alpha ** (k + 4) - alpha ** (l + 3)) * g / (
        (alpha ** (l + 3) - 1 / alpha**l) * (k + l + 4) * (k - l + 1) * (2 * l + 1) * nu
    )
    C = -0.5 * Rp ** (-l + 1) * (alpha ** (k + 4) - 1 / alpha**l) * g / (
        (alpha ** (l + 3) - 1 / alpha**l) * (k + l + 4) * (k - l + 1) * (2 * l + 1) * nu
    )
    D = -0.5 * Rp ** (l + 2) * (alpha ** (k + 3) - alpha**l) * g / (
        (alpha**l - alpha ** (-l + 1)) * (k + l + 2) * (k - l + 3) * (2 * l + 1) * nu
    )
    E = g / (Rp**k * (k + l + 4) * (k + l + 2) * (k - l + 3) * (k - l + 1) * nu)
    return A, B, C, D, E


def coefficients_sphere_smooth_ns(Rp, Rm, k, l, g, nu):
    r"""Construct the spherical Kramer zero-slip radial coefficients.

    Parameters
    ----------
    Rp : real
        Positive outer radius in the chosen length scaling; ``Rm < Rp``.
    Rm : real
        Positive inner radius in the same scaling as ``Rp``.
    k : int
        Radial exponent of smooth density ``(r/Rp)**k``. Resonant powers require a
        logarithmic family and are excluded.
    l : int
        Spherical-harmonic degree, at least 2.
    g : real
        Signed coefficient of the smooth radial body force ``-g*(r/Rp)**k*Y*e_r``.
    nu : real
        Positive constant dynamic viscosity in the same scaling as the forcing.

    Returns
    -------
    tuple of real or sympy expressions
        Radial coefficients ``(A, B, C, D, E)`` in the published radial-power ordering.

    Notes
    -----
    These low-level helpers assume validated radii, modes and nonsingular powers. Use the public
    Stokes class for input validation. No mesh, PDE solve or unit conversion is performed; the
    angular coefficient normalization is supplied by the caller.
    """
    alpha = Rm / Rp
    gamma = (
        (alpha ** (l + 1) + alpha ** (l - 3)) * (2 * l + 1) ** 2
        - 2 * alpha ** (l - 1) * (2 * l + 3) * (2 * l - 1)
        - 4 * alpha ** (3 * l)
        - 4 * alpha ** (-l - 2)
    ) * (k + l + 4) * (k + l + 2) * (k - l + 3) * (k - l + 1)

    A = (
        (alpha ** (k + 2) + alpha ** (l - 1)) * (k + l + 2) * (2 * l + 3)
        - (alpha**k + alpha ** (l + 1)) * (k + l + 4) * (2 * l + 1)
        - 2 * (alpha ** (k + 2 * l + 3) + alpha ** (-l - 2)) * (k - l + 1)
    ) * Rp ** (-l + 3) * g / (gamma * nu)
    B = (
        (alpha ** (k + 2 * l + 1) + alpha ** (l + 1)) * (k - l + 3) * (2 * l + 1)
        - (alpha ** (k + 2 * l + 3) + alpha ** (l - 1)) * (k - l + 1) * (2 * l - 1)
        - 2 * (alpha ** (k + 2) + alpha ** (3 * l)) * (k + l + 2)
    ) * Rp ** (l + 4) * g / (gamma * nu)
    C = -(
        (alpha ** (k + 2) + alpha ** (l - 3)) * (k + l + 2) * (2 * l + 1)
        - (alpha**k + alpha ** (l - 1)) * (k + l + 4) * (2 * l - 1)
        - 2 * (alpha ** (k + 2 * l + 1) + alpha ** (-l - 2)) * (k - l + 3)
    ) * Rp ** (-l + 1) * g / (gamma * nu)
    D = -(
        (alpha ** (k + 2 * l + 1) + alpha ** (l - 1)) * (k - l + 3) * (2 * l + 3)
        - (alpha ** (k + 2 * l + 3) + alpha ** (l - 3)) * (k - l + 1) * (2 * l + 1)
        - 2 * (alpha**k + alpha ** (3 * l)) * (k + l + 4)
    ) * Rp ** (l + 2) * g / (gamma * nu)
    E = g / (Rp**k * (k + l + 4) * (k + l + 2) * (k - l + 3) * (k - l + 1) * nu)
    return A, B, C, D, E


def build_delta_solution(Rp, Rm, rp, l, m, g, nu, sign, no_slip):
    r"""Construct the spherical Kramer selected-wall radial coefficients.

    Parameters
    ----------
    Rp : real
        Positive outer radius in the chosen length scaling; ``Rm < Rp``.
    Rm : real
        Positive inner radius in the same scaling as ``Rp``.
    rp : real
        Delta-sheet radius strictly between ``Rm`` and ``Rp``.
    l : int
        Spherical-harmonic degree, at least 2.
    m : int
        Spherical-harmonic order, ``0 <= m <= l``; retained in the branch metadata.
    g : real
        Signed Kramer forcing coefficient. Delta normal-stress jump is ``g*Y`` and its
        physical sheet force is ``-g*Y*e_r``; smooth body force is ``-g*(r/Rp)**k*Y*e_r``.
    nu : real
        Positive constant dynamic viscosity in the same scaling as the forcing.
    sign : {-1, 1}
        Select the inner (-1) or outer (+1) branch relative to the delta sheet.
    no_slip : bool
        True selects zero wall velocity; False selects impermeable free slip at both radii.

    Returns
    -------
    types.SimpleNamespace
        Branch coefficients ``ABCD`` (delta) or ``ABCDE`` (smooth), mode/viscosity/forcing
        metadata and pressure coefficients for the symbolic field builders.

    Notes
    -----
    These low-level helpers assume validated radii, modes and nonsingular powers. Use the public
    Stokes class for input validation. No mesh, PDE solve or unit conversion is performed; the
    angular coefficient normalization is supplied by the caller.
    """
    coeffs = coefficients_sphere_delta_ns(Rp, Rm, rp, l, g, nu, sign) if no_slip else coefficients_sphere_delta_fs(Rp, Rm, rp, l, g, nu, sign)
    _, _, C, D = coeffs
    return SimpleNamespace(
        l=l,
        m=m,
        g=g,
        nu=nu,
        ABCD=coeffs,
        G=-2 * nu * (l + 1) * (2 * l + 3) * C,
        H=-2 * nu * l * (2 * l - 1) * D,
    )


def build_smooth_solution(Rp, Rm, k, l, m, g, nu, no_slip):
    r"""Construct the spherical Kramer selected-wall radial coefficients.

    Parameters
    ----------
    Rp : real
        Positive outer radius in the chosen length scaling; ``Rm < Rp``.
    Rm : real
        Positive inner radius in the same scaling as ``Rp``.
    k : int
        Radial exponent of smooth density ``(r/Rp)**k``. Resonant powers require a
        logarithmic family and are excluded.
    l : int
        Spherical-harmonic degree, at least 2.
    m : int
        Spherical-harmonic order, ``0 <= m <= l``; retained in the branch metadata.
    g : real
        Signed Kramer forcing coefficient. Delta normal-stress jump is ``g*Y`` and its
        physical sheet force is ``-g*Y*e_r``; smooth body force is ``-g*(r/Rp)**k*Y*e_r``.
    nu : real
        Positive constant dynamic viscosity in the same scaling as the forcing.
    no_slip : bool
        True selects zero wall velocity; False selects impermeable free slip at both radii.

    Returns
    -------
    types.SimpleNamespace
        Branch coefficients ``ABCD`` (delta) or ``ABCDE`` (smooth), mode/viscosity/forcing
        metadata and pressure coefficients for the symbolic field builders.

    Raises
    ------
    NotImplementedError
        A smooth builder is asked for a resonant radial power.

    Notes
    -----
    These low-level helpers assume validated radii, modes and nonsingular powers. Use the public
    Stokes class for input validation. No mesh, PDE solve or unit conversion is performed; the
    angular coefficient normalization is supplied by the caller.
    """
    if (k + 1) * (k + 2) == l * (l + 1) or (k + 3) * (k + 4) == l * (l + 1):
        raise NotImplementedError(f"Smooth solution not implemented for k={k}, l={l}")
    coeffs = coefficients_sphere_smooth_ns(Rp, Rm, k, l, g, nu) if no_slip else coefficients_sphere_smooth_fs(Rp, Rm, k, l, g, nu)
    _, _, C, D, _ = coeffs
    return SimpleNamespace(
        l=l,
        m=m,
        k=k,
        g=g,
        nu=nu,
        ABCDE=coeffs,
        G=-2 * nu * (l + 1) * (2 * l + 3) * C,
        H=-2 * nu * l * (2 * l - 1) * D,
        K=-g * (k + 2) / ((k + 1) * (k + 2) - l * (l + 1)) / Rp**k,
    )


def analytical_velocity_cartesian_sympy(soln, r_sym, y_sym, theta_sym, phi_sym, rrotN):
    r"""Construct the separated Kramer symbolic field.

    Parameters
    ----------
    soln : types.SimpleNamespace
        Branch returned by ``build_delta_solution`` or ``build_smooth_solution`` with its
        radial and pressure coefficients.
    r_sym : sympy.Symbol
        Positive symbolic radial coordinate; the shell formulas exclude the origin.
    y_sym : sympy expression
        Chosen spherical harmonic, including the caller's order, real component and normalization.
    theta_sym : sympy.Symbol
        Colatitude variable in radians, used to differentiate ``y_sym``.
    phi_sym : sympy.Symbol
        Azimuth variable in radians, used to differentiate ``y_sym``.
    rrotN : sympy.Matrix
        Cartesian-to-polar/spherical rotation matrix; its transpose maps the local velocity
        components back to Cartesian coordinates.

    Returns
    -------
    sympy.Matrix
        Cartesian velocity column with two components (cylindrical) or three (spherical), in
        the input velocity scaling.

    Notes
    -----
    No point evaluation or PDE solve is performed. The separated spherical velocity formula uses
    angular-coordinate derivatives; use the public spherical class's Cartesian evaluator for finite
    values at poles.
    """
    if hasattr(soln, "ABCD"):
        A, B, C, D = soln.ABCD
        P_l = A * r_sym**soln.l + B * r_sym ** (-soln.l - 1) + C * r_sym ** (soln.l + 2) + D * r_sym ** (-soln.l + 1)
    elif hasattr(soln, "ABCDE"):
        A, B, C, D, E = soln.ABCDE
        P_l = A * r_sym**soln.l + B * r_sym ** (-soln.l - 1) + C * r_sym ** (soln.l + 2) + D * r_sym ** (-soln.l + 1) + E * r_sym ** (soln.k + 3)
    else:
        raise TypeError(f"Unsupported analytical solution type: {type(soln)}")

    dPldr = sp.diff(P_l, r_sym)
    prefactor = -(P_l / r_sym + dPldr)
    u_r = -soln.l * (soln.l + 1) * P_l * y_sym / r_sym
    u_theta = prefactor * sp.diff(y_sym, theta_sym)
    u_phi = prefactor * sp.diff(y_sym, phi_sym) / sp.sin(theta_sym)
    return rrotN.T * sp.Matrix([u_r, u_theta, u_phi])


def analytical_pressure_sympy(soln, r_sym, y_sym):
    r"""Construct the separated Kramer symbolic field.

    Parameters
    ----------
    soln : types.SimpleNamespace
        Branch returned by ``build_delta_solution`` or ``build_smooth_solution`` with its
        radial and pressure coefficients.
    r_sym : sympy.Symbol
        Positive symbolic radial coordinate; the shell formulas exclude the origin.
    y_sym : sympy expression
        Chosen spherical harmonic, including the caller's order, real component and normalization.

    Returns
    -------
    sympy expression
        Symbolic pressure, radial stress or radial poloidal scalar in the branch scaling;
        total normal stress is ``tau_rr-p``.

    Notes
    -----
    No point evaluation or PDE solve is performed. The separated spherical velocity formula uses
    angular-coordinate derivatives; use the public spherical class's Cartesian evaluator for finite
    values at poles.
    """
    p_sym = (soln.G * r_sym**soln.l + soln.H * r_sym ** (-soln.l - 1)) * y_sym
    if hasattr(soln, "K"):
        p_sym += soln.K * r_sym ** (soln.k + 1) * y_sym
    return p_sym


def analytical_Pl_sympy(soln, r_sym):
    r"""Construct the separated Kramer symbolic field.

    Parameters
    ----------
    soln : types.SimpleNamespace
        Branch returned by ``build_delta_solution`` or ``build_smooth_solution`` with its
        radial and pressure coefficients.
    r_sym : sympy.Symbol
        Positive symbolic radial coordinate; the shell formulas exclude the origin.

    Returns
    -------
    sympy expression
        Radial poloidal scalar P_l; this is not the angular Legendre polynomial.

    Notes
    -----
    No point evaluation or PDE solve is performed. The separated spherical velocity formula uses
    angular-coordinate derivatives; use the public spherical class's Cartesian evaluator for finite
    values at poles.
    """
    if hasattr(soln, "ABCD"):
        A, B, C, D = soln.ABCD
        return A * r_sym**soln.l + B * r_sym ** (-soln.l - 1) + C * r_sym ** (soln.l + 2) + D * r_sym ** (-soln.l + 1)
    if hasattr(soln, "ABCDE"):
        A, B, C, D, E = soln.ABCDE
        return A * r_sym**soln.l + B * r_sym ** (-soln.l - 1) + C * r_sym ** (soln.l + 2) + D * r_sym ** (-soln.l + 1) + E * r_sym ** (soln.k + 3)
    raise TypeError(f"Unsupported analytical solution type: {type(soln)}")


def analytical_radial_normal_stress_sympy(soln, r_sym, y_sym):
    r"""Native analytical radial normal stress: sigma_rr = tau_rr - p.

    Parameters
    ----------
    soln : types.SimpleNamespace
        Branch returned by ``build_delta_solution`` or ``build_smooth_solution`` with its
        radial and pressure coefficients.
    r_sym : sympy.Symbol
        Positive symbolic radial coordinate; the shell formulas exclude the origin.
    y_sym : sympy expression
        Chosen spherical harmonic, including the caller's order, real component and normalization.

    Returns
    -------
    sympy expression
        Symbolic pressure, radial stress or radial poloidal scalar in the branch scaling;
        total normal stress is ``tau_rr-p``.

    Notes
    -----
    No point evaluation or PDE solve is performed. The separated spherical velocity formula uses
    angular-coordinate derivatives; use the public spherical class's Cartesian evaluator for finite
    values at poles.
    """

    P_l = analytical_Pl_sympy(soln, r_sym)
    dPldr = sp.diff(P_l, r_sym)
    tau_rr = -2 * soln.nu * soln.l * (soln.l + 1) * (dPldr - P_l / r_sym) * y_sym / r_sym
    return sp.simplify(tau_rr - analytical_pressure_sympy(soln, r_sym, y_sym))


# Historical name retained for existing benchmark consumers; no assess import.
analytical_radial_stress_assess_sympy = analytical_radial_normal_stress_sympy


import numpy as np
from numbers import Integral

from .._base import AnalyticSolution, free_slip, prescribed_velocity
from .._kramer_geoid import _KramerGeoidMixin
from underworld3.maths.spherical_harmonics import (
    AXISYMMETRIC_UNNORMALIZED, solid_harmonic_sympy, validate_harmonic,
)


class KramerSphericalStokes(_KramerGeoidMixin, AnalyticSolution):
    r"""Native symbolic Kramer spherical shell with cached Cartesian evaluation.

    Parameters ell>=2, 0<=m<=ell, component ('cos'/'sin') and normalization
    specify the same real harmonic in forcing, velocity, pressure and stress.
    The default basis is unnormalised P_ell for m=0; use 'real_orthonormal'
    for nonzero m. Delta forcing g is the outward radial normal-stress jump,
    so its physical sheet force is -g*Y*e_r. Smooth forcing is
    -g*(r/r_outer)**k*Y*e_r. Defaults g=1, viscosity=1.

    Resonant smooth powers and ell=0/1 Stokes families are explicitly excluded.
    Both poles have exact Cartesian values; only the origin is excluded.
    Points at the delta radius select the outer branch. Bulk residuals should
    be checked away from the sheet, whose jumps must be checked separately.

    Parameters
    ----------
    mesh : underworld3.discretisation.Mesh
        Mesh with the solution's dimension and coordinate symbols. Construction
        builds expressions/evaluators; it does not solve a PDE or create a mesh.
    ell : int, default=2
        Harmonic degree ell >= 2; exceptional ell=0/1 Stokes families are excluded.
    m : int, default=0
        Harmonic order satisfying 0 <= m <= ell.
    k : int, default=2
        Smooth density exponent, unused for delta density. Resonances
        ``(k+1)*(k+2)==ell*(ell+1)`` or ``(k+3)*(k+4)==ell*(ell+1)`` are excluded.
    r_inner, r_outer : float
        Finite positive shell radii, default 1.22 and 2.22, ordered inner < outer.
    r_int : float, default=1.72
        Finite sheet radius strictly inside the shell for delta density;
        unused for smooth density.
    density : {"delta", "smooth"}, default="delta"
        Conforming sheet or radial power-law volume density times the selected Y_lm.
    boundary : {"free", "zero"}, default="free"
        Impermeable free slip or zero velocity on both walls.
    viscosity : float, default=1
        Finite positive constant viscosity in the chosen scaling.
    forcing : float, default=1
        Finite signed Kramer coefficient g; no automatic density/unit conversion.
    component : {"cos", "sin"}, default="cos"
        Real angular component; order zero requires cos.
    normalization : str, default="axisymmetric_unnormalized"
        Either "axisymmetric_unnormalized" or "real_orthonormal", setting
        the common forcing/field normalization. The unnormalized basis is defined
        only for m=0; use real_orthonormal for nonzero order.

    Raises
    ------
    TypeError
        The radial power (or spherical degree/order) has an invalid type.
    ValueError
        Mesh dimension, mode, geometry, density/wall choice, viscosity or forcing
        violates the constructor contract.
    NotImplementedError
        The selected smooth power belongs to a resonant logarithmic family.

    Notes
    -----
    Use one consistent coordinate, viscosity and force scaling; no conversion
    is performed. Delta bulk force is zero and the outward normal-stress jump
    is ``g*Y``, so its physical sheet force in ``div(sigma)+f=0`` is ``-g*Y*e_r``.
    Smooth body force is ``-g*(r/r_outer)**k*Y*e_r``. At the delta radius numerical
    evaluation selects the outer branch; check bulk equations away from the sheet.
    ``evaluation_cse`` controls cached NumPy compilation and leaves fields unchanged.
    """

    dim = 3
    geometry = "spherical"
    boundaries = ("Upper", "Lower")
    reference = "Kramer, Davies & Wilson (2021), doi:10.5194/gmd-14-1899-2021."
    evaluation_cse = True
    _fields = dict(AnalyticSolution._fields,
                   radial_normal_stress="fn_radial_normal_stress",
                   radial_shear_stress="fn_radial_shear_stress")

    def __init__(self, mesh, ell=2, m=0, k=2, r_inner=1.22, r_outer=2.22,
                 r_int=1.72, density="delta", boundary="free", viscosity=1.0,
                 forcing=1.0, component="cos", normalization=AXISYMMETRIC_UNNORMALIZED):
        super().__init__(mesh)
        self.ell, self.m, self.component, self.normalization = validate_harmonic(
            ell, m, component, normalization)
        if self.ell < 2:
            raise ValueError("ell must be >=2; ell=0/1 Stokes families are not implemented.")
        if isinstance(k, bool) or not isinstance(k, Integral):
            raise TypeError("k must be an integer.")
        if density not in ("delta", "smooth") or boundary not in ("free", "zero"):
            raise ValueError("density must be delta/smooth and boundary must be free/zero.")
        self.k, self.density, self.boundary = int(k), density, boundary
        self.r_inner, self.r_outer, self.r_int = map(float, (r_inner, r_outer, r_int))
        self.viscosity, self.g = float(viscosity), float(forcing)
        if not np.all(np.isfinite((self.r_inner, self.r_outer, self.r_int, self.viscosity, self.g))):
            raise ValueError("Radii, viscosity and forcing must be finite.")
        if not 0 < self.r_inner < self.r_outer or self.viscosity <= 0:
            raise ValueError("Expected 0 < r_inner < r_outer and viscosity > 0.")
        if density == "delta" and not self.r_inner < self.r_int < self.r_outer:
            raise ValueError("r_int must lie strictly inside the shell.")
        if density == "delta":
            self._below, self._above = [build_delta_solution(
                self.r_outer, self.r_inner, self.r_int, self.ell, self.m,
                self.g, self.viscosity, sign, boundary == "zero") for sign in (-1, 1)]
        else:
            self._below = self._above = build_smooth_solution(
                self.r_outer, self.r_inner, self.k, self.ell, self.m,
                self.g, self.viscosity, boundary == "zero")
        self._xyz = sp.symbols("_kramer_x _kramer_y _kramer_z", real=True)
        self._r = r = sp.Symbol("_kramer_radius", positive=True)
        X = sp.Matrix(self._xyz)
        H = solid_harmonic_sympy(self._xyz, degree=self.ell, order=self.m,
                                component=self.component, normalization=self.normalization)
        gradient = sp.Matrix([sp.diff(H, value) for value in self._xyz])
        hessian = sp.hessian(H, self._xyz)
        ell = self.ell
        self._cartesian_fields = []
        for branch in (self._below, self._above):
            P = analytical_Pl_sympy(branch, r)
            derivative = sp.diff(P, r)
            F = sp.expand(-(P/r + derivative)*r**(1-ell))
            J = sp.expand(ell*(derivative-ell*P/r)*r**(-ell-1))
            velocity = F*gradient + J*H*X
            # Differentiate the radial factors and the Cartesian polynomial
            # separately, without introducing theta/phi divisions at the poles.
            jacobian = (F*hessian + sp.diff(F, r)/r*gradient*X.T
                        + J*(X*gradient.T + H*sp.eye(3))
                        + sp.diff(J, r)/r*H*X*X.T)
            strain = (jacobian+jacobian.T)/2
            pressure = analytical_pressure_sympy(branch, r, H/r**ell)
            stress = 2*self.viscosity*strain-pressure*sp.eye(3)
            radial = (-2*self.viscosity*ell*(ell+1)*(derivative-P/r)/r
                      *H/r**ell-pressure)
            shear = (self.viscosity*((2-ell*(ell+1))*P/r**2-sp.diff(P,r,2))
                     *(r**(1-ell)*gradient-ell*H*X/r**(ell+1)))
            bodyforce = (sp.zeros(3, 1) if density == "delta" else
                         -self.g*(r/self.r_outer)**self.k*H*X/r**(ell+1))
            self._cartesian_fields.append(dict(
                velocity=velocity, pressure=pressure, strainrate=strain, stress=stress,
                viscosity=sp.Float(self.viscosity), bodyforce=bodyforce,
                radial_normal_stress=radial, radial_shear_stress=shear))
        radius = sp.sqrt(sum(value*value for value in mesh.X))
        substitutions = dict(zip(self._xyz, list(mesh.X))) | {r: radius}

        def spatial(field):
            lower, upper = [branch[field] for branch in self._cartesian_fields]
            if density == "smooth":
                return lower.subs(substitutions)
            if isinstance(lower, sp.MatrixBase):
                return sp.Matrix(lower.rows, lower.cols, lambda i,j: sp.Piecewise(
                    (upper[i,j], r >= self.r_int), (lower[i,j], True))).subs(substitutions)
            return sp.Piecewise((upper, r >= self.r_int), (lower, True)).subs(substitutions)

        self.set_fields(velocity=list(spatial("velocity")), pressure=spatial("pressure"),
                        viscosity=self.viscosity, bodyforce=list(spatial("bodyforce")),
                        stress=spatial("stress"))
        self.fn_radial_normal_stress = spatial("radial_normal_stress")
        # Cartesian tangential traction vector, not an angular scalar.
        self.fn_radial_shear_stress = spatial("radial_shear_stress")
        self.fn_harmonic = (H/r**ell).subs(substitutions)
        self.fn_interface_stress_jump = self.g*self.fn_harmonic if density == "delta" else sp.S.Zero
        self._evaluators = {}

    def apply_boundary_conditions(self, solver):
        r"""Apply boundary conditions.

        Parameters
        ----------
        solver : underworld3.systems.Stokes
            Solver receiving the reference wall data on Upper and Lower before solving.

        Returns
        -------
        None
            Mutates wall conditions and enables the pressure nullspace.

        Notes
        -----
        Select homogeneous free/zero slip according to boundary. This performs no solve.
        """
        if self.boundary == "zero":
            prescribed_velocity(solver, self.boundaries, (0., 0., 0.))
        else:
            free_slip(solver, self.boundaries)
        solver.petsc_use_pressure_nullspace = True

    def apply_forcing(self, solver, internal_boundary="Internal"):
        r"""Apply forcing.

        Parameters
        ----------
        solver : underworld3.systems.Stokes
            Solver on the reference mesh whose bodyforce is configured before solving.
        internal_boundary : str, default="Internal"
            Label for a conforming delta sheet, required only for delta density.

        Returns
        -------
        None
            Mutates solver forcing; it does not run the solver.

        Raises
        ------
        ValueError
            Kramer solver/reference meshes differ or a delta interface label is missing.

        Notes
        -----
        Call once, separately from applying the wall conditions. Kramer delta cases add the natural
        stress-jump datum g*Y*e_r with opposite physical-force sign; smooth cases set the bulk
        force.
        """
        if solver.mesh is not self.mesh:
            raise ValueError("Solver and solution must share the same mesh.")
        if self.density == "delta" and internal_boundary not in self.mesh.boundaries.__members__:
            raise ValueError(f"Delta forcing needs a conforming {internal_boundary!r} interface.")
        solver.bodyforce = self.fn_bodyforce
        if self.density == "delta":
            X = sp.Matrix(list(self.mesh.X))
            solver.add_natural_bc(self.fn_interface_stress_jump*X/sp.sqrt(X.dot(X)), internal_boundary)

    def compile_evaluators(self, *fields):
        r"""Compile evaluators.

        Parameters
        ----------
        *fields : str
            Names of fields to prepare; use the same names accepted by evaluate.
            An empty argument list compiles nothing.

        Returns
        -------
        None
            Cached evaluators are prepared without sampling points or solving a PDE.

        Raises
        ------
        ValueError
            A requested field is unknown.

        Notes
        -----
        The cache includes evaluation_cse; toggling it selects separately compiled
        common-subexpression-elimination evaluators.
        """
        for field in fields:
            if field not in self._cartesian_fields[0]:
                raise ValueError(f"Unknown Kramer field {field!r}.")
            for branch in range(2 if self.density == "delta" else 1):
                key = (field, branch, bool(self.evaluation_cse))
                if key not in self._evaluators:
                    expression = self._cartesian_fields[branch][field]
                    terms = list(expression) if isinstance(expression, sp.MatrixBase) else [expression]
                    self._evaluators[key] = sp.lambdify((*self._xyz, self._r), terms, "numpy",
                                                      cse=bool(self.evaluation_cse), docstring_limit=0)

    def evaluate(self, field, coords):
        r"""Evaluate the analytical reference field.

        Parameters
        ----------
        field : str or sympy expression
            Named field or symbolic expression. Native names include velocity,
            pressure, viscosity, bodyforce, strainrate, stress and radial_normal_stress;
            Additional named traction field: radial_shear_stress.
        coords : array-like, shape (N, 3)
            Finite Cartesian coordinates in the reference scaling; the origin is
            excluded. Shell membership is the caller's responsibility.

        Returns
        -------
        numpy.ndarray
            One value per point: scalar (N,), vector (N, 3), or tensor
            (N, 3, 3). Spherical Kramer radial_shear_stress is a
            Cartesian tangential traction vector. Arbitrary expressions use the UW3
            evaluator's expression shape. Empty arrays retain the corresponding shape.

        Raises
        ------
        ValueError
            Coordinates or the requested field are invalid.

        Notes
        -----
        Named fields use cached vectorized NumPy evaluators; symbolic expressions delegate to the
        shared JIT evaluator. This performs no Stokes solve. Delta sheet points select the outer
        branch; Cartesian spherical evaluators are finite at both poles.
        """
        if not isinstance(field, str):
            return super().evaluate(field, coords)
        if field not in self._cartesian_fields[0]:
            raise ValueError(f"Unknown Kramer field {field!r}.")
        points = np.asarray(coords, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
            raise ValueError("coords must be a finite array of shape (N,3).")
        radii = np.linalg.norm(points, axis=1)
        if np.any(radii == 0):
            raise ValueError("The spherical shell solution is undefined at the origin.")
        expression = self._cartesian_fields[0][field]
        shape = expression.shape if isinstance(expression, sp.MatrixBase) else ()
        if field in ("velocity", "bodyforce", "radial_shear_stress"):
            shape = (3,)
        result = np.empty((len(points), *shape))
        above = radii >= self.r_int if self.density == "delta" else np.zeros(len(points), dtype=bool)
        self.compile_evaluators(field)
        for branch, mask in ((0, ~above), (1, above)):
            if np.any(mask):
                values = self._evaluators[(field,branch,bool(self.evaluation_cse))](
                    *points[mask].T, radii[mask])
                result[mask] = np.column_stack([np.broadcast_to(v, (mask.sum(),)) for v in values]).reshape(
                    (mask.sum(), *shape))
        return result

    def sample_points(self, count=12):
        r"""Sample points.

        Parameters
        ----------
        count : int, default=12
            Requested number of interior diagnostic points; use a positive count.

        Returns
        -------
        numpy.ndarray, shape (count, 3)
            Deterministic Cartesian points strictly between the two reference radii.

        Notes
        -----
        This samples the continuous reference independently of mesh nodes. Kramer perturbs points
        that would coincide with the internal sheet.
        """
        radii = self.r_inner+(self.r_outer-self.r_inner)*(np.arange(count)+0.37)/count
        radii[np.isclose(radii,self.r_int)] += 1e-4*(self.r_outer-self.r_inner)
        z = 1-2*(np.arange(count)+0.5)/count
        phi = np.arange(count)*2.399963229728653
        return radii[:,None]*np.column_stack((np.sqrt(1-z*z)*np.cos(phi),
                                             np.sqrt(1-z*z)*np.sin(phi),z))


# Historical name refers to the canonical class; no second implementation.
SphericalStokes = KramerSphericalStokes
