"""Real spherical-harmonic conventions used by the physical benchmarks.

The orthonormal basis follows the Condon--Shortley convention used by
``scipy.special.lpmv`` and ``sympy.assoc_legendre``:

``Y_lm^c = sqrt(2) Re(Y_l^m)`` and
``Y_lm^s = sqrt(2) Im(Y_l^m)`` for ``m>0``.  The ``m=0`` cosine member is
the usual complex-harmonic ``Y_l^0`` and the sine member does not exist.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import lpmv
from numbers import Integral
import sympy as sp


REAL_ORTHONORMAL = "real_orthonormal"
AXISYMMETRIC_UNNORMALIZED = "axisymmetric_unnormalized"


def validate_harmonic(
    degree: int,
    order: int,
    component: str,
    normalization: str,
) -> tuple[int, int, str, str]:
    if isinstance(degree, bool) or not isinstance(degree, Integral):
        raise TypeError("harmonic degree must be an integer.")
    if isinstance(order, bool) or not isinstance(order, Integral):
        raise TypeError("harmonic order must be an integer.")
    degree = int(degree)
    order = int(order)
    component = str(component).strip().lower()
    normalization = str(normalization).strip().lower()
    if degree < 1:
        raise ValueError("harmonic degree must be at least one.")
    if order < 0 or order > degree:
        raise ValueError("harmonic order must satisfy 0 <= m <= ell.")
    if component not in ("cos", "sin"):
        raise ValueError("harmonic component must be cos or sin.")
    if order == 0 and component != "cos":
        raise ValueError("The m=0 sine harmonic is identically zero.")
    if normalization not in (REAL_ORTHONORMAL, AXISYMMETRIC_UNNORMALIZED):
        raise ValueError(
            "harmonic normalization must be real_orthonormal or " "axisymmetric_unnormalized."
        )
    if normalization == AXISYMMETRIC_UNNORMALIZED and order != 0:
        raise ValueError("axisymmetric_unnormalized is defined only for m=0.")
    return degree, order, component, normalization


def harmonic_norm(
    degree: int,
    order: int = 0,
    component: str = "cos",
    normalization: str = AXISYMMETRIC_UNNORMALIZED,
) -> float:
    """Return the angular integral of the squared requested real harmonic."""

    degree, _order, _component, normalization = validate_harmonic(
        degree, order, component, normalization
    )
    if normalization == REAL_ORTHONORMAL:
        return 1.0
    return float(4.0 * np.pi / (2 * degree + 1))


def _orthonormal_factor(degree: int, order: int) -> float:
    return math.sqrt(
        (2 * degree + 1)
        / (4.0 * math.pi)
        * math.factorial(degree - order)
        / math.factorial(degree + order)
    )


def real_harmonic_values(
    coordinates: np.ndarray,
    *,
    degree: int,
    order: int = 0,
    component: str = "cos",
    normalization: str = AXISYMMETRIC_UNNORMALIZED,
) -> np.ndarray:
    """Evaluate a real harmonic at Cartesian coordinates."""

    degree, order, component, normalization = validate_harmonic(
        degree, order, component, normalization
    )
    coordinates = np.asarray(coordinates, dtype=float)
    scalar_input = coordinates.ndim == 1
    coordinates = np.atleast_2d(coordinates)
    if coordinates.shape[1] != 3:
        raise ValueError("coordinates must have shape (N, 3).")
    radii = np.linalg.norm(coordinates, axis=1)
    if np.any(radii <= 0.0):
        raise ValueError("spherical harmonics are undefined at the origin.")
    cosine_theta = np.clip(coordinates[:, 2] / radii, -1.0, 1.0)
    if normalization == AXISYMMETRIC_UNNORMALIZED:
        values = lpmv(0, degree, cosine_theta)
    else:
        azimuth = np.arctan2(coordinates[:, 1], coordinates[:, 0])
        values = _orthonormal_factor(degree, order) * lpmv(order, degree, cosine_theta)
        if order > 0:
            values *= np.sqrt(2.0)
            values *= np.cos(order * azimuth) if component == "cos" else np.sin(order * azimuth)
    return values[0] if scalar_input else values


def real_harmonic_sympy(
    theta,
    phi,
    *,
    degree: int,
    order: int = 0,
    component: str = "cos",
    normalization: str = AXISYMMETRIC_UNNORMALIZED,
):
    """Return the same real harmonic as a SymPy expression."""

    degree, order, component, normalization = validate_harmonic(
        degree, order, component, normalization
    )
    associated = sp.assoc_legendre(degree, order, sp.cos(theta))
    if normalization == AXISYMMETRIC_UNNORMALIZED:
        return associated
    factor = sp.sqrt(
        sp.Rational(2 * degree + 1, 4)
        / sp.pi
        * sp.factorial(degree - order)
        / sp.factorial(degree + order)
    )
    if order == 0:
        return factor * associated
    azimuthal = sp.cos(order * phi) if component == "cos" else sp.sin(order * phi)
    return sp.sqrt(2) * factor * associated * azimuthal


def harmonic_case_label(
    degree: int,
    order: int,
    component: str,
    normalization: str,
) -> str:
    degree, order, component, normalization = validate_harmonic(
        degree, order, component, normalization
    )
    suffix = "" if order == 0 else component[0]
    norm = "real" if normalization == REAL_ORTHONORMAL else "legacy"
    return f"l{degree}_m{order}{suffix}_{norm}"


def solid_harmonic_sympy(
    coordinates, *, degree, order=0, component="cos",
    normalization=AXISYMMETRIC_UNNORMALIZED,
):
    """Return the Cartesian polynomial r**ell Y_ell^m, including both poles.

    Multiplying the associated Legendre polynomial by the real/imaginary
    part of (x+i*y)**m cancels the apparent azimuthal singularity exactly.
    No angular coordinates or divisions by the cylindrical radius are used.
    """
    degree, order, component, normalization = validate_harmonic(
        degree, order, component, normalization
    )
    if len(coordinates) != 3:
        raise ValueError("coordinates must contain three Cartesian expressions.")
    x, y, z = coordinates
    cosine = sp.Symbol("_harmonic_cosine", real=True)
    derivative = sp.Poly(sp.diff(sp.legendre(degree, cosine), cosine, order), cosine)
    azimuth = sum(sp.binomial(order, j)*(-1)**(j//2)*x**(order-j)*y**j
                  for j in range(0 if component == "cos" else 1, order+1, 2))
    radius_squared = x*x + y*y + z*z
    polynomial = (-1)**order * azimuth * sum(
        coefficient * z**power[0] * radius_squared**((degree-order-power[0])//2)
        for power, coefficient in derivative.terms()
    )
    if normalization == REAL_ORTHONORMAL:
        factor = sp.sqrt(sp.Rational(2*degree+1, 4)/sp.pi
                         * sp.factorial(degree-order)/sp.factorial(degree+order))
        if order:
            factor *= sp.sqrt(2)
        polynomial *= factor
    return sp.expand(polynomial)


def real_harmonic_cartesian_sympy(coordinates, **parameters):
    """Return a pole-safe real harmonic as a Cartesian shell expression."""
    return solid_harmonic_sympy(coordinates, **parameters) / sp.sqrt(
        sum(value*value for value in coordinates)
    )**parameters["degree"]


__all__ = [
    "AXISYMMETRIC_UNNORMALIZED",
    "REAL_ORTHONORMAL",
    "harmonic_case_label",
    "harmonic_norm",
    "real_harmonic_sympy",
    "real_harmonic_values",
    "solid_harmonic_sympy",
    "real_harmonic_cartesian_sympy",
    "validate_harmonic",
]
