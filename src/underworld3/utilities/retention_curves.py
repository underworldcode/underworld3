r"""
Retention curve functions for variably-saturated porous media flow.

Provides retention curve functions that return SymPy expressions
suitable for use with the Richards equation solver:

- **Van Genuchten--Mualem** model: general-purpose, widely used.
- **Gardner exponential** model: simpler, admits exact analytical
  solutions for steady-state Richards equation with gravity.
- **Haverkamp** model: rational-function form with independent
  parameters for retention and conductivity; used in the Vauclin
  (1979) water-table recharge benchmark.

All functions accept a SymPy symbol or expression for pressure head
(typically ``psi_field.sym[0]``) and return SymPy expressions that
can be assigned directly to solver properties.

References
----------
Van Genuchten, M. Th. (1980). A closed-form equation for predicting
the hydraulic conductivity of unsaturated soils.
*Soil Science Society of America Journal*, 44(5), 892--898.

Mualem, Y. (1976). A new model for predicting the hydraulic
conductivity of unsaturated porous media.
*Water Resources Research*, 12(3), 513--522.

Gardner, W. R. (1958). Some steady-state solutions of the
unsaturated moisture flow equation with application to evaporation
from a water table. *Soil Science*, 85(4), 228--232.

Haverkamp, R. et al. (1977). A comparison of numerical simulation
models for one-dimensional infiltration.
*Soil Science Society of America Journal*, 41(2), 285--294.

Examples
--------
>>> import sympy
>>> from underworld3.utilities.retention_curves import (
...     van_genuchten_Se, van_genuchten_K, van_genuchten_C,
... )
>>> psi = sympy.Symbol("psi")
>>> Se = van_genuchten_Se(psi, alpha=3.35, n=2.0)
>>> K = van_genuchten_K(psi, Ks=1e-4, alpha=3.35, n=2.0)
>>> C = van_genuchten_C(psi, theta_r=0.045, theta_s=0.43, alpha=3.35, n=2.0)

Gardner model with analytical steady-state solution:

>>> from underworld3.utilities.retention_curves import (
...     gardner_K, gardner_C, gardner_steady_state_psi,
... )
>>> K = gardner_K(psi, Ks=1e-4, alpha=1.0)
>>> C = gardner_C(psi, theta_r=0.05, theta_s=0.4, alpha=1.0)

Haverkamp model (Vauclin benchmark parameters):

>>> from underworld3.utilities.retention_curves import (
...     haverkamp_theta, haverkamp_K, haverkamp_C,
... )
>>> theta = haverkamp_theta(psi, theta_r=0.075, theta_s=0.287,
...     alpha=1.611e6, beta=3.96)
>>> K = haverkamp_K(psi, Ks=9.44e-5, A=1.175e6, B=4.74)
"""

import sympy


def van_genuchten_Se(psi, alpha, n, m=None):
    r"""Effective saturation (Van Genuchten model).

    .. math::

        S_e(\psi) = \begin{cases}
            \left[1 + (\alpha |\psi|)^n\right]^{-m} & \psi < 0 \\
            1 & \psi \ge 0
        \end{cases}

    Parameters
    ----------
    psi : sympy expression
        Pressure head (negative in unsaturated zone).
    alpha : float
        Inverse of the air-entry pressure [1/length].
    n : float
        Pore-size distribution parameter (n > 1).
    m : float, optional
        Van Genuchten parameter. Default: ``1 - 1/n``.

    Returns
    -------
    sympy.Piecewise
        Effective saturation expression.
    """
    if m is None:
        m = 1 - 1 / sympy.Rational(n) if isinstance(n, int) else 1 - 1 / n

    alpha = sympy.sympify(alpha)
    n = sympy.sympify(n)
    m = sympy.sympify(m)

    Se_unsat = (1 + (alpha * (-psi)) ** n) ** (-m)
    return sympy.Piecewise((sympy.S.One, psi >= 0), (Se_unsat, True))


def van_genuchten_theta(psi, theta_r, theta_s, alpha, n, m=None):
    r"""Volumetric water content (Van Genuchten model).

    .. math::

        \theta(\psi) = \theta_r + (\theta_s - \theta_r)\, S_e(\psi)

    Parameters
    ----------
    psi : sympy expression
        Pressure head.
    theta_r : float
        Residual water content.
    theta_s : float
        Saturated water content.
    alpha : float
        Inverse of the air-entry pressure [1/length].
    n : float
        Pore-size distribution parameter.
    m : float, optional
        Default: ``1 - 1/n``.

    Returns
    -------
    sympy.Expr
        Water content expression.
    """
    theta_r = sympy.sympify(theta_r)
    theta_s = sympy.sympify(theta_s)
    Se = van_genuchten_Se(psi, alpha, n, m)
    return theta_r + (theta_s - theta_r) * Se


def van_genuchten_K(psi, Ks, alpha, n, m=None):
    r"""Hydraulic conductivity (Van Genuchten–Mualem model).

    .. math::

        K(\psi) = \begin{cases}
            K_s \, S_e^{1/2}
            \left[1 - \left(1 - S_e^{1/m}\right)^m\right]^2
            & \psi < 0 \\
            K_s & \psi \ge 0
        \end{cases}

    Parameters
    ----------
    psi : sympy expression
        Pressure head.
    Ks : float
        Saturated hydraulic conductivity.
    alpha : float
        Inverse of the air-entry pressure [1/length].
    n : float
        Pore-size distribution parameter.
    m : float, optional
        Default: ``1 - 1/n``.

    Returns
    -------
    sympy.Piecewise
        Hydraulic conductivity expression.
    """
    if m is None:
        m = 1 - 1 / sympy.Rational(n) if isinstance(n, int) else 1 - 1 / n

    Ks = sympy.sympify(Ks)
    m = sympy.sympify(m)

    Se = van_genuchten_Se(psi, alpha, n, m)
    # For the unsaturated branch, extract the unsaturated Se expression
    alpha_s = sympy.sympify(alpha)
    n_s = sympy.sympify(n)
    Se_expr = (1 + (alpha_s * (-psi)) ** n_s) ** (-m)

    K_unsat = Ks * Se_expr ** sympy.Rational(1, 2) * (
        1 - (1 - Se_expr ** (1 / m)) ** m
    ) ** 2

    return sympy.Piecewise((Ks, psi >= 0), (K_unsat, True))


def van_genuchten_C(psi, theta_r, theta_s, alpha, n, m=None, Ss=0.0):
    r"""Specific moisture capacity (Van Genuchten model).

    .. math::

        C(\psi) = \frac{d\theta}{d\psi} = \begin{cases}
            \alpha\, m\, n\, (\theta_s - \theta_r)\,
            (\alpha |\psi|)^{n-1}\,
            \left[1 + (\alpha |\psi|)^n\right]^{-(m+1)}
            & \psi < 0 \\
            S_s & \psi \ge 0
        \end{cases}

    Parameters
    ----------
    psi : sympy expression
        Pressure head.
    theta_r : float
        Residual water content.
    theta_s : float
        Saturated water content.
    alpha : float
        Inverse of the air-entry pressure [1/length].
    n : float
        Pore-size distribution parameter.
    m : float, optional
        Default: ``1 - 1/n``.
    Ss : float, optional
        Specific storage for the saturated zone (default 0).
        Set to a small positive value (e.g. 1e-4) to avoid
        a singular mass matrix when the domain is fully saturated.

    Returns
    -------
    sympy.Piecewise
        Specific moisture capacity expression.
    """
    if m is None:
        m = 1 - 1 / sympy.Rational(n) if isinstance(n, int) else 1 - 1 / n

    alpha = sympy.sympify(alpha)
    n = sympy.sympify(n)
    m = sympy.sympify(m)
    theta_r = sympy.sympify(theta_r)
    theta_s = sympy.sympify(theta_s)
    Ss = sympy.sympify(Ss)

    C_unsat = (
        alpha
        * m
        * n
        * (theta_s - theta_r)
        * (alpha * (-psi)) ** (n - 1)
        * (1 + (alpha * (-psi)) ** n) ** (-(m + 1))
    )

    return sympy.Piecewise((Ss, psi >= 0), (C_unsat, True))


# =====================================================================
# Gardner exponential model
# =====================================================================


def gardner_K(psi, Ks, alpha):
    r"""Hydraulic conductivity (Gardner exponential model).

    .. math::

        K(\psi) = \begin{cases}
            K_s \exp(\alpha\,\psi) & \psi < 0 \\
            K_s & \psi \ge 0
        \end{cases}

    Parameters
    ----------
    psi : sympy expression
        Pressure head (negative in unsaturated zone).
    Ks : float
        Saturated hydraulic conductivity.
    alpha : float
        Sorptive number [1/length]. Larger values give a
        sharper transition near saturation.

    Returns
    -------
    sympy.Piecewise
        Hydraulic conductivity expression.
    """
    Ks = sympy.sympify(Ks)
    alpha = sympy.sympify(alpha)

    K_unsat = Ks * sympy.exp(alpha * psi)
    return sympy.Piecewise((Ks, psi >= 0), (K_unsat, True))


def gardner_theta(psi, theta_r, theta_s, alpha):
    r"""Volumetric water content (Gardner exponential model).

    .. math::

        \theta(\psi) = \begin{cases}
            \theta_r + (\theta_s - \theta_r)\,\exp(\alpha\,\psi)
            & \psi < 0 \\
            \theta_s & \psi \ge 0
        \end{cases}

    Parameters
    ----------
    psi : sympy expression
        Pressure head.
    theta_r : float
        Residual water content.
    theta_s : float
        Saturated water content.
    alpha : float
        Sorptive number [1/length].

    Returns
    -------
    sympy.Piecewise
        Water content expression.
    """
    theta_r = sympy.sympify(theta_r)
    theta_s = sympy.sympify(theta_s)
    alpha = sympy.sympify(alpha)

    theta_unsat = theta_r + (theta_s - theta_r) * sympy.exp(alpha * psi)
    return sympy.Piecewise((theta_s, psi >= 0), (theta_unsat, True))


def gardner_C(psi, theta_r, theta_s, alpha, Ss=0.0):
    r"""Specific moisture capacity (Gardner exponential model).

    .. math::

        C(\psi) = \frac{d\theta}{d\psi} = \begin{cases}
            \alpha\,(\theta_s - \theta_r)\,\exp(\alpha\,\psi)
            & \psi < 0 \\
            S_s & \psi \ge 0
        \end{cases}

    Parameters
    ----------
    psi : sympy expression
        Pressure head.
    theta_r : float
        Residual water content.
    theta_s : float
        Saturated water content.
    alpha : float
        Sorptive number [1/length].
    Ss : float, optional
        Specific storage for the saturated zone (default 0).

    Returns
    -------
    sympy.Piecewise
        Specific moisture capacity expression.
    """
    theta_r = sympy.sympify(theta_r)
    theta_s = sympy.sympify(theta_s)
    alpha = sympy.sympify(alpha)
    Ss = sympy.sympify(Ss)

    C_unsat = alpha * (theta_s - theta_r) * sympy.exp(alpha * psi)
    return sympy.Piecewise((Ss, psi >= 0), (C_unsat, True))


def gardner_steady_state_psi(y, psi_0, psi_L, L, alpha):
    r"""Analytical steady-state pressure head for Gardner model with gravity.

    For a 1D vertical column of height *L* with the Gardner conductivity
    model, steady-state Richards equation with gravity reduces to

    .. math::

        K(\psi)\left(\frac{d\psi}{dy} + 1\right) = q = \text{const}

    The substitution :math:`u = \exp(\alpha\psi)` linearises the ODE.
    The exact solution with boundary conditions
    :math:`\psi(0)=\psi_0` (bottom) and :math:`\psi(L)=\psi_L` (top) is

    .. math::

        \psi(y) = \frac{1}{\alpha}\,\ln\!\Bigl[
            \bigl(u_0 - q^*\bigr)\,e^{-\alpha y} + q^*
        \Bigr]

    where :math:`u_0 = e^{\alpha\psi_0}`,
    :math:`u_L = e^{\alpha\psi_L}`, and

    .. math::

        q^* \equiv \frac{q}{K_s}
        = \frac{u_L - u_0\,e^{-\alpha L}}{1 - e^{-\alpha L}}

    Parameters
    ----------
    y : float or array
        Vertical coordinate (0 = bottom, *L* = top).
    psi_0 : float
        Pressure head at the bottom boundary.
    psi_L : float
        Pressure head at the top boundary.
    L : float
        Column height.
    alpha : float
        Gardner sorptive number [1/length].

    Returns
    -------
    float or array
        Exact pressure head profile :math:`\psi(y)`.

    Notes
    -----
    This is a *numpy* function (not sympy) intended for comparing
    numerical solutions against the analytical benchmark.
    """
    import numpy as np

    u_0 = np.exp(alpha * psi_0)
    u_L = np.exp(alpha * psi_L)

    # Normalised steady-state flux  q* = q / Ks
    q_star = (u_L - u_0 * np.exp(-alpha * L)) / (1.0 - np.exp(-alpha * L))

    return (1.0 / alpha) * np.log((u_0 - q_star) * np.exp(-alpha * y) + q_star)


def gardner_transient_psi(y, t, psi_dry, psi_wet, L, Ks, alpha, theta_r, theta_s):
    r"""Analytical transient wetting-front solution for Gardner model.

    Applies the **Ogata–Banks** (1961) solution to Richards equation
    with Gardner conductivity in a vertical column of height *L*.

    The substitution :math:`u = \exp(\alpha\psi)` transforms the
    nonlinear Richards equation into linear advection–diffusion:

    .. math::

        \frac{\partial u}{\partial t}
        = D\,\frac{\partial^2 u}{\partial z^2}
        + V\,\frac{\partial u}{\partial z}

    where :math:`z = L - y` (depth from the top),
    :math:`D = K_s / (\alpha\,\Delta\theta)`,
    :math:`V = K_s / \Delta\theta`, and
    :math:`\Delta\theta = \theta_s - \theta_r`.

    With a **step change** at the top (:math:`z = 0`) from dry to wet
    and a semi-infinite column approximation, the Ogata–Banks solution
    gives

    .. math::

        u(z, t) = u_{\rm dry}
        + (u_{\rm wet} - u_{\rm dry})\,H(z, t)

    where

    .. math::

        H(z, t) = \tfrac{1}{2}\,\operatorname{erfc}\!\left(
            \frac{z - Vt}{2\sqrt{Dt}}\right)
        + \tfrac{1}{2}\,\exp\!\left(\frac{Vz}{D}\right)\,
          \operatorname{erfc}\!\left(\frac{z + Vt}{2\sqrt{Dt}}\right)

    Finally, :math:`\psi(y, t) = \ln(u) / \alpha`.

    Parameters
    ----------
    y : float or array
        Vertical coordinate (0 = bottom, *L* = top).
    t : float
        Time since the wet boundary was applied (must be > 0).
    psi_dry : float
        Initial (dry) pressure head throughout the column.
    psi_wet : float
        Pressure head imposed at the top boundary.
    L : float
        Column height.
    Ks : float
        Saturated hydraulic conductivity.
    alpha : float
        Gardner sorptive number [1/length].
    theta_r : float
        Residual water content.
    theta_s : float
        Saturated water content.

    Returns
    -------
    float or array
        Pressure head profile :math:`\psi(y, t)`.

    Notes
    -----
    This is a *numpy* function (not sympy) intended for comparing
    numerical solutions against the analytical benchmark.

    The semi-infinite approximation is excellent when the wetting
    front has not yet reached the bottom boundary.

    References
    ----------
    Ogata, A. and Banks, R. B. (1961). A solution of the differential
    equation of longitudinal dispersion in porous media.
    *US Geological Survey Professional Paper* 411-A.
    """
    import numpy as np
    from scipy.special import erfc

    delta_theta = theta_s - theta_r
    D = Ks / (alpha * delta_theta)
    V = Ks / delta_theta

    u_dry = np.exp(alpha * psi_dry)
    u_wet = np.exp(alpha * psi_wet)

    # Depth from the top (z = 0 at top, z = L at bottom)
    z = L - np.asarray(y, dtype=float)

    sqrt_Dt = np.sqrt(D * t)

    # Ogata-Banks solution
    H = (
        0.5 * erfc((z - V * t) / (2.0 * sqrt_Dt))
        + 0.5 * np.exp(V * z / D) * erfc((z + V * t) / (2.0 * sqrt_Dt))
    )

    u = u_dry + (u_wet - u_dry) * H

    return (1.0 / alpha) * np.log(np.maximum(u, 1e-30))


# =====================================================================
# Haverkamp model
# =====================================================================


def haverkamp_theta(psi, theta_r, theta_s, alpha, beta):
    r"""Volumetric water content (Haverkamp model).

    .. math::

        \theta(\psi) = \begin{cases}
            \theta_r + \dfrac{\alpha\,(\theta_s - \theta_r)}
                              {\alpha + |\psi|^{\beta}}
            & \psi < 0 \\[6pt]
            \theta_s & \psi \ge 0
        \end{cases}

    Unlike Van Genuchten, the retention and conductivity curves have
    **independent** parameters, which gives extra flexibility when
    fitting laboratory data.

    Parameters
    ----------
    psi : sympy expression
        Pressure head (negative in unsaturated zone).
    theta_r : float
        Residual water content.
    theta_s : float
        Saturated water content.
    alpha : float
        Retention shape parameter (dimensionless or [length]^beta,
        depending on convention).
    beta : float
        Retention exponent.

    Returns
    -------
    sympy.Piecewise
        Water content expression.

    References
    ----------
    Haverkamp, R. et al. (1977). A comparison of numerical simulation
    models for one-dimensional infiltration.
    *Soil Science Society of America Journal*, 41(2), 285--294.
    """
    theta_r = sympy.sympify(theta_r)
    theta_s = sympy.sympify(theta_s)
    alpha = sympy.sympify(alpha)
    beta = sympy.sympify(beta)

    theta_unsat = theta_r + alpha * (theta_s - theta_r) / (alpha + (-psi) ** beta)
    return sympy.Piecewise((theta_s, psi >= 0), (theta_unsat, True))


def haverkamp_K(psi, Ks, A, B):
    r"""Hydraulic conductivity (Haverkamp model).

    .. math::

        K(\psi) = \begin{cases}
            K_s\,\dfrac{A}{A + |\psi|^B} & \psi < 0 \\[6pt]
            K_s & \psi \ge 0
        \end{cases}

    The conductivity parameters *A* and *B* are independent of the
    retention parameters *alpha* and *beta*.

    Parameters
    ----------
    psi : sympy expression
        Pressure head (negative in unsaturated zone).
    Ks : float
        Saturated hydraulic conductivity.
    A : float
        Conductivity shape parameter.
    B : float
        Conductivity exponent.

    Returns
    -------
    sympy.Piecewise
        Hydraulic conductivity expression.

    References
    ----------
    Haverkamp, R. et al. (1977). A comparison of numerical simulation
    models for one-dimensional infiltration.
    *Soil Science Society of America Journal*, 41(2), 285--294.
    """
    Ks = sympy.sympify(Ks)
    A = sympy.sympify(A)
    B = sympy.sympify(B)

    K_unsat = Ks * A / (A + (-psi) ** B)
    return sympy.Piecewise((Ks, psi >= 0), (K_unsat, True))


def haverkamp_C(psi, theta_r, theta_s, alpha, beta, Ss=0.0):
    r"""Specific moisture capacity (Haverkamp model).

    .. math::

        C(\psi) = \frac{d\theta}{d\psi} = \begin{cases}
            \dfrac{\alpha\,\beta\,(\theta_s - \theta_r)\,|\psi|^{\beta - 1}}
                  {\bigl(\alpha + |\psi|^{\beta}\bigr)^2}
            & \psi < 0 \\[6pt]
            S_s & \psi \ge 0
        \end{cases}

    Parameters
    ----------
    psi : sympy expression
        Pressure head.
    theta_r : float
        Residual water content.
    theta_s : float
        Saturated water content.
    alpha : float
        Retention shape parameter.
    beta : float
        Retention exponent.
    Ss : float, optional
        Specific storage for the saturated zone (default 0).

    Returns
    -------
    sympy.Piecewise
        Specific moisture capacity expression.

    References
    ----------
    Haverkamp, R. et al. (1977). A comparison of numerical simulation
    models for one-dimensional infiltration.
    *Soil Science Society of America Journal*, 41(2), 285--294.
    """
    theta_r = sympy.sympify(theta_r)
    theta_s = sympy.sympify(theta_s)
    alpha = sympy.sympify(alpha)
    beta = sympy.sympify(beta)
    Ss = sympy.sympify(Ss)

    abs_psi = -psi  # psi < 0, so |psi| = -psi
    C_unsat = (
        alpha * beta * (theta_s - theta_r) * abs_psi ** (beta - 1)
        / (alpha + abs_psi ** beta) ** 2
    )

    return sympy.Piecewise((Ss, psi >= 0), (C_unsat, True))
