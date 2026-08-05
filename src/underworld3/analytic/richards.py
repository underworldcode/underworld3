r"""Exact solutions for unsaturated flow — the Gardner Richards problems.

Richards is the one nonlinear scalar equation in the suite:

.. math::
    C(\psi)\,\frac{\partial\psi}{\partial t}
        = \nabla\cdot\!\left[K(\psi)\left(\nabla\psi + \hat y\right)\right]

with the conductivity depending on the unknown head :math:`\psi`. Gardner's
exponential model :math:`K = K_s e^{\alpha\psi}` is the case that admits closed
forms, because the substitution :math:`u = e^{\alpha\psi}` linearises it
*exactly* — not as an approximation. Under that substitution the equation becomes
linear advection–diffusion in :math:`u`, which is why the transient solution here
is an Ogata–Banks form and why it shares a residual check with
:class:`~underworld3.analytic.transport.AdvectedFront`.

These two solutions already existed in
:mod:`underworld3.utilities.retention_curves` as NumPy functions. Those functions
remain, unchanged in signature, and now delegate here; what they gain is the
residual check and a place in the registry.

See Also
--------
underworld3.analytic.transport : the linear scalar solutions.
"""

import sympy

from ._base import AnalyticSolution


def gardner_steady_saturation(y, psi_0, psi_L, L, alpha):
    r"""The linearised variable :math:`u = e^{\alpha\psi}` for steady flow.

    Given as :math:`u` rather than as the head because that is the form the
    formula is actually in — :math:`\psi = \ln u/\alpha` is the last step, and
    the NumPy wrapper in :mod:`underworld3.utilities.retention_curves` needs to
    interpose a floor before taking the log. Building the head here and undoing
    it there would be two spellings of one formula, which is how they drift.
    """

    alpha = sympy.sympify(alpha)
    u_0 = sympy.exp(alpha * sympy.sympify(psi_0))
    u_L = sympy.exp(alpha * sympy.sympify(psi_L))
    decay = sympy.exp(-alpha * sympy.sympify(L))

    # Normalised steady flux q/Ks, fixed by the two boundary heads.
    flux = (u_L - u_0 * decay) / (1 - decay)

    return (u_0 - flux) * sympy.exp(-alpha * y) + flux, flux


def gardner_transient_saturation(
    y, t, psi_dry, psi_wet, L, Ks, alpha, theta_r, theta_s
):
    r"""The linearised variable :math:`u = e^{\alpha\psi}` for a wetting front.

    The Ogata–Banks solution of :math:`u_t = Du_{zz} - Vu_z` with :math:`z` the
    depth below the surface. See :class:`GardnerTransient`.
    """

    alpha = sympy.sympify(alpha)
    Ks = sympy.sympify(Ks)
    spread = sympy.sympify(theta_s) - sympy.sympify(theta_r)

    D = Ks / (alpha * spread)
    V = Ks / spread
    depth = sympy.sympify(L) - y

    u_dry = sympy.exp(alpha * sympy.sympify(psi_dry))
    u_wet = sympy.exp(alpha * sympy.sympify(psi_wet))

    scale = 2 * sympy.sqrt(D * t)
    front = (
        sympy.erfc((depth - V * t) / scale)
        + sympy.exp(V * depth / D) * sympy.erfc((depth + V * t) / scale)
    ) / 2

    return u_dry + (u_wet - u_dry) * front, D, V


class _Gardner(AnalyticSolution):
    """Shared Gardner parameters and boundary conditions."""

    solves = "richards"
    dim = 2
    reference = (
        "Gardner (1958), Soil Sci. 85, 228-232; transient form after "
        "Ogata & Banks (1961), USGS Prof. Paper 411-A. Previously NumPy "
        "functions in underworld3/utilities/retention_curves.py."
    )

    def apply_boundary_conditions(self, solver):
        """Prescribe the exact head on every wall."""

        for boundary in self.boundaries:
            solver.add_dirichlet_bc([self.fn_solution], boundary)

    def _gardner_material(self, head):
        r"""Conductivity and capacity for a given head expression.

        :math:`K = K_s e^{\alpha\psi}` and
        :math:`C = \mathrm d\theta/\mathrm d\psi = \alpha\,\Delta\theta\,e^{\alpha\psi}`
        — the unsaturated branches. The saturated branches in
        :mod:`~underworld3.utilities.retention_curves` do not apply because
        these solutions are posed with :math:`\psi < 0` throughout.
        """

        saturation = sympy.exp(self.alpha * head)
        return (
            self.Ks * saturation,
            self.alpha * (self.theta_s - self.theta_r) * saturation,
        )


class GardnerSteady(_Gardner):
    r"""Steady infiltration through a vertical column.

    With gravity, steady Richards reduces to a constant flux,

    .. math::
        K(\psi)\left(\frac{\mathrm d\psi}{\mathrm dy} + 1\right) = q,

    and the Gardner substitution :math:`u = e^{\alpha\psi}` linearises it to give

    .. math::
        \psi(y) = \frac1\alpha\ln\!\left[(u_0 - q^*)e^{-\alpha y} + q^*\right],
        \qquad q^* = \frac{u_L - u_0 e^{-\alpha L}}{1 - e^{-\alpha L}}.

    The flux being *constant* is the whole content of the solution, and it is a
    sharper test of a Richards solver than the head profile: the head can look
    right while the conductivity is being evaluated at the wrong place, and the
    flux then drifts down the column.

    Parameters
    ----------
    mesh : Mesh
        A 2D mesh; the column is the vertical direction.
    psi_0, psi_L : float
        Head at the bottom and the top. Both should be negative
        (unsaturated); the Gardner model is only valid there.
    L : float
        Column height.
    alpha : float
        Gardner sorptive number, :math:`1/\mathrm{length}`.
    Ks : float
        Saturated conductivity.
    theta_r, theta_s : float
        Residual and saturated water content.
    """

    eqn_solution = (
        r"\frac1\alpha\ln\left[(u_0 - q^*)e^{-\alpha y} + q^*\right]"
    )

    def __init__(
        self,
        mesh,
        psi_0=-1.0,
        psi_L=-5.0,
        L=1.0,
        alpha=1.0,
        Ks=1.0,
        theta_r=0.05,
        theta_s=0.45,
    ):
        super().__init__(mesh)

        if float(alpha) <= 0.0 or float(Ks) <= 0.0:
            raise ValueError("alpha and Ks must be positive.")
        if not float(theta_s) > float(theta_r):
            raise ValueError("theta_s must exceed theta_r.")

        self.alpha = sympy.Rational(str(alpha))
        self.Ks = sympy.Rational(str(Ks))
        self.theta_r = sympy.Rational(str(theta_r))
        self.theta_s = sympy.Rational(str(theta_s))
        self.L = sympy.Rational(str(L))
        self.psi_0 = float(psi_0)
        self.psi_L = float(psi_L)

        y = mesh.X[mesh.dim - 1]

        saturation, self.flux = gardner_steady_saturation(
            y,
            sympy.Rational(str(psi_0)),
            sympy.Rational(str(psi_L)),
            self.L,
            self.alpha,
        )
        head = sympy.log(saturation) / self.alpha

        self.set_richards_field(head, *self._gardner_material(head))


class GardnerTransient(_Gardner):
    r"""A wetting front advancing down a dry column.

    The Gardner substitution turns Richards into linear advection–diffusion in
    :math:`u = e^{\alpha\psi}`, with

    .. math::
        D = \frac{K_s}{\alpha\,\Delta\theta}, \qquad V = \frac{K_s}{\Delta\theta},

    and the Ogata–Banks solution for a step applied at the surface gives

    .. math::
        u = u_{\rm dry} + (u_{\rm wet} - u_{\rm dry})\left[
            \tfrac12\mathrm{erfc}\frac{z - Vt}{2\sqrt{Dt}}
            + \tfrac12 e^{Vz/D}\,\mathrm{erfc}\frac{z + Vt}{2\sqrt{Dt}}\right]

    with :math:`z = L - y` the depth below the surface. Then
    :math:`\psi = \ln u / \alpha`.

    Time is the symbol :attr:`t`, so a solver is checked at whatever time it
    reached::

        exact = sol.fn_solution.subs(sol.t, t_end)

    The solution is semi-infinite, so it satisfies the equation exactly
    everywhere but only satisfies the *bottom* boundary condition while the front
    has yet to arrive. Compare before then — :attr:`front_depth` says where the
    front is.

    Parameters
    ----------
    mesh : Mesh
        A 2D mesh; the column is the vertical direction.
    psi_dry : float
        Initial head throughout the column.
    psi_wet : float
        Head imposed at the surface. Wetter means less negative.
    L, alpha, Ks, theta_r, theta_s
        As for :class:`GardnerSteady`.
    """

    eqn_solution = (
        r"\frac1\alpha\ln\left[u_{\rm dry} "
        r"+ (u_{\rm wet} - u_{\rm dry})H(z,t)\right]"
    )
    singular_at_origin = True

    def __init__(
        self,
        mesh,
        psi_dry=-5.0,
        psi_wet=-0.5,
        L=1.0,
        alpha=1.0,
        Ks=1.0,
        theta_r=0.05,
        theta_s=0.45,
    ):
        super().__init__(mesh)

        if float(alpha) <= 0.0 or float(Ks) <= 0.0:
            raise ValueError("alpha and Ks must be positive.")
        if not float(theta_s) > float(theta_r):
            raise ValueError("theta_s must exceed theta_r.")
        if not float(psi_wet) > float(psi_dry):
            raise ValueError("psi_wet must be wetter (less negative) than psi_dry.")

        self.alpha = sympy.Rational(str(alpha))
        self.Ks = sympy.Rational(str(Ks))
        self.theta_r = sympy.Rational(str(theta_r))
        self.theta_s = sympy.Rational(str(theta_s))
        self.L = sympy.Rational(str(L))
        self.psi_dry = float(psi_dry)
        self.psi_wet = float(psi_wet)
        self.t = sympy.Symbol("t", positive=True)

        y = mesh.X[mesh.dim - 1]

        saturation, D, V = gardner_transient_saturation(
            y,
            self.t,
            sympy.Rational(str(psi_dry)),
            sympy.Rational(str(psi_wet)),
            self.L,
            self.Ks,
            self.alpha,
            self.theta_r,
            self.theta_s,
        )
        self.diffusivity = float(D)
        self.speed = float(V)

        head = sympy.log(saturation) / self.alpha

        self.set_richards_field(head, *self._gardner_material(head))

    def front_depth(self, time):
        """Depth of the wetting front below the surface at *time*.

        The front advects at :math:`V = K_s/\\Delta\\theta`; compare against the
        column height to know whether the semi-infinite form still holds.
        """

        return self.speed * float(time)
