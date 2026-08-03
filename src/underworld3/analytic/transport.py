r"""Exact solutions for the scalar transport problems.

Diffusion, Darcy flow, Richards flow and Poisson: one unknown field satisfying
:math:`\nabla\cdot(k\nabla u) + f = 0`, rather than the velocity-and-pressure
pair the Velic family solves for.

These were already in the repository, written inline in the tests that used them
or as helper functions beside the constitutive models. Collected here they gain
what the Stokes solutions have: a residual check that consults nothing external,
a place to be found, and a name to cite.

See Also
--------
underworld3.analytic.velic : the Stokes family.
"""

import sympy

from ._base import AnalyticSolution


class _Transport(AnalyticSolution):
    """Shared metadata and boundary conditions for the scalar solutions.

    These prescribe the field itself on the boundary, so they cannot reuse the
    Stokes mixins — `FixedWalls` applies a velocity, which a scalar solution does
    not have.
    """

    solves = "transport"
    dim = 2

    def apply_boundary_conditions(self, solver):
        """Prescribe the exact field on every wall."""

        for boundary in self.boundaries:
            solver.add_dirichlet_bc([self.fn_solution], boundary)


class Poisson1D(_Transport):
    r"""One-dimensional Poisson, with a source of your choosing.

    Solves :math:`\nabla^2 u + f = 0` on the unit box with :math:`u` varying in
    :math:`z` only, for three sources that between them cover the cases worth
    separating:

    ==============  ==========================  ==============================
    ``source``      :math:`f`                   :math:`u`
    ==============  ==========================  ==============================
    ``"none"``      :math:`0`                   :math:`1 - z`
    ``"constant"``  :math:`2`                   :math:`z(1-z)`
    ``"sinusoid"``  :math:`\pi^2\sin(\pi z)`    :math:`\sin(\pi z)`
    ==============  ==========================  ==============================

    The simplest thing in the suite, and the first thing to run when a scalar
    solver is suspect: a linear profile is exact in any sensible discretisation,
    a quadratic is exact from second order up, and only the sinusoid actually
    tests convergence. A solver that fails the first two has a problem that has
    nothing to do with accuracy.

    Parameters
    ----------
    mesh : Mesh
        A 2D mesh on the unit box.
    source : {"none", "constant", "sinusoid"}
    """

    reference = "Standard; previously inline in tests/test_1000_poissonCart.py."

    def __init__(self, mesh, source="sinusoid"):
        super().__init__(mesh)

        x, z = mesh.X
        exact = {
            "none": (1 - z, sympy.Integer(0)),
            "constant": (z * (1 - z), sympy.Integer(2)),
            "sinusoid": (sympy.sin(sympy.pi * z), sympy.pi**2 * sympy.sin(sympy.pi * z)),
        }
        if source not in exact:
            raise ValueError(
                f"source must be one of {sorted(exact)}; got {source!r}"
            )

        self.source = source
        solution, forcing = exact[source]

        self.eqn_solution = {"none": r"1 - z", "constant": r"z(1-z)",
                             "sinusoid": r"\sin(\pi z)"}[source]
        self.set_scalar_field(solution=solution, coefficient=1, source=forcing)


class ErfcDiffusion(_Transport):
    r"""The error-function diffusion profile.

    :math:`h(z, t) = \mathrm{erfc}\!\left(z / 2\sqrt{Dt}\right)`, the response of
    a semi-infinite column to a step held at its boundary.

    Time appears as a symbol, :attr:`t`, so a transient solver can be checked at
    whatever time it reached::

        exact = sol.fn_solution.subs(sol.t, t_end)

    Starting a comparison at :math:`t > 0` rather than at the step itself is the
    point of using this: the initial condition is then smooth and representable
    on the mesh, where a sharp step is not.

    Parameters
    ----------
    mesh : Mesh
        A 2D mesh; the profile varies with :math:`z`.
    diffusivity : float
        :math:`D`. For a Darcy problem this is conductivity over storativity.
    """

    reference = (
        "Standard; previously inline in "
        "tests/test_1005_TransientDarcyCartesian.py."
    )
    eqn_solution = r"\mathrm{erfc}\left(z / 2\sqrt{Dt}\right)"

    def __init__(self, mesh, diffusivity=1.0):
        super().__init__(mesh)

        if float(diffusivity) <= 0.0:
            raise ValueError("diffusivity must be positive.")

        self.diffusivity = float(diffusivity)
        self.t = sympy.Symbol("t", positive=True)

        x, z = mesh.X
        profile = sympy.erfc(z / (2 * sympy.sqrt(self.diffusivity * self.t)))

        # Diffusion has no source; the field is driven entirely by its boundary,
        # so the steady residual is not the right check here — the transient one
        # is du/dt = D lap(u), which `diffusion_residual` verifies.
        self.set_scalar_field(profile, coefficient=self.diffusivity, source=0)


class AdvectedFront(_Transport):
    r"""An advecting, diffusing top hat.

    Two error functions, the exact response of a rectangular pulse between
    :math:`x_0` and :math:`x_1` carried at speed :math:`u` while diffusing with
    :math:`\kappa`. Time is the symbol :attr:`t`, as for :class:`ErfcDiffusion`.

    This is the solution `tests/test_1100_AdvDiffCartesian.py` needs. Its own
    note says the test is fragile because a step initial condition is not
    representable on the mesh, and that the fix is to start from a smooth profile
    at :math:`t > 0` — which is exactly what evaluating this at a positive time
    gives.

    Parameters
    ----------
    mesh : Mesh
        A 2D mesh; the front travels in :math:`x`.
    kappa : float
        Diffusivity.
    speed : float
        Advection speed.
    x0, x1 : float
        Edges of the initial pulse.
    """

    reference = (
        "Ogata & Banks (1961) form; previously inline in "
        "tests/test_1100_AdvDiffCartesian.py."
    )
    eqn_solution = (
        r"\tfrac12\left[\mathrm{erf}\frac{x_1 - x + ut}{2\sqrt{\kappa t}}"
        r" + \mathrm{erf}\frac{x - x_0 - ut}{2\sqrt{\kappa t}}\right]"
    )

    def __init__(self, mesh, kappa=1.0e-3, speed=1.0, x0=0.1, x1=0.3):
        super().__init__(mesh)

        if float(kappa) <= 0.0:
            raise ValueError("kappa must be positive.")

        self.kappa = float(kappa)
        self.speed = float(speed)
        self.x0 = float(x0)
        self.x1 = float(x1)
        self.t = sympy.Symbol("t", positive=True)

        x, z = mesh.X
        spread = 2 * sympy.sqrt(self.kappa * self.t)
        drift = self.speed * self.t
        profile = (
            sympy.erf((self.x1 - x + drift) / spread)
            + sympy.erf((x - self.x0 - drift) / spread)
        ) / 2

        self.set_scalar_field(
            profile, coefficient=self.kappa, source=0, advection=(self.speed, 0)
        )


class TwoLayerDarcy(_Transport):
    r"""Steady Darcy flow through two layers of different permeability.

    A piecewise-linear pressure with a kink at the interface, set by continuity
    of flux across it: the steeper gradient is in the less permeable layer.

    The simplest problem with a *coefficient* discontinuity rather than a source
    one, which makes it the scalar counterpart of SolCx — and the fastest way to
    tell whether a Darcy solver handles a permeability contrast at all.

    With gravity the flux is :math:`q = -k(\partial_z p + S)`; setting
    :math:`S = 0` recovers the pressure-driven case. Either way :math:`q` is
    constant down the column, and that is what fixes the interface pressure:

    .. math::
        q = \frac{\Delta p - S(z_1 - z_0)}{d_{\rm low}/k_{\rm low}
                                         + d_{\rm up}/k_{\rm up}}

    with :math:`d` the layer thicknesses. The profile is then linear in each
    layer with slope :math:`-q/k - S`.

    Parameters
    ----------
    mesh : Mesh
        A 2D mesh; the layering is in the vertical coordinate.
    k_lower, k_upper : float
        Permeability below and above the interface.
    interface : float
        Height of the layer boundary. Must lie strictly inside the column.
    pressure_drop : float
        Pressure at the bottom, taking the top as zero.
    gravity : float
        :math:`S` in the flux law. Zero for pressure-driven flow.
    bottom, top : float
        Extent of the column. Defaults to the unit interval; the Darcy tests
        use :math:`(-1, 0)`.
    """

    reference = (
        "Standard; previously inline in tests/test_1004_DarcyCartesian.py."
    )
    eqn_solution = r"\text{piecewise linear, kinked at the interface}"

    def __init__(
        self,
        mesh,
        k_lower=1.0,
        k_upper=0.1,
        interface=0.5,
        pressure_drop=1.0,
        gravity=0.0,
        bottom=0.0,
        top=1.0,
    ):
        super().__init__(mesh)

        if not (float(k_lower) > 0.0 and float(k_upper) > 0.0):
            raise ValueError("permeabilities must be positive.")
        if not float(bottom) < float(interface) < float(top):
            raise ValueError(
                f"interface must lie strictly inside ({bottom}, {top}); "
                f"got {interface}"
            )

        self.k_lower = float(k_lower)
        self.k_upper = float(k_upper)
        self.interface = float(interface)
        self.pressure_drop = float(pressure_drop)
        self.gravity = float(gravity)
        self.bottom = float(bottom)
        self.top = float(top)

        z = mesh.X[mesh.dim - 1]

        k_low = sympy.Rational(str(k_lower))
        k_up = sympy.Rational(str(k_upper))
        z_int = sympy.Rational(str(interface))
        z_bot = sympy.Rational(str(bottom))
        z_top = sympy.Rational(str(top))
        drop = sympy.Rational(str(pressure_drop))
        S = sympy.Rational(str(gravity))

        thick_low = z_int - z_bot
        thick_up = z_top - z_int

        # Constant flux, from p(top) = 0 and p(bottom) = drop. Derived here
        # rather than transcribed, so that agreeing with the form the Darcy test
        # carried is a check on both rather than a copy of one.
        self.flux = (drop - S * (z_top - z_bot)) / (
            thick_low / k_low + thick_up / k_up
        )
        self.at_interface = thick_up * (self.flux / k_up + S)

        slope_low = -self.flux / k_low - S
        slope_up = -self.flux / k_up - S

        self.set_scalar_field(
            sympy.Piecewise(
                (self.at_interface + slope_low * (z - z_int), z < z_int),
                (self.at_interface + slope_up * (z - z_int), True),
            ),
            # Piecewise-constant k makes the gravity term vanish inside each
            # layer, so the source is zero away from the interface — where flux
            # continuity is imposed by construction rather than by a source.
            coefficient=sympy.Piecewise((k_low, z < z_int), (k_up, True)),
            source=0,
        )
