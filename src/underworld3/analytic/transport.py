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

from ._base import AnalyticSolution, prescribed_scalar


class _Transport(AnalyticSolution):
    """Shared metadata and boundary conditions for the scalar solutions.

    The whole family is posed with its own exact field on every wall, which is
    why the condition is stated once here. A member that needs something else on
    one boundary — a no-flux base, an inflow face — writes its own
    ``apply_boundary_conditions`` and composes the helpers per boundary.
    """

    solves = "transport"
    dim = 2

    def apply_boundary_conditions(self, solver):
        """Prescribe the exact field on every wall.

        There is no pressure nullspace to remove: the field is pinned by its own
        Dirichlet data everywhere on the boundary.
        """

        prescribed_scalar(solver, self.boundaries, self.fn_solution)


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
    whatever time it reached.

    .. warning::
       **Singular at** :math:`t = 0`, and so is every diffusive similarity
       solution. At the origin the profile is a step with unbounded gradient —
       a state no finite element space can hold. Do not start a benchmark
       there, and do not quote an error at one time: a transient comparison is
       a *pair* of times, and the error depends on the start as much as the
       finish.

       ``sol.at(t)`` refuses :math:`t \le 0` outright, and
       ``sol.earliest_resolvable_time(h)`` gives the resolution-dependent floor
       below which you are measuring interpolation error rather than the
       solver::

           t0 = sol.earliest_resolvable_time(mesh.get_min_radius())
           field.array = uw.function.evaluate(sol.at(t0), field.coords)
           ...                                  # step to t1
           error = sol.error("solution", field) # against sol.at(t1)

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
    singular_at_origin = True

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
    note said the test was fragile because a step initial condition is not
    representable on the mesh — which is the singularity below, and the reason
    that test has to name a start time as well as an end time.

    .. warning::
       **Singular at** :math:`t = 0`, and so is every diffusive similarity
       solution. At the origin the profile is a step with unbounded gradient —
       a state no finite element space can hold. Do not start a benchmark
       there, and do not quote an error at one time: a transient comparison is
       a *pair* of times, and the error depends on the start as much as the
       finish.

       ``sol.at(t)`` refuses :math:`t \le 0` outright, and
       ``sol.earliest_resolvable_time(h)`` gives the resolution-dependent floor
       below which you are measuring interpolation error rather than the
       solver::

           t0 = sol.earliest_resolvable_time(mesh.get_min_radius())
           field.array = uw.function.evaluate(sol.at(t0), field.coords)
           ...                                  # step to t1
           error = sol.error("solution", field) # against sol.at(t1)

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
    singular_at_origin = True

    def __init__(self, mesh, kappa=1.0e-3, speed=1.0, x0=0.1, x1=0.3):
        super().__init__(mesh)

        if float(kappa) <= 0.0:
            raise ValueError("kappa must be positive.")

        self.kappa = float(kappa)
        self.diffusivity = float(kappa)
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


class RotatingGaussian(_Transport):
    r"""A Gaussian carried round the origin by rigid rotation while it diffuses.

    The velocity :math:`\mathbf{u} = \omega(-y, x)` is solenoidal and rigid, so
    it commutes with the Laplacian: the exact field is the free-space diffusing
    Gaussian with its centre following the rotation,

    .. math::
        \phi(\mathbf{x}, t) = \frac{\sigma^2}{\sigma^2 + 2\kappa t}
            \exp\!\left(-\frac{|\mathbf{x} - \mathbf{c}(t)|^2}
                               {2(\sigma^2 + 2\kappa t)}\right),
        \qquad
        \mathbf{c}(t) = R\,(\cos(\omega t + \varphi_0),\ \sin(\omega t + \varphi_0)).

    The transport test with a known answer at every time: after one
    revolution, :math:`t = 2\pi/\omega`, a pure-advection field must return
    to its initial state, so the round-trip error is an absolute measure and
    the quarter-turn errors give the growth in between. With
    :math:`\kappa = 0` the solution is regular at :math:`t = 0` and a
    benchmark may start there.

    The domain is whatever mesh is supplied; the solution is exact on the
    plane, so the walls should sit where the field is negligible (a few
    :math:`\sigma` from the orbit) and carry :math:`\phi = 0`.

    Parameters
    ----------
    mesh : Mesh
        A 2D mesh containing the orbit.
    sigma : float
        Standard deviation of the initial Gaussian.
    centre_radius : float
        Orbit radius :math:`R`.
    omega : float
        Angular velocity; the period is :math:`2\pi/\omega`.
    diffusivity : float
        :math:`\kappa \ge 0`; zero is pure advection.
    phase : float
        Initial angular position :math:`\varphi_0` of the centre.
    """

    reference = (
        "Rigid rotation of a diffusing Gaussian; classical (e.g. the rotating "
        "cone/Gaussian tests of Zalesak 1979 and LeVeque 1996, here in closed form)."
    )
    eqn_solution = (
        r"\frac{\sigma^2}{\sigma^2 + 2\kappa t}"
        r"\exp\left(-\frac{|\mathbf{x}-\mathbf{c}(t)|^2}{2(\sigma^2+2\kappa t)}\right)"
    )
    singular_at_origin = False

    def __init__(self, mesh, sigma=0.12, centre_radius=0.5, omega=1.0,
                 diffusivity=0.0, phase=0.0):
        super().__init__(mesh)

        if float(sigma) <= 0.0:
            raise ValueError("sigma must be positive.")
        if float(diffusivity) < 0.0:
            raise ValueError("diffusivity must not be negative.")

        self.sigma = float(sigma)
        self.centre_radius = float(centre_radius)
        self.omega = float(omega)
        self.diffusivity = float(diffusivity)
        self.kappa = float(diffusivity)
        self.phase = float(phase)
        self.t = sympy.Symbol("t", positive=True)

        x, y = mesh.X
        angle = self.omega * self.t + self.phase
        cx = self.centre_radius * sympy.cos(angle)
        cy = self.centre_radius * sympy.sin(angle)
        variance = self.sigma ** 2 + 2 * self.kappa * self.t
        profile = (self.sigma ** 2 / variance) * sympy.exp(
            -((x - cx) ** 2 + (y - cy) ** 2) / (2 * variance))

        self.set_scalar_field(
            profile, coefficient=self.kappa, source=0,
            advection=(-self.omega * y, self.omega * x))

    @property
    def period(self):
        r"""Time of one revolution, :math:`2\pi/\omega`."""
        return 2.0 * sympy.pi.evalf() / self.omega


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
