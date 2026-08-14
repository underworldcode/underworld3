r"""Barr & Houseman (1996) analytic solution for a fault embedded in a viscous
medium — the linear (:math:`n = 1`) plane-strain case.

Reference
---------
T. D. Barr & G. A. Houseman, *Deformation fields around a fault embedded in a
non-linear ductile medium*, Geophys. J. Int. **125**, 473-490 (1996);
Appendix, equations (A1)-(A9). The companion letter, Barr & Houseman,
Geophys. Res. Lett. **19**, 1145-1148 (1992), gives the near-tip asymptotics.

Implementation follows that of @gthyagi, who has been using this solution for
fault benchmarking.

Why this solution is unusual, and useful
----------------------------------------
The deformation around a terminating fault is written in polar coordinates
with the **origin at the fault tip** and the fault along :math:`\theta = 0`.
The stream function separates into a Fourier series in :math:`m = q/2`, and the
two halves of that series mean different things:

* **whole-integer** :math:`m` — continuous deformation, no fault;
* **half-integer** :math:`m` — the fault-type discontinuity.

Boundedness of the velocity at :math:`r = 0` admits only one negative index,
:math:`m = -1/2`. That single mode carries the whole fault singularity, and it
is why the slip goes as :math:`\sqrt{r}` and the stress as
:math:`1/\sqrt{r}` — the exponents are a property of the fault's own Fourier
mode, not an assumption.

The solution below is their plane-strain test problem: prescribe the velocity
(:meth:`boundary_velocity`) on the perimeter of a disc of radius :math:`R_0`,
impose the fault conditions on :math:`\theta = 0`, and the interior field is
exact.

Only plane strain is implemented. The paper also gives a thin-viscous-sheet
(plane-stress) solution, equations (A10)-(A13), in which the in-plane
divergence is NOT zero — a different equation set from Underworld's
incompressible Stokes, so it is not a benchmark for this solver.

Conventions
-----------
Their constitutive relation is :math:`\tau_{ij} = B \dot E^{(1/n - 1)}
\dot\varepsilon_{ij}` with :math:`B = 2\eta_0` at :math:`n = 1`, so
:math:`\tau = 2 \eta \dot\varepsilon` as usual.

Their pressure takes **extension as positive**, so their force balance is
:math:`\partial_j \tau_{ij} + \partial_i p = 0` — a sign opposite to the more
common convention. :attr:`pressure` follows the paper. Negate it to compare
against a solver whose pressure is compression-positive.
"""
import numpy as np
import sympy


class BarrHouseman:
    r"""The linear plane-strain fault-tip solution on a disc of radius ``R0``.

    The fault occupies :math:`\theta = 0` from the tip at the origin to the
    perimeter. Slip is the jump in the fault-parallel velocity across it, and
    is :math:`2 U_0 \sqrt{r/R_0}` — so the relative slip velocity at the
    perimeter is :math:`2 U_0`, the paper's normalisation.

    Parameters
    ----------
    U0 : float
        Velocity scale. The slip at the perimeter is ``2 * U0``.
    R0 : float
        Radius of the disc, and the fault's length.
    eta : float
        Newtonian viscosity of the medium.

    Examples
    --------
    The solution and the boundary datum that produces it:

    >>> sol = BarrHouseman(U0=1.0, R0=1.0, eta=1.0)
    >>> float(sol.slip(1.0))
    2.0

    Notes
    -----
    The field is multivalued around the tip — that is what a fault is — so it
    is a function of :math:`(r, \theta)` with :math:`\theta \in [0, 2\pi)`,
    NOT of the Cartesian coordinates alone. The branch cut lies **on the
    fault**. :meth:`evaluate` places it there by taking
    ``arctan2(y, x) mod 2*pi``; a Cartesian expression using a bare
    ``atan2`` would put the cut on the negative :math:`x` axis instead and
    silently return the wrong side of the fault.
    """

    def __init__(self, U0=1.0, R0=1.0, eta=1.0):
        # The parameters may be SymPy symbols. That is not a convenience: a
        # symbolic check with U0 = R0 = eta = 1 would be satisfied by a
        # transcription carrying the wrong power of R0 in the pressure, so the
        # verification is strictly stronger with them left free.
        for name, value in (("R0", R0), ("eta", eta)):
            if not isinstance(value, sympy.Basic) and not float(value) > 0.0:
                raise ValueError(f"{name} must be positive.")
        self.U0 = U0 if isinstance(U0, sympy.Basic) else float(U0)
        self.R0 = R0 if isinstance(R0, sympy.Basic) else float(R0)
        self.eta = eta if isinstance(eta, sympy.Basic) else float(eta)
        self._symbols = None

    @property
    def _is_symbolic(self):
        return any(isinstance(v, sympy.Basic)
                   for v in (self.U0, self.R0, self.eta))

    # ------------------------------------------------------------------ sympy
    @property
    def symbols(self):
        """The polar symbols ``(r, theta)`` the expressions are written in.

        ``r`` is positive — the solution is singular at ``r = 0`` and never
        evaluated there — but ``theta`` is only REAL. It runs over
        :math:`[0, 2\pi)` and the fault conditions are checked at
        :math:`\theta = 0`, which a ``positive=True`` assumption excludes;
        SymPy would then be entitled to simplify a substitution that the
        assumption says cannot happen.

        Cached on the instance, so repeated access returns the same objects
        rather than relying on SymPy's global symbol cache for identity.
        """
        if self._symbols is None:
            self._symbols = (sympy.Symbol("r", positive=True),
                             sympy.Symbol("theta", real=True))
        return self._symbols

    def _polar(self):
        r, t = self.symbols
        R = r / self.R0
        U0, eta, R0 = self.U0, self.eta, self.R0

        # Whole-integer (continuous) modes, then the half-integer (fault) mode.
        # The half-integer group is the entire singular content: sqrt(R) in the
        # velocity, 1/sqrt(R) in the pressure.
        u_r = (U0 / 4) * (
            R**2 * (sympy.sin(t) - sympy.sin(3 * t))
            - R**3 * (2 * sympy.sin(2 * t) - 2 * sympy.sin(4 * t))
            + sympy.sqrt(R) * (sympy.cos(t / 2) + 3 * sympy.cos(3 * t / 2))
        )
        u_t = (U0 / 4) * (
            R**2 * (3 * sympy.cos(t) - sympy.cos(3 * t))
            - R**3 * (4 * sympy.cos(2 * t) - 2 * sympy.cos(4 * t))
            - sympy.sqrt(R) * (3 * sympy.sin(t / 2) + 3 * sympy.sin(3 * t / 2))
        )
        p = (eta * U0 / R0) * (
            -2 * R * sympy.sin(t)
            + 3 * R**2 * sympy.sin(2 * t)
            + sympy.cos(t / 2) / sympy.sqrt(R)
        )
        return u_r, u_t, p

    @property
    def velocity_polar(self):
        """``(u_r, u_theta)`` as SymPy expressions in ``r`` and ``theta``."""
        u_r, u_t, _p = self._polar()
        return u_r, u_t

    @property
    def pressure_polar(self):
        """Pressure as a SymPy expression, EXTENSION POSITIVE (see module doc)."""
        return self._polar()[2]

    def boundary_velocity(self):
        r"""The velocity datum on :math:`r = R_0` that produces the solution.

        Their equations (A8a, A8b). Returned as ``(U_r, U_theta)`` SymPy
        expressions in ``theta``; this is what a solver's Dirichlet condition
        on the disc perimeter must impose.
        """
        _r, t = self.symbols
        u_r, u_t = self.velocity_polar
        return (sympy.simplify(u_r.subs(_r, self.R0)),
                sympy.simplify(u_t.subs(_r, self.R0)))

    # ------------------------------------------------------------------ numpy
    def evaluate(self, coords):
        r"""Velocity and pressure at Cartesian ``coords`` measured FROM THE TIP.

        Parameters
        ----------
        coords : array_like, shape (N, 2)
            Points relative to the fault tip, with the fault along ``+x``.

        Returns
        -------
        velocity : ndarray, shape (N, 2)
            Cartesian components.
        pressure : ndarray, shape (N,)

        Notes
        -----
        ``theta`` is taken as ``arctan2(y, x) mod 2*pi`` so the branch cut sits
        ON the fault, which is where the field is genuinely discontinuous. A
        point exactly on the fault returns the ``theta = 0`` side; approach
        from ``y < 0`` to obtain the other.
        """
        velocity = self.evaluate_velocity(coords)
        return velocity, self.evaluate_pressure(coords)

    def evaluate_velocity(self, coords):
        r"""Velocity at Cartesian ``coords`` measured from the tip.

        Defined AT the tip: every term of the velocity carries a positive
        power of :math:`r`, so the limit is zero and is returned. It is the
        pressure that is singular there, not the velocity — see
        :meth:`evaluate_pressure`.
        """
        r, t, R = self._polar_of(coords)
        u_r = (self.U0 / 4) * (
            R**2 * (np.sin(t) - np.sin(3 * t))
            - R**3 * (2 * np.sin(2 * t) - 2 * np.sin(4 * t))
            + np.sqrt(R) * (np.cos(t / 2) + 3 * np.cos(3 * t / 2))
        )
        u_t = (self.U0 / 4) * (
            R**2 * (3 * np.cos(t) - np.cos(3 * t))
            - R**3 * (4 * np.cos(2 * t) - 2 * np.cos(4 * t))
            - np.sqrt(R) * (3 * np.sin(t / 2) + 3 * np.sin(3 * t / 2))
        )
        return np.column_stack([u_r * np.cos(t) - u_t * np.sin(t),
                                u_r * np.sin(t) + u_t * np.cos(t)])

    def evaluate_pressure(self, coords):
        r"""Pressure at Cartesian ``coords``; refuses the tip.

        The pressure carries the :math:`r^{-1/2}` term of the
        :math:`m = -1/2` mode and genuinely diverges at :math:`r = 0`.
        """
        r, t, R = self._polar_of(coords)
        if np.any(r == 0.0):
            raise ValueError(
                "the pressure is singular at the fault tip; exclude r = 0 "
                "(the velocity is defined there — use evaluate_velocity)")
        return (self.eta * self.U0 / self.R0) * (
            -2 * R * np.sin(t) + 3 * R**2 * np.sin(2 * t)
            + np.cos(t / 2) / np.sqrt(R)
        )

    def _polar_of(self, coords):
        """(r, theta, r/R0) from Cartesian coordinates, cut ON the fault."""
        if self._is_symbolic:
            raise ValueError(
                "this solution was built with symbolic parameters; give U0, "
                "R0 and eta numeric values to evaluate it")
        X = np.asarray(coords, dtype=float)
        if X.ndim != 2 or X.shape[1] != 2:
            raise ValueError("coords must have shape (N, 2)")
        r = np.hypot(X[:, 0], X[:, 1])
        t = np.mod(np.arctan2(X[:, 1], X[:, 0]), 2.0 * np.pi)
        return r, t, r / self.R0

    def slip(self, r):
        r"""Fault slip :math:`2 U_0 \sqrt{r/R_0}` at radius ``r`` from the tip.

        The jump in fault-parallel velocity between the two faces of the fault.
        The :math:`\sqrt{r}` dependence is the :math:`m = -1/2` mode and is the
        quantity a discrete model can be asked to reproduce — unlike the
        stress, which is singular at the tip.
        """
        if self._is_symbolic:
            raise ValueError(
                "this solution was built with symbolic parameters; give U0, "
                "R0 and eta numeric values to evaluate it")
        r = np.asarray(r, dtype=float)
        return 2.0 * self.U0 * np.sqrt(r / self.R0)
