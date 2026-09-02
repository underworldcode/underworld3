r"""Barr & Houseman (1996): flow around a fault that terminates inside a medium.

The linear (:math:`n = 1`) plane-strain case of their Appendix.

Reference
---------
T. D. Barr & G. A. Houseman, *Deformation fields around a fault embedded in a
non-linear ductile medium*, Geophys. J. Int. **125**, 473-490 (1996); Appendix,
equations (A1)-(A9). The companion letter, Barr & Houseman, Geophys. Res. Lett.
**19**, 1145-1148 (1992), gives the near-tip asymptotics.

Implementation follows that of @gthyagi, who uses this solution for fault
benchmarking, and who verified it independently against both papers and against
two numerical models — see the module notes below and PR #550.

Why this solution is unusual, and useful
----------------------------------------
Every other solution in this suite is a smooth field on a box. This one has a
**fault** in it: an internal boundary carrying zero shear traction, with
continuous normal velocity and continuous normal stress, and with its tip inside
the medium. That makes it the only absolute standard here for a fault
calculation — everything else a fault model can be measured against is another
discretisation.

The deformation is written in polar coordinates with the **origin at the fault
tip** and the fault along :math:`\theta = 0`. The stream function separates into
a Fourier series in :math:`m = q/2`, and the two halves of that series mean
different things:

* **whole-integer** :math:`m` — continuous deformation, no fault;
* **half-integer** :math:`m` — the fault-type discontinuity.

Boundedness of the velocity at :math:`r = 0` admits only one negative index,
:math:`m = -1/2`. That single mode carries the whole fault singularity, and it is
why the slip goes as :math:`\sqrt{r}` and the stress as :math:`1/\sqrt{r}` — the
exponents are a property of the fault's own Fourier mode, not an assumption.

The solution below is their plane-strain test problem: prescribe the velocity
(:meth:`FaultedMedium.boundary_velocity`) on the perimeter of a disc of radius
:math:`R_0`, impose the fault conditions on :math:`\theta = 0`, and the interior
field is exact.

Only plane strain is implemented. The paper also gives a thin-viscous-sheet
(plane-stress) solution, equations (A10)-(A13), in which the in-plane divergence
is NOT zero — a different equation set from Underworld's incompressible Stokes,
so it is not a benchmark for this solver.

Two transcription points, both settled by mathematics
-----------------------------------------------------
**The sign of the half-integer sine terms.** They appear with one sign in the
paper's boundary datum (A8b) and the opposite sign in its solution (A9b).
Incompressibility settles it without anyone adjudicating a scanned minus sign:
for :math:`u_r = A\sqrt{R}f(\theta)` and :math:`u_\theta = \sqrt{R}g(\theta)`,
:math:`\nabla\cdot\mathbf u = 0` forces :math:`g' = -\tfrac32 A f`, which
integrates to (A8b)'s sign. Flip it and the divergence becomes
:math:`(0.75\cos(\theta/2) + 2.25\cos(3\theta/2))/\sqrt r` — non-zero purely in
the fault modes.

**A9b also has the wrong function.** @gthyagi's independent reading of both
papers found that printed A9b carries :math:`\cos(3\theta/2)` where
:math:`\sin(3\theta/2)` belongs, as well as the opposite sign; the form used
here reproduces A8b exactly at :math:`r = R_0` and satisfies the equations,
while printed A9b fails both. It is a typographical error in the paper rather
than an alternative convention.

Conventions
-----------
Their constitutive relation is :math:`\tau_{ij} = B\dot E^{(1/n - 1)}
\dot\varepsilon_{ij}` with :math:`B = 2\eta_0` at :math:`n = 1`, so
:math:`\tau = 2\eta\dot\varepsilon` as usual.

Their **pressure takes extension as positive**, so the paper's force balance is
:math:`\partial_j\tau_{ij} + \partial_i p = 0`. Underworld3 — and therefore this
suite — takes pressure positive in compression, so :attr:`fn_pressure` and
:attr:`FaultedMedium.pressure_polar` are the **negative** of the paper's (A9c).
That is a convention difference, not a defect in the paper, and it is recorded
with the family's other errata; it is also measured rather than asserted, in two
independent ways:

* symbolically, by this suite's momentum residual, which is formed as
  :math:`\nabla\cdot\sigma + \mathbf f` with :math:`\sigma = 2\eta\dot\varepsilon
  - p\mathbf I`;
* numerically, by @gthyagi against a UW3 Stokes solve on a Gmsh slit disc —
  compared against :math:`-p_{\rm BH96}` the pressure error converges (13.1%,
  6.9%, 3.3% at :math:`h` = 0.20, 0.10, 0.05), and compared against the paper's
  own sign it is about 199% at every resolution.

See Also
--------
underworld3.analytic._base : the contract this solution satisfies.
"""

import numpy as np
import sympy

from ._base import AnalyticSolution


class FaultedMedium(AnalyticSolution):
    r"""Barr & Houseman's fault-tip solution on a disc of radius :math:`R_0`.

    The fault occupies :math:`\theta = 0` from the tip to the perimeter. Slip is
    the jump in fault-parallel velocity across it, :math:`2U_0\sqrt{r/R_0}`, so
    the relative slip velocity at the perimeter is :math:`2U_0` — the paper's
    normalisation.

    Parameters
    ----------
    mesh : Mesh
        Supplies the coordinate symbols the exact fields are written in.
    U0 : float or sympy.Symbol
        Velocity scale. The slip at the perimeter is ``2 * U0``.
    R0 : float or sympy.Symbol
        Radius of the disc, and the length of the fault.
    eta : float or sympy.Symbol
        Newtonian viscosity of the medium.
    tip : sequence of 2 floats
        Where the fault tip sits in the mesh's coordinates. The fault runs from
        there along :math:`+x`.

    Notes
    -----
    **The domain is the solution's, not the mesh's.** The disc of radius
    :math:`R_0` about the tip is where this field means something; a mesh only
    supplies the coordinate symbols, and :meth:`sample_points` returns points on
    the disc rather than on the mesh. Validating the solution therefore does not
    need the slit-disc mesh that solving it does — which is just as well, since
    Underworld cannot yet build one (see :meth:`apply_boundary_conditions`).

    **The field is multivalued about the tip** — that is what a fault is — so it
    is a function of :math:`(r, \theta)` with :math:`\theta\in[0, 2\pi)`, not of
    the Cartesian coordinates alone. The branch cut lies **on the fault**, and
    both representations here put it there deliberately: the NumPy evaluators
    take ``arctan2(y, x) mod 2*pi``, and the symbolic Cartesian fields use a
    half-angle form of :math:`\theta` whose own cut is the positive :math:`x`
    axis. A bare ``atan2`` would put the cut on the *negative* :math:`x` axis and
    silently return the wrong face of the fault.

    The parameters may be SymPy symbols. That is not a convenience: a symbolic
    check with :math:`U_0 = R_0 = \eta = 1` is satisfied by a transcription
    carrying the wrong power of :math:`R_0` in the pressure, so leaving them free
    makes the verification strictly stronger for the same runtime. The NumPy
    evaluators refuse a symbolic instance, naming the reason.
    """

    dim = 2
    solves = "stokes"
    nonlinear = False

    # Half a dozen terms per field: every gate on it runs in well under a
    # second, so it belongs in the per-PR tier.
    expensive_to_validate = False

    reference = (
        "T. D. Barr & G. A. Houseman, Geophys. J. Int. 125, 473-490 (1996), "
        "Appendix (A1)-(A9), plane strain, n = 1. Implementation follows "
        "@gthyagi; verified by him against both BH papers and against numerical "
        "UW3 models (PR #550)."
    )

    eqn_velocity = (
        r"u_r = \tfrac{U_0}{4}\left[R^2(\sin\theta - \sin 3\theta) "
        r"- 2R^3(\sin 2\theta - \sin 4\theta) "
        r"+ \sqrt{R}\left(\cos\tfrac{\theta}{2} + 3\cos\tfrac{3\theta}{2}\right)\right]"
    )
    eqn_pressure = (
        r"p = -\frac{\eta U_0}{R_0}\left[-2R\sin\theta + 3R^2\sin 2\theta "
        r"+ R^{-1/2}\cos\tfrac{\theta}{2}\right] \quad (R = r/R_0)"
    )
    eqn_viscosity = r"\eta = \text{const}"
    eqn_bodyforce = r"\mathbf f = 0 \quad\text{(driven by the perimeter datum)}"

    def __init__(self, mesh, U0=1.0, R0=1.0, eta=1.0, tip=(0.0, 0.0)):
        super().__init__(mesh)

        for name, value in (("R0", R0), ("eta", eta)):
            if not isinstance(value, sympy.Basic) and not float(value) > 0.0:
                raise ValueError(f"{name} must be positive.")

        self.U0 = U0 if isinstance(U0, sympy.Basic) else float(U0)
        self.R0 = R0 if isinstance(R0, sympy.Basic) else float(R0)
        self.eta = eta if isinstance(eta, sympy.Basic) else float(eta)
        self.tip = np.asarray(tip, dtype=float)

        self._symbols = None
        self._stress_fn = None

        velocity, pressure, strainrate = self._cartesian()

        self.set_fields(
            velocity=velocity,
            pressure=pressure,
            viscosity=self.eta,
            bodyforce=(0, 0),
            strainrate=strainrate,
        )

    @property
    def _is_symbolic(self):
        return any(
            isinstance(value, sympy.Basic) for value in (self.U0, self.R0, self.eta)
        )

    # --------------------------------------------------------------- polar form
    @property
    def symbols(self):
        r"""The polar symbols ``(r, theta)`` the fault-frame expressions use.

        ``r`` is positive — the solution is singular at ``r = 0`` and never
        evaluated there — but ``theta`` is only REAL. It runs over
        :math:`[0, 2\pi)` and the fault conditions are checked at
        :math:`\theta = 0`, which a ``positive=True`` assumption excludes; SymPy
        would then be entitled to simplify away a substitution the assumption
        says cannot happen.

        Cached on the instance, so repeated access returns the same objects
        rather than relying on SymPy's global symbol cache for identity.
        """

        if self._symbols is None:
            self._symbols = (
                sympy.Symbol("r", positive=True),
                sympy.Symbol("theta", real=True),
            )
        return self._symbols

    def _polar(self):
        """``(u_r, u_theta, p)`` in the fault frame, in UW3's pressure sign."""

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

        # Their (A9c) is extension-positive; UW3 is compression-positive, so the
        # sign is flipped HERE, once, and nowhere else. See the module notes.
        p = -(eta * U0 / R0) * (
            -2 * R * sympy.sin(t)
            + 3 * R**2 * sympy.sin(2 * t)
            + sympy.cos(t / 2) / sympy.sqrt(R)
        )

        return u_r, u_t, p

    def _strainrate_polar(self):
        r"""``(e_rr, e_r_theta, e_theta_theta)`` from the polar velocity."""

        r, t = self.symbols
        u_r, u_t, _p = self._polar()

        e_rr = sympy.diff(u_r, r)
        e_tt = sympy.diff(u_t, t) / r + u_r / r
        e_rt = (r * sympy.diff(u_t / r, r) + sympy.diff(u_r, t) / r) / 2

        return e_rr, e_rt, e_tt

    @property
    def velocity_polar(self):
        """``(u_r, u_theta)`` as SymPy expressions in ``r`` and ``theta``."""

        u_r, u_t, _p = self._polar()
        return u_r, u_t

    @property
    def pressure_polar(self):
        r"""Pressure in the fault frame, **positive in compression**.

        This is :math:`-1` times the paper's (A9c), which is extension-positive.
        The suite follows Underworld3's convention rather than any paper's; see
        the module notes for the two measurements that pin it.
        """

        return self._polar()[2]

    def boundary_velocity(self):
        r"""The velocity datum on :math:`r = R_0` that produces the solution.

        Their equations (A8a, A8b). Returned as ``(U_r, U_theta)`` SymPy
        expressions in ``theta``; this is what a solver's Dirichlet condition on
        the disc perimeter must impose.
        """

        r, _t = self.symbols
        u_r, u_t = self.velocity_polar

        return (
            sympy.simplify(u_r.subs(r, self.R0)),
            sympy.simplify(u_t.subs(r, self.R0)),
        )

    def fault_normal_velocity(self):
        r"""The fault-normal velocity :math:`u_\theta` on :math:`\theta = 0`.

        A SymPy expression in ``r``. It is the *same* on both faces — that
        continuity is one of the paper's three fault conditions — so it is the
        datum a split-node model imposes on each face while leaving the
        tangential traction natural. The tangential velocity is what jumps; see
        :meth:`slip`.
        """

        _r, t = self.symbols
        _u_r, u_t = self.velocity_polar

        return sympy.simplify(u_t.subs(t, 0))

    def slip(self, r):
        r"""Fault slip :math:`2U_0\sqrt{r/R_0}` at radius ``r`` from the tip.

        The jump in fault-parallel velocity between the two faces. The
        :math:`\sqrt r` dependence is the :math:`m = -1/2` mode, and it is the
        quantity a discrete model can be asked to reproduce — unlike the stress,
        which is singular at the tip.
        """

        self._refuse_if_symbolic()

        return 2.0 * self.U0 * np.sqrt(np.asarray(r, dtype=float) / self.R0)

    # ----------------------------------------------------- Cartesian, on mesh.X
    def _cartesian(self):
        """The contract's fields, in the mesh coordinates.

        The polar expressions with :math:`(r, \\theta)` replaced by their
        Cartesian forms about the tip, and the polar components rotated. The
        rotation uses :math:`\\cos\\theta = x/r` and :math:`\\sin\\theta = y/r`
        directly rather than trigonometry of an arctangent — same value, far
        smaller expression.
        """

        r_sym, t_sym = self.symbols

        x = self.mesh.X[0] - self.tip[0]
        y = self.mesh.X[1] - self.tip[1]
        r = sympy.sqrt(x**2 + y**2)

        # theta in [0, 2*pi) with the branch cut ON the fault, from the
        # half-angle identity tan(theta/2) = (r - x)/y.
        #
        # The two branches are the SAME function — (r - x)(r + x) = y^2, so the
        # first is the second with both arguments of atan2 scaled by the
        # positive quantity (r + x), which leaves the angle alone. The split is
        # purely for conditioning: near the fault r - x is a difference of two
        # nearly equal numbers, and forming it directly loses the relative
        # accuracy of theta as 1/theta^2. Measured, at 1e-6 radians off the
        # fault: momentum residual 3e-6 formed directly, 3e-15 formed this way.
        # Both branches are also division-free, so the unselected one cannot
        # raise on the axis where its denominator would have vanished.
        theta = 2 * sympy.Piecewise(
            (sympy.atan2(y**2, y * (r + x)), x > 0),
            (sympy.atan2(r - x, y), True),
        )

        into_cartesian = [(r_sym, r), (t_sym, theta)]

        u_r, u_t, pressure = (
            expression.subs(into_cartesian, simultaneous=True)
            for expression in self._polar()
        )
        e_rr, e_rt, e_tt = (
            expression.subs(into_cartesian, simultaneous=True)
            for expression in self._strainrate_polar()
        )

        cos_t, sin_t = x / r, y / r

        velocity = (
            u_r * cos_t - u_t * sin_t,
            u_r * sin_t + u_t * cos_t,
        )
        strainrate = self._rotate(e_rr, e_rt, e_tt, cos_t, sin_t)

        return velocity, pressure, strainrate

    @staticmethod
    def _rotate(a_rr, a_rt, a_tt, cos_t, sin_t):
        """A symmetric polar tensor as its ``[[xx, xy], [xy, yy]]`` components."""

        return (
            (
                a_rr * cos_t**2 - 2 * a_rt * cos_t * sin_t + a_tt * sin_t**2,
                (a_rr - a_tt) * cos_t * sin_t + a_rt * (cos_t**2 - sin_t**2),
            ),
            (
                (a_rr - a_tt) * cos_t * sin_t + a_rt * (cos_t**2 - sin_t**2),
                a_rr * sin_t**2 + 2 * a_rt * cos_t * sin_t + a_tt * cos_t**2,
            ),
        )

    def sample_points(self, count=12):
        r"""Points on the disc, away from the tip and off both faces of the fault.

        The generic sampler is a box, which this solution is not posed on, and it
        would land on both of the places the field is not a function of position:
        the tip, where the pressure and stress diverge, and the fault itself,
        where the velocity is genuinely discontinuous.

        The fixed points carry the sampling that matters: a decade of radii, the
        perimeter, and a point :math:`10^{-6}` radians off *each* face — close
        enough that a wrongly placed branch cut, or a badly conditioned
        :math:`\theta`, shows up in the residual gates rather than only in a
        dedicated test.
        """

        self._refuse_if_symbolic()

        rng = np.random.default_rng(20260815)

        radii = self.R0 * rng.uniform(0.15, 0.95, count)
        angles = rng.uniform(1.0e-3, 2.0 * np.pi - 1.0e-3, count)

        near = 1.0e-6
        radii = np.r_[radii, self.R0 * np.array([0.05, 0.2, 0.5, 0.5, 1.0, 1.0])]
        angles = np.r_[
            angles,
            np.array([np.pi / 3, 3 * np.pi / 2, near, 2 * np.pi - near, np.pi / 2, np.pi]),
        ]

        return np.column_stack(
            [
                self.tip[0] + radii * np.cos(angles),
                self.tip[1] + radii * np.sin(angles),
            ]
        )

    # --------------------------------------------------- boundaries and solving
    @property
    def boundaries(self):
        """The perimeter of the disc and the two faces of the fault."""

        return ["Perimeter", "FaultUpper", "FaultLower"]

    def apply_boundary_conditions(self, solver):
        """Refused: this solution needs a mesh Underworld cannot yet build.

        Every other solution in the family states its conditions here by
        composing :func:`~underworld3.analytic.free_slip` and friends. This one
        refuses, and the reason is the mesh rather than the boundary conditions:

        * The fault is an **internal** boundary whose two faces must be separate
          degrees of freedom at the same coordinates. That is a property of the
          mesh, not of the solver, and Underworld cannot yet build it for a fault
          that reaches the domain boundary (#549).
        * Its conditions are per-component — the fault-normal velocity is
          prescribed on both faces while the tangential traction is left natural
          — so they need a component-wise Dirichlet condition on an internal
          boundary rather than any of the whole-velocity helpers.

        The pieces a model needs are all here:
        :meth:`boundary_velocity` (the perimeter datum, their A8),
        :meth:`fault_normal_velocity` (the common datum on both faces) and
        :meth:`slip` (the answer to check against). @gthyagi's Gmsh slit-disc
        model in PR #550 is a worked example of assembling them.
        """

        raise NotImplementedError(
            "FaultedMedium is posed on a slit disc: the fault is an internal "
            "boundary with two coincident faces, and its conditions are "
            "per-component (fault-normal velocity prescribed, tangential "
            "traction natural). UW3 cannot yet mesh a fault that reaches the "
            "domain boundary (#549). Build the "
            "conditions from boundary_velocity(), fault_normal_velocity() and "
            "slip() — see the docstring."
        )

    # ------------------------------------------------------- NumPy, fault frame
    def evaluate_velocity(self, coords):
        r"""Velocity at Cartesian ``coords``, in the mesh's coordinates.

        Defined AT the tip: every term of the velocity carries a positive power
        of :math:`r`, so the limit is zero and is returned. It is the pressure
        that is singular there, not the velocity.
        """

        _r, t, R = self._polar_of(coords)

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

        return np.column_stack(
            [
                u_r * np.cos(t) - u_t * np.sin(t),
                u_r * np.sin(t) + u_t * np.cos(t),
            ]
        )

    def evaluate_pressure(self, coords):
        r"""Pressure at Cartesian ``coords``; refuses the tip.

        Positive in compression, as everywhere else in this suite. The pressure
        carries the :math:`r^{-1/2}` term of the :math:`m = -1/2` mode and
        genuinely diverges at :math:`r = 0`.
        """

        r, t, R = self._polar_of(coords)
        self._refuse_the_tip(r, "pressure")

        return -(self.eta * self.U0 / self.R0) * (
            -2 * R * np.sin(t) + 3 * R**2 * np.sin(2 * t) + np.cos(t / 2) / np.sqrt(R)
        )

    def evaluate_traction(self, coords, normal):
        r"""Traction :math:`\sigma\cdot\hat n` at Cartesian ``coords``.

        With :math:`\sigma = 2\eta\dot\varepsilon - p\mathbf I` and the pressure
        positive in compression — the suite's convention, and the same total
        stress the paper writes as :math:`\tau + p_{\rm BH}\mathbf I`. Refuses
        the tip, where the stress diverges.

        This is what a boundary needs if its NORMAL velocity component is left
        free rather than prescribed. Leaving one component free is worth doing:
        with velocity Dirichlet on every wall the pressure is determined only up
        to a constant AND the datum must carry exactly zero net flux, and a
        traction condition removes both requirements at once. It is also what
        Barr & Houseman do — their left-hand boundary carries a constant normal
        stress, not a prescribed normal velocity.
        """

        r, t, _R = self._polar_of(coords)
        self._refuse_the_tip(r, "stress")

        s_rr, s_rt, s_tt = self._stress_polar_numeric(r, t)
        (sxx, sxy), (_syx, syy) = self._rotate(s_rr, s_rt, s_tt, np.cos(t), np.sin(t))

        n = np.asarray(normal, dtype=float)
        if n.ndim == 1:
            n = np.broadcast_to(n, (len(r), 2))

        return np.column_stack(
            [sxx * n[:, 0] + sxy * n[:, 1], sxy * n[:, 0] + syy * n[:, 1]]
        )

    def _stress_polar_numeric(self, r, t):
        """``(sigma_rr, sigma_r_theta, sigma_theta_theta)``, built once via SymPy."""

        if self._stress_fn is None:
            r_sym, t_sym = self.symbols
            e_rr, e_rt, e_tt = self._strainrate_polar()
            pressure = self.pressure_polar
            two_eta = 2 * self.eta

            self._stress_fn = sympy.lambdify(
                (r_sym, t_sym),
                [
                    two_eta * e_rr - pressure,
                    two_eta * e_rt,
                    two_eta * e_tt - pressure,
                ],
                "numpy",
            )

        return self._stress_fn(r, t)

    def _polar_of(self, coords):
        """``(r, theta, r/R0)`` about the tip, with the cut ON the fault.

        ``arctan2`` is accurate close to either face here — it is only the
        *symbolic* Cartesian form that needs care there — so the NumPy path takes
        the direct route and shifts the branch cut by a modulus.
        """

        self._refuse_if_symbolic()

        X = np.asarray(coords, dtype=float)
        if X.ndim != 2 or X.shape[1] != 2:
            raise ValueError("coords must have shape (N, 2)")

        x = X[:, 0] - self.tip[0]
        y = X[:, 1] - self.tip[1]

        r = np.hypot(x, y)
        t = np.mod(np.arctan2(y, x), 2.0 * np.pi)

        return r, t, r / self.R0

    def _refuse_if_symbolic(self):
        if self._is_symbolic:
            raise ValueError(
                "this solution was built with symbolic parameters; give U0, R0 "
                "and eta numeric values to evaluate it"
            )

    @staticmethod
    def _refuse_the_tip(r, quantity):
        if np.any(r == 0.0):
            raise ValueError(
                f"the {quantity} is singular at the fault tip; exclude r = 0. "
                f"The velocity is defined there — evaluate_velocity returns the "
                f"limit, which is zero."
            )
