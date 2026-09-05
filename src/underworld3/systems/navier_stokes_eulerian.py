r"""Navier-Stokes with Eulerian SUPG momentum transport.

The incompressible Navier-Stokes equations solved on the mesh with the
momentum advection assembled implicitly in the saddle-point residual and
stabilised by the streamline-upwind Petrov-Galerkin term, the vector
counterpart of :class:`~underworld3.systems.AdvDiffusionSUPG`. The time
scheme is the same multistep family: Crank-Nicolson (the theta rule) at
order 1, BDF2 at order 2, with the history held on the mesh by the
Eulerian history manager. No stress history is carried: the viscous stress
at an earlier level is rebuilt from the stored velocity level through the
constitutive model.

The advecting velocity :math:`\mathbf{a}` in :math:`(\mathbf{a}\cdot\nabla)\mathbf{u}^{n+1}`
is a choice (``advection=``): ``"extrapolated"`` (default) uses
:math:`2\mathbf{u}^n - \mathbf{u}^{n-1}`, a second-order lag that makes
each step one linear (Oseen) solve; ``picard_iterations`` re-solves with
the latest iterate for the fully implicit fixed point; ``"implicit"`` puts
:math:`\mathbf{u}^{n+1}` itself in the advection and lets the SNES take
Newton steps on the quadratic term.

Design note: ``docs/developer/design/eulerian-supg-transport.md``.
"""

import warnings

import numpy as np
import sympy
from typing import Optional, Union

import underworld3 as uw
import underworld3.timing as timing
from underworld3.function import expression as public_expression
from underworld3.systems.ddt import Eulerian as Eulerian_DDt
from underworld3.systems.solvers import SNES_Stokes

_ADVECTION_MODES = ("extrapolated", "implicit")


class SNES_NavierStokes_SUPG(SNES_Stokes):
    r"""Navier-Stokes solver with Eulerian SUPG momentum transport.

    Solves

    .. math::
        \rho\left(\frac{\partial \mathbf{u}}{\partial t}
        + (\mathbf{u}\cdot\nabla)\mathbf{u}\right)
        - \nabla\cdot\left[\boldsymbol{\tau}(\mathbf{u}) - p\mathbf{I}\right]
        = \mathbf{f}, \qquad \nabla\cdot\mathbf{u} = 0,

    with :math:`\boldsymbol{\tau}` the deviatoric stress of the constitutive
    model. In the pointwise form the momentum residual is

    .. math::
        \mathbf{f}_0 = \mathbf{R}, \qquad
        \mathbf{F}_1 = \sum_k w_k\,\boldsymbol{\tau}(\mathbf{u}^{(k)})
                       - p_\mathrm{mech}\mathbf{I} + \tau_\mathrm{s}\,\mathbf{R}\otimes\mathbf{a},

    where :math:`\mathbf{R} = \rho\,(\dot{\mathbf{u}} + \sum_k w_k (\mathbf{a}_k\cdot\nabla)\mathbf{u}^{(k)})
    + \nabla p - \mathbf{f}` is the strong residual of the time scheme (first
    derivatives only, so without the viscous term; :math:`\mathbf{f}_0` carries
    it without :math:`\nabla p`, which enters through the flux), :math:`w_k` the weights of the spatial operator at each time level
    (:math:`w_0 = 1` for BDF, the Adams-Moulton weights for the theta rule),
    :math:`\mathbf{a}_0 = \mathbf{a}` the advecting velocity at the new level
    and :math:`\mathbf{a}_k = \mathbf{u}^{(k)}` at the stored ones. The last
    term of :math:`\mathbf{F}_1` is the Petrov-Galerkin perturbation
    :math:`\tau_\mathrm{s}(\mathbf{a}\cdot\nabla)\mathbf{w}` applied to
    :math:`\mathbf{R}`, with

    .. math::
        \tau_\mathrm{s} = \left[\left(\frac{C_t c_0}{\Delta t}\right)^2
            + \left(\frac{C_u |\mathbf{a}|}{h}\right)^2
            + \left(\frac{C_\nu\, \nu}{h^2}\right)^2\right]^{-1/2},
        \qquad \nu = \eta / \rho,

    :math:`h` the local cell size and the three weights runtime constants
    (``tau_weights``). The pressure equation is the incompressibility
    constraint, unchanged from the Stokes solver; Taylor-Hood elements need
    no pressure stabilisation.

    Parameters
    ----------
    mesh, velocityField, pressureField
        As for :class:`~underworld3.systems.Stokes`.
    rho : float or expression, default 1.0
        Density.
    order : int, default 1
        Time scheme: 1 is the theta rule (Crank-Nicolson at ``theta=0.5``),
        2 is BDF2.
    theta : float, optional
        Crank-Nicolson blend at order 1 (0.5 default; 1.0 backward Euler).
        Order 2 takes ``theta=1.0`` and refuses anything else.
    advection : {"extrapolated", "implicit"}, default "extrapolated"
        The advecting velocity at the new time level: the second-order
        extrapolation :math:`2\mathbf{u}^n - \mathbf{u}^{n-1}` (a linear
        step) or the unknown itself (Newton on the quadratic term).
    picard_iterations : int, default 0
        With ``"extrapolated"``, the number of further passes per step that
        re-solve with the latest iterate as the advecting velocity, stopping
        early when the velocity stops changing (``picard_tolerance``). The
        fixed point is the fully implicit scheme without a tangent.
    picard_tolerance : float, default 1e-4
        Relative change of the velocity (max norm) below which the Picard
        passes stop.
    degree, p_continuous, verbose
        As for :class:`~underworld3.systems.Stokes`.

    Notes
    -----
    - ``DFDt`` (a stress history) is refused: the theta rule forms the
      viscous stress at level n from the stored velocity as
      :math:`2\eta\,\dot\varepsilon(\mathbf{u}^n)` with the current effective
      viscosity, which is exact for a constant viscosity; use ``order=2``
      (all spatial terms at n+1) with a strain-rate dependent viscosity.
    - The linear solver is the Stokes fieldsplit: the velocity block is
      nonsymmetric, which its smoother and flexible outer solver already
      allow for, while the pressure Schur approximation is the viscous-limit
      one and costs outer iterations as :math:`\rho|\mathbf{a}|\Delta t/\eta`
      grows.
    - The velocity history levels, the advecting-velocity field and the
      extrapolation level are mesh variables the solver owns.
    """

    @timing.routine_timer_decorator
    def __init__(
        self,
        mesh: uw.discretisation.Mesh,
        velocityField: uw.discretisation.MeshVariable,
        pressureField: uw.discretisation.MeshVariable,
        rho=1.0,
        order: int = 1,
        theta: Optional[float] = None,
        advection: str = "extrapolated",
        picard_iterations: int = 0,
        picard_tolerance: float = 1.0e-4,
        degree: Optional[int] = 2,
        p_continuous: Optional[bool] = True,
        verbose: bool = False,
        DuDt: Optional[Eulerian_DDt] = None,
        DFDt=None,
        restore_points_func=None,
    ):
        if DFDt is not None:
            raise ValueError(
                "AdvDiffusionSUPG-style Navier-Stokes carries no stress history: "
                "the viscous stress at earlier levels is rebuilt from the stored "
                "velocity. Do not pass DFDt."
            )
        if restore_points_func is not None:
            warnings.warn(
                "NavierStokesSUPG ignores restore_points_func: it configures the "
                "semi-Lagrangian trace-back and the Eulerian scheme has none.",
                stacklevel=2,
            )
        order = int(order)
        if order not in (1, 2):
            raise ValueError(f"order must be 1 or 2, not {order}.")
        theta = float(theta) if theta is not None else (0.5 if order == 1 else 1.0)
        if theta != 1.0 and order != 1:
            raise ValueError(
                "theta applies at order 1 only (0.5 is Crank-Nicolson, 1.0 is "
                "backward Euler); order 2 takes theta=1.0."
            )
        advection = str(advection)
        if advection not in _ADVECTION_MODES:
            raise ValueError(f"advection must be one of {_ADVECTION_MODES}, not {advection!r}.")

        super().__init__(
            mesh, velocityField, pressureField, degree, p_continuous, verbose,
            DuDt=None, DFDt=None,
        )

        self._time_order = order
        self._theta = theta
        self._integrator = "am" if order == 1 else "bdf"
        self._advection_mode = advection
        self._picard_iterations = int(picard_iterations)
        self._picard_tolerance = float(picard_tolerance)
        self._picard_count = 0
        self._last_timestep = None
        self._last_change_rate = None

        tag = self.instance_number
        self._rho = public_expression(rf"\rho_{{{tag}}}", rho, "Density")
        self._delta_t = public_expression(rf"\Delta t_{{{tag}}}", 1.0, "Navier-Stokes timestep")
        self._supg_weight = public_expression(
            rf"w^{{\mathrm{{SUPG}}}}_{{{tag}}}", 1.0, "SUPG term weight (0 = Galerkin)")
        self._tau_weights = [
            public_expression(rf"C^{{\tau}}_{{t,{tag}}}", 2.0, "tau transient weight"),
            public_expression(rf"C^{{\tau}}_{{u,{tag}}}", 2.0, "tau advective weight"),
            public_expression(rf"C^{{\tau}}_{{\nu,{tag}}}", 4.0, "tau viscous weight"),
        ]

        u = self.Unknowns.u
        if DuDt is None:
            self.Unknowns.DuDt = Eulerian_DDt(
                self.mesh,
                u,
                vtype=uw.VarType.VECTOR,
                degree=u.degree,
                continuous=u.continuous,
                V_fn=None,
                theta=theta,
                varsymbol=u.symbol,
                verbose=verbose,
                bcs=self.essential_bcs,
                order=order,
                smoothing=0.0,
            )
        else:
            if DuDt.order < order:
                raise ValueError(
                    f"DuDt supplied is order {DuDt.order} but order {order} was requested.")
            if getattr(DuDt, "V_fn", None) is not None:
                raise ValueError(
                    "DuDt must be built with V_fn=None: advection is assembled "
                    "implicitly by this solver, not as an explicit history correction.")
            self.Unknowns.DuDt = DuDt

        # The advecting velocity at the new level (values set before each
        # solve: the extrapolation, or the latest Picard iterate) and the
        # level n-1 the extrapolation needs beyond what the history holds.
        self._a_var = uw.discretisation.MeshVariable(
            f"a_NSSUPG_{tag}", self.mesh, self.mesh.dim, degree=u.degree,
            continuous=u.continuous, varsymbol=rf"\mathbf{{a}}_{{{tag}}}")
        self._u_prev = uw.discretisation.MeshVariable(
            f"u_prev_NSSUPG_{tag}", self.mesh, self.mesh.dim, degree=u.degree,
            continuous=u.continuous, varsymbol=rf"\mathbf{{u}}^{{n-1}}_{{{tag}}}")
        self._history_primed = False

    # ------------------------------------------------------------------
    # Scheme description and knobs
    # ------------------------------------------------------------------

    @property
    def integrator(self) -> str:
        """``"am"`` (the theta rule) at order 1, ``"bdf"`` at order 2."""
        return self._integrator

    @property
    def order(self) -> int:
        """Time scheme order."""
        return self._time_order

    @property
    def theta(self) -> float:
        """Adams-Moulton blend at order 1 (1.0 backward Euler, 0.5 Crank-Nicolson)."""
        return self._theta

    @theta.setter
    def theta(self, value):
        value = float(value)
        if value != 1.0 and self._time_order != 1:
            raise ValueError("theta applies at order 1 only; order 2 takes theta=1.0.")
        self._theta = value
        self.DuDt.theta = value

    @property
    def advection(self) -> str:
        """``"extrapolated"`` (linear Oseen step) or ``"implicit"`` (Newton)."""
        return self._advection_mode

    @advection.setter
    def advection(self, value):
        value = str(value)
        if value not in _ADVECTION_MODES:
            raise ValueError(f"advection must be one of {_ADVECTION_MODES}, not {value!r}.")
        if value != self._advection_mode:
            self._advection_mode = value
            self.is_setup = False

    @property
    def picard_iterations(self) -> int:
        return self._picard_iterations

    @picard_iterations.setter
    def picard_iterations(self, value):
        self._picard_iterations = int(value)

    @property
    def picard_count(self) -> int:
        """Picard passes the last step took beyond the first solve."""
        return self._picard_count

    @property
    def rho(self):
        """Density (a UW expression)."""
        return self._rho

    @rho.setter
    def rho(self, value):
        self._rho.sym = value

    @property
    def delta_t(self):
        r"""The timestep :math:`\Delta t` as a UW expression (a runtime constant)."""
        return self._delta_t

    @delta_t.setter
    def delta_t(self, value):
        value = self._nondimensional_time(value)
        self._delta_t.sym = value
        self._last_timestep = value

    @property
    def supg_weight(self) -> float:
        """Weight of the SUPG term; 0 gives the plain Galerkin scheme."""
        return float(self._supg_weight.sym)

    @supg_weight.setter
    def supg_weight(self, value):
        self._supg_weight.sym = float(value)

    @property
    def tau_weights(self):
        """The three weights of tau: transient, advective, viscous."""
        return tuple(float(w.sym) for w in self._tau_weights)

    @tau_weights.setter
    def tau_weights(self, values):
        ct, cu, cv = values
        for w, v in zip(self._tau_weights, (ct, cu, cv)):
            w.sym = float(v)

    # ------------------------------------------------------------------
    # The residual
    # ------------------------------------------------------------------

    def _states(self):
        r"""``[u^{n+1}, u^{n}, u^{n-1}, ...]`` as ``(1, dim)`` row matrices."""
        return [self.u.sym] + [ps.sym for ps in self.DuDt.psi_star]

    def _spatial_weights(self):
        """Weight of the spatial operator at each level of ``_states``."""
        n = len(self.DuDt.psi_star)
        if self._integrator == "bdf":
            return [sympy.Integer(1)] + [sympy.Integer(0)] * n
        return self.DuDt.am_coefficient_expressions[: n + 1]

    def _advecting_velocity(self):
        """The advecting velocity at the new level, as a ``(1, dim)`` row."""
        if self._advection_mode == "implicit":
            return self.u.sym
        return self._a_var.sym

    def _time_derivative(self):
        if self._integrator == "bdf":
            return self.DuDt.bdf() / self._delta_t
        u_new, u_old = self._states()[:2]
        return (u_new - u_old) / self._delta_t

    def _convective(self, a, u):
        r"""``(a . grad) u`` as a ``(1, dim)`` row for rows ``a`` and ``u``."""
        dim = self.mesh.dim
        X = self.mesh.X
        return sympy.Matrix([[sum(a[0, j] * u[0, i].diff(X[j]) for j in range(dim))
                              for i in range(dim)]])

    def _advection(self):
        states = self._states()
        total = sympy.zeros(1, self.mesh.dim)
        for k, (w, u_k) in enumerate(zip(self._spatial_weights(), states)):
            if w == 0:
                continue
            a_k = self._advecting_velocity() if k == 0 else u_k
            total = total + w * self._convective(a_k, u_k)
        return total

    def _strong_residual(self, with_pressure=False):
        r"""The strong momentum residual of the time scheme, first derivatives only.

        ``with_pressure=False`` gives the terms that live in :math:`\mathbf{f}_0`:
        density times the time derivative and the advection, less the body
        force. ``with_pressure=True`` adds :math:`\nabla p`, the residual the
        SUPG term must see: the pressure is applied through the flux
        :math:`-p\mathbf{I}` in :math:`\mathbf{F}_1`, so it must not appear in
        :math:`\mathbf{f}_0`, but a strong residual without it is O(1) at the
        exact solution and the stabilisation then injects an O(tau) error
        (measured on Kovasznay flow: 50 times the Galerkin error). The
        viscous term needs second derivatives the kernels do not see; it is
        the remaining inconsistency for P2 velocity.
        """
        # The body-force setter may store a column; the residual is a row.
        dim = self.mesh.dim
        f = sympy.Matrix(self.bodyforce.sym).reshape(1, dim)
        R = self._rho * (self._time_derivative() + self._advection()) - f
        if with_pressure:
            X = self.mesh.X
            R = R + sympy.Matrix([[self.p.sym[0].diff(X[i]) for i in range(dim)]])
        return R

    def _viscous_stress(self, u_row):
        r"""Deviatoric stress ``2 eta strain(u)`` for a velocity row, with the
        current effective viscosity of the constitutive model."""
        eta = self.constitutive_model.K
        return 2 * eta * sympy.Matrix(self.mesh.vector.strain_tensor(u_row))

    def _viscous_flux(self):
        states = self._states()
        weights = self._spatial_weights()
        total = weights[0] * self.stress_deviator
        for w, u_k in zip(weights[1:], states[1:]):
            if w == 0:
                continue
            total = total + w * self._viscous_stress(u_k)
        return total

    def _tau(self):
        dim = self.mesh.dim
        a = self._advecting_velocity()
        a_mag2 = sum(a[0, i] ** 2 for i in range(dim))
        h = self.mesh.cell_size()
        nu = self.constitutive_model.K / self._rho
        if self._integrator == "bdf":
            c0 = self.DuDt.bdf_coefficient_expressions[0]
        else:
            c0 = sympy.Integer(1)
        ct, cu, cv = self._tau_weights
        transient = (ct * c0 / self._delta_t) ** 2
        advective = (cu * sympy.sqrt(a_mag2) / h) ** 2
        viscous = (cv * nu / h ** 2) ** 2
        return self._supg_weight / sympy.sqrt(transient + advective + viscous + 1.0e-30)

    @property
    def F0(self):
        """Pointwise momentum residual: strong residual of the time scheme."""
        f0 = public_expression(
            r"\mathbf{f}_0\left( \mathbf{u} \right)",
            self._strong_residual(),
            "Navier-Stokes SUPG: strong residual (time derivative, advection, body force)",
        )
        self._u_f0 = f0
        return f0

    @property
    def F1(self):
        """Pointwise flux: weighted viscous stress, mechanical pressure, SUPG term."""
        dim = self.mesh.dim
        mechanical_pressure = (
            self.p.sym[0] - self.penalty * self.constitutive_model.K * self.div_u)
        R = self._strong_residual(with_pressure=True)
        a = self._advecting_velocity()
        F1 = public_expression(
            r"\mathbf{F}_1\left( \mathbf{u} \right)",
            self._viscous_flux() - sympy.eye(dim) * mechanical_pressure
            + self._tau() * (R.T * a),
            "Navier-Stokes SUPG: viscous flux of the time scheme, pressure, tau R (x) a",
        )
        self._u_f1 = F1
        return F1

    # ------------------------------------------------------------------
    # Timestep and solve
    # ------------------------------------------------------------------

    def _set_advecting_velocity(self, values):
        self._a_var.array[...] = values

    def _prime_history(self):
        """First solve: the extrapolation level equals the current velocity."""
        if not self._history_primed:
            self._u_prev.array[...] = self.u.array[...]
            self._history_primed = True

    @timing.routine_timer_decorator
    def estimate_dt(self, fraction: float = 0.02, basis: str = "accuracy"):
        r"""A timestep for this scheme.

        ``basis="accuracy"`` (default): the step at which the velocity changes
        by ``fraction`` of its range, from the realised rate of the last step;
        before the first solve, and whenever nothing has changed yet, the
        cell-crossing time of the Stokes solver (``basis="resolution"``).
        """
        if basis == "resolution" or self._last_change_rate is None:
            return SNES_Stokes.estimate_dt(self)
        if basis != "accuracy":
            raise ValueError(f"basis must be 'accuracy' or 'resolution', not {basis!r}.")
        from mpi4py import MPI
        comm = uw.mpi.comm
        speed = np.linalg.norm(np.asarray(self.u.array).reshape(-1, self.mesh.dim), axis=1)
        hi = comm.allreduce(float(speed.max()) if speed.size else 0.0, op=MPI.MAX)
        rate = self._last_change_rate
        dt = fraction * hi / rate if rate > 0.0 else np.inf
        if np.isinf(dt) or hi <= 0.0:
            return SNES_Stokes.estimate_dt(self)
        return dt

    @timing.routine_timer_decorator
    def solve(
        self,
        zero_init_guess: Optional[bool] = None,
        timestep=None,
        _force_setup: bool = False,
        verbose: bool = False,
        picard_iterations: Optional[int] = None,
        divergence_retries: int = 0,
        **kwargs,
    ):
        r"""Advance the velocity and pressure by one step.

        ``timestep`` sets :attr:`delta_t`; omit it to reuse the last value.
        With ``advection="extrapolated"`` the step is one linear solve, plus
        up to ``picard_iterations`` further solves with the latest iterate as
        the advecting velocity; with ``"implicit"`` the SNES solves the
        quadratic term by Newton iteration.
        """
        for name in ("time", "order", "evalf", "_evalf", "homotopy"):
            kwargs.pop(name, None)
        if kwargs:
            warnings.warn(f"NavierStokesSUPG.solve ignores {sorted(kwargs)}", stacklevel=2)
        if timestep is not None:
            self.delta_t = timestep
        elif self._last_timestep is None:
            raise ValueError("solve() needs a timestep: pass timestep=<dt> or set solver.delta_t first.")
        dt = self._last_timestep

        if _force_setup:
            self._needs_function_rewire = True
        if not self.constitutive_model._solver_is_setup:
            self._needs_function_rewire = True
        # The base _build resolves the preconditioner choice against the mesh
        # before the SNES reads its options; the setup stages must not be run
        # directly here (they mark the solver set up first, #683).
        self._build(verbose)

        self._prime_history()
        u_n = np.array(self.u.array[...])
        if self._advection_mode == "extrapolated":
            self._set_advecting_velocity(2.0 * u_n - np.asarray(self._u_prev.array[...]))
        self.DuDt.update_pre_solve(dt, verbose=verbose)

        passes = 1
        if self._advection_mode == "extrapolated":
            n_picard = self._picard_iterations if picard_iterations is None else int(picard_iterations)
            passes += max(n_picard, 0)
        from mpi4py import MPI
        comm = uw.mpi.comm
        self._picard_count = 0
        for k in range(passes):
            if k > 0:
                previous = np.array(self.u.array[...])
                self._set_advecting_velocity(previous)
            SNES_Stokes.solve(
                self, zero_init_guess if k == 0 else False,
                _force_setup=_force_setup if k == 0 else False,
                verbose=verbose, picard=0, divergence_retries=divergence_retries,
            )
            if k > 0:
                self._picard_count = k
                change = np.abs(np.asarray(self.u.array[...]) - previous).max() if previous.size else 0.0
                scale = np.abs(np.asarray(self.u.array[...])).max() if previous.size else 0.0
                change = comm.allreduce(float(change), op=MPI.MAX)
                scale = comm.allreduce(float(scale), op=MPI.MAX)
                if change <= self._picard_tolerance * max(scale, 1.0e-300):
                    break

        # Realised rate of change of the velocity, for estimate_dt.
        change = np.linalg.norm(
            (np.asarray(self.u.array[...]) - u_n).reshape(-1, self.mesh.dim), axis=1)
        local = float(change.max()) if change.size else 0.0
        self._last_change_rate = comm.allreduce(local, op=MPI.MAX) / dt

        # Shift the extrapolation level, then the history.
        self._u_prev.array[...] = self.DuDt.psi_star[0].array[...]
        self.DuDt.update_post_solve(dt, verbose=verbose)

        self.is_setup = True
        self.constitutive_model._solver_is_setup = True
