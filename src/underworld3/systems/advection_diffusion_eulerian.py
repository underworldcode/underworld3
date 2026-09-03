r"""Fully implicit Eulerian advection-diffusion with SUPG stabilisation.

The scalar transport equation

.. math::
    \frac{\partial \phi}{\partial t} + \mathbf{u}\cdot\nabla\phi
        - \nabla\cdot(\kappa\nabla\phi) = f

discretised on the mesh with a linear multistep rule in time and a
streamline-upwind Petrov-Galerkin (SUPG) term in space. Every time level
is a mesh variable held by an :class:`~underworld3.systems.ddt.Eulerian`
history manager, so the scheme's order is a construction argument and the
timestep and multistep coefficients are runtime constants of the compiled
kernels: neither changes the generated code.

The companion of :class:`~underworld3.systems.solvers.SNES_AdvectionDiffusion`
(semi-Lagrangian). The Eulerian scheme is stable at any cell Courant number
and its accuracy is set by how far the transported feature moves in one
step; the semi-Lagrangian scheme's accuracy is set by how far a characteristic
turns in one step. See ``docs/developer/design/eulerian-supg-transport.md``.
"""

import numpy as np
import sympy
from typing import Optional

import underworld3 as uw
import underworld3.timing as timing
from underworld3.systems import SNES_Scalar
from underworld3.utilities._api_tools import Template
from underworld3.function import expression as public_expression
from underworld3.systems.ddt import Eulerian as Eulerian_DDt
from underworld3.systems.solvers import (
    _advective_diffusive_dt,
    _dimensionalise_dt,
    _invalidate_solution_cache,
    _nondimensionalise_timestep,
)


def _as_row_vector(V_fn, dim):
    """Coerce a velocity expression to a ``(1, dim)`` sympy row Matrix."""
    if isinstance(V_fn, uw.discretisation.MeshVariable):
        V_fn = V_fn.sym
    if isinstance(V_fn, sympy.MatrixBase):
        if V_fn.shape == (1, dim):
            return V_fn
        if V_fn.shape == (dim, 1):
            return V_fn.T
        raise ValueError(
            f"V_fn has shape {V_fn.shape} but the mesh is {dim}-D; expected a "
            f"(1, {dim}) row vector such as `v.sym` of a vector MeshVariable."
        )
    raise ValueError(
        f"V_fn must be a (1, {dim}) sympy Matrix or a vector MeshVariable, "
        f"not {type(V_fn).__name__}."
    )


class SNES_AdvectionDiffusion_SUPG(SNES_Scalar):
    r"""Eulerian advection-diffusion solver, implicit in time, SUPG in space.

    .. math::
        \frac{\partial \phi}{\partial t} + \mathbf{u}\cdot\nabla\phi
            - \nabla\cdot(\kappa\nabla\phi) = f

    Two families of time integration are built from the same stored history
    :math:`\phi^{n}, \phi^{n-1}, \dots` (real mesh variables, so their
    gradients are available inside the kernels):

    ``integrator="bdf"`` (backward differentiation, order 1-3)

    .. math::
        \frac{1}{\Delta t}\sum_{k=0}^{N} c_k\,\phi^{n+1-k}
            + \mathbf{u}\cdot\nabla\phi^{n+1}
            - \nabla\cdot(\kappa\nabla\phi^{n+1}) = f

    ``integrator="am"`` (Adams-Moulton, order 1-3; ``theta`` at order 1)

    .. math::
        \frac{\phi^{n+1}-\phi^{n}}{\Delta t}
            + \sum_{k=0}^{N} a_k\left[\mathbf{u}\cdot\nabla\phi^{n+1-k}
            - \nabla\cdot(\kappa\nabla\phi^{n+1-k})\right] = f

    Order 1 with ``theta=1`` is backward Euler in both families;
    ``integrator="am", order=1, theta=0.5`` is Crank-Nicolson. The BDF
    coefficients :math:`c_k` and Adams-Moulton weights :math:`a_k` are the
    ones the :class:`~underworld3.systems.ddt.Eulerian` manager maintains;
    both ramp from first order over the opening steps unless a history is
    planted with ``solver.DuDt.set_initial_history``. A BDF3 request falls
    back to variable-step BDF2 whenever consecutive timesteps differ by more
    than 5%.

    **Weak form.** With the strong residual of the chosen scheme
    :math:`R(\phi)` (time derivative and advection; see below) the residual
    assembled through PETSc's pointwise interface is

    .. math::
        f_0 = R(\phi), \qquad
        \mathbf{f}_1 = \sum_k w_k\,\kappa\nabla\phi^{n+1-k}
            + \tau\,R(\phi)\,\mathbf{u},

    where :math:`w_k` are the weights of the spatial operator (:math:`w_0 = 1`
    for BDF, :math:`w_k = a_k` for Adams-Moulton). The SUPG contribution is
    the Petrov-Galerkin test-function perturbation
    :math:`\tau\,\mathbf{u}\cdot\nabla w` written as a flux against
    :math:`\nabla w`, so PETSc needs no modified test space. The strong
    residual carries no diffusion term because the pointwise kernels see
    first derivatives only; for linear elements that term vanishes
    identically, for higher orders it is the usual inconsistency of SUPG
    without a Laplacian reconstruction.

    **Stabilisation parameter.**

    .. math::
        \tau = \left[\left(\frac{2 c_0}{\Delta t}\right)^2
            + \left(\frac{2|\mathbf{u}|}{h}\right)^2
            + \left(\frac{4\kappa}{h^2}\right)^2\right]^{-1/2}

    with :math:`h` the local cell size (``mesh.cell_size()``, the
    :math:`\mathrm{volume}^{1/d}` equivalent radius) and :math:`c_0` the
    leading multistep coefficient. The three weights are runtime constants
    (``tau_weights``) and ``supg_weight`` scales the whole term, so a
    Galerkin baseline needs no rebuild.

    **What limits the timestep.** Nothing, for stability. The implicit
    scheme is stable at any cell Courant number, including on cells refined
    for a Stokes problem that the scalar does not need. Accuracy is set by
    how far the transported feature moves per step relative to its own
    width: the error grows as :math:`(\mathbf{u}\Delta t)^2` for the
    second-order schemes, and the transient term of :math:`\tau` cannot
    hide that. :meth:`estimate_dt` returns the cell-crossing time as a
    resolution guide only.

    Parameters
    ----------
    mesh : Mesh
    u_Field : MeshVariable
        Continuous scalar field :math:`\phi`.
    V_fn : MeshVariable or sympy Matrix
        Advecting velocity, ``(1, dim)``.
    order : int, default 1
        Order of the time integration, 1 to 3.
    integrator : {"bdf", "am"}, default "bdf"
    theta : float, default 1.0
        Adams-Moulton blend at order 1 only (0.5 is Crank-Nicolson). Must be
        1.0 for BDF and for Adams-Moulton above order 1.
    verbose : bool, default False
    DuDt : Eulerian, optional
        A pre-built history manager (order at least ``order``, no ``V_fn``).

    Notes
    -----
    The diffusivity is set through the constitutive model, as for the other
    scalar solvers; the solver starts with a
    :class:`~underworld3.constitutive_models.DiffusionModel` at
    :math:`\kappa = 0` (pure advection)::

        adv = uw.systems.AdvDiffusionSUPG(mesh, T, v.sym, order=2)
        adv.constitutive_model.Parameters.diffusivity = 1.0e-3
        adv.add_dirichlet_bc(0.0, "Left")
        adv.solve(timestep=dt)

    The linear system is nonsymmetric, so the preconditioner defaults are
    GMRES with an additive-Schwarz ILU preconditioner rather than the
    algebraic multigrid the symmetric scalar solvers use.
    """

    _INTEGRATORS = ("bdf", "am")

    @timing.routine_timer_decorator
    def __init__(
        self,
        mesh: uw.discretisation.Mesh,
        u_Field: uw.discretisation.MeshVariable,
        V_fn,
        order: int = 1,
        integrator: str = "bdf",
        theta: float = 1.0,
        verbose: bool = False,
        DuDt: Optional[Eulerian_DDt] = None,
    ):
        if not u_Field.continuous:
            raise ValueError(
                "u_Field must be a continuous MeshVariable: the SUPG weak form "
                "is continuous Galerkin."
            )
        if integrator not in self._INTEGRATORS:
            raise ValueError(
                f"integrator must be one of {self._INTEGRATORS}, not {integrator!r}."
            )
        order = int(order)
        if order not in (1, 2, 3):
            raise ValueError(f"order must be 1, 2 or 3, not {order}.")
        theta = float(theta)
        if theta != 1.0 and not (integrator == "am" and order == 1):
            raise ValueError(
                "theta applies to integrator='am' at order 1 only "
                "(0.5 is Crank-Nicolson); higher orders and BDF take theta=1."
            )

        super().__init__(mesh, u_Field, u_Field.degree, verbose, DuDt=DuDt, DFDt=None)

        self.f = sympy.Matrix.zeros(1, 1)
        self._integrator = integrator
        self._time_order = order
        self._theta = theta
        self._V_fn = _as_row_vector(V_fn, mesh.dim)

        tag = self.instance_number
        self._delta_t = public_expression(
            rf"\Delta t_{{{tag}}}", 1.0, "Eulerian advection-diffusion timestep")
        self._last_timestep = None

        # SUPG on/off and the three tau weights are runtime constants: the
        # compiled kernels read them from PETSc's constants[] array.
        self._supg_weight = public_expression(
            rf"w^{{\mathrm{{SUPG}}}}_{{{tag}}}", 1.0, "SUPG term weight (0 = Galerkin)")
        self._tau_weights = [
            public_expression(rf"C^{{\tau}}_{{t,{tag}}}", 2.0, "tau transient weight"),
            public_expression(rf"C^{{\tau}}_{{u,{tag}}}", 2.0, "tau advective weight"),
            public_expression(rf"C^{{\tau}}_{{\kappa,{tag}}}", 4.0, "tau diffusive weight"),
        ]

        if DuDt is None:
            self.Unknowns.DuDt = Eulerian_DDt(
                self.mesh,
                u_Field,
                vtype=uw.VarType.SCALAR,
                degree=u_Field.degree,
                continuous=u_Field.continuous,
                V_fn=None,
                theta=theta,
                varsymbol=u_Field.symbol,
                verbose=verbose,
                bcs=self.essential_bcs,
                order=order,
                smoothing=0.0,
            )
        else:
            if DuDt.order < order:
                raise ValueError(
                    f"DuDt supplied is order {DuDt.order} but order {order} was requested."
                )
            if getattr(DuDt, "V_fn", None) is not None:
                raise ValueError(
                    "DuDt must be built with V_fn=None: advection is assembled "
                    "implicitly by this solver, not as an explicit history correction."
                )
            self.Unknowns.DuDt = DuDt

        # Diffusivity lives on the constitutive model, as for every scalar
        # solver; kappa = 0 until the user sets it.
        self.constitutive_model = uw.constitutive_models.DiffusionModel
        self.constitutive_model.Parameters.diffusivity = 0.0

        # Nonsymmetric operator: opt out of the managed GAMG/FMG block and
        # use GMRES with an additive-Schwarz ILU preconditioner. RCM
        # ordering improves the ILU fill on convection-dominated operators.
        self._pc_option_prefix = None
        self.petsc_options["ksp_type"] = "gmres"
        self.petsc_options["ksp_gmres_restart"] = 200
        self.petsc_options["pc_type"] = "asm"
        self.petsc_options["sub_pc_type"] = "ilu"
        self.petsc_options["sub_pc_factor_mat_ordering_type"] = "rcm"
        self.petsc_options["snes_rtol"] = 1.0e-8
        self.petsc_options["snes_max_it"] = 20

    # ------------------------------------------------------------------
    # Scheme description
    # ------------------------------------------------------------------

    @property
    def integrator(self) -> str:
        """``"bdf"`` or ``"am"``."""
        return self._integrator

    @property
    def order(self) -> int:
        """Requested order of the time integration."""
        return self._time_order

    @property
    def theta(self) -> float:
        """Adams-Moulton blend at order 1 (1.0 backward Euler, 0.5 Crank-Nicolson)."""
        return self._theta

    @property
    def delta_t(self):
        r"""The timestep :math:`\Delta t` as a UW expression (set by :meth:`solve`)."""
        return self._delta_t

    @property
    def V_fn(self):
        """Advecting velocity, ``(1, dim)``."""
        return self._V_fn

    @V_fn.setter
    def V_fn(self, value):
        self._V_fn = _as_row_vector(value, self.mesh.dim)
        self.is_setup = False

    @property
    def f(self):
        """Volumetric source term."""
        return self._f

    @f.setter
    def f(self, value):
        self._f = sympy.Matrix((value,))
        self._needs_function_rewire = True

    @property
    def supg_weight(self) -> float:
        """Scale of the SUPG term: 1 (default) or 0 for plain Galerkin. No rebuild."""
        return float(self._supg_weight.sym)

    @supg_weight.setter
    def supg_weight(self, value):
        self._supg_weight.sym = float(value)

    @property
    def tau_weights(self):
        r"""The weights :math:`(C_t, C_u, C_\kappa)` of the three terms in :math:`\tau`."""
        return tuple(float(w.sym) for w in self._tau_weights)

    @tau_weights.setter
    def tau_weights(self, values):
        ct, cu, ck = (float(v) for v in values)
        self._tau_weights[0].sym = ct
        self._tau_weights[1].sym = cu
        self._tau_weights[2].sym = ck

    # ------------------------------------------------------------------
    # Residual pieces (raw field symbols only, so the Jacobian sees them)
    # ------------------------------------------------------------------

    def _states(self):
        r"""``[phi^{n+1}, phi^{n}, phi^{n-1}, ...]`` as scalar field symbols."""
        return [self.u.sym[0]] + [ps.sym[0] for ps in self.DuDt.psi_star]

    def _spatial_weights(self):
        """Weight of the spatial operator at each time level of ``_states``."""
        n = len(self.DuDt.psi_star)
        if self._integrator == "bdf":
            return [sympy.Integer(1)] + [sympy.Integer(0)] * n
        return self.DuDt.am_coefficient_expressions[: n + 1]

    def _time_derivative(self):
        if self._integrator == "bdf":
            return self.DuDt.bdf()[0] / self._delta_t
        phi_new, phi_old = self._states()[:2]
        return (phi_new - phi_old) / self._delta_t

    def _advection(self):
        dim = self.mesh.dim
        u = self._V_fn
        total = sympy.Integer(0)
        for w, phi in zip(self._spatial_weights(), self._states()):
            if w == 0:
                continue
            grad = self.mesh.vector.gradient(phi)
            total = total + w * sum(u[0, i] * grad[0, i] for i in range(dim))
        return total

    def _diffusive_flux(self):
        r"""``(1, dim)`` flux :math:`\sum_k w_k\,\nabla\phi^{(k)}\cdot\kappa` from the constitutive tensor."""
        dim = self.mesh.dim
        c = self.constitutive_model.c
        total = sympy.zeros(1, dim)
        for w, phi in zip(self._spatial_weights(), self._states()):
            if w == 0:
                continue
            grad = self.mesh.vector.gradient(phi)
            total = total + w * (grad * c)
        return total

    def _strong_residual(self):
        return self._time_derivative() + self._advection() - self._f[0]

    def _scalar_diffusivity(self):
        kappa = self.constitutive_model.Parameters.diffusivity
        if isinstance(kappa, sympy.MatrixBase):
            raise ValueError(
                "The SUPG parameter needs a scalar diffusivity; anisotropic "
                "diffusion is not supported by this solver."
            )
        return kappa

    def _tau(self):
        dim = self.mesh.dim
        u = self._V_fn
        u_mag2 = sum(u[0, i] ** 2 for i in range(dim))
        h = self.mesh.cell_size()
        kappa = self._scalar_diffusivity()
        if self._integrator == "bdf":
            c0 = self.DuDt.bdf_coefficient_expressions[0]
        else:
            c0 = sympy.Integer(1)
        ct, cu, ck = self._tau_weights
        transient = (ct * c0 / self._delta_t) ** 2
        advective = (cu * sympy.sqrt(u_mag2) / h) ** 2
        diffusive = (ck * kappa / h ** 2) ** 2
        return self._supg_weight / sympy.sqrt(transient + advective + diffusive + 1.0e-30)

    F0 = Template(
        r"f_0(\phi)",
        lambda self: sympy.Matrix([[self._strong_residual()]]),
        "Strong residual of the time scheme: time derivative, advection and source.",
    )
    F1 = Template(
        r"\mathbf{F}_1(\phi)",
        lambda self: self._diffusive_flux() + self._tau() * self._strong_residual() * self._V_fn,
        "Diffusive flux of the time scheme plus the SUPG flux tau R u.",
    )

    # ------------------------------------------------------------------
    # Timestep and solve
    # ------------------------------------------------------------------

    @timing.routine_timer_decorator
    def estimate_dt(self, direction_aware: bool = False, percentile: float = 0.0):
        r"""Cell-crossing timestep, as a resolution guide.

        The minimum over cells of :math:`h/|\mathbf{u}|` and
        :math:`h^2/\kappa`, exactly as for the semi-Lagrangian solver. It is
        not a stability limit for this scheme, and on a mesh refined for
        another problem it is far smaller than the timestep the transported
        field needs. Choose the timestep from the feature being transported:
        :math:`|\mathbf{u}|\Delta t` should be a fraction of its width.

        Parameters
        ----------
        direction_aware : bool, default False
            Use the per-cell extent along the local velocity.
        percentile : float, default 0.0
            Global percentile of the per-cell timesteps instead of the minimum.
        """
        dt_estimate, dt_adv, dt_diff = _advective_diffusive_dt(
            self.constitutive_model.K, self._V_fn, self.mesh,
            direction_aware=direction_aware, percentile=percentile)
        self.dt_adv = dt_adv if not np.isinf(dt_adv) else 0.0
        self.dt_diff = dt_diff if not np.isinf(dt_diff) else 0.0
        if np.isinf(dt_estimate):
            return np.inf
        return _dimensionalise_dt(dt_estimate)

    def solve(
        self,
        *,
        timestep=None,
        zero_init_guess: Optional[bool] = None,
        verbose: bool = False,
        _force_setup: bool = False,
        divergence_retries: int = 0,
    ):
        r"""Advance :math:`\phi` by one step of size ``timestep``.

        ``timestep`` is required and keyword-only. Changing it between calls
        updates a runtime constant of the compiled kernels; nothing is
        recompiled.
        """
        if timestep is None:
            raise ValueError(
                "solve() requires timestep=<dt>; there is no default timestep."
            )
        dt = float(_nondimensionalise_timestep(timestep))
        if dt <= 0.0:
            raise ValueError(f"timestep must be positive, not {dt}.")
        if dt != self._last_timestep:
            self._delta_t.sym = dt
            self._last_timestep = dt

        if _force_setup:
            self._needs_function_rewire = True
        if not self.constitutive_model._solver_is_setup:
            self._needs_function_rewire = True
        if not self.is_setup:
            self._setup_pointwise_functions(verbose)
            self._setup_discretisation(verbose)
            self._setup_solver(verbose)

        self.DuDt.update_pre_solve(dt, verbose=verbose)
        super().solve(zero_init_guess, _force_setup, divergence_retries=divergence_retries)
        _invalidate_solution_cache(self.u)
        self.DuDt.update_post_solve(dt, verbose=verbose)

        self.is_setup = True
        self.constitutive_model._solver_is_setup = True
