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

The SUPG weak form, the Petrov-Galerkin test-function perturbation written
as a flux so that PETSc needs no modified test space, and its first
implementation on PetscDS are NengLu's (issue #657, branch ``levelset``);
this module keeps that formulation and its stabilisation parameter.
"""

import warnings
import math
from dataclasses import dataclass

from petsc4py import PETSc
from underworld3.checkpoint.state import SnapshottableState

import numpy as np
import sympy
from typing import Callable, Optional, Union

import underworld3 as uw
import underworld3.timing as timing
from underworld3.systems import SNES_Scalar
from underworld3.utilities._api_tools import Template
from underworld3.function import expression as public_expression
from underworld3.systems.ddt import Eulerian as Eulerian_DDt
from underworld3.systems.solvers import (
    _advective_diffusive_dt,
    _centroid_velocities_nd,
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


@dataclass
class AdvDiffusionSUPGState(SnapshottableState):
    """Integrator metadata; fields and DDt history are captured separately."""

    time_integrator: str = "implicit"
    rate_initialised: bool = False
    last_timestep: Optional[float] = None
    last_change_rate: Optional[float] = None
    order: int = 1
    theta: float = 0.5
    adv_gamma: float = 0.5
    corrector_steps: int = 2


class SNES_AdvectionDiffusion_SUPG(SNES_Scalar):
    r"""Scalar transport with shared SUPG assembly and selectable time integration.

    .. math::
        \partial_t T + \mathbf{u}\cdot\nabla T
        - \nabla\cdot(\kappa\nabla T) = f.

    Velocity and material coefficients are frozen during each update.

    Parameters
    ----------
    mesh : Mesh
        Computational volume mesh.
    u_Field : MeshVariable
        Continuous scalar field.
    V_fn : MeshVariable or sympy Matrix
        Advecting velocity.
    order : int, default 1
        Implicit history order: 1, 2 or 3. Leave at 1 for CitcomS,
        which manages its own rate instead of DDt history.
    theta : float, optional
        At order 1, 0.5 selects Crank-Nicolson (implicit default) and 1
        backward Euler. Orders 2 and 3 require 1. Leave unset for CitcomS.
    time_integrator : {"implicit", "citcoms", "bdf"}, default "implicit"
        "implicit" selects CN/BE/BDF2/BDF3 through order and theta.
        "citcoms" selects the P1 lumped-mass predictor-corrector used by
        CitcomS-style mantle-convection benchmarks. "bdf" retains the
        previous BDF selection, including backward Euler at order 1.
    temperature_rate_field : MeshVariable, optional
        Separate continuous P1 field storing the CitcomS rate. A stable
        name such as Tdot is useful for field checkpoints. Created internally
        if omitted; not used by implicit integrators.
    adv_gamma : float, default 0.5
        CitcomS predictor/corrector weight, in (0, 1].
    corrector_steps : int, default 2
        Number of fixed CitcomS residual corrections.
    tau : scalar expression, optional
        Explicit stabilisation parameter; zero gives Galerkin transport.
    tau_model : {"generic", "citcoms"}, optional
        Defaults to the time integrator's model. Generic implicit transport
        uses the transient norm of time, advection and diffusion scales.
        CitcomS uses a clipped steady parameter on directional simplex
        lengths. Its automatic operations require triangles or tetrahedra.
    DuDt : Eulerian, optional
        Pre-built implicit history manager with V_fn=None. Not used by CitcomS.
    verbose : bool, default False
        Solver verbosity.
    restore_points_func, monotone_mode, old_frame_traceback, DFDt
        SLCN-only compatibility arguments, ignored with a warning for
        implicit transport. CitcomS rejects supplied history operators.

    Notes
    -----
    All methods share the same pointwise assembly:

    .. math::
        F_0 = R,\qquad
        \mathbf{F}_1 = \kappa\nabla T + \tau R\mathbf{u}.

    The implicit method's spatial history weights apply to diffusion and
    advection. Diffusion is omitted only from the strong SUPG residual.
    It vanishes identically for affine P1 elements with elementwise constant
    diffusivity; curved mappings, variable diffusivity and higher-order fields
    need separately validated flux-divergence recovery for full consistency.

    CitcomS predicts with (1-gamma)*dt*Tdot, resets the rate, then applies
    fixed corrections delta_rate=-M_L^-1*F to the rate and gamma*dt*delta_rate
    to temperature. Boundary values are reinserted at every correction.
    Its default timestep is 0.9*min(dt_adv, dt_diff), not the implicit
    field-change accuracy estimate.

    Implicit transport defaults to GMRES/ASM-ILU. preconditioner="fmg"
    selects geometric multigrid when a mesh hierarchy is available.
    CN may ring for under-resolved features; BDF3 may amplify pure advection.
    The field-change timestep estimate is an accuracy heuristic, not a
    guarantee of bounded temperature.

    The default Model's save_state() captures integrator metadata and all
    registered fields. Pass file=... for a persistent PETSc-backed snapshot.
    Restore into a matching model, mesh and integration method.

    Examples
    --------
    >>> thermal = uw.systems.AdvDiffusionSUPG(mesh, T, U.sym,
    ...     time_integrator="citcoms", temperature_rate_field=Tdot)
    >>> thermal.constitutive_model.Parameters.diffusivity = 1.0
    >>> thermal.solve(timestep=thermal.estimate_dt())
    """

    @timing.routine_timer_decorator
    def __init__(
        self,
        mesh: uw.discretisation.Mesh,
        u_Field: uw.discretisation.MeshVariable,
        V_fn,
        order: int = 1,
        theta: Optional[float] = None,
        verbose: bool = False,
        DuDt: Optional[Eulerian_DDt] = None,
        DFDt=None,
        restore_points_func: Optional[Callable] = None,
        monotone_mode: Optional[str] = None,
        old_frame_traceback: bool = False,
        *,
        time_integrator: str = "implicit",
        temperature_rate_field: Optional[uw.discretisation.MeshVariable] = None,
        adv_gamma: float = 0.5,
        corrector_steps: int = 2,
        tau=None,
        tau_model: Optional[str] = None,
    ):
        if not u_Field.continuous:
            raise ValueError(
                "u_Field must be a continuous MeshVariable: the SUPG weak form "
                "is continuous Galerkin."
            )
        if time_integrator not in ("implicit", "bdf", "citcoms"):
            raise ValueError("time_integrator must be 'implicit', 'bdf' or 'citcoms'.")
        if u_Field.num_components != 1:
            raise ValueError("u_Field must be scalar.")
        if mesh.dim != mesh.cdim:
            raise NotImplementedError("SUPG currently requires a volume mesh.")
        if time_integrator == "citcoms":
            if u_Field.degree != 1:
                raise ValueError("The CitcomS predictor-corrector requires continuous P1 temperature.")
            if order != 1 or (theta is not None and float(theta) != 1.0):
                raise ValueError("CitcomS uses gamma, not order/theta; leave order=1 and theta unset.")
            if DuDt is not None or DFDt is not None:
                raise ValueError("CitcomS manages its own derivative; do not supply DuDt or DFDt.")
            if not 0.0 < float(adv_gamma) <= 1.0:
                raise ValueError("adv_gamma must be in (0, 1].")
            if int(corrector_steps) != corrector_steps or corrector_steps < 1:
                raise ValueError("corrector_steps must be a positive integer.")
            if temperature_rate_field is not None and (
                temperature_rate_field is u_Field
                or temperature_rate_field.mesh is not mesh
                or temperature_rate_field.degree != 1
                or not temperature_rate_field.continuous
                or temperature_rate_field.num_components != 1
            ):
                raise ValueError("temperature_rate_field must be a separate continuous scalar P1 variable on the solver mesh.")
        elif temperature_rate_field is not None or adv_gamma != 0.5 or corrector_steps != 2:
            raise ValueError("temperature_rate_field, adv_gamma and corrector_steps configure CitcomS only.")
        if time_integrator in ("bdf", "citcoms"):
            if theta is not None and float(theta) != 1.0:
                raise ValueError("The bdf and citcoms modes require theta=1.0.")
            theta = 1.0
        if tau_model is None:
            tau_model = "citcoms" if time_integrator == "citcoms" else "generic"
        if tau_model not in ("generic", "citcoms"):
            raise ValueError("tau_model must be 'generic' or 'citcoms'.")
        if time_integrator == "citcoms" and tau_model != "citcoms":
            raise ValueError("CitcomS requires its steady tau model; supply tau for a custom value.")
        ignored = [name for name, value in (
            ("restore_points_func", restore_points_func),
            ("monotone_mode", monotone_mode),
            ("old_frame_traceback", old_frame_traceback),
            ("DFDt", DFDt),
        ) if value]
        if ignored:
            warnings.warn(
                f"AdvDiffusionSUPG ignores {', '.join(ignored)}: these configure "
                "the semi-Lagrangian trace-back and the Eulerian scheme has none.",
                stacklevel=2,
            )
        order = int(order)
        if order not in (1, 2, 3):
            raise ValueError(f"order must be 1, 2 or 3, not {order}.")
        # theta means what it means for the semi-Lagrangian solver: the
        # Crank-Nicolson blend at order 1. Left unset, order 2 and 3 take the
        # only consistent value; set explicitly to 0.5 there, it is refused.
        theta = float(theta) if theta is not None else (0.5 if order == 1 else 1.0)
        # The multistep family follows the order: the Adams-Moulton (theta)
        # rule at order 1, backward differentiation above. Adams-Moulton at
        # orders 2 and 3 is assembled by the same code but is not offered:
        # its bounded stability region blows up on an advection operator
        # from about Courant 1 (design note, integrator study).
        integrator = ("citcoms" if time_integrator == "citcoms" else
                      "bdf" if time_integrator == "bdf" or order > 1 else "am")
        if theta != 1.0 and order != 1:
            raise ValueError(
                "theta applies at order 1 only (0.5 is Crank-Nicolson, 1.0 is "
                "backward Euler); order 2 and 3 take theta=1.0, the same rule as "
                "the semi-Lagrangian solver (a BDF stencil pairs with terms at n+1, "
                "not with a centred flux)."
            )

        super().__init__(mesh, u_Field, u_Field.degree, verbose, DuDt=DuDt, DFDt=None)

        self.time_integrator = time_integrator
        self.tau_model = tau_model
        self.adv_gamma = float(adv_gamma)
        self.corrector_steps = int(corrector_steps)
        self.f = sympy.Matrix.zeros(1, 1)
        self._integrator = integrator
        self._time_order = order
        self._theta = theta
        self._V_fn = _as_row_vector(V_fn, mesh.dim)

        tag = self.instance_number
        self._delta_t = public_expression(
            rf"\Delta t_{{{tag}}}", 1.0, "Eulerian advection-diffusion timestep")
        self._last_timestep = None
        self._last_change_rate = None

        # SUPG on/off and the three tau weights are runtime constants: the
        # compiled kernels read them from PETSc's constants[] array.
        self._supg_weight = public_expression(
            rf"w^{{\mathrm{{SUPG}}}}_{{{tag}}}", 1.0, "SUPG term weight (0 = Galerkin)")
        self._tau_weights = [
            public_expression(rf"C^{{\tau}}_{{t,{tag}}}", 2.0, "tau transient weight"),
            public_expression(rf"C^{{\tau}}_{{u,{tag}}}", 2.0, "tau advective weight"),
            public_expression(rf"C^{{\tau}}_{{\kappa,{tag}}}", 4.0, "tau diffusive weight"),
        ]

        if time_integrator == "citcoms":
            self.Unknowns.DuDt = None
        elif DuDt is None:
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
            if not isinstance(DuDt, Eulerian_DDt):
                raise TypeError("DuDt must be an Eulerian history manager.")
            if DuDt.order < order:
                raise ValueError(
                    f"DuDt supplied is order {DuDt.order} but order {order} was requested."
                )
            if getattr(DuDt, "V_fn", None) is not None:
                raise ValueError(
                    "DuDt.V_fn must be None: advection is assembled "
                    "implicitly by this solver, not as an explicit history correction."
                )
            self.Unknowns.DuDt = DuDt

        # Diffusivity lives on the constitutive model, as for every scalar
        # solver; kappa = 0 until the user sets it.
        self.constitutive_model = uw.constitutive_models.DiffusionModel
        self.constitutive_model.Parameters.diffusivity = 0.0

        # Linear solver: additive-Schwarz ILU by default, the managed multigrid
        # block on request (see ``preconditioner``). One Newton iteration per
        # step: the operator is linear in phi, so the Krylov tolerance must
        # reach the SNES tolerance or the SNES takes a second step, and a
        # second Jacobian assembly costs more than every linear solve of the
        # step (design note, "Preconditioner").
        self._set_linear_solver(multigrid=False)
        self.petsc_options["snes_rtol"] = 1.0e-8
        self.petsc_options["ksp_rtol"] = 1.0e-9
        self.petsc_options["snes_max_it"] = 20

        self._tau_override = None if tau is None else sympy.sympify(tau)
        self._automatic_tau = tau is None and tau_model == "citcoms"
        self._supg_h = None
        self._supg_tau = None
        self._temperature_rate = None
        self._lumped_mass = None
        self._lumped_mass_mesh_version = None
        self._citcoms_work_vectors = None
        self._citcoms_work_mesh_version = None
        self._simplex_data_cache = None
        self._simplex_data_mesh_version = None
        self._directional_rate_work = None
        self._directional_rate_mesh_version = None
        self._diffusion_dt_cache = None
        self._rate_initialised = False
        if time_integrator == "citcoms":
            self._temperature_rate = temperature_rate_field
            if self._temperature_rate is None:
                self._temperature_rate = uw.discretisation.MeshVariable(
                    f"_supg_dTdt_{tag}", mesh, 1, degree=1, continuous=True)
        if self._automatic_tau:
            self._supg_h = uw.discretisation.MeshVariable(
                f"_supg_h_{tag}", mesh, 1, degree=0, continuous=False)
            self._supg_tau = uw.discretisation.MeshVariable(
                f"_supg_tau_{tag}", mesh, 1, degree=0, continuous=False)
        elif self._tau_override is None:
            # Generic tau needs this field on a fresh-process checkpoint restore,
            # before the first residual build would otherwise create it lazily.
            mesh.cell_size()
        uw.get_default_model()._register_state_bearer(self)

    # ------------------------------------------------------------------
    # Linear solver
    # ------------------------------------------------------------------

    _SCHWARZ_OPTIONS = {
        "ksp_type": "gmres",
        "ksp_gmres_restart": 200,
        "pc_type": "asm",
        "sub_pc_type": "ilu",
        # RCM ordering improves the ILU fill on a convection-dominated operator.
        "sub_pc_factor_mat_ordering_type": "rcm",
    }

    def _set_linear_solver(self, multigrid: bool):
        """Own the linear solver (GMRES + additive-Schwarz ILU) or hand it to
        the managed multigrid block.

        Measured on the level-set advection step at 256^2 and 512^2 (design
        note, "Preconditioner"): with the Krylov tolerance matched to the
        SNES tolerance, additive Schwarz with ILU is the cheaper linear solve
        at every Courant number from 1/2 to 32, its iteration count does not
        change between one and eight ranks, and the geometric multigrid's
        cycle count grows with the Courant number nearly as fast as the
        Schwarz iteration count while each cycle costs about three Schwarz
        iterations. The linear solve is under a tenth of the step either way;
        assembly is the rest. Multigrid keeps its coarse space for a rank
        count where a one-level method runs out of one, which is what
        ``preconditioner = "fmg"`` is for.
        """
        from underworld3.utilities import multigrid_options

        opts = self.petsc_options
        bundle_keys = set()
        for bundle in (multigrid_options.gamg_bundle(),
                       multigrid_options.geometric_mg_bundle()):
            bundle_keys |= set(bundle.settings) | set(bundle.stale)
        if multigrid:
            # The managed block starts from the scalar solver's own keys
            # (GMRES + the GAMG bundle) and _apply_preconditioner_options
            # resolves the request against the mesh hierarchy at build time.
            self._pc_option_prefix = ""
            for key in self._SCHWARZ_OPTIONS:
                opts.delValue(key)
            self._push_managed_option("ksp_type", "gmres")
            for key, value in multigrid_options.gamg_bundle().settings.items():
                self._push_managed_option(key, value)
        else:
            self._pc_option_prefix = None
            for key in bundle_keys | {"ksp_type"}:
                opts.delValue(key)
                self._managed_pc_options.pop(self.petsc_options_prefix + key, None)
            for key, value in self._SCHWARZ_OPTIONS.items():
                opts[key] = value

    @property
    def preconditioner(self):
        """Linear preconditioner: ``"auto"`` (default), ``"fmg"`` or ``"gamg"``.

        ``"auto"`` is GMRES with an additive-Schwarz ILU preconditioner, the
        measured choice for this operator (see :meth:`_set_linear_solver`).
        ``"fmg"`` hands the block to the managed geometric-multigrid route:
        custom-P transfers over the mesh's refinement hierarchy or an adapt
        child's coarse tail, installed on the live PC at the first solve,
        under a flexible GMRES outer solver; without a hierarchy it warns and
        degrades to GAMG. ``"gamg"`` is algebraic multigrid. Setting the
        property rebuilds the solver at the next solve.
        """
        return self._preconditioner

    @preconditioner.setter
    def preconditioner(self, value):
        SNES_Scalar.preconditioner.fset(self, value)
        self._set_linear_solver(multigrid=self._preconditioner != "auto")

    def _object_viewer(self):
        from IPython.display import Latex, display

        super()._object_viewer()
        scheme = {("am", 1): f"Adams-Moulton order 1, theta = {self._theta}",
                  ("bdf", 1): "backward Euler"}.get(
            (self._integrator, self._time_order),
            f"{self._integrator.upper()} order {self._time_order}")
        display(Latex(r"$\quad\mathrm{u} = $ " + self.u.sym._repr_latex_()))
        display(Latex(r"$\quad\mathbf{v} = $ " + self._V_fn._repr_latex_()))
        display(Latex(r"$\quad\Delta t = $ " + self._delta_t._repr_latex_()))
        display(Latex(rf"$\quad$ time scheme: {scheme}"))

    # ------------------------------------------------------------------
    # Scheme description
    # ------------------------------------------------------------------

    @property
    def integrator(self) -> str:
        """The multistep family in use: ``"am"`` (the theta rule) at order 1, ``"bdf"`` above."""
        return self._integrator

    @property
    def order(self) -> int:
        """Requested order of the time integration."""
        return self._time_order

    @property
    def theta(self) -> float:
        """Adams-Moulton blend at order 1 (1.0 backward Euler, 0.5 Crank-Nicolson).

        Settable after construction, as on the semi-Lagrangian solver: the
        blend is a runtime constant of the compiled kernels, refreshed from
        the history manager before every solve, so nothing is recompiled.
        """
        return self._theta

    @theta.setter
    def theta(self, value):
        value = float(value)
        if value != 1.0 and self._time_order != 1:
            raise ValueError(
                "theta applies at order 1 only (0.5 is Crank-Nicolson, 1.0 is "
                "backward Euler); order 2 and 3 take theta=1.0."
            )
        if self.time_integrator == "citcoms" and value != 1.0:
            raise ValueError("CitcomS uses adv_gamma, not theta.")
        self._theta = value
        if self.DuDt is not None:
            self.DuDt.theta = value

    @property
    def delta_t(self):
        r"""The timestep :math:`\Delta t` as a UW expression.

        Set by :meth:`solve`, or assign it directly (a number or a quantity
        with time units) and call ``solve()`` without ``timestep``, as with
        the semi-Lagrangian solver. A new value updates a runtime constant of
        the compiled kernels; nothing is recompiled.
        """
        return self._delta_t

    @delta_t.setter
    def delta_t(self, value):
        dt = float(_nondimensionalise_timestep(value))
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"timestep must be positive, not {dt}.")
        if dt != self._last_timestep:
            self._delta_t.sym = dt
            self._last_timestep = dt

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
        if self.time_integrator == "citcoms":
            return [self.u.sym[0]]
        return [self.u.sym[0]] + [ps.sym[0] for ps in self.DuDt.psi_star]

    def _spatial_weights(self):
        """Weight of the spatial operator at each time level of ``_states``."""
        if self.time_integrator == "citcoms":
            return [sympy.Integer(1)]
        n = len(self.DuDt.psi_star)
        if self._integrator == "bdf":
            return [sympy.Integer(1)] + [sympy.Integer(0)] * n
        return self.DuDt.am_coefficient_expressions[: n + 1]

    def _time_derivative(self):
        if self.time_integrator == "citcoms":
            return self._temperature_rate.sym[0]
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
        value = sympy.sympify(uw.function.unwrap(kappa))
        if value.is_number and (not np.isfinite(float(value)) or float(value) < 0.0):
            raise ValueError("SUPG diffusivity must be finite and non-negative.")
        return kappa

    @property
    def tau(self):
        """SUPG parameter used by the shared residual."""
        return self._tau()

    def _tau(self):
        if self._tau_override is not None:
            return self._supg_weight * self._tau_override
        if self._automatic_tau:
            return self._supg_weight * self._supg_tau.sym[0]
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
    def estimate_dt(self, fraction: float = 0.02, basis: Optional[str] = None,
                    direction_aware: bool = False, percentile: float = 0.0):
        r"""A timestep for this scheme, chosen for accuracy.

        The implicit scheme has no stability limit, so the cell-crossing time
        the semi-Lagrangian solver reports says nothing about how large a step
        this solver can take. What bounds the error is how much the field
        changes per step, and that is what the default estimate measures:

        .. math::
            \Delta t = f\,\frac{\max\phi - \min\phi}
                                {\max\left|\dot\phi\right|}

        with :math:`\dot\phi` the rate of change of the field. Before the
        first solve that rate is the advective one, :math:`|\mathbf{u}\cdot
        \nabla\phi|` at the mesh vertices; after a solve it is the rate the
        last step actually produced, :math:`|\phi^{n+1}-\phi^{n}|/\Delta t`,
        which includes diffusion and sources. The estimate is independent of
        the mesh, so a band of cells refined for another problem does not
        shrink it; it does shrink for a feature that is genuinely
        under-resolved, which is the honest answer.

        On the rotating Gaussian (``docs/developer/design/eulerian-supg-transport.md``)
        ``fraction=0.02`` gives Crank-Nicolson a round-trip error of a few
        tenths of a per cent after one revolution and BDF2 about 1.5%;
        ``fraction=0.07`` gives 2.5% and 9%.

        Parameters
        ----------
        fraction : float, default 0.02
            Allowed change of the field per step as a fraction of its range.
        basis : {"accuracy", "resolution"}
            ``"resolution"`` returns the cell-crossing / diffusion time the
            semi-Lagrangian solver's ``estimate_dt`` returns, for scripts that
            size the step in Courant numbers.
        direction_aware, percentile
            Forwarded to the resolution estimate; ignored otherwise.

        Returns
        -------
        pint.Quantity or float
            With physical time units if a model with reference scales is
            active, otherwise nondimensional. ``inf`` if nothing changes.
        """
        from mpi4py import MPI

        if self.time_integrator == "citcoms":
            if basis not in (None, "stability"):
                raise ValueError("CitcomS requires basis='stability', not an implicit accuracy estimate.")
            if fraction != 0.02 or direction_aware or percentile != 0.0:
                raise ValueError("CitcomS uses its fixed 0.9 stability factor and directional simplex length.")
            return _dimensionalise_dt(self._estimate_citcoms_dt())
        if basis is None:
            basis = "accuracy"
        if basis == "resolution":
            dt_estimate, dt_adv, dt_diff = _advective_diffusive_dt(
                self.constitutive_model.K, self._V_fn, self.mesh,
                direction_aware=direction_aware, percentile=percentile)
            self.dt_adv = dt_adv if not np.isinf(dt_adv) else 0.0
            self.dt_diff = dt_diff if not np.isinf(dt_diff) else 0.0
            if np.isinf(dt_estimate):
                return np.inf
            return _dimensionalise_dt(dt_estimate)
        if basis != "accuracy":
            raise ValueError(f"basis must be 'accuracy' or 'resolution', not {basis!r}.")

        comm = uw.mpi.comm
        values = np.asarray(self.u.array).reshape(-1)
        lo = comm.allreduce(float(values.min()) if values.size else np.inf, op=MPI.MIN)
        hi = comm.allreduce(float(values.max()) if values.size else -np.inf, op=MPI.MAX)
        field_range = hi - lo

        if self._last_change_rate is not None:
            rate = self._last_change_rate
        else:
            rate = self._advective_rate()
        self.dt_accuracy = fraction * field_range / rate if rate > 0.0 else np.inf
        if np.isinf(self.dt_accuracy) or field_range <= 0.0:
            return np.inf
        return _dimensionalise_dt(self.dt_accuracy)

    def _advective_rate(self):
        r"""Global maximum of :math:`|\mathbf{u}\cdot\nabla\phi|` at the mesh vertices.

        The gradient is the Clement recovery at the vertices (no point
        location, so it is safe on a mesh carrying many variables) and the
        velocity is evaluated at the same points.
        """
        from mpi4py import MPI
        from underworld3.function.gradient_evaluation import compute_clement_gradient_at_nodes

        coords = np.asarray(self.mesh.X.coords)
        n = coords.shape[0]
        if n:
            grad = np.asarray(compute_clement_gradient_at_nodes(self.u), dtype=float).reshape(n, -1)
            vel = uw.function.evaluate(self._V_fn, coords)
            vel = np.asarray(getattr(vel, "magnitude", vel), dtype=float).reshape(n, -1)
            local = float(np.abs((vel[:, :grad.shape[1]] * grad).sum(axis=1)).max())
        else:
            local = 0.0
        return uw.mpi.comm.allreduce(local, op=MPI.MAX)

    def solve(
        self,
        zero_init_guess: Optional[bool] = None,
        timestep=None,
        _force_setup: bool = False,
        _evalf: bool = False,
        verbose: bool = False,
        divergence_retries: int = 0,
    ):
        r"""Advance :math:`\phi` by one step.

        Same signature as the semi-Lagrangian solver. ``timestep`` sets
        :attr:`delta_t`; omit it to reuse the value already set. Changing it
        between calls updates a runtime constant of the compiled kernels;
        nothing is recompiled.
        """
        if timestep is not None:
            self.delta_t = timestep
        elif self._last_timestep is None:
            raise ValueError(
                "solve() needs a timestep: pass timestep=<dt> or set solver.delta_t first."
            )
        dt = self._last_timestep

        if _force_setup:
            self._needs_function_rewire = True
        if self.time_integrator == "citcoms":
            return self._solve_citcoms(dt, verbose=verbose)
        self._update_automatic_tau()
        if not self.constitutive_model._solver_is_setup:
            self._needs_function_rewire = True
        # The base ``_build`` resolves the preconditioner choice against the
        # mesh hierarchy before the SNES reads its options. Running the three
        # setup stages directly here (the semi-Lagrangian solvers' pattern)
        # marks the solver set up, so ``_build`` returned early and the
        # geometric-multigrid request was silently inert (#683).
        self._build(verbose)

        self.DuDt.update_pre_solve(dt, verbose=verbose)
        super().solve(zero_init_guess, _force_setup, divergence_retries=divergence_retries)
        _invalidate_solution_cache(self.u)
        # The realised rate of change of the field over this step feeds the
        # accuracy-based estimate_dt; psi_star[0] still holds phi^n here.
        from mpi4py import MPI
        change = np.abs(np.asarray(self.u.array).reshape(-1)
                        - np.asarray(self.DuDt.psi_star[0].array).reshape(-1))
        local = float(change.max()) if change.size else 0.0
        self._last_change_rate = uw.mpi.comm.allreduce(local, op=MPI.MAX) / dt
        self.DuDt.update_post_solve(dt, verbose=verbose)

        self.is_setup = True
        self.constitutive_model._solver_is_setup = True

    @property
    def temperature_rate(self):
        """Stored derivative for CitcomS, or None for an implicit method."""
        return self._temperature_rate

    @property
    def state(self):
        """Integrator metadata for snapshots; fields are captured by their mesh."""
        return AdvDiffusionSUPGState(
            time_integrator=self.time_integrator,
            rate_initialised=self._rate_initialised,
            last_timestep=self._last_timestep,
            last_change_rate=self._last_change_rate,
            order=self.order, theta=self.theta,
            adv_gamma=self.adv_gamma, corrector_steps=self.corrector_steps,
        )

    @state.setter
    def state(self, state):
        if not isinstance(state, AdvDiffusionSUPGState):
            raise TypeError("AdvDiffusionSUPG state has the wrong type.")
        if (state.time_integrator != self.time_integrator
                or state.order != self.order
                or state.adv_gamma != self.adv_gamma
                or state.corrector_steps != self.corrector_steps):
            raise ValueError("AdvDiffusionSUPG integration settings changed since snapshot.")
        self.theta = state.theta
        self._rate_initialised = bool(state.rate_initialised)
        self._last_timestep = None
        if state.last_timestep is not None:
            self.delta_t = state.last_timestep
        self._last_change_rate = state.last_change_rate

    def _simplex_data(self):
        """Return local simplex data; validate the layout collectively on rebuild."""
        from underworld3.meshing.smoothing import _tet_cells, _tri_cells

        mesh_version = getattr(self.mesh, "_mesh_version", 0)
        if (
            self._simplex_data_cache is not None
            and self._simplex_data_mesh_version == mesh_version
        ):
            return self._simplex_data_cache

        cells = (
            _tri_cells(self.mesh.dm)
            if self.mesh.dim == 2
            else _tet_cells(self.mesh.dm) if self.mesh.dim == 3 else None
        )
        cell_start, cell_end = self.mesh.dm.getHeightStratum(0)
        invalid = (
            (uw.mpi.rank, cell_end - cell_start, self.mesh.dim, self.mesh.cdim)
            if cells is None or self.mesh.dim != self.mesh.cdim else None
        )
        invalid_ranks = [item for item in uw.mpi.comm.allgather(invalid) if item is not None]
        if invalid_ranks:
            raise NotImplementedError(
                "Automatic CitcomS operations require a non-empty 2-D or 3-D "
                "volume simplex partition on every rank. Unsupported local "
                f"layouts (rank, cells, dim, cdim): {invalid_ranks}."
            )

        coords = np.asarray(self.mesh.X.coords)
        cell_coords = coords[cells]
        edges = cell_coords[:, 1:, :] - cell_coords[:, :1, :]
        try:
            inverse_edges = np.linalg.inv(edges)
        except np.linalg.LinAlgError as error:
            raise RuntimeError("Cannot operate on a singular simplex.") from error

        gradients = np.empty_like(cell_coords)
        gradients[:, 1:, :] = np.transpose(inverse_edges, (0, 2, 1))
        gradients[:, 0, :] = -gradients[:, 1:, :].sum(axis=1)
        volumes = np.abs(np.linalg.det(edges)) / math.factorial(self.mesh.dim)
        self._simplex_data_cache = (cells, gradients, volumes)
        self._simplex_data_mesh_version = mesh_version
        return self._simplex_data_cache

    def _streamline_directional_rate(self, gradients, velocity):
        """Return ``sum_a |u.grad(N_a)|`` using reusable cell work arrays."""
        mesh_version = getattr(self.mesh, "_mesh_version", 0)
        cell_count = velocity.shape[0]
        if (
            self._directional_rate_work is None
            or self._directional_rate_mesh_version != mesh_version
            or self._directional_rate_work[0].shape != (cell_count,)
        ):
            self._directional_rate_work = (
                np.empty(cell_count, dtype=float),
                np.empty(cell_count, dtype=float),
            )
            self._directional_rate_mesh_version = mesh_version

        directional_rate, projection = self._directional_rate_work
        directional_rate.fill(0.0)
        for basis_index in range(gradients.shape[1]):
            np.einsum(
                "cd,cd->c",
                gradients[:, basis_index, :],
                velocity,
                out=projection,
            )
            np.abs(projection, out=projection)
            np.add(directional_rate, projection, out=directional_rate)
        return directional_rate

    def _cell_diffusivity(self, cell_count):
        """Evaluate non-negative scalar diffusivity at cell centroids."""
        diffusivity_expr = sympy.sympify(self.constitutive_model.K)
        if isinstance(diffusivity_expr, sympy.MatrixBase):
            raise NotImplementedError(
                "Automatic SUPG operations require scalar isotropic "
                "diffusivity; supply tau explicitly for tensor diffusivity."
            )
        diffusivity = uw.function.evaluate(diffusivity_expr, self.mesh._centroids)
        if hasattr(diffusivity, "units") and diffusivity.units is not None:
            diffusivity = uw.non_dimensionalise(diffusivity)
        elif hasattr(diffusivity, "magnitude"):
            diffusivity = diffusivity.magnitude
        diffusivity = np.asarray(diffusivity, dtype=float).reshape(-1)
        if diffusivity.size == 1:
            diffusivity = np.full(cell_count, diffusivity.item())
        if diffusivity.shape != (cell_count,):
            raise ValueError("Diffusivity must evaluate to one scalar per cell.")
        if np.any(diffusivity < 0.0):
            raise ValueError("SUPG diffusivity must be non-negative.")
        return diffusivity

    def _update_automatic_tau(self):
        """Update local simplex streamline lengths and automatic tau values."""
        if not self._automatic_tau:
            if self._tau_override is None:
                self._scalar_diffusivity()
            return
        if self.constitutive_model is None:
            raise RuntimeError(
                "Set constitutive_model before solving AdvDiffusionSUPG."
            )

        _, gradients, _ = self._simplex_data()

        velocity = _centroid_velocities_nd(self.V_fn, self.mesh)
        speed = np.linalg.norm(velocity, axis=1)
        directional_rate = self._streamline_directional_rate(gradients, velocity)
        h_stream = np.divide(
            2.0 * speed,
            directional_rate,
            out=np.zeros_like(speed),
            where=directional_rate > 0.0,
        )

        diffusivity = self._cell_diffusivity(speed.size)

        tau_steady = np.zeros_like(speed)
        moving = speed > np.finfo(float).eps
        diffusive = moving & (diffusivity > 0.0)
        nondiffusive = moving & ~diffusive

        if np.any(diffusive):
            pe = speed[diffusive] * h_stream[diffusive] / (2.0 * diffusivity[diffusive])
            tau_steady[diffusive] = (
                h_stream[diffusive]
                * np.maximum(0.0, 1.0 - 1.0 / pe)
                / (2.0 * speed[diffusive])
            )
        tau_steady[nondiffusive] = h_stream[nondiffusive] / (2.0 * speed[nondiffusive])

        tau_values = tau_steady

        if self._supg_h.array.shape[0] != h_stream.size:
            raise RuntimeError("SUPG P0 field and local simplex counts do not match.")
        self._supg_h.array[:, 0, 0] = h_stream
        self._supg_tau.array[:, 0, 0] = tau_values

    def _setup_citcoms_residual(self, verbose=False):
        """Build the reusable residual assembler for predictor-corrector steps."""
        if not self.constitutive_model._solver_is_setup:
            self._needs_function_rewire = True
        self._build(verbose)
        self.is_setup = True
        self.constitutive_model._solver_is_setup = True

    def _assemble_lumped_mass(self):
        """Assemble positive P1 simplex row-sum masses on free global DOFs."""
        mesh_version = getattr(self.mesh, "_mesh_version", 0)
        if (
            self._lumped_mass is not None
            and self._lumped_mass_mesh_version == mesh_version
        ):
            return self._lumped_mass
        if self._lumped_mass is not None:
            self._lumped_mass.destroy()
            self._lumped_mass = None

        from underworld3.meshing.smoothing import _owned_cell_mask

        cells, _, volumes = self._simplex_data()
        owned = _owned_cell_mask(self.mesh.dm)

        local_mass = self.dm.createLocalVector()
        global_mass = self.dm.createGlobalVector()
        local_mass.set(0.0)
        global_mass.set(0.0)
        section = self.dm.getLocalSection()
        vertex_start, _ = self.mesh.dm.getDepthStratum(0)

        for cell_index in np.flatnonzero(owned):
            contribution = volumes[cell_index] / (self.mesh.dim + 1)
            for vertex_index in cells[cell_index]:
                offset = section.getOffset(vertex_start + int(vertex_index))
                if offset >= 0:
                    local_mass.array[offset] += contribution

        self.dm.localToGlobal(
            local_mass,
            global_mass,
            addv=PETSc.InsertMode.ADD_VALUES,
        )
        local_mass.destroy()
        if global_mass.getLocalSize() and np.any(global_mass.array <= 0.0):
            global_mass.destroy()
            raise RuntimeError("CitcomS P1 lumped mass contains non-positive rows.")

        self._lumped_mass = global_mass
        self._lumped_mass_mesh_version = mesh_version
        return self._lumped_mass

    def _citcoms_vectors(self):
        """Return reusable global vectors for predictor-corrector updates."""
        mesh_version = getattr(self.mesh, "_mesh_version", 0)
        if (
            self._citcoms_work_vectors is not None
            and self._citcoms_work_mesh_version == mesh_version
        ):
            return self._citcoms_work_vectors

        if self._citcoms_work_vectors is not None:
            for vector in self._citcoms_work_vectors:
                vector.destroy()

        solution = self.dm.createGlobalVector()
        residual = solution.duplicate()
        delta_rate = solution.duplicate()
        rate = solution.duplicate()
        self._citcoms_work_vectors = (solution, residual, delta_rate, rate)
        self._citcoms_work_mesh_version = mesh_version
        return self._citcoms_work_vectors

    @timing.routine_timer_decorator
    def _estimate_citcoms_dt(self):
        """Estimate a simplex advection-diffusion timestep.

        The CitcomS-compatible predictor-corrector uses
        ``0.9 * min(1/max(lambda_adv), 2/max(rowsum(abs(M_L^-1 K))))``.
        Generic implicit transport retains its separate Eulerian estimator.
        """
        from mpi4py import MPI
        from underworld3.meshing.smoothing import _owned_cell_mask

        cells, gradients, volumes = self._simplex_data()
        velocity = _centroid_velocities_nd(self.V_fn, self.mesh)
        directional_rate = self._streamline_directional_rate(gradients, velocity)
        local_adv_rate = (
            float(np.max(directional_rate)) if directional_rate.size else 0.0
        )
        adv_rate = uw.mpi.comm.allreduce(local_adv_rate, op=MPI.MAX)
        dt_adv = 1.0 / adv_rate if adv_rate > 0.0 else np.inf

        diffusivity = self._cell_diffusivity(len(cells))
        has_diffusivity = bool(
            uw.mpi.comm.allreduce(
                int(np.any(diffusivity > 0.0)),
                op=MPI.MAX,
            )
        )
        if not has_diffusivity:
            dt_diff = np.inf
        else:
            self._setup_citcoms_residual()
            mass = self._assemble_lumped_mass()
            diffusion_signature = (
                getattr(self.mesh, "_mesh_version", 0),
                hash(diffusivity.tobytes()),
            )
            local_cache_valid = (
                self._diffusion_dt_cache is not None
                and self._diffusion_dt_cache[0] == diffusion_signature
            )
            cache_valid = bool(
                uw.mpi.comm.allreduce(int(local_cache_valid), op=MPI.MIN)
            )
            if cache_valid:
                dt_diff = self._diffusion_dt_cache[1]
                self.dt_adv = dt_adv
                self.dt_diff = dt_diff
                return 0.9 * min(dt_adv, dt_diff)

            stiffness = self.dm.createMatrix()
            stiffness.setOption(PETSc.Mat.Option.NEW_NONZERO_LOCATION_ERR, False)
            section = self.dm.getLocalSection()
            vertex_start, _ = self.mesh.dm.getDepthStratum(0)
            owned = _owned_cell_mask(self.mesh.dm)

            for cell_index in np.flatnonzero(owned):
                points = [vertex_start + int(index) for index in cells[cell_index]]
                local_dofs = [section.getOffset(point) for point in points]
                element_stiffness = (
                    diffusivity[cell_index]
                    * volumes[cell_index]
                    * gradients[cell_index].dot(gradients[cell_index].T)
                )
                stiffness.setValuesLocal(
                    local_dofs,
                    local_dofs,
                    element_stiffness,
                    addv=PETSc.InsertMode.ADD_VALUES,
                )
            stiffness.assemble()

            row_start, row_end = stiffness.getOwnershipRange()
            local_diff_rate = 0.0
            for row in range(row_start, row_end):
                _, values = stiffness.getRow(row)
                row_sum = float(np.sum(np.abs(values)))
                local_diff_rate = max(
                    local_diff_rate,
                    row_sum / mass.array[row - row_start],
                )
            diff_rate = uw.mpi.comm.allreduce(local_diff_rate, op=MPI.MAX)
            stiffness.destroy()
            dt_diff = 2.0 / diff_rate if diff_rate > 0.0 else np.inf
            self._diffusion_dt_cache = (diffusion_signature, dt_diff)

        self.dt_adv = dt_adv
        self.dt_diff = dt_diff
        return 0.9 * min(dt_adv, dt_diff)

    def _compute_citcoms_residual(self, solution=None, residual=None):
        """Assemble the residual at the current temperature and rate."""
        if solution is None:
            solution = self.dm.createGlobalVector()
        if residual is None:
            residual = solution.duplicate()
        solution.set(0.0)
        self.dm.localToGlobal(self.u.vec, solution, addv=False)
        residual.set(0.0)
        self.mesh.update_lvec()
        self.dm.setAuxiliaryVec(self.mesh.lvec, None)
        self._update_constants()
        self.snes.computeFunction(solution, residual)
        return solution, residual

    def _solve_citcoms(self, timestep, verbose=False):
        """Advance one CitcomS-compatible predictor-corrector timestep."""
        if timestep is None:
            timestep = float(self.delta_t.data)
        self.delta_t = timestep
        dt = float(self.delta_t.data)
        if dt <= 0.0:
            raise ValueError("AdvDiffusionSUPG requires a positive timestep.")

        self._update_automatic_tau()
        self._setup_citcoms_residual(verbose)
        mass = self._assemble_lumped_mass()
        temperature_global, residual, delta_rate, rate_global = self._citcoms_vectors()

        if not self._rate_initialised:
            self._temperature_rate.array[:, 0, 0] = 0.0
            self._compute_citcoms_residual(temperature_global, residual)
            delta_rate.pointwiseDivide(residual, mass)
            delta_rate.scale(-1.0)
            self._temperature_rate.vec.set(0.0)
            self.dm.globalToLocal(delta_rate, self._temperature_rate.vec)
            self.mesh._stale_lvec = True
            self._rate_initialised = True

        self.u.array[:, 0, 0] += (
            (1.0 - self.adv_gamma) * dt * self._temperature_rate.array[:, 0, 0]
        )
        self._temperature_rate.array[:, 0, 0] = 0.0
        self.mesh._stale_lvec = True

        from underworld3.cython.petsc_discretisation import (
            petsc_dm_insert_boundary_values,
        )

        for _ in range(self.corrector_steps):
            self._compute_citcoms_residual(temperature_global, residual)
            delta_rate.pointwiseDivide(residual, mass)
            delta_rate.scale(-1.0)

            rate_global.set(0.0)
            self.dm.localToGlobal(self._temperature_rate.vec, rate_global, addv=False)
            rate_global.axpy(1.0, delta_rate)
            temperature_global.axpy(self.adv_gamma * dt, delta_rate)

            self._temperature_rate.vec.set(0.0)
            self.u.vec.set(0.0)
            self.dm.globalToLocal(rate_global, self._temperature_rate.vec)
            self.dm.globalToLocal(temperature_global, self.u.vec)
            petsc_dm_insert_boundary_values(self.dm, self.u.vec)
            self.mesh._stale_lvec = True

        _invalidate_solution_cache(self.u)
        _invalidate_solution_cache(self._temperature_rate)
        self.is_setup = True
        self.constitutive_model._solver_is_setup = True
        return
