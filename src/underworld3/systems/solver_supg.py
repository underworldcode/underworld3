import sympy
from sympy import sympify
import numpy as np
import warnings

from typing import Optional, Callable, Union

import underworld3 as uw
from underworld3.systems import SNES_Scalar, SNES_Vector, SNES_Stokes_SaddlePt
from underworld3.cython.generic_solvers import SNES_MultiComponent
from underworld3 import VarType
import underworld3.timing as timing
from underworld3.utilities import memprobe
from underworld3.utilities._api_tools import (
    uw_object,
    SymbolicProperty,
    Parameter,
    Template,
    ExpressionProperty,
)

from underworld3.function import expression as public_expression

# estimate_dt() below needs two module-level helpers that live alongside
# SNES_AdvectionDiffusion (the SLCN solver) in underworld3/systems/solvers.py.
# They are prefixed with an underscore (module-private by convention) but
# not otherwise protected, so a direct import works fine.
from underworld3.systems.solvers import (
    _global_max_diffusivity,
    _centroid_velocities_nd,
)

def _as_row_vector(V_fn, dim):
    r"""Coerce a velocity expression into a ``(1, dim)`` sympy row-vector
    Matrix matching the mesh's dimensionality, or raise a CLEAR error at
    construction time instead of the cryptic sympy ``IndexError`` that
    would otherwise surface much later, deep inside a compiled residual
    lambda (``u[0, i]`` for ``i in range(dim)`` on a too-narrow matrix --
    e.g. a scalar/1-component velocity on a 2-D mesh).

    A common source of this mismatch: a "1D" test built on a thin 2-D
    strip mesh (``mesh.dim == 2``) with a genuinely 1-component velocity
    field/expression -- UW3 doesn't have a bare 1-D mesh type for this,
    so the velocity still needs an explicit second (zero) component,
    e.g. ``sympy.Matrix([[vx, 0]])``, not a plain scalar ``vx``.
    """
    if isinstance(V_fn, sympy.MatrixBase):
        if V_fn.shape == (1, dim):
            return V_fn
        if V_fn.shape == (dim, 1):
            # Common transpose slip (column vector instead of row).
            return V_fn.T
        raise ValueError(
            f"V_fn has shape {V_fn.shape}, but the mesh is {dim}-D -- "
            f"expected a (1, {dim}) row vector (e.g. `v.sym` from a "
            f"{dim}-component vector MeshVariable). If this is meant to "
            f"be a 1-D-in-x flow on a thin {dim}-D mesh, pass an explicit "
            f"{dim}-component vector, e.g. `sympy.Matrix([[vx, 0]])` for "
            f"dim=2, not a bare scalar or a mismatched-shape Matrix."
        )
    # Plain scalar (python number or bare sympy scalar expression, not
    # wrapped in a Matrix at all).
    if dim == 1:
        return sympy.Matrix([[V_fn]])
    raise ValueError(
        f"V_fn is a scalar, but the mesh is {dim}-D -- expected a "
        f"(1, {dim}) row vector. If this is meant to be a 1-D-in-x flow "
        f"on a thin {dim}-D mesh, pass e.g. `sympy.Matrix([[V_fn, 0]])` "
        f"(dim=2) rather than the bare scalar `V_fn`."
    )


class SNES_AdvectionDiffusion_SUPG(SNES_Scalar):
    r"""Advection-diffusion equation solver using Crank-Nicolson time integration + 
    streamline-upwind Petrov-Galerkin (SUPG, Brooks & Hughes 1982) stabilisation:
    
    .. math::
        \frac{\partial \phi}{\partial t} + \mathbf{u}\cdot\nabla\phi
            - \nabla\cdot(\kappa\nabla\phi) = 0,

    Diffusivity :math:`\kappa` defaults to ``0.0`` -- pure advection,
    identical to the original single-purpose advection-only version of
    this class. Set :attr:`diffusivity` (a plain number or a
    symbolic/field expression) to solve advection-diffusion instead; no
    other API changes are needed and :meth:`solve` is unchanged either way.

    Weak form
    ---------
    Writing the ADVECTION-ONLY strong-form residual at the
    Crank-Nicolson-averaged state

    .. math::
        R = \frac{\phi^{n+1}-\phi^{n}}{\Delta t}
            + \mathbf{u}\cdot\nabla\phi_{CN},
        \qquad
        \phi_{CN} = \theta\,\phi^{n+1} + (1-\theta)\,\phi^{n}
        \quad(\theta=0.5 \Rightarrow \text{Crank-Nicolson}),

    the SUPG-perturbed test function :math:`w + \tau\,\mathbf{u}\cdot\nabla w`
    gives the weak form actually assembled here (UW3's residual convention
    :math:`\int_\Omega (w F_0 + \nabla w \cdot \mathbf{F}_1)\,d\Omega = 0`):

    .. math::
        F_0 = R,
        \qquad
        \mathbf{F}_1 = \kappa\,\nabla\phi_{CN} + \tau\,R\,\mathbf{u}.

    The diffusive term enters ONLY through :math:`\mathbf{F}_1`, as a
    standard consistent Galerkin flux -- exactly how ``SNES_Poisson``
    builds :math:`\kappa\nabla u` -- never as a literal second derivative
    inside :math:`F_0` (which the pointwise-residual PETSc API cannot
    express directly; :math:`\int_\Omega w\,[-\nabla\cdot(\kappa\nabla\phi)]
    \,d\Omega = \int_\Omega \nabla w\cdot\kappa\nabla\phi\,d\Omega` after
    integration by parts, with the boundary term dropped -- i.e. a natural
    zero-flux/Neumann condition on any boundary without an explicit
    Dirichlet BC). Diffusion is elliptic/self-adjoint and does not need
    Petrov-Galerkin stabilisation, so :math:`R` -- and hence the SUPG term
    :math:`\tau R\mathbf{u}` -- stays advection-only regardless of
    :math:`\kappa`; this is standard SUPG practice, not a simplification.

    :math:`\tau` is the Tezduyar-style inverse-norm stabilisation
    parameter, safely regularised for :math:`\Delta t\to0`,
    :math:`|\mathbf u|\to0` AND (now) :math:`\kappa\to0`:

    .. math::
        \tau = \left[ \left(\frac{2}{\Delta t}\right)^2
                     + \left(\frac{2|\mathbf u|}{h}\right)^2
                     + \left(\frac{4\kappa}{h^2}\right)^2
               \right]^{-1/2},

    with :math:`h` = ``mesh.cell_size()`` (UW3's per-cell characteristic
    length). The extra diffusive term keeps :math:`\tau` (and hence the
    SUPG correction) from over-stabilising a diffusion-dominated (low
    element-Peclet-number) problem, where diffusion's own ellipticity
    already provides stability; it vanishes identically at
    :math:`\kappa=0`, recovering the pure-advection :math:`\tau` unchanged.

    A separate ``phi_old`` MeshVariable carries :math:`\phi^{n}`, updated
    by :meth:`solve` before each call -- there is no ``DuDt``/
    ``SemiLagrangian``/``Lagrangian`` time-derivative handler involved
    anywhere in this class.

    Parameters
    ----------
    mesh : Mesh
        The computational mesh.
    u_Field : MeshVariable
        Scalar field :math:`\phi` being advected (and, optionally,
        diffused). Must be continuous (shared vertex/edge DOFs) -- SUPG
        assembles a continuous-Galerkin weak form.
    V_fn : MeshVariable.sym or sympy expression
        Advecting velocity :math:`\mathbf{u}`.
    theta : float, optional
        Crank-Nicolson blend (default 0.5); 1.0 is backward-Euler.
    diffusivity : float or sympy expression, optional
        Diffusivity :math:`\kappa` (default ``0.0`` -- pure advection). A
        plain number is captured as a literal inside the compiled F0/F1
        kernels, exactly like ``dt`` -- re-assign :attr:`diffusivity` to
        change it later (this forces a rebuild, see the setter). A
        symbolic/field expression (e.g. another MeshVariable's ``.sym``)
        already updates its own live value with no rebuild needed.
    discontinuity_capturing : bool, optional
        Add a crosswind discontinuity-capturing (DC) term to F1 (default
        ``False``, i.e. plain streamline-only SUPG). Pure SUPG has no
        mechanism to damp oscillations ACROSS a steep, under-resolved
        front -- this appears as trailing Gibbs-like ringing behind a
        translating front, REGARDLESS of diffusivity (it happens even
        at diffusivity=0). Turn this on if you see that. See
        :meth:`_dc_flux` for the full rationale/formula.
    dc_coefficient : float, optional
        Discontinuity-capturing strength :math:`C_{dc}` (default
        ``1.0``); only matters when ``discontinuity_capturing=True``.
    dc_streamwise_weight : float, optional
        How much of the ALONG-FLOW component of :math:`\nabla\phi` the
        DC flux includes, in ``[0, 1]`` (default ``0.0``). ``0.0`` is
        pure crosswind (textbook Hughes-Mallet -- correct for genuinely
        multi-D fronts, where the streamline SUPG term already handles
        the along-flow direction, but goes essentially INERT for a front
        varying only along the flow, e.g. 1D-in-x on a thin 2D strip
        mesh with no y-variation). ``1.0`` is the full gradient, which
        DOES engage on such a front but now double-counts diffusion in
        the same direction SUPG already stabilises -- expect visible
        peak-amplitude loss alongside the ripple suppression. Intermediate
        values (e.g. ``0.2-0.5``) trade between the two: sweep this
        (and/or ``dc_coefficient``) to find the smallest combination that
        still suppresses ringing without eating into the peak.
    verbose : bool, optional
        Enable verbose SNES output.

    Examples
    --------
    Pure advection (identical behaviour to the original single-purpose
    class this generalises):

    >>> adv = SNES_AdvectionDiffusion_SUPG(mesh, phi, v.sym)
    >>> adv.solve(timestep=1e-3)

    Advection-diffusion:

    >>> adv = SNES_AdvectionDiffusion_SUPG(mesh, phi, v.sym, diffusivity=1.0e-4)
    >>> adv.solve(timestep=1e-3)
    >>> adv.diffusivity = 2.0e-4   # change later; rebuilds automatically
    >>> adv.solve(timestep=1e-3)

    Steep-front advection with trailing-ripple suppression:

    >>> adv = SNES_AdvectionDiffusion_SUPG(mesh, phi, v.sym,
    ...                                     discontinuity_capturing=True)
    >>> adv.solve(timestep=1e-3)
    """

    @timing.routine_timer_decorator
    def __init__(
        self,
        mesh: uw.discretisation.Mesh,
        u_Field: uw.discretisation.MeshVariable,
        V_fn,
        theta: float = 0.5,
        diffusivity=0.0,
        discontinuity_capturing: bool = False,
        dc_coefficient: float = 1.0,
        dc_streamwise_weight: float = 0.0,
        verbose: bool = False,
    ):
        if not u_Field.continuous:
            raise ValueError(
                "`u_Field` must be a CONTINUOUS MeshVariable -- SUPGAdvection "
                "assembles a continuous-Galerkin weak form and needs shared "
                "vertex/edge DOFs across cells."
            )

        super().__init__(mesh, u_Field, degree=u_Field.degree, verbose=verbose)

        self._constitutive_model = uw.constitutive_models.Constitutive_Model(self.Unknowns)

        if isinstance(V_fn, uw.discretisation.MeshVariable):
            V_fn = V_fn.sym
        self._V_fn = _as_row_vector(V_fn, mesh.dim)
        self.theta_cn = float(theta)

        self.phi_old = uw.discretisation.MeshVariable(
            rf"\phi^{{n}}_{{{id(self)}}}",
            mesh, 1, degree=u_Field.degree, continuous=u_Field.continuous,
        )
        self.phi_old.data[:, 0] = u_Field.data[:, 0]

        self._dt_value = 1.0
        self._last_dt = None

        self._diffusivity = diffusivity
        self._last_diffusivity = None

        # Discontinuity-capturing (DC) term: OFF by default, so nothing
        # changes for existing callers. See _dc_flux() for the rationale
        # -- pure streamline SUPG has no crosswind damping, so a steep,
        # under-resolved front can ring (Gibbs-like oscillations) even
        # with diffusivity=0. This is additive to F1, not a separate
        # code path: discontinuity_capturing=False makes _dc_flux()
        # symbolically zero, exactly like diffusivity=0 recovers pure
        # advection through the SAME F1 expression rather than a branch.
        self._discontinuity_capturing = bool(discontinuity_capturing)
        self._dc_coefficient = float(dc_coefficient)
        # dc_streamwise_weight=0.0 (default) restricts the DC flux to the
        # component of grad(phi) ORTHOGONAL to the flow, on the reasoning
        # that tau*R*u already handles the along-flow direction -- this
        # is the textbook Hughes-Mallet formulation and is the right
        # choice for genuinely multi-D fronts. But it goes essentially
        # INERT for a problem where phi varies (near-)only ALONG the
        # flow direction (e.g. a 1D-in-x front on a thin 2D strip mesh,
        # with no y-variation): there, grad(phi) is already ~parallel to
        # u, so the crosswind component ~0 and DC contributes nothing,
        # regardless of dc_coefficient. Set this closer to 1.0 to include
        # more of the along-flow component (accepting some double-
        # counting with the streamline term in exchange for DC actually
        # engaging) -- a continuous knob rather than an all-or-nothing
        # switch, since dc_streamwise_weight=1.0 (full gradient) visibly
        # eats into peak amplitude alongside suppressing ripples.
        self._dc_streamwise_weight = float(np.clip(dc_streamwise_weight, 0.0, 1.0))

        self.petsc_options["snes_rtol"] = 1.0e-8
        self.petsc_options["snes_max_it"] = 20

        # KSP/PC defaults for a genuinely non-symmetric operator: unlike
        # SLCN (whose SemiLagrangian trace-back leaves a much more
        # diffusion/Poisson-like, closer-to-SPD system after each step),
        # this class solves the FULL SUPG-stabilised convection-diffusion
        # operator directly every step -- at high element Peclet number
        # (advection-dominated) that operator is strongly non-symmetric
        # and non-normal.
        #
        # PETSc's bare defaults (GMRES, restart=30) commonly STAGNATE on
        # exactly this kind of operator: it can land back in essentially
        # the same Krylov subspace every 30 iterations, producing a
        # residual that's bit-identical for thousands of iterations
        # rather than slowly decaying -- easy to mistake for "just needs
        # more iterations" when it actually needs a bigger subspace
        # and/or a preconditioner that respects the non-symmetry. GAMG
        # (algebraic multigrid) is the wrong FAMILY here for the same
        # reason: its classical smoothed-aggregation coarsening and
        # default Chebyshev/SOR smoothers assume a near-SPD operator
        # (elasticity/Poisson) and silently misbehave on this one rather
        # than failing loudly.
        #
        # ASM+ILU with a larger GMRES restart is the standard robust
        # choice for non-symmetric SUPG systems at small-to-moderate
        # scale; RCM reordering measurably helps ILU's fill-in quality on
        # convection-dominated operators specifically. None of this is
        # exact (unlike direct LU, which IS exact and fine to use instead
        # while your problem stays small/test-scale -- just won't scale
        # to a production-size mesh). All overridable after construction,
        # e.g. `adv_diff.petsc_options.setValue('pc_type', 'lu')`.
        self.petsc_options["ksp_type"] = "gmres"
        self.petsc_options["ksp_gmres_restart"] = 200
        self.petsc_options["pc_type"] = "asm"
        self.petsc_options["sub_pc_type"] = "ilu"
        self.petsc_options["sub_pc_factor_mat_ordering_type"] = "rcm"

    @property
    def diffusivity(self):
        r"""Diffusivity :math:`\kappa`. ``0.0`` (default) is pure
        advection. Re-assigning a genuinely different value forces a
        kernel rebuild on the next :meth:`solve` -- see the class
        docstring."""
        return self._diffusivity

    @diffusivity.setter
    def diffusivity(self, value):
        # Same reasoning as the `dt`-change guard in solve(): a plain
        # number is captured as a literal inside the compiled F0/F1
        # kernels (via _tau() and F1's sympy expression), so a genuine
        # VALUE change needs those kernels re-evaluated and rebuilt. A
        # symbolic/field expression (e.g. a MeshVariable.sym) already
        # updates its own live value with no rebuild -- the isclose()
        # check below can't meaningfully compare those, so it
        # conservatively treats a non-numeric value as "changed".
        try:
            unchanged = np.isclose(
                float(self._diffusivity), float(value), rtol=1e-12, atol=1e-15)
        except (TypeError, ValueError):
            unchanged = False
        self._diffusivity = value
        if not unchanged:
            self.is_setup = False

    @property
    def discontinuity_capturing(self):
        """Whether the crosswind discontinuity-capturing (DC) term is
        added to F1 -- see :meth:`_dc_flux`. ``False`` (default) is
        plain streamline-only SUPG, unchanged from before this feature
        existed. Re-assigning forces a kernel rebuild."""
        return self._discontinuity_capturing

    @discontinuity_capturing.setter
    def discontinuity_capturing(self, value):
        value = bool(value)
        if value != self._discontinuity_capturing:
            self._discontinuity_capturing = value
            self.is_setup = False

    @property
    def dc_coefficient(self):
        r"""Discontinuity-capturing coefficient :math:`C_{dc}` (default
        ``1.0``) -- only matters when :attr:`discontinuity_capturing` is
        True. Larger values damp front-adjacent ringing more aggressively
        but smear the front more; there's no universal "correct" value,
        it's a per-problem tuning knob (literature values commonly range
        ~0.5-2.0). Re-assigning forces a kernel rebuild."""
        return self._dc_coefficient

    @dc_coefficient.setter
    def dc_coefficient(self, value):
        try:
            unchanged = np.isclose(
                float(self._dc_coefficient), float(value), rtol=1e-12, atol=1e-15)
        except (TypeError, ValueError):
            unchanged = False
        self._dc_coefficient = float(value)
        if not unchanged:
            self.is_setup = False

    @property
    def dc_streamwise_weight(self):
        """How much of the along-flow component of grad(phi) the DC
        flux includes, in [0, 1] (default 0.0 -- pure crosswind, the
        textbook choice, but inert for a front varying only along the
        flow -- see the constructor docstring). 1.0 is the full gradient
        (double-counts with the streamline SUPG term; expect peak-
        amplitude loss). Sweep this alongside dc_coefficient to find the
        smallest combination that suppresses ringing without eating
        into the peak. Re-assigning forces a kernel rebuild."""
        return self._dc_streamwise_weight

    @dc_streamwise_weight.setter
    def dc_streamwise_weight(self, value):
        value = float(np.clip(value, 0.0, 1.0))
        try:
            unchanged = np.isclose(
                self._dc_streamwise_weight, value, rtol=1e-12, atol=1e-15)
        except (TypeError, ValueError):
            unchanged = False
        self._dc_streamwise_weight = value
        if not unchanged:
            self.is_setup = False

    def _sync_diffusivity_from_constitutive_model(self):
        """Compatibility bridge for the SLCN-solver idiom
        ``adv_diff.constitutive_model = uw.constitutive_models.DiffusionModel;
        adv_diff.constitutive_model.Parameters.diffusivity = X``.

        F0/F1 on THIS class are hand-written and never read
        ``constitutive_model.flux``/``.K`` (see class docstring) -- the
        ``constitutive_model`` attribute otherwise exists only as a
        non-None placeholder the base class expects. Without this
        bridge, that (very natural, SLCN-idiomatic) assignment is a
        SILENT no-op: no error, but the residual keeps using whatever
        :attr:`diffusivity` already was (0.0/pure-advection by default),
        which is a dangerous trap -- it can produce a well-posed-looking
        script that's actually solving the wrong PDE, or (as with
        Dirichlet BCs on both ends of a would-be diffusive problem) an
        ill-posed one that fails opaquely deep inside the linear solve.

        Best-effort: swallows anything unexpected about the constitutive
        model's shape (it's a bridge for an API this class doesn't own),
        and only overrides :attr:`diffusivity` when it finds a genuinely
        different value to adopt.
        """
        cm = getattr(self, "_constitutive_model", None)
        if cm is None:
            return
        try:
            cm_kappa = cm.Parameters.diffusivity
        except AttributeError:
            return
        if cm_kappa is None:
            return
        # Unwrap a UWexpression-like Parameter to its underlying symbol/value.
        cm_kappa_val = getattr(cm_kappa, "sym", cm_kappa)
        try:
            unchanged = np.isclose(
                float(self._diffusivity), float(cm_kappa_val),
                rtol=1e-12, atol=1e-15)
        except (TypeError, ValueError):
            unchanged = False
        if not unchanged:
            warnings.warn(
                "SNES_AdvectionDiffusion_SUPG: adopting diffusivity="
                f"{cm_kappa_val} from constitutive_model.Parameters.diffusivity "
                "(this class's F0/F1 don't read the constitutive model "
                "directly -- set `adv_diff.diffusivity = ...` instead to "
                "avoid relying on this compatibility bridge).",
                stacklevel=2,
            )
            self.diffusivity = cm_kappa_val

    def _tau(self):
        """Tezduyar-style advection-diffusion SUPG parameter. 
        The diffusive term vanishes identically at
        diffusivity=0, recovering the pure-advection tau unchanged."""
        dim = self.mesh.dim
        u = self._V_fn
        u_mag2 = sum(u[0, i] ** 2 for i in range(dim))
        h = self.mesh.cell_size()
        inv_dt_term = (2.0 / self._dt_value) ** 2
        inv_h_term = (2.0 * sympy.sqrt(u_mag2) / h) ** 2
        inv_diff_term = (4.0 * self._diffusivity / h**2) ** 2
        return 1.0 / sympy.sqrt(inv_dt_term + inv_h_term + inv_diff_term + 1.0e-30)

    def _phi_cn(self):
        """Crank-Nicolson blended state phi_CN = theta*phi + (1-theta)*phi_old."""
        phi = self.u.sym[0, 0]
        phi_old = self.phi_old.sym[0, 0]
        return self.theta_cn * phi + (1.0 - self.theta_cn) * phi_old

    def _grad_phi_cn(self):
        """(1, dim) row matrix grad(phi_CN), shared by the SUPG residual
        and the diffusive flux so both see the SAME Crank-Nicolson state."""
        return self.mesh.vector.gradient(self._phi_cn())

    def _strong_residual(self):
        """ADVECTION-ONLY strong residual R = (phi-phi_old)/dt +
        u.grad(phi_CN). This is what SUPG stabilises (F1's tau*R*u term)
        and is ALSO the complete F0 Galerkin term: diffusion contributes
        nothing here by construction (see class docstring) -- it enters
        only as a consistent Galerkin flux in F1, so this residual is
        identical whether diffusivity is zero or not."""
        dim = self.mesh.dim
        grad_cn = self._grad_phi_cn()
        u = self._V_fn
        advective = sum(u[0, i] * grad_cn[0, i] for i in range(dim))
        phi = self.u.sym[0, 0]
        phi_old = self.phi_old.sym[0, 0]
        return (phi - phi_old) / self._dt_value + advective

    def _dc_flux(self):
        r"""Discontinuity-capturing (DC) flux, Hughes & Mallet (1986) /
        Codina (1993) style. Zero (a (1, dim) zero row) unless
        :attr:`discontinuity_capturing` is True -- additive to F1
        exactly like the diffusive term, no separate code path.

        Pure streamline SUPG (the `tau*R*u` term) only damps
        oscillations ALONG the flow direction; it has no mechanism to
        damp them CROSSWIND. On a steep, under-resolved front this shows
        up as Gibbs-like ringing trailing the front -- and crucially,
        this happens regardless of physical diffusivity (it appears even
        at diffusivity=0, pure advection): it's a property of streamline
        SUPG's stabilisation, not an interaction with the diffusive
        term.

        The fix adds isotropic-looking but effectively CROSSWIND-ONLY
        artificial diffusion (the along-flow component is subtracted
        out, since tau*R*u already handles that direction -- adding it
        again here would double up the streamline diffusion):

        .. math::
            \nu_{dc} = C_{dc}\,h\,\frac{|R^{n}|}{\|\nabla\phi^{n}\|},
            \qquad
            \mathbf{F}_{1,dc} = \nu_{dc}\,
                \left(\nabla\phi_{CN} - (\nabla\phi_{CN}\cdot\hat{\mathbf u})\,\hat{\mathbf u}\right)

        Note the coefficient :math:`\nu_{dc}` uses the KNOWN, previous-
        timestep state (:math:`R^n`, :math:`\nabla\phi^n`, both from
        ``phi_old``) -- see the note below on why -- while it multiplies
        the CURRENT (unknown) :math:`\nabla\phi_{CN}`. :math:`R^n` is
        the advective part of the strong residual evaluated at
        ``phi_old`` alone, so :math:`\nu_{dc}` is automatically near-zero
        away from steep fronts (where :math:`\nabla\phi^n` is already
        small or the flow is locally well-resolved) and only activates
        where genuinely needed -- it doesn't add diffusion uniformly
        across the domain.
        Both the residual-magnitude and gradient-magnitude in the ratio
        are regularised (``+1e-30`` inside the sqrt) against division by
        a vanishing gradient in smooth regions.

        Crucially, :math:`\nu_{dc}` (the coefficient) is evaluated from
        ``phi_old`` ONLY -- never the unknown :math:`\phi^{n+1}` -- and
        is applied multiplying :math:`\nabla\phi_{CN}` (linear in the
        unknown). A first version of this used the CURRENT strong
        residual/gradient for :math:`\nu_{dc}` too, which makes the
        whole term genuinely nonlinear in :math:`\phi`: a
        :math:`|R|/\|\nabla\phi\|` ratio evaluated at the unknown is a
        well-known source of Newton stagnation (the SNES residual
        locking onto an exact plateau, ``DIVERGED_LINE_SEARCH`` /
        ``DIVERGED_MAX_IT``) rather than a clean single-iteration linear
        solve. Freezing :math:`\nu_{dc}` at ``phi_old`` (a standard
        lagged/Picard treatment for shock-capturing terms, e.g. Codina
        1993) keeps the whole solve LINEAR -- one Newton iteration, like
        the rest of this class -- at the cost of a one-timestep-lagged
        coefficient, which is a good trade since :math:`\phi` doesn't
        move far in a single step.
        """
        dim = self.mesh.dim
        if not self._discontinuity_capturing:
            return sympy.zeros(1, dim)

        u = self._V_fn
        u_mag2 = sum(u[0, i] ** 2 for i in range(dim))
        u_mag = sympy.sqrt(u_mag2 + 1.0e-30)
        u_hat = u / u_mag

        # --- nu_dc computed from phi_old ONLY (known, not the SNES
        # unknown) -- see docstring above. ---------------------------
        grad_old = self.mesh.vector.gradient(self.phi_old.sym[0, 0])
        advective_old = sum(u[0, i] * grad_old[0, i] for i in range(dim))
        grad_norm_old = sympy.sqrt(
            sum(grad_old[0, i] ** 2 for i in range(dim)) + 1.0e-30)
        # sympy.Abs() would be mathematically correct but sympy can
        # rewrite it via re()/im() (real/imaginary part) when it hasn't
        # been told the argument is real -- UW3's C99 JIT printer
        # doesn't support those. sqrt(x**2 + eps) is a standard
        # regularised abs() that sidesteps Abs/re/im entirely, and is
        # smooth (differentiable) at 0, which is preferable for
        # Newton's method anyway.
        abs_R_old = sympy.sqrt(advective_old ** 2 + 1.0e-30)

        h = self.mesh.cell_size()
        nu_dc = self._dc_coefficient * h * abs_R_old / grad_norm_old

        # --- applied to the CURRENT (unknown) CN gradient -- blended
        # between crosswind-only (weight=0, default, textbook
        # Hughes-Mallet) and the full gradient (weight=1, needed for a
        # front that varies only along the flow direction, where the
        # crosswind component is ~0 and weight=0 would be inert -- see
        # dc_streamwise_weight's docstring). Either way this stays
        # LINEAR in phi: nu_dc above is now just a known field and
        # grad(phi_CN) is a linear (gradient) operator on the unknown.
        grad_cn = self._grad_phi_cn()
        grad_cn_along_mag = sum(grad_cn[0, i] * u_hat[0, i] for i in range(dim))
        grad_cn_along = grad_cn_along_mag * u_hat
        grad_cn_cross = grad_cn - grad_cn_along
        grad_cn_target = grad_cn_cross + self._dc_streamwise_weight * grad_cn_along

        return nu_dc * grad_cn_target

    F0 = Template(
        r"f_0(\phi)",
        lambda self: sympy.Matrix([[self._strong_residual()]]),
        "Galerkin (w-tested) part of the residual -- the advection-only "
        "strong residual R itself. Diffusion never appears here (it's a "
        "flux term, see F1); this term is IDENTICAL whether diffusivity "
        "is zero or not.",
    )
    F1 = Template(
        r"\mathbf{F}_1(\phi)",
        lambda self: (
            self._diffusivity * self._grad_phi_cn()
            + self._tau() * self._strong_residual() * self._V_fn
            + self._dc_flux()
        ),
        r"Consistent Galerkin diffusive flux kappa*grad(phi_CN) (zero at "
        r"diffusivity=0), the SUPG-stabilised advective flux tau*R*u "
        r"(\nabla w$-tested), plus the crosswind discontinuity-capturing "
        r"flux (zero unless discontinuity_capturing=True).",
    )

    # ------------------------------------------------------------------
    @timing.routine_timer_decorator
    def estimate_dt(self, direction_aware: bool = False, percentile: float = 0.0):
        r"""
        Estimate an appropriate timestep for the advection-diffusion solver.

        Ported from ``SNES_AdvectionDiffusion.estimate_dt`` (the SLCN
        solver) -- see that docstring for the full rationale. The only
        difference is where :math:`\kappa` comes from: SLCN reads it off
        a constitutive model (``self.constitutive_model.K``), whereas
        this class carries a plain :attr:`diffusivity` attribute (float
        or symbolic/field expression), used directly below.

        Unlike SLCN, this is an EXPLICIT-in-structure SUPG scheme, not
        unconditionally stable -- the returned :math:`\delta t` is not
        just a *convenience* estimate here, it is closer to a genuine
        stability/accuracy requirement for the advective part; see the
        ``percentile`` note below for how much margin different
        reductions give you.

        This is an implicit (per-step SNES) solver so the returned
        :math:`\delta t` is the minimum of:

        - :math:`\delta t_{\textrm{diff}}`: typical time for diffusion across an element
        - :math:`\delta t_{\textrm{adv}}`: typical element-crossing time for a fluid parcel

        Parameters
        ----------
        direction_aware : bool, default False
            If True, the advective dt uses the per-cell extent
            *along the local velocity direction* — `h_eff_c =
            max_i(s_i) - min_i(s_i)` where `s_i = (x_i -
            centroid) · v̂` over the cell vertices. This is the
            distance material actually traverses through the cell
            per unit ``|v|``, and is **always ≥ the isotropic
            mesh._radii estimate**, by 1.5–3× for equant cells
            (geometric factor) and up to ~10× for cells that the
            mover has stretched along the flow direction. On
            adapted meshes the gain is substantial; on uniform
            meshes it's the geometric factor only. Off by
            default to preserve historical behaviour; safe to
            enable everywhere once validated.
        percentile : float, default 0.0
            How the per-element timesteps are reduced to one global
            value. ``0`` (the default) takes the strict global
            MINIMUM — a single cell sets the limit. A value ``> 0``
            takes that global percentile of the per-element dt
            instead (``50`` = median), so a few anisotropic sliver
            cells (velocity *across* a thin cell) cannot collapse
            the timestep. Unlike SLCN, this solver is NOT
            unconditionally stable, so a nonzero ``percentile`` here
            trades a guaranteed margin for a less conservative dt --
            validate against ``percentile=0`` before trusting it in
            production.

        Returns
        -------
        pint.Quantity or float
            The recommended timestep with physical time units if a model
            with reference scales is available, otherwise nondimensional.
        """

        ### required modules
        from mpi4py import MPI

        comm = uw.mpi.comm

        # See _sync_diffusivity_from_constitutive_model()'s docstring:
        # picks up `constitutive_model.Parameters.diffusivity` if that's
        # how the caller set it (the SLCN idiom), so the estimate matches
        # what solve() will actually use.
        self._sync_diffusivity_from_constitutive_model()

        ## global max diffusivity. SLCN reads this off a constitutive
        ## model's unified .K property; this class has no constitutive
        ## model wired up for diffusion (see class docstring) -- its
        ## diffusivity lives directly on self._diffusivity (float or a
        ## symbolic/field expression), which _global_max_diffusivity
        ## accepts exactly the same way it accepts self.constitutive_model.K.
        diffusivity_glob = _global_max_diffusivity(
            self._diffusivity, self.mesh)

        ### velocity values at element centroids (nondimensional)
        vel = _centroid_velocities_nd(self._V_fn, self.mesh)

        # Get per-element velocity magnitudes
        vel_magnitudes = np.linalg.norm(vel, axis=1)

        # Get per-element radii (characteristic element size)
        element_radii = self.mesh._radii

        ## estimate dt of adv and diff components using per-element approach
        ## dt_adv_i = h_i / |v_i| for advection
        ## dt_diff_i = h_i^2 / κ for diffusion (using global κ for now)

        # Reduce per-element dt to one global value. Default (percentile=0) =
        # strict global MINIMUM — one cell sets the limit. percentile>0 takes the
        # Nth global percentile (50 = median) of the per-element dt instead, so a
        # few anisotropic SLIVER cells (velocity ACROSS a thin cell) don't collapse
        # dt -- see the percentile note above: SLCN is unconditionally stable so
        # this trade-off is free there, it is NOT free here.
        def _reduce_dt(per_elem):
            fin = per_elem[np.isfinite(per_elem)] if len(per_elem) else per_elem
            if percentile and percentile > 0:
                gathered = comm.allgather(np.ascontiguousarray(fin, dtype=float))
                allv = (np.concatenate([a for a in gathered if a.size])
                        if any(a.size for a in gathered) else np.empty(0))
                return float(np.percentile(allv, percentile)) if allv.size else np.inf
            loc = float(np.min(fin)) if len(fin) else np.inf
            return comm.allreduce(loc, op=MPI.MIN)

        # Per-element diffusive timestep (all elements use same diffusivity)
        if diffusivity_glob > 0:
            dt_diff_per_element = (element_radii ** 2) / diffusivity_glob
        else:
            dt_diff_per_element = np.array([np.inf])

        # Per-element advective timestep — either isotropic
        # (mesh._radii / |v|) or direction-aware (v-aligned cell
        # extent / |v|).
        if direction_aware:
            # Per-cell vertex indices (triangle / tet).
            from underworld3.meshing.smoothing import _tri_cells
            tris = _tri_cells(self.mesh.dm)
            if tris is None:
                # Fall back to isotropic for non-triangle meshes.
                h_per_element = element_radii
            else:
                coords = np.asarray(self.mesh.X.coords)
                centroids = coords[tris].mean(axis=1)
                # v-hat per cell (use centroid v we already have)
                vhat = np.where(
                    vel_magnitudes[:, None] > 0,
                    vel / np.maximum(vel_magnitudes[:, None],
                                      1.0e-30),
                    0.0)
                D = coords[tris] - centroids[:, None, :]
                # Signed projections along v̂ per cell vertex
                s = np.einsum('cvd,cd->cv', D, vhat)
                h_per_element = s.max(axis=1) - s.min(axis=1)
                # Sanity-floor — for zero-velocity cells s=0
                # ⇒ h_eff=0 ⇒ dt_adv=inf via the where below
                h_per_element = np.maximum(
                    h_per_element, 0.0)
        else:
            h_per_element = element_radii

        with np.errstate(divide='ignore', invalid='ignore'):
            dt_adv_per_element = np.where(
                vel_magnitudes > 0,
                h_per_element / vel_magnitudes,
                np.inf
            )
        # Global reduction — strict min (percentile=0) or Nth percentile (median).
        min_dt_diff_glob = _reduce_dt(dt_diff_per_element)
        min_dt_adv_glob = _reduce_dt(dt_adv_per_element)

        # Store for user inspection
        self.dt_adv = min_dt_adv_glob if not np.isinf(min_dt_adv_glob) else 0.0
        self.dt_diff = min_dt_diff_glob if not np.isinf(min_dt_diff_glob) else 0.0

        # Take overall minimum (respecting infinity for zero velocity/diffusivity cases)
        dt_estimate = min(min_dt_diff_glob, min_dt_adv_glob)

        # If both are infinite (no velocity and no diffusivity), return infinity
        if np.isinf(dt_estimate):
            return np.inf

        # Dimensionalise the result to physical time
        try:
            return uw.dimensionalise(np.squeeze(dt_estimate), {'[time]': 1})
        except Exception:
            # Fallback: return plain nondimensional number
            return np.squeeze(dt_estimate)

    # ------------------------------------------------------------------
    def solve(self, *, timestep: float = None, zero_init_guess: bool = False, **kwargs) -> None:
        """Advance phi by one Crank-Nicolson step of size `timestep`. Updates
        phi_old from the current field *before* solving, then performs one
        implicit weak-form SNES solve for phi^{n+1} (warm-started from the
        current field unless zero_init_guess=True). Identical for pure
        advection (diffusivity=0, the default) and advection-diffusion
        (diffusivity != 0) -- there is no separate code path.

        `timestep` (matching SNES_AdvectionDiffusion's/SLCN's calling
        convention, `.solve(timestep=dt)`) is keyword-only DELIBERATELY:
        an earlier version of this class named the parameter `dt` and took
        it positionally, and a caller written against SLCN's
        `solve(zero_init_guess, timestep, ...)` order that passed a plain
        `.solve(dt)` positional call silently landed `dt` in
        `zero_init_guess` instead, leaving `timestep` at its default and
        producing a `None`-propagation crash two calls deeper (see
        ``ddt.py``'s `_trace_departure_points`, `0.5 * dt_for_calc` with
        `dt_for_calc=None`) instead of a clear error at the call site
        itself. Keyword-only trades that silent mis-binding for an
        immediate, loud `TypeError` if a caller ever gets this wrong again.
        """
        if timestep is None:
            raise ValueError(
                "SNES_AdvectionDiffusion_SUPG.solve() requires `timestep` "
                "(e.g. `adv_diff.solve(timestep=dt)`) -- there is no default."
            )
        self._sync_diffusivity_from_constitutive_model()
        dt = float(timestep)
        self.phi_old.data[:, 0] = self.u.data[:, 0]
        if dt != self._last_dt:
            # dt is captured as a plain float inside the F0/F1 lambdas, so
            # a genuine change in dt needs the residual re-evaluated (and
            # hence the DS/JIT kernels rebuilt) -- but only THEN, not on
            # every call with an unchanged dt, which would force a needless
            # rebuild every single timestep.
            self._dt_value = dt
            self._last_dt = dt
            self.is_setup = False
        super().solve(zero_init_guess=zero_init_guess, **kwargs)