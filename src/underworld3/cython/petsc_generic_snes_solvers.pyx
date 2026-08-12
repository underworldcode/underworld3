import collections

import numpy as np
import sympy

from typing import Optional, Union
from petsc4py import PETSc

import underworld3
import underworld3 as uw
from   underworld3.utilities._jitextension import getext, JITCallbackSet
import underworld3.timing as timing

from underworld3.utilities._api_tools import uw_object
from underworld3.utilities import multigrid_options


class _StrategyName(str):
    """A strategy name that also reports what it resolved to.

    Subclasses ``str`` deliberately: ``solver.strategy == "fast"``, string
    formatting and serialisation all behave exactly as before, but displaying it —
    in a REPL, a notebook, or a log line — shows the preconditioner it actually
    configured. Asking "what am I running?" should not require knowing which nine
    PETSc option keys to look up.

    Use :attr:`SolverBaseClass.preconditioner_settings` for the machine-readable
    form.
    """

    def __new__(cls, name, summary=""):
        obj = super().__new__(cls, name)
        obj._summary = summary
        return obj

    def __reduce__(self):
        # `str.__reduce_ex__` reconstructs via `cls(value)` with ONE argument, which
        # a two-argument `__new__` cannot accept — so without this, pickling, copy
        # and deepcopy of a strategy value all raise TypeError. The summary is
        # derived state and is carried along rather than recomputed, because the
        # solver it came from is not part of the pickle.
        return (self.__class__, (str(self), self._summary))

    def __repr__(self):
        return f"{str.__repr__(self)} — {self._summary}"

    def _repr_markdown_(self):
        return f"**`{str(self)}`** — {self._summary}"

from underworld3.function import expression as public_expression
expression = lambda *x, **X: public_expression(*x, _unique_name_generation=True, **X)

from underworld3.function.expressions import unwrap_expression as _unwrap_expression


def _jacobian_unwrap(expr):
    """Expand UWexpressions down to (but NOT including) constant atoms, for use
    as the input to a Jacobian derivative (``derive_by_array`` / ``diff``).

    Applied element-wise over a sympy ``Matrix``/``Array`` so atoms embedded in
    the residual flux are reached. Non-constant UWexpressions (e.g. the
    effective viscosity ``Min(eta0, tau_y/2/eps_II)``) are expanded so the
    derivative sees their field / grad-v dependence and forms the full Newton
    tangent. Truly-constant atoms (``eta0``, ``tau_y``, ...) are kept as the
    *same* symbol object so the JIT ``constants[]`` runtime-update mechanism is
    preserved — the keep-constants predicate is shared with
    ``getext()._extract_constants`` so the two cannot drift apart.

    This is a no-op for constant-viscosity problems (eta has no grad-v
    dependence), so those Jacobians stay bit-identical.

    The unwrapped result is additionally made DIFFERENTIATION-SAFE: any
    ``sqrt(g)`` whose argument carries non-constant symbols becomes
    ``sqrt(g + 1e-36)``. Differentiating a bare invariant
    :math:`\dot\varepsilon_{II} = \sqrt{g}` produces
    :math:`\partial\sqrt{g}/\partial L = \dot\varepsilon/(2\dot\varepsilon_{II})`
    — the DIRECTION of the strain rate, which is 0/0 at a state of rest —
    so every consistent-tangent assembly at a cold (v = 0) start filled
    the operator with NaN (measured: J(0) norm = nan for a ViscoPlastic
    model at ANY yield stress, surfacing as GAMG's "Computed maximum
    singular value as zero", error 77; the alpha-blended continuation
    kernel inherits it even at alpha = 0 because IEEE 0*NaN = NaN). The
    guard makes the derivative exactly zero at the singular point and
    perturbs it by under one part in 1e24 at any resolvable strain rate.
    The RESIDUAL is never routed through here, and the default (Picard)
    tangent never calls this function, so both remain bit-identical.

    See ``docs/developer/design/jacobian-unwrap-constants-bug.md``.
    """
    eps2 = sympy.Float(1.0e-36)

    def _guard_sqrts(e):
        # every HALF-INTEGER power: +1/2 (the invariant itself), -1/2
        # (its reciprocal in eta_pl = tau_y/(2 edot_II)), -3/2 (their
        # derivatives), ... — all singular in value or derivative at a
        # zero-argument state
        return e.replace(
            lambda n: (n.is_Pow and n.exp.is_Rational
                       and n.exp.q == 2 and n.args[0].free_symbols),
            lambda n: sympy.Pow(n.args[0] + eps2, n.exp))

    f = lambda e: _guard_sqrts(
        _unwrap_expression(e, mode="symbolic_keep_constants"))
    if isinstance(expr, sympy.MatrixBase):
        return expr.applyfunc(f)
    if isinstance(expr, sympy.NDimArray):
        return sympy.Array([f(e) for e in expr], expr.shape)
    return f(expr)  # scalar expression


include "petsc_extras.pxi"

class SolverBaseClass(uw_object):
    r"""
    The Generic `Solver` is used to build the `SNES Solvers`
        - `SNES_Scalar`
        - `SNES_Vector`
        - `SNES_Stokes`

    This class is not intended to be used directly
    """

    def __init__(self, mesh):

        super().__init__()

        self.mesh = mesh
        self.mesh_dm_coordinate_hash = None
        self.compiled_extensions = None
        self.constants_manifest = []

        # Jacobian tangent selection — validated property, see the
        # consistent_jacobian docstring below for the mode semantics.
        self.consistent_jacobian = False
        # Picard->Newton continuation parameter (constants[]-routed so it can be
        # ramped at solve time without a JIT recompile). 0 = Picard, 1 = Newton.
        # Created LAZILY (see _get_newton_alpha) only when continuation is used,
        # so the default path constructs no extra UWexpression and therefore
        # cannot perturb global symbol-naming / JIT-cache state.
        self._newton_alpha = None
        # Switch threshold: ramp alpha toward 1 once the relative residual falls
        # below this (the basin-of-attraction heuristic for Newton).
        self.newton_switch_rtol = 1.0e-2

        # Fine-grained rebuild flags backing the is_setup property. See the
        # is_setup docstring and _build() for how these are consumed.
        self._needs_dm_rebuild = True
        self._needs_bc_reregister = True
        self._needs_function_rewire = True

        # Warm-start status, public (read-only) via `has_solution`. Set True only
        # after a converged solve; reset on a structural rebuild (is_setup=False)
        # so a remesh / adapt / mesh-mover never warm-starts off stale field
        # data. Kept through coefficient changes (viscosity, yield softness δ, BC
        # values, time step). See has_solution / _record_convergence_status and
        # docs/developer/design/nonlinear-solver-homotopy-warmstart.md (Layer 1).
        self._has_solution = False

        self.Unknowns = self._Unknowns(self)

        self._order = 0
        self._constitutive_model = None
        self._rebuild_after_mesh_update = self._build

        self.name = "Solver_{}_".format(self.instance_number)
        self.petsc_options_prefix = self.name
        self.petsc_options = PETSc.Options(self.petsc_options_prefix)

        self.dm = None
        self.dm_hierarchy = []
        self.snes = None
        self.petsc_fe_u = None
        self.petsc_fe_p = None

        # Solution decomposition caches (IS sets)
        self._velocity_is = None
        self._pressure_is = None
        self._subdict = {}

        # Per-Newton-iteration callbacks (PETSc SNESSetUpdate). Empty by default
        # -> no hook is installed and the solve path is unchanged. See
        # add_update_callback().
        self._snes_update_callbacks = []

        # Solver difficulty / convergence reporting (default-on; see the
        # solve_report / solve_history properties and _capture_solve_report).
        # A SolveReport is recorded after every solve; solve_history keeps a
        # short bounded trail so continuation / time-stepping loops can read
        # trends without logging each step themselves.
        self._solve_report = None
        self._solve_history = collections.deque(maxlen=32)
        # Bounded/resumable difficulty probe state (see estimate_difficulty).
        # _difficulty_probe gates probe-only behaviour (iteration-cap solves,
        # suppressed divergence warnings, restart anchoring); _difficulty_max_it
        # is the active cap; _resume_abs_target anchors a chunked start/stop/restart
        # chain to the ORIGINAL ||F0|| so it terminates at the same point an
        # uninterrupted solve would. None = not in a resumable chain.
        self._difficulty_probe = False
        self._difficulty_max_it = None
        self._resume_abs_target = None
        # PETSc-level instrumentation: the sub-solve work gauge, and the
        # wall-clock deadline once guard() arms it. Created on the first solve
        # (systems/solver_health.py) because it needs a live SNES to attach to.
        self._instrumentation = None

        # Preconditioner selection — see the `preconditioner` property.
        # `_pc_option_prefix` is set by subclasses that participate in the easy
        # FMG/GAMG switch ("" for scalar/vector, "fieldsplit_velocity_" for
        # Stokes); None means "this solver manages its own PC options" and the
        # helper below is a no-op.
        self._preconditioner = "auto"
        self._pc_option_prefix = None
        # The pc_type value this helper last managed. Subclasses that opt in set
        # their __init__ default ("gamg"); used in "auto" mode to tell an
        # untouched framework default (eligible for FMG upgrade) apart from an
        # explicit user override of pc_type, which must be respected.
        self._pc_managed_value = "gamg"
        # Latches once the user (or harness) is seen to have set the PC options
        # themselves. _apply_preconditioner_options() runs on EVERY _build (so
        # "auto" can re-resolve after a remesh), and without this latch the
        # override detection only holds on the first build: once we adopted the
        # user's pc_type as our managed value, a later rebuild could no longer
        # tell "user set mg" from "we set mg" and would clobber their tuned
        # smoother / coarse-solver options with the framework FMG bundle.
        self._pc_user_override = False
        # Every multigrid option value UW3 itself has written on the managed block,
        # keyed by full option name. This is what lets the bundle honour a
        # user-set smoother while still managing the keys the user left alone: a
        # present key whose value is not the one we recorded writing is theirs.
        # Ownership is RECORDED, never inferred from the value — inference fails
        # the moment a second internal writer touches the same key, which is how
        # the `tolerance` and `strategy` setters defeated an earlier attempt (#477).
        # Internal writers therefore go through _push_managed_option().
        self._managed_pc_options = {}
        # Has _apply_preconditioner_options actually made the resolution decision
        # yet? Until it has, the options database holds only this solver's __init__
        # defaults, which is NOT what the next solve will run — reporting them as
        # resolved would be exactly the stale-but-authoritative-looking summary this
        # reporting exists to prevent.
        self._pc_resolved = False

        # Custom multigrid prolongation hierarchy (see set_custom_mg /
        # utilities.custom_mg). None => standard FMG/GAMG path, unchanged.
        self._custom_mg = None

    @property
    def preconditioner_settings(self):
        """The option values the managed preconditioner block is configured with.

        A read-only dict of the multigrid keys UW3 currently has in the options
        database for this solver's managed block, so what actually got applied can
        be asserted on instead of inferred from timings. Empty for a solver with no
        managed block (``_pc_option_prefix is None``), or before the first build
        resolves one.

        The values are what the strategy resolved to, *including* any key you set
        yourself — those are respected (see :attr:`petsc_options`). Use
        :attr:`strategy` for a readable summary of the same thing.
        """
        prefix = self._pc_option_prefix
        if prefix is None:
            return {}
        keys = set(multigrid_options.gamg_bundle().settings)
        for coarse in multigrid_options.GEOMETRIC_MG_COARSE_SOLVERS:
            keys |= set(multigrid_options.geometric_mg_bundle(coarse=coarse).settings)
        out = {}
        for key in sorted(keys):
            name = prefix + key
            if self.petsc_options.hasName(name):
                out[key] = self.petsc_options.getString(name)
        return out

    @property
    def _user_overridden_pc_options(self):
        """The managed-block keys the USER set, as (key, value) pairs.

        A key present in the options database whose value is not the one UW3
        recorded writing is theirs — the same test the bundle writer uses to decide
        what to leave alone."""
        prefix = self._pc_option_prefix
        if prefix is None:
            return ()
        qualified = self.petsc_options_prefix
        return tuple(
            (key, value) for key, value in self.preconditioner_settings.items()
            if self._managed_pc_options.get(qualified + prefix + key) != value)

    @property
    def _mg_smoother_variant(self):
        """Which measured smoother regime this solver's strategy asks for.

        ``solver.strategy`` is the named intent ("I want speed" / "I want this to
        converge"); the values live in ``utilities.multigrid_options``. Solvers with
        no strategy axis get the robust default. See
        :func:`multigrid_options.geometric_mg_bundle` for the measurements."""
        return "fast" if getattr(self, "_strategy", "default") == "fast" else "robust"

    def _push_managed_option(self, key, value):
        """Write a PETSc option UW3 owns, recording that we wrote it.

        Use this for any option a solver sets on its own behalf that the multigrid
        bundles also write (``utilities.multigrid_options``). A plain
        ``self.petsc_options[key] = value`` is indistinguishable from a user's own
        write, and the bundle would then back off from a key nobody asked for.
        """
        self.petsc_options[key] = value
        # Key the record by the GLOBAL option name. `self.petsc_options` is a
        # prefixed view (`Solver_N_`), but custom_mg._configure_pcmg reads the
        # global database using the live PC's own full prefix — so an unqualified
        # record makes every key look user-owned over there and the bundle
        # silently stops applying.
        self._managed_pc_options[self.petsc_options_prefix + key] = \
            multigrid_options.option_string(value)

    @property
    def consistent_jacobian(self):
        r"""Jacobian tangent selection: ``False`` | ``True`` | ``"continuation"``.

        Selects the tangent used by :meth:`_jacobian_source` and the solve
        dispatch; the residual is never affected, so the converged solution
        always satisfies the exact constitutive law.

        ``False`` (default)
            Differentiate the residual flux *as wrapped* — the effective
            viscosity is frozen, giving a Picard / defect-correction tangent.
            Bit-identical to the long-standing behaviour. Globally robust;
            load-bearing for the tuned hard-yield viscoplastic paths.
        ``True``
            Unwrap the flux before differentiation so the tangent captures
            :math:`\partial\eta/\partial(\nabla v)` (full Newton). Fast near
            the solution; its yield kink can stall the line search far from it.
        ``"continuation"``
            Picard :math:`\rightarrow` Newton. Blend
            :math:`J(\alpha) = J_{\mathrm{picard}} + \alpha\,(J_{\mathrm{newton}}
            - J_{\mathrm{picard}})` with :math:`\alpha` a ``constants[]``
            parameter ramped :math:`0 \rightarrow 1` by a SNES monitor as the
            residual drops. Picard locates the basin, Newton gives quadratic
            convergence inside it (cf. Spiegelman et al. 2016; ASPECT
            defect-correction-then-Newton). :math:`\alpha = 0` is bit-identical
            to Picard, so no recompile is needed to switch.

        The Newton flux for a model whose flux has a non-smooth yield kink is
        the model's own smooth law (``constitutive_model.flux_jacobian``) when
        it provides one; otherwise the exact unwrapped flux. See
        ``docs/developer/design/jacobian-unwrap-constants-bug.md``.

        Raises
        ------
        ValueError
            On assignment of anything other than ``False``, ``True`` or
            ``"continuation"``. Falsy values (``None``, ``0``, ``""``)
            normalize to ``False`` (they already selected the Picard tangent).
            Before validation, any other truthy value (``1``, ``"picard"``,
            ``"Continuation"``) silently selected the full-Newton tangent.
        """
        return self._consistent_jacobian

    @consistent_jacobian.setter
    def consistent_jacobian(self, mode):
        if not mode:
            self._consistent_jacobian = False
        elif mode is True or mode == "continuation":
            self._consistent_jacobian = mode
        else:
            raise ValueError(
                f"consistent_jacobian must be False, True or 'continuation'; "
                f"got {mode!r}")

    def _jacobian_source(self, expr, newton_expr=None):
        """Prepare a residual flux for Jacobian differentiation.

        ``expr`` is the exact (Picard) flux; ``newton_expr`` is the consistent
        (Newton) flux to use — when None it is the unwrapped ``expr`` (the
        derivative then captures d(eta)/d(grad v)). The residual itself is never
        passed through here, so the converged solution always satisfies the
        exact constitutive law.

        Returns, by ``consistent_jacobian`` mode:
          * False  -> ``expr``  (frozen viscosity; bit-identical Picard tangent)
          * True   -> ``newton_expr``  (full consistent Newton tangent)
          * "continuation" -> ``expr + alpha*(newton_expr - expr)`` with alpha a
            constants[] parameter ramped 0->1 at solve time. Differentiating
            this gives ``G_picard + alpha*(G_newton - G_picard)`` because alpha
            is constant w.r.t. the unknowns. alpha=0 is bit-identical to Picard.

        No-op for constant viscosity (``newton_expr`` == ``expr``).
        """
        mode = self.consistent_jacobian
        if not mode:
            return expr
        if newton_expr is None:
            newton_expr = _jacobian_unwrap(expr)
        if mode == "continuation":
            a = self._get_newton_alpha()
            if isinstance(expr, sympy.MatrixBase):
                ne = newton_expr if isinstance(newton_expr, sympy.MatrixBase) \
                    else sympy.Matrix(newton_expr)
                return expr + a * (ne - expr)
            if isinstance(expr, sympy.NDimArray):
                # flatten to scalars first — iterating an N-d Array yields
                # sub-arrays, not elements
                ex = expr.reshape(expr._loop_size)
                ne = sympy.Array(newton_expr).reshape(expr._loop_size)
                return sympy.Array(
                    [e + a * (n - e) for e, n in zip(ex, ne)], expr.shape
                )
            return expr + a * (newton_expr - expr)  # scalar
        return newton_expr

    def _newton_flux(self, exact_F1):
        """Newton (consistent) flux for the bulk ``F1`` Jacobian source.

        Returns the constitutive model's own smooth tangent law
        (``constitutive_model.flux_jacobian``) when it supplies one — matched to
        the container/shape of ``exact_F1`` — otherwise ``None`` so that
        :meth:`_jacobian_source` falls back to the exact flux unwrapped. Lets a
        model whose flux has a non-smooth yield kink provide a physically
        motivated smooth tangent without changing the residual.
        """
        # Default (Picard) path never needs the Newton flux — short-circuit so
        # the (potentially expensive) model.flux_jacobian is not even evaluated,
        # keeping the default assembly allocation-free and bit-identical.
        if not self.consistent_jacobian:
            return None
        cm = getattr(self, "constitutive_model", None)
        smooth = getattr(cm, "flux_jacobian", None) if cm is not None else None
        if smooth is None:
            return None
        try:
            sm = sympy.Array(smooth)
            ex = sympy.Array(exact_F1)
            if sm.shape != ex.shape:
                sm = sm.reshape(*ex.shape)
            if isinstance(exact_F1, sympy.MatrixBase):
                return sympy.Matrix(exact_F1.shape[0], exact_F1.shape[1], list(sm))
            return sm
        except Exception:
            return None

    def _get_newton_alpha(self):
        """Lazily construct the Picard->Newton continuation parameter.

        Built on first use (continuation mode only) so the default Picard path
        creates no extra UWexpression and leaves global symbol-naming / JIT-cache
        state byte-identical to historical behaviour.
        """
        if self._newton_alpha is None:
            self._newton_alpha = uw.function.expression(
                r"\alpha_{N}", sympy.Float(0.0),
                "Picard->Newton continuation fraction (0=Picard, 1=Newton)",
            )
        return self._newton_alpha

    def _set_newton_alpha(self, value):
        """Set the Picard->Newton continuation fraction and push it to the DS.

        alpha is routed through ``constants[]`` (it appears as a constant atom
        in the blended Jacobian), so this updates the live tangent WITHOUT a JIT
        recompile. No-op outside ``consistent_jacobian == "continuation"`` (alpha
        is then absent from the constants manifest).
        """
        self._get_newton_alpha().sym = sympy.Float(value)
        try:
            self._update_constants()
        except Exception:
            pass

    def set_custom_mg(self, coarse_meshes, kind="barycentric", verbose=False):
        r"""Drive geometric multigrid with a prolongation we build ourselves.

        Supplies a sequence of (possibly **non-nested**) coarse meshes from which
        a barycentric or RBF prolongation ``P`` is assembled and installed into
        the PCMG via ``PC.setMGInterpolation``; coarse operators are formed by
        Galerkin RAP. This decouples geometric multigrid from a nested
        ``refine()`` hierarchy — it works even when the solver mesh has no
        refinement hierarchy at all.

        Parameters
        ----------
        coarse_meshes : list of Mesh
            Coarsest-first list of coarse meshes (the finest level is the
            solver's own mesh). Need not be nested with the solver mesh.
        kind : {"barycentric", "rbf"}
            Prolongation builder. ``barycentric`` is FE-exact; ``rbf`` is a
            polyharmonic RBF (Shepard-normalised). Default ``barycentric``.
        verbose : bool
            Print the per-level DOF counts at injection.

        Notes
        -----
        Supported both on single-field (scalar / vector) solvers — where the
        prolongation is installed directly on the solver's ``PCMG`` — and on the
        **Stokes velocity block** (the ``fieldsplit_velocity_`` sub-PC), where the
        velocity sub-PC is only reachable once the monolithic Jacobian has been
        assembled; the install there assembles the Jacobian, descends the
        fieldsplit to the velocity sub-PC, and rebuilds it as a fresh ``PCMG``
        driven by our ``P``. Injection happens at solve time (after
        ``setFromOptions`` / nullspace attach) via
        :func:`underworld3.utilities.custom_mg.inject_custom_mg`. See
        :mod:`underworld3.utilities.custom_mg`.

        .. deprecated:: 2026-07
            This is the legacy **serial-only, finest-only-reduction,
            single-field** path; the Stokes velocity-block support described
            above is delivered by :meth:`set_custom_fmg`, which is the
            canonical entry point (parallel-capable, BC-per-level reduction).
        """
        import warnings
        warnings.warn(
            "set_custom_mg is deprecated (legacy serial-only, single-field "
            "custom-MG path); use set_custom_fmg(coarse_meshes, "
            "builder=..., field_id=...) instead",
            DeprecationWarning, stacklevel=2)
        if kind not in ("barycentric", "rbf"):
            raise ValueError("kind must be 'barycentric' or 'rbf'")
        if not coarse_meshes:
            raise ValueError("coarse_meshes must be a non-empty coarsest-first list")
        self._custom_mg = {"coarse_meshes": list(coarse_meshes),
                           "kind": kind, "verbose": verbose}
        self.is_setup = False

    def set_custom_fmg(self, coarse_meshes, *, builder="barycentric",
                       field_id=None, verbose=False):
        r"""Drive geometric multigrid with a prolongation built from
        ``coarse_meshes`` — the canonical custom-MG entry point.

        Registers a BC-per-level reduced hierarchy on the solver so the next
        :meth:`solve` builds and installs it (build-time injection). Works in
        parallel and on the **Stokes velocity block** (pass ``field_id=0`` on a
        saddle-point solver). This supersedes the legacy
        :meth:`set_custom_mg` path (serial-only, finest-only reduction,
        single-field).

        Parameters
        ----------
        coarse_meshes : list of Mesh
            Coarsest-first list of coarse meshes (the finest level is the
            solver's own mesh). They need only carry the same boundary labels
            as the solver's mesh.
        builder : {"barycentric", "rbf"}, default "barycentric"
            Prolongation builder. ``barycentric`` is FE-exact; ``rbf`` is a
            polyharmonic RBF (Shepard-normalised).
        field_id : int, optional
            Target field for a multi-field (saddle-point) solver; ``0`` is the
            Stokes velocity block. ``None`` (default) for single-field solvers.
        verbose : bool, default False
            Print the per-level DOF counts at injection.

        See Also
        --------
        underworld3.utilities.custom_mg.set_custom_fmg : the implementation.
        """
        from underworld3.utilities.custom_mg import set_custom_fmg as _set_custom_fmg
        _set_custom_fmg(self, coarse_meshes, builder=builder,
                        field_id=field_id, verbose=verbose)

    def add_update_callback(self, callback):
        r"""Register a callback fired at the start of every nonlinear (SNES) iteration.

        The callback is invoked as ``callback(solver, iteration)``. Immediately
        before the call the current Newton iterate is scattered into the solver's
        field MeshVariables (so the callback can read ``v``, ``p``, ... at the
        current iterate); immediately afterwards the (possibly modified) fields
        are gathered back into the iterate. Typical uses:

        - re-fire an auxiliary solve each iteration — e.g. a Helmholtz/Projection
          smoother that supplies a regularised field the residual depends on
          (gradient-plasticity / shear-band stabilisation);
        - impose a gauge consistently inside the nonlinear solve — e.g. remove the
          mean pressure on a surface so the pressure null space is pinned
          (see :meth:`set_pressure_gauge` on the Stokes solver).

        Callbacks run in registration order. Registering one forces a re-setup so
        the PETSc ``SNESSetUpdate`` hook is attached. With no callbacks registered
        no hook is installed and the solve path is byte-for-byte unchanged.
        """
        self._snes_update_callbacks.append(callback)
        self._needs_function_rewire = True
        return callback

    def _attach_snes_update_hook(self):
        """Attach the SNESSetUpdate dispatcher iff callbacks are registered."""
        if self.snes is not None and self._snes_update_callbacks:
            self.snes.setUpdate(self._dispatch_snes_update)

    def _nondimensional_time(self, time):
        """Return ``time`` as a plain float in solver (non-dimensional) units.

        Pint quantities and UWQuantity values are non-dimensionalised through
        the scaling system; bare numbers pass through unchanged.
        """
        if hasattr(time, 'magnitude') or hasattr(time, '_pint_qty'):
            return float(uw.non_dimensionalise(time))
        return float(time)

    def _scatter_global_to_fields(self, gvec):
        """Scatter the global iterate into the solver's field MeshVariable,
        correct on driven (non-zero Dirichlet) boundaries.

        Single-field solvers (scalar / vector / multi-component) store the whole
        solution in ``self.u`` on ``self.dm``. A plain ``globalToLocal`` leaves a
        field's non-zero Dirichlet (driven) boundary DOFs stale — those are not in
        the global vector; they are imposed on the LOCAL vector by
        ``DMPlexSNESComputeBoundaryFEM``. So a callback reading the field on a
        driven boundary would otherwise see a wrong value. This mirrors the
        post-solve copy-back: globalToLocal -> boundary FEM -> copy into the field.

        Stokes overrides this to split its multi-field DM (see
        ``SNES_Stokes_SaddlePt._scatter_global_to_fields``).
        """
        cdef DM dm = self.dm
        cdef Vec clvec = self.dm.getLocalVec()
        self.dm.globalToLocal(gvec, clvec)
        ierr = DMPlexSNESComputeBoundaryFEM(dm.dm, <void*>clvec.vec, NULL); CHKERRQ(ierr)
        self.u.vec.array[:] = clvec.array[:]
        self.dm.restoreLocalVec(clvec)

        self.mesh._stale_lvec = True
        base = getattr(self.u, "_base_var", self.u)
        if hasattr(base, "_sync_lvec_to_gvec"):
            base._sync_lvec_to_gvec()
        if hasattr(base, "_canonical_data"):
            base._canonical_data = None

    def _gather_fields_to_global(self, gvec):
        """Gather the solver's field MeshVariable back into the global iterate.

        Single-field default (``localToGlobal`` drops the boundary DOFs, which are
        re-imposed next iteration). Stokes overrides for its multi-field DM.
        """
        self.dm.localToGlobal(self.u.vec, gvec)

    def _refresh_auxiliary_vec(self):
        """Rebuild the mesh auxiliary vector so the residual / nested solves see the
        current field values (callbacks may have changed v, p, or auxiliary fields)."""
        self.mesh.update_lvec()
        self.dm.setAuxiliaryVec(self.mesh.lvec, None)

    def _dispatch_snes_update(self, snes, iteration):
        """PETSc SNESSetUpdate hook: sync iterate->fields, run callbacks, sync back.

        The mesh auxiliary vector is refreshed (a) before callbacks, so a callback
        reading v/p (e.g. a Helmholtz/Projection smoother) sees the current iterate,
        and (b) after, so the outer residual sees any fields the callbacks updated.
        """
        gvec = snes.getSolution()
        self._scatter_global_to_fields(gvec)
        self._refresh_auxiliary_vec()
        for callback in self._snes_update_callbacks:
            callback(self, iteration)
        self._gather_fields_to_global(gvec)
        self._refresh_auxiliary_vec()

    @property
    def preconditioner(self):
        """Preconditioner selection for the (velocity) block.

        One of:

        - ``"auto"`` (default) — use geometric Full Multigrid (FMG) when the
          mesh carries a genuine refinement hierarchy
          (``len(mesh.dm_hierarchy) > 1``, i.e. built with ``refinement >= 1``),
          otherwise fall back to algebraic multigrid (GAMG).
        - ``"fmg"`` (alias ``"mg"``) — force geometric multigrid. Requires a
          refinement hierarchy; warns and falls back to GAMG if none exists.
        - ``"gamg"`` — force algebraic multigrid (the historical default).

        Geometric multigrid is inherently robust to mesh anisotropy (it is built
        from the geometric refinement hierarchy, not the operator connection
        graph), which makes it the preferred choice on adapted/deformed meshes.
        The hierarchy survives the coordinate-deforming adaptation movers and
        only collapses under a true remesh — in which case ``"auto"``
        transparently reverts to GAMG.

        For Stokes this governs the velocity fieldsplit block; for scalar/vector
        solvers it governs the top-level preconditioner. Other solvers ignore it.
        """
        return self._preconditioner

    @preconditioner.setter
    def preconditioner(self, value):
        choice = str(value).lower()
        if choice == "mg":
            choice = "fmg"
        if choice not in ("auto", "fmg", "gamg"):
            raise ValueError(
                f"preconditioner must be 'auto', 'fmg', or 'gamg' (got {value!r})"
            )
        self._preconditioner = choice
        # Force a full rebuild so the new option bundle is pushed to PETSc.
        self.is_setup = False

    def _apply_preconditioner_options(self):
        """Push the PETSc option bundle implied by ``self.preconditioner``.

        Called from ``_build`` so that ``"auto"`` re-resolves against the
        *current* mesh — e.g. after a remesh that collapses the refinement
        hierarchy. Solvers opt in by setting ``self._pc_option_prefix`` (``""``
        or ``"fieldsplit_velocity_"``); the default (None) makes this a no-op.
        """
        prefix = self._pc_option_prefix
        if prefix is None:
            return
        # Every path from here is a resolution decision, including "the user owns
        # these options, leave them alone".
        self._pc_resolved = True

        opts = self.petsc_options

        # The mesh is the source of truth for hierarchy depth: the solver's own
        # dm_hierarchy is a clone of it and may not be populated this early.
        n_levels = len(getattr(self.mesh, "dm_hierarchy", []) or [])

        if self._preconditioner == "auto":
            # Auto mode is deliberately conservative: it only ever *adds*
            # geometric multigrid, never rewrites an existing GAMG/other
            # configuration (internal utility solvers — mesh smoothing, OT,
            # projections — set their own carefully tuned pc options that must
            # be left intact).
            if self._pc_user_override:
                # User/harness owns the PC options — never re-apply our bundle,
                # even across remesh rebuilds (see _pc_user_override init note).
                return
            current = opts.getString(f"{prefix}pc_type")
            if current and current != self._pc_managed_value:
                # Preconditioner was set explicitly elsewhere — respect it, and
                # latch so later rebuilds keep respecting it.
                self._pc_user_override = True
                self._pc_managed_value = current
                return
            if n_levels > 1:
                want_fmg = True
            else:
                # No hierarchy. Only act to revert a previous FMG upgrade of
                # ours (e.g. after a remesh collapsed the hierarchy); otherwise
                # leave the existing default/tuned options untouched.
                if self._pc_managed_value != "mg":
                    return
                want_fmg = False
        elif self._preconditioner == "fmg":
            want_fmg = n_levels > 1
        else:  # "gamg" — explicit, always applied
            want_fmg = False

        # Native geometric FMG relies on DMCreateInjection between the refined
        # DMPlex levels. For a **single-field** (scalar / vector) discretisation
        # this is fragile: whether PETSc can build the injection depends on the
        # geometry×element-degree×refinement combination, and it fails at solve
        # time with err62 "Could not locate matching functional for injection"
        # on the common curved-shell cases (issue #276) as well as some flat
        # high-degree ones. The Stokes velocity sub-block (prefix
        # "fieldsplit_velocity_") is the validated, robust native-FMG path and is
        # unaffected. So never auto-route a single-field solver to native FMG —
        # fall back to GAMG. Geometric MG on a scalar/vector solver is available,
        # robustly, via ``underworld3.utilities.custom_mg.set_custom_fmg`` (own
        # barycentric/RBF prolongation + Galerkin coarse operators; no injection).
        if want_fmg and prefix == "":
            if self._preconditioner == "fmg" and uw.mpi.rank == 0:
                import warnings
                warnings.warn(
                    f"[{self.name}] preconditioner='fmg' is not supported on a "
                    f"single-field (scalar/vector) solver: native geometric FMG "
                    f"needs DMCreateInjection, which PETSc cannot reliably build "
                    f"on a refined DMPlex (issue #276). Falling back to GAMG. For "
                    f"geometric MG on this solver use "
                    f"underworld3.utilities.custom_mg.set_custom_fmg().",
                    stacklevel=2,
                )
            want_fmg = False

        # The option VALUES live in utilities.multigrid_options, which is the
        # single owner shared with the custom-P routes (custom_mg, rotated_bc) —
        # the routes are the same preconditioner reached three ways and must not
        # be configured from three places (#468). Coarse solve: the native
        # hierarchy is not rotated, so its coarse operator carries no inherited
        # null space and redundant+LU is right here.
        # `_managed_pc_options` makes the bundle respect a key the USER set while
        # still managing the ones they left alone. Before this, the bundle was
        # applied wholesale on every rebuild and the only escape was the
        # `_pc_user_override` latch above — which keys on `pc_type` ALONE, so a user
        # who set (say) `mg_levels_ksp_max_it` had it silently discarded unless they
        # also set `pc_type` to the value it already had. Explicit
        # `preconditioner="fmg"` was worse: it skips the latch entirely, so the
        # clearer the request the less control it carried.
        if want_fmg:
            multigrid_options.geometric_mg_bundle(
                smoother=self._mg_smoother_variant).apply(
                    PETSc.Options(), self.petsc_options_prefix + prefix,
                    owned=self._managed_pc_options)
            self._pc_managed_value = "mg"
        else:
            if self._preconditioner == "fmg" and n_levels <= 1 and uw.mpi.rank == 0:
                import warnings
                warnings.warn(
                    f"[{self.name}] preconditioner='fmg' requested but the mesh "
                    f"has no refinement hierarchy; falling back to GAMG. Build the "
                    f"mesh with refinement >= 1 to enable geometric multigrid.",
                    stacklevel=2,
                )
            multigrid_options.gamg_bundle().apply(
                PETSc.Options(), self.petsc_options_prefix + prefix,
                owned=self._managed_pc_options)
            self._pc_managed_value = "gamg"

    def _enforce_galerkin_for_geometric_mg(self):
        """Geometric multigrid in UW3 REQUIRES Galerkin (RAP) coarse operators.

        UW3 installs no residual/Jacobian callbacks on the coarse DMs of the
        refinement hierarchy, so PETSc cannot re-discretise the operator there.
        With Galerkin RAP the coarse operators are assembled as R*A*P from the
        fine operator and the coarse DMs are used only for interpolation — which
        is the only mode that works. If ``pc_type=mg`` is selected on a block we
        manage but Galerkin is unset (or explicitly ``none``), PETSc instead
        tries to build coarse operators itself and fails cryptically: PETSc
        error 73 ("KSPSetDM without ComputeOperators") in serial, or
        ``DMCoarsen -> DMAdaptMetric -> ParMmg`` (which is 3D-only) in parallel.

        So rather than let users trip those, force ``pc_mg_galerkin=both``
        whenever a managed block uses geometric MG, regardless of who selected
        it (user, harness, or our own auto/fmg bundle). A bare ``=None`` flag
        already engages RAP (reads back as ``''`` but is *present*); only a
        genuinely-unset or ``none`` value is overridden.
        """
        prefix = self._pc_option_prefix
        if prefix is None:
            return
        opts = self.petsc_options
        if opts.getString(f"{prefix}pc_type") != "mg":
            return
        gkey = f"{prefix}pc_mg_galerkin"
        if (not opts.hasName(gkey)) or opts.getString(gkey) == "none":
            if uw.mpi.rank == 0:
                import warnings
                warnings.warn(
                    f"[{self.name}] geometric multigrid (pc_type=mg) requires "
                    f"Galerkin coarse operators in UW3 — forcing {gkey}=both. "
                    f"(UW3 installs no coarse-DM operator callbacks, so coarse "
                    f"re-discretisation is unsupported and fails as PETSc error "
                    f"73 / ParMmg.)",
                    stacklevel=2,
                )
            opts[gkey] = "both"

    def _check_expression_meshes(self):
        """Check that all MeshVariable symbols in solver expressions
        belong to this solver's mesh.

        Raises a clear error if variables from a different mesh are
        found, rather than letting the JIT fail with a cryptic message.
        """
        from underworld3.function.expressions import extract_meshes

        solver_mesh = self.mesh

        # Collect all sympy expressions from the solver
        exprs = []

        if hasattr(self, 'bodyforce') and self.bodyforce is not None:
            if hasattr(self.bodyforce, 'atoms'):
                exprs.append(self.bodyforce)

        for bc in getattr(self, 'natural_bcs', []):
            for attr in ('fn_f', 'fn_F', 'fn_p'):
                fn = getattr(bc, attr, None)
                if fn is not None and hasattr(fn, 'atoms'):
                    exprs.append(fn)

        if hasattr(self, '_constitutive_model') and self._constitutive_model is not None:
            cm = self._constitutive_model
            # Check parameter expressions rather than cm.flux — the flux
            # property triggers tensor contraction which can fail for
            # some model/solver combinations before setup is complete.
            if hasattr(cm, 'Parameters'):
                for attr_name in dir(cm.Parameters):
                    if attr_name.startswith('_'):
                        continue
                    try:
                        val = getattr(cm.Parameters, attr_name)
                        if hasattr(val, 'atoms'):
                            exprs.append(val)
                    except (AttributeError, TypeError):
                        pass

        # Extract all meshes from all expressions
        foreign_meshes = set()
        for expr in exprs:
            meshes = extract_meshes(expr)
            for m in meshes:
                if m is not solver_mesh:
                    foreign_meshes.add(m)

        if foreign_meshes:
            raise ValueError(
                f"Solver expressions contain MeshVariable symbols from "
                f"{len(foreign_meshes)} foreign mesh(es). All variables in "
                f"a solver expression must belong to the solver's mesh. "
                f"Use var.copy_into() to transfer data before building expressions."
            )


        return

    @property
    def is_setup(self):
        """True when the solver's PETSc DM/DS/SNES state is consistent with
        the current settings.

        Backed by three fine-grained flags:

        - ``_needs_dm_rebuild``       — mesh or field layout changed
        - ``_needs_bc_reregister``    — boundary conditions changed
        - ``_needs_function_rewire``  — only pointwise functions changed

        Assigning ``self.is_setup = False`` pessimistically raises all three
        (safe default — forces full DM teardown on next _build). Assigning
        ``True`` clears all three.

        Narrower opt-in: directly assign ``_needs_function_rewire = True`` when
        only F0/F1/Jacobian pointwise functions changed (constitutive model or
        time-derivative manager swap). _build() uses the in-place
        PetscDSSetResidual/SetJacobian path for that case, skipping DM/SNES
        teardown entirely.
        """
        return not (
            self._needs_dm_rebuild
            or self._needs_bc_reregister
            or self._needs_function_rewire
        )

    @is_setup.setter
    def is_setup(self, value):
        if value:
            self._needs_dm_rebuild = False
            self._needs_bc_reregister = False
            self._needs_function_rewire = False
        else:
            self._needs_dm_rebuild = True
            self._needs_bc_reregister = True
            self._needs_function_rewire = True
            # A structural invalidation (mesh change / adaptivity / mesh-mover /
            # explicit _force_setup) means any stored solution no longer matches
            # the operators — drop the warm-start claim so the next solve()
            # cold-starts rather than warming off a stale iterate. Coefficient-only
            # updates (new viscosity, δ, BC values, time step) route through
            # _update_constants / a direct _needs_function_rewire and keep
            # has_solution, so continuation and time-stepping warm-start correctly.
            self._has_solution = False

    @property
    def has_solution(self):
        """``True`` when the solver holds a converged solution usable as a warm start.

        Set ``True`` only after a solve whose SNES reported a converged reason
        (``> 0``); ``False`` initially, after a diverged solve, and after any
        structural rebuild (mesh change / adaptivity / mesh-mover / explicit
        ``_force_setup`` — the ``is_setup = False`` invalidation hook). It
        survives coefficient changes (viscosity, yield softness :math:`\\delta`,
        boundary-condition *values*, time step) so parameter continuation and
        time-stepping warm-start correctly.

        Read-only status flag. Because a diverged solve leaves
        ``has_solution == False``, the next :meth:`solve` automatically
        cold-starts (Picard warm-up) rather than warming off a corrupted iterate.

        See :doc:`nonlinear-solver-homotopy-warmstart` (Layer 1).
        """
        return self._has_solution

    def _solution_is_trivially_zero(self):
        """True when the solution field is still identically zero.

        The design's *secondary* cold signal: a solver may be told to warm-start
        (``zero_init_guess=False``) while its solution variable has never been
        written, which is a cold start in everything but name. It matters because a
        viscoplastic tangent is undefined there — the plastic viscosity is
        :math:`\\tau_y / (2\\dot\\varepsilon_{II})`, so at :math:`v = 0` the
        *residual* is finite (the soft-min carries the infinite plastic branch to
        the viscous one) but its derivative is not, and assembling the consistent
        tangent produces NaN.

        Uses the PETSc vector norm, which is collective and therefore rank-uniform,
        so every rank reaches the same decision.
        """
        u = getattr(self, "u", None)
        if u is None:
            return False
        return float(u.vec.norm()) == 0.0

    def _resolve_zero_init_guess(self, zero_init_guess):
        """Resolve the tri-state ``zero_init_guess`` argument of ``solve()``.

        ``None`` (the default) auto-detects: cold when the solver holds no converged
        solution, warm when it does. Detection is safe by construction — guessing
        *cold* when a solution was in fact available costs one extra iteration from a
        good starting point, while the harmful direction (warming off stale field data
        after a remesh or a diverged solve) cannot happen, because
        :attr:`has_solution` is cleared by both.

        ``True`` forces a fresh start (discard any solution); ``False`` insists on
        warming from the current field values.
        """
        if zero_init_guess is None:
            return not self.has_solution
        return bool(zero_init_guess)

    def _solve_yield_homotopy(self, homotopy_options=None, verbose=False,
                              solve_kwargs=None):
        """Run ``solve(homotopy=True)``: a multi-solve δ-continuation on the yield law.

        The constitutive model advertises the homotopy (``supports_yield_homotopy``)
        and describes it (``_yield_homotopy_control()``: how to set δ, and which
        tangent to pair with it); the continuation driver marches δ from a large,
        benign value down to the sharp yield surface, warm-starting each step from the
        previous converged solution. See
        :func:`~underworld3.systems.yield_continuation.yield_continuation` and
        :doc:`nonlinear-solver-homotopy-warmstart` (Layer 2).
        """
        cm = self.constitutive_model
        if cm is None or not getattr(cm, "supports_yield_homotopy", False):
            raise TypeError(
                f"solve(homotopy=True) needs a constitutive model with a yield law to "
                f"sharpen, but {type(cm).__name__ if cm is not None else None} does not "
                f"advertise supports_yield_homotopy. Use a viscoplastic (or VEP) model, "
                f"or solve without homotopy."
            )
        from underworld3.systems.yield_continuation import yield_continuation
        # The march's own options win: an explicit homotopy_options["verbose"] is a
        # deliberate choice about the march, distinct from the solve's verbosity.
        options = dict(homotopy_options or {})
        options.setdefault("verbose", verbose)
        return yield_continuation(self, solve_kwargs=solve_kwargs, **options)

    def _record_convergence_status(self, converged=None):
        """Refresh :attr:`has_solution` from the just-completed solve.

        Called at the end of every ``solve()``. With ``converged`` unset the
        status is read from the SNES converged reason (``> 0`` ⇒ converged);
        the rotated free-slip path, which runs its own KSP loop rather than
        driving ``self.snes``, passes the flag from its result dict.
        """
        if converged is None:
            converged = self.snes is not None and self.snes.getConvergedReason() > 0
        self._has_solution = bool(converged)

    class _Unknowns:
        """
        Manager for solver unknown variables and derived quantities.

        This class manages the primary unknown variable (e.g., velocity, temperature)
        and provides automatic computation of derived quantities like velocity gradients,
        strain rates, and vorticity.

        Attributes
        ----------
        u : MeshVariable
            The primary unknown variable being solved for.
        DuDt : SemiLagrangian_DDt, optional
            Time derivative manager for advection-diffusion problems.
        DFDt : SemiLagrangian_DDt, optional
            Flux time derivative manager for viscoelastic problems.
        L : sympy.Matrix
            Velocity gradient tensor :math:`L_{ij} = \\partial u_i / \\partial x_j`.
        E : sympy.Matrix
            Strain rate tensor :math:`E = (L + L^T) / 2` (symmetric part of L).
        W : sympy.Matrix
            Vorticity tensor :math:`W = (L - L^T) / 2` (antisymmetric part of L).
        Einv2 : sympy.Expr
            Second invariant of strain rate :math:`\\sqrt{E_{ij} E_{ij} / 2}`.
        """

        def __init__(inner_self, _owning_solver):
            inner_self._owning_solver = _owning_solver
            inner_self._u = None
            inner_self._DuDt = None
            inner_self._DFDt = None

            inner_self._L = None
            inner_self._E = None
            inner_self._W = None
            inner_self._Einv2 = None

            return

        ## properties

        @property
        def u(inner_self):
            """Primary unknown variable (MeshVariable) being solved for."""
            return inner_self._u

        @u.setter
        def u(inner_self, new_u):

            if new_u is not None:
                inner_self._u = new_u
                inner_self._L = new_u.sym.jacobian(new_u.mesh.CoordinateSystem.N)

                # can build suitable E and W operators of the unknowns
                if inner_self._L.is_square:
                    inner_self._E = (inner_self._L + inner_self._L.T) / 2
                    inner_self._W = (inner_self._L - inner_self._L.T) / 2

                    inner_self._Einv2 = sympy.sqrt((sympy.Matrix(inner_self._E) ** 2).trace() / 2)

                inner_self._owning_solver.is_setup = False
            return

        @property
        def DuDt(inner_self):
            """Time derivative manager for the unknown variable (advection-diffusion)."""
            return inner_self._DuDt

        @DuDt.setter
        def DuDt(inner_self, new_DuDt):
            inner_self._DuDt = new_DuDt
            # Time-derivative manager affects F0/F1 pointwise functions only;
            # DM, fields, and BCs are unchanged. Use the in-place rewire path.
            inner_self._owning_solver._needs_function_rewire = True
            return

        @property
        def DFDt(inner_self):
            """Flux time derivative manager (viscoelastic problems)."""
            return inner_self._DFDt

        @DFDt.setter
        def DFDt(inner_self, new_DFDt):
            inner_self._DFDt = new_DFDt
            inner_self._owning_solver._needs_function_rewire = True
            return

        @property
        def E(inner_self):
            """Strain rate tensor: symmetric part of velocity gradient L."""
            return inner_self._E

        @property
        def L(inner_self):
            """Velocity gradient tensor: :math:`L_{ij} = \\partial u_i / \\partial x_j`."""
            return inner_self._L

        @property
        def W(inner_self):
            """Vorticity tensor: antisymmetric part of velocity gradient L."""
            return inner_self._W

        @property
        def Einv2(inner_self):
            """Second invariant of strain rate tensor."""
            return inner_self._Einv2

        @property
        def CoordinateSystem(inner_self):
            """Coordinate system of the underlying mesh."""
            return inner_self._owning_solver.mesh.CoordinateSystem

    def _object_viewer(self):
        '''This will add specific information about this object to the generic class viewer
        '''
        from IPython.display import Latex, Markdown, display
        from textwrap import dedent

        display(Markdown(fr"### Boundary Conditions"))

        display(Markdown(fr"This solver is formulated as {self.mesh.dim} dimensional problem with a {self.mesh.cdim} dimensional mesh"))

        return

    def _reset(self):

        self.natural_bcs = []
        self.essential_bcs = []
        # A teardown means the next solve is a different discrete problem: a resume
        # anchor from the old problem's ||F0|| must not survive into it.
        self._resume_abs_target = None

        if self.snes is not None:
            # Drop the instrumentation's references to this SNES's KSP hierarchy first,
            # or destroy() only decrements and the old hierarchy stays resident.
            if getattr(self, "_instrumentation", None) is not None:
                self._instrumentation.release()
            self.snes.destroy()
            self.snes = None

        if self.dm is not None:
            for coarse_dm in self.dm_hierarchy:
                coarse_dm.destroy()

            self.dm = None
            self.dm_hierarchy = [None]

        self.is_setup = False

        return

    @property
    def solve_report(self):
        """Difficulty / convergence record of the most recent solve (read-only).

        A :class:`~underworld3.systems.solve_report.SolveReport` populated by default after
        every ``solve()`` — no opt-in. ``None`` before the first solve. Exposes the converged
        reason, nonlinear/linear iteration counts, final and initial ‖F‖, the residual
        reduction and contraction ρ, and the residual ladder. See also ``solve_history`` and
        ``estimate_difficulty``.
        """
        return self._solve_report

    @solve_report.setter
    def solve_report(self, value):
        raise AttributeError("solve_report is read-only (set by solve()).")

    @property
    def solve_history(self):
        """Bounded trail of recent :class:`SolveReport`s (read-only, most recent last).

        A ``collections.deque`` (``maxlen=32``) so continuation / time-stepping loops can read
        difficulty trends without logging each solve themselves.
        """
        return self._solve_history

    @solve_history.setter
    def solve_history(self, value):
        raise AttributeError("solve_history is read-only (set by solve()).")

    def _capture_solve_report(self, *, bounded=False):
        """Record a SolveReport for the just-completed solve into ``self._solve_report`` and
        append it to ``self._solve_history``. Reads SNES state only (side-effect free).

        INVARIANT: every physical ``self.snes.solve(...)`` site must be followed by a call to
        this method, so that reporting covers every solve path — including any that returns
        early and bypasses ``_snes_solve_with_retries`` / ``_warn_on_divergence`` (e.g. the
        rotated-free-slip handoff in ``utilities/rotated_bc.py``). Do NOT anchor capture on
        ``_warn_on_divergence`` (some paths skip it).
        """
        from underworld3.systems.solve_report import (
            SolveReport, reason_string, contraction,
        )

        if getattr(self, "snes", None) is None:
            self._solve_report = None
            return None

        reason = int(self.snes.getConvergedReason())
        nl_its = int(self.snes.getIterationNumber())
        ksp_its = int(self.snes.getLinearSolveIterations())
        # Sanctioned swallows (each optional-field read below): PETSc raises from these
        # getters on SNES types/states that never computed the quantity (e.g. ksponly
        # before a function evaluation, builds without the counter). The report field
        # degrades to its honest "unavailable" value; nothing else is masked.
        try:
            fnorm = float(self.snes.getFunctionNorm())
        except Exception:
            fnorm = float("nan")
        try:
            fev = int(self.snes.getFunctionEvaluations())
        except Exception:
            fev = None
        try:
            hist = tuple(float(h) for h in self.snes.getConvergenceHistory()[0])
        except Exception:
            hist = ()
        fnorm0 = hist[0] if hist else None
        reduction = (fnorm / fnorm0) if (fnorm0 not in (None, 0.0)) else None
        rho = contraction(hist)

        # Sub-solve work, and whether a wall-clock guard cut this solve short. Absent
        # for a solver whose instrumentation has never attached (no solve yet).
        instrumentation = self._instrumentation
        sub = instrumentation.sub_reports() if instrumentation is not None else {}
        deadline_expired = (instrumentation is not None
                            and instrumentation.deadline_expired)

        report = SolveReport(
            reason=reason,
            reason_str=reason_string(reason),
            converged=reason > 0,
            nl_its=nl_its,
            ksp_its=ksp_its,
            fnorm=fnorm,
            fnorm0=fnorm0,
            reduction=reduction,
            rho=rho,
            fev=fev,
            history=hist,
            bounded=bool(bounded),
            sub=sub,
            deadline_expired=deadline_expired,
        )
        self._solve_report = report
        self._solve_history.append(report)
        return report

    def _capture_rotated_report(self, info):
        """Record a SolveReport for a rotated-free-slip solve, which runs its own
        ``ksp.solve()`` on the rotated operator (``utilities/rotated_bc.py``) and never
        touches ``self.snes`` — so the generic reader would see stale SNES state.

        The rotated result dict is the manual Newton loop's (``nonlinear_iterations``,
        ``ksp_its`` a per-increment LIST, an outer ``converged`` flag from the
        residual/step-norm tests, and ``ksp_reason`` the LAST increment's KSP code) —
        rendered with the KSP reason table, since the SNES table shares the integers
        but names different outcomes. A malformed dict raises: this reader and the
        dict in ``rotated_bc.py`` are a contract, and a silent fallback here
        previously masked a broken one."""
        from underworld3.systems.solve_report import SolveReport, ksp_reason_string

        info = info or {}
        reason = int(info.get("ksp_reason", 0))
        raw_its = info.get("ksp_its", 0)
        if isinstance(raw_its, (list, tuple)):
            ksp_its = int(sum(raw_its))                 # nonlinear: one count per Newton step
        else:
            ksp_its = int(raw_its)
        nl_its = int(info.get("nonlinear_iterations", 1))   # linear path = one outer solve
        if "converged" in info:
            # The nonlinear loop's own verdict. The last KSP reason alone would mislabel
            # a stalled Newton chain whose final linear solve happened to converge.
            converged = bool(info["converged"])
        else:
            converged = reason > 0
        rnorm = info.get("rnorm")
        fnorm = float(rnorm) if rnorm is not None else float("nan")
        rnorm0 = info.get("rnorm0")
        fnorm0 = float(rnorm0) if rnorm0 else None
        reduction = (fnorm / fnorm0) if (fnorm0 and fnorm == fnorm) else None
        report = SolveReport(
            reason=reason, reason_str=ksp_reason_string(reason), converged=converged,
            nl_its=nl_its, ksp_its=ksp_its, fnorm=fnorm,
            fnorm0=fnorm0, reduction=reduction, rho=None, fev=None, history=(),
            bounded=False,
        )
        self._solve_report = report
        self._solve_history.append(report)
        return report

    def guard(self, *, wall_per_step):
        r"""Bound the wall-clock time of each Newton step. Opt-in; changes termination.

        A hard nonlinear solve can grind for hours *inside a single Newton step*, and no
        iteration cap can stop it: the grind happens within one outer Krylov iteration,
        so the counter the cap watches never advances. Nor can a Python timer stop it —
        Python runs signal handlers only between bytecodes, and control is inside PETSc
        the whole time. The deadline therefore lives in PETSc's own convergence tests,
        where it is checked between the *inner* iterations of the fieldsplit blocks.

        When the budget runs out the solve stops with ``DIVERGED_LINEAR_SOLVE`` and
        ``solve_report.deadline_expired`` set, leaving the current iterate in the fields.
        It is a report, not an error: a continuation driver reads it and steps back.

        Parameters
        ----------
        wall_per_step : float
            Seconds of wall clock allowed per Newton step. The clock restarts at the
            beginning of every step, so a solve taking ``n`` steps may run for up to
            roughly ``n * wall_per_step`` seconds. Once the deadline fires it stays
            fired for the rest of that ``solve()`` call, so a Picard-to-Newton
            continuation or a warm-start retry cannot quietly buy itself a fresh
            budget.

        Notes
        -----
        The budget is enforced at inner-iteration granularity, so the overrun beyond it
        is at most the cost of one multigrid cycle. In parallel the expiry decision is
        reduced over the solver's communicator, because PETSc deadlocks if convergence
        tests return different verdicts on different ranks.

        Not available with rotated free-slip BCs: that path runs its own Krylov loop
        outside ``self.snes`` (``utilities/rotated_bc.py``), which the deadline cannot
        reach — so arming a guard there would look like protection and provide none.

        On a solver's FIRST solve the fieldsplit blocks do not exist until PETSc has set
        up the preconditioner, part-way through that solve, so the first preconditioner
        application runs unguarded. Every solve after that is fully covered. A driver
        that needs the first one covered should do one cheap solve before arming.

        Examples
        --------
        >>> stokes.guard(wall_per_step=150.0)
        >>> stokes.solve()
        >>> if stokes.solve_report.deadline_expired:
        ...     ...                      # too hard at these parameters; back off
        >>> stokes.unguard()

        See Also
        --------
        unguard : remove the deadline.
        estimate_difficulty : bound the *work* (an iteration count) instead.
        """
        if getattr(self, "_rotated_freeslip_bcs", None) \
                or getattr(self, "_fault_contact_faults", None):
            raise NotImplementedError(
                "guard() is not available with rotated free-slip / fault-contact "
                "BCs: that path runs "
                "its own Krylov loop outside self.snes (utilities/rotated_bc.py), so "
                "the deadline cannot reach it and the guard would be silently inert."
            )
        self._solver_instrumentation().arm(wall_per_step)

    def unguard(self):
        """Remove the wall-clock deadline set by :meth:`guard`, restoring PETSc defaults."""
        if self._instrumentation is not None:
            self._instrumentation.disarm()

    def _solver_instrumentation(self):
        """The solver's PETSc instrumentation, created on first use."""
        if self._instrumentation is None:
            from underworld3.systems.solver_health import SolverInstrumentation
            self._instrumentation = SolverInstrumentation()
        return self._instrumentation

    def estimate_difficulty(self, max_nl_its, *, warm=True, **solve_kwargs):
        """Run a bounded, resumable solve to *estimate solver difficulty* and return its report.

        Caps the solve at ``max_nl_its`` nonlinear iterations (for THIS call only), runs the
        normal ``solve()`` — which leaves the partial iterate in the fields — and returns the
        :class:`SolveReport` (``bounded=True``). The reported effort (iterations, residual
        reduction, contraction ρ) IS the difficulty estimate. Continue losslessly with another
        ``estimate_difficulty(warm=True)`` (a large cap runs it to completion); a chunked
        start/stop/restart chain terminates at the same point an uninterrupted solve would,
        because the chain anchors convergence to the original ‖F0‖.

        This bounds *solver work predictably* (an iteration count) — it is NOT a wall-time limit.

        Parameters
        ----------
        max_nl_its : int
            Cap on nonlinear (SNES) iterations for this call.
        warm : bool, default True
            ``True`` continues from the current fields (a restart within a chain);
            ``False`` starts a fresh chain from a zero initial guess.
        **solve_kwargs
            Forwarded to ``solve()`` (e.g. ``timestep``, ``picard`` for Stokes/VE).
            ``zero_init_guess`` is owned by ``warm`` and ``divergence_retries`` by the
            probe (always 0) — passing either raises.

        Returns
        -------
        SolveReport
            The (``bounded=True``) report for this capped solve; also available as
            ``self.solve_report``.

        Notes
        -----
        Not available with rotated free-slip BCs: that path solves outside
        ``self.snes``, so the cap and anchor cannot reach it. Under
        ``consistent_jacobian="continuation"`` the cap applies to EACH stage (Picard
        then Newton), so one probe may run up to twice ``max_nl_its``.
        """
        if getattr(self, "_rotated_freeslip_bcs", None) \
                or getattr(self, "_fault_contact_faults", None):
            raise NotImplementedError(
                "estimate_difficulty() is not available with rotated free-slip / "
                "fault-contact BCs: "
                "that path solves outside self.snes (utilities/rotated_bc.py), so the "
                "iteration cap and resume anchor cannot be applied to it."
            )
        if "zero_init_guess" in solve_kwargs:
            raise TypeError(
                "estimate_difficulty() sets the initial guess from warm= "
                "(warm=False starts a fresh chain from zero); do not pass zero_init_guess."
            )
        if solve_kwargs.get("divergence_retries"):
            raise TypeError(
                "estimate_difficulty() does not accept divergence_retries: a capped probe "
                "ends DIVERGED_MAX_IT by design, and retrying on it would run multiples of "
                "the advertised cap."
            )
        if not warm:
            self._resume_abs_target = None      # fresh chain: forget any prior anchor
        self._difficulty_probe = True
        self._difficulty_max_it = int(max_nl_its)
        try:
            self.solve(zero_init_guess=(not warm), **solve_kwargs)
        finally:
            self._difficulty_probe = False
            self._difficulty_max_it = None

        # Anchor lifecycle. A CONVERGED probe ends the chain: clear the anchor so it
        # cannot leak into a later chain on different physics. This also covers the
        # warm first call on an already-converged state — its ||F0|| is the *converged*
        # residual, and anchoring to tolerance*||F_converged|| would set an unreachable
        # target that silently degrades the chain to restart-relative semantics.
        # An UNCONVERGED (capped) probe with no anchor yet establishes the chain,
        # anchored to the original ||F0||: subsequent warm restarts — including a
        # large-cap call to run to completion — terminate at tolerance*||F0||, the same
        # point an uninterrupted solve would, regardless of where the caps fell. The
        # anchor is consulted ONLY under _difficulty_probe (see
        # _snes_solve_with_retries), so plain solves are unaffected.
        report = self._solve_report
        if report is not None and report.converged:
            self._resume_abs_target = None
        elif self._resume_abs_target is None and report is not None:
            f0 = report.fnorm0
            if f0:
                self._resume_abs_target = self.tolerance * float(f0)
        return report

    def get_snes_diagnostics(self):
        """
        Extract comprehensive SNES convergence diagnostics with string representations.

        Returns:
        --------
        dict
            Comprehensive convergence diagnostics including:
            - converged: bool - Whether solver converged
            - diverged: bool - Whether solver diverged
            - convergence_reason: int - Numerical convergence reason
            - convergence_reason_string: str - Human-readable convergence reason
            - snes_iterations: int - Number of SNES iterations
            - linear_iterations: int - Total number of linear iterations
            - zero_iterations: bool - Whether SNES took zero iterations
            - tolerances: dict - SNES tolerance settings
        """

        if not hasattr(self, 'snes') or self.snes is None:
            return {
                'error': 'SNES not initialized - call solve() first',
                'snes_available': False
            }

        # Get basic convergence info
        converged_reason = self.snes.getConvergedReason()
        snes_iterations = self.snes.getIterationNumber()
        linear_iterations = self.snes.getLinearSolveIterations()
        rtol, atol, stol, maxit = self.snes.getTolerances()

        # Determine convergence status
        converged = converged_reason > 0
        diverged = converged_reason < 0

        # Format "NAME - explanation" from the class-level table shared with
        # _warn_on_divergence.
        if converged_reason in self._convergence_reasons:
            _name, _explanation = self._convergence_reasons[converged_reason]
            convergence_reason_string = f"{_name} - {_explanation}"
        else:
            convergence_reason_string = f"UNKNOWN_CONVERGENCE_REASON_{converged_reason}"

        return {
            'snes_available': True,
            'converged': converged,
            'diverged': diverged,
            'convergence_reason': converged_reason,
            'convergence_reason_string': convergence_reason_string,
            'snes_iterations': snes_iterations,
            'linear_iterations': linear_iterations,
            'zero_iterations': snes_iterations == 0,
            'linear_solver_failed': converged_reason == -3,
            'nan_residual': converged_reason == -4,
            'tolerances': {
                'relative_tolerance': rtol,
                'absolute_tolerance': atol,
                'step_tolerance': stol,
                'max_iterations': maxit
            }
        }

    def check_snes_convergence(self, raise_on_divergence=True, print_diagnostics=False):
        """
        Check SNES convergence and optionally raise exceptions or print diagnostics.

        Parameters:
        -----------
        raise_on_divergence : bool
            Whether to raise an exception if solver diverged
        print_diagnostics : bool
            Whether to print diagnostic information

        Returns:
        --------
        dict
            SNES diagnostics

        Raises:
        -------
        RuntimeError
            If solver diverged and raise_on_divergence=True
        """

        diagnostics = self.get_snes_diagnostics()

        if not diagnostics.get('snes_available', False):
            if raise_on_divergence:
                raise RuntimeError(diagnostics.get('error', 'SNES diagnostics not available'))
            return diagnostics

        if print_diagnostics:
            print(f"\n=== SNES DIAGNOSTICS ===")
            print(f"Status: {'✓ CONVERGED' if diagnostics['converged'] else '✗ DIVERGED'}")
            print(f"Reason: {diagnostics['convergence_reason_string']}")
            print(f"Iterations: {diagnostics['snes_iterations']} SNES, {diagnostics['linear_iterations']} linear")

            tol = diagnostics['tolerances']
            print(f"Tolerances: rtol={tol['relative_tolerance']:.1e}, "
                  f"atol={tol['absolute_tolerance']:.1e}")

            # Issue-specific warnings
            if diagnostics['zero_iterations']:
                print(f"⚠️  WARNING: Zero SNES iterations!")
                print(f"   Possible scaling issues - consider geological scaling")

            if diagnostics['linear_solver_failed']:
                print(f"⚠️  LINEAR SOLVER FAILURE!")
                print(f"   Often caused by poor matrix conditioning")

        # Raise exception if requested and solver diverged
        if diagnostics['diverged'] and raise_on_divergence:
            error_msg = f"SNES solver diverged: {diagnostics['convergence_reason_string']}\n"
            error_msg += f"Iterations: {diagnostics['snes_iterations']} SNES, {diagnostics['linear_iterations']} linear"

            if diagnostics['zero_iterations']:
                error_msg += "\nZERO ITERATIONS: Scaling or tolerance issues likely"
                error_msg += "\nSUGGESTION: Try geological scaling"

            if diagnostics['linear_solver_failed']:
                error_msg += "\nLINEAR SOLVER FAILURE: Matrix conditioning problems"
                error_msg += "\nSUGGESTION: Check scaling or solver options"

            raise RuntimeError(error_msg)

        return diagnostics

    def solve_with_diagnostics(self,
                              check_convergence=True,
                              raise_on_divergence=False,
                              print_diagnostics=False,
                              **solve_kwargs):
        """
        Solve with automatic SNES convergence checking and diagnostics.

        Parameters:
        -----------
        check_convergence : bool
            Whether to check convergence after solving
        raise_on_divergence : bool
            Whether to raise exception on divergence
        print_diagnostics : bool
            Whether to print diagnostic information
        **solve_kwargs
            Additional arguments passed to solve()

        Returns:
        --------
        dict or None
            SNES diagnostics if check_convergence=True, None otherwise

        Raises:
        -------
        RuntimeError
            If solver diverged and raise_on_divergence=True
        """

        # Call the original solve method
        self.solve(**solve_kwargs)

        # Check convergence if requested
        if check_convergence:
            return self.check_snes_convergence(
                raise_on_divergence=raise_on_divergence,
                print_diagnostics=print_diagnostics
            )

        return None

    # SNES convergence reasons: code -> (NAME, explanation). The NAMES are the same
    # table as solve_report.REASON_STRINGS, kept here with an explanation string for
    # get_convergence_diagnostics (formats "NAME - explanation") and _warn_on_divergence
    # (uses NAME only). Both copies are pinned to petsc4py's enum by test_1055 — the
    # positive codes here were shifted by one until 2026-07 (there is no code 1, and a
    # step-norm stop was reported as CONVERGED_ITS).
    _convergence_reasons = {
        # Positive reasons = converged
        2: ("CONVERGED_FNORM_ABS", "||F|| < atol"),
        3: ("CONVERGED_FNORM_RELATIVE", "||F|| < rtol*||F_initial||"),
        4: ("CONVERGED_SNORM_RELATIVE", "||x|| < stol"),
        5: ("CONVERGED_ITS", "Maximum iterations reached"),
        # Zero = still iterating (shouldn't see after solve)
        0: ("CONVERGED_ITERATING", "Still iterating (unexpected after solve)"),
        # Negative reasons = diverged
        -1: ("DIVERGED_FUNCTION_DOMAIN", "Function domain error"),
        -2: ("DIVERGED_FUNCTION_COUNT", "Too many function evaluations"),
        -3: ("DIVERGED_LINEAR_SOLVE", "Linear solver failed"),
        -4: ("DIVERGED_FUNCTION_NANORINF", "||F|| is Not-a-Number or infinite"),
        -5: ("DIVERGED_MAX_IT", "Maximum iterations exceeded"),
        -6: ("DIVERGED_LINE_SEARCH", "Line search failed"),
        -7: ("DIVERGED_INNER", "Inner solve failed"),
        -8: ("DIVERGED_LOCAL_MIN", "Local minimum reached"),
        -9: ("DIVERGED_DTOL", "||F|| increased by divtol"),
        -10: ("DIVERGED_JACOBIAN_DOMAIN", "Jacobian calculation failed"),
        -11: ("DIVERGED_TR_DELTA", "Trust region delta too small"),
        -13: ("DIVERGED_OBJECTIVE_DOMAIN", "Objective function domain error"),
        -14: ("DIVERGED_OBJECTIVE_NANORINF", "Objective is Not-a-Number or infinite"),
    }

    def _warn_on_divergence(self, phase="solve"):
        """Check SNES convergence and warn on genuine divergence.

        Parameters
        ----------
        phase : str
            ``"picard"`` -- ignore ``DIVERGED_MAX_IT`` (truncation is
            intentional during the Picard-to-Newton transition).
            ``"solve"``  -- any divergence is a real problem.
        """

        reason = self.snes.getConvergedReason()

        if reason >= 0:
            return                          # converged -- nothing to report

        # Picard truncation (DIVERGED_MAX_IT) is expected and harmless
        if phase == "picard" and reason == -5:
            return

        # Bounded difficulty probe: hitting the iteration cap (DIVERGED_MAX_IT) is
        # the intended stop, not a failure — the effort is reported, not warned.
        if self._difficulty_probe and reason == -5:
            return

        its = self.snes.getIterationNumber()
        _entry = self._convergence_reasons.get(reason)
        reason_str = _entry[0] if _entry is not None else f"UNKNOWN({reason})"

        uw.pprint(
            f"\nSNES {phase} diverged after {its} iterations: {reason_str}\n"
            f"  The solution vector may not have been updated.\n"
            f"  Use  <solver>.petsc_options.setValue('snes_converged_reason', None)\n"
            f"  and  <solver>.petsc_options.setValue('snes_monitor', None)\n"
            f"  to investigate.\n"
        )

    def _snes_solve_with_retries(self, gvec, divergence_retries=0, verbose=False):
        """Call self.snes.solve(None, gvec) with warm-start retries on divergence.

        Useful for nonlinear problems whose residual has kinks (e.g. VEP at
        the yield surface Min/softmin cutoff) where Newton can land on a
        bad iterate that trips DIVERGED_MAX_IT or DIVERGED_LINE_SEARCH. A
        single warm-start re-solve from the just-computed iterate commonly
        steps off the kink.

        The first solve is always performed. If the SNES converged reason is
        negative, up to ``divergence_retries`` additional calls to
        ``snes.solve(None, gvec)`` are made; ``gvec`` is retained between
        calls so each retry is a warm start. Returns once the SNES reports
        converged or the retry budget is exhausted.

        Parameters
        ----------
        gvec : PETSc.Vec
            Global solution vector (modified in place by snes.solve).
        divergence_retries : int, default=0
            Maximum retries on DIVERGED. 0 preserves legacy behaviour.
            Typically 1 is enough for VEP kink-related divergence.
        verbose : bool, default=False
            Log each retry on rank 0.
        """
        # Attach the per-iteration callback dispatcher here (after all
        # setFromOptions in the solve path). No-op when no callbacks registered.
        self._attach_snes_update_hook()
        # Arm the residual-history buffer for THIS solve (reset clears stale data so the
        # reported contraction is computed on the current ladder only). Done here — every
        # solve funnels through this method and reads self.snes freshly — so a recreated
        # SNES (new object after a setup-dirtying _build) is armed automatically. Guarded:
        # not all PETSc builds expose it identically.
        try:
            self.snes.setConvergenceHistory(reset=True)
        except Exception:
            pass

        # Attach the sub-solve gauge (and the wall-clock deadline, if guard() armed
        # one) to the CURRENT SNES and clear its per-solve counters. Here rather than
        # in _build because a rebuilt solver gets a new SNES and everything attached to
        # the old one is dropped; every solve funnels through this method.
        self._solver_instrumentation().begin_solve(self.snes)

        # Single control point for the bounded/resumable difficulty solve. Runs AFTER
        # every per-solve setFromOptions (base and Stokes), so overrides here win. Gated
        # on _difficulty_probe so it is active ONLY inside estimate_difficulty — a plain
        # solve() is byte-for-byte unchanged and can never pick up a stale chain anchor.
        #   - iteration cap: cap nonlinear iterations for this probe;
        #   - resume anchor: once a chain is established (_resume_abs_target set from the
        #     first solve's ||F0||), terminate at the ORIGINAL ||F0|| via an absolute tol
        #     = tolerance*||F0||. This abs tol is LOOSER than the restart-relative
        #     rtol*||F_restart||, so it fires first — no need to zero rtol. So a chunked
        #     start/stop/restart terminates at the same point an uninterrupted solve would.
        # Tolerances are saved and restored so nothing leaks into later solves.
        _probe = bool(self._difficulty_probe)
        _saved_tol = None
        if _probe:
            _saved_tol = self.snes.getTolerances()
            if self._difficulty_max_it is not None:
                self.snes.setTolerances(max_it=int(self._difficulty_max_it))
            if self._resume_abs_target is not None:
                self.snes.setTolerances(atol=float(self._resume_abs_target))

        try:
            if self.consistent_jacobian == "continuation":
                self._continuation_solve(gvec, verbose=verbose)
            else:
                self.snes.solve(None, gvec)
            if divergence_retries > 0:
                for _r in range(divergence_retries):
                    reason = self.snes.getConvergedReason()
                    if reason >= 0:
                        break
                    if verbose and uw.mpi.rank == 0:
                        print(
                            f"SNES DIVERGED (reason={reason}); "
                            f"warm-start retry {_r + 1}/{divergence_retries}",
                            flush=True,
                        )
                    self.snes.solve(None, gvec)
        finally:
            if _saved_tol is not None:
                self.snes.setTolerances(rtol=_saved_tol[0], atol=_saved_tol[1],
                                        stol=_saved_tol[2], max_it=_saved_tol[3])
            # Default-on difficulty report — single tail so every path is covered,
            # INCLUDING a solve that raises: the report then reflects the failed
            # attempt's SNES state rather than leaving the previous solve's converged
            # report behind for recovery logic to misread. bounded=True marks a report
            # produced under an estimate_difficulty cap.
            self._capture_solve_report(bounded=_probe)

    def _continuation_solve(self, gvec, verbose=False):
        """Picard -> Newton continuation via the constants[]-routed alpha.

        Stage 1 solves with the frozen (Picard) tangent (alpha=0) to a loose
        tolerance to enter Newton's basin of attraction; stage 2 ramps to the
        consistent (Newton) tangent (alpha=1) and warm-starts to the requested
        tolerance. alpha is toggled through ``constants[]`` so neither stage
        triggers a JIT recompile (cf. Spiegelman et al. 2016; ASPECT).
        """
        rtol, atol, stol, max_it = self.snes.getTolerances()

        # Stage 1 — Picard (alpha=0), loose tolerance.
        self._set_newton_alpha(0.0)
        self.snes.setTolerances(rtol=max(self.newton_switch_rtol, rtol))
        self.snes.solve(None, gvec)
        if verbose and uw.mpi.rank == 0:
            print(f"continuation Picard: reason={self.snes.getConvergedReason()} "
                  f"it={self.snes.getIterationNumber()}", flush=True)

        # Stage 2 — Newton (alpha=1), requested tolerance, warm-started.
        self._set_newton_alpha(1.0)
        self.snes.setTolerances(rtol=rtol, atol=atol, stol=stol, max_it=max_it)  # restore
        self.snes.solve(None, gvec)
        if verbose and uw.mpi.rank == 0:
            print(f"continuation Newton: reason={self.snes.getConvergedReason()} "
                  f"it={self.snes.getIterationNumber()}", flush=True)

        # Restore a clean Picard tangent for any subsequent solve (next step).
        self._set_newton_alpha(0.0)

    @timing.routine_timer_decorator
    def _build(self,
                    verbose: bool = False,
                    debug: bool = False,
                    debug_name: str = None,
                    ):

        self._check_expression_meshes()

        if self.is_setup:
            return

        # Resolve the preconditioner choice (auto/fmg/gamg) against the current
        # mesh hierarchy and push the option bundle before the per-class
        # _setup_solver runs snes.setFromOptions(). No-op unless the solver
        # opted in via _pc_option_prefix (Stokes / scalar / vector).
        self._apply_preconditioner_options()
        self._enforce_galerkin_for_geometric_mg()

        # === Fast path 1: constants-only change ===
        # If JIT cache key matches, the compiled code is identical — only
        # constant values (dt_elastic, scalar viscosity, BDF coeffs) differ.
        # Refresh PetscDS constants and skip all rebuilding.
        #
        # BUGFIX(#122): must also respect _needs_dm_rebuild. A mesh deform
        # leaves equations (and therefore the JIT cache key) unchanged, but
        # the cached DM still carries the pre-deform coordinate layout.
        # Without this guard, Fast Path 1 fires and the solver solves on
        # stale coords, computing F(v_prev) ≈ 0 and "converging" in zero
        # iterations without updating the solution. Fast Path 2 already
        # checks this flag; Fast Path 1 was missing the same guard.
        if self.dm is not None and hasattr(self, '_last_jit_cache_key') \
                and not self._needs_dm_rebuild:
            self._setup_pointwise_functions(verbose, debug=debug, debug_name=debug_name)

            if hasattr(self, '_current_jit_cache_key') and \
               self._current_jit_cache_key == self._last_jit_cache_key:
                self._update_constants()
                self.is_setup = True
                if hasattr(self, "constitutive_model") and \
                   self.constitutive_model is not None and \
                   hasattr(self.constitutive_model, "_solver_is_setup"):
                    self.constitutive_model._solver_is_setup = True
                return

        # === Fast path 2: function rewire in place ===
        # Pointwise functions changed but DM structure is unchanged.
        # Re-JIT and call PetscDSSet* on the existing DS.
        if (self.dm is not None
                and not self._needs_dm_rebuild
                and not self._needs_bc_reregister
                and self._needs_function_rewire):
            if verbose and uw.mpi.rank == 0:
                print(f"Rewire solver pointwise functions in place", flush=True)

            self._setup_pointwise_functions(verbose, debug=debug, debug_name=debug_name)
            self._setup_solver(verbose, _rewire_only=True)

            self.is_setup = True

            # _last_jit_cache_key must track "which bundle the solver is
            # currently wired for", not "what the last full-build was".
            # Otherwise a C → S → C transition can make fast path 1 trigger
            # on a subsequent _build() call while the solver is still wired
            # for S — returning S's result for a C solve.
            if hasattr(self, '_current_jit_cache_key'):
                self._last_jit_cache_key = self._current_jit_cache_key
            return

        # === Full rebuild path — teardown DM/SNES and reconstruct ===
        # BUGFIX(#157): two PETSc objects were leaking on every is_setup=False
        # rebuild:
        #   (1) self.snes — _setup_solver() does
        #         self.snes = PETSc.SNES().create(...)
        #       unconditionally; the previous SNES (with its KSP, PC, and
        #       full GAMG hierarchy of coarse operator matrices) was just
        #       overwritten.
        #   (2) self.dm_hierarchy[:-1] — only self.dm == dm_hierarchy[-1]
        #       was destroyed here; coarse DMs from the previous build
        #       leaked.
        # Tight is_setup=False loops (notably SNES_Tensor_Projection.solve()
        # cycling 6 symmetric tensor components in 3D) accumulate one SNES
        # plus one full coarse-DM hierarchy per iteration — large enough at
        # Gadi scale to push past memory limits during stress recovery
        # postprocessing. _reset() already had the correct destroy
        # sequence; this brings _build() into line with it.
        # NB self.snes / self.dm_hierarchy may not exist yet on the first
        # build, so use getattr/hasattr-style guards rather than `is not None`.
        if getattr(self, "snes", None) is not None:
            if verbose and uw.mpi.rank == 0:
                print(f"Destroy solver SNES", flush=True)
            # Instrumentation holds petsc4py references to this SNES's KSP and its
            # fieldsplit blocks, which reference-count the PC and the whole multigrid
            # hierarchy. Drop them BEFORE the destroy or the old hierarchy survives
            # alongside the new one -- the leak BUGFIX(#157) above exists to prevent.
            if getattr(self, "_instrumentation", None) is not None:
                self._instrumentation.release()
            self.snes.destroy()
            self.snes = None

        if self.dm is not None:
            if verbose and uw.mpi.rank == 0:
                print(f"Destroy solver DM hierarchy", flush=True)

            # Clean up field decomposition cache (Stokes specific but safe to check here)
            if hasattr(self, "_pressure_is") and self._pressure_is is not None:
                self._pressure_is.destroy()
                self._pressure_is = None
            if hasattr(self, "_velocity_is") and self._velocity_is is not None:
                self._velocity_is.destroy()
                self._velocity_is = None

            if getattr(self, "_multiplier_is", None):
                for mis in self._multiplier_is.values():
                    mis.destroy()
                self._multiplier_is = {}

            if hasattr(self, "_subdict") and self._subdict:
                for name, (is_set, subdm) in self._subdict.items():
                    is_set.destroy()
                    subdm.destroy()
                self._subdict = {}

            if hasattr(self, "dm_hierarchy") and self.dm_hierarchy:
                # Destroys each level — including dm_hierarchy[-1] which
                # is self.dm — so no separate self.dm.destroy() needed.
                for coarse_dm in self.dm_hierarchy:
                    if coarse_dm is not None:
                        coarse_dm.destroy()
                self.dm_hierarchy = [None]
            else:
                self.dm.destroy()
            self.dm = None
            if hasattr(self, "_stokes_nullspace"):
                self._stokes_nullspace = None
            if hasattr(self, "_stokes_nullspace_basis"):
                self._stokes_nullspace_basis = ()
            if hasattr(self, "_velocity_rotation_nullspace"):
                self._velocity_rotation_nullspace = None
            if hasattr(self, "_constant_nullspace_obj"):
                self._constant_nullspace_obj = None

        if verbose:
            uw.pprint("Build pointwise functions")
        self._setup_pointwise_functions(verbose, debug=debug, debug_name=debug_name)
        if verbose:
            uw.pprint("Set up spatial discretisation")
        self._setup_discretisation(verbose)
        if verbose:
            uw.pprint("Setup solver")
        self._setup_solver(verbose)

        self.is_setup = True

        # Record cache key after full build — used by the fast path
        # on subsequent _build() calls to detect constants-only changes.
        if hasattr(self, '_current_jit_cache_key'):
            self._last_jit_cache_key = self._current_jit_cache_key

        return


    def _set_constants_on_ds(self, ds):
        """Pack current constant values and call PetscDSSetConstants.

        Parameters
        ----------
        ds : PETSc DS object
            The PetscDS to set constants on.
        """
        if not self.constants_manifest:
            return

        from underworld3.utilities._jitextension import _pack_constants
        import numpy as np

        values = _pack_constants(self.constants_manifest)

        cdef DS cds = ds
        cdef int n_constants = len(values)
        cdef double[::1] vals_view = np.ascontiguousarray(values, dtype=np.float64)
        CHKERRQ(PetscDSSetConstants(cds.ds, n_constants, <const PetscScalar*>&vals_view[0]))

    def _update_constants(self):
        """Re-pack current UWexpression values and call PetscDSSetConstants.

        Called before each solve() to ensure constants are current without
        requiring JIT recompilation.
        """
        if not self.constants_manifest or self.dm is None:
            return

        ds = self.dm.getDS()
        self._set_constants_on_ds(ds)

        # Also propagate to coarse DMs in multigrid hierarchy
        if hasattr(self, 'dm_hierarchy') and self.dm_hierarchy:
            for coarse_dm in self.dm_hierarchy[:-1]:
                try:
                    coarse_ds = coarse_dm.getDS()
                    self._set_constants_on_ds(coarse_ds)
                except Exception:
                    pass

    # Deprecate in favour of properties for solver.F0, solver.F1
    @timing.routine_timer_decorator
    def _setup_problem_description(self):
        raise RuntimeError("Contact Developers - shouldn't be calling SolverBaseClass _setup_problem_description")

    @timing.routine_timer_decorator
    def add_condition(self, f_id, c_type, conds, label, components=None):
        """
        Add a dirichlet or neumann condition to the mesh.

        This function prepares UW data to use PetscDSAddBoundary().

        Parameters
        ----------
        f_id: int
            Index of the solver's field (equation) to apply the condition.
            Note: The solvers field id is usually different to the mesh's field ids.
        c_type: string
            BC type. Either dirichlet (essential) or neumann (natural) conditions.
        conds: array_like of floats or a sympy.Matrix
            eg. For a 3D model with an unconstraint x component: (None, 5, 1.2) or sympy.Matrix([sympy.oo, 5, 1.2])
        label: string
            The label name to apply the BC. To find a label/boundary name run something like
            mesh.view()
        components: array_like, single int value or None.
            (optional) tuple, or int of active conds components to use. Use 'None' for all conds to be used.
            If 'None' and components in 'cond' equal sympy.oo or -sympy.oo those components won't be used.
            eg. For the 3D example cond = (2, 5, 1.2), components = (1,2) the x components is ignored and uncontrainted.
        """
        if not isinstance(f_id, int):
            raise TypeError("Error: f_id argument must be of type 'int' representing the solver's fields")

        if c_type not in ['dirichlet', 'neumann']:
            raise ValueError("'c_type' unknown. Value must be either 'dirichlet' or 'neumann'")

        self.is_setup = False
        import numpy as np
        import underworld3 as uw

        # process conds and error check
        if isinstance(conds, (tuple, list)):
            # remove all None for sympy.oo, and handle UWQuantity/Pint Quantity objects
            processed_conds = []
            for x in conds:
                if x is None:
                    processed_conds.append(sympy.oo)
                elif isinstance(x, uw.function.quantities.UWQuantity):
                    # Convert UWQuantity to SI base units (dimensionless number)
                    if hasattr(x, '_pint_qty'):
                        # Use Pint to convert to base units (m, s, kg, etc.)
                        base_qty = x._pint_qty.to_base_units()
                        processed_conds.append(base_qty.magnitude)
                    else:
                        # Fallback: use the value as-is
                        processed_conds.append(x.value)
                elif hasattr(x, 'magnitude') and hasattr(x, 'units'):
                    # Direct Pint Quantity (from 300 * uw.units("K"))
                    import pint
                    if isinstance(x, pint.Quantity):
                        base_qty = x.to_base_units()
                        processed_conds.append(base_qty.magnitude)
                    else:
                        processed_conds.append(x)
                else:
                    processed_conds.append(x)
            conds = processed_conds
        elif isinstance(conds, float):
            conds = (conds,)
        elif isinstance(conds, int):
            conds = (conds,)
        elif isinstance(conds, uw.function.quantities.UWQuantity):
            # Single UWQuantity value
            if hasattr(conds, '_pint_qty'):
                # Use Pint to convert to base units
                base_qty = conds._pint_qty.to_base_units()
                conds = (base_qty.magnitude,)
            else:
                # Fallback: use the value as-is
                conds = (conds.value,)
        elif hasattr(conds, 'magnitude') and hasattr(conds, 'units'):
            # Single Pint Quantity (from 300 * uw.units("K"))
            import pint
            if isinstance(conds, pint.Quantity):
                base_qty = conds.to_base_units()
                conds = (base_qty.magnitude,)
            else:
                raise ValueError(f"Unknown quantity type: {type(conds)}")
        elif isinstance(conds, sympy.Matrix):
            conds = conds.T
        else:
            raise ValueError("Unsupported BC conds: \n" +
                  "array_like,   i.e. conds = [None, 5, 1.2]\n" +
                  "UWQuantity,   i.e. conds = uw.quantity(10, 'metre')\n" +
                  "Pint Quantity, i.e. conds = 10*uw.units('metre')\n" +
                  "sympy.Matrix, i.e. conds = sympy.Matrix([sympy.oo, 5, 1.2])\n")

        if isinstance(components, (tuple, list, int)):
            import warnings
            warnings.warn(
                "The 'components' argument is deprecated; select components "
                "with None / sympy.oo entries in 'conds' instead, e.g. "
                "conds=(None, 5, 1.2)",
                DeprecationWarning, stacklevel=3)
            components = np.array(components, dtype=np.int32, ndmin=1)

        elif components is None:
            cpts_list = []
            for i, fn in enumerate(conds):
                if fn != sympy.oo and fn != -sympy.oo:
                    cpts_list.append(i)

            components = np.array(cpts_list, dtype=np.int32, ndmin=1)
        else:
            raise TypeError("Unsupported BC 'components' argument")

        # ======================================================================
        # Apply non-dimensional scaling to BC values if ND is enabled
        # IMPORTANT: Only scale numeric values (floats, UWQuantity), NOT symbolic
        # expressions. Symbolic expressions (like Gamma_N, v_soln.sym) go through
        # the standard unwrap() pipeline during JIT compilation.
        # ======================================================================
        if uw.is_nondimensional_scaling_active():
            # Get the field variable for this f_id
            field_var = None
            if f_id == 0:
                field_var = self.Unknowns.u
            elif f_id == 1 and hasattr(self.Unknowns, 'p'):
                field_var = self.Unknowns.p
            else:
                # For other field IDs, try to get from Unknowns (future-proofing)
                pass

            # If we have a field variable with a scaling coefficient, scale numeric BCs
            if field_var is not None and hasattr(field_var, 'scaling_coefficient'):
                scale = field_var.scaling_coefficient
                if scale != 1.0 and scale != 0.0:
                    # Scale ONLY numeric values: dimensional → ND by dividing by scale
                    # This converts e.g. T=1000K to T*=1000/1000=1.0 (ND)
                    # Symbolic expressions are left unchanged - they go through unwrap()
                    def is_numeric_only(val):
                        """Check if value is a pure number (no symbols)."""
                        if val == sympy.oo or val == -sympy.oo:
                            return False  # Don't scale infinity
                        if isinstance(val, (int, float)):
                            return True
                        if isinstance(val, sympy.Basic):
                            # Check if it has any free symbols (i.e., is it symbolic?)
                            # Pure numbers like sympy.Float(1.0) have no free_symbols
                            return len(val.free_symbols) == 0
                        return False

                    scaled_conds = []
                    for val in conds:
                        if val == sympy.oo or val == -sympy.oo:
                            # Don't scale infinity (unconstrained components)
                            scaled_conds.append(val)
                        elif is_numeric_only(val):
                            # Scale only pure numeric values
                            scaled_conds.append(val / scale)
                        else:
                            # Symbolic expression - leave unchanged, will go through unwrap()
                            scaled_conds.append(val)
                    conds = scaled_conds

        sympy_fn = sympy.Matrix(conds).as_immutable()

        from collections import namedtuple
        if c_type == 'neumann':
            # mesh.Gamma in a natural-BC expression resolves per boundary:
            # external boundaries keep the exact per-quadrature petsc_n[];
            # internal boundaries substitute the declared analytic normal
            # (petsc_n is orientation-ambiguous there — issue #327).
            sympy_fn = sympy.Matrix(
                self.mesh._resolve_boundary_normals(sympy_fn, label)
            ).as_immutable()
            BC = namedtuple('NaturalBC', ['f_id', 'components', 'fn_f', 'fn_F', 'fn_p', 'boundary', 'boundary_label_val', 'type', 'PETScID', 'fns'])
            self.natural_bcs.append(BC(f_id, components, sympy_fn, None, None, label, -1, "natural", -1, {}))
        elif c_type == 'dirichlet':
            BC = namedtuple('EssentialBC', ['f_id', 'components', 'fn', 'boundary', 'boundary_label_val', 'type', 'PETScID'])
            self.essential_bcs.append(BC(f_id, components,sympy_fn, label, -1,  'essential', -1))


    def _value_first_bc_args(self, method, conds, boundary, alias=None,
                             alias_name="g"):
        """Normalize BC arguments to the canonical value-first order.

        The canonical BC signature is ``method(conds, boundary, ...)`` with the
        prescribed datum named ``conds`` (Style Charter, API conventions;
        maintainer decisions D2/D3, 2026-07). Two legacy spellings are shimmed
        here, each with exactly one DeprecationWarning:

        * **boundary-first order** — detected conservatively: the first
          positional argument is a string (a boundary label; a BC datum is
          never a string — see :meth:`add_condition`) while the second is not.
          The two arguments are swapped.
        * **the** ``g=`` **keyword alias** for the datum — forwarded to
          ``conds``. Supplying both ``conds`` and ``g`` is an error.

        Returns the normalized ``(conds, boundary)`` pair; ``boundary`` is
        required to be a string on exit.
        """
        legacy_order = False
        legacy_alias = False
        if isinstance(conds, str) and not isinstance(boundary, str):
            conds, boundary = boundary, conds
            legacy_order = True
        if alias is not None:
            if conds is not None:
                raise TypeError(
                    f"{method}() received the boundary datum twice "
                    f"(as 'conds' and as '{alias_name}='); pass it once, "
                    f"as 'conds'")
            conds = alias
            legacy_alias = True
        if legacy_order or legacy_alias:
            # Name the legacy form the caller actually used (Copilot review
            # of #334: a one-size message misdescribed keyword-only calls).
            if legacy_order and legacy_alias:
                legacy_form = f"{method}(boundary, {alias_name}=...)"
            elif legacy_order:
                legacy_form = f"{method}(boundary, conds, ...) positional order"
            else:
                legacy_form = f"the '{alias_name}=' keyword of {method}()"
            import warnings
            warnings.warn(
                f"{legacy_form} is deprecated; "
                f"use {method}(conds, boundary, ...)",
                DeprecationWarning, stacklevel=3)
        if not isinstance(boundary, str):
            raise TypeError(
                f"{method}() requires a boundary label string; "
                f"got {type(boundary).__name__}")
        return conds, boundary

    # Use FE terminology note f_id is 0.
    @timing.routine_timer_decorator
    def add_essential_bc(self, conds, boundary, components=None):
        """
        Add an essential (Dirichlet) boundary condition.

        Alias for :meth:`add_dirichlet_bc`. Essential BCs constrain the
        solution to specified values at boundary nodes.

        Parameters
        ----------
        conds : array-like, float, or sympy.Matrix
            Boundary condition values. Use ``None`` or ``sympy.oo`` for
            unconstrained components.
        boundary : str
            Name of the boundary label (e.g., ``"Top"``, ``"Bottom"``).
            **Case-sensitive**: must match mesh boundary names exactly.
        components : array-like or None, optional
            Deprecated. Use ``None`` in ``conds`` for unconstrained components.

        Examples
        --------
        >>> # Scalar field: fix temperature at boundary
        >>> diffusion.add_essential_bc(300.0, "Top")

        >>> # Vector field: fix both velocity components
        >>> stokes.add_essential_bc([0.0, 0.0], "Bottom")

        >>> # Vector field: fix x-component only, leave y free
        >>> stokes.add_essential_bc([0.0, None], "Left")

        >>> # Symbolic expression as boundary condition
        >>> import sympy
        >>> x, y = stokes.mesh.X
        >>> stokes.add_essential_bc([sympy.sin(x), 0.0], "Top")

        See Also
        --------
        add_dirichlet_bc : Equivalent method (preferred name).
        add_natural_bc : For flux/traction boundary conditions.
        """
        self.add_condition(0, 'dirichlet', conds, boundary, components)
        return

    @timing.routine_timer_decorator
    def add_natural_bc(self, conds, boundary, components=None):
        """
        Add a natural (Neumann) boundary condition.

        Natural BCs specify flux or traction at boundaries. These are
        incorporated as surface integrals in the weak form rather than
        direct constraints on the solution.

        Parameters
        ----------
        conds : array-like, float, or sympy.Matrix
            Boundary condition values representing flux (scalar problems)
            or traction (vector problems).
        boundary : str
            Name of the boundary label (e.g., ``"Top"``, ``"Bottom"``).
            **Case-sensitive**: must match mesh boundary names exactly.
        components : array-like or None, optional
            Deprecated. Use ``None`` in ``conds`` for unconstrained components.

        Examples
        --------
        >>> # Scalar: specify heat flux at boundary (insulated if 0)
        >>> diffusion.add_natural_bc(0.0, "Left")  # Insulated boundary

        >>> # Scalar: specify inward heat flux
        >>> diffusion.add_natural_bc(100.0, "Bottom")

        >>> # Vector: apply traction to boundary
        >>> normal = stokes.mesh.CoordinateSystem.unit_e_0
        >>> stokes.add_natural_bc(pressure * normal, "Right")

        >>> # Free-slip on arbitrary curved surface (spherical models)
        >>> # Uses penalty method with surface normal from mesh.Gamma
        >>> import sympy
        >>> penalty = 1e5
        >>> Gamma = mesh.Gamma  # Surface normal vector field
        >>> Gamma_N = Gamma / sympy.sqrt(Gamma.dot(Gamma))  # Normalize
        >>> # Penalize normal velocity component, allow tangential slip
        >>> stokes.add_natural_bc(penalty * Gamma_N.dot(v.sym) * Gamma_N, "Upper")

        Notes
        -----
        For Stokes problems, natural BCs represent tractions
        :math:`\\mathbf{t} = \\boldsymbol{\\sigma} \\cdot \\mathbf{n}`.

        For free-slip boundary conditions, consider using
        :meth:`add_nitsche_bc` instead of the penalty approach shown
        above. Nitsche provides variationally consistent enforcement
        without penalty tuning and is more robust on spherical shells.

        See Also
        --------
        add_nitsche_bc : Nitsche free-slip (recommended for curved boundaries).
        add_dirichlet_bc : For fixed-value boundary conditions.
        """
        self.add_condition(0, 'neumann', conds, boundary, components)

    @timing.routine_timer_decorator
    def add_dirichlet_bc(self, conds, boundary, components=None):
        """
        Add a Dirichlet (essential) boundary condition.

        Dirichlet BCs fix the solution value at boundary nodes. This is
        the most common type of boundary condition for prescribing known
        values (e.g., fixed temperature, no-slip walls).

        Parameters
        ----------
        conds : array-like, float, or sympy.Matrix
            Boundary condition values. Use ``None`` or ``sympy.oo`` for
            unconstrained components (partial Dirichlet conditions).
        boundary : str
            Name of the boundary label (e.g., ``"Top"``, ``"Bottom"``).
            **Case-sensitive**: must match mesh boundary names exactly.
        components : array-like or None, optional
            Deprecated. Use ``None`` in ``conds`` for unconstrained components.

        Examples
        --------
        >>> # Scalar problem: fix temperature at boundaries
        >>> diffusion.add_dirichlet_bc(300.0, "Top")
        >>> diffusion.add_dirichlet_bc(500.0, "Bottom")

        >>> # Vector problem: no-slip walls (zero velocity)
        >>> stokes.add_dirichlet_bc([0.0, 0.0], "Top")
        >>> stokes.add_dirichlet_bc([0.0, 0.0], "Bottom")

        >>> # Free-slip: fix normal component, leave tangential free
        >>> stokes.add_dirichlet_bc([0.0, None], "Left")   # x=0 at left
        >>> stokes.add_dirichlet_bc([None, 0.0], "Bottom") # y=0 at bottom

        >>> # Lid-driven cavity: moving top boundary
        >>> stokes.add_dirichlet_bc([1.0, 0.0], "Top")

        >>> # Symbolic boundary condition
        >>> x, y = mesh.X
        >>> T_boundary = 300 + 100 * sympy.sin(x * sympy.pi)
        >>> diffusion.add_dirichlet_bc(T_boundary, "Top")

        Raises
        ------
        KeyError
            If ``boundary`` name doesn't match any mesh boundary label.
            Check ``mesh.boundaries.keys()`` for available boundary names.

        See Also
        --------
        add_essential_bc : Alias for this method.
        add_natural_bc : For flux/traction boundary conditions.
        """
        self.add_condition(0, 'dirichlet', conds, boundary, components)

    @property
    def F0(self):
        raise RuntimeError("Contact Developers - SolverBaseClass F0 is being used")

    @property
    def F1(self):
        raise RuntimeError("Contact Developers - SolverBaseClass F1 is being used")

    @property
    def u(self):
        """
        Primary unknown variable (MeshVariable) being solved for.

        For scalar problems (Poisson, advection-diffusion), this is typically a
        scalar field like temperature. For vector problems (Stokes), this is
        typically velocity.
        """
        return self.Unknowns.u

    @u.setter
    def u(self, new_u):
        self.Unknowns.u = new_u
        return

    @property
    def DuDt(self):
        """
        Time derivative manager for advection-diffusion problems.

        This is a :class:`~underworld3.systems.ddt.SemiLagrangian_DDt` object
        that handles material derivatives using semi-Lagrangian advection.
        """
        return self.Unknowns.DuDt

    @DuDt.setter
    def DuDt(self, new_du):
        self.Unknowns.DuDt = new_du
        return

    @property
    def DFDt(self):
        """
        Flux time derivative manager for viscoelastic problems.

        This is a :class:`~underworld3.systems.ddt.SemiLagrangian_DDt` object
        that handles time evolution of stress/flux fields.
        """
        return self.Unknowns.DFDt

    @DFDt.setter
    def DFDt(self, new_dF):
        self.Unknowns.DFDt = new_dF
        return


    @property
    def constitutive_model(self):
        """
        Constitutive model defining the material behavior.

        The constitutive model provides the stress-strain relationship
        (for Stokes) or diffusivity (for advection-diffusion). Can be set
        as either a class or an instance.

        See Also
        --------
        underworld3.constitutive_models : Available constitutive models.
        """
        return self._constitutive_model

    @constitutive_model.setter
    def constitutive_model(self, model_or_class):

        ### checking if it's an instance - it will need to be reset
        if isinstance(model_or_class, uw.constitutive_models.Constitutive_Model):
            self._constitutive_model = model_or_class
            self._constitutive_model.Unknowns = self.Unknowns
            self._constitutive_model._solver_is_setup = False
            # Only override the constitutive model's order if the solver has
            # an explicit VE order (> 0). Otherwise preserve the model's default.
            if self._order > 0:
                self._constitutive_model.order = self._order
            # Establish bidirectional reference so parameter changes can propagate to solver
            self._constitutive_model.Parameters._solver = self


        ### checking if it's a class
        elif type(model_or_class) == type(uw.constitutive_models.Constitutive_Model):
            self._constitutive_model = model_or_class(self.Unknowns)
            if self._order > 0:
                self._constitutive_model.order = self._order
            # Establish bidirectional reference so parameter changes can propagate to solver
            self._constitutive_model.Parameters._solver = self



        ### Raise an error if it's neither
        else:
            raise RuntimeError(
                "constitutive_model must be a valid class or instance of a valid class"
            )

        # If the constitutive model requires stress history (e.g. VEP), create the
        # DFDt infrastructure lazily. This means users don't need to choose between
        # Stokes and VE_Stokes — the solver adapts to the constitutive model.
        if self._constitutive_model.requires_stress_history and self.Unknowns.DFDt is None:
            self._create_stress_history_ddt(order=self._constitutive_model.order)

        # May not work due to flux being incomplete
        if self.Unknowns.DFDt is not None:
            self.Unknowns.DFDt.psi_fn = self._constitutive_model.flux.T



    @property
    def tau(self):
        r"""Computed flux from the constitutive model, projected onto the mesh.

        Returns a :class:`~underworld3.discretisation.MeshVariable` containing
        the flux that the solver actually computed. The variable is created
        lazily on first access and updated by solving a projection each time
        the property is read.

        For Stokes-family solvers, this is the deviatoric stress tensor
        :math:`\boldsymbol{\tau}`. For Poisson/diffusion, it is the
        diffusive flux :math:`\kappa \nabla u`. The projection handles
        derivative evaluation internally — users should use this property
        rather than manually reconstructing the flux from the constitutive
        formula.

        Subclasses may override this to return pre-computed values (e.g.
        VE_Stokes returns the stress stored during the solve).

        Returns
        -------
        MeshVariable
            Mesh variable containing the projected flux values.
            Access numerical data via ``.data`` or ``.array``, symbolic
            expression via ``.sym``.
        """

        if self._constitutive_model is None:
            raise RuntimeError(
                "No constitutive model set. Assign solver.constitutive_model first."
            )

        # Lazy initialization of projection infrastructure
        if not hasattr(self, '_tau_var') or self._tau_var is None:
            self._setup_tau_projection()

        # Project the constitutive flux onto the mesh variable
        # Transpose if needed: flux may be (dim,1) column but projector expects row
        flux = self._constitutive_model.flux
        if hasattr(flux, 'shape') and flux.shape[1] == 1 and flux.shape[0] > 1:
            flux = flux.T

        if getattr(self, '_tau_use_multicomponent', False):
            # Symmetric tensor: flatten flux to a row of independent components,
            # solve one multi-component projection, fan result back to SYM_TENSOR.
            # Only set uw_function on first call — the expression structure is
            # stable across timesteps, and constant values (dt_elastic, BDF coeffs)
            # flow through PetscDS constants[] without recompilation.
            if not getattr(self, '_tau_projector_initialised', False) or not self.constitutive_model._solver_is_setup:
                import sympy
                indep = self._tau_indep_indices
                row = sympy.Matrix([[flux[i, j] for (i, j) in indep]])
                self._tau_projector.uw_function = row
                self._tau_projector.smoothing = 0.0
                self._tau_projector_initialised = True
            self._tau_projector.solve()
            for k, (i, j) in enumerate(indep):
                vals = self._tau_projector.u.array[:, 0, k]
                self._tau_var.array[:, i, j] = vals
                if i != j:
                    self._tau_var.array[:, j, i] = vals
        else:
            self._tau_projector.uw_function = flux
            self._tau_projector.smoothing = 0.0
            self._tau_projector.solve()

        return self._tau_var

    def _setup_tau_projection(self):
        """Create the mesh variable and projector for tau (lazy init)."""

        import math
        flux = self._constitutive_model.flux
        dim = self.mesh.dim
        rows, cols = flux.shape

        # Determine variable type and create appropriate projection solver
        if rows == cols and rows == dim:
            # Symmetric tensor flux (Stokes / VE_Stokes stress).
            # User-facing: SYM_TENSOR so downstream .array[:, i, j] reads are unchanged.
            # Internal:    (1, Nc) MATRIX variable driven by SNES_MultiComponent_Projection,
            #              eliminating the per-component DM cycling of SNES_Tensor_Projection.
            self._tau_var = uw.discretisation.MeshVariable(
                f"tau_{self.instance_number}",
                self.mesh,
                (dim, dim),
                vtype=uw.VarType.SYM_TENSOR,
                degree=self.u.degree,
                continuous=True,
                varsymbol=r"{\tau}",
            )
            Nc = math.comb(dim + 1, 2)
            self._tau_indep_indices = [
                (i, j) for i in range(dim) for j in range(i, dim)
            ]
            self._tau_flat_var = uw.discretisation.MeshVariable(
                f"tau_flat_{self.instance_number}",
                self.mesh,
                (1, Nc),
                vtype=uw.VarType.MATRIX,
                degree=self.u.degree,
                continuous=True,
                varsymbol=r"{\tau_{\mathrm{flat}}}",
            )
            from underworld3.systems.solvers import SNES_MultiComponent_Projection
            self._tau_projector = SNES_MultiComponent_Projection(
                self.mesh,
                u_Field=self._tau_flat_var,
                n_components=Nc,
                degree=self.u.degree,
                verbose=False,
            )
            self._tau_use_multicomponent = True

        elif rows == dim and cols == 1:
            # Vector flux (Poisson heat flux)
            self._tau_var = uw.discretisation.MeshVariable(
                f"tau_{self.instance_number}",
                self.mesh,
                dim,
                vtype=uw.VarType.VECTOR,
                degree=self.u.degree,
                continuous=True,
                varsymbol=r"{\mathbf{q}}",
            )
            from underworld3.systems.solvers import SNES_Vector_Projection
            self._tau_projector = SNES_Vector_Projection(
                self.mesh, self._tau_var, verbose=False
            )

        elif cols == dim and rows == 1:
            # Transposed vector flux
            self._tau_var = uw.discretisation.MeshVariable(
                f"tau_{self.instance_number}",
                self.mesh,
                dim,
                vtype=uw.VarType.VECTOR,
                degree=self.u.degree,
                continuous=True,
                varsymbol=r"{\mathbf{q}}",
            )
            from underworld3.systems.solvers import SNES_Vector_Projection
            self._tau_projector = SNES_Vector_Projection(
                self.mesh, self._tau_var, verbose=False
            )

        else:
            raise RuntimeError(
                f"Cannot create tau projection for flux shape {flux.shape}. "
                f"Expected ({dim},{dim}) tensor, ({dim},1) or (1,{dim}) vector."
            )

    def validate_solver(self):
        """
        Checks to see if the required properties have been set.
        Over-ride this one if you want to check specifics for your solver"""

        name = self.__class__.__name__

        if not isinstance(self.u, uw.discretisation._MeshVariable):
            print(f"Vector of unknowns required")
            print(f"{name}.u = uw.discretisation.MeshVariable(...)")

        if not isinstance(self.constitutive_model, uw.constitutive_models.Constitutive_Model):
            print(f"Constitutive model required")
            print(f"{name}.constitutive_model = uw.constitutive_models...")

        return

    def get_dof_partition(self,
                          section_type: str,
                          filename: Optional[str | None] = None,
                          outputPath: Optional[str] = ""):
        """
        Obtains how the degrees of freedom (DOF) are distributed/divided among the processors and saves them in an h5 file.
        Parameters
        ----------
        section_type:
            Can be: "local" which includes DOFs from ghost points or "global" which differentiates DOFs from ghost points by having negative values.
        filename:
            Output file name. If None, will print out results; if set to a string, the final output file will be <filename>_<section_type>.u.h5.
        outputPath:
            Path of directory where data is saved. If left empty it will save the data in the current working directory.
        """

        self.validate_solver()  # mainly check if self.u is properly set

        u_id = self.Unknowns.u.field_id
        fname = None if filename is None else f"{filename}_{section_type}.u.h5"

        self._get_dof_partition_by_field_id(section_type    = section_type,
                                            field_id        = u_id,
                                            filename        = fname,
                                            outputPath      = outputPath)

        return


    def _get_dof_partition_by_field_id(self,
                                       section_type: str,
                                       field_id: int,
                                       filename: Optional[str | None] = None,
                                       outputPath: Optional[str] = ""):
        """
        Private version of get_dof_partition with field_id as an additional parameter.
        Parameters
        ----------
        section_type:
            Can be: "local" which includes DOFs from ghost points or "global" which differentiates DOFs from ghost points by having negative values.
        field_id:
            The field id
        filename:
            Output file name. If None, will print out results; if set to a string, resulting h5 file has the following keys: field_id, rank, dof.
        outputPath:
            Path of directory where data is saved. If left empty it will save the data in the current working directory.
        """

        import os
        import h5py
        import numpy as np

        # check if section type is valid
        if section_type not in ['local', 'global']:
            raise ValueError("'section_type' unknown. Value must be either 'local' or 'global'")

        # check if path exists
        if os.path.exists(os.path.abspath(outputPath)):  # easier to debug abs
            pass
        else:
            raise RuntimeError(f"{os.path.abspath(outputPath)} does not exist")

        # check if we have write access
        if os.access(os.path.abspath(outputPath), os.W_OK):
            pass
        else:
            raise RuntimeError(f"No write access to {os.path.abspath(outputPath)}")


        # get all points in the DAG of this partition
        if section_type == "local":
            section = self.mesh.dm.getLocalSection()
        elif section_type == "global":
            section = self.mesh.dm.getGlobalSection()

        # NOTE: negative DOFs mean that these are ghost ones and owned by a different process

        ptStart, ptEnd = section.getChart() # will give all DOFs including ghosts

        fdofs = [section.getFieldDof(pt, field_id) for pt in range(ptStart, ptEnd)]
        fdofs = np.array(fdofs)
        pos_dof_data = np.array([field_id, uw.mpi.rank, fdofs[fdofs > 0].sum()])

        if section_type == "global":
            neg_dof_data = np.array([field_id, uw.mpi.rank, fdofs[fdofs < 0].sum()])

        comm = uw.mpi.comm

        # Gather the arrays on rank 0
        gath_pos_dof_data = comm.gather(pos_dof_data, root = 0)
        if section_type == "global":
            gath_neg_dof_data = comm.gather(neg_dof_data, root = 0)

        # pack data and save to a dataframe for formatted opening
        if uw.mpi.rank == 0:
            gath_dof_data = np.vstack(gath_pos_dof_data)
            if section_type == "global":
                gath_dof_data = np.vstack([gath_pos_dof_data, gath_neg_dof_data])

            if filename is None: # print out
                print(f"Section type: {section_type}")
                print(f"| Field ID      | Rank           | # DOFs        |")
                print(f"| ---------------------------------------------- |")
                for i in range(gath_dof_data.shape[0]):
                    print(
                        f"| {gath_dof_data[i, 0]:<15}|{gath_dof_data[i, 1]:<15}|{gath_dof_data[i, 2]:<15}|"
                    )
                print(f"| ---------------------------------------------- |")
                print("\n", flush = True)

            else: # save
                with h5py.File(f"{outputPath}/{filename}", "w") as f:
                    f.create_dataset("field_id", data = gath_dof_data[:, 0])
                    f.create_dataset("rank", data = gath_dof_data[:, 1])
                    f.create_dataset("dof", data = gath_dof_data[:, 2])

        return

    def _assemble_volume_reaction(self, time=None, verbose=False):
        """RAW per-rank FEM VOLUME residual (no boundary terms) in the DM-local layout,
        as a numpy array.

        At an essential-BC (Dirichlet) node this residual IS the consistent boundary
        reaction — the integrated nodal flux :math:`\\int_\\Gamma (F\\cdot\\hat n)\\phi_i`
        (heat flux for a scalar diffusion solve, traction for Stokes). Interior nodes are
        ~0. General across scalar / vector / Stokes solvers: the current solution is
        gathered from ``self.fields`` when present (Stokes) else the single
        ``Unknowns.u`` field.

        NOTE: the returned array is NOT globally assembled. The DM has overlap=0, so each
        rank computes only its OWNED cells' contribution; a boundary node shared across a
        partition cut therefore holds only this rank's PARTIAL reaction. The complete
        reaction is assembled by the caller (``utilities.boundary_flux._desmear``) by
        SUMMING each rank's partial by coordinate — not by a hand-rolled localToGlobal.
        """
        cdef DM dm
        cdef Vec xvec
        cdef Vec fvec
        cdef DM _time_dm_reaction
        cdef PetscFormKey key
        cdef IS ccell_is
        cdef PetscReal residual_time = 0.0
        cdef PetscReal implicit_form_time = <PetscReal>-1.7976931348623157e308

        self._build(verbose, False, None)

        if time is not None:
            t_nd = self._nondimensional_time(time)
            _time_dm_reaction = self.dm
            UW_DMSetTime(_time_dm_reaction.dm, <PetscReal>t_nd)
            residual_time = <PetscReal>t_nd

        self.mesh.update_lvec()
        self.dm.setAuxiliaryVec(self.mesh.lvec, None)
        self._update_constants()

        gvec = self.dm.getGlobalVec()
        xlocal = self.dm.getLocalVec()
        flocal = self.dm.getLocalVec()
        gvec.setArray(0.0)
        xlocal.setArray(0.0)
        flocal.setArray(0.0)

        try:
            # gather the current solution into the global vector (field-structure agnostic)
            if getattr(self, "fields", None):
                for name, var in self.fields.items():
                    sgvec = gvec.getSubVector(self._subdict[name][0])
                    self._subdict[name][1].localToGlobal(var.vec, sgvec)
                    gvec.restoreSubVector(self._subdict[name][0], sgvec)
            else:
                _names, _iss, _subdms = self.dm.createFieldDecomposition()
                sgvec = gvec.getSubVector(_iss[0])
                _subdms[0].localToGlobal(self.Unknowns.u.vec, sgvec)
                gvec.restoreSubVector(_iss[0], sgvec)

            self.dm.globalToLocal(gvec, xlocal)

            dm = self.dm
            xvec = xlocal
            fvec = flocal
            # Constrained (Dirichlet) DOFs are ABSENT from the global vector,
            # so the localToGlobal/globalToLocal round trip above leaves them
            # ZERO in xlocal. Insert the essential boundary values before
            # integrating: without this the residual is evaluated against a
            # state whose boundary values are wrong wherever g != 0, and the
            # 'reaction' on inhomogeneous Dirichlet boundaries is garbage
            # (issue #407 — g=0 boundaries were accidentally correct).
            CHKERRQ(DMPlexInsertBoundaryValues(dm.dm, PETSC_TRUE, xvec.vec,
                                               residual_time, NULL, NULL, NULL))
            CHKERRQ(DMPlexSNESComputeResidualFEM(dm.dm, xvec.vec, fvec.vec, NULL))

            # Return the RAW local residual: each rank has computed its OWNED cells'
            # contribution to its local nodes (the DM has overlap=0, so a boundary node
            # shared across a partition cut holds only this rank's PARTIAL contribution).
            # The caller assembles the complete reaction by summing these partials across
            # ranks by coordinate (boundary_flux._desmear), consistent with the boundary-
            # mass gather — this reproduces the rock-solid volume integral at cut nodes.
            return np.array(flocal.array, copy=True)
        finally:
            self.dm.restoreLocalVec(flocal)
            self.dm.restoreLocalVec(xlocal)
            self.dm.restoreGlobalVec(gvec)

    def boundary_flux(self, boundary, mass="auto", remove_mean=False, normal=None):
        r"""Consistent boundary flux on ``boundary``, recovered from the essential-BC
        reaction of the last solve (the Consistent Boundary Flux method).

        Returns ``(xs, flux)`` with one entry per boundary node on this rank: for a
        **scalar** solver the outward normal flux :math:`F\cdot\hat n` (e.g. surface heat
        flux :math:`-k\,\partial T/\partial n`, whose boundary mean is the Nusselt
        number); for a **vector** solver the traction :math:`\sigma\cdot\hat n` (pass
        ``normal`` to get the scalar normal component :math:`\hat n\cdot\sigma\cdot\hat n`).

        ``mass`` de-smears the nodal reaction with ``"lumped"`` or ``"consistent"``
        boundary mass. ``"auto"`` (default) selects lumped recovery for 2D traces and
        3D P1 triangles, and the required consistent solve for 3D P2 triangles.
        ``remove_mean`` subtracts the boundary mean — leave ``False`` for a physical
        flux (the mean is the Nusselt number); ``True`` gives a gauge-free field.

        Three-dimensional recovery supports triangular P1/P2 traces; quadrilateral
        traces raise explicitly. Reaction and mass assembly are partition-independent.
        For vector fluxes, supply an analytic ``normal`` when strict partition
        independence of the normal projection is required; geometric facet-normal
        averaging at partition seams has a small pre-existing partition sensitivity.

        .. warning::
           On CURVED boundaries, P2 **vertex** values converge only slowly: the P2
           vertex basis has zero surface mean, so vertex reactions carry only the
           O(h) facet-geometry error, which the recovery faithfully reconstructs
           (measured: 93%→55% error under one refinement, while midpoints go
           2.8%→0.7%). Pointwise consumers on curved boundaries should use
           **edge-midpoint values** or integral/fitted quantities, never vertex
           values (issue #414). Flat boundaries are exact up to solver tolerance."""
        from underworld3.utilities.boundary_flux import boundary_flux as _bf
        return _bf(self, boundary, mass=mass, remove_mean=remove_mean, normal=normal)

    def boundary_flux_field(self, boundary, field, mass="auto",
                            remove_mean=False, scale=1.0, normal=None):
        r"""Write the consistent boundary flux (see :meth:`boundary_flux`) onto a scalar
        MeshVariable ``field`` at the boundary nodes (interior untouched), multiplied by
        ``scale``. This is the field hand-off for downstream machinery (surface heat
        flux for coupling, or — with ``remove_mean=True`` and ``scale=-1/(\Delta\rho g)``
        — dynamic topography). Returns ``field``.

        Note that ``scale`` is a generic multiplier, NOT the ``buoyancy_scale``
        taken by :meth:`dynamic_topography` / ``topography``: for topography the
        relationship is ``scale = -1 / buoyancy_scale`` (there the division by
        :math:`\Delta\rho\,g` and the minus sign are internal)."""
        from underworld3.utilities.boundary_flux import boundary_flux_field as _bff
        return _bff(self, boundary, field, mass=mass, remove_mean=remove_mean,
                    scale=scale, normal=normal)

## Specific to dimensionality


class SNES_Scalar(SolverBaseClass):
    r"""
    General scalar equation solver using PETSc SNES.

    Solves the scalar conservation problem for unknown :math:`u`:

    .. math::

        \nabla \cdot \mathbf{F}(u, \nabla u, \dot{u}, \nabla\dot{u})
        - f(u, \nabla u, \dot{u}, \nabla\dot{u}) = 0

    where :math:`f` is a source term, :math:`\mathbf{F}` is a flux term relating
    :math:`u` to its gradients :math:`\nabla u`, and :math:`\dot{u}` is the
    Lagrangian time derivative.

    The unknown :math:`u` is a scalar mesh variable, and :math:`f`, :math:`\mathbf{F}`
    are arbitrary sympy expressions of mesh coordinate variables.

    This class is the base layer for building solvers that translate physical
    conservation laws into this general mathematical form.

    Parameters
    ----------
    mesh : underworld3.discretisation.Mesh
        The computational mesh.
    u_Field : MeshVariable, optional
        Pre-existing scalar field variable. If None, creates a new variable.
    degree : int, default=2
        Polynomial degree for finite element discretization.
    verbose : bool, default=False
        Enable verbose solver output (monitors convergence).
    DuDt : SemiLagrangian or Lagrangian, optional
        Time derivative handler for the unknown (advection-diffusion problems).
    DFDt : SemiLagrangian or Lagrangian, optional
        Time derivative handler for flux (viscoelastic problems).

    Attributes
    ----------
    u : MeshVariable
        The scalar unknown being solved for.
    F0 : UWexpression
        Source/force term :math:`f`.
    F1 : UWexpression
        Flux term :math:`\mathbf{F}`.
    constitutive_model : Constitutive_Model
        Material model defining flux-gradient relationship.
    tolerance : float
        Solver convergence tolerance.

    Examples
    --------
    >>> import underworld3 as uw
    >>> mesh = uw.meshing.StructuredQuadBox(elementRes=(16, 16))
    >>> poisson = uw.systems.Poisson(mesh)
    >>> poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    >>> poisson.constitutive_model.Parameters.diffusivity = 1.0
    >>> poisson.f = 1.0  # Source term
    >>> poisson.add_dirichlet_bc(0.0, "Bottom")
    >>> poisson.solve()

    See Also
    --------
    SNES_Vector : For vector-valued equations.
    SNES_Stokes_SaddlePt : For coupled velocity-pressure (Stokes) problems.
    """

    @timing.routine_timer_decorator
    def __init__(self,
                 mesh     : uw.discretisation.Mesh,
                 u_Field  : uw.discretisation.MeshVariable = None,
                 degree: int = 2,
                 verbose    = False,
                 DuDt          : Union[uw.systems.ddt.SemiLagrangian, uw.systems.ddt.Lagrangian] = None,
                 DFDt          : Union[uw.systems.ddt.SemiLagrangian, uw.systems.ddt.Lagrangian] = None,
                 ):

        super().__init__(mesh)

        ## Keep track

        ## Todo: some validity checking on the size / type of u_Field supplied
        if u_Field is None:
            # A scalar unknown has one component. (MeshVariable already forced
            # this: vtype=SCALAR overrides num_components, so the old
            # num_components=mesh.dim here was ignored, never over-allocated —
            # issue #367.)
            self.Unknowns.u = uw.discretisation.MeshVariable( mesh=mesh, num_components=1,
                                                      varname="Us{}".format(SNES_Scalar._obj_count),
                                                      vtype=uw.VarType.SCALAR, degree=degree, )
        else:
            self.Unknowns.u = u_Field
        self.Unknowns.DuDt = DuDt
        self.Unknowns.DFDt = DFDt

        self.verbose = verbose
        self._tolerance = 1.0e-4

        # Here we can set some defaults for this set of KSP / SNES solvers

        ## FAST as possible for simple problems:
        ## MG,

        # ROBUST and general GAMG, heavy-duty solvers in the suite

        # Participate in the auto FMG/GAMG switch (see the `preconditioner`
        # property). The pc/mg keys below are the GAMG default;
        # _apply_preconditioner_options() upgrades this block to geometric FMG
        # at build time when the mesh carries a refinement hierarchy, and
        # otherwise leaves these defaults in place.
        self._pc_option_prefix = ""

        self.petsc_options["snes_type"] = "newtonls"
        self._push_managed_option("ksp_type", "gmres")
        self._push_managed_option("pc_type", "gamg")
        self._push_managed_option("pc_gamg_type", "agg")
        self._push_managed_option("pc_gamg_repartition", True)
        self._push_managed_option("pc_mg_type", "additive")
        self._push_managed_option("pc_gamg_agg_nsmooths", 2)
        self._push_managed_option("mg_levels_ksp_max_it", 3)
        self._push_managed_option("mg_levels_ksp_converged_maxits", None)

        self.petsc_options["snes_rtol"] = 1.0e-4
        self._push_managed_option("mg_levels_ksp_max_it", 3)

        if self.verbose == True:
            self.petsc_options["ksp_monitor"] = None
            self.petsc_options["snes_converged_reason"] = None
            self.petsc_options["snes_monitor_short"] = None
        else:
            self.petsc_options.delValue("ksp_monitor")
            self.petsc_options.delValue("snes_monitor")
            self.petsc_options.delValue("snes_monitor_short")
            self.petsc_options.delValue("snes_converged_reason")

        self.dm = None


        self.essential_bcs = []
        self.natural_bcs = []
        self.bcs = self.essential_bcs
        self.boundary_conditions = False

        self.verbose = verbose

        self._rebuild_after_mesh_update = self._build  # Maybe just reboot the dm

        # Some other setup

        self.mesh._equation_systems_register.append(self)

        self.is_setup = False

        # Optional constant nullspace for pure-Neumann scalar
        # problems (e.g. an equidistribution / mesh-motion
        # potential). Off by default; see ``constant_nullspace``.
        self._constant_nullspace = False
        # Cached constant PETSc.NullSpace — built lazily in
        # _attach_constant_nullspace and invalidated on DM/SNES rebuild.
        self._constant_nullspace_obj = None

    @property
    def constant_nullspace(self):
        """Attach a constant nullspace to the Jacobian before solve.

        For a scalar problem with only natural (Neumann) boundary
        conditions the operator is singular up to an additive
        constant. Set ``True`` to attach a constant
        ``PETSc.NullSpace`` to the Jacobian (and its transpose /
        the preconditioner), which both projects the constant mode
        out of the Krylov solve and makes PETSc remove the
        (consistent) component of the RHS — the scalar analogue of
        the Stokes pressure-nullspace handling. The RHS must be
        compatible (zero mean) for the Neumann problem to be
        solvable.
        """
        return self._constant_nullspace

    @constant_nullspace.setter
    def constant_nullspace(self, value):
        self._constant_nullspace = bool(value)

    def _attach_constant_nullspace(self):
        """Attach a constant nullspace to the (already set-up) SNES
        Jacobian. Scalar analogue of ``_attach_stokes_nullspace``."""
        if not self._constant_nullspace:
            return

        # A constant nullspace is only valid for a pure-Neumann operator.
        # If essential (Dirichlet) BCs pin the field, the constant mode is
        # NOT in the operator's nullspace and projecting it out would
        # corrupt the solution — guard exactly as the Stokes pressure
        # nullspace guards against pressure Dirichlet BCs.
        if len(self.essential_bcs) > 0:
            boundaries = ", ".join(sorted(
                {str(getattr(bc, "boundary", "?")) for bc in self.essential_bcs}))
            raise ValueError(
                "constant_nullspace=True is only valid for pure-Neumann "
                "scalar problems, but essential (Dirichlet) boundary "
                f"conditions are present on: {boundaries}. Remove them or "
                "set constant_nullspace=False."
            )

        self.snes.setUp()

        jacobian = self.snes.getJacobian()
        operator_matrix = jacobian[0]
        preconditioner_matrix = jacobian[1] if len(jacobian) > 1 else None

        # Cache the constant nullspace on the instance and reuse it across
        # solves (it depends only on the comm); invalidated to None when the
        # DM/SNES is rebuilt (see _build).
        nullspace = self._constant_nullspace_obj
        if nullspace is None:
            nullspace = PETSc.NullSpace().create(
                constant=True, comm=self.dm.comm)
            self._constant_nullspace_obj = nullspace

        operator_matrix.setNullSpace(nullspace)
        operator_matrix.setTransposeNullSpace(nullspace)
        # GAMG (the default PC) builds its coarse hierarchy from the
        # near-nullspace; for a pure-Neumann operator the constant
        # mode MUST be supplied here or the PC setup fails
        # (DIVERGED_LINEAR_SOLVE at 0 iterations on re-solves).
        operator_matrix.setNearNullSpace(nullspace)

        if preconditioner_matrix is not None:
            preconditioner_matrix.setNullSpace(nullspace)
            preconditioner_matrix.setTransposeNullSpace(nullspace)
            preconditioner_matrix.setNearNullSpace(nullspace)

        if self.verbose and uw.mpi.rank == 0:
            print(f"SNES_Scalar ({self.name}): attached constant "
                  f"nullspace", flush=True)

    @property
    def petsc_use_constant_nullspace(self):
        """Backwards-compatible alias for :attr:`constant_nullspace`.

        The manifold-PDE work (PR #202) introduced this name on an
        older base; ``development`` independently landed the same
        capability as :attr:`constant_nullspace` (with an internal
        pure-Neumann guard and a cached nullspace object). Both names
        now refer to that single canonical implementation.
        """
        return self.constant_nullspace

    @petsc_use_constant_nullspace.setter
    def petsc_use_constant_nullspace(self, value):
        self.constant_nullspace = value

    @property
    def tolerance(self):
        """
        Solver convergence tolerance for SNES and KSP.

        Setting this value automatically configures related PETSc tolerances:
        - ``snes_rtol``: Set to ``tolerance``
        - ``ksp_rtol``: Set to ``tolerance * 0.1``
        - ``ksp_atol``: Set to ``tolerance * 1e-6``

        Returns
        -------
        float
            Current solver tolerance.

        Examples
        --------
        >>> solver.tolerance = 1e-6  # Tighter convergence
        >>> solver.solve()
        """
        return self._tolerance

    @tolerance.setter
    def tolerance(self, value):
        self._tolerance = value
        self.petsc_options["snes_rtol"] = self._tolerance
        self.petsc_options["ksp_rtol"] = self._tolerance * 1.0e-1
        self.petsc_options["ksp_atol"]  = self._tolerance * 1.0e-6

    @timing.routine_timer_decorator
    def _setup_discretisation(self, verbose=False):

        # Grab the mesh
        mesh = self.mesh

        import xxhash
        import numpy as np

        xxh = xxhash.xxh64()
        xxh.update(np.ascontiguousarray(mesh.X.coords))
        mesh_dm_coord_hash = xxh.intdigest()

        # if we already set up the dm and the coordinates in the mesh dm have not
        # changed then we do not need to do everything here

        if self.dm is not None and self.mesh_dm_coordinate_hash == mesh_dm_coord_hash:
            if verbose and uw.mpi.rank == 0:
                print("SNES_Scalar: Discretisation does not need to be rebuilt", flush=True)
            return

        # Keep a note of the coordinates that we use for this setup
        self.mesh_dm_coordinate_hash = mesh_dm_coord_hash


        degree = self.u.degree
        mesh = self.mesh

        if self.verbose:
            print(f"{uw.mpi.rank}: Building dm for {self.name}")

        if mesh.qdegree < degree:
            print(f"Caution - the mesh quadrature ({mesh.qdegree})is lower")
            print(f"than {degree} which is required by the {self.name} solver")

        self.dm_hierarchy = mesh.clone_dm_hierarchy()
        self.dm = self.dm_hierarchy[-1]

        if self.verbose:
            print(f"{uw.mpi.rank}: Building FE / quadrature for {self.name}", flush=True)

        # create private variables using standard quadrature order from the mesh

        options = PETSc.Options()
        options.setValue("private_{}_petscspace_degree".format(self.petsc_options_prefix), degree) # for private variables
        options.setValue("private_{}_petscdualspace_lagrange_continuity".format(self.petsc_options_prefix), self.u.continuous)
        options.setValue("private_{}_petscdualspace_lagrange_node_endpoints".format(self.petsc_options_prefix), False)

        # Should del these when finished

        self.petsc_fe_u = PETSc.FE().createDefault(mesh.dim, 1, mesh.isSimplex, mesh.qdegree, "private_{}_".format(self.petsc_options_prefix), PETSc.COMM_SELF,)
        self.petsc_fe_u_id = self.dm.getNumFields()
        self.dm.setField( self.petsc_fe_u_id, self.petsc_fe_u )
        self.petsc_fe_u.setName("_scalar_unknown_")

        self.is_setup = False

        if self.verbose:
            print(f"{uw.mpi.rank}: Building DS for {self.name}")

        ## This part is done once on the solver dm ... not required every time we update the functions ...
        ## the values of the natural bcs can be updated

        self.dm.createDS()

        # set functions
        cdef int ind=1
        cdef int [::1] comps_view  # for numpy memory view
        cdef DM cdm = self.dm
        cdef DS ds =  self.dm.getDS()
        cdef PtrContainer ext = self.compiled_extensions

        # TODO(BUG): natural BC on an INTERNAL surface is partition-dependent.
        # An interior facet has two support cells; the FE-assembly DM is kept
        # non-overlapped on purpose (overlap double-counts volume assembly via
        # LocalToGlobal+ADD — see discretisation_mesh.py ~L760). At a partition
        # seam ~few interior facets have only ONE local support cell, and PETSc's
        # DMPlexComputeBdResidual_Single_Internal attributes the whole per-facet
        # elemVec to support[0]'s cell closure (plexfem.c ~L4928) with the facet
        # normal oriented outward from support[0] (dmfieldds.c ~L790). When the
        # local-only support cell is the OPPOSITE side from the serial support[0],
        # the integral is the same value but is scattered through a different
        # cell closure; closure DOFs that are non-owned on this rank are added
        # locally then dropped at global assembly → the assembled load UNDER-counts
        # by ~0.027% (the force *function* integral / RMS is identical to machine
        # precision). In an ill-conditioned solve (augmented Stokes_Constrained)
        # this amplifies to ~0.1% velocity.
        #
        # Three cheap workarounds were TESTED and do NOT work:
        #   * editing the boundary LABEL (owner-only / ghost-strip): no-op — the
        #     ghost facet copies already contribute nothing (PETSc integrates each
        #     facet once), so stripping them is bit-identical;
        #   * a 1-cell partition OVERLAP on the assembly DM: no-op — verified the
        #     overlap propagates (+ghost cells) yet F0 is bit-identical, so overlap
        #     does NOT recover the dropped DOFs (and it double-counts volume anyway);
        #   * assembling the surface load on a codim-1 SUBMESH (extract_surface):
        #     blocked — UW3 FE on an embedded surface fails at setup because the
        #     gradient/flux term has cdim components but is reshaped to dim
        #     (the manifold vector-FE gap).
        # A deterministic support[0] cannot be imposed from UW3 (support order
        # follows the DMPlex point partition). Genuine fixes, all substantial:
        #   (a) MANUALLY assemble the boundary load (per-facet ∫ f·φ keyed by GLOBAL
        #       velocity DOF, each facet once) into a partition-independent vector
        #       and inject it via a guarded SNES function wrap — reimplements the bd
        #       residual but sidesteps PETSc's support[0] scatter;
        #   (b) fix UW3 manifold vector-FE, then assemble the load on the surface
        #       submesh and map to the parent by coincident-node coordinate;
        #   (c) a PETSc patch making the interior-facet bd residual scatter to the
        #       OWNED support cell.
        # See planning file (underworld.md, Bugs) and
        # project_stokes_constrained_parallel_session.
        for index,bc in enumerate(self.natural_bcs):

            components = bc.components
            if uw.mpi.rank == 0 and self.verbose:
                print("Setting bc {} ({})".format(index, bc.type))
                print(" - field:      {}".format(bc.f_id))
                print(" - components: {}".format(bc.components))
                print(" - boundary:   {}".format(bc.boundary))
                print(" - fn:         {} ".format(bc.fn_f))

            boundary = bc.boundary
            value = mesh.boundaries[bc.boundary].value
            ind = value

            bc_label = self.dm.getLabel(boundary)
            bc_is = bc_label.getStratumIS(value)
            self.natural_bcs[index] = self.natural_bcs[index]._replace(boundary_label_val=value)

            # use type 5 bc for `DM_BC_ESSENTIAL_FIELD` enum
            # use type 6 bc for `DM_BC_NATURAL_FIELD` enum

            bc_type = 6
            num_constrained_components = bc.components.shape[0]
            comps_view = bc.components
            bc = PetscDSAddBoundary_UW(cdm.dm,
                                bc_type,
                                str(boundary+f"{bc.components}").encode('utf8'),
                                "UW_Boundaries".encode('utf8'),   # consolidated boundary label
                                bc.f_id,  # field ID in the DM
                                num_constrained_components,
                                <const PetscInt *> &comps_view[0],
                                <void (*)() noexcept>NULL,
                                NULL,
                                1,
                                <const PetscInt *> &ind,
                                NULL, )

            self.natural_bcs[index] = self.natural_bcs[index]._replace(PETScID=bc, boundary_label_val=ind)


        for index,bc in enumerate(self.essential_bcs):

            component = bc.components
            if uw.mpi.rank == 0 and self.verbose:
                print("Setting bc {} ({})".format(index, bc.type))
                print(" - field:      {}".format(bc.f_id))
                print(" - component:  {}".format(bc.components))
                print(" - boundary:   {}".format(bc.boundary))
                print(" - fn:         {} ".format(bc.fn))

            boundary = bc.boundary
            value = mesh.boundaries[bc.boundary].value
            ind = value

            # use type 5 bc for `DM_BC_ESSENTIAL_FIELD` enum
            # use type 6 bc for `DM_BC_NATURAL_FIELD` enum

            bc_type = 5
            fn_index = self.ext_dict.ebc[sympy.Matrix([[bc.fn]]).as_immutable()]
            num_constrained_components = bc.components.shape[0]
            comps_view = bc.components
            bc = PetscDSAddBoundary_UW(cdm.dm,
                                bc_type,
                                str(boundary+f"{bc.components}").encode('utf8'),
                                str(boundary).encode('utf8'),
                                bc.f_id,  # field ID in the DM
                                num_constrained_components,
                                <const PetscInt *> &comps_view[0],
                                <void (*)() noexcept>ext.fns_bcs[fn_index],
                                NULL,
                                1,
                                <const PetscInt *> &ind,
                                NULL, )

            self.essential_bcs[index] = self.essential_bcs[index]._replace(PETScID=bc, boundary_label_val=value)

        return

    @timing.routine_timer_decorator
    def _setup_pointwise_functions(self, verbose=False, debug=False, debug_name=None):
        import sympy

        # Any property changes will trigger this
        if self.is_setup:
            if verbose and uw.mpi.rank == 0:
                print(f"SNES_Scalar ({self.name}): Pointwise functions do not need to be rebuilt", flush=True)
            return
        else:
            if verbose and uw.mpi.rank == 0:
                print(f"SNES_Scalar ({self.name}): Pointwise functions need to be built", flush=True)



        mesh = self.mesh
        N = mesh.N
        dim = mesh.dim
        cdim = mesh.cdim

        sympy.core.cache.clear_cache()

        # RESIDUAL: don't unwrap here — let getext()'s two-phase unwrap handle
        # it (preserves constant UWexpressions as symbols for constants[]).
        f0  = sympy.Array(self.F0.sym).reshape(1).as_immutable()
        # F1 is the flux vector, which lives in the embedded coordinate
        # space (cdim components). For volume meshes dim==cdim so this
        # is unchanged; for manifold meshes (dim=2, cdim=3) the flux
        # is genuinely 3-component.
        F1  = sympy.Array(self.F1.sym).reshape(cdim).as_immutable()

        self._u_f0 = f0
        self._u_F1 = F1

        U = sympy.Array(self.u.sym).reshape(1).as_immutable() # scalar works better in derive_by_array
        L = sympy.Array(self.Unknowns.L).reshape(cdim).as_immutable() # unpack one index here too

        fns_residual = [self._u_f0, self._u_F1]

        # JACOBIAN: unwrap (keep constants) + smooth Min/Max kinks so the
        # derivative sees the field-dependence of any nonlinear coefficient
        # (full Newton) while the residual keeps the exact form. No-op for
        # constant coefficients -> bit-identical. See _jacobian_source.
        f0_jac = self._jacobian_source(f0)
        F1_jac = self._jacobian_source(F1, self._newton_flux(F1))

        G0 = sympy.derive_by_array(f0_jac, U)
        G1 = sympy.derive_by_array(f0_jac, L)
        G2 = sympy.derive_by_array(F1_jac, U)
        G3 = sympy.derive_by_array(F1_jac, L)

        # Re-organise if needed / make hashable

        self._G0 = sympy.ImmutableMatrix(G0)
        self._G1 = sympy.ImmutableMatrix(G1)
        self._G2 = sympy.ImmutableMatrix(G2)
        self._G3 = sympy.ImmutableMatrix(G3)

        ##################

        fns_jacobian = (self._G0, self._G1, self._G2, self._G3)

        ##################


        # Now natural bcs (compiled into boundary integral terms)
        # Need to loop on them all ...

        fns_bd_residual = []
        fns_bd_jacobian = []

        for index, bc in enumerate(self.natural_bcs):

            boundary = bc.boundary
            value = mesh.boundaries[bc.boundary].value

            bc_label = mesh.dm.getLabel(boundary)
            bc_is = bc_label.getStratumIS(value)

            if bc.fn_f is not None:
                
                bd_F0  = sympy.Array(bc.fn_f)
                bc.fns["u_f0"] = sympy.ImmutableDenseMatrix(bd_F0)
                
                G0 = sympy.derive_by_array(bd_F0, U)
                G1 = sympy.derive_by_array(bd_F0, L)

                bc.fns["uu_G0"] = sympy.ImmutableMatrix(G0.reshape(1, 1)) 
                bc.fns["uu_G1"] = sympy.ImmutableMatrix(G1.reshape(dim, 1))

                fns_bd_residual += [bc.fns["u_f0"]]
                fns_bd_jacobian += [bc.fns["uu_G0"], bc.fns["uu_G1"]]

            # Similar to SNES_Vector, will leave these out for now, perhaps a different user-interface altogether is required for flux-like bcs


        self._fns_bd_residual = fns_bd_residual
        self._fns_bd_jacobian = fns_bd_jacobian
        
        # generate JIT code.
        # first, we must specify the primary fields.
        # these are fields for which the corresponding sympy functions
        # should be replaced with the primary (instead of auxiliary) petsc
        # field value arrays. in this instance, we want to switch out
        # `self.u` and `self.p` for their primary field
        # petsc equivalents. without specifying this list,
        # the aux field equivalents will be used instead, which
        # will give incorrect results for non-linear problems.
        # note also that the order here is important.

        if self.verbose and uw.mpi.rank==0:
            print(f"Scalar SNES: Jacobians complete, now compile", flush=True)

        prim_field_list = [self.u]
        _getext_result = getext(
            self.mesh,
            JITCallbackSet(
                residual=tuple(fns_residual),
                bcs=tuple(x.fn for x in self.essential_bcs),
                jacobian=tuple(fns_jacobian),
                bd_residual=tuple(fns_bd_residual),
                bd_jacobian=tuple(fns_bd_jacobian),
            ),
            prim_field_list,
            verbose=verbose,
            debug=debug,
        )
        self.compiled_extensions = _getext_result.ptrobj
        self.ext_dict = _getext_result.fn_dicts
        self.constants_manifest = _getext_result.constants_manifest
        self._current_jit_cache_key = _getext_result.cache_key

        return


    @timing.routine_timer_decorator
    def _setup_solver(self, verbose=False, _rewire_only=False):

        if self.is_setup == True:
            if verbose and uw.mpi.rank == 0:
                print(f"SNES_Scalar ({self.name}): SNES solver does not need to be rebuilt", flush=True)
            return

        # set functions
        cdef int ind=1
        cdef int [::1] comps_view  # for numpy memory view
        cdef DM cdm = self.dm
        cdef DS ds =  self.dm.getDS()
        cdef PtrContainer ext = self.compiled_extensions

        i_res = self.ext_dict.res

        PetscDSSetResidual(ds.ds, 0, ext.fns_residual[i_res[self._u_f0]], ext.fns_residual[i_res[self._u_F1]])
        # TODO: check if there's a significant performance overhead in passing in
        # identically `zero` pointwise functions instead of setting to `NULL`

        i_jac = self.ext_dict.jac
        PetscDSSetJacobian(ds.ds, 0, 0,
                ext.fns_jacobian[i_jac[self._G0]],
                ext.fns_jacobian[i_jac[self._G1]],
                ext.fns_jacobian[i_jac[self._G2]],
                ext.fns_jacobian[i_jac[self._G3]],
                )

        ## Now add the boundary residual / jacobian terms

        cdef DMLabel c_label

        for bc in self.natural_bcs:

            boundary = bc.boundary
            boundary_id = bc.PETScID

            value = self.mesh.boundaries[bc.boundary].value
            bc_label = self.dm.getLabel("UW_Boundaries")

            label_val = value

            i_bd_res = self.ext_dict.bd_res
            i_bd_jac = self.ext_dict.bd_jac

            c_label = bc_label

            if bc.fn_f is not None:

                UW_PetscDSSetBdResidual(ds.ds, c_label.dmlabel, label_val, boundary_id,
                                0, 0,
                                ext.fns_bd_residual[i_bd_res[bc.fns["u_f0"]]],
                                NULL,
                                )

                UW_PetscDSSetBdJacobian(ds.ds, c_label.dmlabel, label_val, boundary_id,
                                0, 0, 0,
                                ext.fns_bd_jacobian[i_bd_jac[bc.fns["uu_G0"]]],
                                ext.fns_bd_jacobian[i_bd_jac[bc.fns["uu_G1"]]],
                                NULL,
                                NULL,
                                )

                UW_PetscDSSetBdJacobianPreconditioner(ds.ds, c_label.dmlabel, label_val, boundary_id,
                                0, 0, 0,
                                ext.fns_bd_jacobian[i_bd_jac[bc.fns["uu_G0"]]],
                                ext.fns_bd_jacobian[i_bd_jac[bc.fns["uu_G1"]]],
                                NULL,
                                NULL,
                                )

        # Set constants on DS before copying to coarse levels
        self._set_constants_on_ds(ds)

        # Propagate the updated DS to coarse DMs (needed on both full and rewire
        # paths — DS contains the pointwise function pointers we just changed).
        for coarse_dm in self.dm_hierarchy:
            if not _rewire_only:
                self.dm.copyFields(coarse_dm)
            self.dm.copyDS(coarse_dm)

        if not _rewire_only:
            # DM/SNES construction — only needed on full rebuild. On the rewire
            # path the existing DM/SNES stay attached; only the DS function
            # pointers have been updated above.
            for coarse_dm in self.dm_hierarchy:
                coarse_dm.createClosureIndex(None)

            self.dm.setUp()

            self.snes = PETSc.SNES().create(PETSc.COMM_WORLD)
            self.snes.setDM(self.dm)
            self.snes.setOptionsPrefix(self.petsc_options_prefix)
            self.snes.setFromOptions()


            UW_DMPlexSetSNESLocalFEM(cdm.dm, PETSC_FALSE, NULL)

        self.is_setup = True
        self.constitutive_model._solver_is_setup = True

    @timing.routine_timer_decorator
    def solve(self,
              zero_init_guess: bool =None,
              _force_setup:    bool =False,
              verbose:         bool=False,
              debug:           bool=False,
              debug_name:      str=None,
              time=None,
              divergence_retries: int=0, ):
        """
        Solve the system of equations.

        Assembles and solves the discretized PDE system using PETSc's SNES
        (Scalable Nonlinear Equations Solvers) framework. The solution is
        stored in the solver's unknown variable(s).

        Parameters
        ----------
        zero_init_guess : bool, optional
            Cold or warm start. The default (``None``) **auto-detects**: cold when the
            solver holds no converged solution, warm when it does (see
            :attr:`has_solution`). ``True`` forces a fresh start, discarding any
            existing solution; ``False`` insists on warming from the current field
            values. Warm-starting improves convergence for time-stepping and
            continuation; the auto default gets that without a flag, and cannot warm
            off stale data because a remesh or a diverged solve clears
            ``has_solution``.
        _force_setup : bool, default=False
            Force rebuild of the solver even if already set up. Useful after
            changing boundary conditions or constitutive parameters.
        verbose : bool, default=False
            Print solver progress and timing information.
        debug : bool, default=False
            Enable debug output including intermediate residuals.
        debug_name : str, optional
            Name prefix for debug output files.
        time : float or Quantity, optional
            Physical time for this solve. Passed as ``petsc_t`` to all
            pointwise functions. Expressions using ``mesh.t`` evaluate at
            this time. Non-dimensionalised when scaling is active.
            Default: None (petsc_t unchanged).
        divergence_retries : int, default=0
            If SNES reports DIVERGED after the solve, re-call it with warm
            start up to this many times. A single retry rescues most VEP
            yield-surface kink divergences. 0 preserves legacy behaviour.

        Returns
        -------
        None
            Solution is stored in ``self.u`` (and ``self.p`` for Stokes).

        Examples
        --------
        >>> # Basic solve
        >>> solver.solve()
        >>> temperature_values = solver.u.array[:, 0, 0]

        >>> # Time-stepping with previous solution as initial guess
        >>> for step in range(n_steps):
        ...     solver.solve(zero_init_guess=False)

        >>> # Check convergence
        >>> print(f"Converged: {solver.snes.getConvergedReason() > 0}")

        Notes
        -----
        This is a **collective operation** - all MPI ranks must call it.
        The solver automatically handles mesh variable synchronization.

        See Also
        --------
        snes : Access to underlying PETSc SNES object for advanced control.
        """


        import petsc4py


        if _force_setup:
            self.is_setup = False
        elif not self.constitutive_model._solver_is_setup:
            # Constitutive model swapped: pointwise functions change but the
            # DM/fields/BCs are unchanged. In-place rewire is sufficient.
            self._needs_function_rewire = True

        # Tri-state: None auto-detects cold-vs-warm from has_solution. Resolved HERE,
        # after _force_setup has had its say: that invalidation clears has_solution,
        # and resolving earlier would warm-start off the flag it just cleared.
        zero_init_guess = self._resolve_zero_init_guess(zero_init_guess)

        self._build(verbose, debug, debug_name)

        # Set time on the DM so petsc_t is available in pointwise functions
        cdef DM _time_dm
        if time is not None:
            t_nd = self._nondimensional_time(time)
            _time_dm = self.dm
            UW_DMSetTime(_time_dm.dm, t_nd)

        gvec = self.dm.getGlobalVec()

        if not zero_init_guess:
            self.dm.localToGlobal(self.u.vec, gvec)
        else:
            gvec.array[:] = 0.0

        ## ----

        cdef DM dm = self.dm
        self.mesh.update_lvec()
        cdef Vec cmesh_lvec = self.mesh.lvec

        # cmesh_lvec = vn2.copy()
        # PETSc == 3.16 introduced an explicit interface
        # for setting the aux-vector which we'll use when available.

        ierr = DMSetAuxiliaryVec_UW(dm.dm, NULL, 0, 0, cmesh_lvec.vec); CHKERRQ(ierr)

        # Update constants (e.g. changed material params) before solve
        self._update_constants()

        # Pure-Neumann scalar problems: attach a constant nullspace
        # to the (now set-up) Jacobian. No-op unless
        # ``constant_nullspace`` was set.
        self._attach_constant_nullspace()

        # Custom multigrid prolongation: inject our P hierarchy before the
        # first PCSetUp (so the Galerkin coarse operators are built from it).
        # Picks up a solver-set (set_custom_mg) OR a mesh-owned (adapt child)
        # hierarchy. No-op unless one is present.
        from underworld3.utilities.custom_mg import auto_inject_custom_mg
        auto_inject_custom_mg(self, field_id=None)

        # solve
        self._snes_solve_with_retries(gvec, divergence_retries, verbose)

        lvec = self.dm.getLocalVec()
        cdef Vec clvec = lvec
        # Copy solution back into user facing variable
        self.dm.globalToLocal(gvec, lvec)
        # add back boundaries.
        ierr = DMPlexSNESComputeBoundaryFEM(dm.dm, <void*>clvec.vec, NULL); CHKERRQ(ierr)
        self.u.vec.array[:] = lvec.array[:]
        self.mesh._stale_lvec = True

        # Sync _gvec so downstream consumers (write, stats) see the result
        target_var = getattr(self.u, "_base_var", self.u)
        target_var._sync_lvec_to_gvec()

        # Invalidate cached data views - PETSc buffer may have changed
        if hasattr(target_var, "_canonical_data"):
            target_var._canonical_data = None

        self.dm.restoreLocalVec(lvec)
        self.dm.restoreGlobalVec(gvec)

        self._warn_on_divergence()

        self._record_convergence_status()

        return

    def _object_viewer(self):
        '''This will add specific information about this object to the generic class viewer
        '''
        from IPython.display import Latex, Markdown, display
        from textwrap import dedent

        f0 = self.F0.sym
        F1 = self.F1.sym

        eqF1 = "$\\tiny \\quad \\nabla \\cdot \\color{Blue}" + sympy.latex( F1 )+"$ + "
        eqf0 = "$\\tiny \\phantom{ \\quad \\nabla \\cdot} \\color{DarkRed}" + sympy.latex( f0 )+"\\color{Black} = 0 $"

        # feedback on this instance
        display(
            Markdown(f"# Underworld / PETSc General Scalar Equation Solver"),
            Markdown(f"Primary problem: "),
            Latex(eqF1), Latex(eqf0),
        )


        exprs = uw.function.fn_extract_expressions(self.F0)
        exprs = exprs.union(uw.function.fn_extract_expressions(self.F1))

        if len(exprs) != 0:
            display(Markdown("*Where:*"))

            for expr in exprs:
                expr._object_viewer()


        display(
            Markdown(fr"# Boundary Conditions"),)

        bc_table = "| Type   | Boundary | Expression | \n"
        bc_table += "|:------------------------ | -------- | ---------- | \n"

        for bc in self.essential_bcs:
            bc_table += f"| **{bc.type}** | {bc.boundary} | ${sympy.latex(bc.fn.T)}  $ | \n"
        for bc in self.natural_bcs:
                bc_table += f"| **{bc.type}** | {bc.boundary} | ${sympy.latex(bc.fn_f.T)}  $ | \n"

        display(Markdown(bc_table))

        display(Markdown(fr"This solver is formulated as a {self.mesh.dim} dimensional problem with a {self.mesh.cdim} dimensional mesh"))





### =================================

# TODO(DESIGN): the PETSc interface may be general enough for one class to
# handle both Vector and Scalar — the multi-solver unification question is
# tracked as worklist rows D-23..D-28 (deferred, maintainer session).

class SNES_Vector(SolverBaseClass):
    r"""
    General vector equation solver using PETSc SNES.

    Solves the vector conservation problem for unknown :math:`\mathbf{u}`:

    .. math::

        \nabla \cdot \mathbf{F}(\mathbf{u}, \nabla \mathbf{u}, \dot{\mathbf{u}},
        \nabla\dot{\mathbf{u}}) - \mathbf{f}(\mathbf{u}, \nabla \mathbf{u},
        \dot{\mathbf{u}}, \nabla\dot{\mathbf{u}}) = 0

    where :math:`\mathbf{f}` is a source term, :math:`\mathbf{F}` is a flux term
    relating :math:`\mathbf{u}` to its gradients :math:`\nabla \mathbf{u}`, and
    :math:`\dot{\mathbf{u}}` is the Lagrangian time derivative.

    The unknown :math:`\mathbf{u}` is a vector mesh variable, and :math:`\mathbf{f}`,
    :math:`\mathbf{F}` are arbitrary sympy expressions that may include mesh
    coordinates and other mesh/swarm variables.

    This class is the base layer for building solvers that translate physical
    conservation laws into this general mathematical form.

    Parameters
    ----------
    mesh : underworld3.discretisation.Mesh
        The computational mesh.
    u_Field : MeshVariable, optional
        Pre-existing vector field variable. If None, creates a new variable.
    degree : int, default=2
        Polynomial degree for finite element discretization.
    verbose : bool, default=False
        Enable verbose solver output (monitors convergence).
    DuDt : SemiLagrangian or Lagrangian, optional
        Time derivative handler for the unknown.
    DFDt : SemiLagrangian or Lagrangian, optional
        Time derivative handler for flux.

    Attributes
    ----------
    u : MeshVariable
        The vector unknown being solved for.
    F0 : UWexpression
        Source/force term :math:`\mathbf{f}`.
    F1 : UWexpression
        Flux term :math:`\mathbf{F}`.
    constitutive_model : Constitutive_Model
        Material model defining flux-gradient relationship.
    tolerance : float
        Solver convergence tolerance.

    Examples
    --------
    >>> import underworld3 as uw
    >>> mesh = uw.meshing.StructuredQuadBox(elementRes=(16, 16))
    >>> # Vector projection solver
    >>> proj = uw.systems.Vector_Projection(mesh)
    >>> proj.uw_function = some_vector_expression
    >>> proj.solve()

    See Also
    --------
    SNES_Scalar : For scalar-valued equations.
    SNES_Stokes_SaddlePt : For coupled velocity-pressure (Stokes) problems.
    """

    @timing.routine_timer_decorator
    def __init__(self,
                 mesh     : uw.discretisation.Mesh,
                 u_Field  : uw.discretisation.MeshVariable = None,
                 degree     = 2,
                 verbose    = False,
                 DuDt          : Union[uw.systems.ddt.SemiLagrangian, uw.systems.ddt.Lagrangian] = None,
                 DFDt          : Union[uw.systems.ddt.SemiLagrangian, uw.systems.ddt.Lagrangian] = None,
                 ):


        super().__init__(mesh)

        self.Unknowns.u = u_Field
        self.Unknowns.DuDt = DuDt
        self.Unknowns.DFDt = DFDt

        ## Keep track

        self.verbose = verbose
        self._tolerance = 1.0e-4

        ## Todo: this is obviously not particularly robust

        # options = PETSc.Options()
        # options["dm_adaptor"]= "pragmatic"

        # Participate in the auto FMG/GAMG switch (see the `preconditioner`
        # property). The pc/mg keys below are the GAMG default;
        # _apply_preconditioner_options() upgrades this block to geometric FMG
        # at build time when the mesh carries a refinement hierarchy, and
        # otherwise leaves these defaults in place.
        self._pc_option_prefix = ""

        # Here we can set some defaults for this set of KSP / SNES solvers
        self.petsc_options["snes_type"] = "newtonls"
        self.petsc_options["ksp_rtol"] = 1.0e-3
        self._push_managed_option("ksp_type", "gmres")
        self._push_managed_option("pc_type", "gamg")
        self._push_managed_option("pc_gamg_type", "agg")
        self._push_managed_option("pc_gamg_repartition", True)
        self._push_managed_option("pc_mg_type", "additive")
        self._push_managed_option("pc_gamg_agg_nsmooths", 2)
        self.petsc_options["snes_rtol"] = 1.0e-3
        self._push_managed_option("mg_levels_ksp_max_it", 3)
        self._push_managed_option("mg_levels_ksp_converged_maxits", None)

        if self.verbose == True:
            self.petsc_options["ksp_monitor"] = None
            self.petsc_options["snes_converged_reason"] = None
            self.petsc_options["snes_monitor_short"] = None
        else:
            self.petsc_options.delValue("ksp_monitor")
            self.petsc_options.delValue("snes_monitor")
            self.petsc_options.delValue("snes_monitor_short")
            self.petsc_options.delValue("snes_converged_reason")


        ## Todo: some validity checking on the size / type of u_Field supplied
        # (the supplied u_Field, if any, was assigned to self.Unknowns.u above;
        # here we only auto-create the default unknown when none was supplied)
        if u_Field is None:
            self.Unknowns.u = uw.discretisation.MeshVariable( mesh=mesh,
                        num_components=mesh.dim, varname="Uv{}".format(SNES_Vector._obj_count),
                        vtype=uw.VarType.VECTOR, degree=degree )


        self.dm = None

        ## sympy.Matrix

        self._U = self.Unknowns.u.sym

        ## sympy.Matrix - gradient tensor

        self.essential_bcs = []
        self.natural_bcs = []
        self.bcs = self.essential_bcs
        self.boundary_conditions = False

        self.is_setup = False
        self.verbose = verbose

        # Build the DM / FE structures (should be done on remeshing)
        self._rebuild_after_mesh_update = self._build

        # Some other setup

        self.mesh._equation_systems_register.append(self)

    @property
    def tolerance(self):
        """
        Solver convergence tolerance for SNES and KSP.

        Setting this value automatically configures related PETSc tolerances:
        - ``snes_rtol``: Set to ``tolerance``
        - ``ksp_rtol``: Set to ``tolerance * 0.1``
        - ``ksp_atol``: Set to ``tolerance * 1e-6``

        Returns
        -------
        float
            Current solver tolerance.
        """
        return self._tolerance

    @tolerance.setter
    def tolerance(self, value):
        self._tolerance = value
        self.petsc_options["snes_rtol"] = self._tolerance
        self.petsc_options["ksp_rtol"] = self._tolerance * 1.0e-1
        self.petsc_options["ksp_atol"]  = self._tolerance * 1.0e-6


    def add_nitsche_bc(self, conds=None, boundary=None, direction=None,
                       normal=None, gamma=10.0, theta=1, mask=None,
                       local_h=True, g=None):
        r"""Add Nitsche weak enforcement of a velocity constraint along a direction.

        For vector solvers (no pressure field), this constrains
        :math:`\mathbf{u} \cdot \mathbf{d} = \mathrm{conds}` on the boundary
        using Nitsche's method with penalty, consistency, and symmetry terms.

        Parameters
        ----------
        conds : sympy expression or float, optional
            Prescribed velocity along the constraint direction. Default zero
            (free-slip when the direction is the surface normal).
        boundary : str
            Boundary label.
        direction : sympy.Matrix or list, optional
            Constraint direction. Default ``None`` uses surface normal.
        normal : sympy.Matrix or list, optional
            Boundary unit normal used in the Nitsche consistency and symmetry
            terms — the same geometric-normal override as on the Stokes
            variant. Default ``None`` uses the per-boundary,
            deformation-tracking ``mesh.boundary_normal(boundary)``.
        gamma : float, default=10.0
            Dimensionless stabilisation parameter.
        theta : {-1, 0, 1}, default=1
            Symmetry parameter (1=symmetric, -1=skew-symmetric).
        mask : sympy expression, optional
            Accepted for signature parity with the Stokes variant, but
            one-sided masking is **not implemented** on vector solvers:
            passing a mask raises ``NotImplementedError``.
        local_h : bool, default=True
            Scale the penalty by a local per-cell mesh size
            (:meth:`Mesh.cell_size`) rather than the global minimum
            (:meth:`Mesh.get_min_radius`). See
            ``SNES_Stokes_SaddlePt.add_nitsche_bc`` for details.
        g : sympy expression or float, optional
            Deprecated keyword alias for ``conds`` (one DeprecationWarning).

        Warnings
        --------
        Exterior boundaries only. See ``SNES_Stokes_SaddlePt.add_nitsche_bc``
        for details on why internal boundaries are not supported.

        Notes
        -----
        The legacy boundary-first call ``add_nitsche_bc(boundary, g=...)`` is
        detected conservatively (first positional argument a string while the
        second is not — a BC datum is never a string) and shimmed with one
        DeprecationWarning; see :meth:`_value_first_bc_args`.

        See Also
        --------
        SNES_Stokes_SaddlePt.add_nitsche_bc : Stokes version with pressure coupling.
        """
        import sympy
        from collections import namedtuple

        conds, boundary = self._value_first_bc_args(
            "add_nitsche_bc", conds, boundary, alias=g)
        g = conds

        if mask is not None:
            raise NotImplementedError(
                "mask= (one-sided internal-boundary application) is not "
                "implemented on SNES_Vector.add_nitsche_bc; it is supported "
                "on SNES_Stokes_SaddlePt.add_nitsche_bc only")

        self.is_setup = False

        mesh = self.mesh
        # For SNES_Vector, the unknown is a vector field with as many
        # components as the embedding space (cdim). On volume meshes
        # cdim == dim. On manifold meshes (dim < cdim) the vector lives
        # in the embedding space with an implicit tangency constraint.
        cdim = mesh.cdim

        # Surface normal components — use this boundary's own deformation-
        # tracking facet normal (see Mesh.boundary_normal) unless the caller
        # overrides the geometric-normal source; the legacy global
        # mesh.Gamma_P1 stays radial on a deformed surface.
        if normal is not None:
            if isinstance(normal, sympy.MatrixBase):
                n = [normal[i] for i in range(cdim)]
            else:
                n = list(normal)
        else:
            bnorm = mesh.boundary_normal(boundary)
            n = [bnorm[i] for i in range(cdim)]

        # Constraint direction: defaults to surface normal
        if direction is not None:
            if isinstance(direction, sympy.MatrixBase):
                d = [direction[i] for i in range(cdim)]
            else:
                d = list(direction)
        else:
            d = n

        # Velocity symbols
        u = self.u.sym

        # Constraint residual: c = u.d - g
        u_dot_d = sum(u[i] * d[i] for i in range(cdim))
        if g is None:
            g = sympy.Integer(0)
        constraint = u_dot_d - g

        # Mesh size for the penalty term (gamma*mu/h). Default: a LOCAL,
        # per-cell size (mesh.cell_size()) that tracks deformation/adaptation
        # so the stabilisation is scaled correctly on a non-uniform mesh. Set
        # local_h=False for the legacy single global-minimum scalar.
        if local_h:
            h_sym = mesh.cell_size()
        else:
            h_sym = uw.function.expression(
                r"h_{\mathrm{Nitsche}}",
                mesh.get_min_radius(),
                "Nitsche mesh size parameter (global)",
            ).sym

        # Viscosity from constitutive model
        mu = self.constitutive_model.viscosity

        # Constitutive flux
        flux = self._constitutive_model.flux

        # Traction projected onto constraint direction: (σ·n)·d
        t_d = sum(
            flux[i, j] * n[j] * d[i]
            for i in range(cdim) for j in range(cdim)
        )

        # f0_bd: velocity boundary residual (value term)
        f0_components = []
        for c in range(cdim):
            f0_c = (gamma * mu / h_sym) * constraint * d[c]    # penalty
            f0_c -= t_d * d[c]                                   # consistency
            f0_components.append(f0_c)

        fn_f = sympy.Matrix(f0_components).as_immutable()

        # f1_bd: symmetry term
        fn_F = None
        if theta != 0:
            f1_components = sympy.zeros(cdim, cdim)
            for c in range(cdim):
                for dd in range(cdim):
                    f1_components[c, dd] = -theta * mu * (
                        n[dd] * constraint * d[c] + d[c] * constraint * n[dd]
                    )
            fn_F = sympy.Matrix(f1_components).as_immutable()

        # No pressure field in vector solver
        BC = namedtuple('NaturalBC', [
            'f_id', 'components', 'fn_f', 'fn_F', 'fn_p',
            'boundary', 'boundary_label_val', 'type', 'PETScID', 'fns',
        ])

        import numpy as np
        components = np.arange(cdim, dtype=np.int32)

        self.natural_bcs.append(BC(
            0, components, fn_f, fn_F, None,
            boundary, -1, "nitsche", -1, {},
        ))


    @timing.routine_timer_decorator
    def _setup_discretisation(self, verbose=False):
        """
        Most of what is in the init phase that is not called by _setup_terms()
        """

        # Grab the mesh
        mesh = self.mesh

        import xxhash
        import numpy as np

        xxh = xxhash.xxh64()
        xxh.update(np.ascontiguousarray(mesh.X.coords))
        mesh_dm_coord_hash = xxh.intdigest()

        # if we already set up the dm and the coordinates in the mesh dm have not
        # changed then we do not need to do everything here


        if self.dm is not None and self.mesh_dm_coordinate_hash == mesh_dm_coord_hash:
            if verbose and uw.mpi.rank == 0:
                print(f"SNES_Vector ({self.name}): Discretisation does not need to be rebuilt", flush=True)
            return

        # Keep a note of the coordinates that we use for this setup
        self.mesh_dm_coordinate_hash = mesh_dm_coord_hash

        cdef PtrContainer ext = self.compiled_extensions

        mesh = self.mesh
        u_degree = self.u.degree

        if mesh.qdegree < u_degree:
            print(f"Caution - the mesh quadrature ({mesh.qdegree})is lower")
            print(f"than {u_degree} which is required by the {self.name} solver")

        self.dm_hierarchy = mesh.clone_dm_hierarchy()
        self.dm = self.dm_hierarchy[-1]

        options = PETSc.Options()
        options.setValue("private_{}_u_petscspace_degree".format(self.petsc_options_prefix), u_degree) # for private variables
        options.setValue("private_{}_u_petscdualspace_lagrange_continuity".format(self.petsc_options_prefix), self.u.continuous)
        options.setValue("private_{}_u_petscdualspace_lagrange_node_endpoints".format(self.petsc_options_prefix), False)

        self.petsc_fe_u = PETSc.FE().createDefault(mesh.dim, mesh.dim, mesh.isSimplex, mesh.qdegree, "private_{}_u_".format(self.petsc_options_prefix), PETSc.COMM_SELF)
        self.petsc_fe_u_id = self.dm.getNumFields()
        self.dm.setField( self.petsc_fe_u_id, self.petsc_fe_u )
        self.petsc_fe_u.setName("_vector_unknown_")

        self.dm.createDS()


        ## This part is done once on the solver dm ... not required every time we update the functions ...
        ## the values of the natural bcs can be updated

        cdef int ind=1
        cdef int [::1] comps_view  # for numpy memory view
        cdef DM cdm = self.dm

        for index,bc in enumerate(self.natural_bcs):

            component = bc.components
            if uw.mpi.rank == 0 and self.verbose:
                print("Setting bc {} ({})".format(index, bc.type))
                print(" - field:      {}".format(bc.f_id))
                print(" - component:  {}".format(bc.components))
                print(" - boundary:   {}".format(bc.boundary))
                print(" - fn:         {} ".format(bc.fn_f))

            boundary = bc.boundary
            value = mesh.boundaries[bc.boundary].value
            ind = value

            bc_label = self.dm.getLabel(boundary)
            bc_is = bc_label.getStratumIS(value)
            self.natural_bcs[index] = self.natural_bcs[index]._replace(boundary_label_val=value)

            # use type 5 bc for `DM_BC_ESSENTIAL_FIELD` enum
            # use type 6 bc for `DM_BC_NATURAL_FIELD` enum

            bc_type = 6
            num_constrained_components = bc.components.shape[0]
            comps_view = bc.components
            bc = PetscDSAddBoundary_UW(cdm.dm,
                                bc_type,
                                str(boundary+f"{bc.components}").encode('utf8'),
                                "UW_Boundaries".encode('utf8'),   # was: str(boundary)
                                bc.f_id,  # field ID in the DM
                                num_constrained_components,
                                <const PetscInt *> &comps_view[0],
                                <void (*)() noexcept>NULL,
                                NULL,
                                1,
                                <const PetscInt *> &ind,
                                NULL, )

            self.natural_bcs[index] = self.natural_bcs[index]._replace(PETScID=bc, boundary_label_val=ind)


        for index,bc in enumerate(self.essential_bcs):
            if uw.mpi.rank == 0 and self.verbose:
                print("Setting bc {} ({})".format(index, bc.type))
                print(" - field:      {}".format(bc.f_id))
                print(" - component:  {}".format(bc.components))
                print(" - boundary:   {}".format(bc.boundary))
                print(" - fn:         {} ".format(bc.fn))

            boundary = bc.boundary
            value = mesh.boundaries[bc.boundary].value
            ind = value

            # use type 5 bc for `DM_BC_ESSENTIAL_FIELD` enum
            # use type 6 bc for `DM_BC_NATURAL_FIELD` enum

            bc_type = 5
            fn_index = self.ext_dict.ebc[sympy.Matrix([[bc.fn]]).as_immutable()]
            num_constrained_components = bc.components.shape[0]
            comps_view = bc.components
            bc = PetscDSAddBoundary_UW(cdm.dm,
                                bc_type,
                                str(boundary+f"{bc.components}").encode('utf8'),
                                "UW_Boundaries".encode('utf8'),   # was: str(boundary)
                                bc.f_id,  # field ID in the DM
                                num_constrained_components,
                                <const PetscInt *> &comps_view[0],
                                <void (*)() noexcept>ext.fns_bcs[fn_index],
                                NULL,
                                1,
                                <const PetscInt *> &ind,
                                NULL, )

            self.essential_bcs[index] = self.essential_bcs[index]._replace(PETScID=bc, boundary_label_val=value)

        for coarse_dm in self.dm_hierarchy:
            self.dm.copyFields(coarse_dm)
            self.dm.copyDS(coarse_dm)

        self.is_setup = False

        return

    # The properties that are used in the problem description
    # F0 is a vector function (can include u, grad_u)
    # F1_i is a vector valued function (can include u, grad_u)

    # We don't add any validation here ... we should check that these
    # can be ingested by the _setup_terms() function


    @timing.routine_timer_decorator
    def _setup_pointwise_functions(self, verbose=False, debug=False, debug_name=None):
        import sympy

        # Any property changes will trigger this
        if self.is_setup:
            if verbose and uw.mpi.rank == 0:
                print(f"SNES_Vector ({self.name}): Pointwise functions do not need to be rebuilt", flush=True)
            return
        else:
            if verbose and uw.mpi.rank == 0:
                print(f"SNES_Vector ({self.name}): Pointwise functions need to be built", flush=True)

        N = self.mesh.N
        # For SNES_Vector, the vector has cdim components in the embedding
        # space — see the boundary-condition setup above. On volume meshes
        # cdim == mesh.dim.
        cdim = self.mesh.cdim

        sympy.core.cache.clear_cache()

        ## The jacobians are determined from the above (assuming we
        ## do not concern ourselves with the zeros)
        # Residual piece shapes: f0 is (cdim,) per-component, F1 is (cdim, cdim).
        # RESIDUAL: don't unwrap here — let getext()'s two-phase unwrap handle
        # it (preserves constant UWexpressions as symbols for constants[]). The
        # Jacobian sources (f0_jac_list / F1_user_jac) are derived below.
        F0_user = sympy.Matrix(self.F0.sym)
        F1_user = sympy.Matrix(self.F1.sym)

        # Normalise F0 into a list of scalar per-component entries so the
        # explicit-index Jacobian construction below can sympy.diff each one.
        if F0_user.shape == (cdim, 1):
            f0_list = [F0_user[c, 0] for c in range(cdim)]
        elif F0_user.shape == (1, cdim):
            f0_list = [F0_user[0, c] for c in range(cdim)]
        elif F0_user.shape == (cdim,):
            f0_list = [F0_user[c] for c in range(cdim)]
        else:
            raise ValueError(
                f"SNES_Vector F0 shape {F0_user.shape} is not compatible with cdim={cdim}."
            )
        if F1_user.shape != (cdim, cdim):
            raise ValueError(
                f"SNES_Vector F1 shape {F1_user.shape} does not match (cdim, cdim)=({cdim}, {cdim})."
            )

        # Residual arrays kept as matrices for the JIT codegen path.
        # f0 stored as (cdim, 1) column; F1 stored as (cdim, cdim).
        self._u_f0 = sympy.ImmutableDenseMatrix([[e] for e in f0_list])
        self._u_F1 = sympy.ImmutableDenseMatrix(F1_user)
        fns_residual = [self._u_f0, self._u_F1]

        # Unknowns in the form we need for the explicit Jacobian loops.
        #   U_list[c] = u-component c            (u.sym is a (1, cdim) row for VECTOR vtype)
        #   L[c, d]   = ∂u_c / ∂x_d              (shape (cdim, cdim))
        U_list = [self.u.sym[0, c] for c in range(cdim)]
        L = self.Unknowns.L

        # JACOBIAN sources: unwrap (keep constants) + smooth Min/Max kinks so
        # each sympy.diff below sees the field-dependence of any nonlinear
        # coefficient (full Newton) while the residual above stays exact. No-op
        # for constant coefficients -> bit-identical. See _jacobian_source.
        f0_jac_list = [self._jacobian_source(e) for e in f0_list]
        F1_jac = self._jacobian_source(F1_user, self._newton_flux(F1_user))

        # Explicit-index Jacobian construction — writes each entry directly
        # into PETSc's flat [fc, gc, df, dg] layout via row-major 2D matrices.
        # See docs/developer/subsystems/petsc-jacobian-layout.md for the
        # convention and why the older derive_by_array + permutedims form
        # was incorrect for non-symmetric F1. Nc == cdim for SNES_Vector.
        Nc = cdim

        # G0[fc*Nc + gc, 0]              = ∂f0[fc] / ∂U[gc]
        G0 = sympy.zeros(Nc, Nc)
        for fc in range(Nc):
            for gc in range(Nc):
                G0[fc, gc] = sympy.diff(f0_jac_list[fc], U_list[gc])

        # G1[fc*Nc + gc, df]             = ∂f0[fc] / ∂L[gc, df]
        G1 = sympy.zeros(Nc * Nc, cdim)
        for fc in range(Nc):
            for gc in range(Nc):
                for df in range(cdim):
                    G1[fc * Nc + gc, df] = sympy.diff(f0_jac_list[fc], L[gc, df])

        # G2[fc*Nc + gc, df]             = ∂F1[fc, df] / ∂U[gc]
        G2 = sympy.zeros(Nc * Nc, cdim)
        for fc in range(Nc):
            for gc in range(Nc):
                for df in range(cdim):
                    G2[fc * Nc + gc, df] = sympy.diff(F1_jac[fc, df], U_list[gc])

        # G3[fc*Nc + gc, df*cdim + dg]    = ∂F1[fc, df] / ∂L[gc, dg]
        G3 = sympy.zeros(Nc * Nc, cdim * cdim)
        for fc in range(Nc):
            for gc in range(Nc):
                for df in range(cdim):
                    for dg in range(cdim):
                        G3[fc * Nc + gc, df * cdim + dg] = sympy.diff(F1_jac[fc, df], L[gc, dg])

        self._G0 = sympy.ImmutableMatrix(G0)
        self._G1 = sympy.ImmutableMatrix(G1)
        self._G2 = sympy.ImmutableMatrix(G2)
        self._G3 = sympy.ImmutableMatrix(G3)

        ##################

        fns_jacobian = (self._G0, self._G1, self._G2, self._G3)

        # Now natural bcs (compiled into boundary integral terms).
        # Natural BC Jacobians follow the same PETSc layout as the main
        # residual Jacobians — build them with the same explicit-index pattern.

        fns_bd_residual = []
        fns_bd_jacobian = []

        for index, bc in enumerate(self.natural_bcs):

            if bc.fn_f is not None:

                bd_F0_mat = sympy.Matrix(bc.fn_f)
                if bd_F0_mat.shape == (cdim, 1):
                    bd_f0_list = [bd_F0_mat[c, 0] for c in range(cdim)]
                elif bd_F0_mat.shape == (1, cdim):
                    bd_f0_list = [bd_F0_mat[0, c] for c in range(cdim)]
                elif bd_F0_mat.shape == (cdim,):
                    bd_f0_list = [bd_F0_mat[c] for c in range(cdim)]
                else:
                    raise ValueError(
                        f"Natural BC fn_f shape {bd_F0_mat.shape} is not compatible with cdim={cdim}."
                    )

                bd_f0 = sympy.ImmutableDenseMatrix([[e] for e in bd_f0_list])
                bc.fns["u_f0"] = bd_f0

                # BC G0[fc*Nc + gc]           = ∂bd_f0[fc]/∂U[gc]
                # BC G1[fc*Nc + gc, df]       = ∂bd_f0[fc]/∂L[gc, df]
                bd_G0 = sympy.zeros(Nc, Nc)
                bd_G1 = sympy.zeros(Nc * Nc, cdim)
                for fc in range(Nc):
                    for gc in range(Nc):
                        bd_G0[fc, gc] = sympy.diff(bd_f0_list[fc], U_list[gc])
                        for df in range(cdim):
                            bd_G1[fc * Nc + gc, df] = sympy.diff(bd_f0_list[fc], L[gc, df])

                bc.fns["uu_G0"] = sympy.ImmutableMatrix(bd_G0)
                bc.fns["uu_G1"] = sympy.ImmutableMatrix(bd_G1)

                fns_bd_residual += [bc.fns["u_f0"]]
                fns_bd_jacobian += [bc.fns["uu_G0"], bc.fns["uu_G1"]]

                # Gradient boundary residual (f1_bd) and its Jacobians (g2, g3)
                # Used by Nitsche-type BCs; None for standard natural BCs.
                if hasattr(bc, 'fn_F') and bc.fn_F is not None:
                    bd_F1 = sympy.Matrix(bc.fn_F)
                    if bd_F1.shape != (cdim, cdim):
                        raise ValueError(
                            f"Natural BC fn_F shape {bd_F1.shape} is not (cdim, cdim)=({cdim}, {cdim})."
                        )
                    bc.fns["u_F1"] = sympy.ImmutableDenseMatrix(bd_F1)
                    fns_bd_residual += [bc.fns["u_F1"]]

                    # Nitsche gradient-traction Jacobian source (unwrap + smooth
                    # kinks); residual u_F1 above stays exact. See _jacobian_source.
                    bd_F1_jac = self._jacobian_source(bd_F1)

                    # BC G2[fc*Nc + gc, df]          = ∂bd_F1[fc, df]/∂U[gc]
                    # BC G3[fc*Nc + gc, df*cdim + dg] = ∂bd_F1[fc, df]/∂L[gc, dg]
                    bd_G2 = sympy.zeros(Nc * Nc, cdim)
                    bd_G3 = sympy.zeros(Nc * Nc, cdim * cdim)
                    for fc in range(Nc):
                        for gc in range(Nc):
                            for df in range(cdim):
                                bd_G2[fc * Nc + gc, df] = sympy.diff(bd_F1_jac[fc, df], U_list[gc])
                                for dg in range(cdim):
                                    bd_G3[fc * Nc + gc, df * cdim + dg] = sympy.diff(bd_F1_jac[fc, df], L[gc, dg])

                    bc.fns["uu_G2"] = sympy.ImmutableMatrix(bd_G2)
                    bc.fns["uu_G3"] = sympy.ImmutableMatrix(bd_G3)
                    fns_bd_jacobian += [bc.fns["uu_G2"], bc.fns["uu_G3"]]

        self._fns_bd_residual = fns_bd_residual
        self._fns_bd_jacobian = fns_bd_jacobian


        ##################

        # generate JIT code.
        # first, we must specify the primary fields.
        # these are fields for which the corresponding sympy functions
        # should be replaced with the primary (instead of auxiliary) petsc
        # field value arrays. in this instance, we want to switch out
        # `self.u` and `self.p` for their primary field
        # petsc equivalents. without specifying this list,
        # the aux field equivalents will be used instead, which
        # will give incorrect results for non-linear problems.
        # note also that the order here is important.

        prim_field_list = [self.u,]
        _getext_result = getext(
            self.mesh,
            JITCallbackSet(
                residual=tuple(fns_residual),
                bcs=tuple(x.fn for x in self.essential_bcs),
                jacobian=tuple(fns_jacobian),
                bd_residual=tuple(fns_bd_residual),
                bd_jacobian=tuple(fns_bd_jacobian),
            ),
            prim_field_list,
            verbose=verbose,
            debug=debug,
        )
        self.compiled_extensions = _getext_result.ptrobj
        self.ext_dict = _getext_result.fn_dicts
        self.constants_manifest = _getext_result.constants_manifest
        self._current_jit_cache_key = _getext_result.cache_key

        cdef PtrContainer ext = self.compiled_extensions

        return


    @timing.routine_timer_decorator
    def _setup_solver(self, verbose=False, _rewire_only=False):


        if self.is_setup == True:
            if verbose and uw.mpi.rank == 0:
                print(f"SNES_Vector ({self.name}): SNES solver does not need to be rebuilt", flush=True)
            return

        # set functions
        cdef int ind=1
        cdef int [::1] comps_view  # for numpy memory view
        cdef DM cdm = self.dm
        cdef DS ds =  self.dm.getDS()
        cdef PtrContainer ext = self.compiled_extensions


        i_res = self.ext_dict.res

        PetscDSSetResidual(ds.ds, 0, ext.fns_residual[i_res[self._u_f0]], ext.fns_residual[i_res[self._u_F1]])
        # TODO: check if there's a significant performance overhead in passing in
        # identically `zero` pointwise functions instead of setting to `NULL`

        i_jac = self.ext_dict.jac
        PetscDSSetJacobian(ds.ds, 0, 0,
                ext.fns_jacobian[i_jac[self._G0]],
                ext.fns_jacobian[i_jac[self._G1]],
                ext.fns_jacobian[i_jac[self._G2]],
                ext.fns_jacobian[i_jac[self._G3]],
                )

        ## SNES VECTOR ADD Boundary terms

        cdef DMLabel c_label

        for bc in self.natural_bcs:

            boundary = bc.boundary
            boundary_id = bc.PETScID

            value = self.mesh.boundaries[bc.boundary].value
            bc_label = self.dm.getLabel("UW_Boundaries")

            label_val = value

            i_bd_res = self.ext_dict.bd_res
            i_bd_jac = self.ext_dict.bd_jac

            c_label = bc_label

            if bc.fn_f is not None:
                _has_f1 = "u_F1" in bc.fns

                if _has_f1:
                    UW_PetscDSSetBdResidual(ds.ds, c_label.dmlabel, label_val, boundary_id,
                                    0, 0,
                                    ext.fns_bd_residual[i_bd_res[bc.fns["u_f0"]]],
                                    ext.fns_bd_residual[i_bd_res[bc.fns["u_F1"]]],
                                    )
                else:
                    UW_PetscDSSetBdResidual(ds.ds, c_label.dmlabel, label_val, boundary_id,
                                    0, 0,
                                    ext.fns_bd_residual[i_bd_res[bc.fns["u_f0"]]],
                                    NULL,
                                    )

                if _has_f1:
                    UW_PetscDSSetBdJacobian(ds.ds, c_label.dmlabel, label_val, boundary_id,
                                    0, 0, 0,
                                    ext.fns_bd_jacobian[i_bd_jac[bc.fns["uu_G0"]]],
                                    ext.fns_bd_jacobian[i_bd_jac[bc.fns["uu_G1"]]],
                                    ext.fns_bd_jacobian[i_bd_jac[bc.fns["uu_G2"]]],
                                    ext.fns_bd_jacobian[i_bd_jac[bc.fns["uu_G3"]]],
                                    )
                else:
                    UW_PetscDSSetBdJacobian(ds.ds, c_label.dmlabel, label_val, boundary_id,
                                    0, 0, 0,
                                    ext.fns_bd_jacobian[i_bd_jac[bc.fns["uu_G0"]]],
                                    ext.fns_bd_jacobian[i_bd_jac[bc.fns["uu_G1"]]],
                                    NULL, NULL,
                                    )

                if _has_f1:
                    UW_PetscDSSetBdJacobianPreconditioner(ds.ds, c_label.dmlabel, label_val, boundary_id,
                                    0, 0, 0,
                                    ext.fns_bd_jacobian[i_bd_jac[bc.fns["uu_G0"]]],
                                    ext.fns_bd_jacobian[i_bd_jac[bc.fns["uu_G1"]]],
                                    ext.fns_bd_jacobian[i_bd_jac[bc.fns["uu_G2"]]],
                                    ext.fns_bd_jacobian[i_bd_jac[bc.fns["uu_G3"]]],
                                    )
                else:
                    UW_PetscDSSetBdJacobianPreconditioner(ds.ds, c_label.dmlabel, label_val, boundary_id,
                                    0, 0, 0,
                                    ext.fns_bd_jacobian[i_bd_jac[bc.fns["uu_G0"]]],
                                    ext.fns_bd_jacobian[i_bd_jac[bc.fns["uu_G1"]]],
                                    NULL, NULL,
                                    )


        if verbose:
            print(f"Weak form (DS)", flush=True)
            UW_PetscDSViewWF(ds.ds)
            print(f"=============", flush=True)

            print(f"Weak form(s) (Natural Boundaries)", flush=True)
            for boundary in self.natural_bcs:
                UW_PetscDSViewBdWF(ds.ds, boundary.PETScID)

        # Set constants on DS before copying to coarse levels
        self._set_constants_on_ds(ds)

        # Propagate updated DS to coarse DMs (both paths need this — the DS
        # holds the pointwise function pointers we just changed).
        for coarse_dm in self.dm_hierarchy:
            if not _rewire_only:
                self.dm.copyFields(coarse_dm)
            self.dm.copyDS(coarse_dm)

        if not _rewire_only:
            for coarse_dm in self.dm_hierarchy:
                coarse_dm.createClosureIndex(None)

            self.dm.setUp()

            self.snes = PETSc.SNES().create(PETSc.COMM_WORLD)
            self.snes.setDM(self.dm)
            self.snes.setOptionsPrefix(self.petsc_options_prefix)
            self.snes.setFromOptions()

            UW_DMPlexSetSNESLocalFEM(cdm.dm, PETSC_FALSE, NULL)


        self.is_setup = True
        self.constitutive_model._solver_is_setup = True


    @timing.routine_timer_decorator
    def solve(self,
              zero_init_guess: bool =None,
              _force_setup:    bool =False,
              verbose=False,
              debug=False,
              debug_name=None,
              divergence_retries: int=0,
               ):
        """
        Solve the vector field system of equations.

        Assembles and solves the discretized PDE system for vector unknowns
        (e.g., velocity in projection problems) using PETSc's SNES framework.

        Parameters
        ----------
        zero_init_guess : bool, optional
            Cold or warm start. The default (``None``) **auto-detects**: cold when the
            solver holds no converged solution, warm when it does (see
            :attr:`has_solution`). ``True`` forces a fresh start, discarding any
            existing solution; ``False`` insists on warming from the current field
            values. Warm-starting improves convergence for time-stepping and
            continuation; the auto default gets that without a flag, and cannot warm
            off stale data because a remesh or a diverged solve clears
            ``has_solution``.
        _force_setup : bool, default=False
            Force rebuild of the solver even if already set up.
        verbose : bool, default=False
            Print solver progress and timing information.
        debug : bool, default=False
            Enable debug output.
        debug_name : str, optional
            Name prefix for debug output files.
        divergence_retries : int, default=0
            If SNES reports DIVERGED after the solve, re-call it with warm
            start up to this many times. 0 preserves legacy behaviour.

        Returns
        -------
        None
            Solution is stored in ``self.u``.

        Notes
        -----
        This is a **collective operation** - all MPI ranks must call it.

        See Also
        --------
        u : The solution vector field variable.
        """


        if _force_setup:
            self.is_setup = False
        elif not self.constitutive_model._solver_is_setup:
            # Constitutive model swapped: pointwise functions change but the
            # DM/fields/BCs are unchanged. In-place rewire is sufficient.
            self._needs_function_rewire = True

        # Tri-state: None auto-detects cold-vs-warm from has_solution. Resolved HERE,
        # after _force_setup has had its say: that invalidation clears has_solution,
        # and resolving earlier would warm-start off the flag it just cleared.
        zero_init_guess = self._resolve_zero_init_guess(zero_init_guess)

        self._build(verbose, debug, debug_name)


        gvec = self.dm.getGlobalVec()

        if not zero_init_guess:
            self.dm.localToGlobal(self.u.vec, gvec)
        else:
            gvec.array[:] = 0.

        # NOTE: do NOT clearDS()/createDS() the aux (mesh) dm here. Those calls
        # are not made by SNES_Scalar (Poisson) or Stokes, and they destroy field
        # registrations — "Invalid field number" errors when variables are created
        # after other solvers have run.

        self.mesh.update_lvec()
        cdef DM dm = self.dm
        cdef Vec cmesh_lvec
        # PETSc == 3.16 introduced an explicit interface
        # for setting the aux-vector which we'll use when available.
        cmesh_lvec = self.mesh.lvec
        ierr = DMSetAuxiliaryVec_UW(dm.dm, NULL, 0, 0, cmesh_lvec.vec); CHKERRQ(ierr)

        # Update constants (e.g. changed material params) before solve
        self._update_constants()

        # Custom geometric-MG prolongation on the (top-level vector) PC, if
        # registered via set_custom_fmg or owned by an adapt() mesh. Mirrors the
        # SNES_Scalar hook.
        from underworld3.utilities.custom_mg import auto_inject_custom_mg
        auto_inject_custom_mg(self, field_id=None)

        # solve
        self._snes_solve_with_retries(gvec, divergence_retries, verbose)

        lvec = self.dm.getLocalVec()
        cdef Vec clvec = lvec
        # Copy solution back into user facing variable

        self.dm.globalToLocal(gvec, lvec)
        if verbose:
            print(f"{uw.mpi.rank}: Copy solution / bcs to user variables", flush=True)

        # add back boundaries.
        # Note that `DMPlexSNESComputeBoundaryFEM()` seems to need to use an lvec
        # derived from the system-dm (as opposed to the var.vec local vector), else
        # failures can occur.

        ierr = DMPlexSNESComputeBoundaryFEM(dm.dm, <void*>clvec.vec, NULL); CHKERRQ(ierr)
        self.u.vec.array[:] = lvec.array[:]
        self.mesh._stale_lvec = True

        # Sync _gvec so downstream consumers (write, stats) see the result
        target_var = getattr(self.u, "_base_var", self.u)
        target_var._sync_lvec_to_gvec()

        # Invalidate cached data views - PETSc buffer may have changed
        if hasattr(target_var, "_canonical_data"):
            target_var._canonical_data = None

        self.dm.restoreLocalVec(lvec)
        self.dm.restoreGlobalVec(gvec)

        self._warn_on_divergence()

        self._record_convergence_status()

        return

    def _object_viewer(self):
        '''This will add specific information about this object to the generic class viewer
        '''
        from IPython.display import Latex, Markdown, display
        from textwrap import dedent

        f0 = self.F0.sym
        F1 = self.F1.sym

        eqF1 = "$\\tiny \\quad \\nabla \\cdot \\color{Blue}" + sympy.latex( F1 )+"$ + "
        eqf0 = "$\\tiny \\phantom{ \\quad \\nabla \\cdot} \\color{DarkRed}" + sympy.latex( f0 )+"\\color{Black} = 0 $"

        # feedback on this instance
        display(
            Markdown(f"# Underworld / PETSc General Vector Equation Solver"),
            Markdown(f"Primary problem: "),
            Latex(eqF1), Latex(eqf0),
        )

        exprs = uw.function.fn_extract_expressions(self.F0)
        exprs = exprs.union(uw.function.fn_extract_expressions(self.F1))

        if len(exprs) != 0:
            display(Markdown("*Where:*"))

            for expr in exprs:
                expr._object_viewer()

        display(
            Markdown(fr"# Boundary Conditions"),)

        bc_table = "| Type   | Boundary | Expression | \n"
        bc_table += "|:------------------------ | -------- | ---------- | \n"

        for bc in self.essential_bcs:
             bc_table += f"| **{bc.type}** | {bc.boundary} | ${sympy.latex(bc.fn.T)}  $ | \n"
        for bc in self.natural_bcs:
                 bc_table += f"| **{bc.type}** | {bc.boundary} | ${sympy.latex(bc.fn_f.T)}  $ | \n"

        display(Markdown(bc_table))

        display(Markdown(fr"This solver is formulated as a {self.mesh.dim} dimensional problem with a {self.mesh.cdim} dimensional mesh"))

### =================================

class SNES_MultiComponent(SolverBaseClass):
    r"""
    General multi-component equation solver using PETSc SNES.

    Generalises :class:`SNES_Vector` to an arbitrary number of DOF
    components per node, decoupled from ``mesh.dim``. Solves for an
    N-component unknown :math:`\mathbf{u}` (stored as a ``(1, Nc)`` row
    matrix variable) with per-component residual and flux terms.

    Unlike :class:`SNES_Vector`, the components of :math:`\mathbf{u}` have
    no inherent physical meaning as a spatial vector — they are N
    independent scalar DOFs sharing one DM. Cross-component coupling is
    allowed through ``F0`` and ``F1`` but not required; the primary use
    case (multi-component projection) is block-diagonal.

    Parameters
    ----------
    mesh : underworld3.discretisation.Mesh
    u_Field : MeshVariable, optional
        Pre-existing ``(1, n_components)`` MATRIX variable. If ``None``,
        one is created.
    n_components : int
        Number of scalar DOF components per node. Required when
        ``u_Field`` is None; otherwise inferred from ``u_Field.shape``.
    degree : int, default=2
    verbose : bool, default=False
    """

    @timing.routine_timer_decorator
    def __init__(self,
                 mesh         : uw.discretisation.Mesh,
                 u_Field      : uw.discretisation.MeshVariable = None,
                 n_components : int = None,
                 degree       = 2,
                 verbose      = False,
                 DuDt         : Union[uw.systems.ddt.SemiLagrangian, uw.systems.ddt.Lagrangian] = None,
                 DFDt         : Union[uw.systems.ddt.SemiLagrangian, uw.systems.ddt.Lagrangian] = None,
                 ):

        super().__init__(mesh)

        if n_components is None:
            if u_Field is None:
                raise ValueError(
                    "SNES_MultiComponent requires n_components or a u_Field with defined shape."
                )
            n_components = int(u_Field.shape[0]) * int(u_Field.shape[1])
        if n_components < 1:
            raise ValueError("n_components must be >= 1")
        # NB: SNES_MultiComponent works on manifold meshes (dim < cdim)
        # because ``n_components`` is decoupled from mesh.dim by design —
        # each component is an independent scalar problem with no
        # cross-coupling. The spatial-derivative iteration inside
        # _setup_pointwise_functions uses mesh.cdim (the gradient lives
        # in the embedded space). Validated on SphericalManifold 2026-05-23.

        self._n_components = int(n_components)

        if u_Field is None:
            self.Unknowns.u = uw.discretisation.MeshVariable(
                mesh=mesh,
                num_components=(1, self._n_components),
                varname="Umc{}".format(SNES_MultiComponent._obj_count),
                vtype=uw.VarType.MATRIX,
                degree=degree,
            )
        else:
            self.Unknowns.u = u_Field

        self.Unknowns.DuDt = DuDt
        self.Unknowns.DFDt = DFDt

        self.verbose = verbose
        self._tolerance = 1.0e-4

        # PETSc options — mirror SNES_Vector defaults
        self.petsc_options["snes_type"] = "newtonls"
        self.petsc_options["ksp_rtol"] = 1.0e-3
        self._push_managed_option("ksp_type", "gmres")
        self._push_managed_option("pc_type", "gamg")
        self._push_managed_option("pc_gamg_type", "agg")
        self._push_managed_option("pc_gamg_repartition", True)
        self._push_managed_option("pc_mg_type", "additive")
        self._push_managed_option("pc_gamg_agg_nsmooths", 2)
        self.petsc_options["snes_rtol"] = 1.0e-3
        self._push_managed_option("mg_levels_ksp_max_it", 3)
        self._push_managed_option("mg_levels_ksp_converged_maxits", None)

        if self.verbose == True:
            self.petsc_options["ksp_monitor"] = None
            self.petsc_options["snes_converged_reason"] = None
            self.petsc_options["snes_monitor_short"] = None
        else:
            self.petsc_options.delValue("ksp_monitor")
            self.petsc_options.delValue("snes_monitor")
            self.petsc_options.delValue("snes_monitor_short")
            self.petsc_options.delValue("snes_converged_reason")

        self.dm = None
        self._U = self.Unknowns.u.sym

        self.essential_bcs = []
        self.natural_bcs = []
        self.bcs = self.essential_bcs
        self.boundary_conditions = False

        self.is_setup = False
        self._rebuild_after_mesh_update = self._build
        self.mesh._equation_systems_register.append(self)

        # Counter used by tests to assert single-DM-build behaviour.
        self._dm_build_count = 0

    @property
    def n_components(self):
        """Number of scalar DOF components per node."""
        return self._n_components

    @property
    def tolerance(self):
        return self._tolerance

    @tolerance.setter
    def tolerance(self, value):
        self._tolerance = value
        self.petsc_options["snes_rtol"] = self._tolerance
        self.petsc_options["ksp_rtol"] = self._tolerance * 1.0e-1
        self.petsc_options["ksp_atol"] = self._tolerance * 1.0e-6

    @timing.routine_timer_decorator
    def _setup_discretisation(self, verbose=False):
        mesh = self.mesh

        import xxhash
        import numpy as np

        xxh = xxhash.xxh64()
        xxh.update(np.ascontiguousarray(mesh.X.coords))
        mesh_dm_coord_hash = xxh.intdigest()

        if self.dm is not None and self.mesh_dm_coordinate_hash == mesh_dm_coord_hash:
            if verbose and uw.mpi.rank == 0:
                print(f"SNES_MultiComponent ({self.name}): Discretisation does not need to be rebuilt", flush=True)
            return

        # Keep a note of the coordinates that we use for this setup
        self.mesh_dm_coordinate_hash = mesh_dm_coord_hash

        cdef PtrContainer ext = self.compiled_extensions

        mesh = self.mesh
        u_degree = self.u.degree

        if mesh.qdegree < u_degree:
            print(f"Caution - the mesh quadrature ({mesh.qdegree})is lower")
            print(f"than {u_degree} which is required by the {self.name} solver")

        self.dm_hierarchy = mesh.clone_dm_hierarchy()
        self.dm = self.dm_hierarchy[-1]

        options = PETSc.Options()
        options.setValue("private_{}_u_petscspace_degree".format(self.petsc_options_prefix), u_degree)
        options.setValue("private_{}_u_petscdualspace_lagrange_continuity".format(self.petsc_options_prefix), self.u.continuous)
        options.setValue("private_{}_u_petscdualspace_lagrange_node_endpoints".format(self.petsc_options_prefix), False)

        # KEY DIFFERENCE from SNES_Vector: n_components, not mesh.dim.
        self.petsc_fe_u = PETSc.FE().createDefault(
            mesh.dim, self._n_components, mesh.isSimplex, mesh.qdegree,
            "private_{}_u_".format(self.petsc_options_prefix), PETSc.COMM_SELF,
        )
        self.petsc_fe_u_id = self.dm.getNumFields()
        self.dm.setField(self.petsc_fe_u_id, self.petsc_fe_u)
        self.petsc_fe_u.setName("_multicomponent_unknown_")

        self.dm.createDS()

        cdef int ind=1
        cdef int [::1] comps_view
        cdef DM cdm = self.dm

        for index, bc in enumerate(self.natural_bcs):

            if uw.mpi.rank == 0 and self.verbose:
                print("Setting bc {} ({})".format(index, bc.type))
                print(" - field:      {}".format(bc.f_id))
                print(" - component:  {}".format(bc.components))
                print(" - boundary:   {}".format(bc.boundary))
                print(" - fn:         {} ".format(bc.fn_f))

            boundary = bc.boundary
            value = mesh.boundaries[bc.boundary].value
            ind = value

            bc_label = self.dm.getLabel(boundary)
            bc_is = bc_label.getStratumIS(value)
            self.natural_bcs[index] = self.natural_bcs[index]._replace(boundary_label_val=value)

            bc_type = 6
            num_constrained_components = bc.components.shape[0]
            comps_view = bc.components
            bc = PetscDSAddBoundary_UW(cdm.dm,
                                bc_type,
                                str(boundary+f"{bc.components}").encode('utf8'),
                                "UW_Boundaries".encode('utf8'),
                                bc.f_id,
                                num_constrained_components,
                                <const PetscInt *> &comps_view[0],
                                <void (*)() noexcept>NULL,
                                NULL,
                                1,
                                <const PetscInt *> &ind,
                                NULL, )

            self.natural_bcs[index] = self.natural_bcs[index]._replace(PETScID=bc, boundary_label_val=ind)

        for index, bc in enumerate(self.essential_bcs):
            if uw.mpi.rank == 0 and self.verbose:
                print("Setting bc {} ({})".format(index, bc.type))
                print(" - field:      {}".format(bc.f_id))
                print(" - component:  {}".format(bc.components))
                print(" - boundary:   {}".format(bc.boundary))
                print(" - fn:         {} ".format(bc.fn))

            boundary = bc.boundary
            value = mesh.boundaries[bc.boundary].value
            ind = value

            bc_type = 5
            fn_index = self.ext_dict.ebc[sympy.Matrix([[bc.fn]]).as_immutable()]
            num_constrained_components = bc.components.shape[0]
            comps_view = bc.components
            bc = PetscDSAddBoundary_UW(cdm.dm,
                                bc_type,
                                str(boundary+f"{bc.components}").encode('utf8'),
                                "UW_Boundaries".encode('utf8'),
                                bc.f_id,
                                num_constrained_components,
                                <const PetscInt *> &comps_view[0],
                                <void (*)() noexcept>ext.fns_bcs[fn_index],
                                NULL,
                                1,
                                <const PetscInt *> &ind,
                                NULL, )

            self.essential_bcs[index] = self.essential_bcs[index]._replace(PETScID=bc, boundary_label_val=value)

        for coarse_dm in self.dm_hierarchy:
            self.dm.copyFields(coarse_dm)
            self.dm.copyDS(coarse_dm)

        self.is_setup = False
        self._dm_build_count += 1

        return

    @timing.routine_timer_decorator
    def _setup_pointwise_functions(self, verbose=False, debug=False, debug_name=None):
        import sympy

        if self.is_setup:
            if verbose and uw.mpi.rank == 0:
                print(f"SNES_MultiComponent ({self.name}): Pointwise functions do not need to be rebuilt", flush=True)
            return
        else:
            if verbose and uw.mpi.rank == 0:
                print(f"SNES_MultiComponent ({self.name}): Pointwise functions need to be built", flush=True)

        N = self.mesh.N
        # Spatial-derivative iteration uses cdim (the embedded gradient
        # has cdim partial derivatives, one per coordinate of the
        # embedding space). On volume meshes cdim == mesh.dim so the
        # behaviour is unchanged. Distinct from mesh.dim, the topological
        # dimension used for FE element construction in _setup_discretisation.
        cdim = self.mesh.cdim
        Nc = self._n_components

        sympy.core.cache.clear_cache()

        # User-provided expressions.
        #   F0 shape: (1, Nc) row matrix  — per-component residual
        #   F1 shape: (Nc, cdim)          — per-component flux
        F0_user = sympy.Matrix(self.F0.sym)
        F1_user = sympy.Matrix(self.F1.sym)

        # Normalise F0 into an (Nc, 1) column for PETSc layout.
        if F0_user.shape == (1, Nc):
            f0_list = [F0_user[0, c] for c in range(Nc)]
        elif F0_user.shape == (Nc, 1):
            f0_list = [F0_user[c, 0] for c in range(Nc)]
        elif F0_user.shape == (Nc,):
            f0_list = [F0_user[c] for c in range(Nc)]
        else:
            raise ValueError(
                f"F0 shape {F0_user.shape} does not match n_components={Nc}; "
                "expected (1, Nc), (Nc, 1) or (Nc,)."
            )

        if F1_user.shape != (Nc, cdim):
            raise ValueError(
                f"F1 shape {F1_user.shape} does not match (n_components, cdim)=({Nc}, {cdim})."
            )

        # Residuals: f0 as (Nc, 1) column, F1 as (Nc, cdim).
        # The JIT generator reads matrix shape to size the output buffer.
        self._u_f0 = sympy.ImmutableDenseMatrix([[e] for e in f0_list])
        self._u_F1 = sympy.ImmutableDenseMatrix(F1_user)
        fns_residual = [self._u_f0, self._u_F1]

        # Unknowns in PETSc-friendly flat form.
        #   U_list[c] = u-component c
        #   L[c, d]   = ∂u_c / ∂x_d      (already shape (Nc, cdim) from Unknowns.u setter)
        U_list = [self.u.sym[0, c] for c in range(Nc)]
        L = self.Unknowns.L

        # JACOBIAN sources: unwrap (keep constants) + smooth Min/Max kinks so
        # each sympy.diff below sees the field-dependence of any nonlinear
        # coefficient (full Newton) while the residual above stays exact. No-op
        # for constant coefficients -> bit-identical. See _jacobian_source.
        f0_jac_list = [self._jacobian_source(e) for e in f0_list]
        F1_jac = self._jacobian_source(F1_user, self._newton_flux(F1_user))

        # ----- Explicit-differentiation Jacobian construction -----
        # PETSc's element-matrix assembly walks the Jacobian arrays in the
        # order [test_component, trial_component, test_deriv, trial_deriv].
        # From `PetscFEUpdateElementMat_Internal` in fe.c:
        #   g0[fc * Nc + gc]
        #   g1[(fc * Nc + gc) * cdim + df]
        #   g2[(fc * Nc + gc) * cdim + df]
        #   g3[((fc * Nc + gc) * cdim + df) * cdim + dg]
        # i.e. fc and gc are the two outer indices; df/dg are the derivative
        # indices inside. Construct each Jacobian as a 2D sympy matrix with
        # row = fc*Nc + gc and col = (df) or (df*cdim + dg), then row-major
        # flatten naturally matches PETSc's layout.

        #  G0[fc*Nc + gc, 0]           = ∂f0[fc] / ∂U[gc]
        G0 = sympy.zeros(Nc, Nc)
        for fc in range(Nc):
            for gc in range(Nc):
                G0[fc, gc] = sympy.diff(f0_jac_list[fc], U_list[gc])

        #  G1[fc*Nc + gc, df]          = ∂f0[fc] / ∂L[gc, df]
        G1 = sympy.zeros(Nc * Nc, cdim)
        for fc in range(Nc):
            for gc in range(Nc):
                for df in range(cdim):
                    G1[fc * Nc + gc, df] = sympy.diff(f0_jac_list[fc], L[gc, df])

        #  G2[fc*Nc + gc, df]          = ∂F1[fc, df] / ∂U[gc]
        G2 = sympy.zeros(Nc * Nc, cdim)
        for fc in range(Nc):
            for gc in range(Nc):
                for df in range(cdim):
                    G2[fc * Nc + gc, df] = sympy.diff(F1_jac[fc, df], U_list[gc])

        #  G3[fc*Nc + gc, df*cdim + dg] = ∂F1[fc, df] / ∂L[gc, dg]
        G3 = sympy.zeros(Nc * Nc, cdim * cdim)
        for fc in range(Nc):
            for gc in range(Nc):
                for df in range(cdim):
                    for dg in range(cdim):
                        G3[fc * Nc + gc, df * cdim + dg] = sympy.diff(F1_jac[fc, df], L[gc, dg])

        self._G0 = sympy.ImmutableMatrix(G0)
        self._G1 = sympy.ImmutableMatrix(G1)
        self._G2 = sympy.ImmutableMatrix(G2)
        self._G3 = sympy.ImmutableMatrix(G3)

        fns_jacobian = (self._G0, self._G1, self._G2, self._G3)

        fns_bd_residual = []
        fns_bd_jacobian = []

        # Natural BCs. Expected shapes: fn_f is (Nc, 1) or (1, Nc) or (Nc,);
        # fn_F (gradient term) is (Nc, cdim) if provided.
        for index, bc in enumerate(self.natural_bcs):

            if bc.fn_f is not None:
                bd_F0_mat = sympy.Matrix(bc.fn_f)
                if bd_F0_mat.shape == (1, Nc):
                    bd_f0_list = [bd_F0_mat[0, c] for c in range(Nc)]
                elif bd_F0_mat.shape == (Nc, 1):
                    bd_f0_list = [bd_F0_mat[c, 0] for c in range(Nc)]
                elif bd_F0_mat.shape == (Nc,):
                    bd_f0_list = [bd_F0_mat[c] for c in range(Nc)]
                else:
                    raise ValueError(
                        f"Natural BC fn_f shape {bd_F0_mat.shape} != n_components={Nc}."
                    )

                bd_f0 = sympy.ImmutableDenseMatrix([[e] for e in bd_f0_list])
                bc.fns["u_f0"] = bd_f0

                bd_G0 = sympy.zeros(Nc, Nc)
                bd_G1 = sympy.zeros(Nc * Nc, cdim)
                for fc in range(Nc):
                    for gc in range(Nc):
                        bd_G0[fc, gc] = sympy.diff(bd_f0_list[fc], U_list[gc])
                        for df in range(cdim):
                            bd_G1[fc * Nc + gc, df] = sympy.diff(bd_f0_list[fc], L[gc, df])

                bc.fns["uu_G0"] = sympy.ImmutableMatrix(bd_G0)
                bc.fns["uu_G1"] = sympy.ImmutableMatrix(bd_G1)

                fns_bd_residual += [bc.fns["u_f0"]]
                fns_bd_jacobian += [bc.fns["uu_G0"], bc.fns["uu_G1"]]

                if hasattr(bc, 'fn_F') and bc.fn_F is not None:
                    bd_F1 = sympy.Matrix(bc.fn_F)
                    if bd_F1.shape != (Nc, cdim):
                        raise ValueError(
                            f"Natural BC fn_F shape {bd_F1.shape} != (n_components, cdim)=({Nc}, {cdim})."
                        )
                    bc.fns["u_F1"] = sympy.ImmutableDenseMatrix(bd_F1)
                    fns_bd_residual += [bc.fns["u_F1"]]

                    # Nitsche gradient-traction Jacobian source (unwrap + smooth
                    # kinks); residual u_F1 above stays exact. See _jacobian_source.
                    bd_F1_jac = self._jacobian_source(bd_F1)

                    bd_G2 = sympy.zeros(Nc * Nc, cdim)
                    bd_G3 = sympy.zeros(Nc * Nc, cdim * cdim)
                    for fc in range(Nc):
                        for gc in range(Nc):
                            for df in range(cdim):
                                bd_G2[fc * Nc + gc, df] = sympy.diff(bd_F1_jac[fc, df], U_list[gc])
                                for dg in range(cdim):
                                    bd_G3[fc * Nc + gc, df * cdim + dg] = sympy.diff(bd_F1_jac[fc, df], L[gc, dg])

                    bc.fns["uu_G2"] = sympy.ImmutableMatrix(bd_G2)
                    bc.fns["uu_G3"] = sympy.ImmutableMatrix(bd_G3)
                    fns_bd_jacobian += [bc.fns["uu_G2"], bc.fns["uu_G3"]]

        self._fns_bd_residual = fns_bd_residual
        self._fns_bd_jacobian = fns_bd_jacobian

        prim_field_list = [self.u,]
        _getext_result = getext(
            self.mesh,
            JITCallbackSet(
                residual=tuple(fns_residual),
                bcs=tuple(x.fn for x in self.essential_bcs),
                jacobian=tuple(fns_jacobian),
                bd_residual=tuple(fns_bd_residual),
                bd_jacobian=tuple(fns_bd_jacobian),
            ),
            prim_field_list,
            verbose=verbose,
            debug=debug,
        )
        self.compiled_extensions = _getext_result.ptrobj
        self.ext_dict = _getext_result.fn_dicts
        self.constants_manifest = _getext_result.constants_manifest

        cdef PtrContainer ext = self.compiled_extensions

        return

    @timing.routine_timer_decorator
    def _setup_solver(self, verbose=False, _rewire_only=False):

        if self.is_setup == True:
            if verbose and uw.mpi.rank == 0:
                print(f"SNES_MultiComponent ({self.name}): SNES solver does not need to be rebuilt", flush=True)
            return

        cdef int ind=1
        cdef int [::1] comps_view
        cdef DM cdm = self.dm
        cdef DS ds = self.dm.getDS()
        cdef PtrContainer ext = self.compiled_extensions

        i_res = self.ext_dict.res

        PetscDSSetResidual(ds.ds, 0, ext.fns_residual[i_res[self._u_f0]], ext.fns_residual[i_res[self._u_F1]])

        i_jac = self.ext_dict.jac
        PetscDSSetJacobian(ds.ds, 0, 0,
                ext.fns_jacobian[i_jac[self._G0]],
                ext.fns_jacobian[i_jac[self._G1]],
                ext.fns_jacobian[i_jac[self._G2]],
                ext.fns_jacobian[i_jac[self._G3]],
                )

        cdef DMLabel c_label

        for bc in self.natural_bcs:
            boundary = bc.boundary
            boundary_id = bc.PETScID

            value = self.mesh.boundaries[bc.boundary].value
            bc_label = self.dm.getLabel("UW_Boundaries")
            label_val = value

            i_bd_res = self.ext_dict.bd_res
            i_bd_jac = self.ext_dict.bd_jac

            c_label = bc_label

            if bc.fn_f is not None:
                _has_f1 = "u_F1" in bc.fns

                if _has_f1:
                    UW_PetscDSSetBdResidual(ds.ds, c_label.dmlabel, label_val, boundary_id,
                                    0, 0,
                                    ext.fns_bd_residual[i_bd_res[bc.fns["u_f0"]]],
                                    ext.fns_bd_residual[i_bd_res[bc.fns["u_F1"]]],
                                    )
                else:
                    UW_PetscDSSetBdResidual(ds.ds, c_label.dmlabel, label_val, boundary_id,
                                    0, 0,
                                    ext.fns_bd_residual[i_bd_res[bc.fns["u_f0"]]],
                                    NULL,
                                    )

                if _has_f1:
                    UW_PetscDSSetBdJacobian(ds.ds, c_label.dmlabel, label_val, boundary_id,
                                    0, 0, 0,
                                    ext.fns_bd_jacobian[i_bd_jac[bc.fns["uu_G0"]]],
                                    ext.fns_bd_jacobian[i_bd_jac[bc.fns["uu_G1"]]],
                                    ext.fns_bd_jacobian[i_bd_jac[bc.fns["uu_G2"]]],
                                    ext.fns_bd_jacobian[i_bd_jac[bc.fns["uu_G3"]]],
                                    )
                else:
                    UW_PetscDSSetBdJacobian(ds.ds, c_label.dmlabel, label_val, boundary_id,
                                    0, 0, 0,
                                    ext.fns_bd_jacobian[i_bd_jac[bc.fns["uu_G0"]]],
                                    ext.fns_bd_jacobian[i_bd_jac[bc.fns["uu_G1"]]],
                                    NULL, NULL,
                                    )

                if _has_f1:
                    UW_PetscDSSetBdJacobianPreconditioner(ds.ds, c_label.dmlabel, label_val, boundary_id,
                                    0, 0, 0,
                                    ext.fns_bd_jacobian[i_bd_jac[bc.fns["uu_G0"]]],
                                    ext.fns_bd_jacobian[i_bd_jac[bc.fns["uu_G1"]]],
                                    ext.fns_bd_jacobian[i_bd_jac[bc.fns["uu_G2"]]],
                                    ext.fns_bd_jacobian[i_bd_jac[bc.fns["uu_G3"]]],
                                    )
                else:
                    UW_PetscDSSetBdJacobianPreconditioner(ds.ds, c_label.dmlabel, label_val, boundary_id,
                                    0, 0, 0,
                                    ext.fns_bd_jacobian[i_bd_jac[bc.fns["uu_G0"]]],
                                    ext.fns_bd_jacobian[i_bd_jac[bc.fns["uu_G1"]]],
                                    NULL, NULL,
                                    )

        if verbose:
            print(f"Weak form (DS)", flush=True)
            UW_PetscDSViewWF(ds.ds)
            print(f"=============", flush=True)

            print(f"Weak form(s) (Natural Boundaries)", flush=True)
            for boundary in self.natural_bcs:
                UW_PetscDSViewBdWF(ds.ds, boundary.PETScID)

        self._set_constants_on_ds(ds)

        for coarse_dm in self.dm_hierarchy:
            if not _rewire_only:
                self.dm.copyFields(coarse_dm)
            self.dm.copyDS(coarse_dm)

        if not _rewire_only:
            for coarse_dm in self.dm_hierarchy:
                coarse_dm.createClosureIndex(None)

            self.dm.setUp()

            self.snes = PETSc.SNES().create(PETSc.COMM_WORLD)
            self.snes.setDM(self.dm)
            self.snes.setOptionsPrefix(self.petsc_options_prefix)
            self.snes.setFromOptions()

            UW_DMPlexSetSNESLocalFEM(cdm.dm, PETSC_FALSE, NULL)

        self.is_setup = True
        self.constitutive_model._solver_is_setup = True

    @timing.routine_timer_decorator
    def solve(self,
              zero_init_guess: bool = None,
              _force_setup:    bool = False,
              verbose=False,
              debug=False,
              debug_name=None,
              divergence_retries: int=0,
               ):
        """Solve the multi-component SNES problem.

        Collective across all MPI ranks. The solution is written back to
        ``self.u.vec`` and made available through ``self.u.array``.

        Parameters
        ----------
        divergence_retries : int, default=0
            If SNES reports DIVERGED after the solve, re-call it with warm
            start up to this many times. 0 preserves legacy behaviour.
        """


        if _force_setup:
            self.is_setup = False
        elif not self.constitutive_model._solver_is_setup:
            # Constitutive model swapped: pointwise functions change but the
            # DM/fields/BCs are unchanged. In-place rewire is sufficient.
            self._needs_function_rewire = True

        # Tri-state: None auto-detects cold-vs-warm from has_solution. Resolved HERE,
        # after _force_setup has had its say: that invalidation clears has_solution,
        # and resolving earlier would warm-start off the flag it just cleared.
        zero_init_guess = self._resolve_zero_init_guess(zero_init_guess)

        self._build(verbose, debug, debug_name)

        gvec = self.dm.getGlobalVec()

        if not zero_init_guess:
            self.dm.localToGlobal(self.u.vec, gvec)
        else:
            gvec.array[:] = 0.

        self.mesh.update_lvec()
        cdef DM dm = self.dm
        cdef Vec cmesh_lvec
        cmesh_lvec = self.mesh.lvec
        ierr = DMSetAuxiliaryVec_UW(dm.dm, NULL, 0, 0, cmesh_lvec.vec); CHKERRQ(ierr)

        self._update_constants()

        self._snes_solve_with_retries(gvec, divergence_retries, verbose)

        lvec = self.dm.getLocalVec()
        cdef Vec clvec = lvec
        self.dm.globalToLocal(gvec, lvec)
        if verbose:
            print(f"{uw.mpi.rank}: Copy solution / bcs to user variables", flush=True)

        ierr = DMPlexSNESComputeBoundaryFEM(dm.dm, <void*>clvec.vec, NULL); CHKERRQ(ierr)
        self.u.vec.array[:] = lvec.array[:]
        self.mesh._stale_lvec = True

        target_var = getattr(self.u, "_base_var", self.u)
        target_var._sync_lvec_to_gvec()
        if hasattr(target_var, "_canonical_data"):
            target_var._canonical_data = None

        self.dm.restoreLocalVec(lvec)
        self.dm.restoreGlobalVec(gvec)

        self._warn_on_divergence()

        self._record_convergence_status()

        return

    def _object_viewer(self):
        from IPython.display import Latex, Markdown, display

        f0 = self.F0.sym
        F1 = self.F1.sym

        eqF1 = "$\\tiny \\quad \\nabla \\cdot \\color{Blue}" + sympy.latex(F1) + "$ + "
        eqf0 = "$\\tiny \\phantom{ \\quad \\nabla \\cdot} \\color{DarkRed}" + sympy.latex(f0) + "\\color{Black} = 0 $"

        display(
            Markdown(f"# Underworld / PETSc General Multi-Component Solver ({self._n_components} components)"),
            Markdown(f"Primary problem: "),
            Latex(eqF1), Latex(eqf0),
        )

### =================================

class SNES_Stokes_SaddlePt(SolverBaseClass):
    r"""
    Saddle point equation solver for constrained problems using PETSc SNES.

    Solves the constrained vector conservation problem for unknown :math:`\mathbf{u}`
    with constraint parameter :math:`p`:

    .. math::

        \nabla \cdot \mathbf{F}(\mathbf{u}, p, \nabla \mathbf{u}, \nabla p,
        \dot{\mathbf{u}}, \nabla\dot{\mathbf{u}}) - \mathbf{f}(\mathbf{u}, p,
        \nabla \mathbf{u}, \nabla p, \dot{\mathbf{u}}, \nabla\dot{\mathbf{u}}) = 0

    .. math::

        f_p(\mathbf{u}, \nabla \mathbf{u}, \dot{\mathbf{u}}, \nabla\dot{\mathbf{u}}) = 0

    where :math:`\mathbf{f}` is a source term, :math:`\mathbf{F}` is a flux term
    relating :math:`\mathbf{u}` to its gradients, :math:`\dot{\mathbf{u}}` is
    the Lagrangian time derivative, and :math:`f_p` expresses the constraints
    on :math:`\mathbf{u}` enforced by parameter :math:`p`.

    The unknown :math:`\mathbf{u}` is a vector mesh variable and :math:`p` is a
    scalar mesh variable. The terms :math:`\mathbf{f}`, :math:`\mathbf{F}`, and
    :math:`f_p` are arbitrary sympy expressions that may include mesh coordinates
    and other mesh/swarm variables.

    This class is the base layer for building solvers that translate physical
    conservation laws into this general mathematical form.

    Parameters
    ----------
    mesh : underworld3.discretisation.Mesh
        The computational mesh.
    velocityField : MeshVariable, optional
        Pre-existing velocity field. If None, creates a new variable.
    pressureField : MeshVariable, optional
        Pre-existing pressure field. If None, creates a new variable.
    degree : int, default=2
        Polynomial degree for velocity (pressure is degree-1).
    p_continuous : bool, default=True
        Whether pressure field is continuous (True) or discontinuous (False).
    verbose : bool, default=False
        Enable verbose solver output.
    DuDt : SemiLagrangian or Lagrangian, optional
        Time derivative handler for velocity (viscoelastic problems).
    DFDt : SemiLagrangian or Lagrangian, optional
        Time derivative handler for stress (viscoelastic problems).

    Attributes
    ----------
    u : MeshVariable
        Velocity field being solved for.
    p : MeshVariable
        Pressure (Lagrange multiplier) field.
    F0 : UWexpression
        Body force term :math:`\mathbf{f}`.
    F1 : UWexpression
        Stress/flux term :math:`\mathbf{F}`.
    PF0 : UWexpression
        Constraint term :math:`f_p` (typically incompressibility).
    constitutive_model : Constitutive_Model
        Viscous/viscoelastic material model.
    tolerance : float
        Solver convergence tolerance.
    bodyforce : sympy.Matrix
        Body force vector (e.g., gravity).
    penalty : float
        Penalty parameter for augmented Lagrangian methods.

    Examples
    --------
    >>> import underworld3 as uw
    >>> mesh = uw.meshing.StructuredQuadBox(elementRes=(32, 32))
    >>> stokes = uw.systems.Stokes(mesh)
    >>> stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    >>> stokes.constitutive_model.Parameters.viscosity = 1.0
    >>> stokes.bodyforce = sympy.Matrix([0, -1])  # Gravity
    >>> stokes.add_dirichlet_bc([0.0, 0.0], "Bottom")
    >>> stokes.add_dirichlet_bc([None, 0.0], "Top")
    >>> stokes.solve()
    >>> velocity = stokes.u.array[:, 0, :]
    >>> pressure = stokes.p.array[:, 0, 0]

    See Also
    --------
    SNES_Scalar : For scalar-valued equations.
    SNES_Vector : For vector-valued equations without constraints.
    """

    class _Unknowns(SolverBaseClass._Unknowns):
        '''Extend the unknowns with the constraint parameter'''

        def __init__(inner_self, owning_solver):

            super().__init__(owning_solver)

            inner_self._p = None
            inner_self._DpDt = None
            inner_self._DFpDt = None

            return

        @property
        def p(inner_self):
            return inner_self._p

        @p.setter
        def p(inner_self, new_p):
            inner_self._p = new_p
            inner_self._owning_solver.is_setup = False
            return


    @timing.routine_timer_decorator
    def __init__(self,
                 mesh          : underworld3.discretisation.Mesh,
                 velocityField : Optional[underworld3.discretisation.MeshVariable] = None,
                 pressureField : Optional[underworld3.discretisation.MeshVariable] = None,
                 degree        : Optional[int] = 2,
                 p_continuous  : Optional[bool] = True,
                 verbose       : Optional[bool]                           =False,
                 DuDt          : Union[uw.systems.ddt.SemiLagrangian, uw.systems.ddt.Lagrangian] = None,
                 DFDt          : Union[uw.systems.ddt.SemiLagrangian, uw.systems.ddt.Lagrangian] = None,
                ):


        super().__init__(mesh)

        self.verbose = verbose
        self.dm = None

        self.Unknowns.u = velocityField
        self.Unknowns.p = pressureField
        self.Unknowns.DuDt = DuDt
        self.Unknowns.DFDt = DFDt

        # Optional saddle-point Lagrange multiplier fields (block-constrained
        # Stokes). Each entry is a full-domain scalar MeshVariable registered
        # as an extra DM field (id 2, 3, ...), grouped with pressure into the
        # Schur split. EMPTY for ordinary Stokes — every multiplier-aware code
        # path below is guarded by `if self._multipliers:` so that with no
        # multiplier the emitted DS is bit-identical to the 2-field solver.
        # _multiplier_screening[k] is the interior screening coefficient
        # (eps M de-singularises the interior h block); see
        # docs/developer/design/CONSTRAINED_FREESLIP_MULTIPLIER.md.
        self._multipliers = []
        self._multiplier_screening = []
        self._block_constraint_bcs = []
        # Rotated strong free-slip BCs: [(boundary, normal), ...]. Registered via
        # add_rotated_freeslip_bc; when non-empty, solve() delegates to
        # underworld3.utilities.rotated_bc (per-node DOF rotation + strong v_n = u_n
        # + reaction = sigma_nn). Empty by default → the solve path is unchanged.
        # _rotated_freeslip_datum maps boundary → prescribed wall-normal datum
        # (the non-zero `conds` of add_rotated_freeslip_bc; absent ⇒ u.n = 0).
        self._rotated_freeslip_bcs = []
        self._rotated_freeslip_datum = {}
        self._rotated_freeslip_info = None
        # Split-fault interface conditions (add_fault_bc): fault names whose
        # coincident DOF pairs carry a contact (the laws themselves live in
        # _fault_interface_laws, set lazily by utilities/fault_contact.py).
        # Non-empty => solve() takes the rotated path, where fault_contact
        # supplies the pair blocks and the interface operator.
        self._fault_contact_faults = []
        # Give the Lagrange-multiplier (lambda) block its own viscosity-scaled
        # Schur preconditioner. The constraint Schur complement S_lambda = C A^-1 C^T
        # scales as 1/mu (since A ~ mu K), exactly like the pressure Schur S_p ~ mu^-1 M_p
        # and independent of the augmentation r. When True, the lambda block's
        # PRECONDITIONER mass (Pmat only) uses 1/mu (= saddle_preconditioner) while the
        # true operator (Amat) (lambda,lambda) block stays = eps (exact Newton). This is
        # the boundary-trace analog of _pp_G0 = 1/mu on pressure, and decouples
        # convergence from r so r can shrink -> cleaner recovered lambda (= topography).
        # When on, the lambda block reuses pressure's already-compiled _pp_G0 = 1/mu
        # for its preconditioner block (same scaling, no extra JIT term). On uniform mu
        # (fieldsplit) it is bit-identical; on moderate contrast it cracks the wall and
        # makes the augmentation optional. Default OFF (opt-in): a monolithic lu solve
        # factorizes the Pmat (pc_use_amat is a no-op here), so this Pmat term is NOT
        # inert for direct solves — flipping the default on regresses the direct-lu
        # constraint enforcement. See the design note
        # docs/developer/design/CONSTRAINED_FREESLIP_MULTIPLIER.md (the warning on the
        # coupled vs sub-block preconditioner placement).
        self._multiplier_schur_pc = False
        # Pin interior (off-boundary) multiplier DOFs to 0 so the solved [p,h]
        # block carries only the boundary trace (~√ndof instead of ~ndof/3 DOFs);
        # the boundary trace is the only physical part. Done by constraining the
        # interior h DOFs directly in the fine local PetscSection
        # (_constrain_interior_multipliers_in_section) — lossless (correct
        # constraint-boundary closure) and refined/FMG-safe (no DMAddBoundary
        # ordering restriction). Default ON: it is both a correctness-neutral DOF
        # reduction AND a conditioning win — leaving the interior h DOFs in place
        # keeps the near-singular ε-screened interior block in the solved system,
        # which an iterative ([p,h] Schur) solve handles poorly. See that method's
        # docstring for why disabling it is ill-conditioned, not "more accurate".
        self._reduce_interior_multiplier = True

        self._degree = degree

        ## Any problem with U,P, just define our own
        if velocityField == None or pressureField == None:

            # Note, ensure names are unique for each solver type
            i = SNES_Stokes_SaddlePt._obj_count
            self.Unknowns.u = uw.discretisation.MeshVariable(f"V{i}", self.mesh, self.mesh.dim, degree=degree, varsymbol=rf"{{\mathbf{{u}}^{{[{i}]}} }}" )
            self.Unknowns.p = uw.discretisation.MeshVariable(f"P{i}", self.mesh, 1, degree=degree-1, continuous=p_continuous, varsymbol=rf"{{\mathbf{{p}}^{{[{i}]}} }}")

            if self.verbose and uw.mpi.rank == 0:
                print(f"SNES Saddle Point Solver {self.instance_number}: creating new mesh variables for unknowns")

        # I expect the following to break for anyone who wants to name their solver _stokes__ etc etc (LM)

        # options = PETSc.Options()
        # options["dm_adaptor"]= "pragmatic"
        # Here we can set some defaults for this set of KSP / SNES solvers

        if self.verbose == True:
            self.petsc_options["ksp_monitor"] = None
            self.petsc_options["snes_converged_reason"] = None
            self.petsc_options["snes_monitor_short"] = None
        else:
            self.petsc_options.delValue("ksp_monitor")
            self.petsc_options.delValue("snes_monitor")
            self.petsc_options.delValue("snes_monitor_short")
            self.petsc_options.delValue("snes_converged_reason")

        self._tolerance = 1.0e-4
        self._strategy = "default"

        # Participate in the auto FMG/GAMG switch on the velocity fieldsplit
        # block (see the `preconditioner` property). The velocity pc/mg keys
        # set below are the GAMG default; _apply_preconditioner_options()
        # upgrades the velocity block to geometric FMG at build time when the
        # mesh carries a refinement hierarchy, and otherwise leaves them.
        self._pc_option_prefix = "fieldsplit_velocity_"

        self.petsc_options["snes_rtol"] = self._tolerance
        self.petsc_options["snes_ksp_ew"] = None
        self.petsc_options["snes_ksp_ew_version"] = 3

        self.petsc_options["pc_type"] = "fieldsplit"
        self.petsc_options["pc_fieldsplit_type"] = "schur"
        self.petsc_options["pc_fieldsplit_schur_fact_type"] = "full"     # diag is an alternative (quick/dirty)
        self.petsc_options["pc_fieldsplit_schur_precondition"] = "a11"   # despite what the docs say for saddle points

        self.petsc_options["pc_fieldsplit_diag_use_amat"] = None
        self.petsc_options["pc_fieldsplit_off_diag_use_amat"] = None
        # self.petsc_options["pc_use_amat"] = None                         # Using this puts more pressure on the inner solve

        p_name = "pressure" # pressureField.clean_name
        v_name = "velocity" # velocityField.clean_name

        # Pressure subsolve — flexible GMRES + GASM. (FGMRES is right-
        # preconditioned by construction; no need to set pc_side explicitly.)
        self.petsc_options[f"fieldsplit_{p_name}_ksp_type"] = "fgmres"
        self.petsc_options[f"fieldsplit_{p_name}_ksp_max_it"] = 200
        self.petsc_options[f"fieldsplit_{p_name}_ksp_rtol"]  = self._tolerance
        self.petsc_options[f"fieldsplit_{p_name}_pc_type"] = "gasm"
        # self.petsc_options[f"fieldsplit_{p_name}_pc_gasm_type"] = "basic"

        ## may be more robust but usually slower
        # self.petsc_options[f"fieldsplit_{p_name}_ksp_type"] = "fgmres"
        # self.petsc_options[f"fieldsplit_{p_name}_ksp_rtol"]  = self._tolerance * 0.1
        # self.petsc_options[f"fieldsplit_{p_name}_pc_type"] = "gamg"
        # self.petsc_options[f"fieldsplit_{p_name}_pc_gamg_type"] = "agg"
        # self.petsc_options[f"fieldsplit_{p_name}_pc_gamg_repartition"] = True

        # Velocity subsolve — flexible GMRES + GAMG.
        #
        # FGMRES (not CG/FCG) is the default to remain robust under non-stationary
        # preconditioning and weakly-indefinite coarse operators. The mg_levels
        # KSP is bounded but variable-iteration (mg_levels_ksp_converged_maxits)
        # so the GAMG application is non-linear; CG/FCG's residual recurrence
        # cannot accommodate this, and at scale (large parallel partitions, sharp
        # internal sources, free-slip Nitsche) PETSc reports DIVERGED_PC_FAILED
        # / indefinite matrix or GMRES residual-recursion mismatch. See issue
        # #147 for the spherical-Kramer benchmark on Gadi that drove this change;
        # gthyagi's validated FGMRES configuration completed at np=144,
        # cellsize=1/32 where CG/FCG and standard GMRES both failed.
        self.petsc_options[f"fieldsplit_{v_name}_ksp_type"] = "fgmres"
        self.petsc_options[f"fieldsplit_{v_name}_ksp_max_it"] = 200
        self.petsc_options[f"fieldsplit_{v_name}_ksp_rtol"]  = self._tolerance * 0.1
        self._push_managed_option(f"fieldsplit_{v_name}_pc_type", "gamg")
        self._push_managed_option(f"fieldsplit_{v_name}_pc_gamg_type", "agg")
        self._push_managed_option(f"fieldsplit_{v_name}_pc_gamg_repartition", True)
        self._push_managed_option(f"fieldsplit_{v_name}_pc_mg_type", "additive")
        self._push_managed_option(f"fieldsplit_{v_name}_pc_gamg_agg_nsmooths", 2)
        self._push_managed_option(f"fieldsplit_{v_name}_mg_levels_ksp_max_it", 3)
        self._push_managed_option(f"fieldsplit_{v_name}_mg_levels_ksp_converged_maxits", None)

        # Create this dict
        self.fields = {}
        self.fields[p_name] = self.p
        self.fields[v_name] = self.u

        # Some other setup

        self.mesh._equation_systems_register.append(self)
        self._rebuild_after_mesh_update = self._build

        self.essential_bcs = []

        self.natural_bcs = []
        self.bcs = self.essential_bcs
        self.boundary_conditions = False
        self._saddle_preconditioner = None
        self._petsc_use_pressure_nullspace = False
        self._petsc_velocity_nullspace_basis = ()
        self._stokes_nullspace = None
        self._stokes_nullspace_basis = ()
        self._velocity_rotation_nullspace = None

        # Construct strainrate tensor for future usage.
        # Grab gradients, and let's switch out to sympy.Matrix notation
        # immediately as it is probably cleaner for this.
        N = mesh.N

        ## sympy.Matrix - gradient tensors
        self._G = self.p.sym.jacobian(self.mesh.CoordinateSystem.N)

        # this attrib records if we need to re-setup
        self.is_setup = False

    def add_rotated_freeslip_bc(self, conds=None, boundary=None, normal=None):
        r"""Add STRONG free-slip (:math:`\mathbf{u}\cdot\hat{\mathbf n}=0`) by rotating
        the boundary velocity DOFs into a per-node (normal, tangential) frame and
        imposing the rotated normal component as an exact Dirichlet constraint.

        Unlike Nitsche/penalty free-slip (weak, leaks :math:`\mathcal O(10^{-3})`),
        this enforces zero wall-normal flow to machine precision, and the constraint
        **reaction** is the consistent boundary normal traction
        :math:`\sigma_{nn}` (see :meth:`boundary_normal_traction`) with no
        augmented-Lagrangian splitting. Correct on deformed / tilted / curved
        boundaries because the normal is taken per node.

        Parameters
        ----------
        conds : float, scalar sympy expression, or None, optional
            Prescribed wall-normal velocity datum :math:`\tilde u_n` (the SCALAR
            component along the outward normal), in the canonical value-first BC
            order (Style Charter, API conventions). Zero (or ``None``) is pure
            free-slip :math:`\mathbf{u}\cdot\hat{\mathbf n}=0`. A non-zero value
            — a number, a sympy expression of ``mesh.X``, or a scalar field read
            such as ``h_dot.sym[0]`` — is imposed strongly (machine precision) as
            :math:`\mathbf{u}\cdot\hat{\mathbf n}=\tilde u_n`, evaluated at the
            boundary nodes at each ``solve()``. On an enclosed boundary the
            datum must be discretely flux-free (:math:`\oint \tilde u_n = 0`)
            for incompressibility. A corner/edge node shared between rotated
            boundaries has no single normal and stays at the free-slip pinning
            (datum ignored there). Vector/matrix values are rejected.
        boundary : str
            Boundary label to constrain.
        normal : None or sympy 1×dim Matrix or array, optional
            Per-node outward normal source. ``None`` uses the geometric facet
            normal (PETSc ``computeCellGeometryFVM``; works in 2D and 3D). A
            sympy ``1×dim`` matrix supplies an analytic normal (exact
            ``X/|X|`` on a spherical cap, a constant on a planar face) — preferred
            on curved boundaries. A constant array is also accepted.

        Notes
        -----
        A node shared by several rotated-free-slip boundaries (a box corner, a 3D
        edge) is constrained on the whole span of its accumulated normals: a 3D
        face frees two tangential directions, a 3D edge frees one (the edge
        tangent), a corner is fully pinned. Registering delegates the solve to
        :mod:`underworld3.utilities.rotated_bc`.

        The legacy boundary-first call ``add_rotated_freeslip_bc(boundary,
        normal)`` is detected conservatively (the first positional argument is
        a string — the datum is never a string) and shimmed with one
        DeprecationWarning: the string becomes ``boundary`` and a second
        positional argument, if present, becomes ``normal``.
        """
        if isinstance(conds, str):
            # legacy boundary-first call: (boundary[, normal])
            if boundary is not None:
                if normal is not None:
                    raise TypeError(
                        "add_rotated_freeslip_bc() received 'normal' twice "
                        "(positionally, legacy order, and as a keyword)")
                normal = boundary
            boundary = conds
            conds = None
            import warnings
            warnings.warn(
                "add_rotated_freeslip_bc(boundary, normal) is deprecated; "
                "use add_rotated_freeslip_bc(conds, boundary, normal=...) "
                "with conds=0 (or boundary=... by keyword)",
                DeprecationWarning, stacklevel=2)
        if not isinstance(boundary, str):
            raise TypeError(
                f"add_rotated_freeslip_bc() requires a boundary label string; "
                f"got {type(boundary).__name__}")
        if conds is not None:
            if isinstance(conds, (sympy.MatrixBase, list, tuple, np.ndarray)):
                raise TypeError(
                    "add_rotated_freeslip_bc(conds=...) takes the SCALAR "
                    "wall-normal datum u.n; got a vector/matrix value")
            # Value comparison, not sympy's structural ==: Float(0.0) != Integer(0)
            # structurally, but both are zero data (the free-slip member of the
            # family). is_zero is True only when sympy can PROVE zero, so a
            # symbolic datum (field read, expression) is correctly kept.
            if sympy.sympify(conds).is_zero is not True:
                self._rotated_freeslip_datum[boundary] = conds
        self._rotated_freeslip_bcs.append((boundary, normal))
        self.is_setup = False
        return

    def add_fault_bc(self, conds=0, boundary=None, normal=None):
        r"""Interface condition on a split-node fault (value-first).

        ``boundary`` names a fault split by ``Mesh.add_fault`` (or
        ``fault_split.split_fault``); the mesh carries its coincident DOF
        pairing. The no-opening constraint :math:`[\mathbf v]\cdot\hat n = 0`
        is always imposed strongly; ``conds`` sets the tangential law:

        * ``conds = 0`` — frictionless (perfectly slippery): zero shear
          traction, the slip emerges (the stress-driven crack).
        * ``conds`` > 0 — viscous interface :math:`\tau = \eta_f V` with
          ``conds`` = :math:`\eta_f` (viscosity per unit length; the
          zero-thickness limit of a band is :math:`\eta_f = \eta_{band}/w`,
          where :math:`\eta_{band}` is the band's OWN weak-zone viscosity —
          not the background's — and :math:`w` its width). ``conds`` large
          removes the fault (only the JUMP is penalised — nothing becomes
          rigid).

        ``normal`` optionally supplies the fault's smooth unit normal —
        a sympy ``1×dim`` Matrix in ``mesh.X`` (same conventions as
        :meth:`add_rotated_freeslip_bc`), the string ``"trace"`` (2-D: the
        smoothed normal is built from the fault's own stored polyline —
        the right choice for digitized traces with no analytic formula),
        or a constant ``(dim,)`` array. Use it whenever the fault trace is
        a SAMPLED SMOOTH CURVE: the default per-node normal averages the
        adjacent facet normals, which zig-zags at the sampling kinks, and
        the no-opening constraint then forbids smooth slip past each kink —
        slip notches and normal-traction sawteeth that GROW under mesh
        refinement. The smooth normal restores the smooth curve's
        mechanics on the same polyline mesh. On a straight fault the
        default is already exact, and a deliberately KINKED fault should
        NOT be smoothed — there the kink response is the physics.

        The solve then takes the rotated strong-constraint path
        (``utilities/rotated_bc.py`` with the pair blocks of
        ``utilities/fault_contact.py``); ``guard()`` /
        ``estimate_difficulty()`` are unavailable, as for rotated free-slip.
        Slip and leak per coincident pair afterwards:
        ``fault_contact.fault_slip(solver, boundary,
        solver._rotated_freeslip_info)``.
        """
        from underworld3.utilities import fault_contact

        if not isinstance(boundary, str):
            raise TypeError(
                f"add_fault_bc() requires the fault's boundary name string; "
                f"got {type(boundary).__name__}")
        eta_f = float(conds)
        if eta_f == 0.0:
            fault_contact.add_frictionless_fault_bc(self, boundary,
                                                    normal=normal)
        else:
            fault_contact.add_viscous_fault_bc(self, eta_f, boundary,
                                               normal=normal)
        self.is_setup = False
        return

    def _residual_is_nonlinear(self, tol=1e-8):
        r"""True if the Stokes residual is nonlinear in the unknowns :math:`(v, p)`
        — i.e. the assembled Jacobian depends on the solution, so Newton / Picard
        iteration is required.

        Detected by a NUMERICAL probe: assemble the Jacobian at two distinct
        velocity states and compare. A symbolic test on ``F1.sym`` cannot see the
        nonlinearity — the effective viscosity's strain-rate (velocity-gradient)
        dependence is carried as a JIT-substituted *placeholder* symbol
        (``\dot\varepsilon_{II}``) in the flux, decoupled from the gradient
        ``L`` in the symbolic form, so it only becomes visible once the operator
        is assembled at a concrete iterate. Constant- or temperature-dependent
        viscosity ⇒ ``J`` independent of ``v`` ⇒ the two assemblies are
        bit-identical ⇒ linear.

        Used as a LAZY guard in the rotated-free-slip loop's picard + pure-Newton
        corner (a Picard warmup is meaningless for a linear residual and impossible
        for pure Newton) — the loop itself needs no up-front probe, it
        self-terminates. The caller must have run the pre-solve preamble
        (auxiliary vector + constants) so the assembly sees the correct
        coefficients.
        """
        snes = self.snes
        dm = self.dm
        snes.setUp()
        J = snes.getJacobian()[0]
        U1 = dm.getGlobalVec(); U2 = dm.getGlobalVec()
        # two distinct smooth, bounded states with non-zero velocity gradients
        # (a linear operator gives the SAME J for both; only a solution-dependent
        # viscosity makes them differ). Ownership-relative index keeps it
        # partition-independent enough for the norm comparison.
        rs, re = U1.getOwnershipRange()
        idx = np.arange(rs, re, dtype=float)
        # write IN PLACE into each vec's PETSc-owned array (getArray returns a writable
        # view). setArray with a fresh numpy temporary on a POOLED getGlobalVec is risky:
        # its storage lifetime/pool reuse is not guaranteed for the later computeJacobian.
        a1 = U1.getArray(); a1[:] = 0.1 * np.sin(0.7 * idx + 0.3)
        a2 = U2.getArray(); a2[:] = 0.1 * np.sin(1.3 * idx + 1.1)
        J1 = J.copy(); J2 = J.copy()
        try:
            snes.computeJacobian(U1, J1)
            snes.computeJacobian(U2, J2)
            J2.axpy(-1.0, J1)                       # J2 <- J(U2) - J(U1)
            rel = J2.norm() / (J1.norm() + 1e-300)
        finally:
            dm.restoreGlobalVec(U1); dm.restoreGlobalVec(U2)
            J1.destroy(); J2.destroy()
        return rel > tol

    def boundary_normal_traction(self, boundary, mass="auto"):
        r"""Return the boundary normal traction :math:`\sigma_{nn}` on a
        rotated-free-slip ``boundary`` as the constraint reaction from the last
        solve — the smooth, bounded quantity used for dynamic topography
        (:math:`h_\infty=-(\sigma_{nn}-\overline{\sigma_{nn}})/\rho g`). Requires a
        prior :meth:`add_rotated_freeslip_bc` on ``boundary`` and a completed
        :meth:`solve`.

        ``mass="auto"`` (default) uses lumped recovery for 2D traces and 3D P1
        triangles, and the consistent surface-mass solve for 3D P2 triangles.
        Explicit ``"lumped"`` and ``"consistent"`` choices remain available where
        mathematically valid, and ``"p1"`` selects P1-PROJECTED recovery on a 3D
        P2 trace (edge-midpoint loads folded onto vertices, lumped P1 triangle
        mass — sound where the consistent P2 path carries the vertex-integral
        checkerboard; the FreeSurface default in 3D). Three-dimensional recovery
        currently supports triangular P1/P2 traces only.

        .. warning::
           On CURVED boundaries, P2 vertex values of :math:`\sigma_{nn}` converge
           only slowly (the vertex basis has zero surface mean, so vertex reactions
           carry only the O(h) facet-geometry error); edge-midpoint values are
           superconvergent. Pointwise consumers on curved boundaries should use
           midpoint or integral/fitted quantities (issue #414)."""
        if self._rotated_freeslip_info is None:
            raise RuntimeError(
                "boundary_normal_traction requires a completed rotated-free-slip solve.")
        from underworld3.utilities.rotated_bc import boundary_normal_traction as _bnt
        return _bnt(self, boundary, self._rotated_freeslip_info, mass=mass)

    def dynamic_topography(self, boundary, field, buoyancy_scale=1.0, mass="auto"):
        r"""Write the dynamic topography
        :math:`h = -(\sigma_{nn}-\overline{\sigma_{nn}})/(\Delta\rho\,g)` on a
        rotated-free-slip ``boundary`` onto a scalar MeshVariable ``field``, from the
        constraint reaction of the last solve. This is the hand-off to the free-surface
        machinery — the 3-number topography integrator drives node motion from a surface
        field, so create a scalar ``field`` (P1 recommended, continuous) up front and
        pass it here after each :meth:`solve`; its boundary nodes are filled and the
        interior left untouched.

        ``buoyancy_scale`` is :math:`\Delta\rho\,g` (traction → length).
        ``mass="auto"`` selects lumped recovery where valid and the consistent
        surface-mass solve for 3D P2 triangles. Requires a prior
        :meth:`add_rotated_freeslip_bc` on ``boundary`` and a completed :meth:`solve`.

        .. warning::
           On CURVED boundaries (annulus/spherical free surfaces), the P2 VERTEX
           values written into ``field`` converge only slowly; edge-midpoint values
           are superconvergent. Downstream pointwise use of curved-boundary
           topography should rely on midpoint/fitted quantities (issue #414)."""
        if self._rotated_freeslip_info is None:
            raise RuntimeError(
                "dynamic_topography requires a completed rotated-free-slip solve.")
        from underworld3.utilities.rotated_bc import dynamic_topography_field as _dtf
        return _dtf(self, boundary, self._rotated_freeslip_info, field,
                    buoyancy_scale=buoyancy_scale, mass=mass)

    def add_nitsche_bc(self, conds=None, boundary=None, direction=None, normal=None,
                       gamma=10.0, theta=1, mask=None, local_h=True, g=None):
        r"""Add Nitsche weak enforcement of a velocity constraint along a direction.

        Nitsche's method provides a variationally consistent alternative to
        penalty-based free-slip that is less sensitive to the penalty magnitude
        and gives optimal convergence rates.

        By default, constrains the normal velocity component
        :math:`\mathbf{u} \cdot \mathbf{n} = \mathrm{conds}` (free-slip when
        ``conds`` is zero). When *direction* is provided, constrains
        :math:`\mathbf{u} \cdot \mathbf{d} = \mathrm{conds}` along that
        direction instead.

        The method constructs boundary residuals and Jacobians for:

        - Penalty/stabilisation: :math:`(\gamma \mu / h)(\mathbf{u} \cdot \mathbf{d} - g) \, \mathbf{d}`
        - Consistency: :math:`-(\boldsymbol{\sigma} \cdot \mathbf{n} \cdot \mathbf{d}) \, \mathbf{d}`
          (boundary traction projected onto constraint direction)
        - Symmetry: :math:`-\theta \mu` adjoint consistency term
        - Pressure: :math:`p \, (\mathbf{n} \cdot \mathbf{d}) \, \mathbf{d}` on velocity
          and :math:`(\mathbf{n} \cdot \mathbf{d})(\mathbf{u} \cdot \mathbf{d} - g)` on pressure

        Parameters
        ----------
        conds : sympy expression or float, optional
            Prescribed velocity along the constraint direction. Default
            ``None`` means zero (:math:`\mathbf{u} \cdot \mathbf{d} = 0`).
        boundary : str
            Boundary label (e.g., ``"Upper"``, ``"Lower"``).
        direction : sympy.Matrix or list, optional
            Constraint direction vector. Default ``None`` uses the boundary
            surface normal (free-slip). Can be spatially varying (e.g.,
            a fault orientation field).
        normal : sympy.Matrix or list, optional
            Boundary unit normal used in the Nitsche consistency, symmetry,
            and pressure-coupling terms. Default ``None`` uses the per-boundary,
            deformation-tracking ``mesh.boundary_normal(boundary)``.
        gamma : float, default=10.0
            Dimensionless stabilisation parameter. Typical values 5--20
            for P2 elements.
        theta : {-1, 0, 1}, default=1
            Symmetry parameter:
             1: symmetric (default — optimal convergence and solver efficiency)
             0: incomplete (no symmetry term)
            -1: skew-symmetric (unconditionally stable but slower convergence)
        mask : sympy expression, optional
            Element-wise mask for one-sided application on internal
            boundaries. Use a DG MeshVariable that is 1 on the active
            side and 0 on the inactive side. The mask multiplies all
            Nitsche terms so that only the active-side cell contributes.
        local_h : bool, default=True
            Scale the penalty term :math:`\gamma\mu/h` by a **local**,
            per-cell mesh size (:meth:`Mesh.cell_size`, deformation- and
            adaptation-tracking) rather than the single **global** minimum
            cell size (:meth:`Mesh.get_min_radius`). On a non-uniform or
            adaptive mesh the local size scales the stabilisation correctly
            on every facet; on a uniform mesh the two coincide. Set ``False``
            to restore the legacy global-h behaviour exactly.
        g : sympy expression or float, optional
            Deprecated keyword alias for ``conds`` (one DeprecationWarning).

        Examples
        --------
        >>> # Free-slip (u.n = 0)
        >>> stokes.add_nitsche_bc(0.0, "Upper", gamma=10)

        >>> # Prescribed normal inflow
        >>> stokes.add_nitsche_bc(1.0, "Left", gamma=10)

        >>> # Constrain along a specific direction (e.g. fault normal)
        >>> fault_normal = sympy.Matrix([0.6, 0.8])
        >>> stokes.add_nitsche_bc(0.0, "Fault", direction=fault_normal, gamma=10)

        Notes
        -----
        The legacy boundary-first call ``add_nitsche_bc(boundary, g=...)`` is
        detected conservatively (first positional argument a string while the
        second is not — a BC datum is never a string) and shimmed with one
        DeprecationWarning; see :meth:`_value_first_bc_args`.

        Warnings
        --------
        This method is for **exterior** boundaries only. On internal
        boundaries (e.g., ``"Internal"`` from ``AnnulusInternalBoundary``),
        the consistency terms cancel between the two adjacent cells,
        producing worse results than no constraint. Use the penalty
        approach (``add_natural_bc``) for internal boundary constraints.

        References
        ----------
        Sime & Wilson (2020), arXiv:2001.10639 — Nitsche free-slip for geodynamics.
        PETSc ``snes/tutorials/ex62.c`` — Nitsche Stokes implementation.
        """
        import sympy
        from collections import namedtuple

        conds, boundary = self._value_first_bc_args(
            "add_nitsche_bc", conds, boundary, alias=g)
        g = conds

        self.is_setup = False

        mesh = self.mesh
        dim = mesh.dim

        # Surface normal components. By default use this boundary's OWN
        # deformation-tracking facet normal (assembled from only this
        # boundary's faces, so a corner shared with another boundary is not
        # averaged across the discontinuity, and a deformed surface is
        # followed correctly). The legacy global mesh.Gamma_P1 point-evaluates
        # petsc_n and stays radial on a deformed surface — see
        # Mesh.boundary_normal.
        if normal is not None:
            if isinstance(normal, sympy.MatrixBase):
                n = [normal[i] for i in range(dim)]
            else:
                n = list(normal)
        else:
            bnorm = mesh.boundary_normal(boundary)
            n = [bnorm[i] for i in range(dim)]

        # Constraint direction: defaults to surface normal
        if direction is not None:
            if isinstance(direction, sympy.MatrixBase):
                d = [direction[i] for i in range(dim)]
            else:
                d = list(direction)
        else:
            d = n  # free-slip: constrain along surface normal

        # Velocity and pressure symbols
        u = self.u.sym   # Matrix (dim, 1)
        p_sym = self.p.sym[0]  # scalar

        # Constraint residual: c = u.d - g
        u_dot_d = sum(u[i] * d[i] for i in range(dim))
        if g is None:
            g = sympy.Integer(0)
        constraint = u_dot_d - g

        # n.d — how much of the constraint direction is normal to the surface
        # Controls pressure coupling (vanishes when d is purely tangential)
        # Use Abs to ensure sign-consistent pressure coupling regardless
        # of whether PETSc face normal points inward or outward
        n_dot_d = sum(n[i] * d[i] for i in range(dim))
        # n_dot_d is always positive when n = d (it's |n|²),
        # so sign of PETSc face normal doesn't affect this term

        # Mesh size for the penalty term (gamma*mu/h). Default: a LOCAL,
        # per-cell characteristic size (mesh.cell_size()), so the Nitsche
        # stabilisation is correctly scaled on every facet of a non-uniform
        # or adaptively-refined mesh — the boundary kernel sees the adjacent
        # cell's size. The field tracks mesh deformation/adaptation. Set
        # local_h=False to restore the legacy single global-minimum scalar
        # (mesh.get_min_radius()); on a uniform mesh the two coincide.
        if local_h:
            h_sym = mesh.cell_size()
        else:
            h_sym = uw.function.expression(
                r"h_{\mathrm{Nitsche}}",
                mesh.get_min_radius(),
                "Nitsche mesh size parameter (global)",
            ).sym

        # Viscosity from constitutive model
        mu = self.constitutive_model.viscosity

        # Constitutive flux (stress tensor) — includes VE history if active
        flux = self._constitutive_model.flux  # dim x dim Matrix

        # Traction projected onto constraint direction:
        # t_d = (σ·n)·d = sum_{ij} flux[i,j] * n[j] * d[i]
        t_d = sum(
            flux[i, j] * n[j] * d[i]
            for i in range(dim) for j in range(dim)
        )

        # f0_bd: velocity boundary residual (value term)
        # = penalty + consistency + pressure flux
        f0_components = []
        for c in range(dim):
            f0_c = (gamma * mu / h_sym) * constraint * d[c]    # penalty
            f0_c -= t_d * d[c]                                   # consistency
            f0_c += p_sym * n_dot_d * d[c]                       # pressure flux
            f0_components.append(f0_c)

        fn_f = sympy.Matrix(f0_components).as_immutable()

        # f1_bd: velocity boundary residual (gradient term) — symmetry
        fn_F = None
        if theta != 0:
            f1_components = sympy.zeros(dim, dim)
            for c in range(dim):
                for dd in range(dim):
                    f1_components[c, dd] = -theta * mu * (
                        n[dd] * constraint * d[c] + d[c] * constraint * n[dd]
                    )
            fn_F = sympy.Matrix(f1_components).as_immutable()

        # fn_p: pressure boundary residual
        # Enforces continuity: (n.d)(u.d - g) on boundary
        # Vanishes when constraint direction is purely tangential
        fn_p = sympy.Matrix([n_dot_d * constraint]).as_immutable()

        # Apply mask for one-sided internal boundary application
        if mask is not None:
            if hasattr(mask, 'sym'):
                mask_expr = mask.sym[0, 0]
            else:
                mask_expr = mask
            fn_f = (fn_f * mask_expr).as_immutable()
            if fn_F is not None:
                fn_F = (fn_F * mask_expr).as_immutable()
            fn_p = (fn_p * mask_expr).as_immutable()

        # Create the NaturalBC with all terms populated
        BC = namedtuple('NaturalBC', [
            'f_id', 'components', 'fn_f', 'fn_F', 'fn_p',
            'boundary', 'boundary_label_val', 'type', 'PETScID', 'fns',
        ])

        import numpy as np
        components = np.arange(dim, dtype=np.int32)

        self.natural_bcs.append(BC(
            0, components, fn_f, fn_F, fn_p,
            boundary, -1, "nitsche", -1, {},
        ))

    # TODO(DESIGN): _setup_history_terms is solver-specific (vector unknowns,
    # SemiLagrangian machinery), not generic — revisit its placement when the
    # solver-class unification (worklist D-23..D-28) is taken up.
    def _setup_history_terms(self):
        self.Unknowns.DuDt = uw.systems.ddt.SemiLagrangian(
                    self.mesh,
                    self.u.sym,
                    self.u.sym,
                    vtype=uw.VarType.VECTOR,
                    degree=self.u.degree,
                    continuous=self.u.continuous,
                    varsymbol=self.u.symbol,
                    verbose=self.verbose,
                    bcs=self.essential_bcs,
                    order=self._order,
                    smoothing=0.0,
                )

        # we will not have a valid constutituve model
        # at this point, so the flux term is empty and
        # will have to be filled later.

        self.Unknowns.DFDt = uw.systems.ddt.SemiLagrangian(
            self.mesh,
            sympy.Matrix.zeros(self.mesh.dim, self.mesh.dim),
            self.u.sym,
            vtype=uw.VarType.SYM_TENSOR,
            degree=self.u.degree - 1,
            continuous=True,
            varsymbol=rf"{{F[ {self.u.symbol} ] }}",
            verbose=self.verbose,
            bcs=None,
            order=self._order,
            smoothing=0.0,
        )

    @property
    def tolerance(self):
        """
        Solver convergence tolerance for the Stokes saddle-point system.

        Setting this value automatically configures PETSc tolerances for the
        coupled velocity-pressure solve using Schur complement fieldsplit:
        - ``snes_rtol``: Set to ``tolerance``
        - ``ksp_atol``: Set to ``tolerance * 1e-6``
        - ``fieldsplit_pressure_ksp_rtol``: Set to ``tolerance * 0.1``
        - ``fieldsplit_velocity_ksp_rtol``: Set to ``tolerance * 0.033``

        Also enables Eisenstat-Walker adaptive tolerance (``snes_ksp_ew``).

        Returns
        -------
        float
            Current solver tolerance.

        Examples
        --------
        >>> stokes.tolerance = 1e-6  # Tighter convergence
        >>> stokes.solve()
        """
        return self._tolerance

    #: Inner-solve tolerance margins, as a fraction of the outer tolerance. The inner
    #: solves are deliberately inexact (Citcom; Moresi & Solomatov 1995) — which is why
    #: the sub-blocks are flexible Krylov methods — but the inexactness must stay well
    #: BELOW the tolerance demanded of the outer solve. These factors are that margin.
    #: Their existence is principled; their size is inherited convention, so they are
    #: DEFAULTS a user may override rather than values this property owns outright.
    _INNER_RTOL_MARGIN = {"fieldsplit_pressure_ksp_rtol": 0.1,
                          "fieldsplit_velocity_ksp_rtol": 0.033}

    @tolerance.setter
    def tolerance(self, value):
        self._tolerance = value
        self.petsc_options["snes_rtol"] = self._tolerance
        self.petsc_options["snes_ksp_ew"] = None
        self.petsc_options["snes_ksp_ew_version"] = 3

        self.petsc_options["ksp_atol"]  = self._tolerance * 1.0e-6

        # Setting the tolerance re-derives the inner margins from it — that is this
        # property's job, and a user who changes the tolerance expects it. What must NOT
        # happen is `solve()` re-deriving them on every call: it did (pyx `solve()`
        # round-trips `self.tolerance` immediately before `setFromOptions()`), which
        # overwrote any user value between it being set and PETSc reading it and made
        # both documented options silently unreachable (#477). `solve()` now re-asserts
        # only the outer keys, via `_reassert_outer_tolerances`.
        for key, margin in self._INNER_RTOL_MARGIN.items():
            self.petsc_options[key] = self._tolerance * margin

    def _resolve_snes_max_it(self, default):
        """The nonlinear iteration cap for this solve, honouring a user-set
        ``snes_max_it``.

        ``solve()`` pushes this option before every solve, so it has to be able to tell
        its OWN previous push from a value the user set — otherwise the first solve makes
        the option permanently unreachable. Call once, before this solve pushes anything.
        """
        pushed = getattr(self, "_snes_max_it_pushed", None)
        user = getattr(self, "_snes_max_it_user", None)
        if self.petsc_options.hasName("snes_max_it"):
            try:
                current = int(self.petsc_options.getInt("snes_max_it"))
            except Exception:
                return default
            if pushed is None or current != pushed:
                # Never pushed by us, or the user has moved it since. Latch: from here on
                # the option is theirs, because the next solve will read back OUR push of
                # THEIR value and would otherwise mistake it for our own default.
                self._snes_max_it_user = current
                return current
            if user is not None:
                return user
        return default

    def _push_snes_max_it(self, value):
        """Push ``snes_max_it`` and remember what was pushed, so a later solve can tell
        this solver's own value apart from a user override."""
        value = int(value)
        self.petsc_options.setValue("snes_max_it", value)
        self._snes_max_it_pushed = value

    def _reassert_outer_tolerances(self):
        """Re-push the OUTER tolerance keys before a solve, leaving the inner margins be.

        `solve()` may have changed `snes_max_it` and the SNES type for a Picard warm-up,
        so the outer settings are re-asserted before the real solve. The sub-block rtols
        are deliberately excluded: they belong to whoever set them last, which may be the
        user (#477). Overwriting them here is what made them unsettable."""
        self.petsc_options["snes_rtol"] = self._tolerance
        self.petsc_options["ksp_atol"] = self._tolerance * 1.0e-6
        # The Eisenstat-Walker flags are NOT re-asserted here. solve() never changes
        # them, so they stay in the options DB from the `tolerance` setter and
        # setFromOptions picks them up regardless — while re-asserting would make
        # `snes_ksp_ew` impossible to switch OFF, which is the same defect as #477 on a
        # knob that matters: EW re-picks the outer KSP rtol every Newton step and
        # OVERRIDES ksp_rtol, so "how hard is the linear solve actually being asked to
        # work" is not answerable without being able to disable it.


    @property
    def strategy(self):
        """
        What this solve should optimise for — the named intent over the
        multigrid smoother's two measured regimes.

        - ``"default"``, ``"robust"``: ``gmres``/4 smoothing. Survives an operator a
          stationary smoother stalls on: Spiegelman notch (:math:`\eta` contrast
          1e26, 4 levels) per-V-cycle contraction 0.56 against richardson's 0.75, the
          margin growing with depth; transversely isotropic rotated annulus 11
          velocity iterations down to 5.
        - ``"fast"``: ``richardson``/3 smoothing. Cheaper per cycle and quicker where
          the operator is benign — on a linear, symmetric annulus at
          :math:`\eta` contrast 1e6 it beats ``"robust"`` on wall clock at every
          hierarchy depth tested (x1.16, x1.30, x1.82 at 2, 3, 4 levels) while taking
          more iterations. It gives up the regime ``"robust"`` exists for, so it is
          an opt-in.

        ``"default"`` is ``"robust"``: the failure it avoids is worse than the cost it
        carries, and it carries that cost exactly where the problem is easy.

        Setting this property also resets the fieldsplit / Schur / pressure sub-solve
        configuration to the framework defaults. It does **not** write the velocity
        block's preconditioner directly — that is applied later, from
        :mod:`underworld3.utilities.multigrid_options`, which is the single writer of
        those options; this property selects which variant it applies. Values written
        by hand into :attr:`petsc_options` are respected and not overwritten.

        Returns
        -------
        str
            Current strategy name.

        The value returned is the strategy name (it compares and formats as the
        plain string) and additionally reports the preconditioner it resolved to
        when displayed::

            >>> stokes.strategy
            'default' — geometric multigrid (3 levels), full cycle,
                        smoother gmresx4 + sor, coarse redundant/lu

        See Also
        --------
        preconditioner : which multigrid FAMILY to use (geometric, algebraic, auto).
        preconditioner_settings : the same information as a dict, for assertions.
        """
        settings = self.preconditioner_settings
        if not settings or not self._pc_resolved:
            summary = "not resolved yet — configured at the first solve"
            if settings:
                summary += (f" (framework defaults in place: "
                            f"{multigrid_options.describe(settings)})")
        else:
            levels = len(getattr(self.mesh, "dm_hierarchy", []) or []) or None
            summary = multigrid_options.describe(
                settings, levels=levels,
                overridden=self._user_overridden_pc_options)
        return _StrategyName(self._strategy, summary)

    @strategy.setter
    def strategy(self, value):
        if value not in ("default", "robust", "fast"):
            raise ValueError(
                f"Unknown solver strategy {value!r}: "
                "expected 'default', 'robust', or 'fast'."
            )
        # 'fast' and 'robust' now select a real smoother variant, via
        # `_mg_smoother_variant` -> `multigrid_options.geometric_mg_bundle`. They were
        # accepted-and-inert placeholders for a long time: validated on input, then
        # configured identically to 'default'. A property that checks your value and
        # then ignores it is the same defect class as #477 and #478 — the failure is
        # invisible, because the solve still converges.

        # self.is_setup = False
        self._strategy = value

        # Common to every strategy: reset the fieldsplit / Schur / pressure
        # sub-solve to the framework defaults. The strategy's effect on the VELOCITY
        # BLOCK is carried by `_mg_smoother_variant`, not by writes from here.

        self.petsc_options["snes_ksp_ew"] = None
        self.petsc_options["snes_ksp_ew_version"] = 3

        self.petsc_options["pc_type"] = "fieldsplit"
        self.petsc_options["pc_fieldsplit_type"] = "schur"
        self.petsc_options["pc_fieldsplit_schur_fact_type"] = "full"     # diag is an alternative (quick/dirty)
        self.petsc_options["pc_fieldsplit_schur_precondition"] = "a11"   # despite what the docs say for saddle points

        self.petsc_options["pc_fieldsplit_diag_use_amat"] = None
        self.petsc_options["pc_fieldsplit_off_diag_use_amat"] = None
        # self.petsc_options["pc_use_amat"] = None                         # Using this puts more pressure on the inner solve

        p_name = "pressure" # pressureField.clean_name
        v_name = "velocity" # velocityField.clean_name

        # Pressure subsolve — flexible GMRES + GASM. (FGMRES is right-
        # preconditioned by construction; no need to set pc_side explicitly.)
        self.petsc_options[f"fieldsplit_{p_name}_ksp_type"] = "fgmres"
        self.petsc_options[f"fieldsplit_{p_name}_ksp_max_it"] = 200
        self.petsc_options[f"fieldsplit_{p_name}_ksp_rtol"]  = self._tolerance
        self.petsc_options[f"fieldsplit_{p_name}_pc_type"] = "gasm"
        self.petsc_options[f"fieldsplit_{p_name}_pc_gasm_type"] = "basic"

        ## may be more robust but usually slower
        # self.petsc_options[f"fieldsplit_{p_name}_ksp_type"] = "fgmres"
        # self.petsc_options[f"fieldsplit_{p_name}_ksp_rtol"]  = self._tolerance * 0.1
        # self.petsc_options[f"fieldsplit_{p_name}_pc_type"] = "gamg"
        # self.petsc_options[f"fieldsplit_{p_name}_pc_gamg_type"] = "agg"
        # self.petsc_options[f"fieldsplit_{p_name}_pc_gamg_repartition"] = True

        # Velocity subsolve — see corresponding block in __init__ for the
        # rationale (FGMRES default for robustness to non-stationary GAMG
        # preconditioning and weakly-indefinite coarse operators; issue #147).
        self.petsc_options[f"fieldsplit_velocity_ksp_type"] = "fgmres"
        self.petsc_options[f"fieldsplit_velocity_ksp_max_it"] = 200
        # The velocity BLOCK's preconditioner is not set here. `strategy` used to
        # write the whole GAMG bundle plus `pc_mg_type=kaskade`, and every one of
        # those writes was DEAD: `_apply_preconditioner_options` runs later (at
        # `_build`) and overwrites them with the geometric bundle on a refined mesh,
        # or the GAMG bundle (`additive`) without one. Measured both orders — the
        # live PC was `mg`/FULL every time, so `kaskade` never once took effect
        # despite the comment warning against changing it. The strategy's effect on
        # the velocity block now runs through `_mg_smoother_variant`, which selects
        # a bundle variant instead of racing the bundle writer.



    @property
    def PF0(self):
        """
        Pressure constraint term (incompressibility and other constraints).

        This is the :math:`\\mathbf{h}_0(p)` term in the saddle-point formulation,
        typically representing the incompressibility constraint
        :math:`\\nabla \\cdot \\mathbf{u} = 0`.

        Returns
        -------
        UWexpression
            Symbolic expression for the constraint term.

        See Also
        --------
        F0 : Velocity force term.
        F1 : Velocity flux/stress term.
        """
        return self._PF0

    @PF0.setter
    def PF0(self, value):
        self.is_setup = False
        # should add test here to make sure k is conformal
        self._PF0 = value

    @property
    def p(self):
        """
        Pressure solution variable (MeshVariable).

        The pressure field from the Stokes solve, typically a discontinuous
        field one degree lower than velocity.

        Returns
        -------
        MeshVariable
            Pressure field variable.

        See Also
        --------
        u : Velocity solution variable.
        """
        return self.Unknowns.p

    @p.setter
    def p(self, new_p):
        self.Unknowns.p = new_p
        return

    @property
    def saddle_preconditioner(self):
        """
        Custom preconditioner for the pressure Schur complement.

        A symbolic expression used to precondition the pressure solve.
        If None (default), uses the mass matrix approximation.

        Returns
        -------
        sympy expression or None
            Custom preconditioner expression.
        """
        return self._saddle_preconditioner

    @saddle_preconditioner.setter
    def saddle_preconditioner(self, function):
        self.is_setup = False
        self._saddle_preconditioner = function

    @property
    def petsc_use_nullspace(self):
        """Enable full nullspace handling: constant pressure + mesh rotation modes.

        Convenience property that enables the pressure nullspace and
        auto-populates velocity nullspace modes from ``mesh.nullspace_rotations``.

        For finer control, use ``petsc_use_pressure_nullspace`` and
        ``petsc_velocity_nullspace_basis`` separately.
        """
        return (self._petsc_use_pressure_nullspace
                or len(self._petsc_velocity_nullspace_basis) > 0)

    @petsc_use_nullspace.setter
    def petsc_use_nullspace(self, value):
        value = bool(value)
        self._petsc_use_pressure_nullspace = value
        if value:
            # Sync velocity nullspace basis to mesh rotations (even if empty)
            # to avoid stale modes carrying over between meshes/runs
            if hasattr(self.mesh, 'nullspace_rotations'):
                self.petsc_velocity_nullspace_basis = self.mesh.nullspace_rotations or ()
            else:
                self._petsc_velocity_nullspace_basis = ()
        else:
            self._petsc_velocity_nullspace_basis = ()
        self._reset_stokes_nullspace()
        self.is_setup = False

    @property
    def petsc_use_pressure_nullspace(self):
        """
        Enable PETSc handling of the constant-pressure nullspace.

        When enabled, the solver attaches the constant-pressure mode to
        the coupled Stokes nullspace basis before solve. Additional
        user-supplied velocity nullspace modes can be configured through
        ``petsc_velocity_nullspace_basis``.

        For free-slip shell problems this is typically used together with
        the rigid-body rotation modes documented on
        ``petsc_velocity_nullspace_basis``.

        Examples
        --------
        2-D annulus with pressure gauge only:
        >>> stokes.petsc_use_pressure_nullspace = True

        2-D annulus with pressure plus rigid rotation:
        >>> x, y = mesh.X
        >>> stokes.petsc_use_pressure_nullspace = True
        >>> stokes.petsc_velocity_nullspace_basis = [sympy.Matrix([-y, x])]

        3-D spherical shell with pressure plus the three rigid rotations:
        >>> x, y, z = mesh.X
        >>> stokes.petsc_use_pressure_nullspace = True
        >>> stokes.petsc_velocity_nullspace_basis = [
        ...     sympy.Matrix([0, -z, y]),
        ...     sympy.Matrix([z, 0, -x]),
        ...     sympy.Matrix([-y, x, 0]),
        ... ]
        """
        return self._petsc_use_pressure_nullspace

    @petsc_use_pressure_nullspace.setter
    def petsc_use_pressure_nullspace(self, value):
        self._petsc_use_pressure_nullspace = bool(value)
        self._reset_stokes_nullspace()
        # Force re-setup so the changed nullspace is re-registered with PETSc if
        # this is toggled after the solver has already been set up (matches
        # `petsc_use_nullspace`; `_attach_stokes_nullspace` runs inside the
        # is_setup-gated `_setup_solver`).
        self.is_setup = False

    @property
    def multiplier_schur_pc(self):
        """
        Give each Lagrange-multiplier (constraint) block its own viscosity-scaled
        Schur preconditioner.

        The constraint Schur complement ``S_lambda = C A^-1 C^T`` scales as
        ``1/mu`` (since ``A ~ mu K``), exactly like the pressure Schur
        ``S_p ~ mu^-1 M_p`` and **independent of the augmentation** ``r``. When
        enabled, the multiplier block's *preconditioner* (Pmat) diagonal reuses
        pressure's ``1/mu`` mass while the true operator (Amat) block stays the
        screening ``eps`` — so Newton is unchanged and this is a pure
        preconditioner term (the boundary-trace analog of the pressure
        ``saddle_preconditioner``).

        Effect: convergence decouples from ``r``. On moderate boundary viscosity
        contrast (e.g. annulus mu_hi ~ 1e3) the augmentation becomes optional
        (``augmentation=0`` converges) and the recovered multiplier (= dynamic
        topography) is cleaner at small ``r``. On extreme contrast (eta jump 1e6)
        a small augmentation floor is still needed, but the requirement shrinks by
        orders of magnitude. Pair with ``snes_type='ksponly'`` for linear Stokes —
        the block solver's default ``newtonls`` defect-corrects a linear system in
        many steps when the Schur approximation is stiff.

        Default ``False`` (opt-in). On the fieldsplit/iterative path it is
        bit-identical on uniform ``mu`` and cracks the moderate-contrast wall;
        but a monolithic ``lu`` solve factorizes the Pmat (``pc_use_amat`` is a
        no-op there), so this term is not inert for direct solves — hence opt-in.
        """
        return self._multiplier_schur_pc

    @multiplier_schur_pc.setter
    def multiplier_schur_pc(self, value):
        self._multiplier_schur_pc = bool(value)
        self.is_setup = False   # force DS re-registration on next solve

    @property
    def petsc_velocity_nullspace_basis(self):
        """
        Optional exact velocity nullspace modes for the coupled Stokes solve.

        Each entry must be a vector-valued SymPy expression defined in the
        mesh coordinate system and representing an exact null mode of the
        configured Stokes operator. Typical examples are rigid-body rotation
        modes for annulus or spherical-shell free-slip problems.

        For centered shell geometries with exact free-slip / no-penetration
        boundary conditions, the rigid-body rotation modes are:

        - 2-D annulus: one mode, ``(-y, x)``, equivalent to ``r e_theta``
        - 3-D spherical shell: three modes,
          ``(0, -z, y)``, ``(z, 0, -x)``, and ``(-y, x, 0)``

        These are the velocity fields generated by rigid rotations
        ``u = omega x x``. They are tangent to concentric circles / spheres
        and have zero strain rate, so they are exact velocity null modes for
        the free-slip shell Stokes operator.

        They are not exact null modes when the boundary conditions select a
        specific tangential velocity, for example:

        - essential velocity boundary conditions
        - penalty boundary conditions on the full velocity error
          ``u - u_analytic``

        To remove shell nullspaces in Stokes, set the pressure mode and then
        provide the exact rotation basis:

        Examples
        --------
        2-D annulus:
        >>> x, y = mesh.X
        >>> stokes.petsc_use_pressure_nullspace = True
        >>> stokes.petsc_velocity_nullspace_basis = [sympy.Matrix([-y, x])]

        3-D spherical shell:
        >>> x, y, z = mesh.X
        >>> stokes.petsc_use_pressure_nullspace = True
        >>> stokes.petsc_velocity_nullspace_basis = [
        ...     sympy.Matrix([0, -z, y]),
        ...     sympy.Matrix([z, 0, -x]),
        ...     sympy.Matrix([-y, x, 0]),
        ... ]
        """
        return self._petsc_velocity_nullspace_basis

    @petsc_velocity_nullspace_basis.setter
    def petsc_velocity_nullspace_basis(self, modes):
        if modes is None:
            modes = ()

        velocity_modes = []
        for mode in modes:
            matrix_mode = sympy.Matrix(mode)
            if matrix_mode.shape == (1, self.mesh.dim):
                matrix_mode = matrix_mode.T
            if matrix_mode.shape != (self.mesh.dim, 1):
                raise ValueError(
                    "Each petsc_velocity_nullspace_basis mode must have shape "
                    f"({self.mesh.dim}, 1) or (1, {self.mesh.dim}); got {matrix_mode.shape}."
                )
            velocity_modes.append(matrix_mode)

        self._petsc_velocity_nullspace_basis = tuple(velocity_modes)
        self._reset_stokes_nullspace()
        self.is_setup = False

    def _reset_stokes_nullspace(self):
        self._stokes_nullspace = None
        self._stokes_nullspace_basis = ()
        # the cached velocity rotation nullspace is built from node coordinates,
        # so it must be invalidated whenever the nullspace config / geometry
        # changes (e.g. mesh deformation) — see _build_velocity_rotation_nullspace.
        self._velocity_rotation_nullspace = None

    def _pressure_dirichlet_bcs(self):
        """Return essential boundary conditions applied to the pressure field."""

        pressure_field_id = getattr(getattr(self, "p", None), "field_id", None)
        if pressure_field_id is None:
            raise RuntimeError("Pressure field is unavailable; cannot inspect pressure Dirichlet BCs.")

        return [bc for bc in self.essential_bcs if bc.f_id == pressure_field_id]

    def _build_pressure_nullspace_vector(self):
        """Create the constant-pressure basis vector for the coupled Stokes DM."""

        template_vec = self.dm.getGlobalVec()
        try:
            null_vec = template_vec.duplicate()
        finally:
            self.dm.restoreGlobalVec(template_vec)

        null_vec.set(0.0)

        pressure_is = self._subdict["pressure"][0]
        pressure_subvec = null_vec.getSubVector(pressure_is)
        pressure_subvec.set(1.0)
        null_vec.restoreSubVector(pressure_is, pressure_subvec)

        return null_vec

    def _build_velocity_nullspace_vector(self, mode):
        """Create a velocity nullspace basis vector from a user-supplied mode."""

        mode_values = np.asarray(uw.function.evaluate(mode, self.u.coords_nd), dtype=np.float64)
        if mode_values.ndim == 1:
            mode_values = mode_values.reshape(-1, 1)
        elif mode_values.ndim > 2:
            mode_values = mode_values.reshape(mode_values.shape[0], -1)

        if mode_values.shape != (self.u.coords_nd.shape[0], self.u.num_components):
            raise ValueError(
                "Velocity nullspace mode evaluation must return an array with shape "
                f"({self.u.coords_nd.shape[0]}, {self.u.num_components}); got {mode_values.shape}."
            )

        template_vec = self.dm.getGlobalVec()
        try:
            null_vec = template_vec.duplicate()
        finally:
            self.dm.restoreGlobalVec(template_vec)

        null_vec.set(0.0)

        velocity_is, velocity_subdm = self._subdict["velocity"]
        velocity_basis = self.u.vec.duplicate()
        try:
            velocity_basis.array[:] = mode_values.reshape(velocity_basis.array.shape)
            velocity_subvec = null_vec.getSubVector(velocity_is)
            velocity_subdm.localToGlobal(velocity_basis, velocity_subvec, addv=False)
            null_vec.restoreSubVector(velocity_is, velocity_subvec)
        finally:
            velocity_basis.destroy()

        return null_vec

    def _build_block_gauge_nullspace_vector(self):
        """Combined constant-(pressure, multiplier) gauge mode (block-constrained).

        On a free-slip (non-Dirichlet) boundary delta_u.n != 0, so
        int_Gamma n.delta_u = int_Omega div(delta_u) != 0, and a constant
        pressure (Bᵀ1) and a constant multiplier (Cᵀ1) couple to the SAME u-row
        functional. The genuine near-null mode is therefore the COMBINED
        (p = +1, h = -1 everywhere, u = 0): the u-row contribution
        (1)·int n.delta_u + (-1)·int n.delta_u cancels, the h-row leaves only
        eM·(-1) ~ 0. This replaces the pure constant-pressure mode for the block
        solver (which is NOT a null mode when the constraint boundary is free).
        """
        template_vec = self.dm.getGlobalVec()
        try:
            null_vec = template_vec.duplicate()
        finally:
            self.dm.restoreGlobalVec(template_vec)

        null_vec.set(0.0)

        pressure_is = self._subdict["pressure"][0]
        p_sub = null_vec.getSubVector(pressure_is)
        p_sub.set(1.0)
        null_vec.restoreSubVector(pressure_is, p_sub)

        for cbc in self._block_constraint_bcs:
            mult_is = self._subdict[cbc.lam._solver_field_name][0]
            m_sub = null_vec.getSubVector(mult_is)
            m_sub.set(-1.0)
            null_vec.restoreSubVector(mult_is, m_sub)

        return null_vec

    def _build_stokes_nullspace(self):
        """Create the configured coupled Stokes nullspace basis."""

        basis_vectors = []

        if self._block_constraint_bcs:
            # Block-constrained free-slip: the gauge mode is the COMBINED
            # constant-(pressure, multiplier) vector, not a pure constant
            # pressure (see _build_block_gauge_nullspace_vector).
            if self._petsc_use_pressure_nullspace:
                basis_vectors.append(self._build_block_gauge_nullspace_vector())
        elif self._petsc_use_pressure_nullspace:
            basis_vectors.append(self._build_pressure_nullspace_vector())

        for mode in self._petsc_velocity_nullspace_basis:
            basis_vectors.append(self._build_velocity_nullspace_vector(mode))

        if not basis_vectors:
            return None

        orthonormal_basis = []
        for basis_vec in basis_vectors:
            for orth_vec in orthonormal_basis:
                basis_vec.axpy(-orth_vec.dot(basis_vec), orth_vec)

            basis_norm = basis_vec.norm()
            if np.isclose(basis_norm, 0.0):
                raise ValueError(
                    "Configured Stokes nullspace basis contains a dependent or zero mode."
                )

            basis_vec.scale(1.0 / basis_norm)
            orthonormal_basis.append(basis_vec)

        self._stokes_nullspace_basis = tuple(orthonormal_basis)
        self._stokes_nullspace = PETSc.NullSpace().create(
            constant=False,
            vectors=self._stokes_nullspace_basis,
            comm=self.dm.comm,
        )

        return self._stokes_nullspace

    def _setup_block_fieldsplit_options(self):
        """Collapse the 3+ field DM to a 2-way velocity | [p,h] Schur split.

        We group by DM FIELD INDEX (pc_fieldsplit_0_fields=0,
        pc_fieldsplit_1_fields=1,2,...) rather than by IS: a field-index split
        keeps the DM-field association, so geometric multigrid / FMG on the
        velocity block can build its interpolation hierarchy (an IS-defined
        split has no DM and PCMG errors out with PETSC_ERR_SUP). The grouped
        splits are named "0"/"1", so we MIRROR the user-facing
        fieldsplit_velocity_* / fieldsplit_pressure_* options (defaults + any
        user/FMG overrides) onto fieldsplit_0_* / fieldsplit_1_* — existing
        configs and FMG harnesses then apply unchanged. Must run before
        setFromOptions. No-op if the user chose a non-fieldsplit pc_type.
        """
        opts = self.petsc_options
        if str(opts.getAll().get("pc_type", "")) != "fieldsplit":
            return  # respect a user's direct (lu) solve of the monolithic system

        group1 = ["1"] + [str(cbc.lam._solver_field_id) for cbc in self._block_constraint_bcs]
        opts["pc_fieldsplit_0_fields"] = "0"
        opts["pc_fieldsplit_1_fields"] = ",".join(group1)

        # Mirror velocity_->0_ and pressure_->1_, but DON'T clobber any
        # fieldsplit_0_*/fieldsplit_1_* the user set explicitly (so the grouped
        # [p,h] block PC can be overridden directly).
        allopts = opts.getAll()
        for key, val in list(allopts.items()):
            mirrored = None
            if key.startswith("fieldsplit_velocity_"):
                mirrored = "fieldsplit_0_" + key[len("fieldsplit_velocity_"):]
            elif key.startswith("fieldsplit_pressure_"):
                mirrored = "fieldsplit_1_" + key[len("fieldsplit_pressure_"):]
            if mirrored is not None and mirrored not in allopts:
                opts[mirrored] = None if val in (None, "") else val

    def _attach_stokes_nullspace(self):
        """Attach the configured coupled Stokes nullspace to the solver matrices."""

        if (not self._petsc_use_pressure_nullspace
                and not self._petsc_velocity_nullspace_basis
                and not self._block_constraint_bcs):
            return

        pressure_bcs = self._pressure_dirichlet_bcs()
        if pressure_bcs:
            boundaries = ", ".join(sorted({bc.boundary for bc in pressure_bcs}))
            raise ValueError(
                "PETSc Stokes nullspace support requires the pressure field to be "
                f"free of Dirichlet boundary conditions. Found pressure Dirichlet BCs on: {boundaries}"
            )

        if "pressure" not in self._subdict or "velocity" not in self._subdict:
            raise RuntimeError("Velocity/pressure field decomposition is unavailable; cannot attach nullspace.")

        self.snes.setUp()

        jacobian = self.snes.getJacobian()
        operator_matrix = jacobian[0]
        preconditioner_matrix = jacobian[1] if len(jacobian) > 1 else None

        nullspace = self._stokes_nullspace
        if nullspace is None:
            nullspace = self._build_stokes_nullspace()

        if nullspace is None:
            return

        operator_matrix.setNullSpace(nullspace)
        operator_matrix.setTransposeNullSpace(nullspace)

        if preconditioner_matrix is not None:
            preconditioner_matrix.setNullSpace(nullspace)
            preconditioner_matrix.setTransposeNullSpace(nullspace)

        if self.verbose and uw.mpi.rank == 0:
            print(
                f"Stokes Saddle Pt ({self.name}): attached Stokes nullspace with "
                f"{len(self._stokes_nullspace_basis)} basis mode(s)",
                flush=True,
            )

    def _build_velocity_rotation_nullspace(self):
        """Cache a velocity-rotation-only ``MatNullSpace`` (zero in pressure /
        multiplier blocks). Used to project the rigid-rotation gauge out of the
        converged solution — see ``_remove_velocity_rotation_gauge``. Cached;
        the cache is invalidated by ``_reset_stokes_nullspace`` / solver rebuild
        because the modes depend on node coordinates."""
        if self._velocity_rotation_nullspace is not None:
            return self._velocity_rotation_nullspace
        if not self._petsc_velocity_nullspace_basis:
            return None

        vecs = [self._build_velocity_nullspace_vector(m)
                for m in self._petsc_velocity_nullspace_basis]
        orthonormal = []
        for bvec in vecs:
            for ovec in orthonormal:
                bvec.axpy(-ovec.dot(bvec), ovec)
            bnorm = bvec.norm()
            if np.isclose(bnorm, 0.0):
                bvec.destroy()
                continue
            bvec.scale(1.0 / bnorm)
            orthonormal.append(bvec)
        if not orthonormal:
            return None

        self._velocity_rotation_nullspace = PETSc.NullSpace().create(
            constant=False, vectors=tuple(orthonormal), comm=self.dm.comm)
        return self._velocity_rotation_nullspace

    def _remove_velocity_rotation_gauge(self, gvec):
        """Project the rigid-body rotation gauge out of the converged solution.

        For a free-slip (non-Dirichlet) velocity, the rigid rotations are a true
        nullspace of the velocity block (A_uu·rotation = 0). The monolithic
        nullspace attached for the solve is NOT removed inside a fieldsplit/Schur
        iteration — the inner velocity KSP has no rotation nullspace, so it leaves
        an unconstrained rigid rotation in the velocity. Because the operator is
        blind to it, the true residual still converges to machine precision, but
        the rotation amplitude is PARTITION-DEPENDENT (the tangential velocity then
        differs serial-vs-parallel by ~0.1-1% while the physical / radial flow is
        partition-clean to round-off). Since the rotation is a genuine nullspace,
        removing it from the converged ``gvec`` does NOT change the residual and
        yields the same rotation-free solution on any decomposition. This is the
        velocity analogue of the constant-pressure gauge (see set_pressure_gauge).
        No-op when there are no velocity rotation modes.

        Why a post-solve projection rather than attaching the rotation nullspace
        to the fieldsplit velocity SUB-BLOCK so the inner KSP removes it in-solve:
        the in-solve route was implemented and MEASURED, and it does not fix the
        gauge. A ``DMSetNullSpaceConstructor(dm, 0, ...)`` (rotations-only, built
        with ``DMPlexCreateRigidBody``) DOES attach the rotation nullspace to the
        fieldsplit velocity block — verified: the velocity sub-block operator
        carries the 1 (2-D) / 3 (3-D) rotation modes after PC setup — yet the
        converged GLOBAL velocity still retained the full rotation gauge (rotation
        coefficient ~0.18, i.e. unchanged). Removing a nullspace gauge is a
        projection on the converged GLOBAL solution; attaching the nullspace to a
        sub-block operator only constrains the inner Krylov solves, and with
        ``fgmres`` + a variable (fieldsplit/Schur) preconditioner that reintroduces
        rotation, the assembled outer solution is not gauge-free. (PETSc also
        dispatches ``DMCreateSubDM`` nullspace constructors by SUB-DM-LOCAL field
        index, so a field-0 constructor needs a block-sniffing guard to avoid
        leaking onto the ``[p,h]`` Schur block — but even with that, the global
        gauge persists.) So the explicit post-solve projection is the correct AND
        necessary mechanism, not a stopgap: it is mathematically exact (the rotation
        is a true nullspace, so it does not change the residual) and robust to
        operator rebuild. Building only the rotation modes (zero in the pressure /
        multiplier blocks) keeps it from touching the pressure/multiplier gauge,
        which is handled by the monolithic nullspace."""
        rot_ns = self._build_velocity_rotation_nullspace()
        if rot_ns is not None:
            rot_ns.remove(gvec)


    # TODO(DESIGN): residual-term naming is inconsistent (F0/F1 vs f0, plus the
    # redundant uf0/uF1 aliases used below); settle one scheme rather than
    # adding new spellings.

    def _object_viewer(self):
        '''This will add specific information about this object to the generic class viewer
        '''
        from IPython.display import Latex, Markdown, display
        from textwrap import dedent

        uf0 = self.F0.sym
        uF1 = self.F1.sym
        pF0 = self.PF0.sym

        if self.penalty.sym == 0:
            uF1 = self.F1.sym.subs(self.penalty, self.penalty.sym)

        eqF1 = "$\\tiny \\quad \\nabla \\cdot \\color{Blue}" + sympy.latex( uF1 )+"$ + "
        eqf0 = "$\\tiny \\phantom{ \\quad \\nabla \\cdot} \\color{DarkRed}" + sympy.latex( uf0 )+"\\color{Black} = 0 $"
        eqp0 = "$\\tiny \\phantom{ \\quad \\nabla \\cdot} " + sympy.latex( pF0 ) + " = 0 $"

        # feedback on this instance
        display(
            Markdown(f"# Underworld / PETSc General Saddle Point Equation Solver"),
            Markdown(f"Primary problem: "),
            Latex(eqF1), Latex(eqf0),
            Markdown(f"Constraint: "),
            Latex(eqp0 ),
        )

        exprs = uw.function.fn_extract_expressions(self.F0)
        exprs = exprs.union(uw.function.fn_extract_expressions(self.F1))
        exprs = exprs.union(uw.function.fn_extract_expressions(self.PF0))

        if len(exprs) != 0:
            display(Markdown("*Where:*"))

            for expr in exprs:
                expr._object_viewer()

        display(
            Markdown(fr"# Boundary Conditions"),)

        bc_table = "| Type   | Boundary | Expression | \n"
        bc_table += "|:------------------------ | -------- | ---------- | \n"

        for bc in self.essential_bcs:
            bc_table += f"| **{bc.type}** | {bc.boundary} | ${sympy.latex(bc.fn.T)}  $ | \n"
        for bc in self.natural_bcs:
                bc_table += f"| **{bc.type}** | {bc.boundary} | ${sympy.latex(bc.fn_f.T)}  $ | \n"

        display(Markdown(bc_table))

        display(Markdown(fr"This solver is formulated as a {self.mesh.dim} dimensional problem with a {self.mesh.cdim} dimensional mesh"))

        return

    def validate_solver(self):
        """Checks to see if the required properties have been set"""

        name = self.__class__.__name__

        if not isinstance(self.u, uw.discretisation._MeshVariable):
            print(f"Vector of unknowns required")
            print(f"{name}.u = uw.discretisation.MeshVariable(...)")
            raise RuntimeError("Unknowns: MeshVariable is required")

        if not isinstance(self.p, uw.discretisation._MeshVariable):
            print(f"Vector of constraint unknowns required")
            print(f"{name}.p = uw.discretisation.MeshVariable(...)")
            raise RuntimeError("Constraint (Pressure): MeshVariable is required")

        if not isinstance(self.constitutive_model, uw.constitutive_models.Constitutive_Model):
            print(f"Constitutive model required")
            print(f"{name}.constitutive_model = uw.constitutive_models...")
            raise RuntimeError("Constitutive Model is required")

        return

    def get_dof_partition(self,
                          section_type: str,
                          filename: Optional[str | None] = None,
                          outputPath: Optional[str] = ""):
        """
        Obtains how the degrees of freedom (DOF) are distributed/divided among the processors and saves them in an h5 file.
        Parameters
        ----------
        section_type:
            Can be: "local" which includes DOFs from ghost points or "global" which differentiates DOFs from ghost points by having negative values.
        filename:
            Output file name. If None, will print out results; if set to a string, the output files will be <filename>_<section_type>.u.h5 and <filename>_<section_type>.p.h5.
        outputPath:
            Path of directory where data is saved. If left empty it will save the data in the current working directory.
        """
        # NOTE: supposed to inherit get_dof_partition from SolverBaseClass
        # NOTE: _get_dof_partition_by_field_id is defined in SolverBaseClass

        self.validate_solver()

        u_id = self.Unknowns.u.field_id
        fname = None if filename is None else f"{filename}_{section_type}.u.h5"

        self._get_dof_partition_by_field_id(section_type    = section_type,
                                            field_id        = u_id,
                                            filename        = fname,
                                            outputPath      = outputPath)

        p_id = self.Unknowns.p.field_id
        fname = None if filename is None else f"{filename}_{section_type}.p.h5"

        self._get_dof_partition_by_field_id(section_type    = section_type,
                                            field_id        = p_id,
                                            filename        = fname,
                                            outputPath      = outputPath)

        return

    @timing.routine_timer_decorator
    def _setup_pointwise_functions(self, verbose=False, debug=False, debug_name=None):
        import sympy

        # Any property changes will trigger this
        if self.is_setup:
            if verbose and uw.mpi.rank == 0:
                print(f"SNES_Stokes_SaddlePt ({self.name}): Pointwise functions do not need to be rebuilt", flush=True)
            return
        else:
            if verbose and uw.mpi.rank == 0:
                print(f"SNES_Stokes_SaddlePt ({self.name}): Pointwise functions need to be built", flush=True)

        dim  = self.mesh.dim
        cdim = self.mesh.cdim
        N = self.mesh.N

        sympy.core.cache.clear_cache()

        # RESIDUAL: don't unwrap here — let getext()'s two-phase unwrap handle
        # it (preserves constant UWexpressions as symbols for constants[]). The
        # JACOBIAN sources are unwrapped separately below (see _jac_source) so
        # the derivative sees through the viscosity — that is the Newton fix.
        F0  = sympy.Array(self.F0.sym)
        F1  = sympy.Array(self.F1.sym)
        PF0  = sympy.Array(self.PF0.sym)

        # JIT compilation needs immutable, matrix input (not arrays)
        self._u_F0 = sympy.ImmutableDenseMatrix(F0)
        self._u_F1 = sympy.ImmutableDenseMatrix(F1)
        self._p_F0 = sympy.ImmutableDenseMatrix(PF0)

        fns_residual = [self._u_F0, self._u_F1, self._p_F0]

        ## jacobian terms

        fns_jacobian = []

        ## NOTE PETSc and sympy require some re-ordering so that
        ## a `for element in Matrix:` loop produces functions
        ## in the order that the PETSc jacobian routines expect.
        ## This needs checking and completion. Especialy if we are
        ## going to do this for arbitrary block systems.
        ## It's a bit easier for Stokes where P is a scalar field

        # This is needed to eliminate extra dims in the tensor
        U = sympy.Array(self.u.sym).reshape(dim)
        P = sympy.Array(self.p.sym).reshape(1)

        # Expand UWexpressions down to (but NOT including) constant atoms,
        # element-wise, for the Jacobian derivative ONLY. This exposes the
        # field / grad-v dependence of the (effective) viscosity so that
        # derive_by_array forms the full Newton tangent (e.g. Min -> Heaviside
        # yield switch), instead of freezing eta_eff as an opaque atom and
        # silently running a Picard / defect-correction tangent. Truly-constant
        # atoms (eta0, tau_y, ...) survive as symbols so the constants[]
        # runtime-update mechanism is preserved (the keep-constants predicate
        # is shared with getext()'s _extract_constants, so they cannot drift).
        # The residual fns above (self._u_F0/_u_F1/_p_F0) are left untouched —
        # getext() unwraps those itself. For constant-viscosity problems this
        # is a no-op (eta has no grad-v dependence) so the Jacobian is
        # bit-identical. See docs/developer/design/jacobian-unwrap-constants-bug.md
        #
        # (see consistent_jacobian / _jacobian_source: default Picard, bit-
        # identical; True -> Newton; "continuation" -> alpha-blended.)
        F0_jac  = self._jacobian_source(F0)
        PF0_jac = self._jacobian_source(PF0)

        # Optional override: differentiate an alternative F1 to build the
        # uu and up Jacobian blocks while leaving the residual F1
        # unchanged. Used for inexact Newton (e.g. softmin Jacobian with
        # Min residual at a yield kink). When None, autodiff F1 itself —
        # the Newton flux being the model's smooth law (flux_jacobian) if it
        # provides one, else the exact flux unwrapped.
        F1_jac_src = getattr(self, "_F1_jacobian_source", None)
        if F1_jac_src is not None:
            F1_for_jac = self._jacobian_source(sympy.Array(F1_jac_src))
        else:
            F1_for_jac = self._jacobian_source(F1, self._newton_flux(F1))
        # Normalise to strict (dim, dim) Array indexing for the explicit
        # Jacobian loops below.
        F1_for_jac = sympy.Array(F1_for_jac).reshape(dim, dim)

        # Explicit-index Jacobian construction — writes each entry directly
        # into PETSc's flat [fc, gc, df, dg] layout via row-major matrices.
        # sympy.derive_by_array is dx-FIRST (derivative indices lead), so the
        # previous derive_by_array + permutedims((0,2,1,3)) form assembled the
        # MAJOR TRANSPOSE of uu_G3: it placed dF1[gc,dg]/dL[fc,df] in the
        # [fc,gc,df,dg] slot. Invisible whenever the tangent has major symmetry
        # (frozen C, isotropic eta(edot) Newton, linear transverse isotropy),
        # wrong exactly when it does not (transverse-isotropic Newton — issue
        # #457). Same explicit-loop construction as SNES_Vector above; the
        # layout contract is docs/developer/subsystems/petsc-jacobian-layout.md.
        U_list = [self.u.sym[0, c] for c in range(dim)]
        L = self.Unknowns.L
        Nc = dim

        # Normalise the F0 Jacobian source to a flat list of dim scalar
        # entries — F0 arrives as (1, dim) or (dim, 1) depending on how the
        # template/bodyforce was written (Array indexing is strict).
        _f0_flat = sympy.Array(F0_jac).reshape(dim)
        f0_jac_list = [_f0_flat[c] for c in range(dim)]

        # uu_G0[fc, gc]                  = dF0[fc] / dU[gc]
        G0 = sympy.zeros(Nc, Nc)
        for fc in range(Nc):
            for gc in range(Nc):
                G0[fc, gc] = sympy.diff(f0_jac_list[fc], U_list[gc])

        # uu_G1[fc*Nc + gc, dg]          = dF0[fc] / dL[gc, dg]
        G1 = sympy.zeros(Nc * Nc, dim)
        for fc in range(Nc):
            for gc in range(Nc):
                for dg in range(dim):
                    G1[fc * Nc + gc, dg] = sympy.diff(f0_jac_list[fc], L[gc, dg])

        # uu_G2[fc*Nc + gc, df]          = dF1[fc, df] / dU[gc]
        G2 = sympy.zeros(Nc * Nc, dim)
        for fc in range(Nc):
            for gc in range(Nc):
                for df in range(dim):
                    G2[fc * Nc + gc, df] = sympy.diff(F1_for_jac[fc, df], U_list[gc])

        # uu_G3[fc*Nc + gc, df*dim + dg] = dF1[fc, df] / dL[gc, dg]
        G3 = sympy.zeros(Nc * Nc, dim * dim)
        for fc in range(Nc):
            for gc in range(Nc):
                for df in range(dim):
                    for dg in range(dim):
                        G3[fc * Nc + gc, df * dim + dg] = sympy.diff(
                            F1_for_jac[fc, df], L[gc, dg]
                        )

        self._uu_G0 = sympy.ImmutableMatrix(G0)
        self._uu_G1 = sympy.ImmutableMatrix(G1)
        self._uu_G2 = sympy.ImmutableMatrix(G2)
        self._uu_G3 = sympy.ImmutableMatrix(G3)

        fns_jacobian += [self._uu_G0, self._uu_G1, self._uu_G2, self._uu_G3]

        # U/P block. The constraint field is scalar (Nc_p == 1), so the g-index
        # is size 1 and only the derivative indices need explicit placement.
        p_scalar = self.p.sym[0]
        Gp = self._G  # (1, dim) row of dp/dx_dg symbols

        # up_G0[fc, 0]                   = dF0[fc] / dp
        G0 = sympy.zeros(dim, 1)
        for fc in range(dim):
            G0[fc, 0] = sympy.diff(f0_jac_list[fc], p_scalar)

        # up_G1[fc, dg]                  = dF0[fc] / d(dp/dx_dg)
        G1 = sympy.zeros(dim, dim)
        for fc in range(dim):
            for dg in range(dim):
                G1[fc, dg] = sympy.diff(f0_jac_list[fc], Gp[0, dg])

        # up_G2[fc, df]                  = dF1[fc, df] / dp
        G2 = sympy.zeros(dim, dim)
        for fc in range(dim):
            for df in range(dim):
                G2[fc, df] = sympy.diff(F1_for_jac[fc, df], p_scalar)

        # up_G3[fc*dim + df, dg]         = dF1[fc, df] / d(dp/dx_dg)
        G3 = sympy.zeros(dim * dim, dim)
        for fc in range(dim):
            for df in range(dim):
                for dg in range(dim):
                    G3[fc * dim + df, dg] = sympy.diff(F1_for_jac[fc, df], Gp[0, dg])

        self._up_G0 = sympy.ImmutableMatrix(G0)  # zero in stokes tests
        self._up_G1 = sympy.ImmutableMatrix(G1)  # zero in stokes tests
        self._up_G2 = sympy.ImmutableMatrix(G2)  # pressure coupling
        self._up_G3 = sympy.ImmutableMatrix(G3)  # zero in stokes tests

        fns_jacobian += [self._up_G0, self._up_G1, self._up_G2, self._up_G3]

        # P/U block (check permutations)

        G0 = sympy.derive_by_array(PF0_jac, self.u.sym)
        G1 = sympy.derive_by_array(PF0_jac, self.Unknowns.L)
        # (there is no FP1 flux term, so the pu_G2 / pu_G3 blocks are empty)

        self._pu_G0 = sympy.ImmutableMatrix(G0.reshape(dim))  # non zero
        self._pu_G1 = sympy.ImmutableMatrix(G1.reshape(dim*dim))  # non-zero
        fns_jacobian += [self._pu_G0, self._pu_G1]

        ## PP block is a preconditioner term, not auto-constructed

        if self.saddle_preconditioner is not None:
            self._pp_G0 = self.saddle_preconditioner
        else:
            self._pp_G0 = 1 / self.constitutive_model.K

        fns_jacobian.append(self._pp_G0)

        ## Lagrange-multiplier rows (block-constrained Stokes). Guarded: no-op
        ## for ordinary Stokes. Each multiplier h_k contributes an interior
        ## screening residual  f0 = eps_k * h_k  and a diagonal mass Jacobian
        ## hh_G0 = eps_k. The screening de-singularises the otherwise-empty
        ## interior h block; with no boundary coupling (M1) h is driven to 0.
        ## Boundary coupling (C, C^T) and off-diagonal blocks are added by the
        ## natural-bc loop in later milestones.
        self._h_F0 = []
        self._hh_G0 = []
        for mvar, eps in zip(self._multipliers, self._multiplier_screening):
            h_F0 = sympy.ImmutableDenseMatrix(sympy.Array([eps * mvar.sym[0]]).reshape(1))
            hh_G0 = sympy.ImmutableMatrix(
                sympy.derive_by_array(sympy.Array([eps * mvar.sym[0]]), mvar.sym).reshape(1, 1)
            )
            self._h_F0.append(h_F0)
            self._hh_G0.append(hh_G0)
            fns_residual.append(h_F0)
            fns_jacobian.append(hh_G0)

        # Now natural bcs (compiled into boundary integral terms)
        # Need to loop on them all ...

        fns_bd_residual = []
        fns_bd_jacobian = []

        for index, bc in enumerate(self.natural_bcs):

            if bc.fn_f is not None:

                bd_F0  = sympy.Array(bc.fn_f)

                bc.fns["u_f0"] = sympy.ImmutableDenseMatrix(bd_F0)
                fns_bd_residual += [bc.fns["u_f0"]]

                # Boundary Jacobians follow the same PETSc [fc, gc, df, dg]
                # layout as the bulk blocks — build them with the same
                # explicit-index loops (the old permutedims form transposed
                # fc/gc here too; harmless only while every natural-BC
                # tangent happened to be symmetric).
                bd_f0_list = list(sympy.Array(bc.fn_f).reshape(dim))

                # uu_G0[fc, gc]         = d bd_F0[fc] / dU[gc]
                G0 = sympy.zeros(dim, dim)
                for fc in range(dim):
                    for gc in range(dim):
                        G0[fc, gc] = sympy.diff(bd_f0_list[fc], U_list[gc])

                # uu_G1[fc*dim + gc, dg] = d bd_F0[fc] / dL[gc, dg]
                G1 = sympy.zeros(dim * dim, dim)
                for fc in range(dim):
                    for gc in range(dim):
                        for dg in range(dim):
                            G1[fc * dim + gc, dg] = sympy.diff(bd_f0_list[fc], L[gc, dg])

                bc.fns["uu_G0"] = sympy.ImmutableMatrix(G0)
                bc.fns["uu_G1"] = sympy.ImmutableMatrix(G1)
                fns_bd_jacobian += [bc.fns["uu_G0"], bc.fns["uu_G1"]]

                # up_G0[fc, 0]          = d bd_F0[fc] / dp
                G0 = sympy.zeros(dim, 1)
                for fc in range(dim):
                    G0[fc, 0] = sympy.diff(bd_f0_list[fc], p_scalar)

                # up_G1[fc, dg]         = d bd_F0[fc] / d(dp/dx_dg)
                G1 = sympy.zeros(dim, dim)
                for fc in range(dim):
                    for dg in range(dim):
                        G1[fc, dg] = sympy.diff(bd_f0_list[fc], Gp[0, dg])

                bc.fns["up_G0"] = sympy.ImmutableMatrix(G0)
                bc.fns["up_G1"] = sympy.ImmutableMatrix(G1)
                fns_bd_jacobian += [bc.fns["up_G0"], bc.fns["up_G1"]]

                # Gradient boundary residual (f1_bd) and its Jacobians (g2, g3)
                # Used by Nitsche-type BCs; None for standard natural BCs.
                if bc.fn_F is not None:
                    bd_F1 = sympy.Array(bc.fn_F).reshape(dim, dim)
                    bc.fns["u_F1"] = sympy.ImmutableDenseMatrix(bd_F1)
                    fns_bd_residual += [bc.fns["u_F1"]]

                    # Nitsche gradient-traction depends on eta(grad v); unwrap +
                    # smooth-kink the Jacobian source so its tangent is Newton-
                    # consistent (same fix as the bulk). Residual u_F1 stays exact.
                    bd_F1_jac = self._jacobian_source(bd_F1)

                    # uu_G2[fc*dim + gc, df]          = d bd_F1[fc, df] / dU[gc]
                    G2 = sympy.zeros(dim * dim, dim)
                    for fc in range(dim):
                        for gc in range(dim):
                            for df in range(dim):
                                G2[fc * dim + gc, df] = sympy.diff(
                                    bd_F1_jac[fc, df], U_list[gc]
                                )

                    # uu_G3[fc*dim + gc, df*dim + dg] = d bd_F1[fc, df] / dL[gc, dg]
                    G3 = sympy.zeros(dim * dim, dim * dim)
                    for fc in range(dim):
                        for gc in range(dim):
                            for df in range(dim):
                                for dg in range(dim):
                                    G3[fc * dim + gc, df * dim + dg] = sympy.diff(
                                        bd_F1_jac[fc, df], L[gc, dg]
                                    )

                    bc.fns["uu_G2"] = sympy.ImmutableMatrix(G2)
                    bc.fns["uu_G3"] = sympy.ImmutableMatrix(G3)
                    fns_bd_jacobian += [bc.fns["uu_G2"], bc.fns["uu_G3"]]

                    # up_G2[fc, df]          = d bd_F1[fc, df] / dp
                    G2 = sympy.zeros(dim, dim)
                    for fc in range(dim):
                        for df in range(dim):
                            G2[fc, df] = sympy.diff(bd_F1[fc, df], p_scalar)

                    # up_G3[fc*dim + df, dg] = d bd_F1[fc, df] / d(dp/dx_dg)
                    G3 = sympy.zeros(dim * dim, dim)
                    for fc in range(dim):
                        for df in range(dim):
                            for dg in range(dim):
                                G3[fc * dim + df, dg] = sympy.diff(
                                    bd_F1[fc, df], Gp[0, dg]
                                )

                    bc.fns["up_G2"] = sympy.ImmutableMatrix(G2)
                    bc.fns["up_G3"] = sympy.ImmutableMatrix(G3)
                    fns_bd_jacobian += [bc.fns["up_G2"], bc.fns["up_G3"]]

                # Pressure boundary residual and Jacobians (pu, pp blocks)
                # Nitsche BCs provide fn_p (pressure boundary residual = u.n - g);
                # standard natural BCs have fn_p = None → zeros.
                if hasattr(bc, 'fn_p') and bc.fn_p is not None:
                    bd_PF0 = sympy.Array(bc.fn_p).reshape(1)
                    bc.fns["p_f0"] = sympy.ImmutableDenseMatrix(bd_PF0)
                    fns_bd_residual += [bc.fns["p_f0"]]

                    G0 = sympy.derive_by_array(bd_PF0, self.Unknowns.u.sym)
                    G1 = sympy.derive_by_array(bd_PF0, self.Unknowns.L)
                    bc.fns["pu_G0"] = sympy.ImmutableMatrix(G0.reshape(dim))
                    bc.fns["pu_G1"] = sympy.ImmutableMatrix(G1.reshape(dim*dim))
                    fns_bd_jacobian += [bc.fns["pu_G0"], bc.fns["pu_G1"]]
                else:
                    bc.fns["pu_G0"] = sympy.ImmutableMatrix(sympy.Matrix.zeros(rows=1, cols=dim))
                    bc.fns["pu_G1"] = sympy.ImmutableMatrix(sympy.Matrix.zeros(rows=dim, cols=dim))
                    fns_bd_jacobian += [bc.fns["pu_G0"], bc.fns["pu_G1"]]

                bc.fns["pp_G0"] = sympy.ImmutableMatrix([0])
                fns_bd_jacobian += [bc.fns["pp_G0"]]


        ## Lagrange-multiplier boundary coupling (block-constrained Stokes).
        ## For each constraint, the boundary contributes the symmetric pair
        ##   C^T :  field-0 traction   fn_f = h * n         (u-row, ∫_Γ h (n·δu))
        ##   C   :  field-2 constraint fn_h = n·u − g       (h-row, ∫_Γ ψ(n·u−g))
        ## off-diagonal boundary Jacobians  uh = ∂fn_f/∂h = n  (0, h)  and
        ## hu = ∂fn_h/∂u = n  (h, 0), plus the augmented-Lagrangian uu boundary
        ## stiffness uu = ∂fn_f/∂u = r·(n⊗n)  (0, 0) which conditions the [p,h]
        ## Schur complement (r=0 ⇒ bare KKT, uu=0). Guarded: no-op for ordinary
        ## Stokes.
        for cbc in self._block_constraint_bcs:
            n_row = cbc.normal           # sympy 1×dim Matrix
            g_sym = cbc.g
            r_sym = cbc.augmentation     # augmented-Lagrangian parameter
            H = sympy.Array(cbc.lam.sym).reshape(1)
            hsym = cbc.lam.sym[0]

            u_dot_n = sum(n_row[i] * self.u.sym[i] for i in range(dim))

            # u-row residual:  fn_f = h·n  +  r(n·u − g)·n
            # The r-term is the augmented-Lagrangian penalty: it adds a uu
            # boundary stiffness r·(n⊗n) that conditions the Schur complement
            # but does NOT bias the multiplier (the h-row stays the exact
            # constraint, so h still converges to the true normal traction).
            fn_f = sympy.Matrix(
                [(hsym + r_sym * (u_dot_n - g_sym)) * n_row[i] for i in range(dim)]
            ).as_immutable()
            cbc.fns["u_f0"] = sympy.ImmutableDenseMatrix(sympy.Array(fn_f).reshape(dim))
            fns_bd_residual += [cbc.fns["u_f0"]]

            # h-row constraint residual  fn_h = n·u − g
            fn_h = sympy.Array([u_dot_n - g_sym]).reshape(1)
            cbc.fns["h_f0"] = sympy.ImmutableDenseMatrix(fn_h)
            fns_bd_residual += [cbc.fns["h_f0"]]

            # uu (0, 0):  ∂fn_f/∂u = r·(n⊗n)  — AL stiffness (mirror Nitsche shape).
            # Explicit [fc, gc] placement like every other Jacobian block (the
            # content is symmetric, but no permutedims survives on principle —
            # see petsc-jacobian-layout.md).
            G0 = sympy.zeros(dim, dim)
            for fc in range(dim):
                for gc in range(dim):
                    G0[fc, gc] = sympy.diff(fn_f[fc], U_list[gc])
            cbc.fns["uu_G0"] = sympy.ImmutableMatrix(G0)
            fns_bd_jacobian += [cbc.fns["uu_G0"]]

            # uh (0, h):  ∂fn_f/∂h = n  — mirror the up_G0 (velocity,scalar) shape
            G0 = sympy.derive_by_array(sympy.Array(fn_f), H)
            cbc.fns["uh_G0"] = sympy.ImmutableMatrix(G0.reshape(dim))
            fns_bd_jacobian += [cbc.fns["uh_G0"]]

            # hu (h, 0):  ∂fn_h/∂u = n  — mirror the pu_G0 (scalar,velocity) shape
            G0 = sympy.derive_by_array(fn_h, self.Unknowns.u.sym)
            cbc.fns["hu_G0"] = sympy.ImmutableMatrix(G0.reshape(dim))
            fns_bd_jacobian += [cbc.fns["hu_G0"]]


        self._fns_bd_residual = fns_bd_residual
        self._fns_bd_jacobian = fns_bd_jacobian

        # generate JIT code.
        # first, we must specify the primary fields.
        # these are fields for which the corresponding sympy functions
        # should be replaced with the primary (instead of auxiliary) petsc
        # field value arrays. in this instance, we want to switch out
        # `self.u` and `self.p` for their primary field
        # petsc equivalents. without specifying this list,
        # the aux field equivalents will be used instead, which
        # will give incorrect results for non-linear problems.
        # note also that the order here is important.

        if self.verbose and uw.mpi.rank==0:
            print(f"Stokes: Jacobians complete, now compile", flush=True)

        prim_field_list = [self.u, self.p]
        if self._multipliers:
            prim_field_list = prim_field_list + list(self._multipliers)

        # Essential-BC value functions. (Block-constrained Stokes reduces the
        # interior multiplier DOFs by constraining them directly in the
        # PetscSection — see _constrain_interior_multipliers_in_section — so no
        # extra compiled essential-BC value is needed here.)
        bc_value_fns = [x.fn for x in self.essential_bcs]

        _getext_result = getext(
            self.mesh,
            JITCallbackSet(
                residual=tuple(fns_residual),
                bcs=tuple(bc_value_fns),
                jacobian=tuple(fns_jacobian),
                bd_residual=tuple(fns_bd_residual),
                bd_jacobian=tuple(fns_bd_jacobian),
            ),
            prim_field_list,
            verbose=verbose,
            debug=debug,
            debug_name=debug_name,
            cache=False,
        )
        self.compiled_extensions = _getext_result.ptrobj
        self.ext_dict = _getext_result.fn_dicts
        self.constants_manifest = _getext_result.constants_manifest
        self._current_jit_cache_key = _getext_result.cache_key

        self.is_setup = False

        return

    @timing.routine_timer_decorator
    def _setup_discretisation(self, verbose=False):
        """
        Most of what is in the init phase that is not called by _setup_terms()
        """

        # Grab the mesh
        mesh = self.mesh

        import xxhash
        import numpy as np

        xxh = xxhash.xxh64()
        xxh.update(np.ascontiguousarray(mesh.X.coords))
        mesh_dm_coord_hash = xxh.intdigest()

        # if we already set up the dm and the coordinates in the mesh dm have not
        # changed then we do not need to do everything here

        if self.dm is not None and self.mesh_dm_coordinate_hash == mesh_dm_coord_hash:
            if verbose and uw.mpi.rank == 0:
                print(f"SNES_Stokes_SaddlePt ({self.name}): Discretisation does not need to be rebuilt", flush=True)
            return

        if self.verbose:
            print(f"{uw.mpi.rank}: Building dm for {self.name}")

        # Keep a note of the coordinates that we use for this setup
        self.mesh_dm_coordinate_hash = mesh_dm_coord_hash

        cdef PtrContainer ext = self.compiled_extensions

        mesh = self.mesh
        u_degree = self.u.degree
        p_degree = self.p.degree
        p_continous = self.p.continuous

        if mesh.qdegree < u_degree:
            print(f"Caution - the mesh quadrature ({mesh.qdegree})is lower")
            print(f"than {u_degree} which is required by the {self.name} solver")

        self.dm_hierarchy = mesh.clone_dm_hierarchy()
        self.dm = self.dm_hierarchy[-1]

        if self.dm.getNumFields() == 0:

            if self.verbose:
               print(f"{uw.mpi.rank}: Building FE / quadrature for {self.name}", flush=True)


            # Both degree AND dual-space options must be set before createDefault().
            # The dual-space options control node placement on simplices: without them,
            # PETSc may choose a different nodal basis than the solver's weak form expects.
            # For P2 the defaults happen to be correct, but P3+ gets wrong node placement.

            options = PETSc.Options()
            options.setValue("private_{}_u_petscspace_degree".format(self.petsc_options_prefix), u_degree)
            options.setValue("private_{}_u_petscdualspace_lagrange_continuity".format(self.petsc_options_prefix), self.u.continuous)
            options.setValue("private_{}_u_petscdualspace_lagrange_node_endpoints".format(self.petsc_options_prefix), False)
            self.petsc_fe_u = PETSc.FE().createDefault(mesh.dim, mesh.dim, mesh.isSimplex, mesh.qdegree, "private_{}_u_".format(self.petsc_options_prefix), PETSc.COMM_SELF)
            self.petsc_fe_u.setName("velocity")
            self.petsc_fe_u_id = self.dm.getNumFields()
            self.dm.setField( self.petsc_fe_u_id, self.petsc_fe_u )

            options.setValue("private_{}_p_petscspace_degree".format(self.petsc_options_prefix), p_degree)
            options.setValue("private_{}_p_petscdualspace_lagrange_continuity".format(self.petsc_options_prefix), p_continous)
            options.setValue("private_{}_p_petscdualspace_lagrange_node_endpoints".format(self.petsc_options_prefix), False)

            self.petsc_fe_p = PETSc.FE().createDefault(mesh.dim,    1, mesh.isSimplex, mesh.qdegree, "private_{}_p_".format(self.petsc_options_prefix), PETSc.COMM_SELF)
            self.petsc_fe_p.setName("pressure")
            self.petsc_fe_p_id = self.dm.getNumFields()
            self.dm.setField( self.petsc_fe_p_id, self.petsc_fe_p)

            # Saddle-point Lagrange multiplier fields (block-constrained Stokes).
            # Registered as extra scalar fields (id 2, 3, ...) at the multiplier's
            # own degree (velocity P2 by default, so the trace reaches every
            # normal-trace DOF). Guarded: no-op for ordinary Stokes.
            for mvar in self._multipliers:
                h_degree = mvar.degree
                h_prefix = "private_{}_{}_".format(self.petsc_options_prefix, mvar._solver_field_name)
                options.setValue(h_prefix + "petscspace_degree", h_degree)
                options.setValue(h_prefix + "petscdualspace_lagrange_continuity", mvar.continuous)
                options.setValue(h_prefix + "petscdualspace_lagrange_node_endpoints", False)
                fe_h = PETSc.FE().createDefault(mesh.dim, 1, mesh.isSimplex, mesh.qdegree, h_prefix, PETSc.COMM_SELF)
                fe_h.setName(mvar._solver_field_name)
                mvar._solver_field_id = self.dm.getNumFields()
                self.dm.setField(mvar._solver_field_id, fe_h)

        self.dm.createDS()

        ## This part is done once on the solver dm ... not required every time we update the functions ...
        ## the values of the natural bcs can be updated

        cdef int ind=1
        cdef int [::1] comps_view  # for numpy memory view
        cdef DM cdm = self.dm

        for index,bc in enumerate(self.natural_bcs):

            if uw.mpi.rank == 0 and self.verbose:
                print("Setting bc {} ({})".format(index, bc.type))
                print(" - field:      {}".format(bc.f_id))
                print(" - component:  {}".format(bc.components))
                print(" - boundary:   {}".format(bc.boundary))
                print(" - fn:         {} ".format(bc.fn_f))

            boundary = bc.boundary
            value = mesh.boundaries[bc.boundary].value
            ind = value

            # use type 5 bc for `DM_BC_ESSENTIAL_FIELD` enum
            # use type 6 bc for `DM_BC_NATURAL_FIELD` enum

            bc_type = 6
            num_constrained_components = bc.components.shape[0]
            comps_view = bc.components
            bc = PetscDSAddBoundary_UW(cdm.dm,
                                bc_type,
                                str(boundary+f"{bc.components}").encode('utf8'),
                                str("UW_Boundaries").encode('utf8'),
                                bc.f_id,  # field ID in the DM
                                num_constrained_components,
                                <const PetscInt *> &comps_view[0],
                                <void (*)() noexcept>NULL,
                                NULL,
                                1,
                                <const PetscInt *> &ind,
                                NULL, )


            self.natural_bcs[index] = self.natural_bcs[index]._replace(PETScID=bc, boundary_label_val=value)


        # Lagrange-multiplier constraint boundaries (block-constrained Stokes).
        # PETSc evaluates a boundary entry's residual/Jacobian ONLY for the
        # field it was registered with (key.field = boundary field). So each
        # constraint registers the boundary TWICE: once for field 0 (carries the
        # u-row traction h·n and the uh Jacobian) and once for the multiplier
        # field (carries the h-row constraint n·u−g and the hu Jacobian).
        # Guarded: no-op for ordinary Stokes.
        cdef int [::1] cbc_comps_view
        cdef int [::1] cbc_hcomps_view
        for cbc in self._block_constraint_bcs:
            cbc_boundary = cbc.boundary
            cbc_value = mesh.boundaries[cbc_boundary].value
            ind = cbc_value
            cbc_fid_h = cbc.lam._solver_field_id

            cbc_comps = np.arange(mesh.dim, dtype=np.int32)
            cbc_comps_view = cbc_comps
            cbc.petsc_id_u = PetscDSAddBoundary_UW(cdm.dm,
                                6,
                                str(cbc_boundary + "_constraint_u").encode('utf8'),
                                str("UW_Boundaries").encode('utf8'),
                                0,  # velocity field: u-row traction + uh Jacobian
                                cbc_comps.shape[0],
                                <const PetscInt *> &cbc_comps_view[0],
                                <void (*)() noexcept>NULL,
                                NULL,
                                1,
                                <const PetscInt *> &ind,
                                NULL, )

            cbc_hcomps = np.array([0], dtype=np.int32)
            cbc_hcomps_view = cbc_hcomps
            cbc.petsc_id_h = PetscDSAddBoundary_UW(cdm.dm,
                                6,
                                str(cbc_boundary + "_constraint_h").encode('utf8'),
                                str("UW_Boundaries").encode('utf8'),
                                cbc_fid_h,  # multiplier field: h-row constraint + hu Jacobian
                                cbc_hcomps.shape[0],
                                <const PetscInt *> &cbc_hcomps_view[0],
                                <void (*)() noexcept>NULL,
                                NULL,
                                1,
                                <const PetscInt *> &ind,
                                NULL, )
            cbc.label_val = cbc_value


        for index,bc in enumerate(self.essential_bcs):
            if uw.mpi.rank == 0 and self.verbose:
                print("Setting bc {} ({})".format(index, bc.type))
                print(" - field:      {}".format(bc.f_id))
                print(" - component:  {}".format(bc.components))
                print(" - boundary:   {}".format(bc.boundary))
                print(" - fn:         {} ".format(bc.fn))
                print(flush=True)


            boundary = bc.boundary
            value = mesh.boundaries[bc.boundary].value
            ind = value

            # use type 5 bc for `DM_BC_ESSENTIAL_FIELD` enum
            # use type 6 bc for `DM_BC_NATURAL_FIELD` enum
            bc_type = 5
            fn_index = self.ext_dict.ebc[sympy.Matrix([[bc.fn]]).as_immutable()]
            num_constrained_components = bc.components.shape[0]
            comps_view = bc.components
            bc = PetscDSAddBoundary_UW(cdm.dm,
                                bc_type,
                                str(boundary+f"{bc.components}").encode('utf8'),
                                str(boundary).encode('utf8'),
                                bc.f_id,  # field ID in the DM
                                num_constrained_components,
                                <const PetscInt *> &comps_view[0],
                                <void (*)() noexcept>ext.fns_bcs[fn_index],
                                NULL,
                                1,
                                <const PetscInt *> &ind,
                                NULL, )

            self.essential_bcs[index] = self.essential_bcs[index]._replace(PETScID=bc, boundary_label_val=value)


        # Boundary-only multiplier reduction (block-constrained Stokes) is applied
        # LATER, in _setup_solver, by constraining the interior (off-constraint-
        # boundary) multiplier DOFs directly in the PetscSection
        # (_constrain_interior_multipliers_in_section). That path is both lossless
        # (the constraint-boundary closure is correct because createClosureIndex
        # has finalised the section) and refined-safe (no DMAddBoundary, which
        # cannot follow section creation). See that method for details.

        for coarse_dm in self.dm_hierarchy:
            self.dm.copyFields(coarse_dm)
            self.dm.copyDS(coarse_dm)

        self.is_setup = False

        return


    def _constrain_interior_multipliers_in_section(self):
        """Lossless, refined-safe boundary-only multiplier reduction.

        Block-constrained Stokes carries one Lagrange multiplier (field ``h``)
        per constraint as a full-domain field, but only its boundary trace is
        physical — the interior ``h`` DOFs are inert and merely inflate the
        ``[p,h]`` Schur block (~ndof/3 extra DOFs). Here we constrain every
        interior (off-constraint-boundary) ``h`` DOF DIRECTLY in the fine DM's
        local PetscSection, so they drop out of the GLOBAL system (smaller Schur
        block) while remaining present in the LOCAL vector at 0 (scatter-back
        stays size-correct, see below).

        This must run AFTER ``createClosureIndex`` (so the local section is
        finalised with the velocity Dirichlet constraints baked in AND the
        constraint-boundary closure is complete — the source of the earlier
        DMAddBoundary path's loss) and BEFORE any consumer of the GLOBAL section
        (the field-index fieldsplit grouping, the SNES, ``createFieldDecomposition``,
        the nullspace). Constraining in the section has no "after section creation"
        restriction, so it is also refined/FMG-safe — unlike the DMAddBoundary
        essential-field pin it replaces.

        Fine ``self.dm`` only: the coarse MG levels never carry an active ``h``
        field in the solve, and ``copyFields``/``copyDS`` copy discretisations and
        the DS (weak forms), not the section, so the velocity MG/FMG hierarchy and
        the field-index fieldsplit grouping are undisturbed (they only see the
        ``[p,h]`` block shrink).

        Note (scatter-back): PetscSection constraints remove DOFs from the GLOBAL
        section only; the LOCAL section still allocates them. The multiplier IS is
        built with ``unconstrained=False`` and (for a scalar ``h``, 1 dof/point)
        appends the local offset of every h-bearing point regardless of
        constraint, so the copy back into the full-domain ``h`` MeshVariable is
        size-correct without change.

        Why this is lossless (and why disabling it is NOT a "more accurate"
        reference). The interior ``h`` rows are the screening block alone,
        ``ε M_ii h_i + ε M_ib h_b = 0`` (no constraint or buoyancy source reaches
        the interior), so the interior multiplier is fully determined by the
        boundary trace, ``h_i = -M_ii^{-1} M_ib h_b`` — a *bounded* O(h_b)
        quantity that feeds back into the boundary row only at O(ε). At ε=1e-6 its
        effect on the physical solution is negligible: pinning ``h_i = 0`` instead
        moves a converged solve by ~1e-8 in velocity and ~1e-5 in topography
        (measured, 2-D annulus). So the pinned DOFs carry no physical signal.

        Disabling the reduction (``_reduce_interior_multiplier = False``) does NOT
        recover lost physics; it re-admits ~ndof/3 interior DOFs whose only
        coupling is the near-singular ``ε M_ii`` block, enlarging and
        ill-conditioning the ``[p,h]`` Schur complement. On a direct/well-converged
        solve the answer is essentially unchanged (lossless, as above); on an
        iterative grouped-Schur solve (e.g. ``snes_type=ksponly`` on the 3-D shell)
        the extra ill-conditioned DOFs degrade convergence and the solve can land
        on a different representative — which is the origin of the "answer moves a
        few percent / parallel spread worsens when off" behaviour. That is a
        conditioning effect, not evidence the knockout drops information. Keep it
        ON; it is the recommended, validated path.
        """
        from petsc4py import PETSc
        import numpy as np

        if not (self._block_constraint_bcs and self._reduce_interior_multiplier):
            return

        dm = self.dm
        Sold = dm.getLocalSection()
        cS, cE = Sold.getChart()
        nF = Sold.getNumFields()

        # 1. Boundary-trace KEEP set + interior-h set, per multiplier field.
        #    createClosureIndex has run, so getTransitiveClosure gives the
        #    COMPLETE closure — no boundary-trace h DOF is ever pinned (lossless).
        interior_pts = {}  # fid_h -> set(points to additionally constrain)
        for cbc in self._block_constraint_bcs:
            fid_h = cbc.lam._solver_field_id
            bvalue = self.mesh.boundaries[cbc.boundary].value
            bd_is = dm.getLabel("UW_Boundaries").getStratumIS(bvalue)
            # Guard for parallel: a rank owning no points with this label value
            # gets back a valid non-None PETSc.IS with size 0 (bool(bd_is) = True,
            # bd_is.handle != 0). Calling .getIndices() on that IS segfaults.
            # `if bd_is` catches the null-handle case; `.getSize() > 0` catches
            # the valid-but-empty case that None / handle checks miss (issue #291).
            keep = set()
            if bd_is and bd_is.getSize() > 0:
                for bp in bd_is.getIndices().tolist():
                    keep.update(dm.getTransitiveClosure(bp)[0].tolist())
            iset = interior_pts.setdefault(fid_h, set())
            for p in range(cS, cE):
                if Sold.getFieldDof(p, fid_h) > 0 and p not in keep:
                    iset.add(p)

        # Nothing to constrain (e.g. boundary-only multiplier discretisation).
        if not any(interior_pts.values()):
            return

        # 2. Build a new local section mirroring the old (chart, fields, dofs and
        #    ALL existing constraints) plus the new interior-h constraints.
        Snew = PETSc.Section().create(comm=dm.getComm())
        Snew.setNumFields(nF)
        Snew.setChart(cS, cE)
        for f in range(nF):
            Snew.setFieldComponents(f, Sold.getFieldComponents(f))
            Snew.setFieldName(f, Sold.getFieldName(f))

        # 2a. DOFs and constraint DOF COUNTS — must precede setUp().
        #     extra_field[(p, f)] = field-local index list of the ADDED constraints
        extra_field = {}
        for p in range(cS, cE):
            Snew.setDof(p, Sold.getDof(p))
            addl_total = 0
            for f in range(nF):
                fdof = Sold.getFieldDof(p, f)
                Snew.setFieldDof(p, f, fdof)
                fc = Sold.getFieldConstraintDof(p, f)
                add_here = 0
                if p in interior_pts.get(f, ()):
                    # Constrain ALL (currently unconstrained) DOFs of this field
                    # at this point — for a scalar h that is the single dof 0.
                    old_ind = Sold.getFieldConstraintIndices(p, f)
                    old_set = set(old_ind.tolist()) if old_ind is not None else set()
                    new_idx = [i for i in range(fdof) if i not in old_set]
                    if new_idx:
                        extra_field[(p, f)] = new_idx
                        add_here = len(new_idx)
                Snew.setFieldConstraintDof(p, f, fc + add_here)
                addl_total += add_here
            Snew.setConstraintDof(p, Sold.getConstraintDof(p) + addl_total)

        Snew.setUp()

        # 2b. Constraint INDICES — must follow setUp(). Rebuild the point-local
        #     aggregate from the field views so the cross-field offset convention
        #     stays internally consistent (offset of field f = sum of lower fdof).
        for p in range(cS, cE):
            point_local = []
            field_offset = 0
            for f in range(nF):
                fdof = Sold.getFieldDof(p, f)
                old_ind = Sold.getFieldConstraintIndices(p, f)
                fidx = set(old_ind.tolist()) if old_ind is not None else set()
                if (p, f) in extra_field:
                    fidx |= set(extra_field[(p, f)])
                if fidx:
                    fidx = sorted(fidx)
                    Snew.setFieldConstraintIndices(p, f, np.array(fidx, dtype=np.int32))
                    point_local.extend(field_offset + i for i in fidx)
                field_offset += fdof
            if point_local:
                Snew.setConstraintIndices(p, np.array(sorted(point_local), dtype=np.int32))

        dm.setLocalSection(Snew)
        # Force the global-section rebuild (and fail fast if malformed); the
        # constrained interior-h DOFs are now excluded from the global system.
        dm.getGlobalSection()

        return


    @timing.routine_timer_decorator
    def _setup_solver(self, verbose=False, _rewire_only=False):

        if self.is_setup == True:
            if verbose and uw.mpi.rank == 0:
                print(f"Stokes Saddle Pt ({self.name}): SNES solver does not need to be rebuilt", flush=True)
            return

        # set functions
        cdef int ind=1
        cdef int [::1] comps_view  # for numpy memory view
        cdef DM cdm = self.dm
        cdef DS ds =  self.dm.getDS()
        cdef PtrContainer ext = self.compiled_extensions

        i_res = self.ext_dict.res

        PetscDSSetResidual(ds.ds, 0, ext.fns_residual[i_res[self._u_F0]], ext.fns_residual[i_res[self._u_F1]])
        PetscDSSetResidual(ds.ds, 1, ext.fns_residual[i_res[self._p_F0]],                          NULL)

        i_jac = self.ext_dict.jac

        PetscDSSetJacobian(              ds.ds, 0, 0, ext.fns_jacobian[i_jac[self._uu_G0]], ext.fns_jacobian[i_jac[self._uu_G1]], ext.fns_jacobian[i_jac[self._uu_G2]], ext.fns_jacobian[i_jac[self._uu_G3]])
        PetscDSSetJacobian(              ds.ds, 0, 1, ext.fns_jacobian[i_jac[self._up_G0]], ext.fns_jacobian[i_jac[self._up_G1]], ext.fns_jacobian[i_jac[self._up_G2]], ext.fns_jacobian[i_jac[self._up_G3]])
        PetscDSSetJacobian(              ds.ds, 1, 0, ext.fns_jacobian[i_jac[self._pu_G0]], ext.fns_jacobian[i_jac[self._pu_G1]],                                 NULL,                                 NULL)
        PetscDSSetJacobianPreconditioner(ds.ds, 0, 0, ext.fns_jacobian[i_jac[self._uu_G0]], ext.fns_jacobian[i_jac[self._uu_G1]], ext.fns_jacobian[i_jac[self._uu_G2]], ext.fns_jacobian[i_jac[self._uu_G3]])
        PetscDSSetJacobianPreconditioner(ds.ds, 0, 1, ext.fns_jacobian[i_jac[self._up_G0]], ext.fns_jacobian[i_jac[self._up_G1]], ext.fns_jacobian[i_jac[self._up_G2]], ext.fns_jacobian[i_jac[self._up_G3]])
        PetscDSSetJacobianPreconditioner(ds.ds, 1, 0, ext.fns_jacobian[i_jac[self._pu_G0]], ext.fns_jacobian[i_jac[self._pu_G1]],                                 NULL,                                 NULL)
        PetscDSSetJacobianPreconditioner(ds.ds, 1, 1, ext.fns_jacobian[i_jac[self._pp_G0]],                                 NULL,                                 NULL,                                 NULL)

        # Lagrange-multiplier rows (block-constrained Stokes). Guarded: no-op
        # for ordinary Stokes. Register the interior screening residual and the
        # diagonal mass Jacobian/preconditioner on each multiplier's field.
        for k, mvar in enumerate(self._multipliers):
            fid = mvar._solver_field_id
            # Operator (Amat) (lambda,lambda) block is ALWAYS the true screening eps so
            # Newton stays exact; only the preconditioner (Pmat) block swaps to the 1/mu
            # Schur mass when _multiplier_schur_pc is on (pure-Pmat, can't corrupt Newton —
            # the exact analog of pressure's _pp_G0 living only in JacobianPreconditioner).
            # Reuse pressure's already-compiled _pp_G0 (= 1/mu, same scaling) so there is
            # NO extra JIT term (the 1/mu Piecewise of SolCx is otherwise compiled twice).
            hh_pc = self._pp_G0 if self._multiplier_schur_pc else self._hh_G0[k]
            PetscDSSetResidual(ds.ds, fid, ext.fns_residual[i_res[self._h_F0[k]]], NULL)
            PetscDSSetJacobian(              ds.ds, fid, fid, ext.fns_jacobian[i_jac[self._hh_G0[k]]], NULL, NULL, NULL)
            PetscDSSetJacobianPreconditioner(ds.ds, fid, fid, ext.fns_jacobian[i_jac[hh_pc]], NULL, NULL, NULL)

        cdef DMLabel c_label

        for bc in self.natural_bcs:

            boundary = bc.boundary
            boundary_id = bc.PETScID

            value = self.mesh.boundaries[bc.boundary].value
            bc_label = self.dm.getLabel("UW_Boundaries")

            label_val = value

            i_bd_res = self.ext_dict.bd_res
            i_bd_jac = self.ext_dict.bd_jac

            c_label = bc_label


            if bc.fn_f is not None:

                _has_f1 = "u_F1" in bc.fns

                # Velocity boundary residual: f0 (value) + f1 (gradient, if present)
                if _has_f1:
                    UW_PetscDSSetBdResidual(ds.ds, c_label.dmlabel, label_val, boundary_id,
                                    0, 0,
                                    ext.fns_bd_residual[i_bd_res[bc.fns["u_f0"]]],
                                    ext.fns_bd_residual[i_bd_res[bc.fns["u_F1"]]],
                                    )
                else:
                    UW_PetscDSSetBdResidual(ds.ds, c_label.dmlabel, label_val, boundary_id,
                                    0, 0,
                                    ext.fns_bd_residual[i_bd_res[bc.fns["u_f0"]]],
                                    NULL,
                                    )

                # Pressure boundary residual (Nitsche: f0_p = u.n - g)
                if "p_f0" in bc.fns:
                    UW_PetscDSSetBdResidual(ds.ds, c_label.dmlabel, label_val, boundary_id,
                                    1, 0,
                                    ext.fns_bd_residual[i_bd_res[bc.fns["p_f0"]]],
                                    NULL)
                else:
                    UW_PetscDSSetBdResidual(ds.ds, c_label.dmlabel, label_val, boundary_id, 1, 0, NULL, NULL)

                # Velocity-velocity boundary Jacobian: g0, g1 always; g2, g3 if f1_bd present
                if _has_f1:
                    UW_PetscDSSetBdJacobian(ds.ds, c_label.dmlabel, label_val, boundary_id,
                                    0, 0, 0,
                                    ext.fns_bd_jacobian[i_bd_jac[bc.fns["uu_G0"]]],
                                    ext.fns_bd_jacobian[i_bd_jac[bc.fns["uu_G1"]]],
                                    ext.fns_bd_jacobian[i_bd_jac[bc.fns["uu_G2"]]],
                                    ext.fns_bd_jacobian[i_bd_jac[bc.fns["uu_G3"]]],
                                    )
                else:
                    UW_PetscDSSetBdJacobian(ds.ds, c_label.dmlabel, label_val, boundary_id,
                                    0, 0, 0,
                                    ext.fns_bd_jacobian[i_bd_jac[bc.fns["uu_G0"]]],
                                    ext.fns_bd_jacobian[i_bd_jac[bc.fns["uu_G1"]]],
                                    NULL, NULL,
                                    )

                # Velocity-pressure boundary Jacobian
                if _has_f1:
                    UW_PetscDSSetBdJacobian(ds.ds, c_label.dmlabel, label_val, boundary_id,
                                    0, 1, 0,
                                    ext.fns_bd_jacobian[i_bd_jac[bc.fns["up_G0"]]],
                                    ext.fns_bd_jacobian[i_bd_jac[bc.fns["up_G1"]]],
                                    ext.fns_bd_jacobian[i_bd_jac[bc.fns["up_G2"]]],
                                    ext.fns_bd_jacobian[i_bd_jac[bc.fns["up_G3"]]],
                                    )
                else:
                    UW_PetscDSSetBdJacobian(ds.ds, c_label.dmlabel, label_val, boundary_id,
                                    0, 1, 0,
                                    ext.fns_bd_jacobian[i_bd_jac[bc.fns["up_G0"]]],
                                    ext.fns_bd_jacobian[i_bd_jac[bc.fns["up_G1"]]],
                                    NULL, NULL)

                # Pressure-velocity boundary Jacobian (pu block)
                UW_PetscDSSetBdJacobian(ds.ds, c_label.dmlabel, label_val, boundary_id,
                                1, 0, 0,
                                ext.fns_bd_jacobian[i_bd_jac[bc.fns["pu_G0"]]],
                                ext.fns_bd_jacobian[i_bd_jac[bc.fns["pu_G1"]]],
                                NULL, NULL)

                # Pressure-pressure boundary Jacobian (pp block)
                UW_PetscDSSetBdJacobian(ds.ds, c_label.dmlabel, label_val, boundary_id,
                                1, 1, 0,
                                ext.fns_bd_jacobian[i_bd_jac[bc.fns["pp_G0"]]],
                                NULL, NULL, NULL)

                # Preconditioner: mirror the Jacobian structure
                if _has_f1:
                    UW_PetscDSSetBdJacobianPreconditioner(ds.ds, c_label.dmlabel, label_val, boundary_id,
                                    0, 0, 0,
                                    ext.fns_bd_jacobian[i_bd_jac[bc.fns["uu_G0"]]],
                                    ext.fns_bd_jacobian[i_bd_jac[bc.fns["uu_G1"]]],
                                    ext.fns_bd_jacobian[i_bd_jac[bc.fns["uu_G2"]]],
                                    ext.fns_bd_jacobian[i_bd_jac[bc.fns["uu_G3"]]],
                                    )
                else:
                    UW_PetscDSSetBdJacobianPreconditioner(ds.ds, c_label.dmlabel, label_val, boundary_id,
                                    0, 0, 0,
                                    ext.fns_bd_jacobian[i_bd_jac[bc.fns["uu_G0"]]],
                                    ext.fns_bd_jacobian[i_bd_jac[bc.fns["uu_G1"]]],
                                    NULL, NULL,
                                    )

                if _has_f1:
                    UW_PetscDSSetBdJacobianPreconditioner(ds.ds, c_label.dmlabel, label_val, boundary_id,
                                    0, 1, 0,
                                    ext.fns_bd_jacobian[i_bd_jac[bc.fns["up_G0"]]],
                                    ext.fns_bd_jacobian[i_bd_jac[bc.fns["up_G1"]]],
                                    ext.fns_bd_jacobian[i_bd_jac[bc.fns["up_G2"]]],
                                    ext.fns_bd_jacobian[i_bd_jac[bc.fns["up_G3"]]],
                                    )
                else:
                    UW_PetscDSSetBdJacobianPreconditioner(ds.ds, c_label.dmlabel, label_val, boundary_id,
                                    0, 1, 0,
                                    ext.fns_bd_jacobian[i_bd_jac[bc.fns["up_G0"]]],
                                    ext.fns_bd_jacobian[i_bd_jac[bc.fns["up_G1"]]],
                                    NULL, NULL)

                UW_PetscDSSetBdJacobianPreconditioner(ds.ds, c_label.dmlabel, label_val, boundary_id,
                                1, 0, 0,
                                ext.fns_bd_jacobian[i_bd_jac[bc.fns["pu_G0"]]],
                                ext.fns_bd_jacobian[i_bd_jac[bc.fns["pu_G1"]]],
                                NULL, NULL)

                UW_PetscDSSetBdJacobianPreconditioner(ds.ds, c_label.dmlabel, label_val, boundary_id,
                                1, 1, 0,
                                ext.fns_bd_jacobian[i_bd_jac[bc.fns["pp_G0"]]],
                                NULL, NULL, NULL)

        # Lagrange-multiplier boundary coupling DS registration (block-constrained
        # Stokes). Attaches the field-0 traction, field-h constraint, and the
        # off-diagonal uh/hu boundary Jacobians to each constraint's boundary
        # region. Guarded: no-op for ordinary Stokes.
        if self._block_constraint_bcs:
            i_bd_res = self.ext_dict.bd_res
            i_bd_jac = self.ext_dict.bd_jac
            c_label = self.dm.getLabel("UW_Boundaries")
            for cbc in self._block_constraint_bcs:
                label_val = cbc.label_val
                fid_h = cbc.lam._solver_field_id
                bid_u = cbc.petsc_id_u   # boundary entry registered for field 0
                bid_h = cbc.petsc_id_h   # boundary entry registered for field h

                # --- field-0 entry: u-row residual + uu (AL) + uh Jacobians ---
                # traction + AL penalty residual  fn_f = h n + r(n·u−g) n
                UW_PetscDSSetBdResidual(ds.ds, c_label.dmlabel, label_val, bid_u,
                                0, 0,
                                ext.fns_bd_residual[i_bd_res[cbc.fns["u_f0"]]],
                                NULL)
                # uu (0, 0):  ∂fn_f/∂u = r·(n⊗n)  — augmented-Lagrangian stiffness
                UW_PetscDSSetBdJacobian(ds.ds, c_label.dmlabel, label_val, bid_u,
                                0, 0, 0,
                                ext.fns_bd_jacobian[i_bd_jac[cbc.fns["uu_G0"]]],
                                NULL, NULL, NULL)
                UW_PetscDSSetBdJacobianPreconditioner(ds.ds, c_label.dmlabel, label_val, bid_u,
                                0, 0, 0,
                                ext.fns_bd_jacobian[i_bd_jac[cbc.fns["uu_G0"]]],
                                NULL, NULL, NULL)
                # uh (0, h):  ∂fn_f/∂h = n
                UW_PetscDSSetBdJacobian(ds.ds, c_label.dmlabel, label_val, bid_u,
                                0, fid_h, 0,
                                ext.fns_bd_jacobian[i_bd_jac[cbc.fns["uh_G0"]]],
                                NULL, NULL, NULL)
                UW_PetscDSSetBdJacobianPreconditioner(ds.ds, c_label.dmlabel, label_val, bid_u,
                                0, fid_h, 0,
                                ext.fns_bd_jacobian[i_bd_jac[cbc.fns["uh_G0"]]],
                                NULL, NULL, NULL)

                # --- field-h entry: h-row constraint + hu Jacobian ---
                # constraint residual  fn_h = n·u − g
                UW_PetscDSSetBdResidual(ds.ds, c_label.dmlabel, label_val, bid_h,
                                fid_h, 0,
                                ext.fns_bd_residual[i_bd_res[cbc.fns["h_f0"]]],
                                NULL)
                # hu (h, 0):  ∂fn_h/∂u = n
                UW_PetscDSSetBdJacobian(ds.ds, c_label.dmlabel, label_val, bid_h,
                                fid_h, 0, 0,
                                ext.fns_bd_jacobian[i_bd_jac[cbc.fns["hu_G0"]]],
                                NULL, NULL, NULL)
                UW_PetscDSSetBdJacobianPreconditioner(ds.ds, c_label.dmlabel, label_val, bid_h,
                                fid_h, 0, 0,
                                ext.fns_bd_jacobian[i_bd_jac[cbc.fns["hu_G0"]]],
                                NULL, NULL, NULL)

        if verbose:
            print(f"Weak form (DS)", flush=True)
            UW_PetscDSViewWF(ds.ds)
            print(f"=============", flush=True)

            print(f"Weak form(s) (Natural Boundaries)", flush=True)
            for boundary in self.natural_bcs:
                UW_PetscDSViewBdWF(ds.ds, boundary.PETScID)


        # self.dm.setUp()
        # self.dm.ds.setUp()

        # Set constants on DS before copying to coarse levels
        self._set_constants_on_ds(ds)

        # Propagate updated DS to coarse DMs (both paths need this — DS
        # holds the pointwise function pointers we just changed).
        for coarse_dm in self.dm_hierarchy:
            if not _rewire_only:
                self.dm.copyFields(coarse_dm)
            self.dm.copyDS(coarse_dm)

        if not _rewire_only:
            for coarse_dm in self.dm_hierarchy:
                coarse_dm.createClosureIndex(None)

            # Boundary-only multiplier reduction: constrain the interior h DOFs
            # directly in the (now finalised) fine local section. Lossless and
            # refined-safe; must precede the fieldsplit grouping / SNES / field
            # decomposition so they see the shrunken [p,h] global block.
            self._constrain_interior_multipliers_in_section()

            # Block-constrained: group [pressure, multipliers] into a single
            # Schur factor by DM FIELD INDEX, so the velocity block keeps its DM
            # hierarchy for geometric MG/FMG. Must precede setFromOptions.
            if self._block_constraint_bcs:
                self._setup_block_fieldsplit_options()

            self.snes = PETSc.SNES().create(PETSc.COMM_WORLD)
            self.snes.setDM(self.dm)
            self.snes.setOptionsPrefix(self.petsc_options_prefix)
            self.snes.setFromOptions()

            UW_DMPlexSetSNESLocalFEM(cdm.dm, PETSC_FALSE, NULL)

            # Setup subdms here too.
            # These will be used to copy back/forth SNES solutions
            # into user facing variables.

            names, isets, dms = self.dm.createFieldDecomposition()
            self._subdict = {}
            for index,name in enumerate(names):
                self._subdict[name] = (isets[index],dms[index])

            self._attach_stokes_nullspace()

        # Apply region DS if active_region was set
        if hasattr(self, '_inactive_region_label') and self._inactive_region_label is not None:
            self._setup_region_ds()

        self.is_setup = True
        self.constitutive_model._solver_is_setup = True

    def set_active_region(self, region_label_name, region_label_value):
        """Configure the solver to assemble only on cells in the given region.

        Cells NOT in the specified region get a trivial DS (no volume
        contributions) and their DOFs should be pinned via Dirichlet BCs.

        Parameters
        ----------
        region_label_name : str
            DM label name for the INACTIVE region (e.g., "Outer").
        region_label_value : int
            Label stratum value for the inactive region (e.g., 102).
        """
        self._inactive_region_label = region_label_name
        self._inactive_region_value = region_label_value
        self.is_setup = False

    def _setup_region_ds(self):
        """Register a trivial DS for the inactive region.

        After _setup_solver populates the default DS with Stokes weak forms,
        this creates an empty DS for cells in the inactive region. PETSc's
        DMGetCellDS dispatches per-cell: inactive cells get the empty DS
        (zero volume contributions), active cells get the default DS.
        """
        cdef DM c_dm = self.dm
        cdef DS ds_default = self.dm.getDS()
        cdef DMLabel c_label

        # Get the inactive region label
        label_name = self._inactive_region_label
        bc_label = self.dm.getLabel(label_name)
        if bc_label is None:
            raise ValueError(f"DM label '{label_name}' not found")
        c_label = bc_label

        # Create a new DS with the same fields but no weak forms
        cdef DS air_ds = PETSc.DS().create(comm=self.dm.comm)

        # Copy field discretisations from the DM's fields
        cdef PetscInt nfields
        nfields = self.dm.getNumFields()

        for f in range(nfields):
            fe, _ = self.dm.getField(f)
            air_ds.setDiscretisation(f, fe)

        # Set coordinate dimension to match the default DS
        CHKERRQ( PetscDSSetCoordinateDimension(air_ds.ds, self.mesh.dim) )

        # Register the empty DS for the inactive region
        CHKERRQ( DMSetRegionDS(c_dm.dm, c_label.dmlabel, NULL, air_ds.ds, NULL) )

        # Copy to coarse levels too
        for coarse_dm in self.dm_hierarchy:
            self.dm.copyDS(coarse_dm)

        if uw.mpi.rank == 0 and self.verbose:
            print(f"Region DS: inactive region '{label_name}' gets trivial DS", flush=True)

    def compute_volume_residual_fields(
        self,
        time=None,
        verbose=False,
        cell_indices=None,
        residual_field_id=None,
        include_boundary_terms=False,
    ):
        """Return PETSc FEM residual fields in each solver field's local layout.

        This is a low-level diagnostic hook for post-processing derived
        boundary quantities such as consistent-boundary-flux traction. By
        default it calls PETSc's ``DMPlexSNESComputeResidualFEM`` directly. If
        ``cell_indices`` is supplied, it instead calls
        a UW wrapper around ``DMPlexComputeResidualByKey`` on a cloned DM with
        a copied ``PetscDS`` that has no registered boundary objects, so the
        selected-cell path returns volume terms only. Set
        ``include_boundary_terms=True`` to call PETSc's original keyed
        residual behavior, which appends registered boundary residuals. The
        returned arrays are local to each rank and have the same flat layout as
        the corresponding MeshVariable PETSc vector.
        """
        cdef DM _time_dm_residual
        cdef DM dm
        cdef Vec xvec
        cdef Vec fvec
        cdef PetscFormKey key
        cdef IS ccell_is
        cdef PetscReal residual_time = 0.0
        cdef PetscReal implicit_form_time = <PetscReal>-1.7976931348623157e308

        self._build(verbose, False, None)

        if time is not None:
            t_nd = self._nondimensional_time(time)
            _time_dm_residual = self.dm
            UW_DMSetTime(_time_dm_residual.dm, t_nd)
            residual_time = <PetscReal>t_nd

        self.mesh.update_lvec()
        self.dm.setAuxiliaryVec(self.mesh.lvec, None)
        self._update_constants()

        gvec = self.dm.getGlobalVec()
        xlocal = self.dm.getLocalVec()
        flocal = self.dm.getLocalVec()
        gvec.setArray(0.0)
        xlocal.setArray(0.0)
        flocal.setArray(0.0)

        try:
            for name, var in self.fields.items():
                sgvec = gvec.getSubVector(self._subdict[name][0])
                subdm = self._subdict[name][1]
                subdm.localToGlobal(var.vec, sgvec)
                gvec.restoreSubVector(self._subdict[name][0], sgvec)

            self.dm.globalToLocal(gvec, xlocal)

            dm = self.dm
            xvec = xlocal
            fvec = flocal
            # Constrained (Dirichlet) DOFs are ABSENT from the global vector, so the
            # round trip above leaves them ZERO in xlocal: insert the essential
            # values before integrating, exactly as _assemble_volume_reaction does,
            # or the residual is garbage wherever g != 0 (issues #407/#411 — this
            # duplicated variant missed the #407 fix).
            CHKERRQ(DMPlexInsertBoundaryValues(dm.dm, PETSC_TRUE, xvec.vec,
                                               residual_time, NULL, NULL, NULL))
            if cell_indices is None:
                CHKERRQ(DMPlexSNESComputeResidualFEM(dm.dm, xvec.vec, fvec.vec, NULL))
            else:
                if residual_field_id is None:
                    residual_field_id = 0
                cell_is = PETSc.IS().createGeneral(
                    list(cell_indices), comm=PETSc.COMM_SELF
                )
                try:
                    ccell_is = cell_is
                    key.label = NULL
                    key.value = 0
                    key.field = <PetscInt>residual_field_id
                    key.part = 0
                    if include_boundary_terms:
                        CHKERRQ(DMPlexComputeResidualByKey(
                            dm.dm, key, ccell_is.iset, implicit_form_time,
                            xvec.vec, NULL, residual_time, fvec.vec, NULL,
                        ))
                    else:
                        CHKERRQ(UW_DMPlexComputeResidualByKeyVolumeOnly(
                            dm.dm, key, ccell_is.iset, implicit_form_time,
                            xvec.vec, NULL, residual_time, fvec.vec, NULL,
                        ))
                finally:
                    cell_is.destroy()

            local_section = self.dm.getLocalSection()
            pStart, pEnd = local_section.getChart()
            out = {}

            for name, var in self.fields.items():
                field_id = getattr(var, "_solver_field_id", None)
                if field_id is None:
                    field_id = getattr(var, "field_id", None)
                if field_id is None:
                    continue

                is_field = None
                created_is_field = False
                if name == "velocity" and getattr(self, "_velocity_is", None) is not None:
                    is_field = self._velocity_is
                elif name == "pressure" and getattr(self, "_pressure_is", None) is not None:
                    is_field = self._pressure_is
                elif getattr(self, "_multiplier_is", None) is not None and name in self._multiplier_is:
                    is_field = self._multiplier_is[name]
                else:
                    indices = []
                    for point in range(pStart, pEnd):
                        dof = local_section.getFieldDof(point, field_id)
                        if dof > 0:
                            offset = local_section.getFieldOffset(point, field_id)
                            for i in range(dof):
                                indices.append(offset + i)

                    is_field = PETSc.IS().createGeneral(indices, comm=PETSc.COMM_SELF)
                    created_is_field = True

                try:
                    subvec = flocal.getSubVector(is_field)
                    try:
                        out[name] = np.array(subvec.array, copy=True)
                    finally:
                        flocal.restoreSubVector(is_field, subvec)
                finally:
                    if created_is_field:
                        is_field.destroy()

            return out
        finally:
            self.dm.restoreLocalVec(flocal)
            self.dm.restoreLocalVec(xlocal)
            self.dm.restoreGlobalVec(gvec)

    def compute_boundary_residual_fields(self, boundary, time=None, verbose=False, residual_field_id=0):
        """Return the registered FEM boundary residual for one named boundary.

        This is a low-level diagnostic hook for weak-boundary-condition
        debugging. It assembles PETSc's boundary residual terms registered on
        ``boundary`` through ``DMPlexComputeBdResidualSingle``. For Nitsche
        free slip, this includes the full registered weak boundary residual,
        not only the scalar penalty term. The returned arrays are local to each
        rank and have the same flat layout as the corresponding MeshVariable
        PETSc vector.
        """
        cdef DM _time_dm_boundary_residual
        cdef DM dm
        cdef Vec xvec
        cdef Vec fvec
        cdef PetscFormKey key
        cdef PetscDS ds
        cdef PetscWeakForm wf
        cdef DMLabel c_label
        cdef PetscReal residual_time = 0.0

        self._build(verbose, False, None)

        boundary_bc = None
        for bc in self.natural_bcs:
            if bc.boundary == boundary and bc.f_id == residual_field_id:
                boundary_bc = bc
                break
        if boundary_bc is None:
            raise ValueError(
                f"No natural/Nitsche boundary residual is registered for "
                f"boundary '{boundary}' and field {residual_field_id}."
            )

        if time is not None:
            t_nd = self._nondimensional_time(time)
            _time_dm_boundary_residual = self.dm
            UW_DMSetTime(_time_dm_boundary_residual.dm, t_nd)
            residual_time = <PetscReal>t_nd

        self.mesh.update_lvec()
        self.dm.setAuxiliaryVec(self.mesh.lvec, None)
        self._update_constants()

        gvec = self.dm.getGlobalVec()
        xlocal = self.dm.getLocalVec()
        flocal = self.dm.getLocalVec()
        gvec.setArray(0.0)
        xlocal.setArray(0.0)
        flocal.setArray(0.0)

        try:
            for name, var in self.fields.items():
                sgvec = gvec.getSubVector(self._subdict[name][0])
                subdm = self._subdict[name][1]
                subdm.localToGlobal(var.vec, sgvec)
                gvec.restoreSubVector(self._subdict[name][0], sgvec)

            self.dm.globalToLocal(gvec, xlocal)

            dm = self.dm
            xvec = xlocal
            fvec = flocal
            # Same insert as _assemble_volume_reaction: constrained DOFs are absent
            # from the global vector, so without this the boundary residual is
            # evaluated against zeroed essential values wherever g != 0
            # (issues #407/#411 — this duplicated variant missed the #407 fix).
            CHKERRQ(DMPlexInsertBoundaryValues(dm.dm, PETSC_TRUE, xvec.vec,
                                               residual_time, NULL, NULL, NULL))
            CHKERRQ(DMGetDS(dm.dm, &ds))
            CHKERRQ(UW_PetscDSGetBoundaryWeakForm(
                ds, <PetscInt>boundary_bc.PETScID, &wf,
            ))

            c_label = self.dm.getLabel("UW_Boundaries")
            key.label = c_label.dmlabel
            key.value = <PetscInt>boundary_bc.boundary_label_val
            key.field = <PetscInt>residual_field_id
            key.part = 0
            CHKERRQ(DMPlexComputeBdResidualSingle(
                dm.dm, wf, key, xvec.vec, NULL, 0.0, fvec.vec,
            ))

            local_section = self.dm.getLocalSection()
            pStart, pEnd = local_section.getChart()
            out = {}

            for name, var in self.fields.items():
                field_id = getattr(var, "_solver_field_id", None)
                if field_id is None:
                    field_id = getattr(var, "field_id", None)
                if field_id is None:
                    continue

                is_field = None
                created_is_field = False
                if name == "velocity" and getattr(self, "_velocity_is", None) is not None:
                    is_field = self._velocity_is
                elif name == "pressure" and getattr(self, "_pressure_is", None) is not None:
                    is_field = self._pressure_is
                elif getattr(self, "_multiplier_is", None) is not None and name in self._multiplier_is:
                    is_field = self._multiplier_is[name]
                else:
                    indices = []
                    for point in range(pStart, pEnd):
                        dof = local_section.getFieldDof(point, field_id)
                        if dof > 0:
                            offset = local_section.getFieldOffset(point, field_id)
                            for i in range(dof):
                                indices.append(offset + i)

                    is_field = PETSc.IS().createGeneral(indices, comm=PETSc.COMM_SELF)
                    created_is_field = True

                try:
                    subvec = flocal.getSubVector(is_field)
                    try:
                        out[name] = np.array(subvec.array, copy=True)
                    finally:
                        flocal.restoreSubVector(is_field, subvec)
                finally:
                    if created_is_field:
                        is_field.destroy()

            return out
        finally:
            self.dm.restoreLocalVec(flocal)
            self.dm.restoreLocalVec(xlocal)
            self.dm.restoreGlobalVec(gvec)

    def _ensure_local_field_index_sets(self, clvec, local_section):
        """Build (once) and cache the LOCAL index sets that decompose a parent-DM
        local vector into the per-field MeshVariable storage: velocity, pressure
        and any block-constraint multipliers.

        The index sets are state-independent given the section, so they are built
        on first use and reused until a rebuild resets ``_pressure_is`` to None
        (see ``_build``). Used both by the post-solve copy-back and the
        mid-solve callback scatter (``_scatter_global_to_fields``).
        """
        if getattr(self, "_pressure_is", None) is not None:
            return

        # Function to get index set for a field
        def get_local_field_is(section, field, unconstrained=False):
            """
            This function returns the index set of unconstrained points if True, or all points if False.
            """
            pStart, pEnd = section.getChart()
            indices = []
            for p in range(pStart, pEnd):
                dof = section.getFieldDof(p, field)
                if dof > 0:
                    offset = section.getFieldOffset(p, field)
                    if not unconstrained and self.Unknowns.p.continuous:
                        indices.append(offset)
                    else:
                        cind = section.getFieldConstraintIndices(p, field)
                        constrained = set(cind) if cind is not None else set()
                        for i in range(dof):
                            if i not in constrained:
                                index = offset + i
                                indices.append(index)
            is_field = PETSc.IS().createGeneral(indices, comm=PETSc.COMM_SELF)
            return is_field

        # Field numbers (adjust based on your setup)
        velocity_field_num = 0
        pressure_field_num = 1

        self._pressure_is = get_local_field_is(local_section, pressure_field_num)

        # Multiplier fields (block-constrained Stokes). Build a local IS per
        # multiplier so its DOFs can be (a) excluded from the velocity
        # complement and (b) copied back into the multiplier MeshVariable.
        # Guarded: empty dict for ordinary Stokes.
        self._multiplier_is = {}
        multiplier_indices = set()
        for mvar in self._multipliers:
            # unconstrained=False -> include ALL multiplier DOFs (the pinned
            # interior ones are present in the LOCAL vec at 0), so the copy
            # back into the full-domain MeshVariable is size-correct.
            mis = get_local_field_is(local_section, mvar._solver_field_id, unconstrained=False)
            self._multiplier_is[mvar._solver_field_name] = mis
            multiplier_indices |= set(mis.getIndices())

        # Get indices for velocity (complement of pressure and multipliers)
        size = clvec.getLocalSize()
        all_indices = set(range(size))
        pressure_indices = set(self._pressure_is.getIndices())
        velocity_indices = sorted(list(all_indices - pressure_indices - multiplier_indices))
        self._velocity_is = PETSc.IS().createGeneral(velocity_indices, comm=PETSc.COMM_SELF)

    def _scatter_global_to_fields(self, gvec):
        """Scatter the global iterate into the field MeshVariables, correctly on
        driven (non-zero Dirichlet) boundaries.

        The base-class scatter (per-field ``subdm.globalToLocal``) fills a field's
        interior and *zero*-Dirichlet DOFs, but NOT its non-zero Dirichlet (driven)
        boundary DOFs: those are not stored in the global vector — they are imposed
        on the *local* vector by ``DMPlexSNESComputeBoundaryFEM``. A callback that
        reads, e.g., the velocity on a lid would otherwise see stale values there
        (measured ~35% error in the Helmholtz smoother test).

        This override mirrors the post-solve copy-back: ``globalToLocal`` into a
        parent-DM local vec, apply the boundary FEM, then split per-field via the
        cached local index sets. Cache-invalidation tail matches the base method.
        """
        cdef DM dm = self.dm
        cdef Vec clvec = self.dm.getLocalVec()
        self.dm.globalToLocal(gvec, clvec)
        ierr = DMPlexSNESComputeBoundaryFEM(dm.dm, <void*>clvec.vec, NULL); CHKERRQ(ierr)

        local_section = self.dm.getLocalSection()
        self._ensure_local_field_index_sets(clvec, local_section)

        for name, var in self.fields.items():
            if name == 'velocity':
                subvec = clvec.getSubVector(self._velocity_is)
                var.vec.array[:] = subvec.array[:]
                clvec.restoreSubVector(self._velocity_is, subvec)
            elif name == 'pressure':
                subvec = clvec.getSubVector(self._pressure_is)
                var.vec.array[:] = subvec.array[:]
                clvec.restoreSubVector(self._pressure_is, subvec)
            elif name in self._multiplier_is:
                mis = self._multiplier_is[name]
                subvec = clvec.getSubVector(mis)
                var.vec.array[:] = subvec.array[:]
                clvec.restoreSubVector(mis, subvec)

        self.dm.restoreLocalVec(clvec)

        # Cache invalidation (identical to the base-class scatter)
        self.mesh._stale_lvec = True
        for name, var in self.fields.items():
            base = getattr(var, "_base_var", var)
            if hasattr(base, "_sync_lvec_to_gvec"):
                base._sync_lvec_to_gvec()
            if hasattr(base, "_canonical_data"):
                base._canonical_data = None

    def _gather_fields_to_global(self, gvec):
        """Gather velocity / pressure / multiplier fields back into the global
        iterate via the per-field sub-DMs (the multi-field counterpart of the
        single-field base-class gather)."""
        for name, var in self.fields.items():
            if name not in self._subdict:
                continue
            gis, subdm = self._subdict[name]
            sgvec = gvec.getSubVector(gis)
            subdm.localToGlobal(var.vec, sgvec)
            gvec.restoreSubVector(gis, sgvec)

    @timing.routine_timer_decorator
    def solve(self,
              zero_init_guess: bool = None,
              picard: int = 0,
              verbose=False,
              debug=False,
              debug_name=None,
              _force_setup: bool =False,
              time=None,
              divergence_retries: int = 0,
              homotopy: bool = False,
              homotopy_options: dict = None, ):
        """
        Solve the Stokes system for velocity and pressure.

        Assembles and solves the coupled velocity-pressure system using a
        saddle-point formulation. Handles nonlinear rheologies through
        Newton or Picard iteration.

        Parameters
        ----------
        zero_init_guess : bool, optional
            Cold or warm start. The default (``None``) **auto-detects**: cold when the
            solver holds no converged solution, warm when it does (see
            :attr:`has_solution`). ``True`` forces a fresh start, discarding any
            existing solution; ``False`` insists on warming from the current field
            values. Warm-starting improves convergence for time-stepping and
            continuation; the auto default gets that without a flag, and cannot warm
            off stale data because a remesh or a diverged solve clears
            ``has_solution``.
        picard : int, default=0
            Number of Picard iterations before switching to Newton.
            Picard iterations use a simplified Jacobian and can help
            convergence for strongly nonlinear problems.
        verbose : bool, default=False
            Print solver progress and timing information.
        debug : bool, default=False
            Enable debug output including residual norms.
        debug_name : str, optional
            Name prefix for debug output files.
        _force_setup : bool, default=False
            Force rebuild of the solver even if already set up.
        time : float or Quantity, optional
            Physical time for this solve. Passed as ``petsc_t`` to all
            pointwise residual and Jacobian functions. Expressions using
            ``mesh.t`` will evaluate at this time. Non-dimensionalised
            automatically when scaling is active. Default: None (petsc_t=0).
        divergence_retries : int, default=0
            If the final SNES solve reports DIVERGED, re-call it with warm
            start up to this many times. A single retry rescues most VEP
            yield-surface kink divergences. 0 preserves legacy behaviour.
        homotopy : bool, default=False
            Solve a yielding (viscoplastic) model by a **yield homotopy** instead of
            a single solve: the constitutive model is put in its smooth,
            δ-parameterised yield mode and δ is marched down to the sharp yield
            surface as a sequence of warm-started solves. This is the robust route
            for a hard Drucker-Prager problem, which does not converge from a cold
            start on the sharp surface. Requires a model with
            ``supports_yield_homotopy`` (raises otherwise). Returns the march
            summary rather than ``None``.
        homotopy_options : dict, optional
            March settings passed to
            :func:`~underworld3.systems.yield_continuation.yield_continuation` —
            ``smoother``, ``delta0``, ``down``, ``dmin``, ``entry_maxit``,
            ``step_maxit``, ``retries``. All are defaulted; tuning them is optional.
            ``smoother`` picks the soft-min family — ``"powermean"`` (default,
            approaches the yield surface from below) or ``"sqrt"`` (from above). Which
            gives the better cold entry is problem-dependent, so it is worth trying both
            when a march will not start. Leave ``delta0`` unset unless you have a
            reason: each family supplies its own entry, and the two δ are not the same
            parameter.

        Returns
        -------
        None or dict
            ``None`` normally — the solution is stored in ``self.u`` (velocity) and
            ``self.p`` (pressure). With ``homotopy=True``, the march summary
            (``settled_delta``, ``reason``, ``steps``, ``reached_dmin``,
            ``converged``).

        Examples
        --------
        >>> # Basic Stokes solve
        >>> stokes.solve()
        >>> velocity = stokes.u.array[:, 0, :]
        >>> pressure = stokes.p.array[:, 0, 0]

        >>> # Nonlinear solve with Picard warmup
        >>> stokes.solve(picard=3)

        >>> # Time-stepping with previous solution
        >>> for step in range(n_steps):
        ...     # Update boundary conditions, material properties...
        ...     stokes.solve(zero_init_guess=False)

        >>> # Hard viscoplastic (Drucker-Prager) yield: march the yield homotopy
        >>> report = stokes.solve(homotopy=True)
        >>> report["settled_delta"]        # smallest yield softness reached

        Notes
        -----
        This is a **collective operation** - all MPI ranks must call it.

        For nonlinear viscosity (e.g., power-law, viscoplastic), the solver
        uses Newton iteration by default. The ``picard`` parameter can help
        with initial convergence by using simpler Jacobian approximations.

        See Also
        --------
        u : Velocity solution variable.
        p : Pressure solution variable.
        constitutive_model : Viscosity and stress definitions.
        """


        if homotopy:
            # The march runs a SEQUENCE of ordinary solves at successively sharper
            # yield surfaces; each one re-enters this method with homotopy=False.
            return self._solve_yield_homotopy(homotopy_options, verbose=verbose)

        if _force_setup:
            self.is_setup = False
        elif not self.constitutive_model._solver_is_setup:
            # Constitutive model swapped: pointwise functions change but the
            # DM/fields/BCs are unchanged. In-place rewire is sufficient.
            self._needs_function_rewire = True

        # Tri-state: None auto-detects cold-vs-warm from has_solution. Resolved HERE,
        # after _force_setup has had its say: that invalidation clears has_solution,
        # and resolving earlier would warm-start off the flag it just cleared.
        zero_init_guess = self._resolve_zero_init_guess(zero_init_guess)

        self._build(verbose, debug, debug_name)

        # Set time on the DM so petsc_t is available in pointwise functions.
        # Non-dimensionalise if the scaling system is active.
        cdef DM _time_dm_stokes

        # Rotated strong free-slip: delegate to the rotated_bc module (per-node DOF
        # rotation + strong v_n=0 + reaction=sigma_nn). Handles the whole assemble/
        # solve/rotate-back/gauge-removal; stashes info for boundary_normal_traction.
        if self._rotated_freeslip_bcs or self._fault_contact_faults:
            # Run the same pre-solve preamble as the standard path so the pointwise
            # functions see the DM time, the auxiliary vector, and updated constants
            # (needed for problems whose coefficients live in auxiliary fields). This
            # must precede the nonlinearity probe below so the trial assemblies see
            # the correct coefficients.
            if time is not None:
                t_nd = self._nondimensional_time(time)
                _time_dm_stokes = self.dm
                UW_DMSetTime(_time_dm_stokes.dm, t_nd)
            self.mesh.update_lvec()
            self.dm.setAuxiliaryVec(self.mesh.lvec, None)
            self._update_constants()

            # guard() refuses rotated free-slip, but the BC can be added AFTER arming.
            # Re-check here: this path never reaches the instrumentation, so an armed
            # guard would be silently inert -- the exact state guard() exists to refuse.
            if (getattr(self, "_instrumentation", None) is not None
                    and self._instrumentation.armed):
                raise NotImplementedError(
                    "a wall-clock guard is armed but this solve takes the rotated "
                    "free-slip path, which runs its own Krylov loop outside self.snes "
                    "and cannot be bounded by it. Call unguard() first."
                )
            # ONE path for linear and nonlinear models: the manual outer
            # Newton/Picard loop that rotates F(u), J(u) and the strong
            # v_n = u_n constraint every iteration. It honours zero_init_guess
            # (warm start), the Picard warmup count, and the consistent_jacobian
            # tangent (Picard / Newton / continuation); a linear model converges
            # after its first increment, so no up-front nonlinearity probe is
            # paid (the loop self-terminates; _residual_is_nonlinear survives as
            # a lazy guard inside the picard+pure-Newton corner).
            from underworld3.utilities.rotated_bc import solve_rotated_freeslip
            self._rotated_freeslip_info = solve_rotated_freeslip(
                self, self._rotated_freeslip_bcs, verbose=verbose,
                zero_init_guess=zero_init_guess, picard=picard)
            # This path solves via ksp.solve on the rotated operator (not self.snes),
            # so give it a report from the rotated result rather than leaving a stale one.
            _rotated_report = self._capture_rotated_report(self._rotated_freeslip_info)
            # The warm-start flag must follow the rotated solve's own verdict — the SNES
            # was never run, so the generic reader would latch a stale reason.
            self._record_convergence_status(converged=_rotated_report.converged)
            return

        if time is not None:
            t_nd = self._nondimensional_time(time)
            _time_dm_stokes = self.dm
            UW_DMSetTime(_time_dm_stokes.dm, t_nd)

        # Keep a record of these set-up parameters
        snes_type = self.snes.getType()
        # Resolved ONCE, before any of this solve's own pushes, so the resolver compares
        # against the previous solve's push rather than this one's. (Was a hardcoded 50
        # pushed unconditionally, which made `snes_max_it` unreachable — worklist rows
        # D-22/D2, ruling D18; same failure shape as the sub-block rtols in #477.)
        snes_max_it = self._resolve_snes_max_it(50)

        self.mesh.update_lvec()
        self.dm.setAuxiliaryVec(self.mesh.lvec, None)

        # Update constants (e.g. changed material params) before solve
        self._update_constants()

        gvec = self.dm.getGlobalVec()
        gvec.setArray(0.0)

        if not zero_init_guess:

            if verbose and uw.mpi.rank == 0:
                print(f"SNES pre-solve - non-zero initial guess", flush=True)


            self._push_snes_max_it(0)
            self.snes.setType("nrichardson")
            self.snes.setFromOptions()
            # PETSc may rebuild operator state after setFromOptions(), so reattach
            # the configured Stokes nullspace before each solve path.
            self._attach_stokes_nullspace()
            self.snes.solve(None, gvec)

            for name,var in self.fields.items():
                sgvec = gvec.getSubVector(self._subdict[name][0])  # Get global subvec off solution gvec.
                subdm   = self._subdict[name][1]                   # Get subdm corresponding to field
                subdm.localToGlobal(var.vec,sgvec)                 # Copy variable data into gvec
                gvec.restoreSubVector(self._subdict[name][0], sgvec)

            self.atol = self.snes.getFunctionNorm() * self.tolerance

        else:
            self.atol = 0.0

        # Automatic cold-start warm-up (Layer 1): a single Picard (frozen-
        # coefficient) step moves a cold guess into the Newton basin — it is
        # defect-correction iteration 1, contractive and cheap. The default
        # (frozen) tangent path is left bit-identical, and an explicit Picard count
        # is honoured as given.
        #
        # This warm-up is a CONVERGENCE aid (move a cold guess toward the Newton
        # basin), NOT the correctness mechanism for the zero-strain-rate state.
        # Two measured facts (issue #507) retired the old "make the NaN state
        # unreachable" framing: (1) one nrichardson sweep only propagates
        # boundary data a single element layer, so on a BOUNDARY-DRIVEN problem
        # the deep interior stays exactly zero after the warm-up (body-force
        # problems fill F(0) everywhere, which is why the yield campaigns never
        # saw it); (2) a rigidly-translating stuck region has edot = 0 at the
        # CONVERGED solution — the state is physics, not a start-up artifact.
        # Finiteness of the consistent tangent at edot = 0 is owned by the
        # half-integer-power guard in _jacobian_unwrap (the derivative's
        # removable-singularity limit, implemented). NOTE the "continuation"
        # tangent is NOT protected by its alpha = 0 phase (the blended kernel
        # still evaluates the Newton branch pointwise, and IEEE 0*NaN = NaN);
        # with the guard in place both tangents are finite everywhere. See
        # docs/developer/design/nonlinear-solver-homotopy-warmstart.md (Layer 1).
        if (picard == 0 and self.consistent_jacobian is True
                and (zero_init_guess or self._solution_is_trivially_zero())):
            picard = 1

        if verbose and uw.mpi.rank == 0:
            print(f"SNES solve - picard = {picard}", flush=True)

        # Picard solves if requested

        if picard != 0:
            self._push_snes_max_it(abs(picard))
            self._reassert_outer_tolerances()
            self.snes.atol = self.atol
            self.snes.setType("nrichardson")
            self.snes.setFromOptions()
            self._attach_stokes_nullspace()
            self.snes.solve(None, gvec)
            self._warn_on_divergence(phase="picard")

        # The standard Newton solve, run whether or not the optional Picard
        # warmup above was taken: restore the configured SNES type and
        # tolerances, then solve.
        self.snes.setType(snes_type)
        self._reassert_outer_tolerances()
        self.snes.atol = self.atol
        self._push_snes_max_it(snes_max_it)
        self.snes.setFromOptions()
        self._attach_stokes_nullspace()
        # Custom geometric-MG prolongation on the velocity block (if registered
        # via set_custom_fmg). Injected here — after setFromOptions/nullspace,
        # before the real solve — because the velocity sub-PC is only reachable
        # once the monolithic Jacobian is assembled (see custom_mg). Picks up
        # a solver-set (set_custom_fmg field_id=0) or mesh-owned (adapt child)
        # hierarchy on the velocity block.
        from underworld3.utilities.custom_mg import auto_inject_custom_mg
        auto_inject_custom_mg(self, field_id=0)
        self._snes_solve_with_retries(gvec, divergence_retries, verbose)

        # Project the rigid-body rotation gauge out of the converged solution.
        # The fieldsplit/Schur inner velocity solve leaves an unconstrained (and
        # partition-dependent) rigid rotation that the true residual is blind to;
        # removing this genuine velocity nullspace makes the tangential velocity
        # partition-independent to round-off without changing the residual.
        self._remove_velocity_rotation_gauge(gvec)

        # SNESSetUpdate fires only at the START of each iteration, so the final
        # converged iterate is otherwise un-hooked. Apply the callbacks once more
        # to it (e.g. so a pressure gauge pins the FINAL pressure, and an
        # auxiliary field is consistent with the converged solution).
        if self._snes_update_callbacks:
            self._dispatch_snes_update(self.snes, -1)

        cdef DM dm = self.dm
        cdef Vec clvec = self.dm.getLocalVec()
        self.dm.globalToLocal(gvec, clvec)
        ierr = DMPlexSNESComputeBoundaryFEM(dm.dm, <void*>clvec.vec, NULL); CHKERRQ(ierr)

        if verbose and uw.mpi.rank == 0:
                 print(f"SNES Compute Boundary FEM Successfull", flush=True)

        # get local section
        local_section = self.dm.getLocalSection()

        # Get index sets for solution decomposition (velocity, pressure, multipliers).
        # Built once and cached (shared with the mid-solve callback scatter).
        self._ensure_local_field_index_sets(clvec, local_section)

        # Copy solution back into pressure and velocity variables
        for name, var in self.fields.items():
            if name=='velocity':
                subvec = clvec.getSubVector(self._velocity_is)
                var.vec.array[:] = subvec.array[:]
                clvec.restoreSubVector(self._velocity_is, subvec)
            elif name=='pressure':
                subvec = clvec.getSubVector(self._pressure_is)
                var.vec.array[:] = subvec.array[:]
                clvec.restoreSubVector(self._pressure_is, subvec)
            elif name in self._multiplier_is:
                mis = self._multiplier_is[name]
                subvec = clvec.getSubVector(mis)
                var.vec.array[:] = subvec.array[:]
                clvec.restoreSubVector(mis, subvec)
        self.mesh._stale_lvec = True

        # Sync _gvec so downstream consumers (write, stats) see the result
        for name, var in self.fields.items():
            target_var = getattr(var, "_base_var", var)
            target_var._sync_lvec_to_gvec()
            if hasattr(target_var, "_canonical_data"):
                target_var._canonical_data = None

        # Clean up local vectors
        self.dm.restoreLocalVec(clvec)
        self.dm.restoreGlobalVec(gvec)

        self._warn_on_divergence()

        self._record_convergence_status()

        return

    @timing.routine_timer_decorator
    def estimate_dt(self):
        """
        Calculates an appropriate advective timestep for the given
        mesh and velocity configuration.
        """
        # we'll want to do this on an element by element basis
        # for more general mesh

        # first let's extract a max global velocity magnitude
        import math

        vel = self.u.data
        magvel_squared = vel[:, 0] ** 2 + vel[:, 1] ** 2
        if self.mesh.dim == 3:
            magvel_squared += vel[:, 2] ** 2

        max_magvel = math.sqrt(magvel_squared.max())

        from mpi4py import MPI

        comm = uw.mpi.comm
        max_magvel_glob = comm.allreduce(max_magvel, op=MPI.MAX)

        min_dx = self.mesh.get_min_radius()
        dt_nd = min_dx / max_magvel_glob

        # Apply unit-aware scaling when model has units
        from underworld3.systems.solvers import _apply_unit_aware_scaling
        return _apply_unit_aware_scaling(dt_nd, self.u, self.mesh)
