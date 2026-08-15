r"""The contract every Underworld3 analytic solution satisfies.

An analytic solution is a closed-form answer to a problem Underworld3 can also
solve numerically. It supplies the coefficients of that problem (viscosity, body
force, source), the exact fields it produces (velocity, pressure, stress), and
the boundary conditions under which the two agree — so a validation run is three
lines rather than a bespoke script.

See Also
--------
underworld3.analytic : the solution registry and the available solutions.
"""

import numpy as np
import sympy

import underworld3 as uw
from underworld3.utilities._api_tools import uw_object


class AnalyticSolution(uw_object):
    r"""Base class for closed-form solutions used to validate a numerical solve.

    A concrete solution builds its exact fields on ``mesh.X`` in ``__init__`` and
    sets the class-level metadata below. Everything else — applying the boundary
    conditions, evaluating a field at points, measuring the error of a computed
    field — is inherited.

    The canonical validation pattern is::

        sol = uw.analytic.SolCx(mesh, eta_A=1.0, eta_B=1.0e6)

        stokes.constitutive_model.Parameters.shear_viscosity_0 = sol.fn_viscosity
        stokes.bodyforce = sol.fn_bodyforce
        sol.apply_boundary_conditions(stokes)

        stokes.solve()
        rel = sol.error("velocity", stokes.u)

    Attributes
    ----------
    dim : int
        Spatial dimension the solution is posed in (2 or 3).
    nonlinear : bool
        Whether the constitutive law depends on the solution itself.
    reference : str
        Citation for the solution, and the provenance of this implementation.
    eqn_velocity, eqn_pressure, eqn_viscosity, eqn_bodyforce : str
        LaTeX for the defining equations, shown by :meth:`view`. These document
        the *problem*, not the answer: a reader should be able to tell from
        ``eqn_viscosity`` and ``eqn_bodyforce`` alone what is being solved.
    fn_velocity : sympy.Matrix
        Exact velocity, shape ``(1, dim)``.
    fn_pressure : sympy.Expr
        Exact pressure.
    fn_stress : sympy.Matrix
        Exact total (Cauchy) stress, shape ``(dim, dim)``.
    fn_strainrate : sympy.Matrix
        Exact strain rate, shape ``(dim, dim)``.
    fn_viscosity : sympy.Expr
        The viscosity the solution is posed with — assign this to the solver.
    fn_bodyforce : sympy.Matrix
        The body force the solution is posed with, shape ``(1, dim)``.

    Notes
    -----
    Solutions are pure SymPy expressions on ``mesh.X``. That is a deliberate
    single form: it means they compile through the normal JIT path when handed to
    a solver, carry their own analytic Jacobian, can be used as Dirichlet boundary
    values, and evaluate through :func:`underworld3.function.evaluate` like any
    other field. Solutions transcribed from a reference C kernel keep that kernel
    in the tree as an independent oracle; see
    ``docs/developer/subsystems/analytic-solutions.md`` for the validation
    protocol every transcription must pass.
    """

    dim = None
    nonlinear = False
    reference = ""

    #: Which equation this solution answers. The suite began as Stokes-only and
    #: the field names still read that way, so a solution that answers something
    #: else says so here rather than being exempted from the checks: the
    #: conformance suite dispatches on this, and applies the residual for the
    #: equation the solution actually claims.
    solves = "stokes"

    #: Whether the fields are SymPy expressions on ``mesh.X``. True for every
    #: solution in the suite proper — that is the one-form rule. A solution that
    #: wraps an external *numeric* library says so here, and thereby says that
    #: the residual gates cannot be applied to it: there is no expression to
    #: differentiate, so it can be compared against a solver but never checked
    #: against the equations it claims to solve. Treat those as oracles rather
    #: than as validated members of the family.
    symbolic = True

    #: Name of an optional package this solution needs, or None. A solution with
    #: an unmet requirement is still listed by :func:`available` — it just says
    #: it is unavailable, because silently omitting it hides the one fact the
    #: user needs.
    requires = None

    #: Whether this solution's published stress is the deviator rather than the
    #: total. Declared, never inferred — the family is not consistent, and one
    #: kernel writes its deviator into an array named ``total_stress``. Getting
    #: it wrong leaves the momentum residual at order :math:`|\mathbf f|` and
    #: invents a body force, which reads like a transcription failure rather
    #: than a convention one. See the table in the subsystem documentation.
    stress_is_deviatoric = False

    #: Whether validating this solution is expensive enough to keep out of the
    #: per-PR test sweep.
    #:
    #: The cost is intrinsic and predictable from the mathematics rather than a
    #: property of any machine: a solution that accumulates over modes (SolC
    #: sums forty) or carries an exponential viscosity (SolKx, SolKz) produces
    #: expressions with tens of thousands of operations, and every residual gate
    #: differentiates them symbolically and runs common-subexpression
    #: elimination over the result. Measured, the five solutions flagged here
    #: account for 565s of the suite's 1010s while the other eight together cost
    #: about 17s.
    #:
    #: This changes WHICH SOLUTIONS a given run covers, never which checks are
    #: applied to a solution. The full family, gates and all, runs from
    #: ``tests/analytic_full/``.
    expensive_to_validate = False

    #: Whether the source published BOTH a stress and a strain rate, set by
    #: :meth:`set_fields` from what it was actually given.
    #:
    #: This decides what the conformance check
    #: ``sigma + p I == 2 eta edot`` is worth. When only one of the two was
    #: supplied, :meth:`set_fields` derived the other from exactly that identity,
    #: so the check is structural rather than evidential — it re-reads a
    #: derivation. When both were supplied it compares two separately published
    #: quantities and is a real check.
    #:
    #: Recorded here rather than inferred from the source text: a solution that
    #: bypasses :meth:`set_fields` altogether keeps the ``False`` default, which
    #: is the honest answer for it and the one a source scan gets wrong (it finds
    #: the parameter names in this class's own signature).
    publishes_both_stress_and_strainrate = False

    eqn_velocity = ""
    eqn_pressure = ""
    eqn_viscosity = ""
    eqn_bodyforce = ""

    # Field name -> attribute holding the exact expression. A solution that adds a
    # field (a temperature, a pressure head) extends this so error() and
    # evaluate() reach it by name.
    _fields = {
        "solution": "fn_solution",
        "coefficient": "fn_coefficient",
        "source": "fn_source",
        "velocity": "fn_velocity",
        "pressure": "fn_pressure",
        "stress": "fn_stress",
        "strainrate": "fn_strainrate",
        "viscosity": "fn_viscosity",
        "bodyforce": "fn_bodyforce",
    }

    def __init__(self, mesh):
        super().__init__()

        if self.dim is not None and mesh.dim != self.dim:
            raise ValueError(
                f"{type(self).__name__} is a {self.dim}D solution; "
                f"this mesh has dim={mesh.dim}."
            )

        self.mesh = mesh

    @property
    def boundaries(self):
        """The domain walls this solution is posed on.

        The box labels, since the classical solutions are all posed on the unit
        box. A solution in another geometry overrides this with its own labels
        (an annulus, for instance, has ``Upper`` and ``Lower``).
        """

        walls = ["Left", "Right", "Bottom", "Top"]
        if self.mesh.dim == 3:
            walls += ["Front", "Back"]
        return walls

    def set_scalar_field(self, solution, coefficient, source, advection=None):
        r"""Populate the fields of a scalar solution.

        For the transport family — steady diffusion, Darcy flow, Richards — where
        the unknown is one field satisfying
        :math:`\nabla\cdot(k\,\nabla u) + f = 0` rather than a velocity and a
        pressure.

        Parameters
        ----------
        solution : sympy expression
            The exact field: temperature, pressure head, hydraulic head.
        coefficient : sympy expression
            Diffusivity, conductivity or permeability — whatever multiplies the
            gradient in this solution's equation.
        source : sympy expression
            The source term the solution is posed with.
        advection : sequence, optional
            Velocity carrying the field, if it is transported as well as
            diffused. Omitted means no advection, which is the common case;
            supplying it is what distinguishes an advection-diffusion solution
            from a purely diffusive one, and the residual check needs to know.
        """

        self.fn_solution = sympy.sympify(solution)
        self.fn_coefficient = sympy.sympify(coefficient)
        self.fn_source = sympy.sympify(source)
        self.fn_advection = (
            sympy.Matrix([list(advection)])
            if advection is not None
            else sympy.zeros(1, self.dim)
        )

    def set_fields(
        self, velocity, pressure, viscosity, bodyforce, stress=None, strainrate=None
    ):
        r"""Populate the ``fn_*`` attributes from a solution's own components.

        Every solution goes through here, so the conventions are applied in one
        place instead of being re-derived each time. In particular
        :attr:`stress_is_deviatoric` is honoured here and nowhere else: a
        solution declares which stress its source publishes and this converts,
        rather than each one remembering to subtract the pressure.

        Parameters
        ----------
        velocity, bodyforce : sequence
            ``dim`` components each.
        pressure, viscosity : sympy expression
        stress : sequence of sequences, optional
            ``dim x dim``. Read as the deviator or the total according to
            :attr:`stress_is_deviatoric`. Derived from the strain rate if
            omitted.
        strainrate : sequence of sequences, optional
            ``dim x dim``. Derived from the stress if omitted.

        Notes
        -----
        Stress and strain rate are related by :math:`\sigma = -p\,I + 2\eta
        \dot\varepsilon`, so either determines the other once the pressure and
        viscosity are known; supplying both is fine and is worth doing when the
        source publishes both, because then they can be checked against each
        other.
        """

        self.fn_velocity = sympy.Matrix([list(velocity)])
        self.fn_bodyforce = sympy.Matrix([list(bodyforce)])
        self.fn_pressure = sympy.sympify(pressure)
        self.fn_viscosity = sympy.sympify(viscosity)

        # What the source actually gave us, not what the class says it gives.
        # See the attribute's docstring: this is what tells the conformance
        # sweep whether its stress/strain-rate check is evidence or bookkeeping.
        self.publishes_both_stress_and_strainrate = (
            stress is not None and strainrate is not None
        )

        identity = sympy.eye(self.dim)

        if stress is not None:
            given = sympy.Matrix(stress)
            deviator = given if self.stress_is_deviatoric else given + self.fn_pressure * identity
        elif strainrate is not None:
            deviator = 2 * self.fn_viscosity * sympy.Matrix(strainrate)
        else:
            raise ValueError("a solution must supply either stress or strainrate")

        self.fn_stress = deviator - self.fn_pressure * identity
        self.fn_strainrate = (
            sympy.Matrix(strainrate)
            if strainrate is not None
            else deviator / (2 * self.fn_viscosity)
        )

    def sample_points(self, count=12):
        """Points at which this solution can meaningfully be evaluated.

        The default is the unit box or cube, stratified and loaded with the
        boundary and the corners. A solution posed on a different domain — or one
        with a singularity somewhere a generic sampler would happily land on —
        overrides this. The checks in :mod:`underworld3.analytic._validation` ask
        the solution rather than assuming, so adding such a solution does not
        require touching them.

        Parameters
        ----------
        count : int
            Number of interior points.

        Returns
        -------
        numpy.ndarray
            Shape ``(N, dim)``.
        """

        from ._validation import adversarial_points

        return adversarial_points(count=count, dim=self.dim)

    #: Whether this solution is singular at :math:`t = 0`. True for every
    #: diffusive similarity solution in the suite: at the origin the profile is
    #: a step and its gradient is unbounded, so the state the solution describes
    #: at :math:`t = 0` **cannot be represented on any mesh**. Such a solution
    #: must be started from a positive time. See :meth:`at`.
    singular_at_origin = False

    #: The diffusive scale that sets how fast the singularity heals — the
    #: :math:`D` in :math:`\sqrt{Dt}`. Transient solutions set this so that
    #: :meth:`earliest_resolvable_time` can answer in numbers rather than
    #: advice.
    diffusivity = None

    def at(self, time, field=None):
        r"""The solution frozen at *time*.

        Parameters
        ----------
        time : float
            Must be **strictly positive** if :attr:`singular_at_origin`.
        field : str, optional
            Which field. Defaults to the solution's primary unknown.

        Returns
        -------
        sympy expression

        Raises
        ------
        ValueError
            If the solution has no time symbol, or if *time* is at or before the
            singular origin.

        Notes
        -----
        **A transient benchmark is a pair of times, never one.** You initialise
        the solver from this solution at :math:`t_0` and compare against it at
        :math:`t_1`, and the error you measure depends on *both* — a small
        :math:`t_0` means starting from a profile the mesh cannot hold, and the
        error you then attribute to the timestepper is mostly initial
        projection error. Quoting an error at a single time says nothing unless
        the start time is quoted with it.

        :math:`t_0` also has a floor that depends on the mesh, not just on the
        solution — see :meth:`earliest_resolvable_time`.
        """

        if getattr(self, "t", None) is None:
            raise ValueError(
                f"{type(self).__name__} is steady — it has no time symbol, so "
                f"there is no time to evaluate it at."
            )

        time = float(time)

        if self.singular_at_origin and time <= 0.0:
            raise ValueError(
                f"{type(self).__name__} is singular at t = 0: the profile there "
                f"is a step with unbounded gradient, which no mesh can "
                f"represent. Start from a positive time instead — "
                f"`earliest_resolvable_time(h)` will tell you how positive it "
                f"has to be for your resolution. Got t = {time}."
            )

        expression = self._exact(field) if field is not None else self.fn_solution

        return expression.subs(self.t, time)

    def earliest_resolvable_time(self, h, elements_across=4):
        r"""The earliest start time this mesh can actually represent.

        The diffusive front has width :math:`\sim 2\sqrt{Dt}`. Asking for it to
        span *elements_across* cells of size *h* gives

        .. math::
            t \ge \frac{1}{D}\left(\frac{n_{\rm el}\,h}{2}\right)^2

        Below that the exact profile varies faster than the discretisation can
        follow, and a benchmark started there measures interpolation error
        rather than anything about the solver. The floor falls as
        :math:`h^2`, so refining buys an earlier start quickly.

        Parameters
        ----------
        h : float
            Element size. ``mesh.get_min_radius()`` is a reasonable source.
        elements_across : int
            How many elements the front should span. Four is a working
            minimum; fewer and the initial condition is visibly stepped.

        Returns
        -------
        float
            Earliest sensible :math:`t_0`.
        """

        if self.diffusivity is None:
            raise ValueError(
                f"{type(self).__name__} declares no diffusivity, so there is no "
                f"diffusive scale to set a floor from."
            )

        return (float(elements_across) * float(h) / 2.0) ** 2 / float(self.diffusivity)

    def _exact(self, field):
        """Resolve a field name — or pass an expression straight through."""

        if not isinstance(field, str):
            return field

        try:
            return getattr(self, self._fields[field])
        except KeyError:
            available = ", ".join(sorted(self._fields))
            raise ValueError(
                f"{type(self).__name__} has no field {field!r}; "
                f"available: {available}"
            ) from None

    def evaluate(self, field, coords):
        """Exact values of ``field`` at ``coords``.

        Parameters
        ----------
        field : str or sympy expression
            A name from the solution's field set (``"velocity"``,
            ``"pressure"``, ...), or any SymPy expression in ``mesh.X``.
        coords : numpy.ndarray
            Evaluation points, shape ``(N, dim)``.

        Returns
        -------
        numpy.ndarray
            Shape ``(N, *field_shape)``.
        """

        return uw.function.evaluate(self._exact(field), np.asarray(coords))

    def error(self, field, meshvar, norm="l2"):
        r"""Relative error of a computed field against the exact solution.

        Parameters
        ----------
        field : str or sympy expression
            The exact field to compare against — see :meth:`evaluate`.
        meshvar : MeshVariable
            The computed field.
        norm : {"l2", "integral"}
            ``"l2"`` (default) is the discrete nodal relative :math:`L_2` norm
            over the variable's own degrees of freedom. ``"integral"`` is the
            continuous :math:`L_2` norm integrated over the mesh, which is the
            right choice when comparing across different discretisations.

        Returns
        -------
        float
            The same value on every rank.

        Notes
        -----
        Both norms are global. The nodal norm reduces the squared differences and
        the squared exact values across ranks *before* dividing, so it does not
        depend on the partition — an earlier rank-local version reported an error
        10–20x larger on whichever rank owned the hardest region, e.g. the SolCx
        viscosity jump (issue #370). Degrees of freedom shared on a partition
        boundary contribute once per owning rank, a small seam weighting that is
        acceptable for a benchmark diagnostic.
        """

        exact = self._exact(field)

        if norm == "integral":
            zero = (
                sympy.zeros(*exact.shape)
                if isinstance(exact, sympy.Matrix)
                else sympy.S.Zero
            )
            magnitude = uw.maths.L2_norm(zero, exact, self.mesh)
            return float(uw.maths.L2_norm(meshvar.sym, exact, self.mesh) / magnitude)

        if norm != "l2":
            raise ValueError(f"norm must be 'l2' or 'integral'; got {norm!r}")

        computed = np.asarray(meshvar.array).reshape(len(meshvar.coords), -1)
        exact_values = np.asarray(self.evaluate(exact, meshvar.coords)).reshape(
            computed.shape
        )

        difference = computed - exact_values
        error_squared = uw.mpi.comm.allreduce(float((difference**2).sum()))
        exact_squared = uw.mpi.comm.allreduce(float((exact_values**2).sum()))

        return float(np.sqrt(error_squared / exact_squared))

    def set_richards_field(self, solution, conductivity, capacity):
        r"""Declare an unsaturated-flow solution.

        Richards is neither of the other two: the coefficient depends on the
        unknown, and there is a capacity term multiplying the time derivative,
        so it gets its own declaration rather than being forced into
        :meth:`set_scalar_field`.

        Parameters
        ----------
        solution : sympy expression
            Pressure head :math:`\psi`.
        conductivity : sympy expression
            :math:`K(\psi)`, in terms of the solution itself.
        capacity : sympy expression
            :math:`C(\psi) = \mathrm d\theta/\mathrm d\psi`.
        """

        self.fn_solution = sympy.sympify(solution)
        self.fn_conductivity = sympy.sympify(conductivity)
        self.fn_capacity = sympy.sympify(capacity)

    def apply_boundary_conditions(self, solver):
        """Impose the boundary conditions this solution is posed under.

        Every solution writes this itself, because the conditions are part of
        the problem it answers and are not always the same on every boundary.
        :func:`free_slip`, :func:`prescribed_velocity` and
        :func:`prescribed_scalar` are here to be composed inside it — each takes
        the boundaries it applies to, so a solution with a slipping wall and a
        driven one calls both.
        """

        raise NotImplementedError(
            f"{type(self).__name__} does not declare its boundary conditions. "
            f"Write apply_boundary_conditions, composing free_slip, "
            f"prescribed_velocity or prescribed_scalar as the problem requires."
        )

    def _object_viewer(self):
        from IPython.display import Markdown, display

        display(Markdown(f"**{type(self).__name__}** — {self.dim}D"))

        if self.reference:
            display(Markdown(f"*{self.reference}*"))

        for label, equation in (
            ("velocity", self.eqn_velocity),
            ("pressure", self.eqn_pressure),
            ("viscosity", self.eqn_viscosity),
            ("body force", self.eqn_bodyforce),
        ):
            if equation:
                display(Markdown(rf"{label}: $\displaystyle {equation}$"))


# The boundary conditions solutions build from. Functions rather than mixins:
# a mixin can only say "every boundary of this solution gets the same
# condition", which is true of the classical box benchmarks and false of most
# of what the suite has to serve next — a spherical shell or annulus with
# different conditions on the two radii, a faulted disc, a channel driven at one
# end. Composed inside a solution's own `apply_boundary_conditions`, these say
# what is imposed WHERE, and a solution that needs two kinds calls two of them.
#
# The pressure nullspace is deliberately NOT set here. It is a property of the
# domain — enclosed, so the pressure is determined only up to a constant — and
# not of any one wall's condition, so each solution states it for itself:
#
#     solver.petsc_use_pressure_nullspace = True
#
# Leaving it out on an enclosed domain is the failure this whole suite exists to
# catch: a direct solve on the singular saddle returns a quiet, wrong answer
# that only an exact solution exposes.


def free_slip(solver, boundaries, normal=None):
    r"""Impose free slip, :math:`\mathbf u\cdot\hat{\mathbf n}=0`, on *boundaries*.

    Applied as a strong *rotated* constraint (satisfied to machine precision)
    rather than by zeroing a velocity component. On an axis-aligned box the two
    agree; on a curved, tilted or adapted boundary only the rotated form is
    correct, so it is the one that still holds when a solution is used to
    validate an adapted mesh.

    Parameters
    ----------
    solver : Stokes
        The solver to configure.
    boundaries : sequence of str
        The boundary labels to constrain — not necessarily all of them.
    normal : sympy 1 x dim Matrix, optional
        Analytic outward normal. The default takes the solver's geometric facet
        normal, which is consistent with the straight-facet integral the
        assembler evaluates and is normally the right choice even on a curved
        wall. Supply one only when the constraint must follow the TRUE surface
        rather than the mesh; see the "Which normal to use" section of
        ``docs/developer/subsystems/rotated-freeslip.md``.
    """

    # Passed only when the solution has actually chosen a normal, so that
    # otherwise the solver's own geometric default governs.
    choice = {} if normal is None else {"normal": normal}

    for boundary in boundaries:
        solver.add_rotated_freeslip_bc(0.0, boundary, **choice)


def prescribed_velocity(solver, boundaries, velocity):
    """Impose a velocity on *boundaries*, component by component.

    For a solution driven through its boundaries rather than by a body force —
    a far-field shear, say — and for any solution whose exact velocity is not
    tangential to the domain.

    Parameters
    ----------
    solver : Stokes
        The solver to configure.
    boundaries : sequence of str
        The boundary labels to constrain.
    velocity : sympy Matrix, shape (1, dim)
        Usually the solution's own ``fn_velocity``.
    """

    for boundary in boundaries:
        solver.add_dirichlet_bc(velocity, boundary)


def prescribed_scalar(solver, boundaries, field):
    """Impose a scalar field on *boundaries*.

    The scalar counterpart of :func:`prescribed_velocity`, for the transport and
    Richards families: temperature, pressure head, hydraulic head.

    Parameters
    ----------
    solver : Poisson or Darcy
        The solver to configure.
    boundaries : sequence of str
        The boundary labels to constrain.
    field : sympy expression
        Usually the solution's own ``fn_solution``.
    """

    for boundary in boundaries:
        solver.add_dirichlet_bc([field], boundary)
