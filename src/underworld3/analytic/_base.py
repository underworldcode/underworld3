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

    eqn_velocity = ""
    eqn_pressure = ""
    eqn_viscosity = ""
    eqn_bodyforce = ""

    # Field name -> attribute holding the exact expression. A solution that adds a
    # field (a temperature, a pressure head) extends this so error() and
    # evaluate() reach it by name.
    _fields = {
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

    def apply_boundary_conditions(self, solver):
        """Impose the boundary conditions this solution is posed under."""

        raise NotImplementedError(
            f"{type(self).__name__} does not declare its boundary conditions. "
            f"Mix in FreeSlipWalls or FixedWalls, or override this method."
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


class FreeSlipWalls:
    r"""Mixin: the solution is posed with free slip on every wall.

    Free slip is imposed as a strong *rotated* constraint
    (:math:`\mathbf u\cdot\hat{\mathbf n}=0` to machine precision) rather than by
    zeroing a velocity component. On an axis-aligned box the two agree; on a
    curved, tilted or adapted boundary only the rotated form is correct, so it is
    the one that still holds when a solution is used to validate an adapted mesh.

    The domain is enclosed, so the pressure carries a constant nullspace and the
    solver is told to remove it. Leaving that out is the failure this whole suite
    exists to catch: a direct solve on the singular saddle returns a quiet, wrong
    answer that only an exact solution exposes.
    """

    def apply_boundary_conditions(self, solver):
        for boundary in self.boundaries:
            solver.add_rotated_freeslip_bc(0.0, boundary)

        solver.petsc_use_pressure_nullspace = True


class FixedWalls:
    """Mixin: velocity is prescribed on every wall, from the exact solution.

    For solutions driven by their boundaries rather than by a body force — a
    far-field shear, say — and for manufactured solutions whose exact velocity is
    not tangential to the domain. The domain is again enclosed, so the pressure
    nullspace is removed.
    """

    def apply_boundary_conditions(self, solver):
        for boundary in self.boundaries:
            solver.add_dirichlet_bc(self.fn_velocity, boundary)

        solver.petsc_use_pressure_nullspace = True
