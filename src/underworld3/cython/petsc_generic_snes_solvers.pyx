from xmlrpc.client import Boolean

import sympy
from sympy import sympify

from typing import Optional, Union, TypeAlias
from petsc4py import PETSc

import underworld3
import underworld3 as uw
from   underworld3.utilities._jitextension import getext
import underworld3.timing as timing

from underworld3.utilities._api_tools import uw_object
from underworld3.utilities._api_tools import class_or_instance_method

from underworld3.function import expression as public_expression
expression = lambda *x, **X: public_expression(*x, _unique_name_generation=True, **X)


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

        self.Unknowns = self._Unknowns(self)

        self._order = 0
        self._constitutive_model = None
        self._rebuild_after_mesh_update = self._build

        self.name = "Solver_{}_".format(self.instance_number)
        self.petsc_options_prefix = self.name
        self.petsc_options = PETSc.Options(self.petsc_options_prefix)


        return

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
            inner_self._owning_solver.is_setup = False
            return

        @property
        def DFDt(inner_self):
            """Flux time derivative manager (viscoelastic problems)."""
            return inner_self._DFDt

        @DFDt.setter
        def DFDt(inner_self, new_DFDt):
            inner_self._DFDt = new_DFDt
            inner_self._owning_solver.is_setup = False
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

        if self.snes is not None:
            self.snes.destroy()
            self.snes = None

        if self.dm is not None:
            for coarse_dm in self.dm_hierarchy:
                coarse_dm.destroy()

            self.dm = None
            self.dm_hierarchy = [None]

        self.is_setup = False

        return

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

        # Map convergence reasons to descriptive strings (PETSc documentation)
        convergence_reason_map = {
            # Positive reasons = converged
            1: "CONVERGED_FNORM_ABS - ||F|| < atol",
            2: "CONVERGED_FNORM_RELATIVE - ||F|| < rtol*||F_initial||",
            3: "CONVERGED_SNORM_RELATIVE - ||x|| < stol",
            4: "CONVERGED_ITS - Maximum iterations reached",

            # Zero = still iterating (shouldn't see after solve)
            0: "ITERATING - Still iterating (unexpected after solve)",

            # Negative reasons = diverged
            -1: "DIVERGED_FUNCTION_DOMAIN - Function domain error",
            -2: "DIVERGED_FUNCTION_COUNT - Too many function evaluations",
            -3: "DIVERGED_LINEAR_SOLVE - Linear solver failed",
            -4: "DIVERGED_FNORM_NAN - ||F|| is Not-a-Number",
            -5: "DIVERGED_MAX_IT - Maximum iterations exceeded",
            -6: "DIVERGED_LINE_SEARCH - Line search failed",
            -7: "DIVERGED_INNER - Inner solve failed",
            -8: "DIVERGED_LOCAL_MIN - Local minimum reached",
            -9: "DIVERGED_DTOL - ||F|| increased by divtol",
            -10: "DIVERGED_JACOBIAN_DOMAIN - Jacobian calculation failed",
            -11: "DIVERGED_TR_DELTA - Trust region delta too small",
        }

        convergence_reason_string = convergence_reason_map.get(
            converged_reason,
            f"UNKNOWN_CONVERGENCE_REASON_{converged_reason}"
        )

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

    @timing.routine_timer_decorator
    def _build(self,
                    verbose: bool = False,
                    debug: bool = False,
                    debug_name: str = None,
                    ):

        if (not self.is_setup):
            if self.dm is not None:
                if verbose and uw.mpi.rank == 0:
                    print(f"Destroy solver DM", flush=True)

                self.dm.destroy()
                self.dm = None  # Should be able to avoid nuking this if we
                            # can insert new functions in template (surface integrals problematic in
                            # the current implementation )

        # This is a workaround for some problem in the PETSc machinery
        # where we need a surface integral term somewhere on every process
        # if we have a contribution from anywhere. We add a fake one here
        # which just integrates nothing over a bunch of points. It's enough
        # to let the rest of the machinery work.

        if len(self.natural_bcs) > 0:
            if not "Null_Boundary" in self.natural_bcs:
                bc = (0,)*self.Unknowns.u.shape[1]
                self.add_natural_bc(bc, "Null_Boundary")

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

        return


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
            raise("Error: f_id argument must be of type 'int' representing the solver's fields")

        if c_type not in ['dirichlet', 'neumann']:
            raise("'c_type' unknown. Value must be either 'dirichlet' or 'neumann'")

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
            # TODO: DECPRECATE
            import warnings
            warnings.warn(category=DeprecationWarning,
                          message="Using the 'components' argument is being DEPRECATED in the next release\n" +
                                  "The same functionality can be setup with the 'conds' argument and using\n" +
                                  "'sympy.oo' or 'None', see docstring")
            components = np.array(components, dtype=np.int32, ndmin=1)

        elif components is None:
            cpts_list = []
            for i, fn in enumerate(conds):
                if fn != sympy.oo and fn != -sympy.oo:
                    cpts_list.append(i)

            components = np.array(cpts_list, dtype=np.int32, ndmin=1)
        else:
            raise("Unsupported BC 'components' argument")

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
            BC = namedtuple('NaturalBC', ['f_id', 'components', 'fn_f', 'boundary', 'boundary_label_val', 'type', 'PETScID', 'fns'])
            self.natural_bcs.append(BC(f_id, components, sympy_fn, label, -1, "natural", -1, {}))
        elif c_type == 'dirichlet':
            BC = namedtuple('EssentialBC', ['f_id', 'components', 'fn', 'boundary', 'boundary_label_val', 'type', 'PETScID'])
            self.essential_bcs.append(BC(f_id, components,sympy_fn, label, -1,  'essential', -1))


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

        The free-slip penalty method is particularly useful for spherical
        geometries where the normal direction varies along the boundary.
        The penalty term enforces :math:`\\mathbf{v} \\cdot \\mathbf{n} = 0`
        weakly while allowing tangential flow.

        See Also
        --------
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
        raise RuntimeError("Contact Developers - SolverBaseClass F0 is being used")

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
            self._constitutive_model.order = self._order
            # Establish bidirectional reference so parameter changes can propagate to solver
            self._constitutive_model.Parameters._solver = self


        ### checking if it's a class
        elif type(model_or_class) == type(uw.constitutive_models.Constitutive_Model):
            self._constitutive_model = model_or_class(self.Unknowns)
            self._constitutive_model.order = self._order
            # Establish bidirectional reference so parameter changes can propagate to solver
            self._constitutive_model.Parameters._solver = self



        ### Raise an error if it's neither
        else:
            raise RuntimeError(
                "constitutive_model must be a valid class or instance of a valid class"
            )

        # May not work due to flux being incomplete
        if self.Unknowns.DFDt is not None:
            self.Unknowns.DFDt.psi_fn = self._constitutive_model.flux.T



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
            raise("'section_type' unknown. Value must be either 'local' or 'global'")

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
            self.Unknowns.u = uw.discretisation.MeshVariable( mesh=mesh, num_components=mesh.dim,
                                                      varname="Us{}".format(SNES_Scalar._obj_count),
                                                      vtype=uw.VarType.SCALAR, degree=degree, )

        self.Unknowns.u = u_Field
        self.Unknowns.DuDt = DuDt
        self.Unknowns.DFDt = DFDt

        self.verbose = verbose
        self._tolerance = 1.0e-4

        # Here we can set some defaults for this set of KSP / SNES solvers

        ## FAST as possible for simple problems:
        ## MG,

        # ROBUST and general GAMG, heavy-duty solvers in the suite

        self.petsc_options["snes_type"] = "newtonls"
        self.petsc_options["ksp_type"] = "gmres"
        self.petsc_options["pc_type"] = "gamg"
        self.petsc_options["pc_gamg_type"] = "agg"
        self.petsc_options["pc_gamg_repartition"]  = True
        self.petsc_options["pc_mg_type"]  = "additive"
        self.petsc_options["pc_gamg_agg_nsmooths"] = 2
        self.petsc_options["mg_levels_ksp_max_it"] = 3
        self.petsc_options["mg_levels_ksp_converged_maxits"] = None

        self.petsc_options["snes_rtol"] = 1.0e-4
        self.petsc_options["mg_levels_ksp_max_it"] = 3

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
        # self._constitutive_model = None

        self.verbose = verbose

        self._rebuild_after_mesh_update = self._build  # Maybe just reboot the dm

        # Some other setup

        self.mesh._equation_systems_register.append(self)

        self.is_setup = False

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
        self.mesh_dm_coordinate_hash == mesh_dm_coord_hash


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
                                str(boundary).encode('utf8'),
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

        # f0 = sympy.Array(self._f0).reshape(1).as_immutable()
        # F1 = sympy.Array(self._f1).reshape(dim).as_immutable()

        # f0  = sympy.Array(uw.function.fn_substitute_expressions(self.F0.sym)).reshape(1).as_immutable()
        # F1  = sympy.Array(uw.function.fn_substitute_expressions(self.F1.sym)).reshape(dim).as_immutable()

        f0  = sympy.Array(uw.function.expressions._unwrap_for_compilation(self.F0.sym, keep_constants=False, return_self=False)).reshape(1).as_immutable()
        F1  = sympy.Array(uw.function.expressions._unwrap_for_compilation(self.F1.sym, keep_constants=False, return_self=False)).reshape(dim).as_immutable()

        self._u_f0 = f0
        self._u_F1 = F1

        U = sympy.Array(self.u.sym).reshape(1).as_immutable() # scalar works better in derive_by_array
        L = sympy.Array(self.Unknowns.L).reshape(cdim).as_immutable() # unpack one index here too

        fns_residual = [self._u_f0, self._u_F1]

        G0 = sympy.derive_by_array(f0, U)
        G1 = sympy.derive_by_array(f0, L)
        G2 = sympy.derive_by_array(F1, U)
        G3 = sympy.derive_by_array(F1, L)

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
            # if bc_is is None:
            #     print(f"{uw.mpi.rank}: Skip bc {boundary}", flush=True)
            #     continue

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

            # if bc.fn_F is not None:

            #     bd_F1  = sympy.Array(bc.fn_F).reshape(dim)
            #     self._bd_f1 = sympy.ImmutableDenseMatrix(bd_F1)

            #     G2 = sympy.derive_by_array(self._bd_f1, U)
            #     G3 = sympy.derive_by_array(self._bd_f1, self.Unknowns.L)

            #     self._bd_uu_G2 = sympy.ImmutableMatrix(G2.reshape(dim)) # sympy.ImmutableMatrix(sympy.permutedims(G2, permutation).reshape(dim*dim,dim))
            #     self._bd_uu_G3 = sympy.ImmutableMatrix(G3.reshape(dim,dim)) # sympy.ImmutableMatrix(sympy.permutedims(G3, permutation).reshape(dim*dim,dim*dim))

            #     fns_bd_residual += [self._bd_f1]
            #     fns_bd_jacobian += [self._bd_G2, self._bd_G3]


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
        self.compiled_extensions, self.ext_dict = getext(self.mesh,
                                       tuple(fns_residual),
                                       tuple(fns_jacobian),
                                       [x.fn for x in self.essential_bcs],
                                       tuple(fns_bd_residual),
                                       tuple(fns_bd_jacobian),
                                       primary_field_list=prim_field_list,
                                       verbose=verbose,
                                       debug=debug,)

        return


    @timing.routine_timer_decorator
    def _setup_solver(self, verbose=False):

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



        # Rebuild this lot

        for coarse_dm in self.dm_hierarchy:
            self.dm.copyFields(coarse_dm)
            self.dm.copyDS(coarse_dm)

        # self.dm.createClosureIndex(None)

        for coarse_dm in self.dm_hierarchy:
            coarse_dm.createClosureIndex(None)

        self.dm.setUp()

        self.snes = PETSc.SNES().create(PETSc.COMM_WORLD)
        self.snes.setDM(self.dm)
        self.snes.setOptionsPrefix(self.petsc_options_prefix)
        self.snes.setFromOptions()


        cdef DM dm = self.dm
        UW_DMPlexSetSNESLocalFEM(dm.dm, PETSC_FALSE, NULL)

        self.is_setup = True
        self.constitutive_model._solver_is_setup = True

    @timing.routine_timer_decorator
    def solve(self,
              zero_init_guess: bool =True,
              _force_setup:    bool =False,
              verbose:         bool=False,
              debug:           bool=False,
              debug_name:      str=None ):
        """
        Solve the system of equations.

        Assembles and solves the discretized PDE system using PETSc's SNES
        (Scalable Nonlinear Equations Solvers) framework. The solution is
        stored in the solver's unknown variable(s).

        Parameters
        ----------
        zero_init_guess : bool, default=True
            If True, use zero as the initial guess. If False, use the current
            values in the solution variable(s) as the initial guess, which can
            improve convergence for time-stepping or continuation methods.
        _force_setup : bool, default=False
            Force rebuild of the solver even if already set up. Useful after
            changing boundary conditions or constitutive parameters.
        verbose : bool, default=False
            Print solver progress and timing information.
        debug : bool, default=False
            Enable debug output including intermediate residuals.
        debug_name : str, optional
            Name prefix for debug output files.

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


        if _force_setup or not self.constitutive_model._solver_is_setup:
            self.is_setup = False

        self._build(verbose, debug, debug_name)

        gvec = self.dm.getGlobalVec()

        if not zero_init_guess:
            # with self.mesh.access():
            self.dm.localToGlobal(self.u.vec, gvec)
        else:
            gvec.array[:] = 0.0

        # Set quadrature to consistent value given by mesh quadrature.
        # self.mesh._align_quadratures()

        ## ----

        cdef DM dm = self.dm
        self.mesh.update_lvec()
        cdef Vec cmesh_lvec = self.mesh.lvec

        # cmesh_lvec = vn2.copy()
        # PETSc == 3.16 introduced an explicit interface
        # for setting the aux-vector which we'll use when available.

        ierr = DMSetAuxiliaryVec_UW(dm.dm, NULL, 0, 0, cmesh_lvec.vec); CHKERRQ(ierr)

        # solve
        self.snes.solve(None, gvec)

        lvec = self.dm.getLocalVec()
        cdef Vec clvec = lvec
        # Copy solution back into user facing variable
        # with self.mesh.access(self.u,):
        self.dm.globalToLocal(gvec, lvec)
        # add back boundaries.
        ierr = DMPlexSNESComputeBoundaryFEM(dm.dm, <void*>clvec.vec, NULL); CHKERRQ(ierr)
        self.u.vec.array[:] = lvec.array[:]
        self.mesh._stale_lvec = True

        # Invalidate cached data views - PETSc buffer may have changed
        # Handle both EnhancedMeshVariable (has _base_var) and direct _MeshVariable
        target_var = getattr(self.u, "_base_var", self.u)
        if hasattr(target_var, "_canonical_data"):
            target_var._canonical_data = None

        self.dm.restoreLocalVec(lvec)
        self.dm.restoreGlobalVec(gvec)

        converged = self.snes.getConvergedReason()
        iterations = self.snes.getIterationNumber()

        if not converged and uw.mpi.rank == 0:
            print(f"Convergence problems after {iterations} its in SNES solver use:\n",
                  f"  <solver>.petsc_options.setValue('ksp_monitor',  None)\n",
                  f"  <solver>.petsc_options.setValue('snes_monitor', None)\n",
                  f"  <solver>.petsc_options.setValue('snes_converged_reason', None)\n",
                  f"to investigate convergence problems",
                  flush=True
            )

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

# LM: this is probably not something we need ... The petsc interface is
# general enough to have one class to handle Vector and Scalar

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

        # self.u = u_Field
        # self.DuDt = DuDt
        # self.DFDt = DFDt

        ## Keep track

        self.verbose = verbose
        self._tolerance = 1.0e-4

        ## Todo: this is obviously not particularly robust

        # options = PETSc.Options()
        # options["dm_adaptor"]= "pragmatic"

        # Here we can set some defaults for this set of KSP / SNES solvers
        self.petsc_options["snes_type"] = "newtonls"
        self.petsc_options["ksp_rtol"] = 1.0e-3
        self.petsc_options["ksp_type"] = "gmres"
        self.petsc_options["pc_type"] = "gamg"
        self.petsc_options["pc_gamg_type"] = "agg"
        self.petsc_options["pc_gamg_repartition"]  = True
        self.petsc_options["pc_mg_type"]  = "additive"
        self.petsc_options["pc_gamg_agg_nsmooths"] = 2
        self.petsc_options["snes_rtol"] = 1.0e-3
        self.petsc_options["mg_levels_ksp_max_it"] = 3
        self.petsc_options["mg_levels_ksp_converged_maxits"] = None

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
        if not u_Field:
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
        # self._constitutive_model = None

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
        self.mesh_dm_coordinate_hash == mesh_dm_coord_hash

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
        dim = self.mesh.dim
        cdim = self.mesh.cdim

        sympy.core.cache.clear_cache()

        ## The jacobians are determined from the above (assuming we
        ## do not concern ourselves with the zeros)
        ## Convert to arrays for the moment to allow 1D arrays (size dim, not 1xdim)
        ## otherwise we have many size-1 indices that we have to collapse

        # f0 = sympy.Array(self.mesh.vector.to_matrix(self._f0)).reshape(dim)
        # F1 = sympy.Array(self._f1).reshape(dim,dim)

        # f0 = sympy.Array(self._f0).reshape(1).as_immutable()
        # F1 = sympy.Array(self._f1).reshape(dim).as_immutable()

        # f0  = sympy.Array(uw.function.fn_substitute_expressions(self.F0.sym)).reshape(dim).as_immutable()
        # F1  = sympy.Array(uw.function.fn_substitute_expressions(self.F1.sym)).reshape(dim,dim).as_immutable()

        f0  = sympy.Array(uw.function.expressions._unwrap_for_compilation(self.F0.sym, keep_constants=False, return_self=False)).reshape(dim).as_immutable()
        F1  = sympy.Array(uw.function.expressions._unwrap_for_compilation(self.F1.sym, keep_constants=False, return_self=False)).reshape(dim,dim).as_immutable()


        self._u_f0 = f0
        self._u_F1 = F1

        # JIT compilation needs immutable, matrix input (not arrays)
        self._u_f0 = sympy.ImmutableDenseMatrix(f0)
        self._u_F1 = sympy.ImmutableDenseMatrix(F1)
        fns_residual = [self._u_f0, self._u_F1]

        # This is needed to eliminate extra dims in the tensor
        U = sympy.Array(self.u.sym).reshape(dim)

        G0 = sympy.derive_by_array(f0, U)
        G1 = sympy.derive_by_array(f0, self.Unknowns.L)
        G2 = sympy.derive_by_array(F1, U)
        G3 = sympy.derive_by_array(F1, self.Unknowns.L)

        # reorganise indices from sympy to petsc ordering
        # reshape to Matrix form
        # Make hashable (immutable)

        permutation = (0,3,1,2)

        self._G0 = sympy.ImmutableMatrix(G0.reshape(dim,dim))
        self._G1 = sympy.ImmutableMatrix(sympy.permutedims(G1, (2,1,0)  ).reshape(dim,dim*dim))
        self._G2 = sympy.ImmutableMatrix(sympy.permutedims(G2, (2,1,0)  ).reshape(dim*dim,dim))
        self._G3 = sympy.ImmutableMatrix(sympy.permutedims(G3, permutation).reshape(dim*dim,dim*dim))

        ##################

        fns_jacobian = (self._G0, self._G1, self._G2, self._G3)

        # Now natural bcs (compiled into boundary integral terms)
        # Need to loop on them all ...

        # Now natural bcs (compiled into boundary integral terms)
        # Need to loop on them all ...

        fns_bd_residual = []
        fns_bd_jacobian = []

        for index, bc in enumerate(self.natural_bcs):

            if bc.fn_f is not None:

                bd_F0  = sympy.Array(bc.fn_f)
                bc.fns["u_f0"] = sympy.ImmutableDenseMatrix(bd_F0)

                G0 = sympy.derive_by_array(bd_F0, U)
                G1 = sympy.derive_by_array(bd_F0, self.Unknowns.L)

                bc.fns["uu_G0"] = sympy.ImmutableMatrix(G0.reshape(dim,dim)) # sympy.ImmutableMatrix(sympy.permutedims(G0, permutation).reshape(dim,dim))
                bc.fns["uu_G1"] = sympy.ImmutableMatrix(G1.reshape(dim*dim,dim)) # sympy.ImmutableMatrix(sympy.permutedims(G1, permutation).reshape(dim,dim*dim))

                fns_bd_residual += [bc.fns["u_f0"]]
                fns_bd_jacobian += [bc.fns["uu_G0"], bc.fns["uu_G1"]]


            # Going to leave these out for now, perhaps a different user-interface altogether is required for flux-like bcs

            # if bc.fn_F is not None:

            #     bd_F1  = sympy.Array(bc.fn_F).reshape(dim)
            #     self._bd_f1 = sympy.ImmutableDenseMatrix(bd_F1)


            #     G2 = sympy.derive_by_array(self._bd_f1, U)
            #     G3 = sympy.derive_by_array(self._bd_f1, self.Unknowns.L)

            #     self._bd_uu_G2 = sympy.ImmutableMatrix(G2.reshape(dim,dim)) # sympy.ImmutableMatrix(sympy.permutedims(G2, permutation).reshape(dim*dim,dim))
            #     self._bd_uu_G3 = sympy.ImmutableMatrix(G3.reshape(dim,dim*dim)) # sympy.ImmutableMatrix(sympy.permutedims(G3, permutation).reshape(dim*dim,dim*dim))

            #     fns_bd_residual += [self._bd_f1]
            #     fns_bd_jacobian += [self._bd_uu_G2, self._bd_uu_G3]


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
        self.compiled_extensions, self.ext_dict = getext(self.mesh,
                                       tuple(fns_residual),
                                       tuple(fns_jacobian),
                                       [x.fn for x in self.essential_bcs],
                                       tuple(fns_bd_residual),
                                       tuple(fns_bd_jacobian),
                                       primary_field_list=prim_field_list,
                                       verbose=verbose,
                                       debug=debug,)

        cdef PtrContainer ext = self.compiled_extensions

        return


    @timing.routine_timer_decorator
    def _setup_solver(self, verbose=False):


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
            #bc_label = self.dm.getLabel(boundary)

            label_val = value

            i_bd_res = self.ext_dict.bd_res
            i_bd_jac = self.ext_dict.bd_jac

            c_label = bc_label

            if True: #  c_label and label_val != -1:
                if bc.fn_f is not None:

                    UW_PetscDSSetBdResidual(ds.ds, c_label.dmlabel, label_val, boundary_id,
                                    0, 0,
                                    ext.fns_bd_residual[i_bd_res[bc.fns["u_f0"]]],
                                    NULL, # ext.fns_bd_residual[i_bd_res[bc.fns["u_F1"]]],
                                    )

                    UW_PetscDSSetBdJacobian(ds.ds, c_label.dmlabel, label_val, boundary_id,
                                    0, 0, 0,
                                    ext.fns_bd_jacobian[i_bd_jac[bc.fns["uu_G0"]]],
                                    ext.fns_bd_jacobian[i_bd_jac[bc.fns["uu_G1"]]],
                                    NULL, # ext.fns_bd_jacobian[i_bd_jac[bc.fns["uu_G2"]]],
                                    NULL, # ext.fns_bd_jacobian[i_bd_jac[bc.fns["uu_G3"]]]
                                    )

                    UW_PetscDSSetBdJacobianPreconditioner(ds.ds, c_label.dmlabel, label_val, boundary_id,
                                    0, 0, 0,
                                    ext.fns_bd_jacobian[i_bd_jac[bc.fns["uu_G0"]]],
                                    ext.fns_bd_jacobian[i_bd_jac[bc.fns["uu_G1"]]],
                                    NULL, # ext.fns_bd_jacobian[i_bd_jac[bc.fns["uu_G2"]]],
                                    NULL, # ext.fns_bd_jacobian[i_bd_jac[bc.fns["uu_G3"]]]
                                    )


        if verbose:
            print(f"Weak form (DS)", flush=True)
            UW_PetscDSViewWF(ds.ds)
            print(f"=============", flush=True)

            print(f"Weak form(s) (Natural Boundaries)", flush=True)
            for boundary in self.natural_bcs:
                UW_PetscDSViewBdWF(ds.ds, boundary.PETScID)

        # Rebuild this lot

        for coarse_dm in self.dm_hierarchy:
            self.dm.copyFields(coarse_dm)
            self.dm.copyDS(coarse_dm)

        for coarse_dm in self.dm_hierarchy:
            coarse_dm.createClosureIndex(None)

        self.dm.setUp()

        self.snes = PETSc.SNES().create(PETSc.COMM_WORLD)
        self.snes.setDM(self.dm)
        self.snes.setOptionsPrefix(self.petsc_options_prefix)
        self.snes.setFromOptions()

        cdef DM dm = self.dm
        UW_DMPlexSetSNESLocalFEM(dm.dm, PETSC_FALSE, NULL)


        self.is_setup = True
        self.constitutive_model._solver_is_setup = True




    @timing.routine_timer_decorator
    def solve(self,
              zero_init_guess: bool =True,
              _force_setup:    bool =False,
              verbose=False,
              debug=False,
              debug_name=None,
               ):
        """
        Solve the vector field system of equations.

        Assembles and solves the discretized PDE system for vector unknowns
        (e.g., velocity in projection problems) using PETSc's SNES framework.

        Parameters
        ----------
        zero_init_guess : bool, default=True
            If True, use zero as the initial guess. If False, use the current
            values in ``self.u`` as the initial guess.
        _force_setup : bool, default=False
            Force rebuild of the solver even if already set up.
        verbose : bool, default=False
            Print solver progress and timing information.
        debug : bool, default=False
            Enable debug output.
        debug_name : str, optional
            Name prefix for debug output files.

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

        if _force_setup or not self.constitutive_model._solver_is_setup:
            self.is_setup = False

        self._build(verbose, debug, debug_name)

        # if (not self.is_setup):
        #     if self.dm is not None:
        #         self.dm.destroy()
        #         self.dm = None  # Should be able to avoid nuking this if we
        #                     # can insert new functions in template (surface integrals problematic in
        #                     # the current implementation )

        #     self._setup_pointwise_functions(verbose, debug=debug, debug_name=debug_name)
        #     self._setup_discretisation(verbose)
        #     self._setup_solver(verbose)
        # else:
        #     # If only the mesh has changed, this will rebuild (and do nothing if unchanged)
        #     self._setup_discretisation(verbose)


        gvec = self.dm.getGlobalVec()

        if not zero_init_guess:
            # with self.mesh.access():
            self.dm.localToGlobal(self.u.vec, gvec)
        else:
            gvec.array[:] = 0.

        # Set quadrature to consistent value given by mesh quadrature.
        # self.mesh._align_quadratures()

        # COMMENTED OUT: These calls are NOT in SNES_Scalar (Poisson) or Stokes
        # They appear to destroy field registrations, causing "Invalid field number" errors
        # when variables are created after other solvers have run.
        # Removing to match working Poisson pattern.
        #
        # # Call `createDS()` on aux dm. This is necessary after the
        # # quadratures are set above, as it generates the tablatures
        # # from the quadratures (among other things no doubt).
        # # TODO: What are the implications of calling this every solve.
        #
        # self.mesh.dm.clearDS()
        # self.mesh.dm.createDS()
        #
        # for cdm in self.mesh.dm_hierarchy:
        #     self.mesh.dm.copyDisc(cdm)

        self.mesh.update_lvec()
        cdef DM dm = self.dm
        cdef Vec cmesh_lvec
        # PETSc == 3.16 introduced an explicit interface
        # for setting the aux-vector which we'll use when available.
        cmesh_lvec = self.mesh.lvec
        ierr = DMSetAuxiliaryVec_UW(dm.dm, NULL, 0, 0, cmesh_lvec.vec); CHKERRQ(ierr)

        # solve
        self.snes.solve(None,gvec)

        lvec = self.dm.getLocalVec()
        cdef Vec clvec = lvec
        # Copy solution back into user facing variable
        # with self.mesh.access(self.u):

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

        # Invalidate cached data views - PETSc buffer may have changed
        # Handle both EnhancedMeshVariable (has _base_var) and direct _MeshVariable
        target_var = getattr(self.u, "_base_var", self.u)
        if hasattr(target_var, "_canonical_data"):
            target_var._canonical_data = None

        self.dm.restoreLocalVec(lvec)
        self.dm.restoreGlobalVec(gvec)

        converged = self.snes.getConvergedReason()
        iterations = self.snes.getIterationNumber()

        if not converged and uw.mpi.rank == 0:
            print(f"Convergence problems after {iterations} its in SNES solver use:\n",
                  f"  <solver>.petsc_options.setValue('ksp_monitor',  None)\n",
                  f"  <solver>.petsc_options.setValue('snes_monitor', None)\n",
                  f"  <solver>.petsc_options.setValue('snes_converged_reason', None)\n",
                  f"to investigate convergence problems",
                  flush=True
            )

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

        # Works / mostly quick
        self.petsc_options[f"fieldsplit_{p_name}_ksp_type"] = "fgmres"
        self.petsc_options[f"fieldsplit_{p_name}_ksp_rtol"]  = self._tolerance
        self.petsc_options[f"fieldsplit_{p_name}_pc_type"] = "gasm"
        # self.petsc_options[f"fieldsplit_{p_name}_pc_gasm_type"] = "basic"

        ## may be more robust but usually slower
        # self.petsc_options[f"fieldsplit_{p_name}_ksp_type"] = "fgmres"
        # self.petsc_options[f"fieldsplit_{p_name}_ksp_rtol"]  = self._tolerance * 0.1
        # self.petsc_options[f"fieldsplit_{p_name}_pc_type"] = "gamg"
        # self.petsc_options[f"fieldsplit_{p_name}_pc_gamg_type"] = "agg"
        # self.petsc_options[f"fieldsplit_{p_name}_pc_gamg_repartition"] = True

        # Great set of options for gamg
        self.petsc_options[f"fieldsplit_{v_name}_ksp_type"] = "cg"
        self.petsc_options[f"fieldsplit_{v_name}_ksp_rtol"]  = self._tolerance * 0.1
        self.petsc_options[f"fieldsplit_{v_name}_pc_type"]  = "gamg"
        self.petsc_options[f"fieldsplit_{v_name}_pc_gamg_type"]  = "agg"
        self.petsc_options[f"fieldsplit_{v_name}_pc_gamg_repartition"]  = True
        self.petsc_options[f"fieldsplit_{v_name}_pc_mg_type"]  = "additive"
        self.petsc_options[f"fieldsplit_{v_name}_pc_gamg_agg_nsmooths"] = 2
        self.petsc_options[f"fieldsplit_{v_name}_mg_levels_ksp_max_it"] = 3
        self.petsc_options[f"fieldsplit_{v_name}_mg_levels_ksp_converged_maxits"] = None

        # Create this dict
        self.fields = {}
        self.fields[p_name] = self.p
        self.fields[v_name] = self.u

        # Some other setup

        self.mesh._equation_systems_register.append(self)
        self._rebuild_after_mesh_update = self._build # probably just needs to boot the DM and then it should work

        # self.F0 = sympy.Matrix.zeros(1, self.mesh.dim)
        # self.gF0 = sympy.Matrix.zeros(1, self.mesh.dim)
        # self.F1 = sympy.Matrix.zeros(self.mesh.dim, self.mesh.dim)
        # self.PF0 = sympy.Matrix.zeros(1, 1)

        self.essential_bcs = []

        self.natural_bcs = []
        self.bcs = self.essential_bcs
        self.boundary_conditions = False
        # self._constitutive_model = None
        self._saddle_preconditioner = None

        # Construct strainrate tensor for future usage.
        # Grab gradients, and let's switch out to sympy.Matrix notation
        # immediately as it is probably cleaner for this.
        N = mesh.N

        ## sympy.Matrix - gradient tensors
        self._G = self.p.sym.jacobian(self.mesh.CoordinateSystem.N)

        # this attrib records if we need to re-setup
        self.is_setup = False

    # @timing.routine_timer_decorator
    # def add_essential_p_bc(self, fn, boundary):
    #     # switch to numpy arrays
    #     # ndmin arg forces an array to be generated even
    #     # where comps/indices is a single value.

    #     self.is_setup = False
    #     import numpy as np

    #     try:
    #         iter(fn)
    #     except:
    #         fn = (fn,)

    #     components = np.array([0], dtype=np.int32, ndmin=1)

    #     sympy_fn = sympy.Matrix(fn).as_immutable()

    #     from collections import namedtuple
    #     BC = namedtuple('EssentialBC', ['components', 'fn', 'boundary', 'boundary_label_val', 'type', 'PETScID'])
    #     self.essential_p_bcs.append(BC(components, sympy_fn, boundary, -1,  'essential', -1))

    ## Why is this here - this is not "generic" at all ??

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

    @tolerance.setter
    def tolerance(self, value):
        self._tolerance = value
        self.petsc_options["snes_rtol"] = self._tolerance
        self.petsc_options["snes_ksp_ew"] = None
        self.petsc_options["snes_ksp_ew_version"] = 3

        self.petsc_options["ksp_atol"]  = self._tolerance * 1.0e-6
        self.petsc_options["fieldsplit_pressure_ksp_rtol"]  = self._tolerance * 0.1  # rule of thumb
        self.petsc_options["fieldsplit_velocity_ksp_rtol"]  = self._tolerance * 0.033


    @property
    def strategy(self):
        """
        Solver strategy controlling preconditioner configuration.

        Currently supports:
        - ``"default"``: Standard Schur complement fieldsplit with GAMG
        - ``"robust"``: (Reserved) More robust but slower configuration
        - ``"fast"``: (Reserved) Faster but less robust configuration

        Setting this property reconfigures the entire preconditioner stack.

        Returns
        -------
        str
            Current strategy name.
        """
        return self._strategy

    @strategy.setter
    def strategy(self, value):
        # self.is_setup = False
        self._strategy = value

        # All strategies: reset to preferred

        self.petsc_options["snes_ksp_ew"] = None
        self.petsc_options["snes_ksp_ew_version"] = 3

        self.petsc_options["pc_type"] = "fieldsplit"
        self.petsc_options["pc_fieldsplit_type"] = "schur"
        self.petsc_options["pc_fieldsplit_schur_fact_type"] = "full"     # diag is an alternative (quick/dirty)
        self.petsc_options["pc_fieldsplit_schur_precondition"] = "a11"   # despite what the docs say for saddle points

        self.petsc_options["pc_fieldsplit_diag_use_amat"] = None
        self.petsc_options["pc_fieldsplit_off_diag_use_amat"] = None
        # self.petsc_options["pc_use_amat"] = None                         # Using this puts more pressure on the inner solve


        if value == "robust":

            pass


        elif value == "fast":

            pass


        else: # "default"

            pass

        p_name = "pressure" # pressureField.clean_name
        v_name = "velocity" # velocityField.clean_name

        # Works / mostly quick
        self.petsc_options[f"fieldsplit_{p_name}_ksp_type"] = "fgmres"
        self.petsc_options[f"fieldsplit_{p_name}_ksp_rtol"]  = self._tolerance
        self.petsc_options[f"fieldsplit_{p_name}_pc_type"] = "gasm"
        self.petsc_options[f"fieldsplit_{p_name}_pc_gasm_type"] = "basic"

        ## may be more robust but usually slower
        # self.petsc_options[f"fieldsplit_{p_name}_ksp_type"] = "fgmres"
        # self.petsc_options[f"fieldsplit_{p_name}_ksp_rtol"]  = self._tolerance * 0.1
        # self.petsc_options[f"fieldsplit_{p_name}_pc_type"] = "gamg"
        # self.petsc_options[f"fieldsplit_{p_name}_pc_gamg_type"] = "agg"
        # self.petsc_options[f"fieldsplit_{p_name}_pc_gamg_repartition"] = True


        self.petsc_options[f"fieldsplit_velocity_ksp_type"] = "cg"
        self.petsc_options[f"fieldsplit_velocity_pc_type"]  = "gamg"
        self.petsc_options[f"fieldsplit_velocity_pc_gamg_type"]  = "agg"
        self.petsc_options[f"fieldsplit_velocity_pc_gamg_repartition"]  = True
        self.petsc_options[f"fieldsplit_velocity_pc_mg_type"]  = "kaskade"
        self.petsc_options[f"fieldsplit_velocity_pc_gamg_agg_nsmooths"] = 2
        self.petsc_options[f"fieldsplit_velocity_mg_levels_ksp_max_it"] = 3
        self.petsc_options[f"fieldsplit_velocity_mg_levels_ksp_converged_maxits"] = None



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


    ## F0, F1 should be f0 and F1, (pf0 for Saddles can be added here)
    ## don't add new ones uf0, uF1 are redundant

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

        # r = self.mesh.CoordinateSystem.N[0]

        # Array form to work well with what is below
        # The basis functions are 3-vectors by default, even for 2D meshes, soooo ...
        # F0  = sympy.Array(self._u_f0)  #.reshape(dim)
        # F1  = sympy.Array(self._u_f1)  # .reshape(dim,dim)
        # PF0 = sympy.Array(self._p_f0)# .reshape(1)

        ## We don't need to use these arrays, we can specify the ordering of the indices
        ## and do these one by one as required by PETSc. However, at the moment, this
        ## is working .. so be careful !!

        F0  = sympy.Array(uw.function.expressions._unwrap_for_compilation(self.F0.sym, keep_constants=False, return_self=False))
        F1  = sympy.Array(uw.function.expressions._unwrap_for_compilation(self.F1.sym, keep_constants=False, return_self=False))
        PF0  = sympy.Array(uw.function.expressions._unwrap_for_compilation(self.PF0.sym, keep_constants=False, return_self=False))

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

        G0 = sympy.derive_by_array(F0, self.u.sym)
        G1 = sympy.derive_by_array(F0, self.Unknowns.L)
        G2 = sympy.derive_by_array(F1, self.u.sym)
        G3 = sympy.derive_by_array(F1, self.Unknowns.L)

        # reorganise indices from sympy to petsc orssdering / reshape to Matrix form
        # ijkl -> LJKI (hence 3120)
        # ij k -> KJ I (hence 210)
        # i jk -> J KI (hence 201)

        # The indices need to be interleaved, but for symmetric problems
        # there are lots of symmetries. This means we can find it hard to debug
        # the required permutation for a non-symmetric problem
        permutation = (0,2,1,3) # ? same symmetry as I_ijkl ? # OK
        # permutation = (0,2,3,1) # ? same symmetry as I_ijkl ? # OK
        # permutation = (3,1,2,0) # ? same symmetry as I_ijkl ? # OK

        self._uu_G0 = sympy.ImmutableMatrix(sympy.permutedims(G0, permutation).reshape(dim,dim))
        self._uu_G1 = sympy.ImmutableMatrix(sympy.permutedims(G1, permutation).reshape(dim,dim*dim))
        self._uu_G2 = sympy.ImmutableMatrix(sympy.permutedims(G2, permutation).reshape(dim*dim,dim))
        self._uu_G3 = sympy.ImmutableMatrix(sympy.permutedims(G3, permutation).reshape(dim*dim,dim*dim))

        fns_jacobian += [self._uu_G0, self._uu_G1, self._uu_G2, self._uu_G3]

        # U/P block (check permutations - hard to validate without a full collection of examples)

        G0 = sympy.derive_by_array(F0, self.p.sym)
        G1 = sympy.derive_by_array(F0, self._G)
        G2 = sympy.derive_by_array(F1, self.p.sym)
        G3 = sympy.derive_by_array(F1, self._G)

        self._up_G0 = sympy.ImmutableMatrix(G0.reshape(dim))  # zero in tests
        self._up_G1 = sympy.ImmutableMatrix(sympy.permutedims(G1, permutation).reshape(dim,dim))  # zero in stokes tests
        self._up_G2 = sympy.ImmutableMatrix(sympy.permutedims(G2, permutation).reshape(dim,dim))  # ?
        self._up_G3 = sympy.ImmutableMatrix(sympy.permutedims(G3, permutation).reshape(dim*dim,dim))  # zeros

        fns_jacobian += [self._up_G0, self._up_G1, self._up_G2, self._up_G3]

        # P/U block (check permutations)

        G0 = sympy.derive_by_array(PF0, self.u.sym)
        G1 = sympy.derive_by_array(PF0, self.Unknowns.L)
        # G2 = sympy.derive_by_array(FP1, U) # We don't have an FP1 !
        # G3 = sympy.derive_by_array(FP1, self.Unknowns.L)

        self._pu_G0 = sympy.ImmutableMatrix(G0.reshape(dim))  # non zero
        self._pu_G1 = sympy.ImmutableMatrix(G1.reshape(dim*dim))  # non-zero
        # self._pu_G2 = sympy.ImmutableMatrix(sympy.derive_by_array(FP1, self.p.sym).reshape(dim,dim))
        # self._pu_G3 = sympy.ImmutableMatrix(sympy.derive_by_array(FP1, self._G).reshape(dim,dim*2))

        # fns_jacobian += [self._pu_G0, self._pu_G1, self._pu_G2, self._pu_G3]
        fns_jacobian += [self._pu_G0, self._pu_G1]

        ## PP block is a preconditioner term, not auto-constructed

        if self.saddle_preconditioner is not None:
            self._pp_G0 = self.saddle_preconditioner
        else:
            self._pp_G0 = sympy.simplify(1 / self.constitutive_model.K)

        fns_jacobian.append(self._pp_G0)

        # Now natural bcs (compiled into boundary integral terms)
        # Need to loop on them all ...

        fns_bd_residual = []
        fns_bd_jacobian = []

        for index, bc in enumerate(self.natural_bcs):

            if bc.fn_f is not None:

                permutation = (0,2,1,3) # ? same symmetry as I_ijkl ? # OK

                bd_F0  = sympy.Array(bc.fn_f)

                bc.fns["u_f0"] = sympy.ImmutableDenseMatrix(bd_F0)
                fns_bd_residual += [bc.fns["u_f0"]]

                G0 = sympy.derive_by_array(bd_F0, self.Unknowns.u.sym)
                G1 = sympy.derive_by_array(bd_F0, self.Unknowns.L)
                bc.fns["uu_G0"] = sympy.ImmutableMatrix(sympy.permutedims(G0, permutation).reshape(dim,dim)) # sympy.ImmutableMatrix(sympy.permutedims(G0, permutation).reshape(dim,dim))
                bc.fns["uu_G1"] = sympy.ImmutableMatrix(sympy.permutedims(G1, permutation).reshape(dim,dim*dim)) # sympy.ImmutableMatrix(sympy.permutedims(G1, permutation).reshape(dim,dim*dim))
                fns_bd_jacobian += [bc.fns["uu_G0"], bc.fns["uu_G1"]]

                G0 = sympy.derive_by_array(bc.fns["u_f0"], P)
                G1 = sympy.derive_by_array(bc.fns["u_f0"], self._G)

                bc.fns["up_G0"] = sympy.ImmutableMatrix(G0.reshape(dim)) # sympy.ImmutableMatrix(sympy.permutedims(G0, permutation).reshape(dim,dim))
                bc.fns["up_G1"] = sympy.ImmutableMatrix(sympy.permutedims(G1, permutation).reshape(dim,dim)) # sympy.ImmutableMatrix(sympy.permutedims(G1, permutation).reshape(dim,dim*dim))
                fns_bd_jacobian += [bc.fns["up_G0"], bc.fns["up_G1"]]

                # bc.fns["pu_G0"] = sympy.ImmutableMatrix(sympy.ImmutableMatrix(sympy.Matrix.zeros(rows=1,cols=dim))) # sympy.ImmutableMatrix(sympy.permutedims(G2, permutation).reshape(dim*dim,dim))
                # bc.fns["pu_G1"] = sympy.ImmutableMatrix(sympy.ImmutableMatrix(sympy.Matrix.zeros(rows=dim,cols=dim))) # sympy.ImmutableMatrix(sympy.permutedims(G3, permutation).reshape(dim*dim,dim*dim))
                # fns_bd_jacobian += [bc.fns["pu_G0"], bc.fns["pu_G1"],]

                # Set this explicitly to zero initially
                # fn_F = sympy.Matrix([[0,0],[0,0]])

                # bd_F1  = sympy.Array(fn_F).reshape(dim,dim)
                # bc.fns["u_F1"] = sympy.ImmutableDenseMatrix(bd_F1)
                # fns_bd_residual += [bc.fns["u_F1"]]

                # G2 = bc.fns["u_F1"].diff(self.Unknowns.u.sym)
                # G3 = bc.fns["u_F1"].diff(self.Unknowns.L)
                # bc.fns["uu_G2"] = sympy.ImmutableMatrix(sympy.permutedims(G2, permutation).reshape(dim*dim,dim)) # sympy.ImmutableMatrix(sympy.permutedims(G2, permutation).reshape(dim*dim,dim))
                # bc.fns["uu_G3"] = sympy.ImmutableMatrix(sympy.permutedims(G3, permutation).reshape(dim*dim,dim*dim)) # sympy.ImmutableMatrix(sympy.permutedims(G3, permutation).reshape(dim*dim,dim*dim))
                # fns_bd_jacobian += [bc.fns["uu_G2"], bc.fns["uu_G3"]]

                # G2 = sympy.derive_by_array(bc.fns["u_F1"], P)
                # G3 = sympy.derive_by_array(bc.fns["u_F1"], self._G)
                # bc.fns["up_G2"] = sympy.ImmutableMatrix(G2.reshape(dim,dim)) # sympy.ImmutableMatrix(sympy.permutedims(G2, permutation).reshape(dim*dim,dim))
                # bc.fns["up_G3"] = sympy.ImmutableMatrix(G3.reshape(dim,dim*dim)) # sympy.ImmutableMatrix(sympy.permutedims(G3, permutation).reshape(dim*dim,dim*dim))
                # fns_bd_jacobian += [bc.fns["up_G2"], bc.fns["up_G3"],]

                bc.fns["pp_G0"] = sympy.ImmutableMatrix([0])
                fns_bd_jacobian += [bc.fns["pp_G0"]]


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
        self.compiled_extensions, self.ext_dict = getext(self.mesh,
                                       tuple(fns_residual),
                                       tuple(fns_jacobian),
                                       [x.fn for x in self.essential_bcs],
                                       tuple(fns_bd_residual),
                                       tuple(fns_bd_jacobian),
                                       primary_field_list=prim_field_list,
                                       verbose=verbose,
                                       debug=debug,
                                       debug_name=debug_name,
                                       cache=False)


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
        self.mesh_dm_coordinate_hash == mesh_dm_coord_hash

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


            options = PETSc.Options()
            options.setValue("private_{}_u_petscspace_degree".format(self.petsc_options_prefix), u_degree) # for private variables
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

            # bc_label = self.dm.getLabel(boundary)
            # bc_is = bc_label.getStratumIS(value)
            # self.natural_bcs[index] = self.natural_bcs[index]._replace(boundary_label_val=value)

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


        for coarse_dm in self.dm_hierarchy:
            self.dm.copyFields(coarse_dm)
            self.dm.copyDS(coarse_dm)

        self.is_setup = False

        return



    @timing.routine_timer_decorator
    def _setup_solver(self, verbose=False):

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

        cdef DMLabel c_label

        for bc in self.natural_bcs:

            boundary = bc.boundary
            boundary_id = bc.PETScID

            value = self.mesh.boundaries[bc.boundary].value
            bc_label = self.dm.getLabel("UW_Boundaries")
            # bc_label = self.dm.getLabel(boundary)

            label_val = value

            i_bd_res = self.ext_dict.bd_res
            i_bd_jac = self.ext_dict.bd_jac

            c_label = bc_label

            if True: #  c_label and label_val != -1:

                if bc.fn_f is not None:

                    UW_PetscDSSetBdResidual(ds.ds, c_label.dmlabel, label_val, boundary_id,
                                    0, 0,
                                    ext.fns_bd_residual[i_bd_res[bc.fns["u_f0"]]],
                                    NULL, # ext.fns_bd_residual[i_bd_res[bc.fns["u_F1"]]],
                                    )

                    UW_PetscDSSetBdResidual(ds.ds, c_label.dmlabel, label_val, boundary_id, 1, 0, NULL, NULL)


                    UW_PetscDSSetBdJacobian(ds.ds, c_label.dmlabel, label_val, boundary_id,
                                    0, 0, 0,
                                    ext.fns_bd_jacobian[i_bd_jac[bc.fns["uu_G0"]]],
                                    ext.fns_bd_jacobian[i_bd_jac[bc.fns["uu_G1"]]],
                                    NULL, # ext.fns_bd_jacobian[i_bd_jac[bc.fns["uu_G2"]]],
                                    NULL, # ext.fns_bd_jacobian[i_bd_jac[bc.fns["uu_G3"]]]
                                    )

                    UW_PetscDSSetBdJacobian(ds.ds, c_label.dmlabel, label_val, boundary_id,
                                    0, 1, 0,
                                    ext.fns_bd_jacobian[i_bd_jac[bc.fns["up_G0"]]],
                                    ext.fns_bd_jacobian[i_bd_jac[bc.fns["up_G1"]]],
                                    NULL, NULL)

                    # UW_PetscDSSetBdJacobian(ds.ds, c_label.dmlabel, label_val, boundary_id,
                    #                 1, 0, 0,
                    #                 NULL, # ext.fns_bd_jacobian[i_bd_jac[bc.fns["pu_G0"]]],
                    #                 NULL, # ext.fns_bd_jacobian[i_bd_jac[bc.fns["pu_G1"]]],
                    #                 NULL, NULL)

                    # UW_PetscDSSetBdJacobian(ds.ds, c_label.dmlabel, label_val, boundary_id,
                    #                 1, 1, 0,
                    #                 ext.fns_bd_jacobian[i_bd_jac[bc.fns["pp_G0"]]],
                    #                 NULL, NULL, NULL)

                    UW_PetscDSSetBdJacobianPreconditioner(ds.ds, c_label.dmlabel, label_val, boundary_id,
                                    0, 0, 0,
                                    ext.fns_bd_jacobian[i_bd_jac[bc.fns["uu_G0"]]],
                                    ext.fns_bd_jacobian[i_bd_jac[bc.fns["uu_G1"]]],
                                    NULL, # ext.fns_bd_jacobian[i_bd_jac[bc.fns["uu_G2"]]],
                                    NULL, # ext.fns_bd_jacobian[i_bd_jac[bc.fns["uu_G3"]]]
                                    )

                    UW_PetscDSSetBdJacobianPreconditioner(ds.ds, c_label.dmlabel, label_val, boundary_id,
                                    0, 1, 0,
                                    ext.fns_bd_jacobian[i_bd_jac[bc.fns["up_G0"]]],
                                    ext.fns_bd_jacobian[i_bd_jac[bc.fns["up_G1"]]],
                                    NULL, NULL)

                    # UW_PetscDSSetBdJacobianPreconditioner(ds.ds, c_label.dmlabel, label_val, boundary_id,
                    #                 1, 0, 0,
                    #                 NULL, # ext.fns_bd_jacobian[i_bd_jac[bc.fns["pu_G0"]]],
                    #                 NULL, # ext.fns_bd_jacobian[i_bd_jac[bc.fns["pu_G1"]]],
                    #                 NULL, NULL)

                    # UW_PetscDSSetBdJacobianPreconditioner(ds.ds, c_label.dmlabel, label_val, boundary_id,
                    #                 1, 1, 0,
                    #                 ext.fns_bd_jacobian[i_bd_jac[bc.fns["pp_G0"]]],
                    #                 NULL,
                    #                 NULL,
                    #                 NULL)

        if verbose:
            print(f"Weak form (DS)", flush=True)
            UW_PetscDSViewWF(ds.ds)
            print(f"=============", flush=True)

            print(f"Weak form(s) (Natural Boundaries)", flush=True)
            for boundary in self.natural_bcs:
                UW_PetscDSViewBdWF(ds.ds, boundary.PETScID)


        # self.dm.setUp()
        # self.dm.ds.setUp()


        # Rebuild this lot

        for coarse_dm in self.dm_hierarchy:
            self.dm.copyFields(coarse_dm)
            self.dm.copyDS(coarse_dm)
            # coarse_dm.createDS()

        for coarse_dm in self.dm_hierarchy:
            coarse_dm.createClosureIndex(None)

        self.snes = PETSc.SNES().create(PETSc.COMM_WORLD)
        self.snes.setDM(self.dm)
        self.snes.setOptionsPrefix(self.petsc_options_prefix)
        self.snes.setFromOptions()

        cdef DM c_dm = self.dm
        UW_DMPlexSetSNESLocalFEM(c_dm.dm, PETSC_FALSE, NULL)

        # Setup subdms here too.
        # These will be used to copy back/forth SNES solutions
        # into user facing variables.

        names, isets, dms = self.dm.createFieldDecomposition()
        self._subdict = {}
        for index,name in enumerate(names):
            self._subdict[name] = (isets[index],dms[index])

        self.is_setup = True
        self.constitutive_model._solver_is_setup = True


    @timing.routine_timer_decorator
    def solve(self,
              zero_init_guess: bool = True,
              picard: int = 0,
              verbose=False,
              debug=False,
              debug_name=None,
              _force_setup: bool =False, ):
        """
        Solve the Stokes system for velocity and pressure.

        Assembles and solves the coupled velocity-pressure system using a
        saddle-point formulation. Handles nonlinear rheologies through
        Newton or Picard iteration.

        Parameters
        ----------
        zero_init_guess : bool, default=True
            If True, use zero as the initial guess. If False, use current
            values in ``self.u`` (velocity) and ``self.p`` (pressure) as
            initial guess. Using False can improve convergence for
            time-stepping or parameter continuation.
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

        Returns
        -------
        None
            Solution stored in ``self.u`` (velocity) and ``self.p`` (pressure).

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

        if _force_setup or not self.constitutive_model._solver_is_setup:
            self.is_setup = False

        self._build(verbose, debug, debug_name)

        # Keep a record of these set-up parameters
        tolerance = self.tolerance
        snes_type = self.snes.getType()
        snes_max_it = 50

        self.mesh.update_lvec()
        self.dm.setAuxiliaryVec(self.mesh.lvec, None)

        gvec = self.dm.getGlobalVec()
        gvec.setArray(0.0)

        if not zero_init_guess:

            if verbose and uw.mpi.rank == 0:
                print(f"SNES pre-solve - non-zero initial guess", flush=True)


            self.petsc_options.setValue("snes_max_it", 0)
            self.snes.setType("nrichardson")
            self.snes.setFromOptions()
            self.snes.solve(None, gvec)

            # with self.mesh.access():
            for name,var in self.fields.items():
                sgvec = gvec.getSubVector(self._subdict[name][0])  # Get global subvec off solution gvec.
                subdm   = self._subdict[name][1]                   # Get subdm corresponding to field
                subdm.localToGlobal(var.vec,sgvec)                 # Copy variable data into gvec
                gvec.restoreSubVector(self._subdict[name][0], sgvec)

            self.atol = self.snes.getFunctionNorm() * self.tolerance

        else:
            self.atol = 0.0

        if verbose and uw.mpi.rank == 0:
            print(f"SNES solve - picard = {picard}", flush=True)

        # Picard solves if requested

        if picard != 0:
            self.petsc_options.setValue("snes_max_it", abs(picard))
            self.tolerance = tolerance
            self.snes.atol = self.atol
            self.snes.setType("nrichardson")
            self.snes.setFromOptions()
            self.snes.solve(None, gvec)

        # Now go back to the original plan
            self.snes.setType(snes_type)
            self.tolerance = tolerance
            self.snes.atol = self.atol
            self.petsc_options.setValue("snes_max_it", snes_max_it)
            self.snes.setFromOptions()
            self.snes.solve(None, gvec)

        else:
        # Standard Newton solve
            self.snes.setType(snes_type)
            self.tolerance = tolerance
            self.snes.atol = self.atol
            self.petsc_options.setValue("snes_max_it", snes_max_it)
            self.snes.setFromOptions()
            self.snes.solve(None, gvec)

        cdef DM dm = self.dm
        cdef Vec clvec = self.dm.getLocalVec()
        self.dm.globalToLocal(gvec, clvec)
        ierr = DMPlexSNESComputeBoundaryFEM(dm.dm, <void*>clvec.vec, NULL); CHKERRQ(ierr)

        if verbose and uw.mpi.rank == 0:
                 print(f"SNES Compute Boundary FEM Successfull", flush=True)

        # get index set of pressure and velocity to separate solution from localvec
        # get local section
        local_section = self.dm.getLocalSection()

        # Get the index sets for velocity and pressure fields
        # Field numbers (adjust based on your setup)
        velocity_field_num = 0
        pressure_field_num = 1

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

        # Get index sets for pressure (both constrained and unconstrained points)
        # we need indexset of pressure field to separate the solution from localvec.
        # so we don't care whether a point is constrained by bc or not
        pressure_is = get_local_field_is(local_section, pressure_field_num)

        # Get the total number of entries in the local vector
        size = self.dm.getLocalVec().getLocalSize()

        # Create a list of all indices
        all_indices = set(range(size))

        # Get indices of the pressure field
        pressure_indices = set(pressure_is.getIndices())

        # Compute the complement for the velocity field
        velocity_indices = sorted(list(all_indices - pressure_indices))

        # Create the index set for velocity
        velocity_is = PETSc.IS().createGeneral(velocity_indices, comm=PETSc.COMM_SELF)

        # Copy solution back into pressure and velocity variables
        # with self.mesh.access(self.Unknowns.p, self.Unknowns.u):
        for name, var in self.fields.items():
            if name=='velocity':
                var.vec.array[:] = clvec.getSubVector(velocity_is).array[:]
            elif name=='pressure':
                var.vec.array[:] = clvec.getSubVector(pressure_is).array[:]
        self.mesh._stale_lvec = True


        self.dm.restoreGlobalVec(clvec)
        self.dm.restoreGlobalVec(gvec)

        converged = self.snes.getConvergedReason()
        iterations = self.snes.getIterationNumber()

        if not converged and uw.mpi.rank == 0:
            print(f"Convergence problems after {iterations} its in SNES solver use:\n",
                  f"  <solver>.petsc_options.setValue('ksp_monitor',  None)\n",
                  f"  <solver>.petsc_options.setValue('snes_monitor', None)\n",
                  f"  <solver>.petsc_options.setValue('snes_converged_reason', None)\n",
                  f"to investigate convergence problems",
                  flush=True
            )

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

        # with self.mesh.access():
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
