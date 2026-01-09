# from mpi4py.MPI import DATATYPE_NULL
from libc.stdlib cimport malloc, free
from typing import Optional, Tuple, Union


import numpy as np
import sympy
from petsc4py import PETSc
cimport numpy as np

import underworld3 as uw
import underworld3.timing as timing
import underworld3

include "../cython/petsc_extras.pxi"

# NOTE: Coordinate conversion removed 2025-11-02 (commit to clean architecture)
# The Python wrapper in functions_unit_system.py handles all dimensional ↔ non-dimensional conversions
# Cython functions evaluate_nd() and global_evaluate_nd() now expect plain numpy arrays in [0-1] space

# Make Cython aware of this type.
cdef extern from "petsc.h" nogil:
    ctypedef struct DMInterpolationInfo:
        pass

cdef extern from "petsc.h" nogil:
    ctypedef enum DMSwarmMigrateType:
        pass

cdef extern from "petsc.h" nogil:
    ctypedef enum DMSwarmType:
        pass

cdef extern from "petsc_tools.h" nogil:
    PetscErrorCode DMInterpolationSetUp_UW(DMInterpolationInfo ipInfo, PetscDM dm, int petscbool, int petscbool, size_t* owning_cell)
    PetscErrorCode DMInterpolationEvaluate_UW(DMInterpolationInfo ipInfo, PetscDM dm, PetscVec x, PetscVec v)

cdef extern from "petsc.h" nogil:
    PetscErrorCode DMInterpolationCreate(MPI_Comm comm, DMInterpolationInfo *ipInfo)
    PetscErrorCode DMInterpolationSetDim(DMInterpolationInfo ipInfo, PetscInt dim)
    PetscErrorCode DMInterpolationSetDof(DMInterpolationInfo ipInfo, PetscInt dof)
    PetscErrorCode DMInterpolationAddPoints(DMInterpolationInfo ipInfo, PetscInt n, PetscReal points[])
    PetscErrorCode DMInterpolationSetUp(DMInterpolationInfo ipInfo, PetscDM dm, int petscbool, int petscbool)
    PetscErrorCode DMInterpolationDestroy(DMInterpolationInfo *ipInfo)
    MPI_Comm MPI_COMM_SELF

cdef extern from "petsc.h" nogil:
    PetscErrorCode DMSwarmSetMigrateType(PetscDM dm, DMSwarmMigrateType mtype)
    PetscErrorCode DMSwarmGetMigrateType(PetscDM dm, DMSwarmMigrateType *mtype)

class UnderworldAppliedFunction(sympy.core.function.AppliedUndef):
    """
    This is largely just a shell class to help us differentiate between UW
    and native Sympy functions.
    """
    def fdiff(self, argindex):
        """
        We provide an explicit derivative function.
        This allows us to control the way derivative objects are printed,
        but in the user interface, but more critically it allows us to
        patch in code printing implementation for derivatives objects,
        as utilised in `_jitextension.py`.
        """
        # Construct and return the required deriv fn.
        return self._diff[argindex-1](*self.args)

    def _latex(self, printer, exp=None):

        try:
            mesh=self.mesh
            if not mesh.CoordinateSystem.CartesianDM:
                coord_latex = r"\mathbf{r}"
            else:
                coord_latex = r"\mathbf{x}"
        except:
            coord_latex = r"\mathbf{x}"

        if exp==None:
            latexstr = fr"{type(self).__name__}({coord_latex})"
        else:
            latexstr = fr"{type(self).__name__}^{{ {exp} }}({coord_latex})"

        return latexstr

class UnderworldAppliedFunctionDeriv(UnderworldAppliedFunction):
    """
    This is largely just to help us differentiate between UW
    and native Sympy functions.
    """
    def fdiff(self,argindex):
        raise RuntimeError("Second derivatives of Underworld functions are not supported at this time.")

class UnderworldFunction(sympy.Function):
    """
    This is a metaclass, so it returns programmatic class objects rather
    than instances. This basically follows the pattern of the `sympy.Function`
    metaclass, with two key differences:
    1. We set `UnderworldAppliedFunction` as the base class. This is really just a
       dummy class (see its definition) which allows us to do calls such
       as `isinstance(someobj, UnderworldAppliedFunction)` to test if a `sympy`
       object is one we've defined.
    2. We grab a weakref of the owning meshvariable onto the *class* itself. Note
       that it's important that it's recorded onto the class (instead of the instance),
       as Sympy internally sometimes uses calls such as `type(obj)(obj.args)` to
       replace objects with cloned instances, and therefore 'extra' info must be
       recorded onto the class so that the clones are _complete_.

    Consider the calling pattern

    >>> newfn = UnderworldFunction(meshvar,name)(*meshvar.mesh.r)

    This is equivalent to

    >>> newfnclass = UnderworldFunction(meshvar,name)   # Here we create a new *class*.
    >>> newfn = newfnclass(*meshvar.mesh.r)             # Here we create an instance of the class.

    Parameters
    ----------
    name:
        The name of the function.
    meshvar:
        The mesh variable corresponding to this function.
    vtype:
        The variable type (scalar,vector,etc).
    component:
        For vector functions, this is the component of the vector.
        For example, component `1` might correspond to `v_y`.
        For tensors, the component is a tuple
        For scalars, this value is ignored.
    """
    def __new__(cls,
                name     : str,
                meshvar  : underworld3.discretisation.MeshVariable,
                vtype    : underworld3.VarType,
                component: Union[int, tuple] = 0,
                data_loc: int = None,
                *args, **options):

        if vtype==uw.VarType.VECTOR:
            fname = name + "_{{ {} }}".format(component)
        elif vtype==uw.VarType.TENSOR or vtype==uw.VarType.SYM_TENSOR or vtype ==uw.VarType.MATRIX:
            fname = name + "_{{ {}{} }}".format(component[0], component[1])
        else: # other types can manage their own component names
            fname = name

        # Create function class with _uw_id for disambiguation (2025-12)
        # When meshes have instance_number > 1, include it in _uw_id
        # This makes f1(x,y) from mesh1 distinct from f1(x,y) from mesh2
        # even if they have the same display name, solving the "funny whitespace" problem
        mesh = meshvar.mesh
        uw_id = mesh.instance_number if mesh.instance_number > 1 else None
        ourcls = sympy.core.function.UndefinedFunction(fname,*args, bases=(UnderworldAppliedFunction,), _uw_id=uw_id, **options)
        # Grab weakref to meshvar.
        import weakref
        ourcls.meshvar   = weakref.ref(meshvar)
        ourcls.component = data_loc # <- this is used to index into the data array so it should not just be the tuple

        ourcls._diff = []
        # go ahead and create the derivative function *classes*
        if   vtype==uw.VarType.SCALAR:
            fname = name + "_{,"
        elif vtype==uw.VarType.VECTOR:
            fname = name + "_{{ {},".format(component)
        elif vtype==uw.VarType.TENSOR or vtype == uw.VarType.SYM_TENSOR or vtype ==uw.VarType.MATRIX:
            fname = name + "_{{ {}{},".format(component[0], component[1])

        for index, difffname in enumerate((fname+"0}",fname+"1}",fname+"2}")):
            # Pass _uw_id for derivative functions too (same mesh disambiguation)
            diffcls = sympy.core.function.UndefinedFunction(difffname, *args, bases=(UnderworldAppliedFunctionDeriv,), _uw_id=uw_id, **options)
            # Grab weakref to var for derivative fn too.
            diffcls.meshvar   = weakref.ref(meshvar)
            diffcls.component = data_loc
            diffcls.diffindex = index
            ourcls._diff.append(diffcls)

        return ourcls


# =============================================================================
# Shared helper functions for evaluation paths
# =============================================================================

def _collect_mesh_varfns(mesh):
    """
    Collect all mesh variable function symbols from a mesh.

    Parameters
    ----------
    mesh : Mesh
        The mesh containing variables

    Returns
    -------
    set
        Set of UnderworldAppliedFunction symbols for all mesh variables
    """
    varfns = set()
    if mesh is not None and mesh.vars is not None:
        for v in mesh.vars.values():
            for sub in v.sym:
                varfns.add(sub)
    return varfns


def _lambdify_and_evaluate(expr, coords, interpolated_results, coord_sys=None, mesh=None):
    """
    Substitute interpolated values and evaluate expression via lambdify.

    This is the shared final step for both PETSc and RBF evaluation paths.
    It replaces mesh variable symbols with placeholder symbols, creates a
    lambdified function, and evaluates it with the interpolated values.

    Parameters
    ----------
    expr : sympy expression
        Expression to evaluate (already simplified/unwrapped)
    coords : ndarray
        Coordinates array, shape (n_points, dim)
    interpolated_results : dict
        Mapping from varfn symbols to interpolated value arrays
    coord_sys : CoordSys3D, optional
        Coordinate system to use
    mesh : Mesh, optional
        Mesh for coordinate system fallback

    Returns
    -------
    ndarray
        Evaluated results, shape (n_points, *expr_shape)
    """
    import string
    import random
    from sympy import lambdify
    from sympy.vector import CoordSys3D

    # 1. Replace mesh variables with random symbol placeholders
    varfns_symbols = {}
    for varfn in interpolated_results.keys():
        randstr = ''.join(random.choices(string.ascii_uppercase, k=5))
        varfns_symbols[varfn] = sympy.Symbol(randstr)

    subbedexpr = expr.subs(varfns_symbols)

    # 2. Set up coordinate system
    dim = coords.shape[1]

    if coord_sys is not None:
        N = coord_sys
    elif mesh is None:
        N = CoordSys3D("N")
    else:
        N = mesh.N

    r = N.base_scalars()[0:dim]

    # 3. Handle vector/dyadic expressions
    if isinstance(subbedexpr, sympy.vector.Vector):
        subbedexpr = subbedexpr.to_matrix(N)[0:dim, 0]
    elif isinstance(subbedexpr, sympy.vector.Dyadic):
        subbedexpr = subbedexpr.to_matrix(N)[0:dim, 0:dim]

    # 4. Create lambdified function and evaluate
    lambfn = lambdify((r, varfns_symbols.values()), subbedexpr, docstring_limit=0)

    coords_list = [coords[:, i] for i in range(dim)]
    results = lambfn(coords_list, interpolated_results.values())

    # 5. Handle result shape
    try:
        shape = expr.shape
    except AttributeError:
        shape = (1, 1)

    try:
        results_shape = results.shape
    except AttributeError:
        results_shape = (1, 1)

    # Broadcast constant results to span all coordinates
    if shape == results_shape:
        results_new = np.zeros((coords.shape[0], *shape))
        results_new[...] = results
        results = results_new
    else:
        results = np.moveaxis(results, -1, 0)

    return results.reshape(-1, *shape)


def global_evaluate_nd(   expr,
                coords=None,
                coord_sys=None,
                other_arguments=None,
                simplify=True,
                verbose=False,
                evalf=False,
                rbf=False,
                data_layout=None,
                check_extrapolated=False,
                force_l2=False,
                smoothing=1e-6,
            ):

    """
    Internal: Parallel-safe expression evaluation (Cython implementation).

    This is the low-level Cython implementation for MPI-parallel evaluation.
    Users should typically use :func:`underworld3.function.global_evaluate`
    which provides automatic unit handling and a cleaner interface.

    Note it is not efficient to call this function to evaluate an expression at
    a single coordinate. Instead the user should provide a numpy array of all
    coordinates requiring evaluation.

    See Also
    --------
    underworld3.function.global_evaluate : User-facing parallel evaluation.

    Parameters
    ----------
    expr: sympy.Basic
        Sympy expression requiring evaluation.
    coords: numpy.ndarray, list, or tuple
        Coordinates to evaluate expression at. Can be:
        - numpy array of doubles (shape: n_points x n_dims)
        - list/tuple of tuples with unit-aware coordinates: [(x1, y1), (x2, y2), ...]
        - list/tuple for single point: [x, y] or [x, y, z]
        Coordinate values can be UWQuantity, pint.Quantity, or numeric (float/int).
        Unit-aware coordinates are automatically converted to SI base units.
    coord_sys: mesh.N vector coordinate system

    other_arguments: dict
        Dictionary of other arguments necessary to evaluate function.
        Not yet implemented.

    """

    # NOTE: Coordinates should be non-dimensional [0-1] at this point
    # Python wrapper in functions_unit_system.py handles dimensional conversions
    # CRITICAL: Use np.array() to force copy and strip subclass (e.g. UnitAwareArray)
    # np.asarray() preserves subclass if dtype matches, causing downstream issues
    coords_array = np.array(coords, dtype=np.double, copy=False).view(np.ndarray)

    mesh, varfns, derivfns = uw.function.expressions.mesh_vars_in_expression(expr)

    if mesh is None: #  or uw.mpi.size==1:
        return evaluate_nd(
            expr,
            coords_array,
            coord_sys,
            other_arguments,
            simplify,
            verbose,
            evalf,
            rbf,
            data_layout,
            check_extrapolated=check_extrapolated,
            force_l2=force_l2,
            smoothing=smoothing,
        )

    # If in parallel, define a swarm, migrate, evaluate, migrate back
    # (this is the routine used in advection - see ddt.c / SemiLagrangian)

    # Set up a swarm, add a variable to represent the result of the computation and an 'original_rank' variable
    # so that we can recover the information. We should add a local-index variable so we know how to reorder the
    # values when the particles come back.

    index = np.array(range(0, coords_array.shape[0])).reshape(-1,1,1)

    evaluation_swarm = uw.swarm.Swarm(mesh)

    original_rank = uw.swarm.SwarmVariable(
        "rank",
        evaluation_swarm,
        vtype=uw.VarType.SCALAR,
        dtype=int,
        _proxy=False,
        varsymbol=r"\cal{R}_o",
    )

    original_index = uw.swarm.SwarmVariable(
        "index",
        evaluation_swarm,
        vtype=uw.VarType.SCALAR,
        dtype=int,
        _proxy=False,
        varsymbol=r"\cal{I}",
    )

    is_extrapolated = uw.swarm.SwarmVariable(
        "is_extrapolated",
        evaluation_swarm,
        vtype=uw.VarType.SCALAR,
        dtype=int,
        _proxy=False,
        varsymbol=r"\cal{X}",
    )


    try:
        expr.shape
    except AttributeError:
        expr = sympy.Matrix(((expr,),))

    expr_shape = expr.shape

    data_container = uw.swarm.SwarmVariable(
        "data",
        evaluation_swarm,
        vtype=uw.VarType.MATRIX,
        size=expr.shape,
        dtype=float,
        _proxy=False,
        varsymbol=r"\cal{D}",
    )

    # Populate with particles

    points = evaluation_swarm.add_particles_with_global_coordinates(coords_array, migrate=False)

    original_rank.array[...] = uw.mpi.rank
    original_index.array[...] = index[...]

    index = original_index.array[:,0,0]
    ranks = original_rank.array[:,0,0]

    evaluation_swarm.migrate(remove_sent_points=True, delete_lost_points=False)
    local_coords = evaluation_swarm._particle_coordinates.array[...].reshape(-1,evaluation_swarm.dim)
    values, extrapolated = evaluate_nd(expr, local_coords, rbf=rbf, evalf=evalf, verbose=verbose, check_extrapolated=True,)

    data_container.array[...] = values[...]
    is_extrapolated.array[:,0,0] = extrapolated[:]

    # set rank to old values and migrate back
    evaluation_swarm._rank_var.array[...] = original_rank.array[...]

    # Bare bones migration - just move particles, no validation at all
    # in the BASIC swarm, dm.migrate does not care about whether points
    # lie inside the domain or not.

    evaluation_swarm.dm.migrate(remove_sent_points=True)
    uw.mpi.barrier()


    index = original_index.array[:,0,0]

    return_value = np.empty_like(data_container.array[...])
    return_value[index,:,:] =  data_container.array[:,:,:]

    return_mask = np.empty_like(is_extrapolated.array[...], dtype=bool)
    return_mask[index] =  is_extrapolated.array[:]

    if not check_extrapolated:
        return return_value
    else:
        return return_value, return_mask


def _project_to_work_variable(expr, mesh, smoothing=1e-6):
    """
    Project expression to a work variable, resolving derivatives to nodal values.

    This is used to handle derivative expressions before the interior/exterior
    split in evaluate_nd. The work variable can then be interpolated using
    standard DMInterpolation (interior) and RBF (exterior) paths.

    Parameters
    ----------
    expr : sympy expression
        Expression to project (may contain derivatives)
    mesh : Mesh
        The mesh
    smoothing : float
        Projection smoothing parameter

    Returns
    -------
    MeshVariable
        Work variable containing projected nodal values
    """
    import underworld3 as uw

    # Handle matrix expressions - need multi-component work variable
    if hasattr(expr, 'shape') and expr.shape != (1, 1):
        rows, cols = expr.shape
        n_components = rows * cols

        # Get or create multi-component work variable
        cache_key = f'_eval_work_{n_components}'
        if not hasattr(mesh, cache_key):
            work_var = uw.discretisation.MeshVariable(
                cache_key, mesh, num_components=n_components, degree=1
            )
            setattr(mesh, cache_key, work_var)
            # Create projectors for each component
            projectors = []
            for c in range(n_components):
                proj = uw.systems.Projection(mesh, work_var, scalar_component=c)
                proj.petsc_options["snes_rtol"] = 1e-6
                projectors.append(proj)
            setattr(mesh, f'{cache_key}_projectors', projectors)

        work_var = getattr(mesh, cache_key)
        projectors = getattr(mesh, f'{cache_key}_projectors')

        # Project each component
        idx = 0
        for i in range(rows):
            for j in range(cols):
                projectors[idx].uw_function = expr[i, j]
                projectors[idx].smoothing = smoothing
                projectors[idx].solve(zero_init_guess=False)
                idx += 1

        return work_var

    # Scalar expression
    if not hasattr(mesh, '_eval_work_scalar'):
        mesh._eval_work_scalar = uw.discretisation.MeshVariable(
            "_eval_work_scalar", mesh, num_components=1, degree=1
        )
        mesh._eval_projector_scalar = uw.systems.Projection(
            mesh, mesh._eval_work_scalar
        )
        mesh._eval_projector_scalar.petsc_options["snes_rtol"] = 1e-6

    work_var = mesh._eval_work_scalar
    projector = mesh._eval_projector_scalar

    # Handle 1x1 matrix wrapper
    scalar_expr = expr[0, 0] if hasattr(expr, 'shape') else expr

    projector.uw_function = scalar_expr
    projector.smoothing = smoothing
    projector.solve(zero_init_guess=False)

    return work_var


def _clement_to_work_variable(expr, mesh, derivfns):
    """
    Evaluate expression at nodes using Clement gradient recovery (no solve).

    This is the quick/dirty path for derivative evaluation. Instead of
    projecting (which requires a solve), we:
    1. Compute Clement gradient at mesh nodes for each derivative
    2. Get mesh variable values at nodes
    3. Lambdify and evaluate the full expression at nodes
    4. Store result in work variable

    Parameters
    ----------
    expr : sympy expression
        Expression containing derivatives
    mesh : Mesh
        The mesh
    derivfns : dict
        Dictionary mapping source variables to their derivative expressions

    Returns
    -------
    MeshVariable
        Work variable containing expression values at nodes
    """
    import underworld3 as uw
    import sympy
    from underworld3.function.gradient_evaluation import compute_clement_gradient_at_nodes

    # Get work variable (scalar, P1)
    if not hasattr(mesh, '_clement_work_scalar'):
        mesh._clement_work_scalar = uw.discretisation.MeshVariable(
            "_clement_work_scalar", mesh, num_components=1, degree=1
        )
    work_var = mesh._clement_work_scalar

    # Get node coordinates
    node_coords = work_var.coords
    n_nodes = node_coords.shape[0]

    # Collect all mesh variable functions in expression
    varfns = _collect_mesh_varfns(mesh)

    # Build dictionary of values at nodes for each varfn
    nodal_values = {}
    for varfn in varfns:
        var = varfn.meshvar()
        comp = varfn.component
        # Get nodal values directly from mesh variable data
        if var.degree == 1:
            # P1 - data is at nodes
            nodal_values[varfn] = var.data[:, comp].flatten()
        else:
            # Higher degree - need to evaluate at node coords
            nodal_values[varfn] = uw.function.evaluate(
                var.sym[comp, 0], node_coords, rbf=True
            ).flatten()

    # Compute Clement gradients for derivative source variables
    gradient_at_nodes = {}
    for source_var in derivfns.keys():
        # Compute gradient at nodes using Clement interpolant
        if source_var.num_components == 1:
            grad = compute_clement_gradient_at_nodes(source_var)
            gradient_at_nodes[(source_var, 0)] = grad
        else:
            # Multi-component: compute gradient for each component
            for c in range(source_var.num_components):
                grad = compute_clement_gradient_at_nodes(source_var, component=c)
                gradient_at_nodes[(source_var, c)] = grad

    # Add derivative values to nodal_values dictionary
    for source_var, deriv_list in derivfns.items():
        comp = 0 if source_var.num_components == 1 else 0  # TODO: handle multi-component
        grad = gradient_at_nodes[(source_var, comp)]  # shape (n_nodes, dim)
        for deriv_expr, diffindex in deriv_list:
            # grad[:, diffindex] gives ∂f/∂x_i at all nodes
            nodal_values[deriv_expr] = grad[:, diffindex]

    # Now lambdify and evaluate expression at nodes
    # Handle 1x1 matrix wrapper
    scalar_expr = expr[0, 0] if hasattr(expr, 'shape') else expr

    # Build substitution: replace varfns and derivs with symbols
    subs_dict = {}
    symbols_list = []
    arrays_list = []

    for key, arr in nodal_values.items():
        sym = sympy.Symbol(f"_node_{len(symbols_list)}")
        subs_dict[key] = sym
        symbols_list.append(sym)
        arrays_list.append(arr)

    # Add coordinate symbols
    for d in range(mesh.dim):
        coord_sym = mesh.X[d]
        sym = sympy.Symbol(f"_coord_{d}")
        subs_dict[coord_sym] = sym
        symbols_list.append(sym)
        arrays_list.append(node_coords[:, d])

    # Substitute and lambdify
    expr_substituted = scalar_expr.xreplace(subs_dict)
    func = sympy.lambdify(symbols_list, expr_substituted, modules='numpy')

    # Evaluate at nodes
    result = func(*arrays_list)
    result = np.atleast_1d(result)

    # Handle scalar result (constant expression)
    if result.shape == ():
        result = np.full(n_nodes, float(result))
    elif len(result) == 1 and n_nodes > 1:
        result = np.full(n_nodes, result[0])

    # Store in work variable
    with mesh.access(work_var):
        work_var.data[:, 0] = result.flatten()

    return work_var


def evaluate_nd(   expr,
                coords=None,
                coord_sys=None,
                other_arguments=None,
                simplify=True,
                verbose=False,
                evalf=False,
                rbf=False,
                data_layout=None,
                check_extrapolated=False,
                force_l2=False,
                smoothing=1e-6):
    """
    Internal: Evaluate expression at coordinates (Cython implementation).

    This is the low-level Cython implementation. Users should typically use
    :func:`underworld3.function.evaluate` which provides automatic unit
    handling and a cleaner interface.

    Note it is not efficient to call this function to evaluate an expression at
    a single coordinate. Instead the user should provide a numpy array of all
    coordinates requiring evaluation.

    See Also
    --------
    underworld3.function.evaluate : User-facing function with unit support.

    Parameters
    ----------
    expr: sympy.Basic
        Sympy expression requiring evaluation.
    coords: numpy.ndarray, list, or tuple
        Coordinates to evaluate expression at. Can be:
        - numpy array of doubles (shape: n_points x n_dims)
        - list/tuple of tuples with unit-aware coordinates: [(x1, y1), (x2, y2), ...]
        - list/tuple for single point: [x, y] or [x, y, z]
        Coordinate values can be UWQuantity, pint.Quantity, or numeric (float/int).
        Unit-aware coordinates are automatically converted to SI base units.
    coord_sys: mesh.N vector coordinate system

    other_arguments: dict
        Dictionary of other arguments necessary to evaluate function.
        Not yet implemented.
    """

    # NOTE: Coordinates should be non-dimensional [0-1] at this point
    # Python wrapper in functions_unit_system.py handles dimensional conversions
    # CRITICAL: Use np.array() to force copy and strip subclass (e.g. UnitAwareArray)
    # np.asarray() preserves subclass if dtype matches, causing downstream issues
    coords_array = np.array(coords, dtype=np.double, copy=False).view(np.ndarray)

    dim = coords_array.shape[1]
    mesh, varfns, derivfns = uw.function.fn_mesh_vars_in_expression(expr)

    # coercion - make everything at least a 1x1 matrix for consistent evaluation results
    try:
        expr.shape
    except AttributeError:
        expr = sympy.Matrix(((expr,),))

    # DERIVATIVE HANDLING: Resolve derivatives to nodal values BEFORE interior/exterior split
    # Two modes:
    # - Quick (rbf=True, force_l2=False): Clement gradient at nodes, no solve
    # - Accurate (force_l2=True or rbf=False): L2 projection, requires solve
    if derivfns and mesh is not None:
        if evalf:
            raise RuntimeError(
                "Derivative expressions cannot be evaluated with evalf=True. "
                "Use rbf=True for quick mode, or default for accurate evaluation."
            )

        if rbf and not force_l2:
            # Quick path: Clement gradient recovery at nodes (no projection solve)
            # Fast but O(h) accurate - good for visualization
            work_var = _clement_to_work_variable(expr, mesh, derivfns)
        else:
            # Accurate path: L2 projection (requires solve)
            # O(h²) accurate - use for quantitative work
            work_var = _project_to_work_variable(expr, mesh, smoothing)

        # Replace expression with work variable (no derivatives, standard path works)
        expr = work_var.sym
        derivfns = None  # Derivatives are now resolved

        # Re-extract mesh info for the new expression
        mesh, varfns, _ = uw.function.fn_mesh_vars_in_expression(expr)

    elif force_l2 and mesh is not None:
        # force_l2 without derivatives - project for smoothing
        if evalf:
            raise RuntimeError(
                "force_l2=True cannot be used with evalf=True."
            )
        work_var = _project_to_work_variable(expr, mesh, smoothing)
        expr = work_var.sym
        mesh, varfns, _ = uw.function.fn_mesh_vars_in_expression(expr)

    # If there are no mesh variables, then we have no need of a mesh to
    # help us to evaluate the expression. The evalf / rbf flag will force rbf_evaluation and
    # does not need mesh information either.

    if evalf==True or rbf==True or mesh is None:
        in_or_not = np.full((coords_array.shape[0]), False, dtype=bool )
        evaluation = rbf_evaluate( expr,
                            coords_array,
                            coord_sys,
                            mesh,
                            simplify=simplify,
                            verbose=verbose,
                            )

    else:
        in_or_not = mesh.points_in_domain(coords_array, strict_validation=False)
        evaluation_interior = petsc_interpolate( expr,
                                    coords_array[in_or_not],
                                    coord_sys,
                                    mesh,
                                    simplify=simplify,
                                    verbose=verbose, )

        evaluation_interior = np.atleast_1d(evaluation_interior) # handle case where there is only 1 interior point

        # Exterior points handled via RBF extrapolation
        # Note: derivatives have already been resolved to nodal values above
        if np.count_nonzero(in_or_not == False) > 0:
            evaluation_exterior = rbf_evaluate( expr,
                                coords_array[~in_or_not],
                                coord_sys,
                                mesh,
                                simplify=simplify,
                                verbose=verbose, )
        else:
            evaluation_exterior = None

        if len(evaluation_interior.shape) == 1:
            evaluation = np.empty(shape=(in_or_not.shape[0],))
        else:
            evaluation = np.empty(shape=(in_or_not.shape[0],)+tuple(evaluation_interior.shape[1::]))

        evaluation[in_or_not,...] = evaluation_interior
        evaluation[~in_or_not,...] = evaluation_exterior
        # evaluation = evaluation.squeeze() # consistent behavior with mesh is None and only 1 coord input

    ## We should change this so both evaluation routines return an array that has
    ## shape == (N,i,j) where N is the number of points and where (i,j) is the shape of the evaluation type
    ## (scalar == (1,1); vector= (1,dim); tensor=(dim,dim) - even if symmetric and internal storage is flat -
    ## and so on. We can let the variables themselves handle the packing of data using their _data_layout

    if not callable(data_layout):
        if check_extrapolated:
            return evaluation, ~in_or_not
        else:
            return evaluation
    else:
        shape = evaluation.shape[1::]
        if len(shape) <= 1:
            if check_extrapolated:
                return evaluation, ~in_or_not
                return evaluation
        else:
            i_size = shape[0]
            j_size = shape[1]
            storage_size = data_layout(-1)
            evaluation_1d = np.empty(shape=(evaluation.shape[0], storage_size))

            for i in range(i_size):
                for j in range(j_size):
                    ij = data_layout(i,j)
                    evaluation_1d[:,ij] = evaluation[:,i,j]

        if check_extrapolated:
            return evaluation_1d, ~in_or_not
        else:
            return evaluation_1d


def petsc_interpolate(   expr,
                np.ndarray coords=None,
                coord_sys=None,
                mesh=None,
                other_arguments=None,
                simplify=True,
                verbose=False, ):
    """
    Evaluate a given expression at a list of coordinates.

    Note it is not efficient to call this function to evaluate an expression at
    a single coordinate. Instead the user should provide a numpy array of all
    coordinates requiring evaluation.

    Parameters
    ----------
    expr: sympy.Basic
        Sympy expression requiring evaluation.
    coords: numpy.ndarray
        Numpy array of coordinates to evaluate expression at.
    coord_sys: mesh.N vector coordinate system

    other_arguments: dict
        Dictionary of other arguments necessary to evaluate function.
        Not yet implemented.

    Notes
    -----
    This function leverages Sympy's `lambdify` function to provide efficient
    expression evaluation. It operates as follows:
        1. Extract all Underworld variables functions from the expression. Note that
           all variables functions must be leaf nodes of the corresponding expression
           tree, as the variable function arguments must simply be the coordinate
           vector `mesh.r`. This is a necessary requirement to avoid complication in the
           domain decomposed parallel runtime situation, where a modified variable function
           argument (such as `mesh.r - (10,0)`) might translate the variable function onto
           a neighbouring subdomain. Handling this would result in great complication and
           inefficiency, and we therefore disallow it.
        2. Each variable function is evaluated at the user provided coordinates to generate
           an array of evaluated results.
        3. Replace all variable function instances within the expression with sympy
           symbol placeholders.
        4. Generate a Sympy lambdified expression. This expression takes as arguments the
           user provided coordinates, and the Underworld variable function placeholders.
        5. Evaluate the generated lambdified expresson using the coordinate array and
           evaluated variable function result arrays.
        6. Return results array for full expression evaluation.


    """

    if not (isinstance( expr, sympy.Basic ) or isinstance( expr, sympy.Matrix ) ):
        raise RuntimeError("`evaluate()` function parameter `expr` does not appear to be a sympy expression.")

    sympy.core.cache.clear_cache()

    if uw.function.fn_is_constant_expr(expr):

        constant_value = uw.function.expressions.unwrap(expr, keep_constants=False)
        return np.multiply.outer(np.ones(coords.shape[0]), np.array(constant_value, dtype=float))


    if (not coords is None) and not isinstance( coords, np.ndarray ):
        raise RuntimeError("`evaluate()` function parameter `input` does not appear to be a numpy array.")

    if coords.shape[1] not in [2,3]:
        raise ValueError("Provided `coords` must be 2 dimensional array of coordinates.\n"
                         "For n coordinates:  [[x_0,y_0,z_0],...,[x_n,y_n,z_n]].\n"
                         "Note also that it is inefficient to call this function for a single evaluation,\n"
                         "and you should instead stack up all necessary evaluations into your `coords` array\n"
                         "and call this function once.")
    if coords.dtype != np.double:
        raise ValueError("Provided `coords` must be an array of doubles.")
    if other_arguments:
        raise RuntimeError("`other_arguments` functionality not yet implemented.")

    # Early return for empty coordinate arrays (SECOND CHECK - top-level function)
    # CRITICAL: Avoid lambdify errors with LaTeX variable names when coords is empty
    # This handles cases where empty arrays pass through from evaluate_nd
    if len(coords) == 0:
        # Determine output shape based on expression type
        try:
            expr_shape = expr.shape
            # Return empty array with correct shape: (0, rows, cols)
            return np.empty([0] + list(expr_shape), dtype=np.double)
        except AttributeError:
            # Scalar expression - return (0,) shaped array
            return np.empty([0], dtype=np.double)

    ## Substitute any UWExpressions for their values before calculation
    ## NOTE: We use _unwrap_expressions directly (not fn_substitute_expressions) to avoid
    ## applying scaling transformations which would cause double-scaling since PETSc
    ## already stores non-dimensional values
    expr = uw.function.expressions._unwrap_expressions(expr, keep_constants=False)

    if simplify:
        expr = sympy.simplify(expr)

    # NOTE: Derivative handling (projection) is done in evaluate_nd BEFORE this
    # function is called. By the time we get here, derivatives have been resolved
    # to nodal values in a work variable.

    if verbose and uw.mpi.rank==0:
        print(f"Expression to be evaluated: {expr}")


    # In general, non-constant expressions means that we have a matrix that has at least
    # one spatially-variable function. That can cause a problem if other Matrix entries
    # are not constants (numpy cannot see this as a uniform array). The mesh.CoordinateSystem.zero_matrix is
    # the fix for this. We add it here (so it is not visible in the user-space)

    if mesh is not None:
        expr = expr + mesh.CoordinateSystem.zero_matrix(expr.shape)
        ## NOTE: Use _unwrap_expressions (not fn_substitute_expressions) to prevent double-scaling
        expr = uw.function.expressions._unwrap_expressions(expr, keep_constants=False)

    # if (len(varfns)==0) and (coords is None):
    #     raise RuntimeError("Interpolation coordinates not specified by supplied expression contains mesh variables.\n"
    #                        "Mesh variables can only be interpolated at coordinates.")

    # Create dictionary which creates a per mesh list of vars.
    # Usually there will only be a single mesh, but this allows for the
    # more general situation.

    varfns = _collect_mesh_varfns(mesh)

    from collections import defaultdict
    interpolant_varfns = defaultdict(lambda : [])

    for varfn in varfns:
        if verbose and uw.mpi.rank == 0:
            print(f"Varfn for interpolation: {varfn}")
        interpolant_varfns[varfn.meshvar().mesh].append(varfn)


    # 2. Evaluate all mesh variables - there is no real
    # computational benefit in interpolating a subset.

    def interpolate_vars_on_mesh( varfns, np.ndarray coords ):
        """
        This function performs the interpolation for the given variables
        on a single mesh.
        """

        import xxhash

        # Grab the mesh
        mesh = varfns[0].meshvar().mesh

        if mesh._evaluation_hash is not None:
            xxh = xxhash.xxh64()
            xxh.update(np.ascontiguousarray(coords))
            coord_hash = xxh.intdigest()

            # Note: special case: re-evaluating at the same points
            # after updating mesh variables. This is not captured
            # by a simple coordinate hash. We kill this in the
            # .access for mesh variables but this is prone to mistakes

            if False and coord_hash == mesh._evaluation_hash:
                # if uw.mpi.rank == 0:
                #     print("Using uw.evaluation cache", flush=True)
                return mesh._evaluation_interpolated_results
            else:
                # if uw.mpi.rank == 0:
                #     print("No uw.evaluation cache", flush=True)
                mesh._evaluation_hash = None
                mesh._evaluation_interpolated_results = None


        # For now, eval over all vars
        vars = mesh.vars.values()

        cdef DM dm = mesh.dm

        # Get and set total count of dofs
        dofcount = 0
        var_start_index = {}
        for var in vars:
            var_start_index[var] = dofcount
            dofcount += var.num_components

        # Make coords contiguous for caching and C access
        coords = np.ascontiguousarray(coords)

        # Early return for empty coordinate arrays
        # CRITICAL: Avoid DMInterpolation setup with zero points
        if len(coords) == 0:
            # Return empty array with correct shape: (0, dofcount)
            return np.empty([0, dofcount], dtype=np.double)

        # === DMInterpolation CACHING ===
        # Declare variables at function scope (Cython requirement)
        cdef np.ndarray cells

        # Try to get cached structure first
        from underworld3.function._dminterp_wrapper import CachedDMInterpolationInfo

        # coords is already np.ndarray type in petsc_interpolate function signature
        cached_info = mesh._dminterpolation_cache.get_structure(coords, dofcount)

        # Create output array
        cdef np.ndarray outarray = np.empty([len(coords), dofcount], dtype=np.double)

        if cached_info is not None:
            # CACHE HIT - Fast path. Evaluate using cached structure
            mesh.update_lvec()  # Ensure fresh values
            cached_info.evaluate(mesh, outarray)

        else:
            # CACHE MISS - Create structure and cache it
            cached_info = CachedDMInterpolationInfo()

            # Get cell hints
            # coords is already np.ndarray type (function signature ensures this)
            cells = mesh.get_closest_cells(coords)

            # Create and set up DMInterpolation structure (EXPENSIVE)
            try:
                # coords is already np.ndarray type (function signature ensures this)
                cached_info.create_structure(mesh, coords, cells, dofcount)
            except RuntimeError as e:
                # Handle DMInterpolationSetUp failures gracefully
                if "outside the domain" in str(e):
                    raise RuntimeError("Error encountered when trying to interpolate mesh variable.\n"
                                     "Interpolation location is possibly outside the domain.")
                else:
                    raise

            # Store in cache for reuse
            # coords is already np.ndarray type (function signature ensures this)
            mesh._dminterpolation_cache.store_structure(coords, dofcount, cached_info)

            # Evaluate
            mesh.update_lvec()
            cached_info.evaluate(mesh, outarray)
        # === END CACHING ===

        # Create map between array slices and variable functions
        #
        varfns_arrays = {}
        for varfn in varfns:
            var  = varfn.meshvar()
            comp = varfn.component
            var_start = var_start_index[var]
            arr = np.ascontiguousarray(outarray[:,var_start+comp])
            varfns_arrays[varfn] = arr

        # Cache these results
        xxh = xxhash.xxh64()
        xxh.update(np.ascontiguousarray(coords))
        coord_hash = xxh.intdigest()
        mesh._evaluation_hash = coord_hash
        mesh._evaluation_interpolated_results = varfns_arrays

        return varfns_arrays


    # Get map of all variable functions
    interpolated_results = {}
    for key, vals in interpolant_varfns.items():
        interpolated_var_values = interpolate_vars_on_mesh(vals, coords)
        interpolated_results.update(interpolated_var_values)

    # NOTE: Derivative handling is done in evaluate_nd before this function is called.
    # By the time we get here, any derivatives have been resolved to nodal values
    # via projection. The Clement interpolant is still accessible directly via:
    #   - gradient_evaluation.evaluate_gradient(var, coords, method="interpolant")
    #   - gradient_evaluation.compute_clement_gradient_at_nodes(var)

    # Symbol substitution, lambdify, and evaluate (shared with rbf_evaluate)
    return _lambdify_and_evaluate(expr, coords, interpolated_results, coord_sys, mesh)

# Go ahead and substitute for the timed version.
# Note that we don't use the @decorator sugar here so that
# we can pass in the `class_name` parameter.
evaluate_nd = timing.routine_timer_decorator(routine=evaluate_nd, class_name="Function")

### ------------------------------

def rbf_evaluate(  expr,
            coords=None,
            coord_sys=None,
            mesh=None,
            other_arguments=None,
            verbose=False,
            simplify=True,):
    """
    Evaluate a given expression at a list of coordinates.

    Note it is not efficient to call this function to evaluate an expression at
    a single coordinate. Instead the user should provide a numpy array of all
    coordinates requiring evaluation.

    Parameters
    ----------
    expr: sympy.Basic
        Sympy expression requiring evaluation.
    coords: numpy.ndarray
        Numpy array of coordinates to evaluate expression at.
    coord_sys: mesh.N vector coordinate system

    other_arguments: dict
        Dictionary of other arguments necessary to evaluate function.
        Not yet implemented.

    Notes
    -----
    This function leverages Sympy's `lambdify` function to provide efficient
    expression evaluation. It operates as follows:
        1. Extract all Underworld variables functions from the expression. Note that
           all variables functions must be leaf nodes of the corresponding expression
           tree, as the variable function arguments must simply be the coordinate
           vector `mesh.r`. This is a necessary requirement to avoid complication in the
           domain decomposed parallel runtime situation, where a modified variable function
           argument (such as `mesh.r - (10,0)`) might translate the variable function onto
           a neighbouring subdomain. Handling this would result in great complication and
           inefficiency, and we therefore disallow it.
        2. Each variable function is evaluated at the user provided coordinates to generate
           an array of evaluated results.
        3. Replace all variable function instances within the expression with sympy
           symbol placeholders.
        4. Generate a Sympy lambdified expression. This expression takes as arguments the
           user provided coordinates, and the Underworld variable function placeholders.
        5. Evaluate the generated lambdified expresson using the coordinate array and
           evaluated variable function result arrays.
        6. Return results array for full expression evaluation.


    """

    ## These checks should be in the calling `evaluate` function

    if not (isinstance( expr, sympy.Basic ) or isinstance( expr, sympy.Matrix ) ):
        raise RuntimeError("`evaluate()` function parameter `expr` does not appear to be a sympy expression.")

    sympy.core.cache.clear_cache()

    if uw.function.fn_is_constant_expr(expr):
        constant_value = uw.function.expressions.unwrap(expr, keep_constants=False)
        return np.multiply.outer(np.ones(coords.shape[0]), np.array(constant_value, dtype=float))

    if (not coords is None) and not isinstance( coords, np.ndarray ):
        raise RuntimeError("`evaluate()` function parameter `input` does not appear to be a numpy array.")



    if coords.shape[1] not in [2,3]:
        raise ValueError("Provided `coords` must be 2 dimensional array of coordinates.\n"
                         "For n coordinates:  [[x_0,y_0,z_0],...,[x_n,y_n,z_n]].\n"
                         "Note also that it is inefficient to call this function for a single evaluation,\n"
                         "and you should instead stack up all necessary evaluations into your `coords` array\n"
                         "and call this function once.")
    if coords.dtype != np.double:
        raise ValueError("Provided `coords` must be an array of doubles.")
    if other_arguments:
        raise RuntimeError("`other_arguments` functionality not yet implemented.")


    ## Substitute any uw_expressions for their values before calculation
    ## NOTE: Use _unwrap_expressions (not fn_substitute_expressions) to avoid
    ## double-scaling - same fix as petsc_interpolate
    expr = uw.function.expressions._unwrap_expressions(expr, keep_constants=False)

    if simplify:
        expr = sympy.simplify(expr)

    if mesh is not None:
        expr = expr + mesh.CoordinateSystem.zero_matrix(expr.shape)
        expr = uw.function.expressions._unwrap_expressions(expr, keep_constants=False)
    else:
        try:
            any_basis_vector = tuple(expr.atoms(sympy.vector.scalar.BaseScalar))[0]
            expr = expr + any_basis_vector.CS.zero_matrix(expr.shape)
            expr = uw.function.expressions._unwrap_expressions(expr, keep_constants=False)
        except IndexError:
            pass


    # 2. Collect and evaluate all mesh variables via RBF interpolation
    varfns = _collect_mesh_varfns(mesh)

    interpolated_results = {}
    for varfn in varfns:
        parent, component = uw.discretisation.meshVariable_lookup_by_symbol(mesh, varfn)
        values = parent.rbf_interpolate(coords, nnn=mesh.dim+1)[:, component]
        interpolated_results[varfn] = values
        if verbose:
            print(f"{varfn} = {parent.name}[{component}]")

    # 3. Symbol substitution, lambdify, and evaluate (shared with petsc_interpolate)
    return _lambdify_and_evaluate(expr, coords, interpolated_results, coord_sys, mesh)


# Go ahead and substitute for the timed version.
# Note that we don't use the @decorator here so that
# we can pass in the `class_name` parameter.

rbf_evaluate = timing.routine_timer_decorator(routine=rbf_evaluate, class_name="Function")

## Not sure these belong with the uw function cython

def dm_swarm_get_migrate_type(swarm):

    # cdef DM dm = swarm.dm
    # cdef PetscErrorCode ierr
    # cdef DMSwarmMigrateType mtype

    # ierr = DMSwarmGetMigrateType(dm.dm, &mtype); CHKERRQ(ierr)

    mtype = _dmswarm_get_migrate_type(swarm.dm)

    return mtype

def dm_swarm_set_migrate_type(swarm, mtype:PETsc.DMSwarm.MigrateType):

    _dmswarm_set_migrate_type(swarm.dm, mtype)

    # cdef DM dm = swarm.dm
    # cdef PetscErrorCode ierr
    # cdef DMSwarmMigrateType mig = mtype

    # ierr = DMSwarmSetMigrateType(dm.dm, mig); CHKERRQ(ierr)

    return

def _dmswarm_get_migrate_type(sdm):

    cdef DM dm = sdm
    cdef PetscErrorCode ierr
    cdef DMSwarmMigrateType mtype

    ierr = DMSwarmGetMigrateType(dm.dm, &mtype); CHKERRQ(ierr)

    return mtype

def _dmswarm_set_migrate_type(sdm, mtype:PETsc.DMSwarm.MigrateType):

    cdef DM dm = sdm
    cdef PetscErrorCode ierr
    cdef DMSwarmMigrateType mig = mtype

    ierr = DMSwarmSetMigrateType(dm.dm, mig); CHKERRQ(ierr)

    return
