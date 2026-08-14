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
    PetscErrorCode DMInterpolationSetUp_UW(DMInterpolationInfo ipInfo, PetscDM dm, int petscbool, int petscbool, size_t* owning_cell, int petscbool)
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
    Applied Underworld function representing a mesh variable evaluated at coordinates.

    This class extends SymPy's AppliedUndef to represent Underworld mesh variables
    in symbolic expressions. When a mesh variable's symbolic representation (e.g.,
    ``velocity.sym``) appears in an expression, it is an instance of this class.

    The class provides:

    - Type identification via ``isinstance(obj, UnderworldAppliedFunction)``
    - Custom derivative handling through :meth:`fdiff`
    - LaTeX rendering that shows the coordinate system (Cartesian or curvilinear)
    - Access to the parent mesh variable via the ``meshvar`` weakref on the class

    Notes
    -----
    Users typically don't instantiate this class directly. It is created
    automatically when accessing ``mesh_variable.sym`` or when mesh variables
    appear in symbolic expressions.

    See Also
    --------
    UnderworldFunction : The metaclass that creates these applied function classes.
    UnderworldAppliedFunctionDeriv : Derivative version of this class.
    """
    def fdiff(self, argindex):
        """
        SymPy protocol: Return derivative with respect to the argindex-th argument.

        This is an internal method called by SymPy's differentiation machinery
        (e.g., ``sympy.diff(expr, x)``). Users should not call this directly.

        By implementing this method, we control how Underworld mesh variables
        are differentiated symbolically, and ensure the resulting derivative
        objects can be compiled by the JIT system in ``_jitextension.py``.
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
    First derivative of an Underworld mesh variable function.

    This class represents spatial derivatives of mesh variables (e.g.,
    :math:`\\partial T / \\partial x`). Instances are created automatically
    when differentiating mesh variable symbols using ``sympy.diff()``.

    The derivative can be evaluated numerically via the standard evaluation
    functions, which use either Clement gradient recovery (fast, approximate)
    or L2 projection (accurate, requires solve).

    Notes
    -----
    Second derivatives are not currently supported. Attempting to differentiate
    an instance of this class will raise a RuntimeError.

    See Also
    --------
    UnderworldAppliedFunction : The parent class for undifferentiated functions.
    """
    def fdiff(self, argindex):
        """SymPy protocol: Second derivatives are not supported."""
        raise RuntimeError("Second derivatives of Underworld functions are not supported at this time.")

class UnderworldFunction(sympy.Function):
    """
    Metaclass that returns programmatic class objects rather than instances.

    This basically follows the pattern of the ``sympy.Function``
    metaclass, with two key differences. First, we set
    ``UnderworldAppliedFunction`` as the base class, which allows
    ``isinstance(someobj, UnderworldAppliedFunction)`` checks.
    Second, we grab a weakref of the owning meshvariable onto the
    class itself (not the instance), because SymPy internally uses
    ``type(obj)(obj.args)`` to clone instances and extra info must
    be on the class so that clones are complete.

    Consider the calling pattern

    >>> newfn = UnderworldFunction(meshvar,name)(*meshvar.mesh.r)

    This is equivalent to

    >>> newfnclass = UnderworldFunction(meshvar,name)   # Here we create a new *class*.
    >>> newfn = newfnclass(*meshvar.mesh.r)             # Here we create an instance of the class.

    Parameters
    ----------
    name : str
        The name of the function.
    meshvar : MeshVariable
        The mesh variable corresponding to this function.
    vtype : VarType
        The variable type (scalar, vector, etc).
    component : int or tuple
        For vector functions, this is the component of the vector.
        For example, component ``1`` might correspond to ``v_y``.
        For tensors, the component is a tuple.
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

    # 2b. Canonicalize coordinate symbols for lambdify.
    # The expression may contain UWCoordinate objects (from mesh.X or
    # mesh.CoordinateSystem.unit_e_0) alongside BaseScalar objects. Since
    # lambdify uses object identity to map arguments to generated code,
    # we must ensure only ONE set of coordinate objects appears.
    # Strategy: collect all coordinate-like symbols from the expression,
    # group by index, and replace all variants with a single canonical
    # sympy.Dummy symbol per coordinate. Use the same Dummy as the
    # lambdify argument.
    from sympy.vector.scalar import BaseScalar
    coord_dummies = [sympy.Dummy(f"_coord_{i}") for i in range(dim)]
    coord_subs = {}
    for sym in subbedexpr.free_symbols:
        if isinstance(sym, BaseScalar):
            idx = sym._id[0]
            if idx < dim:
                coord_subs[sym] = coord_dummies[idx]
    if coord_subs:
        subbedexpr = subbedexpr.xreplace(coord_subs)
    r = coord_dummies

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
                local_fallback=True,
            ):

    """
    Internal: Parallel-safe expression evaluation (Cython implementation).

    This is the low-level Cython implementation for MPI-parallel evaluation.
    Users should typically use :func:`underworld3.function.global_evaluate`
    which provides automatic unit handling and a cleaner interface.

    Contract: this is a faithful *parallel* counterpart of :func:`evaluate` —
    a query point is interpolated wherever in the mesh it lands (on any rank),
    a point just outside the mesh is extrapolated from its nearest cell (the
    globally nearest by a centroid-distance heuristic, with the lowest rank as
    the tie-break in parallel),
    and ``check_extrapolated`` returns an inside/outside flag per point. The
    result is independent of the number of ranks (up to the rank-local
    extrapolation residual near partition seams). Points that no rank can
    locate in-cell are resolved by a best-claim reduction over ranks (see the
    out-of-domain block below); pass ``local_fallback=False`` to restore the
    legacy behaviour where such points returned silently-wrong values. The
    ``GE_LOCAL_FALLBACK`` environment variable, if set, overrides the kwarg
    (an operator escape hatch retained from the parallel-deadlock debugging
    history; the kwarg is the supported control surface).

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
    coords_array = np.array(coords, dtype=np.float64, copy=False).view(np.ndarray)

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
    n_input_points = coords_array.shape[0]

    evaluation_swarm.migrate(remove_sent_points=True, delete_lost_points=False)
    local_coords = evaluation_swarm._particle_coordinates.array[...].reshape(-1,evaluation_swarm.cdim)
    values, extrapolated = evaluate_nd(expr, local_coords, rbf=rbf, evalf=evalf, verbose=verbose, check_extrapolated=True,)

    if local_coords.shape[0] > 0:
        data_container.array[...] = values[...]
        is_extrapolated.array[:,0,0] = extrapolated[:]

    # set rank to old values and migrate back
    evaluation_swarm._rank_var.array[...] = original_rank.array[...]

    # Bare bones migration - just move particles, no validation at all
    # in the BASIC swarm, dm.migrate does not care about whether points
    # lie inside the domain or not.

    evaluation_swarm.dm.migrate(remove_sent_points=True)
    uw.mpi.barrier()

    # Invalidate cached data after bare-bones dm.migrate —
    # particle count and values changed but Swarm.migrate() was bypassed.
    evaluation_swarm._invalidate_canonical_data()

    # Pre-allocate with NaN so the shape is always correct. If any points
    # are lost during the migration round-trip, they remain NaN rather than
    # causing a shape mismatch or returning uninitialised data.
    return_value = np.full((n_input_points,) + expr_shape, np.nan, dtype=np.float64)
    return_mask = np.full((n_input_points, 1, 1), True, dtype=bool)

    n_returned = original_index.array.shape[0]
    if n_returned > 0:
        index = original_index.array[:, 0, 0].astype(int)
        return_value[index, :, :] = data_container.array[:, :, :]
        return_mask[index] = is_extrapolated.array[:]

    # ------------------------------------------------------------------
    # Out-of-domain extrapolation — keep the parallel result a faithful
    # match for the serial ``evaluate()`` contract: interpolate a point
    # wherever it lands across ranks, extrapolate a point just outside the
    # mesh, and flag inside/outside.
    #
    # After the migrate round-trip, a query point that NO rank could locate
    # in one of its cells returns flagged-extrapolated but valued from
    # whichever rank the bare dm.migrate happened to strand it on — typically
    # a geometrically far, WRONG cell (the classic symptom is an annulus
    # boundary point reading a value from the opposite side of the domain).
    # Serial ``evaluate()`` instead extrapolates from the TRUE nearest cell.
    # Restore that contract with a "best-claim" reduction over the (small,
    # boundary-layer) stranded set:
    #
    #   1. allgather the extrapolated points so every rank holds the SAME
    #      global set;
    #   2. each rank reports, per point, its nearest-local-cell distance and
    #      its LOCAL rbf extrapolation of the field there;
    #   3. Allreduce(MIN distance) + Allreduce(MIN rank) tie-break picks the
    #      rank whose nearest cell is globally closest, and Allreduce(SUM of
    #      the winner-only value/flag) scatters that rank's extrapolation back.
    #
    # A point some rank actually contains (distance ~ 0) naturally wins, so
    # only genuinely-stranded points are corrected. Cost is O(boundary points)
    # — no dense global tree, no exhaustive search.
    #
    # DEADLOCK SAFETY — read before editing. Every collective here (allgather,
    # Allreduce) runs unconditionally on the IDENTICAL global set on every
    # rank, so all ranks stay in lockstep (n_ext_total is itself a reduced
    # value, so the `> 0` guard is taken identically everywhere). The per-rank
    # value MUST come from the LOCAL rbf path (rbf=True): the FE interpolation
    # path (petsc_interpolate / DMInterpolation) is itself collective and would
    # desync here, because each rank classifies the same global set against its
    # own domain (different interior-point counts) → hang. Never route the
    # fallback value through FE interpolation.
    #
    # Serial is left untouched (the serial path above already extrapolates from
    # the true nearest cell). Escape hatch: GE_LOCAL_FALLBACK=0 restores the
    # legacy (silently-wrong out-of-domain) behaviour; default on.
    # ------------------------------------------------------------------
    import os
    # The kwarg is the supported control; an explicitly-set env var overrides
    # it (operator escape hatch — see DEADLOCK SAFETY / contract docstring).
    _env_fallback = os.environ.get("GE_LOCAL_FALLBACK")
    if _env_fallback is not None:
        _local_fallback = _env_fallback.strip().lower() not in ("0", "off", "false", "no", "")
    else:
        _local_fallback = bool(local_fallback)
    if uw.mpi.size > 1 and _local_fallback:
        from mpi4py import MPI

        comm = uw.mpi.comm
        ext_idx = np.where(return_mask[:, 0, 0])[0]
        ext_coords = np.ascontiguousarray(coords_array[ext_idx], dtype=np.float64)

        counts = np.array(comm.allgather(ext_coords.shape[0]), dtype=int)
        n_ext_total = int(counts.sum())

        if n_ext_total > 0:
            parts = comm.allgather(ext_coords)
            all_ext = np.concatenate(
                [p for p in parts if p.size], axis=0).reshape(n_ext_total, -1)

            # This rank's local rbf extrapolation of the global set. NON-collective
            # value path — see DEADLOCK SAFETY above (must be rbf=True, never FE).
            ext_vals, ext_flag = evaluate_nd(
                expr, all_ext, rbf=True, evalf=False, verbose=False,
                check_extrapolated=True,)
            ext_vals = np.ascontiguousarray(
                np.asarray(ext_vals, dtype=np.float64).reshape((n_ext_total,) + expr_shape))
            ext_flag = np.asarray(ext_flag).reshape(n_ext_total).astype(np.int32)

            # Nearest-local-cell distance for every point (local kd-tree query).
            mesh._build_kd_tree_index()
            dist2, _ = mesh._centroid_index.query(all_ext, k=1, sqr_dists=True)
            dist2 = np.ascontiguousarray(np.asarray(dist2, dtype=np.float64).ravel())

            # Globally-nearest cell per point, lowest rank as the tie-break.
            min_dist2 = np.empty(n_ext_total, dtype=np.float64)
            comm.Allreduce([dist2, MPI.DOUBLE], [min_dist2, MPI.DOUBLE], op=MPI.MIN)
            my_claim = np.where(dist2 <= min_dist2 * (1.0 + 1e-12) + 1e-300,
                                comm.rank, comm.size).astype(np.int32)
            win_rank = np.empty(n_ext_total, dtype=np.int32)
            comm.Allreduce([my_claim, MPI.INT], [win_rank, MPI.INT], op=MPI.MIN)
            i_win = (win_rank == comm.rank)

            # Winner contributes value+flag, everyone else zero; SUM selects it.
            contrib_val = np.ascontiguousarray(
                np.where(i_win[:, None, None], ext_vals, 0.0))
            best_val = np.empty_like(contrib_val)
            comm.Allreduce([contrib_val, MPI.DOUBLE], [best_val, MPI.DOUBLE], op=MPI.SUM)
            contrib_flag = np.where(i_win, ext_flag, 0).astype(np.int32)
            best_flag = np.empty(n_ext_total, dtype=np.int32)
            comm.Allreduce([contrib_flag, MPI.INT], [best_flag, MPI.INT], op=MPI.SUM)

            # Scatter this rank's segment of the global set back to its points.
            offset = int(counts[:comm.rank].sum())
            seg = slice(offset, offset + ext_coords.shape[0])
            if ext_idx.size:
                return_value[ext_idx, :, :] = best_val[seg]
                return_mask[ext_idx, 0, 0] = best_flag[seg].astype(bool)

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
    import sympy

    # Handle matrix/tensor expressions — need a multi-component projection.
    # Implementation: project the (rows, cols) expression into a flat
    # (1, Nc) MATRIX work variable using SNES_MultiComponent_Projection
    # (one solve, shared DM), then fan the result out into a (rows, cols)
    # shaped work variable so the caller's ``work_var.sym`` preserves the
    # original tensor shape.
    if hasattr(expr, 'shape') and expr.shape != (1, 1):
        rows, cols = expr.shape
        n_components = rows * cols

        shape_key = f'{rows}x{cols}'
        cache_key = f'_eval_work_{shape_key}'
        flat_key = f'{cache_key}_flat'
        projector_key = f'{cache_key}_projector'

        if not hasattr(mesh, cache_key):
            # Tensor-shaped result variable that the caller sees via .sym
            if rows == mesh.dim and cols == mesh.dim:
                vtype_out = uw.VarType.TENSOR
            else:
                vtype_out = uw.VarType.MATRIX
            work_var = uw.discretisation.MeshVariable(
                cache_key,
                mesh,
                (rows, cols),
                vtype=vtype_out,
                degree=1,
            )
            setattr(mesh, cache_key, work_var)

            # Flat (1, Nc) target for the multi-component projector
            flat_var = uw.discretisation.MeshVariable(
                flat_key,
                mesh,
                (1, n_components),
                vtype=uw.VarType.MATRIX,
                degree=1,
            )
            setattr(mesh, flat_key, flat_var)

            projector = uw.systems.MultiComponent_Projection(
                mesh,
                u_Field=flat_var,
                n_components=n_components,
                degree=1,
            )
            projector.petsc_options["snes_rtol"] = 1e-6
            setattr(mesh, projector_key, projector)

        work_var = getattr(mesh, cache_key)
        flat_var = getattr(mesh, flat_key)
        projector = getattr(mesh, projector_key)

        # Build flat row-matrix source: [[expr[0,0], expr[0,1], ..., expr[r-1,c-1]]]
        flat_source = sympy.Matrix(
            [[expr[i, j] for i in range(rows) for j in range(cols)]]
        )
        projector.uw_function = flat_source
        projector.smoothing = smoothing
        # NB: a plain solve (not _force_setup=True) — the cached projector
        # refreshes correctly against changed field data on the current cache
        # machinery. Forcing a full DM+SNES rebuild every evaluate() re-introduced
        # the O(100 MiB) leak guarded by tests/test_0006_memory_leak.py. If a
        # genuine cache-staleness recurs, invalidate the cached projector
        # targetedly rather than rebuilding on every call (issue #215, Bug 2).
        projector.solve(zero_init_guess=False)

        # Fan flat result back to the tensor work variable
        for idx in range(n_components):
            i, j = divmod(idx, cols)
            work_var.array[:, i, j] = flat_var.array[:, 0, idx]

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
    # _force_setup=True: rebuild solver state to avoid stale cached
    # projector after Stokes/DM modifications (issue #215, Bug 2).
    projector.solve(zero_init_guess=False, _force_setup=True)

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
            # Higher degree - need to evaluate at node coords. Evaluate the
            # component's own applied function directly: var.sym is a row
            # matrix (1, cdim) for vectors (and (rows, cols) for tensors), so
            # indexing it by the flat data-column index is wrong / can raise.
            nodal_values[varfn] = np.asarray(uw.function.evaluate(
                varfn, node_coords, rbf=True
            )).flatten()

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

    # Add derivative values to nodal_values dictionary. Each derivative
    # expression carries its own flat data-column index (diffcls.component,
    # set at UnderworldFunction registration) — use it to retrieve the
    # matching per-component gradient, so e.g. v[1].diff(x) reads the
    # component-1 gradient rather than component 0.
    for source_var, deriv_list in derivfns.items():
        for deriv_expr, diffindex in deriv_list:
            grad = gradient_at_nodes[(source_var, deriv_expr.component)]  # (n_nodes, dim)
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
    coords_array = np.array(coords, dtype=np.float64, copy=False).view(np.ndarray)

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
        # CRITICAL: update_lvec() calls dm.globalToLocal() which is COLLECTIVE.
        # It MUST be called by ALL ranks before any rank enters petsc_interpolate,
        # because ranks with zero interior points would skip petsc_interpolate
        # (and its internal update_lvec call), deadlocking the ranks that do enter.
        mesh.update_lvec()

        # Interior (FE) vs exterior (RBF) classification. points_in_domain()
        # itself defers to the bulletproof barycentric locator on parallel
        # simplex/manifold meshes (mesh._eval_use_robust_location()), so on-
        # face / partition-seam / domain-boundary node points are classified
        # interior (FE path) rather than being rejected to rank-local RBF — the
        # same fix that lets swarm migration claim them. Serial / non-simplex
        # keep the cell-wall test (bit-identical). See
        # parallel-repeated-solve-corruption.md.
        #
        # The classification also hands back the cells it located on the way,
        # so petsc_interpolate does not look those points up again (#551
        # item 2).
        in_or_not, cell_hints = mesh._classify_points_in_domain(
            coords_array, strict_validation=False)
        evaluation_interior = petsc_interpolate( expr,
                                    coords_array[in_or_not],
                                    coord_sys,
                                    mesh,
                                    simplify=simplify,
                                    verbose=verbose,
                                    cell_hints=cell_hints[in_or_not], )

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
                verbose=False,
                cell_hints=None, ):
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
    cell_hints: numpy.ndarray, optional
        One owning cell index per coordinate, as returned by
        ``Mesh._classify_points_in_domain``: a cell the classification
        already located, or ``-1`` for "not looked up", which this function
        then locates itself. Supplying it means those points are not located
        twice. Hints must have been located against ``mesh``; hints for any
        other mesh in the expression are ignored and that mesh locates its
        own.

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
    if coords.dtype != np.float64:
        raise ValueError("Provided `coords` must be an array of doubles.")
    if other_arguments:
        raise RuntimeError("`other_arguments` functionality not yet implemented.")

    # NOTE: Do NOT early-return for empty coords here. petsc_interpolate
    # calls DMLocatePoints which is COLLECTIVE on the mesh DM communicator.
    # If some ranks skip it (empty coords) while others enter it, MPI deadlocks.
    # Empty coords are handled inside interpolate_vars_on_mesh after the
    # collective operations complete.

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

    # Any cell hints the caller supplied were located against THIS mesh; an
    # expression spanning two meshes must locate the second one itself.
    hinted_mesh = mesh

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


        # For now, eval over all vars.
        #
        # MATERIALISE THE LIST. ``mesh.vars`` is a weakref.WeakValueDictionary
        # and its ``.values()`` is a GENERATOR, not a view: the dofcount loop
        # below consumes it, and every later reader (the continuity gate, the
        # RBF fallback rung) then iterates an empty sequence. That silently
        # disabled the fallback — points the locator returns -1 for kept the
        # NaN written by DMInterpolationEvaluate_UW and handed it back to the
        # caller — and it silently pinned the continuity gate at True.
        vars = list(mesh.vars.values())

        cdef DM dm = mesh.dm

        # Get and set total count of dofs
        dofcount = 0
        var_start_index = {}
        for var in vars:
            var_start_index[var] = dofcount
            dofcount += var.num_components

        # Make coords contiguous for caching and C access
        coords = np.ascontiguousarray(coords)

        # NOTE: No early return for empty coords here. DMLocatePoints
        # (inside DMInterpolationSetUp_UW) is COLLECTIVE on the mesh DM
        # communicator. All ranks must participate, even with zero points.

        # === DMInterpolation CACHING ===
        # Declare variables at function scope (Cython requirement)
        cdef np.ndarray cells

        # Try to get cached structure first
        from underworld3.function._dminterp_wrapper import CachedDMInterpolationInfo

        # Location policy: is the cell-wall hint authoritative for THIS
        # evaluation? Decided by the mesh's measured capability (face
        # planarity + convexity, mesh._location_capability) combined with the
        # continuity of the variables being interpolated — "continuous"
        # capability (small-sagitta warped hexes, e.g. the cubed sphere) is
        # authoritative only when every field is continuous, because a
        # face-aligned jump inside the misclassification slab would see
        # O(jump) wrong-side errors. The policy participates in the cache key
        # so the same coords evaluated with a different field mix cannot
        # reuse a structure built under the other policy.
        #
        # The continuity test runs over the variables THIS CALL ASKED FOR,
        # not every variable on the mesh. The structure carries all of them
        # (dofcount above), but only the requested slices are read, and the
        # gate exists to protect a field whose jump sits on a cell face. Over
        # the whole mesh instead, one discontinuous variable anywhere would
        # take every evaluation off the authoritative path: measured on a
        # warped hex box (capability "continuous") carrying a P1 and a P0,
        # that costs the CONTINUOUS field a factor 15 in accuracy (linear
        # field, max error 8.0e-3 -> 1.2e-1 at 62 of 1500 interior points)
        # for no correctness gain. Scoped to the request, the continuous
        # field is bit-identical and only the P0 moves.
        #
        # NOTE this gate has never bound before: `vars` was an exhausted
        # generator (see above) so `all()` was vacuously True.
        all_continuous = all(
            getattr(varfn.meshvar(), "continuous", True) for varfn in varfns)
        authoritative = mesh._hint_is_authoritative(all_continuous)
        location_policy = "auth" if authoritative else "locate"

        # coords is already np.ndarray type in petsc_interpolate function signature
        cached_info = mesh._dminterpolation_cache.get_structure(
            coords, dofcount, policy=location_policy)

        # Create output array
        cdef np.ndarray outarray = np.empty([len(coords), dofcount], dtype=np.float64)

        if cached_info is not None:
            # CACHE HIT - Fast path. Evaluate using cached structure
            # swarm_sync=False: petsc_interpolate is reached by only the
            # ranks that hold interior points — the swarm-dependency hook
            # does collective reductions and must not run on a subset.
            # Freshness comes from the all-ranks update_lvec() in evaluate().
            mesh.update_lvec(swarm_sync=False)  # Ensure fresh values
            cached_info.evaluate(mesh, outarray)

        else:
            # CACHE MISS - Create structure and cache it
            cached_info = CachedDMInterpolationInfo()

            # Cell hints, by policy:
            # AUTHORITATIVE — the hint bypasses DMLocatePoints, so it has to
            # be a cell that CONTAINS the point. _robust_owning_cells is the
            # containment-checked locator: it returns a cell whose walls the
            # point is inside (any one of them, for a point on a shared face)
            # and -1 when no local cell contains it. Every authoritative mesh
            # takes the same route. Serial simplex meshes used to take the
            # nearest-CONTROL-POINT lookup (get_closest_cells) with no
            # containment test at all; on a tetrahedron the reference-coord
            # clamp downstream is a box clamp and cannot rescue that, so a
            # query on a shared edge was evaluated by extrapolating the basis
            # of a cell that does not contain it (#432, a recurrence of #390).
            # NOT AUTHORITATIVE — no hint at all: DMLocatePoints decides,
            # dropped points surface in unlocated_mask and are filled by the
            # RBF fallback below.
            #
            # Cells the caller's classification already located are reused;
            # only the ones it left at -1 are searched for, and only here, on
            # the cache miss that actually needs them. That is what makes it
            # one location per point per call rather than two.
            if authoritative:
                if cell_hints is not None and mesh is hinted_mesh:
                    # COPY: the unhinted entries are filled in below, and
                    # ascontiguousarray hands back the caller's own array when
                    # it is already int64 and contiguous. petsc_interpolate
                    # takes cell_hints as a documented keyword, so writing
                    # through it would mutate somebody else's array.
                    cells = np.array(cell_hints, dtype=np.int64, copy=True,
                                     order="C")
                    if cells.shape[0] != coords.shape[0]:
                        raise RuntimeError(
                            "cell_hints must carry one cell index per coordinate "
                            f"({cells.shape[0]} hints for {coords.shape[0]} points)."
                        )
                    unhinted = np.where(cells < 0)[0]
                    if unhinted.shape[0] > 0:
                        cells[unhinted] = mesh._robust_owning_cells(coords[unhinted])
                else:
                    cells = mesh._robust_owning_cells(coords)
            else:
                cells = None

            # Create and set up DMInterpolation structure (EXPENSIVE)
            # This calls DMLocatePoints which is COLLECTIVE — all ranks must enter.
            try:
                # coords is already np.ndarray type (function signature ensures this)
                cached_info.create_structure(mesh, coords, cells, dofcount,
                                             hint_authoritative=authoritative)
            except RuntimeError as e:
                # Handle DMInterpolationSetUp failures gracefully
                if "outside the domain" in str(e):
                    raise RuntimeError("Error encountered when trying to interpolate mesh variable.\n"
                                     "Interpolation location is possibly outside the domain.")
                else:
                    raise

            # Store in cache for reuse
            # coords is already np.ndarray type (function signature ensures this)
            mesh._dminterpolation_cache.store_structure(
                coords, dofcount, cached_info, policy=location_policy)

            # Evaluate
            # swarm_sync=False: see the cache-hit branch above — only a
            # subset of ranks reaches petsc_interpolate.
            mesh.update_lvec(swarm_sync=False)
            cached_info.evaluate(mesh, outarray)

        # RBF fallback rung: points no cell owns (dropped by DMLocatePoints
        # on a non-authoritative mesh, or hinted -1) hold NaN in outarray.
        # Fill them per-variable with the bounded, topology-free RBF
        # interpolant — the same machinery exterior points already use. NaN
        # survives only if this plumbing is bypassed, which is exactly when
        # it should be visible.
        unlocated = getattr(cached_info, "unlocated_mask", None)
        if unlocated is not None and unlocated.any():
            fallback_coords = coords[unlocated]
            for var in vars:
                var_start = var_start_index[var]
                rbf_vals = np.asarray(var.rbf_interpolate(fallback_coords))
                rbf_vals = rbf_vals.reshape(len(fallback_coords), var.num_components)
                outarray[unlocated, var_start:var_start + var.num_components] = rbf_vals
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
    if coords.dtype != np.float64:
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

## Swarm migration type utilities (PETSc DMSwarm interface)

def dm_swarm_get_migrate_type(swarm):
    """
    Get the migration type for a PETSc DMSwarm.

    The migration type controls how particles are transferred between
    MPI ranks during swarm migration operations.

    Parameters
    ----------
    swarm : Swarm
        The Underworld swarm object.

    Returns
    -------
    PETSc.DMSwarm.MigrateType
        The current migration type setting.

    See Also
    --------
    dm_swarm_set_migrate_type : Set the migration type.
    """
    mtype = _dmswarm_get_migrate_type(swarm.dm)

    return mtype

def dm_swarm_set_migrate_type(swarm, mtype):
    """
    Set the migration type for a PETSc DMSwarm.

    The migration type controls how particles are transferred between
    MPI ranks during swarm migration operations.

    Parameters
    ----------
    swarm : Swarm
        The Underworld swarm object.
    mtype : PETSc.DMSwarm.MigrateType
        The migration type to set.

    See Also
    --------
    dm_swarm_get_migrate_type : Get the current migration type.
    """
    _dmswarm_set_migrate_type(swarm.dm, mtype)

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
