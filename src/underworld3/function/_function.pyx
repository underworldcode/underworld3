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

        ourcls = sympy.core.function.UndefinedFunction(fname,*args, bases=(UnderworldAppliedFunction,), **options)
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
            diffcls = sympy.core.function.UndefinedFunction(difffname, *args, bases=(UnderworldAppliedFunctionDeriv,), **options)
            # Grab weakref to var for derivative fn too.
            diffcls.meshvar   = weakref.ref(meshvar)
            diffcls.component = data_loc
            diffcls.diffindex = index
            ourcls._diff.append(diffcls)

        return ourcls

def global_evaluate_nd(   expr,
                coords=None,
                coord_sys=None,
                other_arguments=None,
                simplify=True,
                verbose=False,
                evalf=False,
                rbf=False,
                data_layout=None,
                check_extrapolated=False
            ):

    """
    Evaluate a given expression at a list of coordinates - parallel-safe version

    Note it is not efficient to call this function to evaluate an expression at
    a single coordinate. Instead the user should provide a numpy array of all
    coordinates requiring evaluation.

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
    coords_array = np.asarray(coords, dtype=np.double)

    mesh, varfns = uw.function.expressions.mesh_vars_in_expression(expr)

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

def evaluate_nd(   expr,
                coords=None,
                coord_sys=None,
                other_arguments=None,
                simplify=True,
                verbose=False,
                evalf=False,
                rbf=False,
                data_layout=None,
                check_extrapolated=False):
    """
    Evaluate a given expression at a list of coordinates.

    Note it is not efficient to call this function to evaluate an expression at
    a single coordinate. Instead the user should provide a numpy array of all
    coordinates requiring evaluation.

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
    coords_array = np.asarray(coords, dtype=np.double)

    dim = coords_array.shape[1]
    mesh, varfns = uw.function.fn_mesh_vars_in_expression(expr)

    # coercion - make everything at least a 1x1 matrix for consistent evaluation results
    try:
        expr.shape
    except AttributeError:
        expr = sympy.Matrix(((expr,),))

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



    ## Substitute any UWExpressions for their values before calculation
    ## NOTE: We use _unwrap_expressions directly (not fn_substitute_expressions) to avoid
    ## applying scaling transformations which would cause double-scaling since PETSc
    ## already stores non-dimensional values
    expr = uw.function.expressions._unwrap_expressions(expr, keep_constants=False)

    if simplify:
        expr = sympy.simplify(expr)

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
    #

    varfns = set()
    if mesh is not None and mesh.vars is not None:
        for v in mesh.vars.values():
            for sub in v.sym:
                varfns.add(sub)

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
        # vars = mesh.vars.values()
        # Now construct and perform the PETSc evaluate of these variables
        # Use MPI_COMM_SELF as following uw2 paradigm, interpolations will be local.
        # TODO: Investigate whether it makes sense to default to global operations here.

        cdef DMInterpolationInfo ipInfo
        cdef PetscErrorCode ierr
        ierr = DMInterpolationCreate(MPI_COMM_SELF, &ipInfo); CHKERRQ(ierr)
        ierr = DMInterpolationSetDim(ipInfo, mesh.dim); CHKERRQ(ierr)

        # Get and set total count of dofs
        dofcount = 0
        var_start_index = {}
        for var in vars:
            var_start_index[var] = dofcount
            dofcount += var.num_components

        ierr = DMInterpolationSetDof(ipInfo, dofcount); CHKERRQ(ierr)

        # Add interpolation points
        # Get c-pointer to data buffer
        # First grab copy, as we're unsure about the underlying array's
        # memory layout

        coords = np.ascontiguousarray(coords)
        cdef double* coords_buff = <double*> coords.data
        ierr = DMInterpolationAddPoints(ipInfo, coords.shape[0], coords_buff); CHKERRQ(ierr)

        # Generate a vector to hold the interpolation results.
        # First create a numpy array of the required size.
        cdef np.ndarray outarray = np.empty([len(coords), dofcount], dtype=np.double)
        # Now create a PETSc vector to wrap the numpy memory.
        cdef Vec outvec = PETSc.Vec().createWithArray(outarray,comm=PETSc.COMM_SELF)

        # INTERPOLATE ALL VARIABLES ON THE DM

        # grab closest cells to use as hint for DMInterpolationSetUp
        cdef np.ndarray cells = mesh.get_closest_cells(coords)
        cdef long unsigned int* cells_buff = <long unsigned int*> cells.data
        ierr = DMInterpolationSetUp_UW(ipInfo, dm.dm, 0, 0, <size_t*> cells_buff)

        if ierr != 0:
            raise RuntimeError("Error encountered when trying to interpolate mesh variable.\n"
                               "Interpolation location is possibly outside the domain.")
        mesh.update_lvec()
        cdef Vec pyfieldvec = mesh.lvec
        # Use our custom routine as the PETSc one is broken.

        ierr = DMInterpolationEvaluate_UW(ipInfo, dm.dm, pyfieldvec.vec, outvec.vec);CHKERRQ(ierr)
        ierr = DMInterpolationDestroy(&ipInfo);CHKERRQ(ierr)

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

        del outarray
        del coords
        del cells
        outvec.destroy()

        return varfns_arrays


    # Get map of all variable functions
    interpolated_results = {}
    for key, vals in interpolant_varfns.items():
        interpolated_var_values = interpolate_vars_on_mesh(vals, coords)
        interpolated_results.update(interpolated_var_values)

    # 3. Replace mesh variables in the expression with sympy symbols
    # First generate random string symbols to act as proxies.
    import string
    import random
    varfns_symbols = {}
    for varfn in interpolated_results.keys():
        randstr = ''.join(random.choices(string.ascii_uppercase, k = 5))
        varfns_symbols[varfn] = sympy.Symbol(randstr)

    # subs variable fns in expression for symbols
    subbedexpr = expr.subs(varfns_symbols)

    # 4. Generate sympy lambdified expression
    from sympy import lambdify
    from sympy.vector import CoordSys3D
    dim = coords.shape[1]

    ## Careful - if we change the names of the base-scalars for the mesh, this will need to be kept in sync

    if coord_sys is not None:
        N = coord_sys
    elif mesh is None:
        N = CoordSys3D(f"N")
    else:
        N = mesh.N

    r = N.base_scalars()[0:dim]

    # This likely never applies any more
    if isinstance(subbedexpr, sympy.vector.Vector):
        subbedexpr = subbedexpr.to_matrix(N)[0:dim,0]
    elif isinstance(subbedexpr, sympy.vector.Dyadic):
        subbedexpr = subbedexpr.to_matrix(N)[0:dim,0:dim]

    lambfn = lambdify( (r, varfns_symbols.values()), subbedexpr, docstring_limit=0 )
    # Leave out modules. This is equivalent to SYMPY_DECIDE and can then include scipy if available

    # 5. Eval generated lambda expression
    coords_list = [ coords[:,i] for i in range(dim) ]

    results = lambfn( coords_list, interpolated_results.values() )

    try:
        shape = expr.shape
    except AttributeError:
        shape = (1,1)
        expr = sympy.Matrix(((expr,)))

    try:
        results_shape = results.shape
    except AttributeError:
        results_shape = (1,1)

    # If passed a constant / constant matrix, then the result will not span the coordinates
    # and we'll need to broadcast the information explicitly

    if shape == results_shape:
        results_new = np.zeros((coords.shape[0], *shape))
        results_new[...] = results
        results = results_new
    else:
        results = np.moveaxis(results, -1, 0)
    # 6. Return results
    #

    return results.reshape(-1, *shape)

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


    # 2. Evaluate all mesh variables - there is no real
    # computational benefit in interpolating a subset.
    #

    varfns = set()
    if mesh is not None and mesh.vars is not None:
        for v in mesh.vars.values():
            for sub in v.sym:
                varfns.add(sub)

    # Get map of all variable functions (no cache)
    interpolated_results = {}
    for varfn in varfns:
        parent, component = uw.discretisation.meshVariable_lookup_by_symbol(mesh, varfn)
        values = parent.rbf_interpolate(coords, nnn=mesh.dim+1)[:,component]
        interpolated_results[varfn] = values
        if verbose:
            print(f"{varfn} = {parent.name}[{component}]")

    # 3. Replace mesh variables in the expression with sympy symbols
    # First generate random string symbols to act as proxies.

    import string
    import random
    varfns_symbols = {}
    for varfn in interpolated_results.keys():
        randstr = ''.join(random.choices(string.ascii_uppercase, k = 5))
        varfns_symbols[varfn] = sympy.Symbol(randstr)
    # subs variable fns in expression for symbols
    subbedexpr = expr.subs(varfns_symbols)

    # 4. Generate sympy lambdified expression
    from sympy import lambdify
    from sympy.vector import CoordSys3D
    dim = coords.shape[1]

    ## Careful - if we change the names of the base-scalars for the mesh, this will need to be kept in sync

    if coord_sys is not None:
        N = coord_sys
    elif mesh is None:
        from sympy.vector import CoordSys3D
        N = CoordSys3D(f"N")
    else:
        N = mesh.N

    r = N.base_scalars()[0:dim]
    lambfn = lambdify( (r, varfns_symbols.values()), subbedexpr, docstring_limit=0 )

    # 5. Eval generated lambda expression
    coords_list = [ coords[:,i] for i in range(dim) ]
    results = lambfn( coords_list, interpolated_results.values())

    # Check shape of original expression

    try:
        shape = expr.shape
    except AttributeError:
        shape = (1,1)
        expr = sympy.Matrix(((expr,)))

    try:
        results_shape = results.shape
    except AttributeError:
        results_shape = (1,)

    # If passed a constant / constant matrix, then the result will not span the coordinates
    # and we'll need to broadcast the information explicitly

    if shape == results_shape:
        results_new = np.zeros((coords.shape[0], *shape))
        results_new[...] = results
        results = results_new

    else:
        results = np.moveaxis(results, -1, 0)

    # 6. Return results

    return results


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
