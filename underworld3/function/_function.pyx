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

# Make Cython aware of this type.
cdef extern from "petsc.h" nogil:
    ctypedef struct DMInterpolationInfo:
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


def evaluate( expr, np.ndarray coords=None, coord_sys=None, other_arguments=None ):
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

    # 1. Extract UW variables.
    # Let's first collect all the meshvariables present in the expression and check
    # them for validity. This is applied recursively across the expression
    # Recurse the expression tree.

    # varfns = set()
    # def get_var_fns(exp):

    #     if isinstance(exp,uw.function._function.UnderworldAppliedFunctionDeriv):
    #         raise RuntimeError("Derivative functions are not handled in evaluations, a projection should be used first to create a mesh Variable.")

    #     isUW = isinstance(exp, uw.function._function.UnderworldAppliedFunction)
    #     if isUW:
    #         varfns.add(exp)
    #         if exp.args != exp.meshvar().mesh.r:
    #             raise RuntimeError(f"Mesh Variable functions can only be evaluated as functions of '{exp.meshvar().mesh.r}'.\n"
    #                                f"However, mesh variable '{exp.meshvar().name}' appears to take the argument {exp.args}." )
    #     else:
    #         # Recurse.
    #         for arg in exp.args:
    #             get_var_fns(arg)

    #     return

    # get_var_fns(expr)

    varfns = set()
    def unpack_var_fns(exp):

        if isinstance(exp,uw.function._function.UnderworldAppliedFunctionDeriv):
            raise RuntimeError("Derivative functions are not handled in evaluations, a projection should be used first to create a mesh Variable.")

        isUW = isinstance(exp, uw.function._function.UnderworldAppliedFunction)
        isMatrix = isinstance(exp, sympy.Matrix)

        if isUW:
            varfns.add(exp)
            if exp.args != exp.meshvar().mesh.r:
                raise RuntimeError(f"Mesh Variable functions can only be evaluated as functions of '{exp.meshvar().mesh.r}'.\n"
                                   f"However, mesh variable '{exp.meshvar().name}' appears to take the argument {exp.args}." )
        elif isMatrix:
            for sub_exp in exp:
                if isinstance(sub_exp, uw.function._function.UnderworldAppliedFunction):
                    varfns.add(sub_exp)
        else:
            # Recurse.
            for arg in exp.args:
                unpack_var_fns(arg)

        return

    unpack_var_fns(expr)

    # Check the same mesh is used for all mesh variables
    mesh = None
    for varfn in varfns:

        if mesh is None:
            mesh = varfn.meshvar().mesh
            #mesh = varfn.mesh
        else:
            if mesh != varfn.meshvar().mesh:
                raise RuntimeError("In this expression there are functions defined on different meshes. This is not supported")

    if (len(varfns)==0) and (coords is None):
        raise RuntimeError("Interpolation coordinates not specified by supplied expression contains mesh variables.\n"
                           "Mesh variables can only be interpolated at coordinates.")

    if mesh is not None:
        varfns = set()
        for var in mesh.vars.values():
            for subvar in var.sym_1d:
                varfns.add(subvar)

    # Create dictionary which creates a per mesh list of vars.
    # Usually there will only be a single mesh, but this allows for the
    # more general situation.

    from collections import defaultdict
    interpolant_varfns = defaultdict(lambda : [])

    for varfn in varfns:
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

            if coord_hash == mesh._evaluation_hash:
                # if uw.mpi.rank == 0:
                #     print("Using evaluation cache", flush=True)
                return mesh._evaluation_interpolated_results
            else:
                # if uw.mpi.rank == 0:
                #     print("No evaluation cache", flush=True)
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


    # Get map of all variable functions across all meshes.
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

    lambfn = lambdify( (r, varfns_symbols.values()), subbedexpr, 'numpy' )

    # 5. Eval generated lambda expression
    coords_list = [ coords[:,i] for i in range(dim) ]
    results = lambfn( coords_list, interpolated_results.values() )

    # Truncated out middle index for vector results
    if isinstance(results,np.ndarray):
        results = results.T
        if len(results.shape)==3 and results.shape[1]==1:
            results = results[:,0,:]

    # Making this explicit while hunting for leaks
    for array_item in interpolated_results.values():
        del array_item

    del interpolated_results

    # 6. Return results
    return results

# Go ahead and substitute for the timed version.
# Note that we don't use the @decorator sugar here so that
# we can pass in the `class_name` parameter.
evaluate = timing.routine_timer_decorator(routine=evaluate, class_name="Function")


### ------------------------------

def evalf( expr, coords, coord_sys=None,  other_arguments=None, verbose=False):
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

    if not (isinstance( expr, sympy.Basic ) or isinstance( expr, sympy.Matrix )):
        raise RuntimeError("`evaluate()` function parameter `expr` does not appear to be a sympy expression.")

    if other_arguments:
        raise RuntimeError("`other_arguments` functionality not yet implemented.")

    # 1. Extract UW variables.

    # Let's first collect all the meshvariables present in the expression and check
    # them for validity. This is applied recursively across the expression
    # Recurse the expression tree.

    varfns = set()
    def unpack_var_fns(exp):

        if isinstance(exp,uw.function._function.UnderworldAppliedFunctionDeriv):
            raise RuntimeError("Derivative functions are not handled in evaluations, a projection should be used first to create a mesh Variable.")

        isUW = isinstance(exp, uw.function._function.UnderworldAppliedFunction)
        isMatrix = isinstance(exp, sympy.Matrix)

        if isUW:
            varfns.add(exp)
            if exp.args != exp.meshvar().mesh.r:
                raise RuntimeError(f"Mesh Variable functions can only be evaluated as functions of '{exp.meshvar().mesh.r}'.\n"
                                   f"However, mesh variable '{exp.meshvar().name}' appears to take the argument {exp.args}." )
        elif isMatrix:
            for sub_exp in exp:
                if isinstance(sub_exp, uw.function._function.UnderworldAppliedFunction):
                    varfns.add(sub_exp)
        else:
            # Recurse.
            for arg in exp.args:
                unpack_var_fns(arg)

        return

    unpack_var_fns(expr)

    mesh = None
    for varfn in varfns:

        if mesh is None:
            mesh = varfn.mesh
        else:
            if mesh != varfn.mesh:
                raise RuntimeError("In this expression there are functions defined on different meshes. This is not supported")

    # 2. Evaluate all mesh variables - there is no real
    # computational benefit in interpolating a subset.

    # Get map of all variable functions from the cache
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
        N = CoordSys3D(f"N")
    else:
        N = mesh.N

    r = N.base_scalars()[0:dim]

    # # This likely never applies any more
    # if isinstance(subbedexpr, sympy.vector.Vector):
    #     subbedexpr = subbedexpr.to_matrix(N)[0:dim,0]
    # elif isinstance(subbedexpr, sympy.vector.Dyadic):
    #     subbedexpr = subbedexpr.to_matrix(N)[0:dim,0:dim]

    lambfn = lambdify( (r, varfns_symbols.values()), subbedexpr, 'numpy' )

    # 5. Eval generated lambda expression
    coords_list = [ coords[:,i] for i in range(dim) ]
    results = lambfn( coords_list, interpolated_results.values() )

    # 6. Return results
    return results.reshape(-1)

# Go ahead and substitute for the timed version.
# Note that we don't use the @decorator sugar here so that
# we can pass in the `class_name` parameter.
evalf = timing.routine_timer_decorator(routine=evalf, class_name="Function")

# This is the interpolation routine used in function-evaluation above

# def _interpolate_all_vars_on_mesh( mesh, np.ndarray coords ):
#     """
#     This function performs the interpolation for the mesh variables
#     on a single mesh.
#     """
#     # Grab the mesh

#     mvars = mesh.vars.values()
#     cdef DM dm = mesh.dm

#     # Now construct and perform the PETSc evaluate of these variables
#     # Use MPI_COMM_SELF as following uw2 paradigm, interpolations will be local.
#     # TODO: Investigate whether it makes sense to default to global operations here.

#     cdef DMInterpolationInfo ipInfo
#     cdef PetscErrorCode ierr
#     ierr = DMInterpolationCreate(MPI_COMM_SELF, &ipInfo); CHKERRQ(ierr)
#     ierr = DMInterpolationSetDim(ipInfo, mesh.dim); CHKERRQ(ierr)

#     # Get and set total count of dofs
#     dofcount = 0
#     var_start_index = {}
#     for var in mvars:
#         var_start_index[var] = dofcount
#         dofcount += var.num_components

#     ierr = DMInterpolationSetDof(ipInfo, dofcount); CHKERRQ(ierr)

#     # Add interpolation points
#     # Get c-pointer to data buffer
#     # First grab copy, as we're unsure about the underlying array's
#     # memory layout

#     coords = np.ascontiguousarray(coords.copy())
#     cdef double* coords_buff = <double*> coords.data
#     ierr = DMInterpolationAddPoints(ipInfo, coords.shape[0], coords_buff); CHKERRQ(ierr)

#     # Generate a vector to hold the interpolation results.
#     # First create a numpy array of the required size.
#     cdef np.ndarray outarray = np.empty([len(coords), dofcount], dtype=np.double)
#     # Now create a PETSc vector to wrap the numpy memory.
#     cdef Vec outvec = PETSc.Vec().createWithArray(outarray,comm=PETSc.COMM_SELF)

#     # INTERPOLATE ALL VARIABLES ON THE DM

#     # grab closest cells to use as hint for DMInterpolationSetUp
#     cdef np.ndarray cells = mesh.get_closest_cells(coords)
#     cdef long unsigned int* cells_buff = <long unsigned int*> cells.data
#     ierr = DMInterpolationSetUp_UW(ipInfo, dm.dm, 0, 0, <size_t*> cells_buff)
#     if ierr != 0:
#         raise RuntimeError(f"Error {ierr} encountered when trying to interpolate mesh variable.\n"
#                             "Interpolation location is possibly outside the domain.")
#     mesh.update_lvec()
#     cdef Vec pyfieldvec = mesh.lvec

#     # Use our custom routine as the PETSc one is broken.
#     ierr = DMInterpolationEvaluate_UW(ipInfo, dm.dm, pyfieldvec.vec, outvec.vec);CHKERRQ(ierr)
#     ierr = DMInterpolationDestroy(&ipInfo);CHKERRQ(ierr)

#     var_arrays = {}
#     for var in mesh.vars.values():
#             for comp, varfn in enumerate(var.sym):
#                 var_start = var_start_index[var] + comp
#                 arr = outarray[:,var_start:var_start+1]
#                 var_arrays[varfn] = arr.copy()

#     del outarray
#     del coords
#     del cells
#     del arr

#     outvec.destroy()

#     return var_arrays
