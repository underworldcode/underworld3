from petsc4py.PETSc cimport DM, PetscDM, Vec, PetscVec, MPI_Comm, PetscIS, IS
from ..petsc_types cimport PetscInt, PetscReal, PetscScalar, PetscErrorCode, PetscBool
from petsc4py import PETSc
import numpy as np
import sympy
import underworld3 as uw
from libc.stdlib cimport malloc, free
cimport numpy as np
import underworld3.timing as timing


cdef extern from "petsc_tools.h" nogil:
    PetscErrorCode DMInterpolationSetUp_UW(void *ipInfo, PetscDM dm, int petscbool, int petscbool, size_t* owning_cell)
    PetscErrorCode DMInterpolationEvaluate_UW(void *ipInfo, PetscDM dm, PetscVec x, PetscVec v)

cdef extern from "petsc.h" nogil:
    PetscErrorCode DMInterpolationCreate(MPI_Comm comm, void *ipInfo)
    PetscErrorCode DMInterpolationSetDim(void *ipInfo, PetscInt dim)
    PetscErrorCode DMInterpolationSetDof(void *ipInfo, PetscInt dof)
    PetscErrorCode DMInterpolationAddPoints(void *ipInfo, PetscInt n, PetscReal points[])
    PetscErrorCode DMInterpolationSetUp(void *ipInfo, PetscDM dm, int petscbool, int petscbool)
    PetscErrorCode DMInterpolationEvaluate(void *ipInfo, PetscDM dm, PetscVec vec, PetscVec out)
    PetscErrorCode DMInterpolationDestroy(void *ipInfo)
    PetscErrorCode DMDestroy(PetscDM *dm)
    PetscErrorCode DMCreateSubDM(PetscDM, PetscInt, const PetscInt *, PetscIS *, PetscDM *)
    MPI_Comm MPI_COMM_SELF

cdef CHKERRQ(PetscErrorCode ierr):
    cdef int interr = <int>ierr
    if ierr != 0: raise RuntimeError(f"PETSc error code '{interr}' was encountered.\nhttps://www.mcs.anl.gov/petsc/petsc-current/include/petscerror.h.html")


class UnderworldAppliedFunctionDeriv(sympy.core.function.AppliedUndef):
    """
    This is largely just to help us differentiate between UW
    and native Sympy functions.  
    """
    def fdiff(self,argindex):
        raise RuntimeError("Second derivatives of Underworld functions are not supported at this time.")

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

    """
    def __new__(cls, meshvar, component, *args, **options):
        ourcls = sympy.core.function.UndefinedFunction(*args, bases=(UnderworldAppliedFunction,), **options)
        # grab weakref to meshvar
        import weakref
        ourcls.meshvar = weakref.ref(meshvar)
        ourcls.component = component

        # go ahead and create the derivative function *classes*
        ourclsname = args[0]
        ourcls._diff = []
        ourcls._diff.append(sympy.core.function.UndefinedFunction(ourclsname+",x", *args[1:], bases=(UnderworldAppliedFunctionDeriv,), **options))
        ourcls._diff.append(sympy.core.function.UndefinedFunction(ourclsname+",y", *args[1:], bases=(UnderworldAppliedFunctionDeriv,), **options))
        ourcls._diff.append(sympy.core.function.UndefinedFunction(ourclsname+",z", *args[1:], bases=(UnderworldAppliedFunctionDeriv,), **options))

        return ourcls


def evaluate( expr, np.ndarray coords=None, other_arguments=None ):
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

    if not isinstance( expr, sympy.Basic ):
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
    # Let's first collect all the meshvariables present in the expression.
    # Recurse the expression tree.
    varfns = set()
    def get_var_fns(exp):
        if isinstance(exp,uw.function._function.UnderworldAppliedFunctionDeriv):
            raise RuntimeError("Derivatives functions are not handled yet unfortunately.")
        isUW = isinstance(exp, uw.function._function.UnderworldAppliedFunction)
        if isUW: 
            varfns.add(exp)
            if exp.args != exp.meshvar().mesh.r:
                raise RuntimeError(f"Mesh Variable functions can only be evaluated as functions of '{exp.meshvar().mesh.r}'.\n"
                                   f"However, mesh variable '{exp.meshvar().name}' appears to take the argument {exp.args}." )
        else:
            # Recurse.
            for arg in exp.args: get_var_fns(arg)
    get_var_fns(expr)

    if (len(varfns)==0) and (coords is None):
        raise RuntimeError("Interpolation coordinates not specified by supplied expression contains mesh variables.\n"
                           "Mesh variables can only be interpolated at coordinates.")

    # Create dictionary which creates a per mesh list of vars.
    # Usually there will only be a single mesh, but this allows for the
    # more general situation.
    from collections import defaultdict
    interpolant_varfns = defaultdict(lambda : [])
    for varfn in varfns:
        interpolant_varfns[varfn.meshvar().mesh].append(varfn)

    # 2. Evaluate mesh variables.
    def interpolate_vars_on_mesh( varfns, np.ndarray coords ):
        """
        This function performs the interpolation for the given variables
        on a single mesh.
        """
        # Grab the mesh
        mesh = varfns[0].meshvar().mesh
        # For now, eval over all vars
        vars = mesh.vars.values()
        cdef DM dm = mesh.dm
        # vars = mesh.vars.values()
        # Now construct and perform the PETSc evaluate of these variables
        # Use MPI_COMM_SELF as following uw2 paradigm, interpolations will be local.
        # TODO: Investigate whether it makes sense to default to global operations here.
        cdef void* ipInfo
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


        # SKIP THE SUBDM METHOD AND INSTEAD JUST INTERPOLATE ALL VARIABLES, FOR NOW.
        # # We'll create a SubDM which records the required fields. 
        # cdef PetscDM subdm
        # cdef PetscInt *fieldarray = <PetscInt *> malloc(len(vars)*sizeof(PetscInt))
        # cdef int i=0
        # for var in vars:
        #     fieldarray[i] = var.field_id
        #     i+=1
        # DMCreateSubDM(dm.dm, i, fieldarray, NULL, &subdm)
        # ierr = DMInterpolationSetUp_UW(ipInfo, subdm, 0, 0); CHKERRQ(ierr)

        # # Create a nestvec containing the variable data to be interpolated.
        # # First generate the required sub vecs.
        # vecs = []
        # for var in vars:
        #     vecs.append(mesh.lvec.getSubVector(IS().createStride(size=var.size, first=var.start, step=1)))
        # # Now generate the nest vec
        # nestvec = Vec().createNest(vecs)
        # cdef Vec pyfieldvec = nestvec
        # # Execute interpolation.
        # ierr = DMInterpolationEvaluate_UW(ipInfo, subdm, pyfieldvec.vec, outvec.vec);CHKERRQ(ierr)
        # # Clean up.
        # ierr = DMDestroy(&subdm);CHKERRQ(ierr)
        # free(fieldarray)

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
            arr = outarray[:,var_start+comp]
            varfns_arrays[varfn] = arr

        return varfns_arrays

    # Get map of all variable functions across all meshes. 
    interpolated_results = {}
    for key, vals in interpolant_varfns.items():
        interpolated_results.update(interpolate_vars_on_mesh(vals, coords))

    # 3. Replace mesh mesh variables in the expression with sympy symbols
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
    N = CoordSys3D("N")
    r = N.base_scalars()[0:dim]
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

    # 6. Return results
    return results

# Go ahead and substituate for the timed version.
# Note that we don't use the @decorator sugar here so that
# we can pass in the `class_name` parameter. 
evaluate = timing.routine_timer_decorator(routine=evaluate, class_name="Function")




    

#     def __new__(cls, mesh, *args, **kwargs):
#         # call the sympy __new__ method without args/kwargs, as unexpected args 
#         # trip up an exception.  
#         obj = super(MeshVariable, cls).__new__(cls, mesh.N.base_vectors()[0:mesh.dim])
#         return obj
    
#     @classmethod
#     def eval(cls, x):
#         return None

#     def _ccode(self, printer):
#         # return f"{type(self).__name__}({', '.join(printer._print(arg) for arg in self.args)})_bobobo"
#         return f"petsc_u[0]"


# class MeshVariableFn(sympy.Function):
#     _printstr = None 
#     _header = None
#     def _ccode(self, printer):
#         # This is just a copy paste from the analytic solutions implementation. 
#         # For usage within element assembly we generally won't need to use this as
#         # PETSc passes in evaluated variables via the (Benny-And-The) Jets 
#         # function. 
#         # However, if we wish to use a variable in a completely independent
#         # system, we may still need to provide an implementation here. We may also
#         # need to provide a code printed implementation for evaluation from within 
#         # Python via `evaluate()` (or equivalent). Alternatively, it might be sufficent
#         # to simply used sympy `evalf` methods, so that overloading type behaviours (+/-/*/etc)
#         # are handled by sympy within its evaluation tree, with c-level implementations 
#         # (such as `DMFieldEvaluate`) accessed at the nodes of the tree. 
#         raise RuntimeError("Not implemented.")
#         printer.headers.add(self._header)
#         param_str = ""
#         for arg in self.args:
#             param_str += printer._print(arg) + ","
#         param_str = param_str[:-1]  # drop final comma
#         if not self._printstr:
#             raise RuntimeError("Trying to print unprintable function.")
#         return self._printstr.format(param_str)
