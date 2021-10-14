import sympy
import numpy as np
import underworld3 
from typing import List
import underworld3.timing as timing


def diff_fn1_wrt_fn2(fn1, fn2):
    """
    This function takes the derivative of a function (fn1) with respect
    to another function (fn2). Sympy does not allow this natively, instead 
    only allowing derivatives with respect to symbols.  Here, we 
    temporarily subsitute fn2 for a dummy symbol, perform the derivative (with
    respect to the dummy symbol), and then replace the dummy for fn2 again. 
    """
    if fn2.is_zero:
        return 0
    # If fn1 doesn't contain fn2, immediately return zero.
    # The full diff method will also return zero, but will be slower.
    if len(fn1.atoms(fn2))==0:
        return 0
    uwderivdummy = sympy.Symbol("uwderivdummy")
    subfn   = fn1.xreplace({fn2:uwderivdummy})      # sub in dummy
    subfn_d = subfn.diff(uwderivdummy)              # actual deriv
    deriv   = subfn_d.xreplace({uwderivdummy:fn2})  # sub out dummy
    return deriv

_ext_dict = {}
@timing.routine_timer_decorator
def getext(mesh, fns_residual, fns_jacobian, fns_bcs, primary_field_list):
    """
    Check if we've already created an equivalent extension
    and use if available.
    """
    import time
    time_s = time.time()

    fns = tuple(fns_residual) + tuple(fns_jacobian) + tuple(fns_bcs)
    import os
    if 'UW_JITNAME' in os.environ:          # If var specified, probably testing.
        jitname = os.environ['UW_JITNAME']
        # Note, extensions cannot be replaced, so need to append count to ensure
        # unique modules.
        jitname += "_" + str(len(_ext_dict.keys()))
    else:                                   # Else name from fns hash
        jitname = abs(hash((mesh,fns)))
    # Create the module if not in dictionary
    if jitname not in _ext_dict.keys():
        _createext(jitname, mesh, fns_residual, fns_jacobian, fns_bcs, primary_field_list)
    module = _ext_dict[jitname]
    ptrobj = module.getptrobj()
    # print(f'jit time {time.time()-time_s}')
    return ptrobj

@timing.routine_timer_decorator
def _createext(name:               str, 
               mesh:               underworld3.mesh.MeshClass,
               fns_residual:       List[sympy.Basic], 
               fns_jacobian:       List[sympy.Basic],
               fns_bcs:            List[sympy.Basic],
               primary_field_list: List[underworld3.mesh.MeshVariable]):
    """
    This creates the required extension which houses the JIT
    fn pointer for PETSc. 

    Note that it is not possible to replace loaded shared libraries
    in Python, so we instead create a new extension for each new function. 

    We hash the functions and create a dictionary of the generated extensions
    to avoid redundantly creating new extensions.

    Params
    ------
    name:
        Name for the extension. It will be prepended with "fn_ptr_ext_"
    mesh:
        Supporting mesh. It is used to get coordinate system and variable
        information.
    fns_residual:
        List of system's residual sympy functions for which JIT equivalents 
        will be generated.
    fns_jacobian:
        List of system's Jacobian sympy functions for which JIT equivalents 
        will be generated.
    fns_bcs:
        List of system's boundary condition sympy functions for which JIT equivalents 
        will be generated.
    primary_field_list
        List of variables that will map from petsc primary variable arrays. All 
        other variables will be obtained from the mesh object and will be mapped to 
        petsc auxiliary variable arrays. Note that *all* the variables in the 
        calling system's corresponding `PetscDM` must be included in this list.
        They must also be ordered according to their `field_id`.

    """
    from sympy import symbols, Eq, MatrixSymbol
    from underworld3 import VarType

    # Note that the order here is important.
    fns = tuple(fns_residual) + tuple(fns_bcs) + tuple(fns_jacobian)
    count_residual_sig = len(fns_residual) + len(fns_bcs)
    count_jacobian_sig = len(fns_jacobian)

    # `_ccode` patching
    def ccode_patch_fns(varlist, prefix_str):
        """ 
        This function patches uw functions with the necessary ccode
        routines for the code printing.

        For a `varlist` consisting of 2d velocity & pressure variables, 
        for example, it'll generate routines which write the follow,
        where `prefix_str="petsc_u"`:
            V_x   : "petsc_u[0]"
            V_y   : "petsc_u[1]"
            P     : "petsc_u[2]"
            V_x_x : "petsc_u_x[0]"
            V_x_y : "petsc_u_x[1]"
            V_y_x : "petsc_u_x[2]"
            V_y_y : "petsc_u_x[3]"
            P_x   : "petsc_u_x[4]"
            P_y   : "petsc_u_x[5]"

        Params
        ------
        varlist: list
            The variables to patch. Note that *all* the variables in the 
            corresponding `PetscDM` must be included. They must also be 
            ordered according to their `field_id`.
        prefix_str: str
            The string prefix to write. 
        """ 
        u_i = 0    # variable increment
        u_x_i = 0  # variable gradient increment
        lambdafunc = lambda self,printer : self._ccodestr
        for var in varlist:
            if var.vtype==VarType.SCALAR:
                # monkey patch this guy into the function
                type(var.fn)._ccodestr = f"{prefix_str}[{u_i}]"
                type(var.fn)._ccode    = lambdafunc
                u_i +=1
                # now patch gradient guy into varfn guy
                for ind in range(mesh.dim):
                    # Note that var.fn._diff[ind] returns the class, so we don't need type(var.fn._diff[ind])
                    var.fn._diff[ind]._ccodestr = f"{prefix_str}_x[{u_x_i}]"
                    var.fn._diff[ind]._ccode    = lambdafunc
                    u_x_i+=1
            elif var.vtype==VarType.VECTOR:
                # Pull out individual sub components
                for bvec in mesh.N.base_vectors()[0:mesh.dim]:
                    comp = var.fn.dot(bvec)
                    # monkey patch
                    type(comp)._ccodestr = f"{prefix_str}[{u_i}]"
                    type(comp)._ccode    = lambdafunc
                    u_i +=1
                    # and also patch gradient guy into varfn guy's comp guy
                    for ind in range(mesh.dim):
                    # Note that var.fn._diff[ind] returns the class, so we don't need type(var.fn._diff[ind])
                        comp._diff[ind]._ccodestr = f"{prefix_str}_x[{u_x_i}]"
                        comp._diff[ind]._ccode    = lambdafunc
                        u_x_i+=1
            else:
                raise RuntimeError("Unsupported type for code generation. Please contact developers.")

    # Patch in `_code` methods. Note that the order here
    # is important, as the secondary call will overwrite 
    # those patched in the first call.
    ccode_patch_fns(mesh.vars.values(),"petsc_a")
    ccode_patch_fns(primary_field_list,"petsc_u")
    # Also patch `BaseScalar` types
    type(mesh.N.x)._ccode = lambda self,printer : f"petsc_x[{self._id[0]}]"


    # Create a custom functions replacement dictionary.
    # Note that this dictionary is really just to appease Sympy,
    # and the actual implementation is printed directly into the 
    # generated JIT files (see `h_str` below). Without specifying
    # this dictionary, Sympy doesn't code print the Heaviside correctly.
    # For example, it will print 
    #    Heaviside(petsc_x[0,1])
    # instead of 
    #    Heaviside(petsc_x[1]).
    # Note that the Heaviside implementation will be printed into all JIT 
    # files now. This is fine for now, but if more complex functions are 
    # required a cleaner solution might be desirable. 
    custom_functions = {
      "Heaviside": [ (lambda *args : len(args)==1, "Heaviside_1"),    # for single arg Heaviside  (defaults to 0.5 at jump).
                     (lambda *args : len(args)==2, "Heaviside_2")]    # for two arg Heavisides    (second arg is jump value).
    }

    # Now go ahead and generate C code from subsituted Sympy expressions.
    # from sympy.printing.c import C99CodePrinter
    # printer = C99CodePrinter(user_functions=custom_functions)
    from sympy.printing.c import c_code_printers
    printer = c_code_printers['c99']({"user_functions":custom_functions})

    # Purge libary/header dictionaries. These will be repopulated 
    # when `doprint` is called below. This ensures that we only link 
    # in libraries where needed. 
    # Note that this generally shouldn't be necessary, as the 
    # extension module should build successfully even where
    # libraries are linked in redundantly. However it does 
    # help to ensure that any potential linking issues are isolated
    # to only those sympy functions (just analytic solutions currently)
    # that require linking. There may also be a performance advantage
    # (faster extension build time) but this is unlikely to be 
    # significant.
    underworld3._incdirs.clear()
    underworld3._libdirs.clear()
    underworld3._libfiles.clear()

    eqns = []
    for index, fn in enumerate(fns):
        if isinstance(fn, sympy.vector.Vector):
            fn = fn.to_matrix(mesh.N)[0:mesh.dim,0]
        elif isinstance(fn, sympy.vector.Dyadic):
            fn = fn.to_matrix(mesh.N)[0:mesh.dim,0:mesh.dim]
        else:
            fn = sympy.Matrix([fn])
        out = sympy.MatrixSymbol("out",*fn.shape)
        eqn = ("eqn_"+str(index), printer.doprint(fn, out))
        if eqn[1].startswith("// Not supported in C:"):
            spliteqn = eqn[1].split("\n")
            raise RuntimeError(f"Error encountered generating JIT extension:\n"
                               f"{spliteqn[0]}\n"
                               f"{spliteqn[1]}\n"
                               f"This is usually because code generation for a Sympy function (or its derivative) is not supported.\n"
                               f"Please contact the developers.")
        eqns.append(eqn)
    MODNAME = "fn_ptr_ext_" + str(name)

    codeguys = []
    # Create a `setup.py`
    setup_py_str = """
try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension
from Cython.Build import cythonize

ext_mods = [Extension(
    '{NAME}', ['cy_ext.pyx',],
    include_dirs={HEADERS},
    library_dirs={LIBDIRS},
    runtime_library_dirs={LIBDIRS},
    libraries={LIBFILES},
    extra_compile_args=['-std=c99'],
    extra_link_args=[]
)]
setup(ext_modules=cythonize(ext_mods))
""".format(NAME=MODNAME,
           HEADERS=list(underworld3._incdirs.keys()),
           LIBDIRS=list(underworld3._libdirs.keys()),
           LIBFILES=list(underworld3._libfiles.keys()),
           )
    codeguys.append( ["setup.py", setup_py_str] )

    residual_sig = "(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar petsc_u[], const PetscScalar petsc_u_t[], const PetscScalar petsc_u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar petsc_a[], const PetscScalar petsc_a_t[], const PetscScalar petsc_a_x[], PetscReal petsc_t,                           const PetscReal petsc_x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar out[])"
    jacobian_sig = "(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar petsc_u[], const PetscScalar petsc_u_t[], const PetscScalar petsc_u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar petsc_a[], const PetscScalar petsc_a_t[], const PetscScalar petsc_a_x[], PetscReal petsc_t, PetscReal petsc_u_tShift, const PetscReal petsc_x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar out[])"

    # Create header top content.
    h_str="""
typedef int PetscInt;
typedef double PetscReal;
typedef double PetscScalar;
typedef int PetscBool;
#include <math.h> 

// Adding missing function implementation 
static inline double Heaviside_1 (double x)                 { return x < 0 ? 0 : x > 0 ? 1 : 0.5;     };
static inline double Heaviside_2 (double x, double mid_val) { return x < 0 ? 0 : x > 0 ? 1 : mid_val; };

"""

    # Create cython top content.
    pyx_str="""
from underworld3.petsc_types cimport PetscInt, PetscReal, PetscScalar, PetscErrorCode, PetscBool, PetscDSResidualFn, PetscDSJacobianFn
from underworld3.petsc_types cimport PtrContainer
from libc.stdlib cimport malloc
from libc.math cimport *

cdef extern from "cy_ext.h" nogil:
"""

    # Generate a random string to prepend to symbol names. 
    # This is generally not required, but on some systems (depending
    # on how Python is configured to dynamically load libraries)
    # it avoids difficulties with symbol namespace clashing which 
    # results in only the first JIT module working (with all 
    # subsequent modules pointing towards the first's symbols).
    # Tags: RTLD_LOCAL, RTLD_Global, Gadi.
    import string
    import random
    randstr = ''.join(random.choices(string.ascii_uppercase, k = 5))

    # Print includes
    for header in printer.headers:
        h_str += "#include \"{}\"\n".format(header)

    h_str += "\n"
    # Print equations
    for eqn in eqns[0:count_residual_sig]:
        h_str  +="void {}_petsc_{}{}\n{{\n{}\n}}\n\n".format(randstr,eqn[0],residual_sig,eqn[1])
        pyx_str+="    void {}_petsc_{}{}\n".format(randstr,eqn[0],residual_sig)

    for eqn in eqns[count_residual_sig:]:
        h_str  +="void {}_petsc_{}{}\n{{\n{}\n}}\n\n".format(randstr,eqn[0],jacobian_sig,eqn[1])
        pyx_str+="    void {}_petsc_{}{}\n".format(randstr,eqn[0],jacobian_sig)

    codeguys.append( ["cy_ext.h", h_str] )
    # Note that the malloc below will cause a leak, but it's just a bunch of function
    # pointers so we don't need to worry about it
    pyx_str+="""
cpdef PtrContainer getptrobj():
    clsguy = PtrContainer()
    clsguy.fns_residual = <PetscDSResidualFn*> malloc({}*sizeof(PetscDSResidualFn))  
    clsguy.fns_jacobian = <PetscDSJacobianFn*> malloc({}*sizeof(PetscDSJacobianFn))
    clsguy.fns_bcs      = <PetscDSResidualFn*> malloc({}*sizeof(PetscDSResidualFn))  
""".format(len(fns_residual),count_jacobian_sig,len(fns_bcs)) 

    for index,eqn in enumerate(eqns[0:len(fns_residual)]):
        pyx_str+="    clsguy.fns_residual[{}] = {}_petsc_{}\n".format(index,randstr,eqn[0])
    for index,eqn in enumerate(eqns[count_residual_sig:]):
        pyx_str+="    clsguy.fns_jacobian[{}] = {}_petsc_{}\n".format(index,randstr,eqn[0])
    for index,eqn in enumerate(eqns[len(fns_residual):count_residual_sig]):
        pyx_str+="    clsguy.fns_bcs[{}] = {}_petsc_{}\n".format(index,randstr,eqn[0])
    pyx_str +="    return clsguy"
    codeguys.append( ["cy_ext.pyx", pyx_str] )

    # Write out files
    import os
    tmpdir = os.path.join("/tmp",MODNAME)
    try:
        os.mkdir(tmpdir)
    except OSError:
        pass
    for thing in codeguys:
        filename = thing[0]
        strguy   = thing[1]
        with open(os.path.join(tmpdir,filename),'w') as f:
            f.write(strguy)

    # Build
    import subprocess
    process = subprocess.Popen('python3 setup.py build_ext --inplace'.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=tmpdir)
    process.communicate()

    # Load and add to dictionary
    from importlib._bootstrap import _load
    def load_dynamic(name, path, file=None):
        """
        Load an extension module.
        Borrowed from:
            https://stackoverflow.com/a/55172547
        """
        import importlib.machinery
        loader = importlib.machinery.ExtensionFileLoader(name, path)

        # Issue #24748: Skip the sys.modules check in _load_module_shims
        # always load new extension
        spec = importlib.machinery.ModuleSpec(
            name=name, loader=loader, origin=path)
        return _load(spec)

    for _file in os.listdir(tmpdir):
        if _file.endswith(".so"): 
            _ext_dict[name] = load_dynamic(MODNAME, os.path.join(tmpdir,_file))

    if name not in _ext_dict.keys():
        raise RuntimeError(f"The Underworld extension module does not appear to have been built successfully. "
                           f"The generated module may be found at:\n    {str(tmpdir)}\n"
                           f"To investigate, you may attempt to build it manually by running\n"
                           f"    python3 setup.py build_ext --inplace\n"
                           f"from the above directory. Note that a new module will always be written by "
                           f"Underworld and therefore any modifications to the above files will not persist into "
                           f"your Underworld runtime.\n"
                           f"Please contact the developers if you are unable to resolve the issue.")

