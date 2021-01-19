import sympy
import numpy as np
import underworld3 
from typing import List

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
    dummy = sympy.Symbol("dummy")
    return fn1.subs(fn2,dummy).diff(dummy).subs(dummy,fn2)

_ext_dict = {}
def getext(mesh, fns_residual, fns_jacobian, fns_bcs, primary_field_list):
    """
    Check if we've already created an equivalent extension
    and use if available.
    """
    import time
    time_s = time.time()

    fns = fns_residual + fns_jacobian + tuple(fns_bcs)
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

def _createext(name:               str, 
               mesh:               underworld3.mesh.Mesh,
               fns_residual:       List[sympy.Function], 
               fns_jacobian:       List[sympy.Function],
               fns_bcs:            List[sympy.Function],
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
    from underworld3.mesh import VarType

    # Note that the order here is important.
    fns = fns_residual + tuple(fns_bcs) + fns_jacobian
    count_residual_sig = len(fns_residual) + len(fns_bcs)
    count_jacobian_sig = len(fns_jacobian)

    # Create some symbol which will force codegen to produce required interface.
    # We'll use MatrixSymbol objects, which sympy simply converts to double* within 
    # the generated code. 
    # Create `MatrixSymbol` vectors of unknown size 1xN. The size doesn't matter
    # as MatrixSymbol entries will simply be replaced with corresponding c-array entry,
    # and then discarded.
    from sympy.abc import N
    petsc_x   = MatrixSymbol(  'petsc_x', 1, N)
    petsc_u   = MatrixSymbol(  'petsc_u', 1, N)
    petsc_u_x = MatrixSymbol('petsc_u_x', 1, N)
    petsc_a   = MatrixSymbol(  'petsc_a', 1, N)
    petsc_a_x = MatrixSymbol('petsc_a_x', 1, N)

    # Now create substitute dictionary to specify how fns will be replaced
    # by corresponsing MatrixSymbol objects (created above).
    substitute = {}

    # First do subs for N.x,N.y,N.z terms.
    for index, base_scalar in enumerate(mesh.N.base_scalars()):
        substitute[base_scalar] = petsc_x[0,index]

    # Now do variable and variable gradient terms.
    # We'll define a function to do this as we need 
    # to repeate it for the primary and aux variables.
    def get_variable_mapping(varlist, petsc_matsymbol, petsc_grad_matsymbol):
        """ 
        This function gets the mapping of variables
        into the arrays PETSc provides for element assembly.

        For a `varlist` consisting of 2d velocity & pressure variables, 
        it'll generate the following mapping:
        {
            V_x   : petsc_matsymbol[0,0],
            V_y   : petsc_matsymbol[0,1],
            P     : petsc_matsymbol[0,2],
            V_x_x : petsc_grad_matsymbol[0,0],
            V_x_y : petsc_grad_matsymbol[0,1],
            V_y_x : petsc_grad_matsymbol[0,2],
            V_y_y : petsc_grad_matsymbol[0,3],
            P_x   : petsc_grad_matsymbol[0,4],
            P_y   : petsc_grad_matsymbol[0,5]
        }

        Params
        ------
        varlist
            List of variables that will map into petsc arrays. Note that
            *all* the variables in the corresponding `PetscDM` must be 
            included. They must also be ordered according to their `field_id`.
        petsc_matsymbol
            A `MatrixSymbol` vector of sufficient size for all variables
        petsc_grad_matsymbol
            A `MatrixSymbol` vector of sufficient size for all variable gradients.
        """ 
        from collections import OrderedDict
        mapping = OrderedDict()
        u_i = 0
        for var in varlist:
            if var.vtype==VarType.SCALAR:
                mapping[var.fn] = petsc_matsymbol[0,u_i]
                u_i +=1
            elif var.vtype==VarType.VECTOR:
                # Pull out individual sub components
                for bvec in mesh.N.base_vectors()[0:mesh.dim]:
                    comp = var.fn.dot(bvec)
                    mapping[comp] = petsc_matsymbol[0,u_i]
                    u_i +=1
            else:
                raise RuntimeError("TODO: Implement VarType.OTHER field codegen.")
        # Now gradient terms
        mapping_grads = OrderedDict()
        u_x_i = 0
        for fn in mapping.keys():
            # Simply generate the required derivative in place, 
            # and set to be substituted by c pointer
            for base_scalar in mesh.N.base_scalars()[0:mesh.dim]:
                mapping_grads[fn.diff(base_scalar)] = petsc_grad_matsymbol[0,u_x_i]
                u_x_i += 1
        # Concatenate dicts
        mapping.update(mapping_grads)
        return mapping

    # Add mapping across aux variables first.
    substitute.update(get_variable_mapping(mesh.vars.values(),petsc_a,petsc_a_x))
    # Now replace with those we want to use from primary dm variables.
    substitute.update(get_variable_mapping(primary_field_list,petsc_u,petsc_u_x))
    # Let's do subsitutions now.
    subbedfns = []
    for fn in fns:
        subbedfns.append(fn.subs(substitute))

    # Now go ahead and generate C code from subsituted Sympy expressions.
    from sympy.printing.ccode import C99CodePrinter
    printer = C99CodePrinter()
    eqns = []
    for index, fn in enumerate(subbedfns):
        if isinstance(fn, sympy.vector.Vector):
            fn = fn.to_matrix(mesh.N)[0:mesh.dim,0]
        elif isinstance(fn, sympy.vector.Dyadic):
            fn = fn.to_matrix(mesh.N)[0:mesh.dim,0:mesh.dim]
        else:
            fn = sympy.Matrix([fn])
        out = sympy.MatrixSymbol("out",*fn.shape)
        eqns.append( ("eqn_"+str(index), printer.doprint(fn, out)) )
    MODNAME = "fn_ptr_ext_" + str(name)

    # Link against function.cpython which contains symbols our custom functions.
    import underworld3.function
    lib = underworld3.function.__file__
    import os
    libdir = os.path.dirname(lib)
    libfile = os.path.basename(lib)
    # Prepend colon to force linking against filename without 'lib' prefix.
    libfile = ":" + libfile  

    # Make lists here in anticipation of future requirements.
    libdirs = [libdir,]
    libfiles = [libfile,]
    incdirs  = [np.get_include(),libdir]

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
           HEADERS=incdirs,
           LIBDIRS=libdirs,
           LIBFILES=libfiles,
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
"""

    # Create cython top content.
    pyx_str="""
from underworld3.petsc_types cimport PetscInt, PetscReal, PetscScalar, PetscErrorCode, PetscBool, PetscDSResidualFn, PetscDSJacobianFn
from underworld3.petsc_types cimport PtrContainer
from libc.stdlib cimport malloc
from libc.math cimport *

cdef extern from "cy_ext.h" nogil:
"""

    # Print includes
    for header in printer.headers:
        h_str += "#include \"{}\"\n".format(header)

    h_str += "\n"
    # Print equations
    for eqn in eqns[0:count_residual_sig]:
        h_str  +="void petsc_{}{}\n{{\n{}\n}}\n\n".format(eqn[0],residual_sig,eqn[1])
        pyx_str+="    void petsc_{}{}\n".format(eqn[0],residual_sig)

    for eqn in eqns[count_residual_sig:]:
        h_str  +="void petsc_{}{}\n{{\n{}\n}}\n\n".format(eqn[0],jacobian_sig,eqn[1])
        pyx_str+="    void petsc_{}{}\n".format(eqn[0],jacobian_sig)

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
        pyx_str+="    clsguy.fns_residual[{}] = petsc_{}\n".format(index,eqn[0])
    for index,eqn in enumerate(eqns[count_residual_sig:]):
        pyx_str+="    clsguy.fns_jacobian[{}] = petsc_{}\n".format(index,eqn[0])
    for index,eqn in enumerate(eqns[len(fns_residual):count_residual_sig]):
        pyx_str+="    clsguy.fns_bcs[{}] = petsc_{}\n".format(index,eqn[0])
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
        raise RuntimeError("Extension module does not appear to have been created.")
