import sympy
import numpy as np

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
def getext(mesh, amesh, fns_residual, fns_jacobian, fns_bcs):
    """
    Check if we've already created an equivalent extension
    and use if available.
    """
    fns = fns_residual + fns_jacobian + tuple(fns_bcs)
    import os
    if 'UW_JITNAME' in os.environ:          # if var specified, probably testing.
        jitname = os.environ['UW_JITNAME']
        # note, extensions cannot be replaced, so need to append count to ensure
        # unique modules.
        jitname += "_" + str(len(_ext_dict.keys()))
    else:                                   # else name from fns hash
        jitname = abs(hash((mesh,amesh,fns)))
    # create the module if not in dictionary
    if jitname not in _ext_dict.keys():
        _createext(jitname, mesh, amesh, fns_residual, fns_jacobian, fns_bcs)
    module = _ext_dict[jitname]
    return module.getptrobj()

def _createext(name, mesh, amesh, fns_residual, fns_jacobian, fns_bcs):
    """
    This creates the required extension which houses the JIT
    fn pointer for PETSc. 

    Note that it is not possible to replace loaded shared libraries
    in Python, so we instead create a new extension for each new function. 

    We hash the functions and create a dictionary of the generated extensions
    to avoid redundantly creating new extensions.

    Params
    ------
    name: str
        Name for the extension. It will be prepended with "fn_ptr_ext_"
    """
    from sympy import symbols, Eq, MatrixSymbol
    from underworld3.mesh import VarType

    # note that the order here is important.
    fns = fns_residual + tuple(fns_bcs) + fns_jacobian
    count_residual_sig = len(fns_residual) + len(fns_bcs)
    count_jacobian_sig = len(fns_jacobian)
    # get fn/fn_grad component totals
    tot_fns = 0
    tot_grad_fns = 0
    for var in mesh.vars.values():
        tot_fns += var.num_components
        tot_grad_fns += var.num_components*mesh.dim

    # get aux/aux_grad component totals
    aux_tot_fns = 0
    aux_tot_grad_fns = 0
    if amesh:
        for var in amesh.vars.values():
            aux_tot_fns += var.num_components
            aux_tot_grad_fns += var.num_components*mesh.dim

    # Create some symbol which will force codegen to produce required interface.
    # We'll use MatrixSymbol objects, which sympy simply converts to double* within 
    # the generated code. 
    petsc_x   = MatrixSymbol(  'petsc_x', 1, 3)  # let's just set this to 3-dim, as it'll be 
                                                    # the max and doesn't matter otherwise.
    petsc_u   = MatrixSymbol(  'petsc_u', 1, tot_fns)
    petsc_u_x = MatrixSymbol('petsc_u_x', 1, tot_grad_fns)
    petsc_a   = MatrixSymbol(  'petsc_a', 1, aux_tot_fns)
    petsc_a_x = MatrixSymbol('petsc_a_x', 1, aux_tot_grad_fns)

    # Now create substitute dictionary to specify how fns will be replaced
    # by corresponsing MatrixSymbol objects (created above).
    # First do subs for N.x,N.y,N.z terms
    substitute = {}
    for index, base_scalar in enumerate(mesh.N.base_scalars()):
        substitute[base_scalar] = petsc_x[index]
    # Now do T, P, V_x, V_y, V_z etc terms
    u_i = 0
    component_fns = []           # We'll use this list later to do gradient terms
    for var in mesh.vars.values():
        if var.vtype==VarType.SCALAR:
            # Substitute all instances of the mesh var with the required c pointer 
            substitute[var.fn] = petsc_u[u_i]
            component_fns.append(var.fn)
            u_i+=1
        elif var.vtype==VarType.VECTOR:
            # pull out individual sub components
            for bvec in mesh.N.base_vectors()[0:mesh.dim]:
                guy = var.fn.dot(bvec)
                substitute[guy] = petsc_u[u_i]
                component_fns.append(guy)                
                u_i+=1
        else:
            raise RuntimeError("TODO: Implement VarType.OTHER field codegen.")
    # Now gradient terms
    u_x_i = 0
    for fn in component_fns:
        # Now process gradients. Simply generate the required derivative in place, 
        # and set to be substituted by c pointer
        for base_scalar in mesh.N.base_scalars()[0:mesh.dim]:
            substitute[fn.diff(base_scalar)] = petsc_u_x[u_x_i]
            u_x_i += 1

    # Now do auxiliary terms
    a_i = 0
    component_fns = []
    if amesh: # weak check of type
        for var in amesh.vars.values():
            if var.vtype==VarType.SCALAR:
                substitute[var.fn] = petsc_a[a_i]
                component_fns.append(var.fn)
                a_i+=1
            elif var.vtype==VarType.VECTOR:
                for bvec in mesh.N.base_vectors()[0:mesh.dim]:
                    guy = var.fn.dot(bvec)
                    substitute[guy] = petsc_a[a_i]
                    component_fns.append(guy)
                    a_i+=1
            else:
                raise RuntimeError("TODO: Implement VarType.OTHER field codegen.")
        # Now gradient terms of auxiliary vars
        a_x_i = 0
        for fn in component_fns:
            for base_scalar in mesh.N.base_scalars()[0:mesh.dim]:
                substitute[fn.diff(base_scalar)] = petsc_a_x[a_x_i]
                a_x_i+=1

    # do subsitutions
    subbedfns = []
    for fn in fns:
        subbedfns.append(fn.subs(substitute))

    # Now go ahead and generate C code from subsituted Sympy expressions
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

    # link against function.cpython which contains symbols our custom functions
    import underworld3.function
    lib = underworld3.function.__file__
    import os
    libdir = os.path.dirname(lib)
    libfile = os.path.basename(lib)
    # prepend colon to force linking against filename without 'lib' prefix.
    libfile = ":" + libfile  

    # make lists here in anticipation of future requirements
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

    # create header top content
    h_str="""
typedef int PetscInt;
typedef double PetscReal;
typedef double PetscScalar;
typedef int PetscBool;
#include <math.h> 
"""

    # create cython top content
    pyx_str="""
from underworld3.petsc_types cimport PetscInt, PetscReal, PetscScalar, PetscErrorCode, PetscBool, PetscDSResidualFn, PetscDSJacobianFn
from underworld3.petsc_types cimport PtrContainer
from libc.stdlib cimport malloc
from libc.math cimport *

cdef extern from "cy_ext.h" nogil:
"""

    # print includes
    for header in printer.headers:
        h_str += "#include \"{}\"\n".format(header)

    h_str += "\n"
    # print equations
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
