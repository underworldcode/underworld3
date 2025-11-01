from typing import List
import subprocess
from xmlrpc.client import boolean
import sympy
import underworld3
import underworld3.timing as timing
from typing import Optional
from collections import namedtuple


## This is not required in sympy >= 1.9

# def diff_fn1_wrt_fn2(fn1, fn2):
#     """
#     This function takes the derivative of a function (fn1) with respect
#     to another function (fn2). Sympy does not allow this natively, instead
#     only allowing derivatives with respect to symbols.  Here, we
#     temporarily subsitute fn2 for a dummy symbol, perform the derivative (with
#     respect to the dummy symbol), and then replace the dummy for fn2 again.
#     """
#     if fn2.is_zero:
#         return 0
#     # If fn1 doesn't contain fn2, immediately return zero.
#     # The full diff method will also return zero, but will be slower.
#     if len(fn1.atoms(fn2))==0:
#         return 0
#     uwderivdummy = sympy.Symbol("uwderivdummy")
#     subfn   = fn1.xreplace({fn2:uwderivdummy})      # sub in dummy
#     subfn_d = subfn.diff(uwderivdummy)              # actual deriv
#     deriv   = subfn_d.xreplace({uwderivdummy:fn2})  # sub out dummy
#     return deriv

_ext_dict = {}


# Generates the C debugging string for the compiled function block
def debugging_text(randstr, fn, fn_type, eqn_no):
    try:
        object_size = len(fn.flat())
    except:
        object_size = 1

    outstr = "out[0]"
    for i in range(1, object_size):
        outstr += f", out[{i}]"

    formatstr = "%6e, " * object_size

    debug_str = f"/* {fn} */\n"
    debug_str += f"/* Size = {object_size} */\n"
    debug_str += f'FILE *fp; fp = fopen( "{randstr}_debug.txt", "a" );\n'
    debug_str += f'fprintf(fp,"{fn_type} - equation {eqn_no} at (%.2e, %.2e, %.2e) -> ", petsc_x[0], petsc_x[1], dim==2 ? 0.0: petsc_x[2]);\n'
    debug_str += f'fprintf(fp,"{formatstr}\\n", {outstr});\n'
    debug_str += f"fclose(fp);"

    return debug_str


def debugging_text_bd(randstr, fn, fn_type, eqn_no):
    try:
        object_size = len(fn.flat())
    except:
        object_size = 1

    outstr = "out[0]"
    for i in range(1, object_size):
        outstr += f", out[{i}]"

    formatstr = "%6e, " * object_size

    debug_str = f"/* {fn} */\n"
    debug_str += f"/* Size = {object_size} */\n"
    debug_str += f'FILE *fp; fp = fopen( "{randstr}_debug.txt", "a" );\n'
    debug_str += f'fprintf(fp,"{fn_type} - equation {eqn_no} X / N (%.2e, %.2e, %.2e / %2.e, %2.e, %.2e ) -> ", petsc_x[0], petsc_x[1], dim==2 ? 0.0: petsc_x[2], petsc_n[0], petsc_n[1], dim==2 ? 0.0: petsc_n[2]);\n'
    debug_str += f'fprintf(fp,"{formatstr}\\n", {outstr});\n'
    debug_str += f"fclose(fp);"

    return debug_str


@timing.routine_timer_decorator
def getext(
    mesh,
    fns_residual,
    fns_jacobian,
    fns_bcs,
    fns_bd_residual,
    fns_bd_jacobian,
    primary_field_list,
    verbose=False,
    debug=False,
    debug_name=None,
    cache=True,
):
    """
    Check if we've already created an equivalent extension
    and use if available.
    """
    import time

    time_s = time.time()

    raw_fns = (
        tuple(fns_residual)
        + tuple(fns_bcs)
        + tuple(fns_jacobian)
        + tuple(fns_bd_residual)
        + tuple(fns_bd_jacobian)
    )

    ## Expand all functions to ensure that changes in constants are recognised
    ## in the caching process.

    expanded_fns = []

    for fn in raw_fns:
        expanded_fns.append(
            underworld3.function.expressions.unwrap(fn, keep_constants=False, return_self=False)
        )

    fns = tuple(expanded_fns)

    if debug and underworld3.mpi.rank == 0:
        print(f"Expanded functions for compilation:")
        for i, fn in enumerate(fns):
            print(f"{i}: {fn}")

    import os

    # if verbose and uw.mpi.rank == 0:
    #     for i, fn in enumerate(fns):
    #         print(f"JIT: [{i:3d}] -> {fn}", flush=True)

    if debug_name is not None:
        jitname = debug_name

    elif "UW_JITNAME" in os.environ:  # If var specified, probably testing.
        jitname = os.environ["UW_JITNAME"]
        # Note, extensions cannot be replaced, so need to append count to ensure
        # unique modules.
        jitname += "_" + str(len(_ext_dict.keys()))

    else:  # Else name from fns hash
        jitname = abs(hash((mesh, fns, tuple(mesh.vars.keys()))))

    # Create the module if not in dictionary
    if jitname not in _ext_dict.keys() or not cache:
        _createext(
            jitname,
            mesh,
            fns_residual,
            fns_bcs,
            fns_jacobian,
            fns_bd_residual,
            fns_bd_jacobian,
            primary_field_list,
            verbose=verbose,
            debug=debug,
            debug_name=debug_name,
        )
    else:
        if verbose and underworld3.mpi.rank == 0:
            print(f"JIT compiled module cached ... {jitname} ", flush=True)

    ## TODO: Return a dictionary to recover the function pointers from the compiled
    ## functions. Note, keep these by category as the same sympy function has
    ## different compiled form depending on the function signature

    module = _ext_dict[jitname]
    ptrobj = module.getptrobj()
    # print(f"jit time {time.time()-time_s}", flush=True)

    i_res = {}
    for index, fn in enumerate(fns_residual):
        i_res[fn] = index

    i_ebc = {}
    for index, fn in enumerate(fns_bcs):
        i_ebc[fn] = index

    i_jac = {}
    for index, fn in enumerate(fns_jacobian):
        i_jac[fn] = index

    i_bd_res = {}
    for index, fn in enumerate(fns_bd_residual):
        i_bd_res[fn] = index

    i_bd_jac = {}
    for index, fn in enumerate(fns_bd_jacobian):
        i_bd_jac[fn] = index

    extn_fn_dict = namedtuple(
        "Functions",
        ["res", "jac", "ebc", "bd_res", "bd_jac"],
    )

    extensions_functions_dicts = extn_fn_dict(i_res, i_jac, i_ebc, i_bd_res, i_bd_jac)

    return ptrobj, extensions_functions_dicts


@timing.routine_timer_decorator
def _createext(
    name: str,
    mesh: underworld3.discretisation.Mesh,
    fns_residual: List[sympy.Basic],
    fns_bcs: List[sympy.Basic],
    fns_jacobian: List[sympy.Basic],
    fns_bd_residual: List[sympy.Basic],
    fns_bd_jacobian: List[sympy.Basic],
    primary_field_list: List[underworld3.discretisation.MeshVariable],
    verbose: Optional[bool] = False,
    debug: Optional[bool] = False,
    debug_name=None,
):
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
    fns_bd_residual:
        List of system's boundary integral sympy functions for which JIT equivalents
        will be generated.
    fns_bd_jacobian:
        List of system's boundary integral jacobian sympy functions for which JIT equivalents
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
    fns = (
        tuple(fns_residual)
        + tuple(fns_bcs)
        + tuple(fns_jacobian)
        + tuple(fns_bd_residual)
        + tuple(fns_bd_jacobian)
    )

    count_residual_sig = len(fns_residual)
    count_bc_sig = len(fns_bcs)
    count_jacobian_sig = len(fns_jacobian)
    count_bd_residual_sig = len(fns_bd_residual)
    count_bd_jacobian_sig = len(fns_bd_jacobian)

    # `_ccode` patching
    def ccode_patch_fns(varlist, prefix_str):
        """
        This function patches uw functions with the necessary ccode
        routines for the code printing.

        For a `varlist` consisting of 2d velocity & pressure variables,
        for example, it'll generate routines which write the following,
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
        u_i = 0  # variable increment
        u_x_i = 0  # variable gradient increment
        lambdafunc = lambda self, printer: self._ccodestr
        for var in varlist:
            if var.vtype == VarType.SCALAR:
                # monkey patch this guy into the function
                type(var.fn)._ccodestr = f"{prefix_str}[{u_i}]"
                type(var.fn)._ccode = lambdafunc
                u_i += 1
                # now patch gradient guy into varfn guy
                for ind in range(mesh.dim):
                    # Note that var.fn._diff[ind] returns the class, so we don't need type(var.fn._diff[ind])
                    var.fn._diff[ind]._ccodestr = f"{prefix_str}_x[{u_x_i}]"
                    var.fn._diff[ind]._ccode = lambdafunc
                    u_x_i += 1
            elif (
                var.vtype == VarType.VECTOR
                or var.vtype == VarType.TENSOR
                or var.vtype == VarType.SYM_TENSOR
                or var.vtype == VarType.MATRIX
            ):
                # Pull out individual sub components
                for comp in var.sym_1d:
                    # monkey patch
                    type(comp)._ccodestr = f"{prefix_str}[{u_i}]"
                    type(comp)._ccode = lambdafunc
                    u_i += 1
                    # and also patch gradient guy into varfn guy's comp guy   # Argh ... too much Mansourness
                    for ind in range(mesh.dim):
                        # Note that var.fn._diff[ind] returns the class, so we don't need type(var.fn._diff[ind])
                        comp._diff[ind]._ccodestr = f"{prefix_str}_x[{u_x_i}]"
                        comp._diff[ind]._ccode = lambdafunc
                        u_x_i += 1
            else:
                raise RuntimeError(
                    f"Unsupported type {var.vtype} for code generation. Please contact developers."
                )

    # Patch in `_code` methods. Note that the order here
    # is important, as the secondary call will overwrite
    # those patched in the first call.

    ccode_patch_fns(mesh.vars.values(), "petsc_a")
    ccode_patch_fns(primary_field_list, "petsc_u")

    # Also patch `BaseScalar` types. Nothing fancy - patch the overall type,
    # make sure each component points to the correct PETSc data

    ## This is set up in the mesh at the moment but this does seem to be the wrong place

    # mesh.N.x._ccodestr = "petsc_x[0]"
    # mesh.N.y._ccodestr = "petsc_x[1]"
    # mesh.N.z._ccodestr = "petsc_x[2]"

    # # Surface integrals also have normal vector information as petsc_n

    # mesh.Gamma_N.x._ccodestr = "petsc_n[0]"
    # mesh.Gamma_N.y._ccodestr = "petsc_n[1]"
    # mesh.Gamma_N.z._ccodestr = "petsc_n[2]"

    type(mesh.N.x)._ccode = lambda self, printer: self._ccodestr
    type(mesh.Gamma_N.x)._ccode = lambda self, printer: self._ccodestr

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
        "Heaviside": [
            (
                lambda *args: len(args) == 1,
                "Heaviside_1",
            ),  # for single arg Heaviside  (defaults to 0.5 at jump).
            (lambda *args: len(args) == 2, "Heaviside_2"),
        ],  # for two arg Heavisides    (second arg is jump value).
    }

    # Now go ahead and generate C code from substituted Sympy expressions.
    # from sympy.printing.c import C99CodePrinter
    # printer = C99CodePrinter(user_functions=custom_functions)
    from sympy.printing.c import c_code_printers

    printer = c_code_printers["c99"]({"user_functions": custom_functions})

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

        fn = underworld3.function.expressions.unwrap(fn, keep_constants=False, return_self=False)

        if isinstance(fn, sympy.vector.Vector):
            fn = fn.to_matrix(mesh.N)[0 : mesh.dim, 0]
        elif isinstance(fn, sympy.vector.Dyadic):
            fn = fn.to_matrix(mesh.N)[0 : mesh.dim, 0 : mesh.dim]
        else:
            fn = sympy.Matrix([fn])

        if verbose:
            print("Processing JIT {:4d} / {}".format(index, fn))

        out = sympy.MatrixSymbol("out", *fn.shape)
        eqn = ("eqn_" + str(index), printer.doprint(fn, out))
        if eqn[1].startswith("// Not supported in C:"):
            spliteqn = eqn[1].split("\n")
            raise RuntimeError(
                f"Error encountered generating JIT extension:\n"
                f"{spliteqn[0]}\n"
                f"{spliteqn[1]}\n"
                f"This is usually because code generation for a Sympy function (or its derivative) is not supported.\n"
                f"Please contact the developers."
                f"---"
                f"The ID of the JIT component that failed is {index}"
                f"The decription of the JIT component that failed:\n {fn}"
            )
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
    extra_compile_args=['-std=c99','-O3'],
    extra_link_args=[]
)]
setup(ext_modules=cythonize(ext_mods))
""".format(
        NAME=MODNAME,
        HEADERS=list(underworld3._incdirs.keys()),
        LIBDIRS=list(underworld3._libdirs.keys()),
        LIBFILES=list(underworld3._libfiles.keys()),
    )
    codeguys.append(["setup.py", setup_py_str])

    residual_sig = "(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar petsc_u[], const PetscScalar petsc_u_t[], const PetscScalar petsc_u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar petsc_a[], const PetscScalar petsc_a_t[], const PetscScalar petsc_a_x[], PetscReal petsc_t,                           const PetscReal petsc_x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar out[])"
    jacobian_sig = "(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar petsc_u[], const PetscScalar petsc_u_t[], const PetscScalar petsc_u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar petsc_a[], const PetscScalar petsc_a_t[], const PetscScalar petsc_a_x[], PetscReal petsc_t, PetscReal petsc_u_tShift, const PetscReal petsc_x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar out[])"
    bd_residual_sig = "(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar petsc_u[], const PetscScalar petsc_u_t[], const PetscScalar petsc_u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar petsc_a[], const PetscScalar petsc_a_t[], const PetscScalar petsc_a_x[], PetscReal petsc_t,                           const PetscReal petsc_x[], const PetscReal petsc_n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar out[])"
    bd_jacobian_sig = "(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar petsc_u[], const PetscScalar petsc_u_t[], const PetscScalar petsc_u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar petsc_a[], const PetscScalar petsc_a_t[], const PetscScalar petsc_a_x[], PetscReal petsc_t, PetscReal petsc_u_tShift, const PetscReal petsc_x[],  const PetscReal petsc_n[],PetscInt numConstants, const PetscScalar constants[], PetscScalar out[])"

    # Create header top content.
    h_str = """
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
    pyx_str = """
from underworld3.cython.petsc_types cimport PetscInt, PetscReal, PetscScalar, PetscErrorCode, PetscBool, PetscDSResidualFn, PetscDSJacobianFn, PetscDSBdResidualFn, PetscDSBdJacobianFn
from underworld3.cython.petsc_types cimport PtrContainer
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
    import os

    if not "UW_JITNAME" in os.environ:
        randstr = "".join(random.choices(string.ascii_uppercase, k=5))
    else:
        if debug_name is None:
            randstr = "FUNC_" + str(len(_ext_dict.keys()))
        else:
            randstr = debug_name

    # Print includes
    for header in printer.headers:
        h_str += '#include "{}"\n'.format(header)

    h_str += "\n"

    # Print equations
    eqn_index_0 = 0
    eqn_index_1 = count_residual_sig
    fn_counter = 0

    for eqn in eqns[eqn_index_0:eqn_index_1]:
        debug_str = debugging_text(randstr, fns[fn_counter], "  res", fn_counter)
        h_str += "void {}_petsc_{}{}\n{{\n{}\n{}\n}}\n\n".format(
            randstr, eqn[0], residual_sig, eqn[1], debug_str if debug else ""
        )
        pyx_str += "    void {}_petsc_{}{}\n".format(randstr, eqn[0], residual_sig)
        fn_counter += 1

    eqn_index_0 = eqn_index_1
    eqn_index_1 = eqn_index_1 + count_bc_sig

    # The bcs have the same signature as the residuals (at present)
    # but we leave this separate in case it changes in later PETSc implementations

    for eqn in eqns[eqn_index_0:eqn_index_1]:
        debug_str = debugging_text(randstr, fns[fn_counter], "  ebc", fn_counter)
        h_str += "void {}_petsc_{}{}\n{{\n{}\n{}\n}}\n\n".format(
            randstr, eqn[0], residual_sig, eqn[1], debug_str if debug else ""
        )
        pyx_str += "    void {}_petsc_{}{}\n".format(randstr, eqn[0], residual_sig)
        fn_counter += 1

    eqn_index_0 = eqn_index_1
    eqn_index_1 = eqn_index_1 + count_jacobian_sig

    for eqn in eqns[eqn_index_0:eqn_index_1]:
        debug_str = debugging_text(randstr, fns[fn_counter], "  jac", fn_counter)

        h_str += "void {}_petsc_{}{}\n{{\n{}\n{}\n}}\n\n".format(
            randstr, eqn[0], jacobian_sig, eqn[1], debug_str if debug else ""
        )
        pyx_str += "    void {}_petsc_{}{}\n".format(randstr, eqn[0], jacobian_sig)
        fn_counter += 1

    eqn_index_0 = eqn_index_1
    eqn_index_1 = eqn_index_1 + count_bd_residual_sig
    for eqn in eqns[eqn_index_0:eqn_index_1]:
        debug_str = debugging_text_bd(randstr, fns[fn_counter], "bdres", fn_counter)
        h_str += "void {}_petsc_{}{}\n{{\n{}\n{}\n}}\n\n".format(
            randstr, eqn[0], bd_residual_sig, eqn[1], debug_str if debug else ""
        )
        pyx_str += "    void {}_petsc_{}{}\n".format(randstr, eqn[0], bd_residual_sig)
        fn_counter += 1

    eqn_index_0 = eqn_index_1
    eqn_index_1 = eqn_index_1 + count_bd_jacobian_sig
    for eqn in eqns[eqn_index_0:eqn_index_1]:
        debug_str = debugging_text_bd(randstr, fns[fn_counter], "bdjac", fn_counter)
        h_str += "void {}_petsc_{}{}\n{{\n{}\n{}\n}}\n\n".format(
            randstr, eqn[0], bd_jacobian_sig, eqn[1], debug_str if debug else ""
        )
        pyx_str += "    void {}_petsc_{}{}\n".format(randstr, eqn[0], bd_jacobian_sig)
        fn_counter += 1

    codeguys.append(["cy_ext.h", h_str])
    # Note that the malloc below will cause a leak, but it's just a bunch of function
    # pointers so we don't need to worry about it (yet)
    pyx_str += """
cpdef PtrContainer getptrobj():
    clsguy = PtrContainer()
    clsguy.fns_residual = <PetscDSResidualFn*> malloc({}*sizeof(PetscDSResidualFn))
    clsguy.fns_bcs      = <PetscDSResidualFn*> malloc({}*sizeof(PetscDSResidualFn))
    clsguy.fns_jacobian = <PetscDSJacobianFn*> malloc({}*sizeof(PetscDSJacobianFn))
    clsguy.fns_bd_residual = <PetscDSBdResidualFn*> malloc({}*sizeof(PetscDSBdResidualFn))
    clsguy.fns_bd_jacobian = <PetscDSBdJacobianFn*> malloc({}*sizeof(PetscDSBdJacobianFn))
""".format(
        len(fns_residual),
        len(fns_bcs),
        len(fns_jacobian),
        len(fns_bd_residual),
        len(fns_bd_jacobian),
    )

    eqn_count = 0
    for index, eqn in enumerate(eqns[eqn_count : eqn_count + len(fns_residual)]):
        pyx_str += "    clsguy.fns_residual[{}] = {}_petsc_{}\n".format(index, randstr, eqn[0])
        eqn_count += 1

    residual_equations = (0, eqn_count)

    for index, eqn in enumerate(eqns[eqn_count : eqn_count + len(fns_bcs)]):
        pyx_str += "    clsguy.fns_bcs[{}] = {}_petsc_{}\n".format(index, randstr, eqn[0])
        eqn_count += 1

    boundary_equations = (residual_equations[1], eqn_count)

    for index, eqn in enumerate(eqns[eqn_count : eqn_count + len(fns_jacobian)]):
        pyx_str += "    clsguy.fns_jacobian[{}] = {}_petsc_{}\n".format(index, randstr, eqn[0])
        eqn_count += 1

    jacobian_equations = (boundary_equations[1], eqn_count)

    for index, eqn in enumerate(eqns[eqn_count : eqn_count + len(fns_bd_residual)]):
        pyx_str += "    clsguy.fns_bd_residual[{}] = {}_petsc_{}\n".format(index, randstr, eqn[0])
        eqn_count += 1

    boundary_residual_equations = (jacobian_equations[1], eqn_count)

    for index, eqn in enumerate(eqns[eqn_count : eqn_count + len(fns_bd_jacobian)]):
        pyx_str += "    clsguy.fns_bd_jacobian[{}] = {}_petsc_{}\n".format(index, randstr, eqn[0])
        eqn_count += 1

    boundary_jacobian_equations = (boundary_residual_equations[1], eqn_count)

    pyx_str += "    return clsguy"
    codeguys.append(["cy_ext.pyx", pyx_str])

    # Write out files
    import os

    import time
    import random

    # Make directory name unique to avoid race conditions between parallel processes
    unique_suffix = f"{os.getpid()}_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
    tmpdir = os.path.join("/tmp", f"{MODNAME}_{unique_suffix}")

    try:
        os.makedirs(tmpdir, exist_ok=True)
    except OSError as e:
        if verbose:
            print(f"Warning: Failed to create tmpdir {tmpdir}: {e}")
        raise RuntimeError(f"Cannot create temporary directory {tmpdir}") from e
    for thing in codeguys:
        filename = thing[0]
        strguy = thing[1]
        with open(os.path.join(tmpdir, filename), "w") as f:
            f.write(strguy)

    # Build
    import sys

    process = subprocess.Popen(
        [sys.executable] + "setup.py build_ext --inplace".split(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=tmpdir,
    )
    stdout, stderr = process.communicate()

    # Check if build process failed
    if process.returncode != 0:
        if verbose:
            print(f"Warning: Build process failed with return code {process.returncode}")
            print(f"stdout: {stdout.decode() if stdout else 'None'}")
            print(f"stderr: {stderr.decode() if stderr else 'None'}")

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
        spec = importlib.machinery.ModuleSpec(name=name, loader=loader, origin=path)
        return _load(spec)

    # Check if tmpdir exists before trying to list it
    if os.path.exists(tmpdir):
        for _file in os.listdir(tmpdir):
            if _file.endswith(".so"):
                _ext_dict[name] = load_dynamic(MODNAME, os.path.join(tmpdir, _file))
    else:
        # tmpdir doesn't exist, likely build process failed
        if verbose:
            print(f"Warning: tmpdir {tmpdir} does not exist - build process may have failed")

    if name not in _ext_dict.keys():
        raise RuntimeError(
            f"The Underworld extension module does not appear to have been built successfully. "
            f"The generated module may be found at:\n    {str(tmpdir)}\n"
            f"To investigate, you may attempt to build it manually by running\n"
            f"    python3 setup.py build_ext --inplace\n"
            f"from the above directory. Note that a new module will always be written by "
            f"Underworld and therefore any modifications to the above files will not persist into "
            f"your Underworld runtime.\n"
            f"Please contact the developers if you are unable to resolve the issue."
        )

    if underworld3.mpi.rank == 0 and verbose:
        print(f"Location of compiled module: {str(tmpdir)}")

        print(
            f"{randstr} Equation count - {eqn_count}",
            flush=True,
        )
        print(
            f"{randstr}   {len(fns_residual):5d}    residuals: {residual_equations[0]}:{residual_equations[1]}",
            flush=True,
        )
        print(
            f"{randstr}   {len(fns_bcs):5d}   boundaries: {boundary_equations[0]}:{boundary_equations[1]}",
            flush=True,
        )
        print(
            f"{randstr}   {len(fns_jacobian):5d}    jacobians: {jacobian_equations[0]}:{jacobian_equations[1]}",
            flush=True,
        )
        print(
            f"{randstr}   {len(fns_bd_residual):5d} boundary_res: {boundary_residual_equations[0]}:{boundary_residual_equations[1]}",
            flush=True,
        )
        print(
            f"{randstr}   {len(fns_bd_jacobian):5d} boundary_jac: {boundary_jacobian_equations[0]}:{boundary_jacobian_equations[1]}",
            flush=True,
        )

    return
