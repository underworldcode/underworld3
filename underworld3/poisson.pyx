from petsc4py.PETSc cimport DM, PetscDM, DS, PetscDS
from .petsc_types cimport PetscInt, PetscReal, PetscScalar, PetscErrorCode, PetscBool, DMBoundaryConditionType, PetscDSResidualFn, PetscDSJacobianFn
from .petsc_types cimport PtrContainer
# TODO
# gil v nogil 
# ctypeds DMBoundaryConditionType etc.. is there a cleaner way? 

cdef extern from "petsc.h":
    PetscErrorCode PetscDSAddBoundary( PetscDS, DMBoundaryConditionType, const char[], const char[], PetscInt, PetscInt, const PetscInt *, void (*)(), PetscInt, const PetscInt *, void *)

cdef extern from "petsc.h" nogil:
    PetscErrorCode PetscDSSetResidual( PetscDS, PetscInt, PetscDSResidualFn, PetscDSResidualFn )
    PetscErrorCode PetscDSSetJacobian( PetscDS, PetscInt, PetscInt, PetscDSJacobianFn, PetscDSJacobianFn, PetscDSJacobianFn, PetscDSJacobianFn)
    PetscErrorCode DMPlexSetSNESLocalFEM( PetscDM, void *, void *, void *)

cdef void g3_uu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[]):
    cdef PetscInt d_i
    for d_i in range(dim):
        g3[d_i*dim+d_i] = 1.0 

cdef PetscErrorCode top_bc(PetscInt dim, PetscReal time, const PetscReal coords[], 
                                  PetscInt Nf, PetscScalar *u, void *ctx):
    u[0] = 0.
    return 0

cdef PetscErrorCode bottom_bc(PetscInt dim, PetscReal time, const PetscReal coords[], 
                                 PetscInt Nf, PetscScalar *u, void *ctx):
    u[0] = 1.
    return 0

from petsc4py import PETSc
    
class Poisson:

    def __init__(self, mesh):
        self.mesh = mesh
        dim = mesh.plex.getDimension()
        self.fe_temp = PETSc.FE().createDefault(dim, 1, False, PETSc.DEFAULT, "temperature_", PETSc.COMM_WORLD)
        mesh.plex.setField(0,self.fe_temp)
        mesh.plex.createDS()
        self._k = 1.
        self._h = 0.

        super().__init__()

    @property
    def temperature(self):
        return self.fe_temp

    @property
    def k(self):
        return self._k
    @k.setter
    def k(self, value):
        # should add test here to make sure k is conformal
        self._k = value

    @property
    def h(self):
        return self._h
    @h.setter
    def h(self, value):
        # should add test here to make sure h is conformal
        self._h = value

    def solve(self):
        cdef PetscInt ids[4]
        ids[:] = [1,2,3,4]
        cdef DS ds = self.mesh.plex.getDS()

        cdef PtrContainer clsguy = self._getext()

        PetscDSSetResidual(ds.ds, 0, clsguy.f0_u, clsguy.f1_u)
        PetscDSSetJacobian(ds.ds, 0, 0, NULL, NULL, NULL, g3_uu)
        PetscDSAddBoundary(ds.ds, 1, NULL, "marker", 0, 0, NULL, <void (*)()>top_bc,    1, &ids[2], NULL)
        PetscDSAddBoundary(ds.ds, 1, NULL, "marker", 0, 0, NULL, <void (*)()>bottom_bc, 1, &ids[0], NULL)
        self.mesh.plex.setUp()

        self.mesh.plex.createClosureIndex(None)
        cdef DM dm = self.mesh.plex
        DMPlexSetSNESLocalFEM(dm.dm, NULL, NULL, NULL)
        self.u = self.mesh.plex.createGlobalVector()

        self.mesh.snes.setDM(self.mesh.plex)
        self.mesh.snes.setFromOptions()
        self.mesh.snes.solve(None,self.u)

    _ext_dict = {}
    def _getext(self):
        """
        Check if we've already created an equivalent extension
        and use if available.
        """
        hashparams = abs(hash((self.k,self.h)))
        try:
            module = self._ext_dict[hashparams]
        except KeyError:
            self._createext(hashparams)
            module = self._ext_dict[hashparams]
        return module.getptrobj()

    def _createext(self, name):
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
        from sympy import symbols, Eq
        out = symbols("out")
        eqnk = Eq(out, self.k)
        eqnh = Eq(out, self.h)

        # Generate C code from Sympy expressions
        MODNAME = "fn_ptr_ext_" + str(name)
        from sympy.utilities.codegen import codegen
        codeguys  = codegen((("eqn_k", eqnk),("eqn_h", eqnh)), prefix="fns", argument_sequence=(self.mesh.x, out), language='c')

        # Create a `setup.py`
        setup_py_str = """
try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

ext_mods = [Extension(
    'NAME', ['cy_ext.pyx', 'fns.c'],
    include_dirs=[np.get_include()],
    library_dirs=[],
    libraries=[],
    extra_compile_args=['-std=c99'],
    extra_link_args=[]
)]
setup(ext_modules=cythonize(ext_mods))
        """.replace("NAME",MODNAME)
        codeguys.append( ["setup.py", setup_py_str] )

        # Create required Cython extension
        pyx_str = """
cdef extern from "fns.h":
    void eqn_k(double*, double*)
    void eqn_h(double*, double*)
 
from underworld3.petsc_types cimport PetscInt, PetscReal, PetscScalar, PetscErrorCode, PetscBool, DMBoundaryConditionType, PetscDSResidualFn, PetscDSJacobianFn
from underworld3.petsc_types cimport PtrContainer

cdef void f0_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[]):

    # now call the C function
    cdef PetscReal h[1]
    eqn_h(<double *> x, <double *> h)
    f0[0] = h[0]

cdef void f1_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[]):

    # now call the C function
    cdef PetscReal k[1]
    eqn_k(<double *> x, <double *> k)
    for d_i in range(dim):
        f0[d_i] = k[0]*u_x[d_i]

cpdef PtrContainer getptrobj():
    clsguy = PtrContainer()
    clsguy.f0_u = f0_u
    clsguy.f1_u = f1_u
    return clsguy
        """
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
        process = subprocess.Popen('python setup.py build_ext --inplace'.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=tmpdir)
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
                self._ext_dict[name] = load_dynamic(MODNAME, os.path.join(tmpdir,_file))

        if name not in self._ext_dict.keys():
            raise RuntimeError("Extension module does not appear to have been created.")
