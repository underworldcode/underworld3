from petsc4py.PETSc cimport DS
from .petsc_types cimport PetscInt, PetscReal, PetscScalar, PetscErrorCode, PetscBool, DMBoundaryConditionType, PetscDSResidualFn, PetscDSJacobianFn
from .petsc_types cimport PtrContainer
# TODO
# gil v nogil 
# ctypeds DMBoundaryConditionType etc.. is there a cleaner way? 

cdef extern from "petsc.h":
    PetscErrorCode PetscDSAddBoundary( DS, DMBoundaryConditionType, const char[], const char[], PetscInt, PetscInt, const PetscInt *, void (*)(), PetscInt, const PetscInt *, void *)

cdef extern from "petsc.h" nogil:
    PetscErrorCode PetscDSSetResidual( DS, PetscInt, PetscDSResidualFn, PetscDSResidualFn )
    PetscErrorCode PetscDSSetJacobian( DS, PetscInt, PetscInt, PetscDSJacobianFn, PetscDSJacobianFn, PetscDSJacobianFn, PetscDSJacobianFn)
    PetscErrorCode DMPlexSetSNESLocalFEM( DM, void *, void *, void *)



cdef void f0_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[]):
    f0[0] = -1#*constants[1]

cdef void f1_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[]):
    cdef PetscInt d_i
    for d_i in range(dim):
        # f0[d_i] = constants[0] * u_x[d_i]
        f0[d_i] = 1.*u_x[d_i]

cdef void g3_uu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[]):
    cdef PetscInt d_i
    for d_i in range(dim):
        g3[d_i*dim+d_i] = 1.0; 

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

    def solve(self):
        cdef PetscInt ids[4]
        ids[:] = [1,2,3,4]
        cdef DS ds = self.mesh.plex.getDS()

        self._buildext()
        import fn_ptr_ext
        cdef PtrContainer clsguy = fn_ptr_ext.getptrobj()

        PetscDSSetResidual(ds, 0, f0_u, clsguy.residual_ptr)
        PetscDSSetJacobian(ds, 0, 0, NULL, NULL, NULL, g3_uu)
        PetscDSAddBoundary(ds, 1, NULL, "marker", 0, 0, NULL, <void (*)()>top_bc,    1, &ids[2], NULL)
        PetscDSAddBoundary(ds, 1, NULL, "marker", 0, 0, NULL, <void (*)()>bottom_bc, 1, &ids[0], NULL)
        self.mesh.plex.setUp()

        self.mesh.plex.createClosureIndex(None)
        DMPlexSetSNESLocalFEM(self.mesh.plex, NULL, NULL, NULL)
        u = self.mesh.plex.createGlobalVector()

        self.mesh.snes.setDM(self.mesh.plex)
        self.mesh.snes.setFromOptions()
        self.mesh.snes.solve(None,u)

    def _buildext(self):
        from sympy import symbols, Eq
        k_out = symbols("k_out")
        eqn = Eq(k_out, self.k)

        NAME = "fn_ptr_ext"
        from sympy.utilities.codegen import codegen
        codeguys  = codegen((NAME, eqn), argument_sequence=(self.mesh.x, k_out), language='c')

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
    'NAME', ['NAME_cy.pyx', 'NAME.c'],
    include_dirs=[np.get_include()],
    library_dirs=[],
    libraries=[],
    extra_compile_args=['-std=c99'],
    extra_link_args=[]
)]
setup(ext_modules=cythonize(ext_mods))
        """.replace("NAME",NAME)

        codeguys.append( ["setup.py",setup_py_str])

        pyx_str = """
cdef extern from "NAME.h":
    void NAME(double *x, double *y);
 
from underworld3.petsc_types cimport PetscInt, PetscReal, PetscScalar, PetscErrorCode, PetscBool, DMBoundaryConditionType, PetscDSResidualFn, PetscDSJacobianFn
from underworld3.petsc_types cimport PtrContainer

cdef void NAME_PETSc(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[]):

    # now call the C function
    cdef PetscReal k[1]
    NAME(<double *> x, <double *> k)
    for d_i in range(dim):
        f0[d_i] = k[0]*u_x[d_i]

cpdef PtrContainer getptrobj():
    clsguy = PtrContainer()
    clsguy.residual_ptr = NAME_PETSc
    return clsguy
        """.replace("NAME",NAME)
        codeguys.append( [NAME+"_cy.pyx",pyx_str])

        import os
        tmpdir = os.path.join("/tmp",NAME)
        try:
            os.mkdir(tmpdir)
        except OSError:
            pass
        for thing in codeguys:
            filename = thing[0]
            strguy   = thing[1]
            with open(os.path.join(tmpdir,filename),'w') as f:
                f.write(strguy)

        import subprocess
        process = subprocess.Popen('python setup.py build_ext --inplace'.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=tmpdir)
        process.communicate()

        import sys
        sys.path.insert(0,tmpdir)

        import fn_ptr_ext
        from importlib import reload  
        fn_ptr_ext = reload(fn_ptr_ext)
