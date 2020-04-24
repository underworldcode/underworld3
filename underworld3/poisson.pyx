from petsc4py.PETSc cimport DM, PetscDM, DS, PetscDS, Vec, PetscVec
from .petsc_types cimport PetscInt, PetscReal, PetscScalar, PetscErrorCode, PetscBool, DMBoundaryConditionType, PetscDSResidualFn, PetscDSJacobianFn
from .petsc_types cimport PtrContainer
import underworld3 as uw
from sympy import sympify
# TODO
# gil v nogil 
# ctypeds DMBoundaryConditionType etc.. is there a cleaner way? 

cdef extern from "petsc.h":
    PetscErrorCode PetscDSAddBoundary( PetscDS, DMBoundaryConditionType, const char[], const char[], PetscInt, PetscInt, const PetscInt *, void (*)(), PetscInt, const PetscInt *, void *)

cdef extern from "petsc.h" nogil:
    PetscErrorCode PetscDSSetResidual( PetscDS, PetscInt, PetscDSResidualFn, PetscDSResidualFn )
    PetscErrorCode PetscDSSetJacobian( PetscDS, PetscInt, PetscInt, PetscDSJacobianFn, PetscDSJacobianFn, PetscDSJacobianFn, PetscDSJacobianFn)
    PetscErrorCode DMPlexSetSNESLocalFEM( PetscDM, void *, void *, void *)
    PetscErrorCode DMPlexSNESComputeBoundaryFEM( PetscDM, void *, void *)

from petsc4py import PETSc
    
class Poisson:

    def __init__(self, mesh, degree=1):
        self.mesh = mesh
        options = PETSc.Options()
        options.setValue("u_petscspace_degree", degree)
        self._u = uw.MeshVariable( 
                                    mesh = mesh, 
                                    num_components = 1,
                                    name = "u", 
                                    isSimplex = False)
        mesh.plex.createDS()
        self._k = 1.
        self._h = 0.

        self.bc_fns = []
        self.bc_inds = []

        super().__init__()

    @property
    def u(self):
        return self._u

    @property
    def k(self):
        return self._k
    @k.setter
    def k(self, value):
        # should add test here to make sure k is conformal
        self._k = sympify(value)

    @property
    def h(self):
        return self._h
    @h.setter
    def h(self, value):
        # should add test here to make sure h is conformal
        self._h = sympify(value)

    def add_dirichlet_bc(self, fn, indices):
        if not isinstance(indices,list):
            indices = [indices,]

        # switch to numpy arrays
        import numpy
        indices = numpy.array(indices, dtype=long)
        
        self.bc_fns.append(sympify(fn))
        self.bc_inds.append(indices)

    def solve(self):

        cdef DS ds = self.mesh.plex.getDS()

        from sympy.vector import gradient
        import sympy

        def diff_fn_wrt_fn(fn_to_diff, wrt_fn):
            dummy = sympy.Symbol("dummy")
            return fn_to_diff.subs(wrt_fn,dummy).diff(dummy).subs(dummy,wrt_fn)

        N = self.mesh.N

        # f0 residual term
        self._f0 = -self.h
        # f1 residual term
        self._f1 = gradient(self.u.fn)*self.k
        # g0 jacobian term
        self._g0 = -diff_fn_wrt_fn(self.h,self.u)
        # g1 jacobian term
        dk_du = diff_fn_wrt_fn(self.k,self.u)
        self._g1 = dk_du*gradient(self.u.fn)
        # g3 jacobian term
        dk_dux = diff_fn_wrt_fn(self.k, self.u.fn.diff(N.x))
        dk_duy = diff_fn_wrt_fn(self.k, self.u.fn.diff(N.y))
        dk_duz = diff_fn_wrt_fn(self.k, self.u.fn.diff(N.z))
        dk = dk_dux*N.i + dk_duy*N.j + dk_duz*N.k
        self._g3  = dk|gradient(self.u.fn)                        # outer product for nonlinear part
        self._g3 += self.k*( (N.i|N.i) + (N.j|N.j) + (N.k|N.k) )  # linear part using dyadic identity

        fns_residual = (self._f0, self._f1)
        fns_jacobian = (self._g0, self._g1, self._g3)
        fns_bcs      = self.bc_fns

        # generate JIT code
        cdef PtrContainer clsguy = self._getext(fns_residual, fns_jacobian, fns_bcs)

        # set functions 
        PetscDSSetResidual(ds.ds, 0, clsguy.fns_residual[0], clsguy.fns_residual[1])
        # TODO: check if there's a significant performance overhead in passing in 
        # identically `zero` pointwise functions instead of setting to `NULL`
        PetscDSSetJacobian(ds.ds, 0, 0, clsguy.fns_jacobian[0], clsguy.fns_jacobian[1], NULL, clsguy.fns_jacobian[2])
        cdef int indcount        # to record length of array
        cdef long [:] narr_view   # for numpy memory view
        for index,indices in enumerate(self.bc_inds):
            indcount = len(indices)
            narr_view = indices
            # use type 5 bc for `DM_BC_ESSENTIAL_FIELD` enum
            PetscDSAddBoundary(ds.ds, 5, NULL, "marker", 0, 0, NULL, <void (*)()>clsguy.fns_bcs[index], indcount, <const PetscInt *> &narr_view[0], NULL)
        self.mesh.plex.setUp()

        self.mesh.plex.createClosureIndex(None)
        cdef DM dm = self.mesh.plex
        DMPlexSetSNESLocalFEM(dm.dm, NULL, NULL, NULL)
        self.u_global = self.mesh.plex.createGlobalVector()
        self.u_local  = self.mesh.plex.createLocalVector()

        self.mesh.snes.setDM(self.mesh.plex)
        self.mesh.snes.setFromOptions()
        self.mesh.snes.solve(None,self.u_global)
        self.mesh.plex.globalToLocal(self.u_global,self.u_local)
        # add back boundaries.. 
        cdef Vec lvec= self.u_local
        DMPlexSNESComputeBoundaryFEM(dm.dm, <void*>lvec.vec, NULL)


    _ext_dict = {}
    def _getext(self, fns_residual, fns_jacobian, fns_bcs):
        """
        Check if we've already created an equivalent extension
        and use if available.
        """
        fns = fns_residual + fns_jacobian + tuple(fns_bcs)
        hashparams = abs(hash(fns))
        try:
            module = self._ext_dict[hashparams]
        except KeyError:
            self._createext(hashparams, fns_residual, fns_jacobian, fns_bcs)
            module = self._ext_dict[hashparams]
        return module.getptrobj()

    def _createext(self, name, fns_residual, fns_jacobian, fns_bcs):
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

        # note that the order here is important.
        fns = fns_residual + tuple(fns_bcs) + fns_jacobian
        count_residual_sig = len(fns_residual) + len(fns_bcs)
        count_jacobian_sig = len(fns_jacobian)
        # get fn/fn_grad component totals
        tot_fns = 0
        tot_grad_fns = 0
        for var in self.mesh.vars.values():
            tot_fns += var.num_components
            tot_grad_fns += var.num_components*self.mesh.dim

        # Create some symbol which will force codegen to produce required interface.
        # Create MatrixSymbol objects, which sympy simply converts to double* within 
        # the generated code. 
        petsc_x   = MatrixSymbol(  'petsc_x', 1, 3)  # let's just set this to 3-dim, as it'll be 
                                                     # the max and doesn't matter otherwise.
        petsc_u   = MatrixSymbol(  'petsc_u', 1, tot_fns)
        petsc_u_x = MatrixSymbol('petsc_u_x', 1, tot_grad_fns)
        petsc_a   = MatrixSymbol(  'petsc_a', 1, 1)  # TODO
        petsc_a_x = MatrixSymbol('petsc_a_x', 1, 1)  # TODO

        # create subs dictionary
        subs = {}
        for index, base_scalar in enumerate(self.mesh.N.base_scalars()):
            subs[base_scalar] = petsc_x[index]
        u_i = 0
        u_x_i = 0
        for var in self.mesh.vars.values():
            if var.num_components==1:
                # Substitute all instances of the mesh var with the required c pointer 
                subs[var.fn] = petsc_u[u_i]
                u_i +=1
                # Now process gradients. Simply generate the required derivative in place, 
                # and set to be substituted by c pointer
                for base_scalar in self.mesh.N.base_scalars()[0:self.mesh.dim]:
                    subs[var.fn.diff(base_scalar)] = petsc_u_x[u_x_i]
                    u_x_i += 1
            else:
                raise RuntimeError("TODO: Implement vector field codegen.")        

        # do subsitutions
        subbedfns = []
        for fn in fns:
            subbedfns.append(fn.subs(subs))

        import sympy
        # Generate C code from Sympy expressions
        from sympy.printing import ccode
        eqns = []
        for index, fn in enumerate(subbedfns):
            if isinstance(fn, sympy.vector.Vector):
                fn = fn.to_matrix(self.mesh.N)[0:self.mesh.dim,0]
            elif isinstance(fn, sympy.vector.Dyadic):
                fn = fn.to_matrix(self.mesh.N)[0:self.mesh.dim,0:self.mesh.dim]
            else:
                fn = sympy.Matrix([fn])
            out = sympy.MatrixSymbol("out",*fn.shape)
            eqns.append( ("eqn_"+str(index), ccode(fn, out)) )
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
import numpy as np

ext_mods = [Extension(
    'NAME', ['cy_ext.pyx',],
    include_dirs=[np.get_include()],
    library_dirs=[],
    libraries=[],
    extra_compile_args=['-std=c99'],
    extra_link_args=[]
)]
setup(ext_modules=cythonize(ext_mods))
        """.replace("NAME",MODNAME)
        codeguys.append( ["setup.py", setup_py_str] )

        pyx_str="""
from underworld3.petsc_types cimport PetscInt, PetscReal, PetscScalar, PetscErrorCode, PetscBool, DMBoundaryConditionType, PetscDSResidualFn, PetscDSJacobianFn
from underworld3.petsc_types cimport PtrContainer
from libc.stdlib cimport malloc
from libc.math cimport *
"""

        for eqn in eqns[0:count_residual_sig]:
            pyx_str+="""
cdef void petsc_{}(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar petsc_u[], const PetscScalar petsc_u_t[], const PetscScalar petsc_u_x[],
                const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar petsc_a[], const PetscScalar petsc_a_t[], const PetscScalar petsc_a_x[],
                PetscReal petsc_t, const PetscReal petsc_x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar out[]):
    {}
""".format(eqn[0],eqn[1].replace("\n","\n    "))   # `replace` here is for required python/cython indenting

        for eqn in eqns[count_residual_sig:]:
            pyx_str+="""
cdef void petsc_{}(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar petsc_u[], const PetscScalar petsc_u_t[], const PetscScalar petsc_u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar petsc_a[], const PetscScalar petsc_a_t[], const PetscScalar petsc_a_x[],
                  PetscReal petsc_t, PetscReal petsc_u_tShift, const PetscReal petsc_x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar out[]):
    {}
""".format(eqn[0],eqn[1].replace("\n","\n    "))   # `replace` here is for required python/cython indenting

        # Note that the malloc below will cause a leak, but it's just a bunch of function
        # pointers so we don't need to worry about it
        pyx_str+="""
cpdef PtrContainer getptrobj():
    clsguy = PtrContainer()
    clsguy.fns_residual = <PetscDSResidualFn*> malloc({}*sizeof(PetscDSResidualFn))  
    clsguy.fns_jacobian = <PetscDSJacobianFn*> malloc({}*sizeof(PetscDSJacobianFn))
    clsguy.fns_bcs      = <PetscDSResidualFn*> malloc({}*sizeof(PetscDSResidualFn))  
""".format(count_residual_sig,count_jacobian_sig,len(fns_bcs)) 

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
