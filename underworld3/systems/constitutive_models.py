import sympy
from sympy import sympify
from sympy.vector import gradient, divergence
import numpy as np 

from typing import Optional, Callable
from typing import NamedTuple, Union

from petsc4py import PETSc

import underworld3 as uw
from   underworld3.systems import SNES_Scalar, SNES_Vector, SNES_SaddlePoint
import underworld3.timing as timing


## This file has constitutive models that can be plugged into the SNES solvers

class Constitutive_Model:
    """
    This is the base class for constitutive models and 
    defines scalar / tensor constitutive equations for 
    the snes solver objects currently in use. 

    It defines the matrix / tensor representations of the material
    properties for the generic solver classes.

    """

    @timing.routine_timer_decorator
    def __init__( self, 
                  dim: int,
                  u_dim: int
        ):

        # Define / identify the various properties in the class but leave 
        # the implementation to child classes. The constitutive tensor is
        # defined as a template here, but should be instantiated via class
        # properties as required.

        # We provide a function that converts gradients / gradient history terms
        # into the relevant flux term. 

        self.dim = dim
        self.u_dim = u_dim
        self._solver = None

        self._material_properties = None

        ## Default consitutive tensor is the identity

        if self.u_dim == 1:
            self._c = sympy.Matrix.eye(self.dim);
        else: # vector problem
            self._c = uw.maths.tensor.rank4_identity(self.dim)    

        self._C = None
        self._symbolic_form()

        super().__init__()


    class Parameters(NamedTuple):
            eta_0 : Union[ float, sympy.Function ]    
            eta_1 : Union[ float, sympy.Function ]  

    @property
    def material_properties(self):
        return self._material_properties

    @material_properties.setter
    def material_properties(self, properties):

        if isinstance(properties, self.Parameters):
            self._material_properties = properties
        else:
            name = self.__class__.__name__
            raise RuntimeError(f"Use {name}.para = {name}.Parameters(...) ")

        d = self.dim
        self._build_c_tensor()

        if isinstance(self._solver, (SNES_Scalar, SNES_Vector, SNES_SaddlePoint)):
            self._solver.is_setup = False

        return

    @property
    def solver(self):
        return self._solver   

    @solver.setter
    def solver(self, solver_object):

        if isinstance(solver_object, (SNES_Scalar, SNES_Vector, SNES_SaddlePoint)):
            self._solver = solver_object
            self._solver.is_setup = False

    ## Properties on all sub-classes

    @property
    def C(self):
        """The matrix form of the constitutive model that relates fluxes to gradients
           - for scalar problem, this is the matrix representation of the rank 2 tensor
           - for vector problems, the Mandel form of the rank 4 tensor is returned
           NOTE: this is an immutable object that is _a view_ of the underlying tensor
           """        

        # c is either a rank 2 or a rank 4 tensor

        d = self.dim

        if self._c.rank() == 2:
            return sympy.Matrix(self._c).as_immutable()
        else: 
            return uw.maths.tensor.rank4_to_mandel(self._c, d).as_immutable()

    @property
    def c(self):
        """The tensor form of the constitutive model that relates fluxes to gradients"""        

        return self._c.as_immutable()

    def flux(self, 
             gradient    :  sympy.Matrix = None, 
             gradient_dt :  sympy.Matrix = None ):
                 
        """Fluxes from the gradients of the unknowns - essentially this is just a wrapper 
           for the tensor multiplication that will differ depending on the size of the 
           unknown vector. 

           Assume that the history of the gradients is available if required. 
        """

        c = self.c
        r = c.rank() 

        # tensor multiplication

        if r==2:
            flux = c * gradient.T
        else: # r==4
            flux = sympy.tensorcontraction(sympy.tensorcontraction(sympy.tensorproduct(c, gradient),(3,5)),(2,3))
        
        return sympy.Matrix(flux)

    def _build_c_tensor(self):
        """Return the identity tensor of appropriate rank (e.g. for projections)"""
           
        d = self.dim
        self._c = uw.maths.tensor.rank4_identity(d) 

        return

    def _symbolic_form(self):
        """This function creates symbolic forms of the constitutive law
            for documentation / introspection."""

        d = self.dim

        # Scalar equation

        if self.u_dim == 1: 
            Q = sympy.Matrix(sympy.symarray(r'q',(d,)))
            C = uw.maths.tensor.rank2_symmetric_sym(r'k', d)
            E = sympy.Matrix(sympy.symarray(r'\partial\phi',(d,)))

            self.equation = sympy.core.relational.Eq(sympy.UnevaluatedExpr(Q), sympy.UnevaluatedExpr(C) * sympy.UnevaluatedExpr(E))

        else:
            tau = uw.maths.tensor.rank2_to_voigt(uw.maths.tensor.rank2_symmetric_sym(r"T", d), d)
            visc = uw.maths.tensor.rank4_to_voigt(uw.maths.tensor.rank4_symmetric_sym(r'c',d),d)
            edot = uw.maths.tensor.rank2_to_voigt(uw.maths.tensor.rank2_symmetric_sym(r"\dot\varepsilon", d), d)

            self.equation = sympy.core.relational.Eq(sympy.UnevaluatedExpr(tau.T), sympy.UnevaluatedExpr(visc) * sympy.UnevaluatedExpr(edot.T))

        return


class ViscousFlowModel(Constitutive_Model):
    """Viscous flow """

    class Parameters(NamedTuple):
        """-"""
        viscosity : Union[ float, sympy.Function ]    

    def __init__(self, dim):

        u_dim = dim
        super().__init__(dim, u_dim)

        return

    def _build_c_tensor(self):
        """For this constitutive law, we expect just a viscosity function"""
           
        d = self.dim
        viscosity = self._material_properties.viscosity
        self._c = uw.maths.tensor.rank4_identity(d) * viscosity

        return


    def _symbolic_form(self):

        """This function creates symbolic forms of the constitutive law
            for documentation / introspection."""

        d = self.dim

        # Scalar equation

        tau = uw.maths.tensor.rank2_to_voigt(uw.maths.tensor.rank2_symmetric_sym(r"\tau", d), d)
        visc = uw.maths.tensor.rank4_to_voigt(uw.maths.tensor.rank4_symmetric_sym(r'\eta',d),d)
        edot = uw.maths.tensor.rank2_to_voigt(uw.maths.tensor.rank2_symmetric_sym(r"\dot\varepsilon", d), d)

        self.equation = sympy.core.relational.Eq(sympy.UnevaluatedExpr(tau.T), sympy.UnevaluatedExpr(visc) * sympy.UnevaluatedExpr(edot.T))

        return

### 

class DiffusionModel(Constitutive_Model):
    """Scalar Diffusion """

    class Parameters(NamedTuple):
        """-"""
        diffusivity : Union[ float, sympy.Function ]    

    def __init__(self, dim):

        self.u_dim = 1
        super().__init__(dim, self.u_dim)

        return

    def _build_c_tensor(self):
        """For this constitutive law, we expect just a viscosity function"""
           
        d = self.dim
        kappa = self._material_properties.diffusivity
        self._c = uw.maths.tensor.rank2_identity(d) * kappa

        return 


    def _symbolic_form(self):
        """This function creates symbolic forms of the constitutive law
           for documentation / introspection."""

        d = self.dim

        # Scalar equation

        Q = sympy.Matrix(sympy.symarray(r'q',(d,)))
        K = uw.maths.tensor.rank2_symmetric_sym(r'\kappa', d)
        Tx = sympy.Matrix(sympy.symarray(r'\partial\phi',(d,)))

        self.equation = sympy.core.relational.Eq(sympy.UnevaluatedExpr(Q), sympy.UnevaluatedExpr(K) * sympy.UnevaluatedExpr(Tx))

        return


class TransverseIsotropicFlowModel(Constitutive_Model):
    """Director-oriented, transversely isotropic flow model"""

    class Parameters(NamedTuple):
        """Transversely isotropic rheology -
        - eta_0: normal viscosity
        - eta_1: shear viscosity
        - director: orientation of weak plane
        See Sharples et al, 2015 for details"""

        eta_0: Union[ float, sympy.Function ]    
        eta_1: Union[ float, sympy.Function ]
        director: sympy.Matrix

    def __init__(self, dim):

        u_dim = dim
        super().__init__(dim, u_dim)

        # default values ... maybe ??
        return

    def _build_c_tensor(self):
        """For this constitutive law, we expect two viscosity functions 
           and a sympy matrix that describes the director components n_{i}"""
           
        d = self.dim
        dv = uw.maths.tensor.idxmap[d][0]

        eta_0 = self._material_properties.eta_0
        eta_1 = self._material_properties.eta_1       
        n = self._material_properties.director

        Delta = eta_1 - eta_0

        lambda_mat = uw.maths.tensor.rank4_identity(d) * eta_0

        for i in range(d):
            for j in range(d):
                for k in range(d):
                    for l in range(d):
                        lambda_mat[i,j,k,l] += Delta * ((n[i]*n[k]*int(j==l) + n[j]*n[k] * int(l==i) + 
                                                        n[i]*n[l]*int(j==k) + n[j]*n[l] * int(k==i))/2 
                                                        - 2 * n[i]*n[j]*n[k]*n[l] )

        lambda_mat = sympy.simplify(uw.maths.tensor.rank4_to_mandel(lambda_mat,d))

        self._c = uw.maths.tensor.mandel_to_rank4(lambda_mat, d)


    def _symbolic_form(self):

        """This function creates symbolic forms of the constitutive law
            for documentation / introspection."""

        d = self.dim

        # Scalar equation

        tau = uw.maths.tensor.rank2_to_voigt(uw.maths.tensor.rank2_symmetric_sym(r"\tau", d), d)
        visc = uw.maths.tensor.rank4_to_voigt(uw.maths.tensor.rank4_symmetric_sym(r'\eta',d),d)
        edot = uw.maths.tensor.rank2_to_voigt(uw.maths.tensor.rank2_symmetric_sym(r"\dot\varepsilon", d), d)

        self.equation = sympy.core.relational.Eq(sympy.UnevaluatedExpr(tau.T), sympy.UnevaluatedExpr(visc) * sympy.UnevaluatedExpr(edot.T))

        return





  



