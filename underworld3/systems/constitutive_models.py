## This file has constitutive models that can be plugged into the SNES solvers

import sympy
from sympy import sympify
from sympy.vector import gradient, divergence
import numpy as np 

from typing import Optional, Callable
from typing import NamedTuple, Union

from petsc4py import PETSc

import underworld3 as uw
from underworld3.systems import SNES_Scalar, SNES_Vector, SNES_SaddlePoint
import underworld3.timing as timing


class Constitutive_Model:
    r'''           
        Constititutive laws relate gradients in the unknowns to fluxes of quantities 
        (for example, heat fluxes are related to temperature gradients through a thermal conductivity)
        The `Constitutive_Model` class is a base class for building `underworld` constitutive laws

        In a scalar problem, the relationship is 

         $$ q_i = k_{ij} \frac{\partial T}{\partial x_j}$$

        and the constitutive parameters describe \( k_{ij}\). The template assumes \( k_{ij} = \delta_{ij} \)

        In a vector problem (such as the Stokes problem), the relationship is 

         $$ t_{ij} = c_{ijkl} \frac{\partial u_k}{\partial x_l} $$

        but is usually written to eliminate the anti-symmetric part of the displacement or velocity gradients (for example):

         $$ t_{ij} = c_{ijkl} \frac{1}{2} \left[ \frac{\partial u_k}{\partial x_l} + \frac{\partial u_l}{\partial x_k} \right] $$

        and the constitutive parameters describe \(c_{ijkl}\). The template assumes 
        \( k_{ij} = \frac{1}{2} \left( \delta_{ik} \delta_{jl} + \delta_{il} \delta_{jk} \right) \) which is the
        4th rank identity tensor accounting for symmetry in the flux and the gradient terms. 
        '''

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

        super().__init__()


    class Parameters(NamedTuple):
        '''Any material properties that are defined by a constitutive relationship are 
           collected in the parameters which can then be defined/accessed by name in 
           individual instances of the class.
        '''
        k : Union[ float, sympy.Function ]    
        

    @property
    def material_properties(self):
        '''The material properties corresponding the the values (functions) for 
        the `Parameters`'''
        return self._material_properties

    @material_properties.setter
    def material_properties(self, properties):

        if isinstance(properties, self.Parameters):
            self._material_properties = properties
        else:
            name = self.__class__.__name__
            raise RuntimeError(f"Use {name}.material_properties = {name}.Parameters(...) ")

        d = self.dim
        self._build_c_tensor()

        if isinstance(self._solver, (SNES_Scalar, SNES_Vector, SNES_SaddlePoint)):
            self._solver.is_setup = False

        return

    @property
    def solver(self):
        '''Each constitutive relationship can, optionally, be associated with one solver object.
           and a solver object _requires_ a constitive relationship to be defined.'''
        return self._solver   
    @solver.setter
    def solver(self, solver_object):
        if isinstance(solver_object, (SNES_Scalar, SNES_Vector, SNES_SaddlePoint)):
            self._solver = solver_object
            self._solver.is_setup = False

    ## Properties on all sub-classes

    @property
    def C(self):
        """The matrix form of the constitutive model (the `c` property) 
           that relates fluxes to gradients.
           For scalar problem, this is the matrix representation of the rank 2 tensor.
           For vector problems, the Mandel form of the rank 4 tensor is returned.
           NOTE: this is an immutable object that is _a view_ of the underlying tensor
           """        

        d = self.dim
        rank = len(self.c.shape)

        if rank == 2:
            return sympy.Matrix(self._c).as_immutable()
        else: 
            return uw.maths.tensor.rank4_to_mandel(self._c, d).as_immutable()

    @property
    def c(self):
        """The tensor form of the constitutive model that relates fluxes to gradients. In scalar
        problems, `c` and `C` are equivalent (matrices), but in vector problems, `c` is a 
        rank 4 tensor. NOTE: `c` is the canonical form of the constitutive relationship. """        

        return self._c.as_immutable()

    def flux(self, 
             gradient    :  sympy.Matrix = None, 
             gradient_dt :  sympy.Matrix = None ):
                 
        """Computes the effect of the constitutive tensor on the gradients of the unknowns.  
           (always uses the `c` form of the tensor). In general cases, the history of the gradients
           may be required to evaluate the flux.
        """

        c = self.c
        rank = len(c.shape)

        # tensor multiplication

        if rank==2:
            flux = c * gradient.T
        else: # rank==4
            flux = sympy.tensorcontraction(sympy.tensorcontraction(sympy.tensorproduct(c, gradient),(3,5)),(2,3))
        
        return sympy.Matrix(flux)

    def _build_c_tensor(self):
        """Return the identity tensor of appropriate rank (e.g. for projections)"""
           
        d = self.dim
        self._c = uw.maths.tensor.rank4_identity(d) 

        return


    def _ipython_display_(self):
        from IPython.display import Latex, Markdown, display
        from textwrap import dedent

        ## Docstring (static)
        docstring = dedent(self.__doc__)
        docstring = docstring.replace(r'\(','$').replace(r'\)','$')
        display(Markdown(docstring))
        display(Markdown(fr"This consititutive model is formulated for {self.dim} dimensional equations"))

        ## Usually, there are constitutive parameters that can be included in the iputho display 



class ViscousFlowModel(Constitutive_Model):
    r'''
        ```python
        class ViscousFlowModel(Constitutive_Model)
        ...
        ```
        ```python
        viscous_model = ViscousFlowModel(dim)
        viscous_model.material_properties = viscous_model.Parameters(viscosity=viscosity_fn)
        solver.constititutive_model = viscous_model
        ```
        $$ \tau_{ij} = \eta_{ijkl} \cdot \frac{1}{2} \left[ \frac{\partial u_k}{\partial x_l} + \frac{\partial u_l}{\partial x_k} \right] $$

        where \( \eta \) is the viscosity, a scalar constant, `sympy` function, `underworld` mesh variable or
        any valid combination of those types. This results in an isotropic (but not necessarily homogeneous or linear)
        relationship between $\tau$ and the velocity gradients. You can also supply \(\eta_{IJ}\), the Mandel form of the 
        constitutive tensor, or \(\eta_{ijkl}\), the rank 4 tensor. 

        The Mandel constitutive matrix is available in `viscous_model.C` and the rank 4 tensor form is
        in `viscous_model.c`.  Apply the constitutive model using:

        ```python
        tau = viscous_model.flux(gradient_matrix)
        ```
        ---
    '''
    class Parameters(NamedTuple):
        """The viscous flow law relates stresses to velocity gradients through the *viscosity* 
        which is a scalar function of position or a fourth-rank tensor"""
        viscosity : Union[ float, sympy.Function ]    

    def __init__(self, dim):

        u_dim = dim
        super().__init__(dim, u_dim)

        return

    def _build_c_tensor(self):
        """For this constitutive law, we expect just a viscosity function"""
           
        d = self.dim
        viscosity = self._material_properties.viscosity

        try:
            self._c = uw.maths.tensor.rank4_identity(d) * viscosity
        except:
            d = self.dim
            dv = uw.maths.tensor.idxmap[d][0]
            if isinstance(viscosity, sympy.Matrix) and viscosity.shape == (dv,dv):
                self._c = uw.maths.tensor.mandel_to_rank4(viscosity, d)
            elif isinstance(viscosity, sympy.Array) and viscosity.shape == (d,d,d,d):
                self._c = viscosity
            else:
                raise RuntimeError("Viscosity is not a known type (scalar, Mandel matrix, or rank 4 tensor")
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


    def _ipython_display_(self):
        from IPython.display import Latex, Markdown, display
        super()._ipython_display_()

        ## feedback on this instance
        display(Latex(r"$\eta = $ "+ sympy.sympify(self.material_properties.viscosity)._repr_latex_()))

### 

class DiffusionModel(Constitutive_Model):
    r'''
        ```python
        class DiffusionModel(Constitutive_Model)
        ...
        ```
        ```python
        diffusion_model = DiffusionModel(dim)
        diffusion_model.material_properties = diffusion_model.Parameters(diffusivity=diffusivity_fn)
        scalar_solver.constititutive_model = diffusion_model
        ```
        $$ q_{i} = \kappa_{ij} \cdot \frac{\partial \phi}{\partial x_j}  $$

        where \( \kappa \) is a diffusivity, a scalar constant, `sympy` function, `underworld` mesh variable or
        any valid combination of those types. Access the constitutive model using:

        ```python
        flux = diffusion_model.flux(gradient_matrix)
        ```
        ---
        '''

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
        self._c = sympy.Matrix.eye(d) * kappa

        return 


    def _ipython_display_(self):
        from IPython.display import Latex, Markdown, display
        super()._ipython_display_()

        ## feedback on this instance
        display(Latex(r"$\kappa = $ "+ sympy.sympify(self.material_properties.diffusivity)._repr_latex_()) )

        return


class TransverseIsotropicFlowModel(Constitutive_Model):
    r'''
        ```python
        class TransverseIsotropicFlowModel(Constitutive_Model)
        ...
        ```
        ```python
        viscous_model = TransverseIsotropicFlowModel(dim)
        viscous_model.material_properties = viscous_model.Parameters(eta_0=viscosity_fn,
                                                                    eta_1=weak_viscosity_fn,
                                                                    director=orientation_vector_fn)
        solver.constititutive_model = viscous_model
        ```
        $$ \tau_{ij} = \eta_{ijkl} \cdot \frac{1}{2} \left[ \frac{\partial u_k}{\partial x_l} + \frac{\partial u_l}{\partial x_k} \right] $$

        where $\eta$ is the viscosity tensor defined as:

        $$ \eta_{ijkl} = \eta_0 \cdot I_{ijkl} + (\eta_0-\eta_1) \left[ \frac{1}{2} \left[
            n_i n_l \delta_{jk} + n_j n_k \delta_{il} + n_i n_l \delta_{jk} + n_j n_l \delta_{ik} \right] - 2 n_i n_j n_k n_l \right] $$

        and \( \hat{\mathbf{n}} \equiv \left\{ n_i \right\} \) is the unit vector 
        defining the local orientation of the weak plane (a.k.a. the director).

        The Mandel constitutive matrix is available in `viscous_model.C` and the rank 4 tensor form is
        in `viscous_model.c`.  Apply the constitutive model using: 

        ```python
        tau = viscous_model.flux(gradient_matrix)
        ```
        ---
        '''

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


    def _ipython_display_(self):
        from IPython.display import Latex, Markdown, display
        super()._ipython_display_()

        ## feedback on this instance
        display(Latex(r"$\eta_0 = $ "+ sympy.sympify(self.material_properties.eta_0)._repr_latex_()))
        display(Latex(r"$\eta_1 = $ "+ sympy.sympify(self.material_properties.eta_1)._repr_latex_()))
        display(Latex(r"$\hat{\mathbf{n}} = $ "+ sympy.sympify(self.material_properties.director.T)._repr_latex_()))

    



