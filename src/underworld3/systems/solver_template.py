"""
Template for creating a new solver class in Underworld3.

This template demonstrates the common structure and patterns used in
Underworld3 solver classes. Modify as needed for your specific PDE.
"""

import sympy
from typing import Optional, Union, Callable
import underworld3 as uw
from underworld3.systems import SNES_Scalar, SNES_Vector, SNES_Stokes_SaddlePt
from underworld3.systems.ddt import SemiLagrangian_DDt, Lagrangian_DDt, Eulerian_DDt
from underworld3 import timing
from underworld3.systems.solvers import expression


class SNES_MyEquation(SNES_Scalar):
    r"""
    This class provides functionality for a discrete representation
    of [Your Equation Name Here].

    The governing equation is:
    $$
    \color{Green}{\underbrace{ \Bigl[ \frac{\partial u}{\partial t} \Bigr]}_{\dot{\mathbf{u}}}} +
    \color{Blue}{\underbrace{\Bigl[ \nabla \cdot \mathbf{F}(u, \nabla u) \Bigr]}_{\text{Flux Term}}} =
    \color{Maroon}{\underbrace{\Bigl[ f(\mathbf{x}, t, u) \Bigr] }_{\text{Source Term}}}
    $$

    Where:
    - $u$ is the unknown scalar field
    - $\mathbf{F}(u, \nabla u)$ is the flux term (typically involving diffusion, advection, etc.)
    - $f(\mathbf{x}, t, u)$ is the source/sink term

    ## Properties

    - The unknown is $u$ (scalar field)
    - The constitutive model relates flux to gradients: $\mathbf{F} = -\kappa \nabla u$
    - Material properties are provided through the `constitutive_model` property
    - Source terms are specified through the `f` property

    ## Key Features

    - Supports time-dependent problems
    - Handles nonlinear constitutive relationships
    - Provides automatic timestep estimation
    - Compatible with various boundary conditions

    Parameters
    ----------
    mesh : uw.discretisation.Mesh
        The computational mesh
    u_Field : uw.discretisation.MeshVariable, optional
        The solution field. If None, will be created automatically
    degree : int, optional
        Polynomial degree for the finite element basis (default: 2)
    verbose : bool, optional
        Enable verbose output (default: False)
    DuDt : DDt object, optional
        Time derivative discretization method
    DFDt : DDt object, optional
        Flux time derivative discretization method

    Examples
    --------
    Create a basic solver:

    >>> import underworld3 as uw
    >>> mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0,0), maxCoords=(1,1))
    >>> solver = uw.systems.SNES_MyEquation(mesh, degree=2)
    >>> solver.f = 1.0  # Set source term
    >>> solver.solve()

    With custom field and constitutive model:

    >>> u_field = uw.discretisation.MeshVariable("u", mesh, 1, degree=2)
    >>> solver = uw.systems.SNES_MyEquation(mesh, u_Field=u_field)
    >>> solver.constitutive_model = uw.constitutive_models.DiffusionModel(mesh)
    >>> solver.constitutive_model.Parameters.diffusivity = 1.0
    """

    # Class variable to track instances
    instances = 0

    @timing.routine_timer_decorator
    def __init__(
        self,
        mesh: uw.discretisation.Mesh,
        u_Field: Optional[uw.discretisation.MeshVariable] = None,
        degree: int = 2,
        verbose: bool = False,
        DuDt: Optional[Union[SemiLagrangian_DDt, Lagrangian_DDt, Eulerian_DDt]] = None,
        DFDt: Optional[Union[SemiLagrangian_DDt, Lagrangian_DDt, Eulerian_DDt]] = None,
    ):
        """
        Initialize the solver.

        Parameters
        ----------
        mesh : uw.discretisation.Mesh
            The computational mesh
        u_Field : uw.discretisation.MeshVariable, optional
            Solution field (created if None)
        degree : int
            Polynomial degree for basis functions
        verbose : bool
            Enable verbose output
        DuDt : DDt object, optional
            Time derivative method for solution
        DFDt : DDt object, optional
            Time derivative method for flux
        """

        # Track instances
        SNES_MyEquation.instances += 1

        # Initialize parent class
        super().__init__(
            mesh,
            u_Field,
            degree,
            verbose,
            DuDt=DuDt,
            DFDt=DFDt,
        )

        # Initialize default property values
        self._f = sympy.Matrix.zeros(1, 1)  # Source term
        self._constitutive_model = None

        # Problem-specific parameters
        self._alpha = 1.0  # Example parameter
        self._beta = 0.0  # Example parameter

    @property
    def F0(self):
        """
        Pointwise source/sink term: $f_0(u)$.

        This represents the volumetric source term in the weak form.
        """
        f0_val = expression(
            r"f_0 \left( u \right)",
            -self.f,  # Note: negative sign for weak form
            "MyEquation pointwise source term: f_0(u)",
        )

        return f0_val

    @property
    def F1(self):
        """
        Flux term: $\mathbf{F}_1(u, \nabla u)$.

        This represents the flux that appears in the divergence term.
        """
        if self.constitutive_model is None:
            # Default: simple diffusion
            flux = -sympy.Matrix(
                [sympy.symbols(f"du_dx{i}") for i in range(self.mesh.dim)]
            )
        else:
            flux = self.constitutive_model.flux.T

        F1_val = expression(
            r"\mathbf{F}_1\left( u, \nabla u \right)",
            sympy.simplify(flux),
            "MyEquation pointwise flux term: F_1(u, ∇u)",
        )

        return F1_val

    @timing.routine_timer_decorator
    def my_equation_problem_description(self):
        """
        Set up the problem description for the PETSc solver.

        This method defines the residual terms for the finite element formulation.
        """
        # Source term (f0 - pointwise integration)
        self._f0 = self.F0.sym

        # Flux term (f1 - integration by parts)
        self._f1 = self.F1.sym

        return

    # Property accessors and setters
    @property
    def f(self):
        """Source term function."""
        return self._f

    @f.setter
    def f(self, value):
        """Set the source term."""
        self.is_setup = False
        self._f = sympy.Matrix((value,))

    @property
    def alpha(self):
        """Example parameter alpha."""
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        """Set parameter alpha."""
        self.is_setup = False
        self._alpha = float(value)

    @property
    def beta(self):
        """Example parameter beta."""
        return self._beta

    @beta.setter
    def beta(self, value):
        """Set parameter beta."""
        self.is_setup = False
        self._beta = float(value)

    @property
    def constitutive_model(self):
        """Constitutive model defining material response."""
        return self._constitutive_model

    @constitutive_model.setter
    def constitutive_model(self, model):
        """Set the constitutive model."""
        self.is_setup = False
        self._constitutive_model = model

    @property
    def CM_is_setup(self):
        """Check if constitutive model is set up."""
        if self._constitutive_model is None:
            return True  # No model needed
        return self._constitutive_model._solver_is_setup

    @timing.routine_timer_decorator
    @uw.collective_operation
    def solve(self, verbose=False, debug=False):
        """
        Solve the equation system.

        Parameters
        ----------
        verbose : bool
            Enable verbose solver output
        debug : bool
            Enable debug output

        Returns
        -------
        Convergence information from PETSc solver
        
        Note: This is a COLLECTIVE operation - all MPI ranks must call it.
        """

        # Set up problem if needed
        if not self.is_setup:
            self.my_equation_problem_description()

        # Call parent solve method
        return super().solve(verbose=verbose, debug=debug)

    @timing.routine_timer_decorator
    def estimate_dt(self, dt_min=1.0e-15, dt_max=1.0):
        """
        Estimate appropriate timestep for stability.

        Parameters
        ----------
        dt_min : float
            Minimum allowed timestep
        dt_max : float
            Maximum allowed timestep

        Returns
        -------
        float
            Estimated timestep
        """

        # Get characteristic length scale
        h_min = self.mesh.get_min_radius()

        # Example: CFL-type condition
        # dt < h^2 / (2 * diffusivity) for diffusion
        if self.constitutive_model is not None:
            # Extract diffusivity parameter (problem-specific)
            diff = getattr(self.constitutive_model.Parameters, "diffusivity", 1.0)
            dt_diffusion = h_min**2 / (2.0 * abs(diff))
        else:
            dt_diffusion = h_min**2 / 2.0

        # Additional stability constraints
        dt_stability = dt_diffusion

        # Apply bounds
        dt_estimate = max(dt_min, min(dt_max, dt_stability))

        if self.verbose:
            print(f"Estimated timestep: {dt_estimate}")
            print(f"  Based on diffusion limit: {dt_diffusion}")

        return dt_estimate


# Alternative template for vector equations
class SNES_MyVectorEquation(SNES_Vector):
    r"""
    Template for vector-valued PDE solver.

    Governing equation:
    $$
    \color{Green}{\underbrace{ \Bigl[ \frac{\partial \mathbf{u}}{\partial t} \Bigr]}_{\dot{\mathbf{u}}}} +
    \color{Blue}{\underbrace{\Bigl[ \nabla \cdot \mathbf{F}(\mathbf{u}, \nabla \mathbf{u}) \Bigr]}_{\text{Flux}}} =
    \color{Maroon}{\underbrace{\Bigl[ \mathbf{f}(\mathbf{x}, t, \mathbf{u}) \Bigr] }_{\text{Source}}}
    $$

    Where $\mathbf{u}$ is a vector field.
    """

    instances = 0

    @timing.routine_timer_decorator
    def __init__(
        self,
        mesh: uw.discretisation.Mesh,
        u_Field: Optional[uw.discretisation.MeshVariable] = None,
        degree: int = 2,
        verbose: bool = False,
    ):

        SNES_MyVectorEquation.instances += 1

        super().__init__(mesh, u_Field, degree, verbose)

        # Initialize vector source term
        self._f = sympy.Matrix.zeros(self.mesh.dim, 1)

    @property
    def F0(self):
        """Vector pointwise source term."""
        f0_val = expression(
            r"\mathbf{f}_0 \left( \mathbf{u} \right)",
            -self.f,
            "Vector equation pointwise source term",
        )
        return f0_val

    @property
    def F1(self):
        """Vector flux term - typically a tensor."""
        # Default: identity tensor (for demonstration)
        flux_tensor = sympy.eye(self.mesh.dim)

        F1_val = expression(
            r"\mathbf{F}_1\left( \mathbf{u}, \nabla \mathbf{u} \right)",
            flux_tensor,
            "Vector equation flux tensor",
        )
        return F1_val

    @property
    def f(self):
        """Vector source term."""
        return self._f

    @f.setter
    def f(self, value):
        """Set vector source term."""
        self.is_setup = False
        if hasattr(value, "__len__") and len(value) == self.mesh.dim:
            self._f = sympy.Matrix(value)
        else:
            self._f = sympy.Matrix([value] * self.mesh.dim)


# Usage examples and notes
"""
Usage Notes:
-----------

1. Choose the appropriate base class:
   - SNES_Scalar: For scalar equations (temperature, concentration, etc.)
   - SNES_Vector: For vector equations (velocity, displacement, etc.)
   - SNES_Stokes_SaddlePt: For saddle point problems (Stokes, Navier-Stokes)

2. Key methods to implement:
   - __init__: Initialize solver and default values
   - F0 property: Pointwise source/reaction terms
   - F1 property: Flux terms (appear in divergence)
   - problem_description(): Set up PETSc formulation
   - Property setters: For user-configurable parameters

3. Common patterns:
   - Use @timing.routine_timer_decorator for performance monitoring
   - Include comprehensive docstrings with LaTeX equations
   - Provide sensible defaults and validation
   - Support both automatic and manual field creation
   - Include estimate_dt() for time-dependent problems

4. Integration with constitutive models:
   - Use self.constitutive_model for material properties
   - Check CM_is_setup property for model readiness
   - Allow fallback to simple default behavior

Example instantiation:
>>> mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0,0), maxCoords=(1,1))
>>> solver = SNES_MyEquation(mesh, degree=2, verbose=True)
>>> solver.f = 10.0  # Source term
>>> solver.solve()
"""
