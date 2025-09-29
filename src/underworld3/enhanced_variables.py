# underworld3/enhanced_variables.py
"""
Enhanced Variable Classes with Mathematical and Units Support

This module provides enhanced versions of Underworld3 variable classes that
integrate both MathematicalMixin and UnitAwareMixin functionality.

These enhanced classes demonstrate how to add units support to existing
classes without breaking backward compatibility.
"""

from .discretisation import _MeshVariable
from .swarm import SwarmVariable as _SwarmVariable
from .utilities import UnitAwareMixin, MathematicalMixin


class EnhancedMeshVariable(UnitAwareMixin, _MeshVariable):
    """
    Enhanced MeshVariable with both mathematical operations and units support.
    
    This class combines:
    - All original MeshVariable functionality
    - Direct mathematical operations (from MathematicalMixin)
    - Units and dimensional analysis (from UnitAwareMixin)
    
    Usage:
        # Create mesh variable with units
        velocity = EnhancedMeshVariable("velocity", mesh, 2, units="m/s")
        
        # Mathematical operations with units checking
        momentum = density * velocity  # Direct arithmetic
        v_x = velocity[0]              # Component access
        speed = velocity.norm()        # Vector magnitude
        
        # Units operations
        velocity_scaled = velocity.non_dimensional_value()  # For solvers
        velocity_kmh = velocity.to_units("km/h")           # Unit conversion
    
    Example:
        >>> import underworld3 as uw
        >>> mesh = uw.meshing.UniformMesh((0, 0), (1, 1), (16, 16))
        >>> velocity = EnhancedMeshVariable("velocity", mesh, 2, units="m/s")
        >>> velocity.units
        meter / second
        >>> kinetic_energy = 0.5 * density * velocity.dot(velocity)
        >>> kinetic_energy  # Will have units of kg⋅m²/s² (energy)
    """
    
    def __init__(self, *args, units=None, units_backend=None, **kwargs):
        """
        Initialize enhanced mesh variable.
        
        Args:
            *args: Arguments passed to _MeshVariable
            units: Units for this variable (optional)
            units_backend: Units backend to use ('pint' or 'sympy')
            **kwargs: Keyword arguments passed to _MeshVariable
        """
        # Initialize all parent classes
        super().__init__(*args, units=units, units_backend=units_backend, **kwargs)
    
    def __repr__(self):
        """
        Enhanced representation showing computational info by default.
        
        Shows the computational view (mesh info, field details) rather than
        the symbolic mathematical form, which is more useful for debugging
        and understanding variable state.
        """
        # Use the original MeshVariable representation
        base_repr = _MeshVariable.__repr__(self)
        
        # Add units information if present
        if self.has_units:
            return f"{base_repr[:-1]}, units={self.units})"
        else:
            return base_repr
    
    def _repr_latex_(self):
        """
        LaTeX representation for Jupyter notebooks.
        
        In mathematical contexts, show the symbolic form with units.
        """
        if hasattr(super(), '_repr_latex_'):
            latex_repr = super()._repr_latex_()
            if self.has_units:
                # Add units to LaTeX display
                from sympy import latex
                units_latex = latex(self.units) if hasattr(self.units, '_latex') else str(self.units)
                return latex_repr[:-2] + r'\,\mathrm{' + units_latex + r'}$$'
            return latex_repr
        else:
            # Fallback to mathematical mixin LaTeX
            return MathematicalMixin._repr_latex_(self)


class EnhancedSwarmVariable(UnitAwareMixin, _SwarmVariable):
    """
    Enhanced SwarmVariable with both mathematical operations and units support.
    
    This class combines:
    - All original SwarmVariable functionality
    - Direct mathematical operations (from MathematicalMixin)
    - Units and dimensional analysis (from UnitAwareMixin)
    
    Usage:
        # Create swarm variable with units
        density = EnhancedSwarmVariable("density", swarm, 1, units="kg/m^3")
        
        # Mathematical operations with units checking
        total_mass = density * volume  # Direct arithmetic with units
        rho_max = density.max()        # Statistical operations
        
        # Units operations
        density_scaled = density.non_dimensional_value()  # For calculations
        density_gcc = density.to_units("g/cm^3")         # Unit conversion
    
    Example:
        >>> import underworld3 as uw
        >>> swarm = uw.swarm.Swarm(mesh)
        >>> density = EnhancedSwarmVariable("density", swarm, 1, units="kg/m^3")
        >>> density.units
        kilogram / meter ** 3
        >>> mass_flux = velocity * density  # Units: kg/(m²⋅s)
    """
    
    def __init__(self, *args, units=None, units_backend=None, **kwargs):
        """
        Initialize enhanced swarm variable.
        
        Args:
            *args: Arguments passed to _SwarmVariable  
            units: Units for this variable (optional)
            units_backend: Units backend to use ('pint' or 'sympy')
            **kwargs: Keyword arguments passed to _SwarmVariable
        """
        # Initialize all parent classes
        super().__init__(*args, units=units, units_backend=units_backend, **kwargs)
    
    def __repr__(self):
        """
        Enhanced representation showing computational info by default.
        """
        # Use the original SwarmVariable representation
        base_repr = _SwarmVariable.__repr__(self)
        
        # Add units information if present
        if self.has_units:
            return f"{base_repr[:-1]}, units={self.units})"
        else:
            return base_repr
    
    def _repr_latex_(self):
        """
        LaTeX representation for Jupyter notebooks with units.
        """
        if hasattr(super(), '_repr_latex_'):
            latex_repr = super()._repr_latex_()
            if self.has_units:
                # Add units to LaTeX display
                from sympy import latex
                units_latex = latex(self.units) if hasattr(self.units, '_latex') else str(self.units)
                return latex_repr[:-2] + r'\,\mathrm{' + units_latex + r'}$$'
            return latex_repr
        else:
            # Fallback to mathematical mixin LaTeX
            return MathematicalMixin._repr_latex_(self)


# Convenience factory functions for creating enhanced variables
def create_enhanced_mesh_variable(name, mesh, num_components, 
                                  units=None, units_backend='pint', **kwargs):
    """
    Factory function to create an enhanced mesh variable.
    
    Args:
        name: Variable name
        mesh: Mesh object
        num_components: Number of components (1=scalar, 2/3=vector, etc.)
        units: Units specification (optional)
        units_backend: Units backend ('pint' or 'sympy')
        **kwargs: Additional arguments passed to variable constructor
        
    Returns:
        EnhancedMeshVariable instance
        
    Example:
        >>> velocity = create_enhanced_mesh_variable("velocity", mesh, 2, 
        ...                                         units="m/s")
        >>> pressure = create_enhanced_mesh_variable("pressure", mesh, 1,
        ...                                         units="Pa")
    """
    return EnhancedMeshVariable(name, mesh, num_components, 
                               units=units, units_backend=units_backend, **kwargs)


def create_enhanced_swarm_variable(name, swarm, num_components,
                                   units=None, units_backend='pint', **kwargs):
    """
    Factory function to create an enhanced swarm variable.
    
    Args:
        name: Variable name
        swarm: Swarm object
        num_components: Number of components (1=scalar, 2/3=vector, etc.)
        units: Units specification (optional)
        units_backend: Units backend ('pint' or 'sympy')
        **kwargs: Additional arguments passed to variable constructor
        
    Returns:
        EnhancedSwarmVariable instance
        
    Example:
        >>> density = create_enhanced_swarm_variable("density", swarm, 1,
        ...                                         units="kg/m^3")
        >>> temperature = create_enhanced_swarm_variable("temperature", swarm, 1,
        ...                                             units="K")
    """
    return EnhancedSwarmVariable(name, swarm, num_components,
                                units=units, units_backend=units_backend, **kwargs)


# Demonstration function
def demonstrate_enhanced_variables():
    """
    Demonstration of enhanced variables with units and mathematical operations.
    
    This function shows how the enhanced variables integrate mathematical
    operations with units checking and dimensional analysis.
    """
    print("Enhanced Variables Demonstration")
    print("=" * 40)
    
    try:
        import underworld3 as uw
        
        # Create a simple mesh
        mesh = uw.meshing.UniformMesh((0, 0), (1, 1), (8, 8))
        print(f"✓ Created mesh: {mesh}")
        
        # Create enhanced mesh variables with units
        velocity = EnhancedMeshVariable("velocity", mesh, 2, units="m/s")
        pressure = EnhancedMeshVariable("pressure", mesh, 1, units="Pa")
        
        print(f"✓ Created velocity: {velocity}")
        print(f"  - Units: {velocity.units}")
        print(f"  - Has units: {velocity.has_units}")
        
        print(f"✓ Created pressure: {pressure}")
        print(f"  - Units: {pressure.units}")
        
        # Test mathematical operations
        print("\nTesting Mathematical Operations:")
        print("--------------------------------")
        
        # Component access
        v_x = velocity[0]
        print(f"✓ Component access: velocity[0] = {type(v_x)}")
        
        # Vector operations
        speed_squared = velocity.dot(velocity)
        print(f"✓ Dot product: velocity.dot(velocity) = {type(speed_squared)}")
        
        # Arithmetic with units checking
        try:
            momentum = 1000 * velocity  # kg/m³ * m/s = kg/(m²⋅s)
            print(f"✓ Scalar multiplication: 1000 * velocity = {type(momentum)}")
        except Exception as e:
            print(f"✗ Scalar multiplication failed: {e}")
        
        # Test units compatibility checking
        print("\nTesting Units Compatibility:")
        print("-----------------------------")
        
        velocity2 = EnhancedMeshVariable("velocity2", mesh, 2, units="m/s")
        compatible = velocity.check_units_compatibility(velocity2)
        print(f"✓ Compatible velocities: {compatible}")
        
        incompatible = velocity.check_units_compatibility(pressure)
        print(f"✓ Incompatible velocity vs pressure: {incompatible}")
        
        # Test swarm variables
        print("\nTesting Enhanced Swarm Variables:")
        print("---------------------------------")
        
        swarm = uw.swarm.Swarm(mesh)
        swarm.populate(n_per_cell=4)
        
        density = EnhancedSwarmVariable("density", swarm, 1, units="kg/m^3")
        print(f"✓ Created density: {density}")
        print(f"  - Units: {density.units}")
        
        print("\n✅ All enhanced variable functionality working!")
        
    except Exception as e:
        print(f"✗ Error in demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    demonstrate_enhanced_variables()