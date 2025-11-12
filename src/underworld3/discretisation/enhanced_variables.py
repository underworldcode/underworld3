# underworld3/enhanced_variables.py
"""
# Enhanced Variable Classes with Mathematical and Units Support

This module provides enhanced versions of Underworld3 variable classes that
integrate mathematical operations, units support, and persistence capabilities.

The enhanced classes provide a clean separation between pure data (base classes)
and enhanced functionality (wrapper classes) while maintaining backward compatibility.

Key Features:
- Mathematical operations (direct arithmetic, component access, vector operations)
- Units support (dimensional analysis, unit conversion, consistency checking)
- Persistence capabilities (collision detection, data transfer for adaptive meshing)
- Controlled delegation (avoids assignment operator issues like +=)

Classes:
- EnhancedMeshVariable: Enhanced mesh variable with all capabilities
- EnhancedSwarmVariable: Enhanced swarm variable with mathematical and units support

Factory Functions:
- create_enhanced_mesh_variable: Convenience function for enhanced mesh variables
- create_enhanced_swarm_variable: Convenience function for enhanced swarm variables
"""

# Import the new enhanced implementations from persistence module
from .persistence import (
    EnhancedMeshVariable,
    create_enhanced_mesh_variable,
)

# For backward compatibility, also provide the old swarm implementation
# TODO: Implement EnhancedSwarmVariable wrapper following same pattern
from ..swarm import SwarmVariable as _SwarmVariable
from ..utilities import UnitAwareMixin, MathematicalMixin


class EnhancedSwarmVariable(UnitAwareMixin, _SwarmVariable):
    """
    Enhanced SwarmVariable with both mathematical operations and units support.

    This class combines:
    - All original SwarmVariable functionality
    - Direct mathematical operations (from MathematicalMixin)
    - Units and dimensional analysis (from UnitAwareMixin)

    Note: This is the legacy implementation. A wrapper-based implementation
    following the same pattern as EnhancedMeshVariable should be created
    for consistency.

    Usage:
        # Create swarm variable with units
        density = EnhancedSwarmVariable("density", swarm, 1, units="kg/m^3")

        # Mathematical operations with units checking
        total_mass = density * volume  # Direct arithmetic with units
        rho_max = density.max()        # Statistical operations

        # Units operations
        density_scaled = density.non_dimensional_value()  # For calculations
        density_gcc = density.to("g/cm^3")               # Unit conversion
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
        if hasattr(super(), "_repr_latex_"):
            latex_repr = super()._repr_latex_()
            if self.has_units:
                # Add units to LaTeX display
                from sympy import latex

                units_latex = (
                    latex(self.units) if hasattr(self.units, "_latex") else str(self.units)
                )
                return latex_repr[:-2] + r"\\,\\mathrm{" + units_latex + r"}$$"
            return latex_repr
        else:
            # Fallback to mathematical mixin LaTeX
            return MathematicalMixin._repr_latex_(self)


def create_enhanced_swarm_variable(
    name: str,
    swarm: "Swarm",
    num_components: int,
    units: str = None,
    units_backend: str = "pint",
    **kwargs,
) -> EnhancedSwarmVariable:
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
    return EnhancedSwarmVariable(
        name, swarm, num_components, units=units, units_backend=units_backend, **kwargs
    )


# Make all enhanced classes and functions available at module level
__all__ = [
    # Enhanced classes
    "EnhancedMeshVariable",
    "EnhancedSwarmVariable",
    # Factory functions
    "create_enhanced_mesh_variable",
    "create_enhanced_swarm_variable",
]


# Demonstration function (for testing and documentation)
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
        mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4), minCoords=(0, 0), maxCoords=(1, 1))
        print(f"✓ Created mesh")

        # Create enhanced mesh variables with units
        velocity = EnhancedMeshVariable("velocity", mesh, 2, units="m/s")
        pressure = EnhancedMeshVariable("pressure", mesh, 1, units="Pa")

        print(f"✓ Created velocity: {velocity.name}")
        print(f"  - Units: {velocity.units}")
        print(f"  - Has units: {velocity.has_units}")
        print(f"  - Is persistent: {velocity.is_persistent}")

        print(f"✓ Created pressure: {pressure.name}")
        print(f"  - Units: {pressure.units}")

        # Test mathematical operations
        print("\\nTesting Mathematical Operations:")
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
        print("\\nTesting Units Compatibility:")
        print("-----------------------------")

        velocity2 = EnhancedMeshVariable("velocity2", mesh, 2, units="m/s")
        compatible = velocity.check_units_compatibility(velocity2)
        print(f"✓ Compatible velocities: {compatible}")

        incompatible = velocity.check_units_compatibility(pressure)
        print(f"✓ Incompatible velocity vs pressure: {incompatible}")

        # Test swarm variables
        print("\\nTesting Enhanced Swarm Variables:")
        print("---------------------------------")

        swarm = uw.swarm.Swarm(mesh)
        swarm.populate(fill_param=2)

        density = EnhancedSwarmVariable("density", swarm, 1, units="kg/m^3")
        print(f"✓ Created density: {density.name}")
        print(f"  - Units: {density.units}")

        print("\\n✅ All enhanced variable functionality working!")

    except Exception as e:
        print(f"✗ Error in demonstration: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    demonstrate_enhanced_variables()
