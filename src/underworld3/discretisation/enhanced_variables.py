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

# Direct implementation of EnhancedMeshVariable (moved from persistence.py)
import weakref
from typing import Optional, Union
import numpy as np

from .discretisation_mesh_variables import _BaseMeshVariable
from ..utilities import MathematicalMixin
from ..utilities.dimensionality_mixin import DimensionalityMixin

# === ENHANCED MESH VARIABLE ===

class EnhancedMeshVariable(DimensionalityMixin, MathematicalMixin):
    """
    Enhanced MeshVariable with mathematical operations, units support, and persistence.

    This class enhances the base MeshVariable with:
    - Mathematical operations (via MathematicalMixin) - direct arithmetic, component access
    - Units support (via DimensionalityMixin) - dimensional analysis, unit conversion
    - Optional persistence for adaptive meshing scenarios
    - Collision-safe registration with qualified naming
    - All original MeshVariable functionality (via controlled delegation)

    The wrapper uses controlled delegation to avoid assignment issues with
    operators like += while providing seamless access to the underlying
    MeshVariable functionality.

    Examples:
        # Basic enhanced variable
        velocity = EnhancedMeshVariable("velocity", mesh, 2, units="m/s")

        # Mathematical operations
        momentum = density * velocity
        v_x = velocity[0]
        speed = velocity.norm("L2")

        # Persistence for adaptive meshing
        persistent_var = EnhancedMeshVariable("pressure", mesh, 1, persistent=True)
        success = persistent_var.transfer_data_from(old_pressure)
    """

    def __new__(cls, varname, mesh, *args, **kwargs):
        """Custom __new__ to ensure proper initialization and registration."""
        # Create the instance
        instance = super().__new__(cls)
        instance._enhanced_initialized = False

        # Perform early registration to override any base variable registration
        import underworld3 as uw

        model = uw.get_default_model()

        # Register the wrapper immediately (this will overwrite any base variable registration)
        if isinstance(varname, str):
            name = varname
        elif isinstance(varname, list):
            name = varname[0]
        else:
            name = str(varname)

        # Store reference for later use in __init__
        instance._early_name = name

        return instance

    def __init__(
        self,
        varname: Union[str, list],
        mesh: "Mesh",
        num_components: Union[int, tuple] = None,
        vtype: Optional["uw.VarType"] = None,
        degree: int = 1,
        continuous: bool = True,
        varsymbol: Union[str, list] = None,
        persistent: bool = False,
        units: Optional[str] = None,
        units_backend: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize enhanced mesh variable.

        Args:
            varname: Variable name
            mesh: Mesh object
            num_components: Number of components (1=scalar, 2/3=vector, etc.)
            vtype: Variable type (optional)
            degree: Polynomial degree
            continuous: Whether variable is continuous
            varsymbol: Symbol for LaTeX representation
            persistent: Whether to enable persistence features
            units: Units for this variable (optional)
            units_backend: Units backend to use ('pint' or 'sympy')
            **kwargs: Additional arguments passed to base MeshVariable
        """

        # Check if already initialized (to prevent duplicate initialization)
        if hasattr(self, "_enhanced_initialized") and self._enhanced_initialized:
            return

        # Store configuration FIRST (before any method calls)
        self._persistent = persistent
        self._mesh_ref = weakref.ref(mesh)  # Weak reference to avoid circular deps

        # Create base variable without registration (we handle registration ourselves)
        self._base_var = _BaseMeshVariable(
            varname=varname,
            mesh=mesh,
            num_components=num_components,
            vtype=vtype,
            degree=degree,
            continuous=continuous,
            varsymbol=varsymbol,
            _register=False,  # Never let base variable register itself
            units=units,  # Pass through units for compatibility
            units_backend=units_backend,
            **kwargs,
        )

        # STRICT UNITS MODE CHECK
        # Enforce units-scales contract: variables with units require reference quantities
        if units is not None:
            import underworld3 as uw
            model = uw.get_default_model()

            # Check if strict mode is enabled
            if uw.is_strict_units_active() and not model.has_units():
                # Format variable name for error message
                if isinstance(varname, str):
                    name_str = f"'{varname}'"
                elif isinstance(varname, list):
                    name_str = f"'{varname[0]}'"
                else:
                    name_str = "variable"

                raise ValueError(
                    f"Strict units mode: Cannot create variable {name_str} with units='{units}' "
                    f"when model has no reference quantities.\n\n"
                    f"Options:\n"
                    f"  1. Set reference quantities FIRST:\n"
                    f"     model = uw.get_default_model()\n"
                    f"     model.set_reference_quantities(\n"
                    f"         domain_depth=uw.quantity(1000, 'km'),\n"
                    f"         plate_velocity=uw.quantity(5, 'cm/year')\n"
                    f"     )\n\n"
                    f"  2. Remove units parameter (use plain numbers):\n"
                    f"     uw.discretisation.MeshVariable({name_str}, mesh, ...)\n\n"
                    f"  3. Disable strict mode (not recommended):\n"
                    f"     uw.use_strict_units(False)\n"
                )

            # If not strict mode and no reference quantities, warn as before
            # (Warning is already in _BaseMeshVariable.__init__)

        # Cache frequently accessed properties IMMEDIATELY
        self._name = self._base_var.name
        self._clean_name = self._base_var.clean_name

        # Initialize mixins
        DimensionalityMixin.__init__(self)  # Initialize dimensionality tracking
        # MathematicalMixin is initialized automatically via inheritance

        # Always register with model (AFTER all attributes are set)
        self._setup_registration()

        # Setup additional persistence features if requested
        if persistent:
            self._setup_persistence_features()

        # Mark as initialized
        self._enhanced_initialized = True

    def _setup_registration(self):
        """Register with default model, replacing any existing registration from base variable."""
        import underworld3 as uw

        # All variables register with default model
        # Force registration of wrapper, even if base variable registered itself
        model = uw.get_default_model()

        # Use either the early name (from __new__) or the processed name (from base variable)
        name = getattr(self, "_early_name", self._name)
        model._register_variable(name, self)  # This will overwrite any existing registration

        # Auto-derive scaling coefficient if model has reference quantities
        if hasattr(model, "_reference_quantities") and model._reference_quantities:
            self._auto_derive_scaling_coefficient(model)

    def _auto_derive_scaling_coefficient(self, model):
        """Automatically derive scaling coefficient from model's reference quantities."""
        try:
            from ..utilities.nondimensional import _derive_scale_for_variable

            if hasattr(self, "dimensionality") and self.dimensionality:
                scale = _derive_scale_for_variable(
                    self,
                    model._fundamental_scales if hasattr(model, "_fundamental_scales") else {},
                    model._reference_quantities,
                )
                if scale is not None and scale != 1.0:
                    self.set_reference_scale(scale)
        except (ImportError, AttributeError):
            pass  # Silently skip if module not available or model incomplete

    def _setup_persistence_features(self):
        """Setup additional persistence capabilities for persistent variables."""
        import underworld3 as uw

        # Store qualified name for later reference (for persistent variables)
        model = uw.get_default_model()
        self._qualified_name = None
        for name, var in model._variables.items():
            if var is self:
                self._qualified_name = name
                break

    # === CRITICAL: Direct property access for assignment operations ===

    @property
    def data(self):
        """Direct access to data array (supports += and other assignment ops)."""
        return self._base_var.data

    @property
    def array(self):
        """
        Direct access to array (delegates to base variable).

        The base variable's array implementation now handles units and
        non-dimensional scaling correctly, so we just delegate to it.
        This eliminates code duplication and ensures consistency.
        """
        # Simply delegate to the base variable's array property
        # The base variable now has the complete, working implementation
        return self._base_var.array

    @array.setter
    def array(self, value):
        """Set array values."""
        self._base_var.array = value

    # === READ-ONLY DELEGATED PROPERTIES ===

    @property
    def name(self) -> str:
        """Variable name."""
        return self._name

    @property
    def clean_name(self) -> str:
        """Cleaned variable name for PETSc."""
        return self._clean_name

    @property
    def mesh(self):
        """The mesh this variable is defined on."""
        mesh = self._mesh_ref()
        if mesh is None:
            raise RuntimeError("Mesh has been garbage collected")
        return mesh

    @property
    def coords(self):
        """Coordinate array from base variable."""
        return self._base_var.coords

    @property
    def num_components(self) -> int:
        """Number of components."""
        return self._base_var.num_components

    @property
    def shape(self):
        """Variable shape."""
        return self._base_var.shape

    @property
    def vtype(self):
        """Variable type."""
        return self._base_var.vtype

    @property
    def degree(self) -> int:
        """Polynomial degree."""
        return self._base_var.degree

    # === UNITS PROTOCOL DELEGATION ===

    @property
    def units(self):
        """Units for this variable."""
        return self._base_var.units

    @property
    def has_units(self) -> bool:
        """Check if this variable has units."""
        return self._base_var.has_units

    @property
    def _units_backend(self):
        """Units backend (for protocol compatibility)."""
        return self._base_var._units_backend

    @property
    def dimensionality(self):
        """Get dimensionality (delegates to DimensionalityMixin or base variable)."""
        # DimensionalityMixin.dimensionality uses self.units, which now delegates to _base_var
        # This ensures consistency
        return DimensionalityMixin.dimensionality.fget(self)

    def units_repr(self):
        """Return string representation with units information."""
        if self.has_units:
            return f"Variable '{self.name}' (units: {self.units})"
        else:
            return f"Variable '{self.name}' (no units)"

    def non_dimensional_value(self, model=None):
        """
        Get non-dimensionalized values of the variable.

        Returns the variable's data array in non-dimensional form based on
        the model's reference quantities.
        """
        import underworld3 as uw

        # Use provided model or get default
        if model is None:
            model = uw.get_default_model()

        # Just return the data divided by scaling coefficient
        if hasattr(self._base_var, 'scaling_coefficient'):
            return self.data / float(self._base_var.scaling_coefficient)
        else:
            # No scaling, return data as-is
            return self.data

    @property
    def continuous(self) -> bool:
        """Whether variable is continuous."""
        return self._base_var.continuous

    @property
    def symbol(self):
        """Variable symbol for LaTeX representation."""
        return self._base_var.symbol

    # === PETSC INTEGRATION ===

    @property
    def vec(self):
        """PETSc vector (for solver integration)."""
        return self._base_var.vec

    @property
    def _lvec(self):
        """Local PETSc vector."""
        return self._base_var._lvec

    @property
    def _gvec(self):
        """Global PETSc vector."""
        return self._base_var._gvec

    # === MATHEMATICAL SYMBOL ===

    @property
    def sym(self):
        """Enhanced mathematical symbol."""
        # Get base symbol from underlying variable
        base_sym = self._base_var.sym
        return base_sym

    # === ESSENTIAL METHODS (delegation) ===

    def _set_vec(self, available=True):
        """Initialize PETSc vector."""
        return self._base_var._set_vec(available=available)

    def pack_uw_data_to_petsc(self, *args, **kwargs):
        """Pack data to PETSc format."""
        return self._base_var.pack_uw_data_to_petsc(*args, **kwargs)

    def unpack_uw_data_from_petsc(self, *args, **kwargs):
        """Unpack data from PETSc format."""
        return self._base_var.unpack_uw_data_from_petsc(*args, **kwargs)

    # === VECTOR CALCULUS METHODS ===

    def divergence(self):
        """Vector divergence calculation."""
        return self._base_var.divergence()

    def gradient(self):
        """Vector gradient calculation."""
        return self._base_var.gradient()

    def curl(self):
        """Vector curl calculation."""
        return self._base_var.curl()

    def jacobian(self):
        """Vector jacobian calculation."""
        return self._base_var.jacobian()

    # === ADDITIONAL DELEGATED METHODS ===

    def clone(self):
        """Clone the variable."""
        return self._base_var.clone()

    def max(self):
        """Maximum value of the variable."""
        return self._base_var.max()

    def mean(self):
        """Mean value of the variable."""
        return self._base_var.mean()

    def min(self):
        """Minimum value of the variable."""
        return self._base_var.min()

    def norm(self, norm_type=None):
        """Compute the norm of the variable."""
        if norm_type is None:
            # Mathematical usage: use SymPy Matrix norm from MathematicalMixin
            return MathematicalMixin.norm(self, norm_type=None)
        else:
            # Computational usage: delegate to PETSc norm
            return self._base_var.norm(norm_type)

    def load_from_h5_plex_vector(self, *args, **kwargs):
        """Load from HDF5 plex vector."""
        return self._base_var.load_from_h5_plex_vector(*args, **kwargs)

    def write(self, *args, **kwargs):
        """Write variable data to HDF5 file."""
        return self._base_var.write(*args, **kwargs)

    @property
    def sym_1d(self):
        """Flattened sympy view of the variable."""
        return self._base_var.sym_1d

    def save(self, *args, **kwargs):
        """Save variable data."""
        return self._base_var.save(*args, **kwargs)

    def read_timestep(self, *args, **kwargs):
        """Read timestep data."""
        return self._base_var.read_timestep(*args, **kwargs)

    def stats(self, *args, **kwargs):
        """Get statistics for the variable."""
        return self._base_var.stats(*args, **kwargs)

    def sum(self, *args, **kwargs):
        """Sum of variable values."""
        return self._base_var.sum(*args, **kwargs)

    def rbf_interpolate(self, *args, **kwargs):
        """RBF interpolation."""
        return self._base_var.rbf_interpolate(*args, **kwargs)

    def pack_raw_data_to_petsc(self, *args, **kwargs):
        """Pack raw data to PETSc format."""
        return self._base_var.pack_raw_data_to_petsc(*args, **kwargs)

    def unpack_raw_data_from_petsc(self, *args, **kwargs):
        """Unpack raw data from PETSc format."""
        return self._base_var.unpack_raw_data_from_petsc(*args, **kwargs)

    # === ADDITIONAL DELEGATED PROPERTIES ===

    @property
    def field_id(self):
        """Field ID in the mesh."""
        return self._base_var.field_id

    @property
    def fn(self):
        """Function representation."""
        return self._base_var.fn

    @property
    def ijk(self):
        """IJK coordinates."""
        return self._base_var.ijk

    @property
    def instance_number(self):
        """Instance number."""
        return self._base_var.instance_number

    @property
    def old_data(self):
        """Legacy data property (for testing)."""
        return self._base_var.old_data

    @property
    def uw_object_counter(self):
        """Object counter from API tools."""
        return self._base_var.uw_object_counter

    # === PERSISTENCE CAPABILITIES ===

    def transfer_data_from(self, source_var):
        """
        Transfer data from another variable using global_evaluate.

        This enables persistent variable identity across mesh changes.

        Args:
            source_var: Source variable to transfer data from

        Returns:
            bool: True if transfer successful
        """
        if not self._persistent:
            raise RuntimeError("Data transfer only available for persistent variables")

        import underworld3 as uw

        model = uw.get_default_model()
        return model.transfer_variable_data(source_var, self)

    @property
    def qualified_name(self) -> Optional[str]:
        """
        Qualified name for collision resolution.

        Returns the fully qualified name used in the model registry
        to resolve namespace collisions (e.g., 'velocity_mesh123456').
        """
        return self._qualified_name if self._persistent else None

    @property
    def is_persistent(self) -> bool:
        """Whether this variable has persistence capabilities."""
        return self._persistent

    # === REPRESENTATION ===

    def __repr__(self):
        """Enhanced representation showing persistence and units info."""
        base_repr = self._base_var.__repr__()

        # Add persistence info
        if self._persistent:
            base_repr = base_repr.replace(">", f", persistent=True>")

        # Add units info
        if self.has_units:
            base_repr = base_repr.replace(">", f", units={self.units}>")

        return base_repr

    def __str__(self):
        """String representation."""
        return self.__repr__()

    def view(self):
        """
        Display detailed information about the enhanced variable including units.

        Shows variable name, dimensions, shape, units information, and sample data.
        """
        print(f"Enhanced MeshVariable: {self.name}")
        print(f"  Components: {self.num_components}")
        print(f"  Degree: {self.degree}")
        print(f"  Array shape: {self.array.shape}")

        # Units information
        if self.has_units:
            print(f"  Units: {self.units}")
            print(f"  Dimensionality: {self.dimensionality}")
            print(f"  Units backend: {type(self._units_backend).__name__}")
        else:
            print(f"  Units: None (dimensionless)")

        # Persistence information
        if self._persistent:
            print(f"  Persistence: Enabled")
        else:
            print(f"  Persistence: Disabled")

        # Sample data (first few elements)
        try:
            if len(self.array.shape) > 0 and self.array.shape[0] > 0:
                print(f"  Data sample: {self.array[:3]}")
            else:
                print(f"  Data sample: No data")
        except:
            print(f"  Data sample: Unable to display")

        # Mathematical capabilities
        if hasattr(self, "sym"):
            print(f"  Symbolic form: {self.sym}")

        return self  # Allow chaining


def create_enhanced_mesh_variable(
    varname: Union[str, list],
    mesh: "Mesh",
    num_components: Union[int, tuple] = None,
    persistent: bool = True,
    units: Optional[str] = None,
    **kwargs,
) -> EnhancedMeshVariable:
    """
    Factory function to create an enhanced mesh variable.

    This is a convenience function that creates an EnhancedMeshVariable
    with persistence enabled by default.

    Args:
        varname: Variable name
        mesh: Mesh object
        num_components: Number of components
        persistent: Enable persistence (default True)
        units: Units specification
        **kwargs: Additional arguments

    Returns:
        EnhancedMeshVariable instance
    """
    return EnhancedMeshVariable(
        varname=varname,
        mesh=mesh,
        num_components=num_components,
        persistent=persistent,
        units=units,
        **kwargs,
    )


# === SWARM VARIABLES ===
# NOTE: SwarmVariable already has all enhanced functionality built-in
#
# Unlike MeshVariable (which uses a wrapper pattern), SwarmVariable directly
# inherits from both DimensionalityMixin and MathematicalMixin. Therefore,
# there is NO NEED for a separate EnhancedSwarmVariable class.
#
# Users should directly use:
#   from underworld3.swarm import SwarmVariable
#
# SwarmVariable already provides:
# - Mathematical operations (via MathematicalMixin)
# - Units and dimensional analysis (via DimensionalityMixin)
# - All standard swarm functionality
#
# See: src/underworld3/swarm.py line 40:
#   class SwarmVariable(DimensionalityMixin, MathematicalMixin, Stateful, uw_object)


# Make all enhanced classes and functions available at module level
__all__ = [
    # Enhanced classes
    "EnhancedMeshVariable",
    # Factory functions
    "create_enhanced_mesh_variable",
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

        # Test swarm variables (SwarmVariable is already enhanced)
        print("\\nTesting Swarm Variables (already enhanced):")
        print("--------------------------------------------")

        swarm = uw.swarm.Swarm(mesh)
        swarm.populate(fill_param=2)

        density = uw.swarm.SwarmVariable("density", swarm, 1, units="kg/m^3")
        print(f"✓ Created density: {density.name}")
        print(f"  - Units: {density.units}")

        print("\\n✅ All enhanced variable functionality working!")

    except Exception as e:
        print(f"✗ Error in demonstration: {e}")
        import traceback

        traceback.print_exc()


# Note: The demonstration function above references EnhancedSwarmVariable
# which doesn't exist - SwarmVariable is already enhanced (see swarm.py).
# Update this demo to use uw.swarm.SwarmVariable directly if needed.
