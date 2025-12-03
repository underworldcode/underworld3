"""
Unit-aware coordinate symbols for Underworld3.

This module provides UnitAwareBaseScalar, a subclass of SymPy's BaseScalar
that adds unit awareness to coordinate symbols while maintaining full compatibility
with the JIT compilation system.
"""

import sympy
from sympy.vector.scalar import BaseScalar


class UnitAwareBaseScalar(BaseScalar):
    """
    A BaseScalar subclass that carries units information.

    This class maintains full compatibility with SymPy's vector system and
    Underworld's JIT compilation while adding unit awareness. The JIT system
    detects coordinates by looking for BaseScalar atoms, so by inheriting from
    BaseScalar, these unit-aware coordinates will be properly recognized and
    compiled.

    Attributes:
        _units: The units of this coordinate (e.g., "meter", "kilometer")
        _ccodestr: The C code string for JIT compilation (set by mesh initialization)
    """

    def __new__(cls, name, index, system, pretty_str=None, latex_str=None, units=None):
        """
        Create a new unit-aware coordinate symbol.

        Parameters:
            name: Name of the coordinate (e.g., "x", "y", "z")
            index: Index of this coordinate (0, 1, or 2)
            system: The parent coordinate system
            pretty_str: Pretty print string (optional)
            latex_str: LaTeX representation (optional)
            units: Units of this coordinate (optional)
        """
        # BaseScalar.__new__ doesn't take 'name' - it infers it from the system
        # Signature: BaseScalar.__new__(cls, index, system, pretty_str, latex_str)
        obj = BaseScalar.__new__(cls, index, system, pretty_str, latex_str)
        # Store units on the new object
        obj._units = units
        return obj

    def __init__(self, name, index, system, pretty_str=None, latex_str=None, units=None):
        """Initialize the unit-aware coordinate with units information."""
        # BaseScalar.__init__ also doesn't take 'name'
        super().__init__(index, system, pretty_str, latex_str)
        # _units already set in __new__

    @property
    def units(self):
        """Get the units of this coordinate."""
        return self._units

    @units.setter
    def units(self, value):
        """Set the units of this coordinate."""
        self._units = value

    # TRANSPARENT CONTAINER PRINCIPLE (2025-11-26):
    # Arithmetic operators return plain SymPy results.
    # Units are derived on demand via uw.get_units() which traverses the
    # expression tree and finds atoms with ._units attributes.
    #
    # This eliminates:
    # - UnitAwareExpression wrapping overhead
    # - Unit sync issues between stored and computed values
    # - Complexity in arithmetic operator implementations
    #
    # The coordinate symbols carry ._units attributes, so get_units() can
    # discover them when needed. No special operator overloading required.

    def get_units(self):
        """
        Get units for compatibility with get_units() function.

        Returns:
            Units of this coordinate or None if dimensionless
        """
        return self._units


def create_unit_aware_coordinate_system(name, units=None):
    """
    Create a coordinate system with unit-aware coordinates.

    This function creates a SymPy CoordSys3D-like object but with
    UnitAwareBaseScalar coordinates that carry units information.

    Parameters:
        name: Name of the coordinate system (e.g., "N")
        units: Units for the coordinates (e.g., "meter", "kilometer")

    Returns:
        A coordinate system object with unit-aware x, y, z coordinates
    """
    from sympy.vector import CoordSys3D

    # Create a standard coordinate system first
    system = CoordSys3D(name)

    # Replace the standard BaseScalar coordinates with unit-aware ones
    # Note: We need to maintain the same structure for JIT compatibility

    # Store original coordinates
    orig_x = system.x
    orig_y = system.y
    orig_z = system.z

    # Create unit-aware replacements
    system.x = UnitAwareBaseScalar(
        "x",
        0,
        system,
        pretty_str=orig_x._pretty_form if hasattr(orig_x, "_pretty_form") else None,
        latex_str=orig_x._latex_form if hasattr(orig_x, "_latex_form") else None,
        units=units,
    )

    system.y = UnitAwareBaseScalar(
        "y",
        1,
        system,
        pretty_str=orig_y._pretty_form if hasattr(orig_y, "_pretty_form") else None,
        latex_str=orig_y._latex_form if hasattr(orig_y, "_latex_form") else None,
        units=units,
    )

    system.z = UnitAwareBaseScalar(
        "z",
        2,
        system,
        pretty_str=orig_z._pretty_form if hasattr(orig_z, "_pretty_form") else None,
        latex_str=orig_z._latex_form if hasattr(orig_z, "_latex_form") else None,
        units=units,
    )

    # Update the base scalars list
    system._base_scalars = (system.x, system.y, system.z)

    return system


def patch_coordinate_units(mesh):
    """
    Patch existing mesh coordinates to be unit-aware.

    This function takes an existing mesh with standard BaseScalar coordinates
    and adds unit awareness to them. This is done by monkey-patching the
    units property onto the existing coordinate objects.

    Parameters:
        mesh: The mesh whose coordinates should be made unit-aware
    """
    # Get mesh units if available
    mesh_units = getattr(mesh, "units", None)

    # If mesh doesn't have explicit units but we have a model with reference scales,
    # use the length units from the model
    if mesh_units is None:
        try:
            import underworld3 as uw

            model = uw.get_default_model()
            if hasattr(model, "_fundamental_scales") and model._fundamental_scales:
                scales = model._fundamental_scales
                if "length" in scales:
                    length_scale = scales["length"]
                    # Store Pint Unit objects, NOT strings
                    if hasattr(length_scale, "units"):
                        mesh_units = length_scale.units  # Already a Pint Unit
                    elif hasattr(length_scale, "_pint_qty"):
                        mesh_units = length_scale._pint_qty.units  # Pint Unit from quantity
        except Exception as e:
            # Silently continue if model not available
            pass

    if mesh_units is not None:
        # Add units attribute to existing coordinates
        # TRANSPARENT CONTAINER PRINCIPLE (2025-11-26):
        # We only set ._units on the coordinate atoms. No operator overloading needed.
        # uw.get_units() traverses expression trees and finds these atoms with ._units.
        # This preserves object identity for JIT compatibility.

        def _patch_coordinate(coord, units):
            """Helper to add units information to a coordinate."""
            # Update units if they've changed or don't exist
            current_units = getattr(coord, "_units", None)
            if current_units != units:
                coord._units = units

            # Add get_units method for compatibility (idempotent)
            if not hasattr(coord, "get_units"):
                coord.get_units = lambda self=coord: self._units

        # Patch the x, y, z coordinates
        for coord in [mesh.N.x, mesh.N.y, mesh.N.z]:
            _patch_coordinate(coord, mesh_units)

        # Also patch the normal vector coordinates if they exist
        if hasattr(mesh, "_Gamma"):
            for coord in [mesh._Gamma.x, mesh._Gamma.y, mesh._Gamma.z]:
                _patch_coordinate(coord, mesh_units)


def get_coordinate_units(coord):
    """
    Get units from a coordinate symbol.

    This function works with both standard BaseScalar (returns None)
    and UnitAwareBaseScalar (returns units).

    Parameters:
        coord: A coordinate symbol (BaseScalar or UnitAwareBaseScalar)

    Returns:
        Units of the coordinate or None if not unit-aware
    """
    if hasattr(coord, "_units"):
        return coord._units
    elif hasattr(coord, "units"):
        return coord.units
    else:
        return None
