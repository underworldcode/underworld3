"""
Hierarchical Unit-Aware Expression Module

DEPRECATED (2025-11-26): UnitAwareExpression has been removed.

Following the Transparent Container Principle, composite expressions now
return raw SymPy objects and units are derived on demand via get_units().

See planning/UNITS_SIMPLIFIED_DESIGN_2025-11.md for architecture details.
"""

# UnitAwareExpression, MathematicalExpression, LazyExpression, create_unit_aware
# have been removed - use UWexpression and get_units() instead

__all__ = []
