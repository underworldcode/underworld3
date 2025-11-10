"""
Hierarchical Unit-Aware Expression Module

This module provides a clean hierarchical architecture for handling
unit-aware symbolic expressions in Underworld3, separating concerns:
- Pure SymPy computation
- Unit metadata tracking
- Mathematical operations
- Lazy evaluation
"""

from .unit_aware_expression import (
    UnitAwareExpression,
    MathematicalExpression,
    LazyExpression,
    create_unit_aware
)

__all__ = [
    'UnitAwareExpression',
    'MathematicalExpression',
    'LazyExpression',
    'create_unit_aware'
]