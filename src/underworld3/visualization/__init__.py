"""
Underworld3 visualization utilities.

This module provides visualization tools for Underworld3 including:
- PyVista-based 3D visualization (visualisation.py)
- Parallel-safe matplotlib plotting (parallel.py)
"""

# Import main visualization functions from visualisation.py
from ..visualisation import (
    mesh_to_pv_mesh,
    scalar_fn_to_pv_points,
    vector_fn_to_pv_points,
    tensor_fn_to_pv_points,
    pv_save_to_disk,
    scalar_colour_points,
    scatter_plot,
    quiver_plot,
)

# Import parallel visualization utilities
from . import parallel
