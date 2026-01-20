"""
Underworld3 visualization utilities.

This module provides visualization tools for Underworld3 including:
- PyVista-based 3D visualization (visualisation.py)
- Parallel-safe matplotlib plotting (parallel.py)
"""

# Import main visualization functions from visualisation.py
from .visualisation import (
    mesh_to_pv_mesh,
    scalar_fn_to_pv_points,
    vector_fn_to_pv_points,
    plot_mesh,
    plot_scalar,
    plot_vector,
    meshVariable_to_pv_cloud,
    meshVariable_to_pv_mesh_object,
    swarm_to_pv_cloud,
)

# Import parallel visualization utilities
from . import parallel
