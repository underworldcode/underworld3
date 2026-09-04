"""
Underworld3 visualization utilities.

This module provides visualization tools for Underworld3 including:
- PyVista-based 3D visualization (visualisation.py)
- Parallel-safe matplotlib plotting (parallel.py)
"""

# Import main visualization functions from visualisation.py
from .visualisation import (
    mesh_to_pv_mesh,
    labelled_facets_to_pv_mesh,
    scalar_fn_to_pv_points,
    vector_fn_to_pv_points,
    plot_mesh,
    plot_mesh_hierarchy,
    MG_LEVEL_COLOURS,
    FAULT_COLOUR,
    plot_scalar,
    plot_vector,
    meshVariable_to_pv_cloud,
    meshVariable_to_pv_mesh_object,
    meshVariable_to_native_pv_mesh,
    swarm_to_pv_cloud,
)

# Principal-stress glyphs and stress trajectories (glyphs.py)
from .glyphs import (
    tensor_fn_to_pv_points,
    principal_stress_glyphs,
    direction_trajectories,
    trajectories_to_pv_lines,
    plot_stress_glyphs,
)

# Import parallel visualization utilities
from . import parallel
