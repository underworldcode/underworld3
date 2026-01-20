# underworld3/persistence.py
"""
Persistence features for adaptive meshing and data transfer.

NOTE: EnhancedMeshVariable has been moved to enhanced_variables.py (2025-01-13).

This module is reserved for future implementation of actual persistence features:
- Data transfer between meshes during adaptive refinement
- Variable state preservation across remeshing operations
- Checkpoint/restart capabilities for long-running simulations
- Mesh-to-mesh interpolation utilities

For enhanced variables with mathematical operations and units support,
see discretisation/enhanced_variables.py

Symbol Disambiguation Note (2025-12-15):
----------------------------------------
When transferring data or expressions between meshes, be aware that:
- MeshVariable symbols are mesh-specific (v1.sym != v2.sym even with same name)
- Coordinate symbols (mesh.N.x, etc.) are also mesh-specific to prevent cache bugs
- For expression portability, use explicit coordinate substitution:
    expr_for_mesh2 = expr.subs({mesh1.N.x: mesh2.N.x, mesh1.N.y: mesh2.N.y})

See: docs/developer/design/SYMBOL_DISAMBIGUATION_2025-12.md

Historical Note:
-----------------
This module previously contained EnhancedMeshVariable, but that class
was moved to enhanced_variables.py where it logically belongs. The name
"persistence" was misleading since the main functionality was mathematical
operations and units support, not data persistence.

The actual persistence features (like transfer_data_from) are integrated
into EnhancedMeshVariable and will be augmented here in the future with
standalone utilities for mesh-to-mesh data transfer.
"""

# For backward compatibility, re-export from enhanced_variables
# This ensures existing code that imports from persistence still works
from .enhanced_variables import (
    EnhancedMeshVariable,
    create_enhanced_mesh_variable,
)

__all__ = [
    "EnhancedMeshVariable",  # Re-exported for backward compatibility
    "create_enhanced_mesh_variable",  # Re-exported for backward compatibility
]

# Future persistence utilities will be added here
# TODO: Implement standalone mesh-to-mesh transfer functions
# TODO: Implement checkpoint/restart utilities
# TODO: Implement adaptive meshing data preservation
