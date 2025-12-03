# Meshing Module Breakup Plan

## Current State
- **File**: `src/underworld3/meshing.py`
- **Size**: 4,443 lines
- **Functions**: 17 mesh factory functions
- **Problem**: Monolithic file that's hard to maintain and navigate

## Proposed Module Structure

### 1. Base Module: `meshing/__init__.py`
**Purpose**: Common imports, utilities, and backward compatibility
**Contents**:
- Import all mesh functions for backward compatibility
- Common imports (gmsh, numpy, timing, etc.)
- Shared utility functions
- Main mesh class imports

```python
# Backward compatibility - import all mesh functions
from .cartesian import UnstructuredSimplexBox, StructuredQuadBox, BoxInternalBoundary
from .spherical import SphericalShell, SphericalShellInternalBoundary, SegmentofSphere, CubedSphere
from .annulus import Annulus, QuarterAnnulus, SegmentofAnnulus, AnnulusWithSpokes, AnnulusInternalBoundary, DiscInternalBoundaries
from .geographic import RegionalSphericalBox
from .segmented import SegmentedSphericalSurface2D, SegmentedSphericalShell, SegmentedSphericalBall

# Make available at top level
__all__ = [
    'UnstructuredSimplexBox', 'StructuredQuadBox', 'BoxInternalBoundary',
    'SphericalShell', 'SphericalShellInternalBoundary', 'SegmentofSphere', 'CubedSphere',
    'Annulus', 'QuarterAnnulus', 'SegmentofAnnulus', 'AnnulusWithSpokes', 'AnnulusInternalBoundary', 'DiscInternalBoundaries',
    'RegionalSphericalBox',
    'SegmentedSphericalSurface2D', 'SegmentedSphericalShell', 'SegmentedSphericalBall'
]
```

### 2. Cartesian Module: `meshing/cartesian.py`
**Purpose**: Rectangular/box meshes in Cartesian coordinates
**Functions**:
- `UnstructuredSimplexBox()` (line 23)
- `StructuredQuadBox()` (line 252)  
- `BoxInternalBoundary()` (line 3992)

**Characteristics**:
- Uses minCoords/maxCoords or elementRes parameters
- Rectangular domains
- Cartesian coordinate systems

### 3. Spherical Module: `meshing/spherical.py`
**Purpose**: Spherical shells and 3D spherical geometries
**Functions**:
- `SphericalShell()` (line 568)
- `SphericalShellInternalBoundary()` (line 728)
- `SegmentofSphere()` (line 960)
- `CubedSphere()` (line 2538)

**Characteristics**:
- Uses radiusInner/radiusOuter parameters
- 3D spherical coordinate systems
- Full 3D shell geometries

### 4. Annulus Module: `meshing/annulus.py`
**Purpose**: 2D cylindrical/annular geometries
**Functions**:
- `Annulus()` (line 1415)
- `QuarterAnnulus()` (line 1247)
- `SegmentofAnnulus()` (line 1591)
- `AnnulusWithSpokes()` (line 1849)
- `AnnulusInternalBoundary()` (line 2114)
- `DiscInternalBoundaries()` (line 2327)

**Characteristics**:
- Uses radiusInner/radiusOuter parameters
- 2D cylindrical coordinate systems
- Annular/disc geometries

### 5. Geographic Module: `meshing/geographic.py`  
**Purpose**: Geographic/geodetic coordinate systems
**Functions**:
- `RegionalSphericalBox()` (line 2755)

**Characteristics**:
- Uses lat/lon coordinates
- Geographic projections
- Regional domains on sphere

### 6. Segmented Module: `meshing/segmented.py`
**Purpose**: Multi-segment and complex spherical geometries
**Functions**:
- `SegmentedSphericalSurface2D()` (line 3063)
- `SegmentedSphericalShell()` (line 3224)
- `SegmentedSphericalBall()` (line 3629)

**Characteristics**:
- Multi-segment geometries
- Complex spherical constructions
- Advanced meshing patterns

## Implementation Strategy

### Phase 1: Create Module Structure
1. Create `src/underworld3/meshing/` directory
2. Create `__init__.py` with backward compatibility imports
3. Create empty module files (cartesian.py, spherical.py, etc.)

### Phase 2: Move Functions
1. Copy common imports and utilities to each module
2. Move function groups to appropriate modules
3. Update MeshParameters for each mesh type
4. Test each module independently

### Phase 3: Integration
1. Update `__init__.py` imports
2. Test backward compatibility
3. Update documentation
4. Run full test suite

### Phase 4: Cleanup
1. Remove original `meshing.py`
2. Update any direct imports in codebase
3. Add module-specific documentation

## Benefits

### 🎯 **Maintainability**
- **Logical grouping**: Related mesh functions together
- **Smaller files**: ~500-800 lines per module vs 4443 lines
- **Clear responsibility**: Each module has specific purpose

### 🔍 **Discoverability**
- **Intuitive imports**: `from underworld3.meshing.cartesian import StructuredQuadBox`
- **Module-specific docs**: Targeted documentation per geometry type
- **Clear function groups**: Users know where to find specific mesh types

### 🔧 **Development**
- **Parallel development**: Different developers can work on different mesh types
- **Easier testing**: Module-specific test suites
- **Reduced conflicts**: Smaller files reduce merge conflicts

### 📚 **Documentation**
- **Focused examples**: Geometry-specific tutorials
- **Better organization**: Clear structure for docs
- **Specialized guides**: Coordinate system specific documentation

## Backward Compatibility

### ✅ **Preserved Patterns**
```python
# All existing imports continue to work
import underworld3 as uw
mesh = uw.meshing.Annulus(...)

# Direct imports still work  
from underworld3.meshing import Annulus
```

### 🆕 **New Import Options**
```python
# More specific imports available
from underworld3.meshing.annulus import Annulus
from underworld3.meshing.cartesian import StructuredQuadBox
```

## File Size Estimates
- **cartesian.py**: ~600 lines (3 functions)
- **spherical.py**: ~800 lines (4 functions) 
- **annulus.py**: ~1200 lines (6 functions)
- **geographic.py**: ~400 lines (1 function)
- **segmented.py**: ~1000 lines (3 functions)
- **__init__.py**: ~100 lines (imports + utilities)

**Total**: ~4100 lines (vs 4443 current)

## Dependencies

### Internal Dependencies
- All modules depend on base Mesh class
- MeshParameters dataclass (in discretisation_mesh.py)
- Common coordinate system types

### External Dependencies  
- gmsh (all modules)
- numpy (all modules)
- PETSc/petsc4py (all modules)
- underworld3.timing (all modules)

## Migration Checklist

### Pre-Migration
- [ ] Analyze function dependencies
- [ ] Identify shared utilities
- [ ] Plan MeshParameters updates
- [ ] Design module interfaces

### Migration Steps
- [ ] Create module directory structure
- [ ] Implement base __init__.py
- [ ] Migrate cartesian functions
- [ ] Migrate spherical functions  
- [ ] Migrate annulus functions
- [ ] Migrate geographic functions
- [ ] Migrate segmented functions
- [ ] Update all imports
- [ ] Test backward compatibility

### Post-Migration
- [ ] Update documentation
- [ ] Create module-specific examples
- [ ] Run full test suite
- [ ] Performance regression testing
- [ ] Remove original meshing.py

## Risks and Mitigations

### Risk: Import Breakage
**Mitigation**: Comprehensive backward compatibility in __init__.py

### Risk: Function Dependencies
**Mitigation**: Careful analysis of shared code and utilities

### Risk: Test Coverage
**Mitigation**: Module-specific test suites + integration tests

### Risk: Documentation Fragmentation  
**Mitigation**: Clear cross-references and unified examples

This plan provides a systematic approach to breaking up the monolithic meshing.py file while maintaining full backward compatibility and improving code organization.