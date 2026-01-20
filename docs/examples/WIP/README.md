# Work in Progress (WIP) Examples

This directory contains examples that are either:
- **Under development** - Not yet fully documented or tested
- **Advanced/specialized** - Require deep knowledge of Underworld3 internals
- **Legacy migration** - Recovered from underworld3-documentation but not yet integrated

## Organization

### 📁 [porous_flow/](porous_flow/)
Advanced porous flow examples that complete the porous flow physics domain.
- Ex_Explicit_Flow_Grains.ipynb - Explicit grain-scale flow modeling
- pramoda_stokes.ipynb - Stokes flow in porous media

### 📁 [mesh_advanced/](mesh_advanced/) 
Low-level mesh manipulation and DMPlex integration examples.
- cuttingtetmesh_mmg.ipynb - Mesh cutting with MMG
- Ex_Dmplex_from_Petsc4py.ipynb - Direct PETSc DMPlex usage

### 📁 [developer_tools/](developer_tools/)
Advanced developer utilities for debugging, performance analysis, and I/O.
- **DMPlex internals**: DMPlex decoding and boundary labelling
- **Checkpointing**: Advanced HDF5 and XDMF checkpoint handling  
- **Mesh refinement**: Adaptive mesh refinement techniques
- **Performance timing**: Benchmarking various solver configurations
- **I/O testing**: HDF5 and swarm data I/O validation

### 📁 [post_processing/](post_processing/)
Advanced visualization and analysis techniques.
- Ex_MoresiSolomatov_Convection_Visualisation-Cart.ipynb - Complex convection visualization

## Usage Notes

⚠️ **These examples may**:
- Require additional dependencies not in standard environment
- Need modification to run with current Underworld3 version
- Be incomplete or experimental
- Lack comprehensive documentation

✅ **Value for**:
- **Advanced users** exploring Underworld3 internals
- **Developers** working on core functionality
- **Researchers** needing specialized techniques
- **Learning** advanced computational geodynamics methods

## Migration Status

These examples were recovered from `underworld3-documentation` and represent ~96% coverage of all original examples:
- **Migrated to main docs**: 121 examples
- **Migrated to WIP**: 25 examples (this directory)
- **Total coverage**: 146/156 files (93.6%)

## Contributing

If you enhance or complete any WIP example:
1. Move it to the appropriate physics domain directory
2. Add proper documentation and parameter descriptions
3. Ensure it passes validation pipeline
4. Update this README