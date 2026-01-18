# Developer Tools - WIP

Advanced developer utilities for Underworld3 internals, debugging, and performance analysis.

## DMPlex and PETSc Internals

### DMPlex Utilities
- **`Decoding_DMPlex.ipynb`** - Understanding PETSc DMPlex data structures
- **`DMPlex_Labelling_boundaries.ipynb`** - Boundary condition labeling techniques

### Mesh Refinement
- **`MeshRefine.py`** - Basic adaptive mesh refinement
- **`MeshRefine-AdaptiveLabel.py`** - Label-based adaptive refinement
- **`MeshRefine-DMPlexTricks.py`** - Advanced DMPlex refinement techniques  
- **`MeshRefine-ReadAdaptedCheckpoint.py`** - Restart from refined mesh checkpoints

## Data I/O and Checkpointing

### Checkpoint Management
- **`Ex_Checkpoint_Read_MeshVariable.py`** - Reading mesh variable checkpoints
- **`Ex_Checkpoint_Read_MeshVariable-vectors.py`** - Vector field checkpoint handling
- **`Ex_Checkpoint_Read_MeshVariable-vectors-old.py`** - Legacy checkpoint format
- **`Ex_Checkpoint_Write_XDMF.py`** - Writing XDMF format checkpoints

### Data I/O Testing
- **`HDF5.py`** - HDF5 file format utilities and testing
- **`swarmIO_test.py`** - Swarm data input/output validation

## Structural Optimization

### Shape Optimization
- **`SOpt.py`** - Structural optimization tests and shape recovery using Stokes flow
- **`SOpt-Viz.py`** - Visualization tools for structural optimization results

## Performance Analysis

### Timing Benchmarks
- **`Timing_Mesh_Build_Save.py`** - Mesh construction and save performance
- **`Timing_NonLinear_ShearBand.py`** - Nonlinear solver performance analysis
- **`Timing_StokesDisc_FSbcs.py`** - Stokes solver on disc with free slip BCs
- **`Timing_StokesSinker.py`** - Stokes sinker problem benchmarking
- **`Timing_StokesSphere_FSbcs.py`** - Spherical Stokes with free slip BCs
- **`Timing_StokesSphere_RT.py`** - Spherical Stokes with RT elements

## Usage Notes

⚠️ **Requirements**:
- Deep understanding of PETSc/DMPlex architecture
- Familiarity with Underworld3 internal APIs
- May require debugging/development builds

🎯 **Use cases**:
- **Core development** - Extending Underworld3 functionality
- **Performance optimization** - Identifying bottlenecks
- **Advanced meshing** - Custom mesh manipulation
- **Debugging** - Understanding internal data structures
- **Research** - Novel computational techniques

💡 **Tips**:
- Start with simpler examples in main documentation
- Consult PETSc documentation for DMPlex concepts
- Use these as reference for advanced implementations