# Work in Progress (WIP) Examples

This directory contains examples that are either:
- **Under development** - Not yet fully documented or tested
- **Advanced/specialized** - Require deep knowledge of Underworld3 internals
- **Legacy migration** - Recovered from underworld3-documentation but not yet integrated

## Organization

### [needs_update/](needs_update/)
Examples that do not run on the current library, moved here by #944 with the reason on each file's first line. Their paths below `needs_update/` are where they were.
- `fluid_mechanics/advanced/Ex_Navier_Stokes_Benchmarks_NS_DFG_2d.py`: particle recycling (Swarm recycle_rate > 1) is not implemented.
- `porous_flow/advanced/Ex_viscousFingering.py`: particle recycling (Swarm recycle_rate > 1) is not implemented.
- `fluid_mechanics/advanced/Ex_Navier_Stokes_Disk_Coriolis.py`: NavierStokes no longer takes velocityStar_fn.
- `fluid_mechanics/advanced/Ex_Navier_Stokes_Disk_FreeSlipBCs-Coriolis.py`: written for SwarmVariable.fn and the old Navier-Stokes history API.
- `fluid_mechanics/advanced/Ex_Navier_Stokes_Disk_NoSlipBCs-Coriolis_ss.py`: written for SwarmVariable.fn and the old Navier-Stokes history API.
- `fluid_mechanics/advanced/Ex_NS_Couette_PostProc.py`: post-processes another user's local output files, which are not in the repository.
- `fluid_mechanics/intermediate/Ex_NS_PipeFlow_PostProc.py`: post-processes another user's local output files, which are not in the repository.
- `fluid_mechanics/intermediate/Ex_Stokes_Disk_CylCoords_DEPRECATED_NATIVE.py`: native coordinate systems were removed in ee29df2a.
- `utilities/advanced/Ex_UsingCoordinates.py`: native coordinate systems were removed in ee29df2a; the periodic native mesh fails its geometry check.
- `utilities/advanced/Manifold_S2.5.py`: uses SegmentedSphere and native coordinates, removed in ee29df2a.
- `utilities/advanced/Manifold_S2.5-SphCoords.py`: uses SegmentedSphere and native coordinates, removed in ee29df2a.
- `utilities/advanced/MeshRefine-AdaptiveMetric-Dynamic.py`: needs a PETSc built with mmg (metric remeshing), which the standard environment does not have.
- `utilities/advanced/MeshRefine-AdaptiveMetric-Static.py`: needs a PETSc built with mmg (metric remeshing), which the standard environment does not have.
- `utilities/intermediate/Convection-AdaptedMesh.py`: needs an adapted-mesh checkpoint from the mmg remeshing examples.
- `free_surface/advanced/AnnulusND_FS-OuterSphere.py`: uses FSSA with the factory boundary normals on a deformed mesh, which the library refuses.
- `convection/advanced/Ex_Convection_Cartesian-Swarm.py`: written for the AdvDiffusionSwarm projection API, which is gone.
- `free_surface/advanced/AnnulusFS.py`: written for the old evaluate shapes and mesh-mover API, updated only in part.
- `utilities/advanced/Ex_Tensor_Variables_etc.py`: uses t_soln without defining it, and the AdvDiffusionSwarm API.
- `solid_mechanics/advanced/Ex_Sheared_Layer_Visualisation.py`: reads swarms with Swarm.load, which is gone.

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