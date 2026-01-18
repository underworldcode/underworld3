---
title: "Checkpointing and Restart System"
---

# Checkpointing and Restart Architecture

Underworld3 provides a flexible checkpointing system for simulation data persistence, restart capabilities, and post-processing workflows.

## System Overview

The checkpointing system is built around **PETSc HDF5** files and uses **KDTree interpolation** for flexible restart scenarios.

### Key Components

1. **`mesh.write_timestep()`** - Primary checkpointing interface
2. **KDTree mapping** - Enables flexible mesh remapping on restart
3. **PETSc HDF5 format** - Parallel-safe, standardized file format
4. **Swarm checkpointing** - Particle data persistence (limited down-sampling)

## Mesh Checkpointing

### Basic Usage

```python
# Save checkpoint
mesh.write_timestep("simulation_step_100.h5", 
                   meshVars=[velocity, pressure, temperature],
                   time=100.0, step=100)

# Restart from checkpoint  
mesh.read_timestep("simulation_step_100.h5", 
                  meshVars=[velocity, pressure, temperature])
```

### Advanced Features

#### Different Mesh Geometries
```python
# Original simulation: 32x32 elements
mesh_coarse = uw.meshing.StructuredQuadBox(elementRes=(32, 32))

# Restart simulation: 64x64 elements (refined)
mesh_fine = uw.meshing.StructuredQuadBox(elementRes=(64, 64))

# KDTree automatically maps data from coarse to fine mesh
mesh_fine.read_timestep("coarse_simulation_step_100.h5",
                       meshVars=[velocity, pressure, temperature])
```

#### Different Parallel Decompositions
```python
# Original run: 4 processes
# mpirun -np 4 python simulation.py

# Restart run: 8 processes (different decomposition)
# mpirun -np 8 python restart_simulation.py

# Checkpointing system handles parallel decomposition changes automatically
mesh.read_timestep("4_process_checkpoint.h5", meshVars=[fields])
```

### Technical Implementation

#### KDTree Mapping Process
1. **Coordinate extraction**: Target mesh node coordinates identified
2. **KDTree search**: Find nearest source mesh nodes for each target
3. **Interpolation weights**: Compute interpolation coefficients
4. **Value mapping**: Interpolate field values to new mesh nodes
5. **Boundary handling**: Special treatment for domain boundaries

#### File Format Details
- **Format**: PETSc HDF5 with parallel I/O
- **Metadata**: Time, step number, mesh geometry info
- **Field data**: All MeshVariable data with proper chunking
- **Compression**: Optional HDF5 compression for large datasets

## Swarm Checkpointing

### Capabilities and Limitations

#### What Works Well
```python
# Save swarm state
swarm.save("particles_step_100.h5")

# Restore to same or different mesh
swarm.load("particles_step_100.h5")
```

#### Key Characteristics
- **Mesh-resolution independent**: Particles can be loaded to different mesh resolutions
- **No down-sampling strategy**: All particles are saved/restored (no intelligent reduction)
- **Spatial integrity**: Particle positions and data preserved exactly
- **Migration compatibility**: Works with mesh parallel decomposition changes

### Use Cases
- **Material tracking**: Persistent material property histories
- **Lagrangian tracers**: Pathline and geological history tracking  
- **Passive markers**: Flow visualization and analysis
- **Material interfaces**: Sharp interface tracking

## Production Workflows

### Checkpoint-Based Visualization

Instead of attempting real-time parallel visualization:

```python
# Simulation loop with regular checkpointing
for step in range(1000):
    # ... solve physics ...
    
    if step % 10 == 0:  # Every 10 steps
        mesh.write_timestep(f"output/step_{step:04d}.h5",
                           meshVars=[velocity, pressure, temperature],
                           time=step * dt, step=step)

# Separate visualization workflow (single process)
import glob
checkpoint_files = sorted(glob.glob("output/step_*.h5"))

for checkpoint in checkpoint_files:
    # Load data for visualization
    mesh.read_timestep(checkpoint, meshVars=[velocity, pressure, temperature])
    
    # Create high-quality visualization
    create_full_field_visualization(mesh, velocity, pressure, temperature)
```

### Multi-Resolution Analysis

```python
# High-resolution simulation checkpoint
mesh_hr = uw.meshing.StructuredQuadBox(elementRes=(256, 256))
# ... run simulation, save checkpoint ...

# Load to lower resolution for analysis
mesh_lr = uw.meshing.StructuredQuadBox(elementRes=(64, 64))  
mesh_lr.read_timestep("hr_simulation_final.h5", meshVars=[fields])

# Fast analysis on reduced data
perform_statistical_analysis(fields)
```

## Performance Characteristics

### Checkpoint File Sizes
- **Mesh variables**: Proportional to mesh resolution and field count
- **Swarm data**: Proportional to particle count (can be very large)
- **Compression**: HDF5 compression can reduce sizes significantly
- **Typical sizes**: 10MB - 10GB depending on problem scale

### I/O Performance  
- **Parallel writing**: Scales well with process count
- **Reading speed**: Fast restart due to parallel HDF5
- **Storage requirements**: Plan for 5-20% of simulation time for I/O

### Memory Usage
- **KDTree construction**: Temporary memory spike during interpolation
- **Field mapping**: Additional memory for intermediate interpolation arrays
- **Swarm loading**: Full swarm data loaded into memory

## Best Practices

### Checkpoint Frequency
```python
# Balance between restart capability and I/O cost
checkpoint_interval = max(10, total_steps // 100)  # At least every 10 steps

if step % checkpoint_interval == 0:
    mesh.write_timestep(f"restart_{step}.h5", ...)
```

### File Management
```python
# Implement checkpoint rotation to manage disk space
def cleanup_old_checkpoints(keep_last=5):
    checkpoints = sorted(glob.glob("restart_*.h5"))
    for old_checkpoint in checkpoints[:-keep_last]:
        os.remove(old_checkpoint)
```

### Metadata Tracking
```python
# Include simulation parameters in checkpoint
checkpoint_metadata = {
    "rayleigh_number": Ra,
    "viscosity_contrast": eta_contrast,
    "boundary_conditions": bc_info,
    "mesh_resolution": mesh.data.shape[0]
}

# Store as HDF5 attributes or separate JSON file
```

## Integration with Model Orchestration

### Future Direction: Model-Centric Checkpointing

The checkpointing system may migrate from `mesh.write_timestep()` to `model.save_checkpoint()`:

```python
# Current approach
mesh.write_timestep("step_100.h5", meshVars=[v, p, T])

# Future model-centric approach  
model = uw.Model(mesh=mesh, velocity=v, pressure=p, temperature=T)
model.save_checkpoint("step_100.h5", include_parameters=True)
```

**Benefits:**
- **Parameter persistence**: Automatically save constitutive parameters
- **Model validation**: Ensure checkpoint compatibility with model structure  
- **Metadata integration**: Rich simulation metadata in checkpoint files
- **Version control**: Track model version and parameter changes

## Error Handling and Robustness

### Common Issues and Solutions

#### Mesh Incompatibility
```python
try:
    mesh.read_timestep("checkpoint.h5", meshVars=[fields])
except uw.CheckpointError as e:
    print(f"Mesh incompatibility: {e}")
    print("Consider using interpolation mode or checking mesh geometry")
```

#### Missing Fields
```python
# Partial field loading when checkpoint has different fields
available_fields = mesh.get_checkpoint_fields("checkpoint.h5")
loadable_fields = [f for f in [v, p, T] if f.name in available_fields]
mesh.read_timestep("checkpoint.h5", meshVars=loadable_fields)
```

#### Parallel Inconsistency
```python
# Ensure all processes participate in checkpoint operations
uw.mpi.barrier()  # Synchronize before checkpoint
mesh.write_timestep("checkpoint.h5", meshVars=[fields])
uw.mpi.barrier()  # Synchronize after checkpoint
```

## Summary

The Underworld3 checkpointing system provides:

- ✅ **Flexible restart**: Different meshes and parallel decompositions
- ✅ **Production-ready**: Robust parallel HDF5 I/O
- ✅ **KDTree interpolation**: Automatic field remapping
- ✅ **Swarm support**: Complete particle state preservation
- ⚠️ **No swarm down-sampling**: Memory constraints for large particle counts
- 🔄 **Future enhancement**: Model-centric approach with rich metadata

The system enables robust simulation workflows, flexible post-processing, and production-scale computational geodynamics applications.