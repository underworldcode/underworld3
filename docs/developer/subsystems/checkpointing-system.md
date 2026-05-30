---
title: "Checkpointing and Restart System"
---

# Checkpointing and Restart System

Underworld3 has two related but distinct output/reload layers:

- **Mesh-variable output**, written with `Mesh.write_timestep()`.
- **Whole-model snapshots**, written with `Model.save_state()`.

Use `Mesh.write_timestep()` when you want selected mesh variables for
visualisation, postprocessing, remapping, or PETSc-native reload. Use
`Model.save_state()` when you want to capture the state of a full model,
including registered meshes, variables, swarms, and Python-side state bearers.

## Mesh-Variable Output

`Mesh.write_timestep()` is the standard mesh and mesh-variable output method.
It writes one mesh HDF5 file and one HDF5 file per requested mesh variable.
Each variable file always contains `/fields` coordinate/value datasets used by
`MeshVariable.read_timestep()`.

Optional payloads are controlled by explicit flags:

| Flag | Payload | Reader / use |
| --- | --- | --- |
| `create_xdmf=True` | XDMF-compatible visualisation datasets and a companion `.xdmf` file | ParaView and other XDMF tools |
| `petsc_reload=True` | PETSc DMPlex section/vector metadata | `MeshVariable.read_checkpoint()` |

### Visualisation and Coordinate Remap

```python
mesh.write_timestep(
    "output",
    index=100,
    outputPath="output",
    meshVars=[velocity, pressure, temperature],
    time=100.0,
    create_xdmf=True,
)
```

Typical files:

```text
output/output.mesh.00100.h5
output/output.mesh.velocity.00100.h5
output/output.mesh.pressure.00100.h5
output/output.mesh.temperature.00100.h5
output/output.mesh.00100.xdmf
```

Reload selected variables through the coordinate/KDTree remap path:

```python
velocity.read_timestep("output", "velocity", 100, outputPath="output")
pressure.read_timestep("output", "pressure", 100, outputPath="output")
```

`read_timestep()` compares saved coordinates with the target variable's live
coordinates and maps values by the coordinate-remap path. This is useful when
the target mesh or MPI decomposition differs from the one used to write the
files. It is not an exact PETSc finite-element vector reload.

### PETSc-Native Reload

For exact finite-element vector reload, add `petsc_reload=True` and load with
`MeshVariable.read_checkpoint()`:

```python
mesh.write_timestep(
    "restart",
    index=100,
    outputPath="output",
    meshVars=[velocity, pressure],
    create_xdmf=False,
    petsc_reload=True,
)

velocity.read_checkpoint(
    "output/restart.mesh.velocity.00100.h5",
    data_name="velocity",
)
pressure.read_checkpoint(
    "output/restart.mesh.pressure.00100.h5",
    data_name="pressure",
)
```

Typical files:

```text
output/restart.mesh.00100.h5
output/restart.mesh.velocity.00100.h5
output/restart.mesh.pressure.00100.h5
```

The variable files contain `/fields` datasets and PETSc reload metadata under
`/topologies/uw_mesh/dms/<variable>/`. `read_checkpoint()` uses PETSc DMPlex
topology, section, vector, and `PetscSF` metadata. It does not use KDTree
remapping.

### Unified Visualisation and PETSc Reload

Set both flags when one output family should support ParaView, coordinate
remap, and PETSc-native reload:

```python
mesh.write_timestep(
    "output",
    index=100,
    outputPath="output",
    meshVars=[velocity, pressure],
    create_xdmf=True,
    petsc_reload=True,
)
```

The same variable file can then be read by:

- `MeshVariable.read_timestep(...)` for coordinate/KDTree remap.
- `MeshVariable.read_checkpoint(...)` for PETSc-native reload.

## Compatibility API

`Mesh.write_checkpoint()` is retained for older scripts and emits a
`FutureWarning`. New code should use:

```python
mesh.write_timestep(..., petsc_reload=True)
```

instead of:

```python
mesh.write_checkpoint(...)
```

The compatibility method writes PETSc-reloadable files, but it uses legacy
checkpoint-style variable filenames. The standard `write_timestep()` method
keeps one naming convention for visualisation, remap, and PETSc reload output.

## Whole-Model Snapshots

`Model.save_state()` and `Model.load_state()` are the model-level snapshot API.

```python
token = model.save_state()
model.load_state(token)
```

Without `file=`, `save_state()` returns an in-memory token for short-lived
backtracking inside a running process.

```python
model.save_state(file="run.snap.h5")
model.load_state("run.snap.h5")
```

With `file=`, `save_state()` writes a persistent snapshot wrapper plus a sibling
bulk directory:

```text
run.snap.h5
run.snap.bulk/
    {mesh}.mesh.00000.h5
    {mesh}.{variable}.00000.h5
    {swarm}.swarm.h5
```

The wrapper file stores model metadata and references to the bulk files. The
mesh-variable bulk files are PETSc-reloadable and are read through
`MeshVariable.read_checkpoint()` during `Model.load_state(...)`.

Disk snapshots are for same-model restart. The model should already contain
compatible registered meshes, variables, swarms, and state bearers when
`load_state()` is called. For remapping onto a different mesh or MPI layout,
use `Mesh.write_timestep()` plus `MeshVariable.read_timestep()`.

## Choosing the Correct Path

| Need | Write | Read |
| --- | --- | --- |
| ParaView/XDMF visualisation | `Mesh.write_timestep(create_xdmf=True)` | External visualisation tool |
| Remap selected variables onto another mesh or decomposition | `Mesh.write_timestep(...)` | `MeshVariable.read_timestep(...)` |
| PETSc-native reload of selected variables | `Mesh.write_timestep(petsc_reload=True)` | `MeshVariable.read_checkpoint(...)` |
| One file family for visualisation, remap, and PETSc reload | `Mesh.write_timestep(create_xdmf=True, petsc_reload=True)` | `read_timestep()` or `read_checkpoint()` depending on intent |
| In-process backtracking | `Model.save_state()` | `Model.load_state(token)` |
| Persistent whole-model restart | `Model.save_state(file=...)` | `Model.load_state(path)` |

## Performance Notes

`read_timestep()` is flexible because it is coordinate based, but the remap can
be memory-heavy for large meshes and high MPI counts.

`read_checkpoint()` is less flexible but better suited to exact restart and
large postprocessing jobs because it reloads PETSc finite-element data directly
from DMPlex metadata.

For production jobs that need both visualisation and restart, prefer:

```python
mesh.write_timestep(
    "output",
    index=step,
    outputPath="output",
    meshVars=[velocity, pressure],
    create_xdmf=True,
    petsc_reload=True,
)
```

This makes the output intent explicit and avoids maintaining separate user APIs
for XDMF output and PETSc reload output.
