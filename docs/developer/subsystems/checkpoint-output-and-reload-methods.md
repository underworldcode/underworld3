# Checkpoint Output And Reload Methods

`write_timestep()` is the standard mesh and mesh-variable output API. It can
write either or both output payloads:

- XDMF/remap payloads for ParaView and `read_timestep()`.
- PETSc DMPlex section/vector payloads for `read_checkpoint()`.

`write_checkpoint()` is retained as a compatibility wrapper for older scripts,
but new code should use `write_timestep(..., petsc_reload=True)`.

## Standard API

`write_timestep()` always writes the mesh file and one HDF5 file per mesh
variable. Mesh-variable files always contain raw coordinate/value datasets under
`/fields`, which are the source data used by `MeshVariable.read_timestep()` for
coordinate/KDTree remapping.

The two optional payloads are selected with explicit flags:

| Flag | Output payload | Reader/use case |
| --- | --- | --- |
| `create_xdmf=True` | `/vertex_fields` or `/cell_fields` compatibility datasets plus a companion `.xdmf` file | ParaView/XDMF visualisation |
| `petsc_reload=True` | PETSc DMPlex section/vector metadata under `/topologies/uw_mesh/dms/...` | `MeshVariable.read_checkpoint()` PETSc-native reload |

### Visualisation And Remap

```python
mesh.write_timestep(
    "output",
    index=0,
    outputPath=str(output_dir),
    meshVars=[velocity, pressure],
    create_xdmf=True,
    petsc_reload=False,
)

velocity.read_timestep("output", "Velocity", 0, outputPath=str(output_dir))
pressure.read_timestep("output", "Pressure", 0, outputPath=str(output_dir))
```

Typical files:

```text
output.mesh.00000.h5
output.mesh.Velocity.00000.h5
output.mesh.Pressure.00000.h5
output.mesh.00000.xdmf
```

The field files contain coordinate/value datasets such as `/fields/<name>` and
`/fields/coordinates`, plus vertex-field datasets for visualisation. Reloading
uses coordinate-based remapping. In practice this means the target variable is
filled by comparing target coordinates to source coordinates, using a KDTree or
similar nearest-neighbour/remap process.

### Unified Visualisation And PETSc Reload

Set both flags to write one file family that supports ParaView/XDMF,
coordinate/KDTree remap, and PETSc-native reload:

```python
mesh.write_timestep(
    "output",
    index=0,
    outputPath=str(output_dir),
    meshVars=[velocity, pressure],
    create_xdmf=True,
    petsc_reload=True,
)

velocity.read_checkpoint(
    output_dir / "output.mesh.Velocity.00000.h5",
    data_name="Velocity",
)
```

With both `create_xdmf=True` and `petsc_reload=True`, the same variable file can
be used by `read_timestep()` for coordinate/KDTree remapping and by
`read_checkpoint()` for exact PETSc-native reload.

### PETSc Reload Without XDMF

For PETSc reload output without ParaView/XDMF payloads, use:

```python
mesh.write_timestep(
    "restart",
    index=0,
    outputPath=str(output_dir),
    meshVars=[velocity, pressure],
    create_xdmf=False,
    petsc_reload=True,
)
```

This still writes raw `/fields` datasets, but it does not write
`/vertex_fields`, `/cell_fields`, or a companion `.xdmf` file.

Typical PETSc-reload-only files still use the timestep naming convention:

```text
restart.mesh.00000.h5
restart.mesh.Velocity.00000.h5
restart.mesh.Pressure.00000.h5
```

The variable files contain raw `/fields` datasets and PETSc reload metadata
under `/topologies/uw_mesh/dms/<variable>/`.

### Advantages

- Produces XDMF/HDF5 files suitable for visualisation workflows.
- Can remap data onto a different mesh or a different node layout.
- Useful for postprocessing where exact finite-element section identity is not
  required.
- Can also be made PETSc-reloadable with `petsc_reload=True`.

### Disadvantages

- Reload is not an exact PETSc FE-vector restart path unless
  `petsc_reload=True` is used and the field is loaded with `read_checkpoint()`.
- The KDTree/remap step can be memory-heavy for large meshes.
- At high MPI counts, remap memory can dominate postprocessing memory use.
- Discontinuous fields and high-order fields rely on coordinate remap behavior
  rather than PETSc section metadata.

## Legacy Compatibility

`write_checkpoint()` is deprecated and retained for existing callers. It emits a
`FutureWarning` directing users to `write_timestep(..., petsc_reload=True)`.

Legacy call:

```python
mesh.write_checkpoint(
    "checkout",
    index=0,
    outputPath=str(output_dir),
    meshVars=[velocity, pressure],
    create_xdmf=False,
)
```

Preferred replacement:

```python
mesh.write_timestep(
    "checkout",
    index=0,
    outputPath=str(output_dir),
    meshVars=[velocity, pressure],
    create_xdmf=False,
    petsc_reload=True,
)
```

The preferred replacement writes timestep-style files:

```text
checkout.mesh.00000.h5
checkout.mesh.Velocity.00000.h5
checkout.mesh.Pressure.00000.h5
```

The legacy call writes checkpoint-style variable filenames:

```text
checkout.mesh.00000.h5
checkout.Velocity.00000.h5
checkout.Pressure.00000.h5
```

Reload:

```python
mesh = uw.discretisation.Mesh("checkout.mesh.00000.h5")
velocity = uw.discretisation.MeshVariable("Velocity", mesh, mesh.dim, degree=2)
pressure = uw.discretisation.MeshVariable("Pressure", mesh, 1, degree=1)

velocity.read_checkpoint("checkout.Velocity.00000.h5", data_name="Velocity")
pressure.read_checkpoint("checkout.Pressure.00000.h5", data_name="Pressure")
```

By default, `write_checkpoint()` writes one checkpoint file per mesh variable.
Use `separate_variable_files=False` to write all variables to one file:

```python
mesh.write_checkpoint(
    "checkout",
    index=0,
    outputPath=str(output_dir),
    meshVars=[velocity, pressure],
    separate_variable_files=False,
)
```

Combined variable file:

```text
checkout.checkpoint.00000.h5
```

These files store PETSc DMPlex HDF5 storage version `3.0.0` data with the
section/vector metadata required to reconstruct finite-element vectors.
Reloading uses PETSc DMPlex topology, section, vector, and `PetscSF` metadata;
it does not use KDTree coordinate remapping.

Set `create_xdmf=True` to route through the unified timestep writer. This writes
XDMF/remap payloads and PETSc reload payloads together, using the timestep file
layout:

```python
mesh.write_checkpoint(
    "checkout",
    index=0,
    outputPath=str(output_dir),
    meshVars=[velocity, pressure],
    create_xdmf=True,
)
```

The variable files are then named
`checkout.mesh.<variable>.<index>.h5` rather than
`checkout.<variable>.<index>.h5`. Because this mode uses the timestep file
layout, it does not support `unique_id=True` or
`separate_variable_files=False`.

New code should prefer the equivalent `write_timestep()` calls above.

## Which Method To Use

| Use case | Recommended method |
| --- | --- |
| ParaView/XDMF visualisation | `write_timestep(..., create_xdmf=True)` |
| Flexible remap onto another mesh | `write_timestep(...)` with `MeshVariable.read_timestep(...)` |
| Exact restart/postprocessing | `write_timestep(..., create_xdmf=False, petsc_reload=True)` |
| Unified visualisation/remap plus PETSc reload | `write_timestep(..., create_xdmf=True, petsc_reload=True)` |
| Avoid KDTree memory growth at high MPI counts | `write_timestep(..., petsc_reload=True)` with `read_checkpoint()` |

For unified output, write both payload families in one call:

```python
mesh.write_timestep(
    "output",
    index=0,
    outputPath=str(output_dir),
    meshVars=[v, p],
    create_xdmf=True,
    petsc_reload=True,
)
```

If separate visualisation and restart-style file families are wanted, use two
`write_timestep()` calls with different base names:

```python
mesh.write_timestep("output", index=0, outputPath=str(output_dir), meshVars=[v, p])
mesh.write_timestep(
    "restart",
    index=0,
    outputPath=str(output_dir),
    meshVars=[v, p],
    create_xdmf=False,
    petsc_reload=True,
)
```

The first output is for visualisation/remap. The second output is for restart or
metrics-from-checkpoint postprocessing.

## Spherical Benchmark Evidence

The spherical Thieulot benchmark exposed the practical difference between
coordinate/KDTree remap and PETSc-native reload. Boundary metric evaluation is
run in a second step after the Stokes solve. The old reload path used
`read_timestep()`; the newer path uses PETSc DMPlex section/vector metadata and
`read_checkpoint()`. New output should be written through
`write_timestep(..., petsc_reload=True)`.

### Resource Usage

| Resolution | Method | NCPUs | Walltime | CPU time | Memory used | Exit status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `1/64` | `read_timestep` remap | 144 | `00:03:43` | `07:04:27` | `211.27 GB` | `0` |
| `1/64` | PETSc `read_checkpoint` reload | 144 | `00:02:41` | `05:21:14` | `233.67 GB` | `0` |
| `1/128` | `read_timestep` remap | 1152 | `00:13:55` | `214:02:57` | `3.92 TB` | `0` |
| `1/128` | PETSc `read_checkpoint` reload | 1152 | `00:03:57` | `64:19:53` | `1.83 TB` | `0` |

For the `1/128` case, checkpoint reload reduced memory by about `2.09 TB` and
reduced walltime by about `3.5x`.

### Metric Agreement

`1/128` spherical Thieulot benchmark:

| Metric | `read_timestep` remap | PETSc `read_checkpoint` reload | Difference |
| --- | ---: | ---: | ---: |
| `v_l2_norm` | `1.4319274480265082e-06` | `1.4319274480231255e-06` | `-3.38e-18` |
| `p_l2_norm` | `5.985841567394967e-04` | `5.985841567395382e-04` | `4.15e-17` |
| `p_l2_norm_abs` | `1.0566381005355924e-03` | `1.0566381005356654e-03` | `7.30e-17` |
| `sigma_rr_l2_norm_lower` | `1.117914337768646e-03` | `1.1256362820288926e-03` | `7.72e-06` |
| `sigma_rr_l2_norm_upper` | `4.461443231341268e-05` | `3.811141458727819e-05` | `-6.50e-06` |
| `u_dot_n_l2_norm_lower_abs` | `2.2509850571644799e-04` | `2.2509850571645164e-04` | `3.65e-18` |
| `u_dot_n_l2_norm_upper_abs` | `5.535239716141496e-05` | `5.535239716141875e-05` | `3.79e-18` |

Velocity, pressure, and normal-velocity metrics agree to roundoff. The
remaining `sigma_rr` differences are small and come from the benchmark stress
recovery path. The checkpoint workflow computes stress after reload by
projecting deviatoric-stress components and forming `sigma_rr`; it does not
reuse the old `read_timestep()` remap path.

`1/64` spherical Thieulot benchmark:

| Metric | `read_timestep` remap | PETSc `read_checkpoint` reload | Difference |
| --- | ---: | ---: | ---: |
| `v_l2_norm` | `1.1662200663950889e-05` | `1.1662200663957042e-05` | `6.15e-18` |
| `p_l2_norm` | `2.7573367818459473e-03` | `2.7573367818460497e-03` | `1.02e-16` |
| `sigma_rr_l2_norm_lower` | `4.368560398155481e-03` | `4.381908965541248e-03` | `1.33e-05` |
| `sigma_rr_l2_norm_upper` | `1.6315543718450765e-04` | `1.6047310456195621e-04` | `-2.68e-06` |

## Notes For Benchmark Scripts

For production benchmark workflows:

- run the solve stage first
- use `write_timestep(..., create_xdmf=True, petsc_reload=True)` if one unified
  file family should support visualisation, remap, and PETSc-native reload
- alternatively, write separate `write_timestep()` outputs for visualisation and
  PETSc reload by changing `create_xdmf` and `petsc_reload`
- exit before metric evaluation
- run a second metrics-from-checkpoint job
- reload mesh from `<base>.mesh.<index>.h5`
- reload fields with `MeshVariable.read_checkpoint(...)`
- compute metrics from reloaded fields

This separates solver memory from postprocessing memory and avoids the KDTree
reload path for large benchmark metric jobs.
