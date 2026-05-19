# Checkpoint Output And Reload Methods

UW3 currently has two output/reload workflows for mesh and mesh-variable data.
They serve different purposes and should not be treated as interchangeable.

## Method A: `write_timestep()` / `read_timestep()`

This is the visualisation and flexible remap workflow.

Example:

```python
mesh.write_timestep(
    "output",
    index=0,
    outputPath=str(output_dir),
    meshVars=[velocity, pressure],
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

### Advantages

- Produces XDMF/HDF5 files suitable for visualisation workflows.
- Can remap data onto a different mesh or a different node layout.
- Useful for postprocessing where exact finite-element section identity is not
  required.

### Disadvantages

- Reload is not an exact PETSc FE-vector restart path.
- The KDTree/remap step can be memory-heavy for large meshes.
- At high MPI counts, remap memory can dominate postprocessing memory use.
- Discontinuous fields and high-order fields rely on coordinate remap behavior
  rather than PETSc section metadata.

## Method B: `write_checkpoint()` / `read_checkpoint()`

This is the restart and exact postprocessing workflow.

Example:

```python
mesh.write_checkpoint(
    "checkout",
    index=0,
    outputPath=str(output_dir),
    meshVars=[velocity, pressure],
)
```

Default files:

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

The checkpoint files store PETSc DMPlex HDF5 storage version `3.0.0` data with
the section/vector metadata required to reconstruct finite-element vectors.
Reloading uses PETSc DMPlex topology, section, vector, and `PetscSF` metadata.
It does not use KDTree coordinate remapping.

### Advantages

- Exact FE-vector reload path for restart and postprocessing.
- Avoids KDTree memory spikes.
- Preserves continuous, vector, and discontinuous variable layouts through
  PETSc section metadata.
- Per-variable files avoid forcing postprocessing to open one large combined
  field checkpoint.
- Better suited to large MPI jobs where memory locality matters.

### Disadvantages

- Does not write XDMF.
- Does not write `/vertex_fields/...` visualisation datasets.
- Assumes the checkpoint mesh and variable checkpoint files are used together.
- Different-rank reload should be validated for each workflow before relying on
  it in production.

## Which Method To Use

| Use case | Recommended method |
| --- | --- |
| ParaView/XDMF visualisation | `write_timestep()` |
| Flexible remap onto another mesh | `write_timestep()` |
| Exact restart/postprocessing | `write_checkpoint()` |
| Large spherical benchmark metric evaluation | `write_checkpoint()` |
| Avoid KDTree memory growth at high MPI counts | `write_checkpoint()` |

It is valid for production scripts to write both:

```python
mesh.write_timestep("output", index=0, outputPath=str(output_dir), meshVars=[v, p])
mesh.write_checkpoint("checkout", index=0, outputPath=str(output_dir), meshVars=[v, p])
```

The first output is for visualisation. The second output is for restart or
metrics-from-checkpoint postprocessing.

## Spherical Benchmark Evidence

The spherical Thieulot benchmark exposed the practical difference between the
two methods. Boundary metric evaluation is run in a second step after the Stokes
solve. The old reload path used `write_timestep()` output and `read_timestep()`;
the new path uses `write_checkpoint()` output and `read_checkpoint()`.

### Resource Usage

| Resolution | Method | NCPUs | Walltime | CPU time | Memory used | Exit status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `1/64` | `write_timestep/read_timestep` | 144 | `00:03:43` | `07:04:27` | `211.27 GB` | `0` |
| `1/64` | `write_checkpoint/read_checkpoint` | 144 | `00:02:41` | `05:21:14` | `233.67 GB` | `0` |
| `1/128` | `write_timestep/read_timestep` | 1152 | `00:13:55` | `214:02:57` | `3.92 TB` | `0` |
| `1/128` | `write_checkpoint/read_checkpoint` | 1152 | `00:03:57` | `64:19:53` | `1.83 TB` | `0` |

For the `1/128` case, checkpoint reload reduced memory by about `2.09 TB` and
reduced walltime by about `3.5x`.

### Metric Agreement

`1/128` spherical Thieulot benchmark:

| Metric | `write_timestep/read_timestep` | `write_checkpoint/read_checkpoint` | Difference |
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

| Metric | `write_timestep/read_timestep` | `write_checkpoint/read_checkpoint` | Difference |
| --- | ---: | ---: | ---: |
| `v_l2_norm` | `1.1662200663950889e-05` | `1.1662200663957042e-05` | `6.15e-18` |
| `p_l2_norm` | `2.7573367818459473e-03` | `2.7573367818460497e-03` | `1.02e-16` |
| `sigma_rr_l2_norm_lower` | `4.368560398155481e-03` | `4.381908965541248e-03` | `1.33e-05` |
| `sigma_rr_l2_norm_upper` | `1.6315543718450765e-04` | `1.6047310456195621e-04` | `-2.68e-06` |

## Notes For Benchmark Scripts

For production benchmark workflows:

- run the solve stage first
- write `write_timestep()` output if visualisation files are needed
- write `write_checkpoint()` output for restart/postprocessing
- exit before metric evaluation
- run a second metrics-from-checkpoint job
- reload mesh from `<base>.mesh.<index>.h5`
- reload fields with `MeshVariable.read_checkpoint(...)`
- compute metrics from reloaded fields

This separates solver memory from postprocessing memory and avoids the KDTree
reload path for large benchmark metric jobs.
