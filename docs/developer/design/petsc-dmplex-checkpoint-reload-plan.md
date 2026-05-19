# PETSc DMPlex Checkpoint Reload Plan

## Commit And Test Workflow

Use small, meaningful commits while implementing this work.

Do `git add` and `git commit` after each coherent feature, fix, or debugging
checkpoint so progress is easy to inspect and bisect. Do not wait until the end
to commit everything as one large change, and do not commit after every tiny
edit.

Create or update unit tests whenever they are needed to prove behavior or
prevent regressions. Prefer adding a failing regression test before the fix when
the failure mode is already known.

## Objective

Provide an exact PETSc DMPlex checkpoint reload path for UW3 mesh variables.
This path is intended for restart and large-scale postprocessing, not
visualisation.

Target workflow:

```python
mesh.write_checkpoint(
    "checkout",
    outputPath=str(output_dir),
    meshVars=[v_soln, p_soln],
    index=0,
)
```

Default output:

```text
checkout.mesh.00000.h5
checkout.Velocity.00000.h5
checkout.Pressure.00000.h5
```

Reload workflow:

```python
mesh = uw.discretisation.Mesh("checkout.mesh.00000.h5")
v_soln = uw.discretisation.MeshVariable("Velocity", mesh, mesh.dim, degree=2)
p_soln = uw.discretisation.MeshVariable("Pressure", mesh, 1, degree=1)

v_soln.read_checkpoint("checkout.Velocity.00000.h5", data_name="Velocity")
p_soln.read_checkpoint("checkout.Pressure.00000.h5", data_name="Pressure")
```

The reload path must not use `KDTree` remapping. It must restore FE data through
PETSc DMPlex topology, section, vector, and `PetscSF` metadata.

## Existing Output Methods

UW3 has two related but different output paths.

| Method | Purpose | Reload method | Strength | Limitation |
| --- | --- | --- | --- | --- |
| `mesh.write_timestep(...)` | Visualisation and flexible field remap | `MeshVariable.read_timestep(...)` | Writes XDMF and vertex-field data; can map data onto a different mesh | Uses coordinate/KDTree remapping; memory-heavy for large meshes and high MPI counts |
| `mesh.write_checkpoint(...)` | Restart and exact postprocessing | `MeshVariable.read_checkpoint(...)` | Uses PETSc DMPlex section/vector metadata; avoids KDTree | Not a visualisation output; no XDMF or vertex-field datasets |

The benchmark scripts should use `write_timestep()` when they need
visualisation files, and `write_checkpoint()` when they need restart-safe,
memory-efficient postprocessing.

## Implemented Design

### Checkpoint Writing

`Mesh.write_checkpoint(...)` writes PETSc DMPlex HDF5 storage version `3.0.0`.

The mesh file is named:

```text
<base>.mesh.<index>.h5
```

With the default `separate_variable_files=True`, each mesh variable is written
to its own checkpoint file:

```text
<base>.<variable>.<index>.h5
```

With `separate_variable_files=False`, all variables are written into:

```text
<base>.checkpoint.<index>.h5
```

Per-variable files are the default because they avoid forcing downstream
postprocessing to open and move through one very large combined checkpoint file.
This is useful for large spherical benchmark cases where velocity and pressure
files can already be large individually.

### Mesh Reload

When a PETSc DMPlex HDF5 mesh is loaded, UW3 keeps the topology-load `PetscSF`
returned by `DMPlexTopologyLoad(...)`. If UW3 distributes the mesh after load,
the topology-load SF is composed with the redistribution SF.

This composed SF is stored on the mesh and is the mapping used by
`MeshVariable.read_checkpoint(...)`.

The mesh DM name is fixed to `uw_mesh` while writing and loading checkpoint
files so PETSc can find the expected topology groups.

### Variable Reload

`MeshVariable.read_checkpoint(filename, data_name=None)`:

- opens the checkpoint HDF5 file in PETSc HDF5 format
- loads the saved DMPlex section for `data_name`
- loads the saved local vector through PETSc's DMPlex local-vector path
- copies values into the target UW3 variable using section offsets
- syncs the UW3 local vector back to its global vector

The implementation uses a small Cython wrapper around:

- `DMPlexSectionLoad(...)`
- `DMPlexLocalVectorLoad(...)`

The wrapper requests only the local-data SF. This avoids failures seen when
the global-data SF path was constructed before the local checkpoint data could
be loaded.

## PETSc Requirements

The relevant PETSc DMPlex HDF5 restart sequence is:

1. Load topology with `DMPlexTopologyLoad(...)`.
2. Keep the `PetscSF` returned by topology load.
3. Load coordinates with `DMPlexCoordinatesLoad(...)`.
4. Load labels with `DMPlexLabelsLoad(...)`.
5. If the mesh is redistributed, compose the topology-load SF with the
   redistribution SF.
6. Load the saved section with `DMPlexSectionLoad(...)`.
7. Load the saved vector with the DMPlex vector-load API.
8. Scatter or copy the loaded values into UW3's mesh-variable storage.

The critical identity requirements are:

- topology DM name must match the saved topology group
- section DM name must match the saved variable group
- vector name must match the saved vector group
- the SF passed to section load must map saved topology points to current
  distributed topology points

PETSc reference: [DMPlex manual](https://petsc.org/main/manual/dmplex/).

## Tests

Current unit coverage is in `tests/test_0003_save_load.py`.

The checkpoint roundtrip test covers:

- scalar variable reload
- vector variable reload
- discontinuous variable reload
- combined checkpoint file reload with `separate_variable_files=False`
- per-variable checkpoint file reload with the default
  `separate_variable_files=True`

Required validation before PR:

```bash
./uw python -m pytest tests/test_0003_save_load.py -q
mpirun -np 2 ./uw python -m pytest \
    tests/test_0003_save_load.py::test_meshvariable_checkpoint_roundtrip -q
```

## Spherical Benchmark Validation

The motivating case is spherical benchmark postprocessing at high MPI counts.
The old `write_timestep()` / `read_timestep()` path can build large KDTree
mapping structures during reload. At `1/128` this used nearly the full 4.5 TB
allocation on Gadi.

The checkpoint method avoids KDTree reload and preserves velocity/pressure
metrics to roundoff. Boundary stress metrics require the benchmark to recover
stress consistently after reload. In the spherical benchmark this is handled by
projecting the six deviatoric-stress components and then forming `sigma_rr`.

### Gadi Evidence

| Resolution | Method | NCPUs | Walltime | Memory used | Status |
| --- | --- | ---: | ---: | ---: | --- |
| `1/64` | `write_timestep/read_timestep` | 144 | `00:03:43` | `211.27 GB` | completed |
| `1/64` | `write_checkpoint/read_checkpoint` | 144 | `00:02:41` | `233.67 GB` | completed |
| `1/128` | `write_timestep/read_timestep` | 1152 | `00:13:55` | `3.92 TB` | completed near memory limit |
| `1/128` | `write_checkpoint/read_checkpoint` | 1152 | `00:03:57` | `1.83 TB` | completed |

The `1/128` checkpoint reload reduced memory by about `2.09 TB` and walltime by
about `3.5x` for the postprocessing run.

### Metric Agreement

`1/128` spherical Thieulot benchmark:

| Metric | `write_timestep/read_timestep` | `write_checkpoint/read_checkpoint` |
| --- | ---: | ---: |
| `v_l2_norm` | `1.4319274480265082e-06` | `1.4319274480231255e-06` |
| `p_l2_norm` | `5.985841567394967e-04` | `5.985841567395382e-04` |
| `p_l2_norm_abs` | `1.0566381005355924e-03` | `1.0566381005356654e-03` |
| `sigma_rr_l2_norm_lower` | `1.117914337768646e-03` | `1.1256362820288926e-03` |
| `sigma_rr_l2_norm_upper` | `4.461443231341268e-05` | `3.811141458727819e-05` |
| `u_dot_n_l2_norm_lower_abs` | `2.2509850571644799e-04` | `2.2509850571645164e-04` |
| `u_dot_n_l2_norm_upper_abs` | `5.535239716141496e-05` | `5.535239716141875e-05` |

Velocity, pressure, and normal-velocity metrics agree to roundoff. The
`sigma_rr` values are close but not bitwise identical because the stress
recovery path changed from the old reload workflow to explicit tau-component
projection after checkpoint reload.

`1/64` spherical Thieulot benchmark:

| Metric | `write_timestep/read_timestep` | `write_checkpoint/read_checkpoint` |
| --- | ---: | ---: |
| `v_l2_norm` | `1.1662200663950889e-05` | `1.1662200663957042e-05` |
| `p_l2_norm` | `2.7573367818459473e-03` | `2.7573367818460497e-03` |
| `sigma_rr_l2_norm_lower` | `4.368560398155481e-03` | `4.381908965541248e-03` |
| `sigma_rr_l2_norm_upper` | `1.6315543718450765e-04` | `1.6047310456195621e-04` |

## Remaining PR Readiness Items

- Run the unit checkpoint tests on the clean checkpoint-only branch.
- Add or record one small local benchmark smoke test for reproducibility.
- Decide whether different-rank checkpoint reload is required for the first PR
  or should be documented as follow-up validation.
- Keep the final checkpoint PR branch free of unrelated JIT and macOS compiler
  commits.

## Acceptance Criteria

The checkpoint reload implementation is ready for review when:

- `write_checkpoint()` writes PETSc DMPlex HDF5 storage version `3.0.0`.
- `write_checkpoint()` supports `outputPath`.
- `write_checkpoint()` defaults to one checkpoint file per variable.
- `write_checkpoint(..., separate_variable_files=False)` still supports a
  combined variable checkpoint file.
- `MeshVariable.read_checkpoint(...)` reloads through PETSc metadata, not
  coordinate/KDTree remapping.
- scalar, vector, and discontinuous variables roundtrip in tests.
- same-rank MPI reload is validated.
- benchmark evidence shows the large-memory KDTree reload issue is avoided.
