---
title: "Create XDMF from PETSc HDF5 Output"
---

# Create XDMF from PETSc HDF5 Output

This guide explains how to generate ParaView-ready XDMF/HDF5 outputs with `mesh.write_timestep(create_xdmf=True)`, why this is needed with newer PETSc HDF5 layouts, and how tensors are converted for correct ParaView interpretation.

This note explains the output-format issue seen with newer PETSc versions and the fix now available in `Mesh.write_timestep()`.

## 1. What new PETSc writes to HDF5

With newer PETSc HDF5 viewer behavior, variable files commonly contain only:

- `/fields/<VariableName>`
- `/fields/coordinates`

Instead of relying on a fixed file example, you can verify this directly:

```python
import h5py
import underworld3 as uw

mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4), minCoords=(0, 0), maxCoords=(1, 1))
v = uw.discretisation.MeshVariable("V", mesh, mesh.dim, degree=1, continuous=True)
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=0, continuous=False)

with uw.synchronised_array_update():
    v.array[:, 0, :] = 1.0
    p.array[:, 0, 0] = 2.0

# Case A: HDF5 only (no XDMF compatibility groups)
mesh.write_timestep(
    "check_a",
    index=0,
    outputPath=".",
    meshVars=[v, p],
    create_xdmf=False,
)

# Case B: create XDMF + compatibility groups
mesh.write_timestep(
    "check_b",
    index=0,
    outputPath=".",
    meshVars=[v, p],
    create_xdmf=True,
)

for fname in ["check_a.mesh.V.00000.h5", "check_b.mesh.V.00000.h5"]:
    print(f"\n{fname}")
    with h5py.File(fname, "r") as h5:
        def _show(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  {name}  {obj.shape}")
            else:
                print(f"  {name}/")
        h5.visititems(_show)
```

Older workflows also had explicit compatibility groups such as:

- `/vertex_fields/...`
- `/cell_fields/...`

which many existing XDMF templates expected.

## 2. Why `/fields` alone is not sufficient for robust XDMF

XDMF needs a clear **data center**:

- node-centered data must match mesh vertices (`Nvertices`)
- cell-centered data must match mesh elements (`Ncells`)

But `/fields` can store high-order DOF layouts that do not equal vertex count or cell count (for example packed element-point coordinates/values).  
If XDMF points `Center="Node"` data to such arrays, ParaView reads mismatched lengths and fails or misinterprets arrays.

So, `/fields` is useful raw data, but it is not always directly visualization-ready for a mesh topology/geometry pair.

## 3. What changed in `write_timestep()` and why

`Mesh.write_timestep()` now supports:

```python
mesh.write_timestep(..., create_xdmf=True)
```

When `create_xdmf=True`:

1. It writes compatibility datasets per variable:
   - node-like vars -> `/vertex_fields/coordinates` and `/vertex_fields/<name>_<name>`
   - cell-like vars -> `/cell_fields/<name>_<name>`
2. It writes XDMF that references these compatibility datasets.

This gives XDMF arrays that match mesh vertex/cell counts, which ParaView expects.

Implementation notes:

- continuous, non-degree-0 variables are treated as node-like
- discontinuous or degree-0 variables are treated as cell-like
- high-order `/fields` layouts are unpacked/remapped to vertex/cell-compatible arrays

### How vertex vs cell field is chosen

For each mesh variable during `write_timestep(..., create_xdmf=True)`:

- **Cell field** if:
  - variable is discontinuous (`continuous == False`), or
  - variable polynomial degree is 0 (`degree == 0`)
- **Vertex field** otherwise.

In code terms, this is equivalent to:

```python
is_cell = (not var.continuous) or (var.degree == 0)
```

Then:

- `is_cell == True`  -> write `/cell_fields/<name>_<name>` and XDMF `Center="Cell"`
- `is_cell == False` -> write `/vertex_fields/<name>_<name>` and XDMF `Center="Node"`

### How `/fields` data is remapped (KDTree)

When a variable is vertex-centered but `/fields` does not already match mesh vertex count:

1. Source coordinates/values are read from `/fields/coordinates` and `/fields/<name>`.
2. Packed high-order layouts are unpacked into point-wise coordinate/value rows.
3. A nearest-neighbor search is done with `uw.kdtree.KDTree` to map source points to mesh vertices.
4. The mapped values are written to `/vertex_fields/<name>_<name>`.

For cell-centered variables, values are written into `/cell_fields/<name>_<name>` using row slices.

### Parallel execution details

This remapping/writing path is parallel-aware:

- work is partitioned by MPI rank (each rank owns a slice of vertices / rows)
- each rank computes mapping for its local slice
- output datasets are written by per-rank slices (no overlapping writes)
- with MPI-enabled `h5py`, HDF5 writes use parallel I/O; otherwise a serial fallback is used

So the compatibility-field generation is computed and written in parallel when MPI-HDF5 is available.

## 4. Tensor representation for ParaView (2D and 3D)

ParaView expects `Attribute Type="Tensor"` data as 9 components per point/cell (flattened `3x3`).

ParaView expects tensors in a `3x3` format, even for 2D problems.

### The key idea

A 2D tensor

```text
[ s_xx  s_xy ]
[ s_yx  s_yy ]
```

must be embedded into a 3D tensor

```text
[ s_xx  s_xy   0 ]
[ s_yx  s_yy   0 ]
[   0     0    0 ]
```

ParaView works internally with 3D tensors, so writing only 2D tensor components is not enough for full tensor-aware filters.

### Expected tensor format in XDMF / HDF5

- XDMF attribute: `Type="Tensor"`
- dataset shape: `(N, 9)`
- component order (row-major):
  - `[Txx, Txy, Txz, Tyx, Tyy, Tyz, Tzx, Tzy, Tzz]`

### 2D tensors

For 2D tensors, values are embedded into 3D by zero-filling Z terms.

Example (full 2D tensor with 4 comps):

- input components interpreted as in-plane terms
- output `9` components with `Txz, Tyz, Tzx, Tzy, Tzz = 0`

Example (2D symmetric with 3 comps):

- `Txy == Tyx`
- output still written as `9` components.

### 3D tensors

- full tensors: `(N, 9)` passed through
- symmetric tensors with 6 comps are expanded into `(N, 9)` with mirrored off-diagonals

### What we changed in Underworld3

When `mesh.write_timestep(..., create_xdmf=True)` is used:

1. Node/cell compatibility arrays are written to `/vertex_fields` and `/cell_fields`.
2. Tensor-like mesh variables are converted to ParaView-ready `9` components before writing:
   - 2D full tensor (`4` comps) -> embedded `3x3` (`9` comps)
   - 2D symmetric (`3` comps) -> embedded `3x3` (`9` comps, mirrored shear terms)
   - 3D full (`9` comps) -> unchanged
   - 3D symmetric (`6` comps) -> expanded to full `3x3` (`9` comps)
3. XDMF attributes for these variables are emitted as `Type="Tensor"` and reference the `9`-component datasets.

This is why tensor variables now appear in HDF5/XDMF as 9-component arrays for ParaView compatibility.

## 5. Post-processing without XDMF or extra HDF5 groups

If you want to avoid writing `/vertex_fields` and `/cell_fields` (to reduce file size and write overhead), you can:

- save only raw `/fields` data (`create_xdmf=False`)
- do mapping in Python during post-processing
- attach mapped arrays directly to a `pyvista` mesh in memory

This avoids storing additional redundant datasets in HDF5.

### 5.1 Write raw HDF5 only

```python
mesh.write_timestep(
    "output",
    index=0,
    outputPath=".",
    meshVars=[v, p, stress],
    create_xdmf=False,  # no xdmf, no /vertex_fields or /cell_fields
)
```

### 5.2 Load mesh + raw field data and map in PyVista

```python
import h5py
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree


def _flatten_fields_layout(field_values, field_coords, dim):
    """Handle packed high-order field layout into point-wise rows."""
    if field_values.ndim == 1:
        field_values = field_values.reshape(-1, 1)

    if field_coords.shape[1] == dim:
        return field_values, field_coords

    if field_coords.shape[1] % dim != 0:
        raise RuntimeError(f"Cannot unpack coords shape {field_coords.shape} for dim={dim}")

    dof_per_row = field_coords.shape[1] // dim
    coords = field_coords.reshape(-1, dim)

    if field_values.shape[1] == dof_per_row:
        values = field_values.reshape(-1, 1)
    elif field_values.shape[1] % dof_per_row == 0:
        ncomp = field_values.shape[1] // dof_per_row
        values = field_values.reshape(field_values.shape[0], dof_per_row, ncomp).reshape(-1, ncomp)
    else:
        raise RuntimeError(
            f"Cannot unpack values shape {field_values.shape} with dof_per_row={dof_per_row}"
        )

    return values, coords


def load_h5_field_to_pvmesh(pvmesh, mesh_h5, var_h5, field_name, location="auto"):
    """
    Attach raw /fields data to pvmesh as point_data or cell_data.
    location: 'point', 'cell', or 'auto'
    """
    with h5py.File(mesh_h5, "r") as mh:
        verts = mh["geometry/vertices"][()]
    nverts = verts.shape[0]
    ncells = pvmesh.n_cells
    dim = verts.shape[1]

    with h5py.File(var_h5, "r") as vh:
        values = vh["fields"][field_name][()]
        coords = vh["fields"]["coordinates"][()]

    if values.ndim == 1:
        values = values.reshape(-1, 1)

    # Direct matches: already node or cell sized
    if location in ("auto", "point") and values.shape[0] == nverts:
        pvmesh.point_data[field_name] = values
        return "point"
    if location in ("auto", "cell") and values.shape[0] == ncells:
        pvmesh.cell_data[field_name] = values
        return "cell"

    # Packed/high-order case: map by nearest coordinate
    unpacked_values, unpacked_coords = _flatten_fields_layout(values, coords, dim)
    tree = cKDTree(unpacked_coords)
    _, idx = tree.query(verts, k=1)
    pvmesh.point_data[field_name] = unpacked_values[idx, :]
    return "point"
```

### 5.3 Example usage

```python
mesh_h5 = "output.mesh.00000.h5"
vel_h5 = "output.mesh.Velocity.00000.h5"
prs_h5 = "output.mesh.Pressure.00000.h5"

# Build pvmesh from the mesh file (reader choice depends on your workflow)
pvmesh = pv.read(mesh_h5)

load_h5_field_to_pvmesh(pvmesh, mesh_h5, vel_h5, "Velocity", location="auto")
load_h5_field_to_pvmesh(pvmesh, mesh_h5, prs_h5, "Pressure", location="auto")

# Continue in-memory analysis/plotting with pyvista
plotter = pv.Plotter()
plotter.add_mesh(pvmesh, scalars="Pressure")
plotter.show()
```

Notes:

- This workflow is ideal for Python-based analysis.
- For direct ParaView file-open workflows, `create_xdmf=True` remains the simpler option.
