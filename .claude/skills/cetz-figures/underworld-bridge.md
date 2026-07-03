# Bridging cetz figures to underworld3 mesh objects (sketch)

This is a **sketch, not an implementation**. When we wire figures up to
real `underworld3` data, this is the contract the drawing code expects.

## The JSON schema

Any cetz figure that draws a mesh reads the same shape:

```json
{
  "vertices":   [[x, y], ...],     // N x 2 floats
  "triangles":  [[i, j, k], ...],  // M x 3 int indices into vertices
  "highlight":  17                  // optional: triangle of interest
}
```

Keep it minimal — cetz script should not have to do geometry or case
analysis. Anything derived (centroids, boundary detection, highlight
selection) is computed upstream and baked into the JSON.

## How this maps to underworld3

Provisional — needs verification against the current `underworld3` mesh
API once we start using it:

| JSON field | `underworld3` source | Notes |
|---|---|---|
| `vertices` | `mesh.data` (or `mesh.X.coords` — see repo style guide) | 2D only for this skill; 3D needs a projection |
| `triangles` | PETSc `DMPlex` cell-vertex connectivity | Simplex meshes only; quads/hexes need different rendering |
| `highlight` | Caller-provided or heuristic (e.g. closest-to-origin) | Chosen offline, not in Typst |

Repo's data-access pattern guide:
`docs/developer/UW3_Style_and_Patterns_Guide.md`.

## Python exporter sketch

```python
# Not the current implementation — a target shape
def mesh_to_figure_json(mesh, *, highlight=None, path):
    """Dump a uw.meshing object to the schema the cetz figures expect."""
    import json
    # 2D coordinates
    coords = mesh.X.coords              # (N, 2) numpy, already preferred over mesh.data
    cells  = mesh.cell_vertex_indices   # placeholder — actual accessor TBD
    data = {
        "vertices":  [[round(float(x), 4), round(float(y), 4)] for x, y in coords],
        "triangles": [[int(i) for i in cell] for cell in cells],
    }
    if highlight is not None:
        data["highlight"] = int(highlight)
    with open(path, "w") as f:
        json.dump(data, f)
```

## What this intentionally does NOT do

- **No coordinate transform / projection.** Caller passes whatever world
  coordinates it wants; Typst draws them as-is. Transforming a spherical
  mesh to a 2D projection is a Python concern.
- **No per-cell data.** This schema is for topology/geometry only. For
  scalar fields, vector fields, colormaps → go the Python-generated-SVG
  route and include via `#image()`.
- **No parallel-safe collection logic.** Caller is expected to gather the
  mesh to rank 0 before exporting. Don't invent a parallel JSON writer.

## Next steps (when we get here)

1. Implement `mesh_to_figure_json` using the correct underworld3 accessors.
   Verify which is idiomatic in the current codebase — `mesh.data` is
   deprecated per `CLAUDE.md`.
2. Write a test figure consuming output from a small real mesh (e.g. one
   of the existing example meshes).
3. Decide whether to ship the exporter as a `uw.utilities.export_figure`
   helper or leave it as a per-paper script.
