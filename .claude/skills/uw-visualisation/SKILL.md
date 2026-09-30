---
name: uw-visualisation
description: Render Underworld3 mesh fields (T, V, viscosity, the adapted mesh) correctly with PyVista. Use whenever you need to SEE a UW3 result — a field colormap, the moving/adapted mesh, streamlines, or compare runs. Reach for THIS before hand-rolling a renderer; getting the four cosmetic settings wrong makes renders look grey/patchy/blocky and wastes a round-trip with Louis.
---

# uw-visualisation

Canonical PyVista recipe for Underworld3 fields. This exists because every fresh
Claude session re-derives the renderer and gets the colormap / background /
lighting / DOF-sampling wrong, producing "grey/patchy/weird" images Louis
rejects. The settings below match his reference renders exactly.

**Use PyVista (`underworld3.visualisation`), NOT matplotlib.** Louis reaffirmed
this even after seeing the legacy matplotlib renderer
(`scripts/fault_convection_frames.py`) — that one is NOT preferred.

## Hard rules (artifacts + output location)

- **Outputs go under `~/+Simulations/...`, NEVER `/tmp`** (Louis can't view /tmp
  or harness task paths). Mirror the run's `--sim-dir`; write `T_<step>.png` into
  the run directory, comparison figures into the sim-dir root.
- `pv.OFF_SCREEN = True` at import; finish with `pl.screenshot(path); pl.close()`.

## The field+mesh pattern (copy this exactly)

```python
import numpy as np, underworld3 as uw, underworld3.visualisation as vis, pyvista as pv
pv.OFF_SCREEN = True

mesh = uw.discretisation.Mesh(f"{label}.mesh.00000.h5")   # or the live mesh
T = uw.discretisation.MeshVariable("T_v2p1", mesh, 1, degree=3, continuous=True)
T.read_timestep(label, "T_v2p1", 0, outputPath=D)          # or use the live var

pv_T = vis.meshVariable_to_pv_mesh_object(T)    # Delaunay through T's OWN DOFs
pv_T.point_data["T"] = np.asarray(T.data[:, 0]) # attach DOF values DIRECTLY (P3-faithful)
edges = vis.mesh_to_pv_mesh(mesh).extract_all_edges()

pl = pv.Plotter(off_screen=True, window_size=(1000, 1000))
pl.set_background("white")                       # rule 2
pl.add_mesh(pv_T, scalars="T", cmap="RdBu_r", clim=(0, 1),   # rules 1 + clim required
            show_edges=False, lighting=False)   # rule 3
pl.add_mesh(edges, color="black", line_width=0.5, lighting=False)  # mesh overlay
pl.view_xy(); pl.camera.zoom(1.3)
pl.screenshot(out); pl.close()
```

## The four things that make renders look bad (all COSMETIC)

1. `cmap="coolwarm"` → muddy grey-lavender midtone — this IS the "blue/grey/red"
   Louis rejects. **Use `cmap="RdBu_r"`** (clean blue→white→red).
2. PyVista's default grey background bleeds through RdBu_r's white (T≈0.5) → dirty
   grey. **Always `pl.set_background("white")`.**
3. Default lighting darkens the colormap. **Always `lighting=False`** on every
   `add_mesh`.
4. Re-evaluating via `scalar_fn_to_pv_points` / vertex-only sampling drops the
   high-order DOFs → blocky. **Attach `T.data[:,0]` directly** to the DOF-cloud
   mesh from `meshVariable_to_pv_mesh_object` (it is correct for annulus/box/disc
   — do NOT avoid it). `clim` MUST be passed (default `clim=""` trips `np.any`).
5. Resampling ANY field (even P1) onto a regular pixel grid via
   `uw.function.evaluate` **dapples at element boundaries** — grid points that
   straddle a facet get located into a neighbouring cell with slightly-off
   reference coords (Louis: "artefacts across the elements", S-fault rig).
   Render derived fields NODALLY on the mesh's own triangulation instead:
   evaluate at `mesh_to_pv_mesh(mesh).points` (exact at vertices for P1,
   whichever cell the locator picks), attach as point_data, let VTK
   interpolate WITHIN elements. On a SPLIT mesh never Delaunay the DOF cloud
   (it re-triangulates across the slit) — use the mesh's own cells.

## Seeing the MESH (adaptation / moving mesh)

The full-annulus T colormap **washes out mesh detail** — at whole-domain zoom the
grading is invisible. To judge adaptation you MUST crop:

- Zoom the feature region with a parallel camera:
  `pl.camera.parallel_projection = True; pl.camera.parallel_scale = half_width;
   pl.camera.focal_point = (cx, cy, 0)`.
- For mesh-only views, drop the field and draw `edges` on white, `line_width≈0.7`.
- Real corruption vs render artifact: apparent "holes / lumps" are often a
  mesh-overlay/low-res artifact. Before calling adaptation broken, CHECK the
  field's value range is bounded and count folded elements (negative cell area)
  programmatically — do NOT diagnose from a render alone.
- **Overlay the feature you're refining to** (a fault trace, an interface): draw it
  as a red `pv.PolyData` line over the mesh. Without it you cannot tell whether the
  refinement sits ON the feature or has drifted off it (a real failure mode — see
  the `adaptive-meshing` skill). Read the geometry from the run manifest so any run
  renders the same way.

## Adaptive / long runs

- **Render each checkpoint as it lands**, not just the last frame: arm a Monitor that
  polls for new `run.mesh.NNNNN.{xdmf,h5}` and emits the index → render on each
  event. A completion-only watch leaves you blind for a multi-hour (e.g. TI) run.
- The per-step mesh GEOMETRY must have been written (`write_timestep(...,
  meshUpdates=True)`) or you'll render deformed fields on the stale step-0 mesh.
  Load the per-step `run.mesh.NNNNN.h5` as the mesh, then `read_timestep` the vars.

## Velocity

Same pattern; use **streamlines, not glyphs**. Build a pv mesh for V, add
`pv_mesh.streamlines(...)` or evaluate V on a line seed. Magnitude with the same
white-bg / lighting=False rules.

## Quantities to judge a convection run (not just pretty pictures)

- `vrms` from `uw.function.evaluate(V.sym.dot(V.sym), mesh.X.coords)` → the clean
  kinetic-energy indicator (more reliable than nodal boundary metrics).
- Surface heat flux Nu via `uw.maths.BdIntegral` on the Upper boundary.
- Mesh quality: fault/bulk nearest-neighbour spacing RATIO (cKDTree) for refinement;
  folded-element count + min cell area for tangling.

## Templates in this skill

- `render_field.py` — single/`--all`-steps T+mesh render of a run directory.
- `render_field_streamlines.py` — T colormap + mesh + **V streamlines** (sparse
  seeds, thin lines, short integration so weak/closed cells read clearly, not
  black spiral-blobs). Use for convection. `--tag <run> --all`.
- `zoom_compare.py` — side-by-side cropped mesh+field for N runs at one step.

Copy these into the run's `scripts/` (or run in place), point `--sim-dir` at the
run, and adjust the field/variable names. They already encode every rule above.

## Related memory

`feedback_use_uw_pyvista_visualisation.md`, `feedback_pyvista_viz_pattern.md`,
`feedback_render_all_steps.md`, `project_adaptation_corruption_was_render_artifact.md`.
