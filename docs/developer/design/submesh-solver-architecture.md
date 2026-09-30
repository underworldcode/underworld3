# Submesh Solver Architecture: Multi-Domain Equation Systems

**Status**: Prototype (2026-04 – 2026-05). The subdomain (`extract_region`) and surface (`extract_surface`) flavours exist in `discretisation_mesh.py`; the resolution-level `coarsened_companion` flavour is not implemented.

## Context

Underworld3 needs to support solving different equations on different subsets of a mesh while maintaining a unified field representation. Use cases include:

- **Air/rock**: Stokes on rock only, full mesh for temperature/gravity
- **Surface evolution**: deforming air mesh coupled to rock Stokes
- **Gravity**: Poisson on full domain, density source from rock only
- **Multi-physics**: different equations on different subdomains (Stokes, Darcy, etc.)

### What we've established (2026-04-05)

1. **`DMPlexFilter`** extracts a submesh with exact shared nodes. The submesh carries a subpoint IS mapping back to the parent via `getSubpointIS()`.

2. **PETSc Region DS** (`DMSetRegionDS`) segfaults during assembly — no examples exist in PETSc, likely incomplete infrastructure. Dead end for now.

3. **Solver `part` parameter** in PETSc boundary assembly (`support[key.part]`) — controls which cell's closure is used for internal boundary integrals. Useful for one-sided boundary assembly but doesn't address the core problem of restricting volume assembly to a subdomain.

4. **Low-viscosity air layer** with discontinuous pressure works reasonably but the air's incompressibility constraint acts as an unintended physical boundary condition. Not equivalent to solving on rock alone.

5. **Normalised `Gamma_N`** (merged) — `mesh.Gamma_N` now returns a unit normal. Penalty and Nitsche BCs are mesh-independent.

## The parent/child model: submeshes and adapted meshes are both *children*

```{note}
**Unified terminology (parent/child DAG).** Every mesh derived from another
carries a `parent` link and a `_relationship_kind`, forming a directed
derivation graph. A *child* comes in two **kinds**:

- **submesh** (`_relationship_kind == "submesh"`) — a *subset* of the parent
  produced by `extract_region` / `extract_surface`. DOFs **coincide** with the
  parent (`subpoint_is`); transfer is exact injection.
- **refinement** (`_relationship_kind == "refinement"`) — a *finer* mesh
  produced by `mesh.adapt(metric, …)` (SBR adapt-on-top). The child is
  **bigger** than the parent; DOFs do **not** coincide, so transfer is FE-exact
  custom-P prolongation (parent→child) and injection restriction (child→parent).
  See [`LAYER2_SBR_ADAPT_ON_TOP.md`](LAYER2_SBR_ADAPT_ON_TOP.md).

`copy_into` / `add_into` dispatch on the kind, so the same call works for both:
`parent_var.copy_into(child_var)` and `child_var.copy_into(parent_var)`.
The "submesh flavours" below are all the **submesh kind** of child.
```

A *submesh* in UW3 is any mesh pulled out of a parent mesh that retains a
lineage link (`parent`, registration in `parent._registered_submeshes`)
and supports explicit field transfer back and forth. There are three
flavours, and they share one usage pattern:

> **get a child → build a solver on it → map fields back and forth**

| | Subdomain | Resolution level | Surface |
|---|---|---|---|
| Constructor | `mesh.extract_region("Inner")` | `coarsened_companion(fine, levels=1)` | `extract_surface(mesh, "Upper")` |
| PETSc mechanism | `DMPlexFilter` (cell subset) | `dm.refine()` nested hierarchy | `DMPlexCreateSubmesh` then `DMPlexFilter(depth, 2)` (strip phantom DAG) |
| dim, cdim | `parent.dim`, `parent.cdim` | `parent.dim`, `parent.cdim` | `parent.dim − 1`, `parent.cdim` |
| Parent ↔ child map | `getSubpointIS()` (point-level) | nested `createInjection` / `createInterpolation` (DOF-level) | `getSubpointIS()` (point-level) |
| Transfer fidelity | exact (shared nodes) | exact (nested FE) | exact (shared nodes) |
| Example | `docs/examples/submesh_investigation/test_region_ds_submesh.py` | `docs/examples/submesh_investigation/example_refined_companion.py` | `docs/examples/submesh_investigation/example_surface_extraction.py` |

The subdomain and resolution-level examples solve the **same** annulus +
radial-buoyancy Stokes problem so those two flavours are directly
comparable: one solves on a *subdomain* of the annulus, the other on a
*coarser resolution* of the whole annulus, and both map the solution
back to the parent.

The surface flavour produces a 2-manifold embedded in 3-space
(`dim=2, cdim=3`) and is the natural carrier for *lateral surface
processes*: surface diffusion, surface advection, surface evolution.
Phase 1 (the lineage layer: extract, restrict/prolongate round-trip,
registration with the parent, navigation, `evaluate(expr, surface_pts)`)
lands as the investigation under
`docs/examples/submesh_investigation/surface_submesh_prototype.py`.
Phase 2 — making a solver actually run on the manifold (Laplace–
Beltrami) — is tracked in the *Surface submesh: solver path* section
below.

### Design contract for the resolution-level flavour: refine-DM mode only

A resolution level is extractable **only when a genuine nested
refinement hierarchy exists** (the mesh was built with
`refinement >= 1`). The hierarchy is the source of truth; any level can
be pulled out as a standalone solver-ready `uw.Mesh`, exactly as a
subdomain is pulled out with `extract_region`.

**If there is no refinement relationship the operation is unavailable.**
There is deliberately:

- **no geometric / `DMPlexComputeInterpolatorGeneral` fallback**, and
- **no KDTree coordinate-matching fallback**.

`coarsened_companion` raises a clear error on a non-refined mesh rather
than silently degrading to an approximate transfer.

Transfer between levels uses PETSc's *nested* interpolator/injector
(`plex.c:10328`, `DMPlexComputeInterpolatorNested` /
`DMPlexComputeInjectorFEM`), which is:

- **exact** — the prolongation is the FE embedding; injection is a pure
  scatter of coincident DOFs;
- **parallel-local** — the injector builds per-rank `COMM_SELF` scatters
  (`plexfem.c:3739-3741`); `DMRefine` preserves the coarse partition, so
  no MPI communication is needed for restrict/prolongate;
- **not dependent on point location** — no grid hashing, no geometry
  search.

```{note}
The nested path triggers only when the fine DM's `getCoarseDM()` *is*
the coarse DM and the regular-refinement flag is set. Independently
cloning two hierarchy levels breaks this and petsc4py exposes no setter
to restore it. The working construction is to build the transfer pair by
**refining a single-field clone of the coarse level**
(`dm_f = dm_c.refine()`): `refine()` itself establishes the linkage and
the flag. See `refined_pair_prototype.py` (`_linked_pair`).
```

Empirically (box and annulus, P2 velocity), prolongating a coarse Stokes
solution to the fine mesh and sampling it back recovers the coarse
solution to **O(1e-15)** — machine precision — with the geometric escape
hatch deliberately removed, proving the transfer is genuinely the nested
operator.

```python
fine   = uw.meshing.Annulus(radiusOuter=1.5, radiusInner=0.5,
                            cellSize=1/16, refinement=2)
coarse = coarsened_companion(fine, levels=1)   # parent = fine

v_c = uw.discretisation.MeshVariable("V", coarse, coarse.dim, degree=2)
stokes = Stokes(coarse, velocityField=v_c, ...)   # solve cheaply, coarse
stokes.solve()

v_f = uw.discretisation.MeshVariable("Vf", fine, fine.dim, degree=2)
prolongate(coarse, v_c, v_f)                      # exact, fills all fine DOFs
```

Status: prototype + both parallel examples + gating tests
(`test_refined_pair_solver.py`) + contract test
(`test_refined_pair_contract.py`) all passing. The investigation under
`docs/examples/submesh_investigation/` is the sign-off artifact;
promotion of `coarsened_companion` into the `Mesh` API proper is a
follow-up.

### Design contract for the surface flavour: two-stage strip, no fallback

A surface submesh is extractable **only when the parent carries a
non-empty boundary-face stratum** for the named label. Mirrors the
loud-failure stance of the other two flavours: `extract_surface` raises
on an unknown label, on a label whose value isn't in the parent's live
value set, or on an empty stratum. There is deliberately no geometric
fallback that might silently produce a degenerate mesh.

The PETSc construction is **two-stage** — both primitives are already
wrapped in UW3's cython layer:

```
sub1 = petsc_dm_create_submesh_from_label(parent.dm, label, value, marked_faces=True)
sub2 = petsc_dm_filter_by_label(sub1, "depth", 2)   # for parent.dim == 3
subpoint_is = compose(sub2.getSubpointIS(), sub1.getSubpointIS())
```

Stage 1 (`DMPlexCreateSubmesh`) gives a cd-1 DM with the right cells,
edges and vertices, but PETSc additionally retains an **upward-DAG
phantom stratum**: depth-3 (= parent.dim) points, one per parent
volume cell, celltype 12, included inside each surface cell's downward
closure tuple. This linkage is there so the resulting DM can still be
navigated as "a slice of the parent" (assemble surface integrals from
parent volume cells). For a standalone surface mesh we don't use that
linkage, and the phantom points are actively harmful — every consumer
that does `closure[-n:]` slicing to grab vertices (the kd-tree
builder, centroid pickers, control-point face markers — three copies
of the same idiom in `discretisation_mesh.py` alone) gets sometimes
the right answer and sometimes a phantom point inside the slice.

Stage 2 (`DMPlexFilter(depth, 2)`) keeps only the cells of `sub1` and
their downward closure (edges, vertices). The result is a **genuine
standalone 2-manifold mesh** with a clean 3-stratum chart
(`depth = 0, 1, 2`; celltypes `[0, 1, 3]`). Cell closures are length 7
(1 cell + 3 edges + 3 vertices) and `closure[-3:]` reliably returns
vertices. The standard `Mesh._build_kd_tree_index` and downstream
navigation code run on this DM without any cd-1 special-casing.

The composed subpoint IS gives a direct surface → parent point map
(point IDs in the parent's chart). The properties carry through:

- `surf.dim = parent.dim − 1`, `surf.cdim = parent.cdim` (2 and 3 for
  the sphere case);
- vertex coordinates land on the parent surface to machine precision
  (we measured 2.2e-16 on `SphericalShell.Upper`);
- parent boundary labels propagate onto the submesh: the *selecting*
  label (e.g. `"Upper"`) survives carrying the surface itself;
  sibling boundary labels (`"Lower"`) survive as empty strata which
  the submesh constructor filters out.

**One PETSc footgun, worth knowing about.** Calling
`label.getStratumIS(value)` on a `DMPlexCreateSubmesh` DM for a value
that isn't in `getValueIS()` hard-aborts PETSc (no Python-catchable
exception). The prototype enumerates `surface_dm.getNumLabels()` by
index and checks each label's live value set first.

**Navigation on the manifold.** For an on-surface query point (the
contract assumption — query points come *from* the manifold), the
centroid kd-tree ranks faces by chord distance, which agrees with
the owning-face ranking to first order on any convex manifold (sphere,
ellipsoid). `get_closest_cells(surface_dof_coords)` returns valid
surface cell IDs and `evaluate(expr, surface_pts)` works to machine
precision. The cell-containment-by-half-space-rule helpers
(`_test_if_points_in_cells_internal`, `_mark_faces_inside_and_out`)
use a 2-D perpendicular construction that doesn't make sense for a
curved cell, so the cd-1 mesh path skips that rejection step and
trusts the centroid kd-tree result. A proper manifold in-cell test
(project the query into the cell's tangent plane, then barycentric)
is Phase 2 work — only needed if we ever support off-surface queries.

```python
shell   = uw.meshing.SphericalShell(radiusOuter=1.0, radiusInner=0.5,
                                    cellSize=0.2)
surface = extract_surface(shell, "Upper")           # parent = shell

# Round-trip a parent scalar via shared-vertex KDTree (1e-10 match)
T_p = uw.discretisation.MeshVariable("T_p", shell,   1, degree=1)
T_s = uw.discretisation.MeshVariable("T_s", surface, 1, degree=1)
surface.restrict(T_p, T_s)                          # parent -> surface, bit-exact
# … work on the surface …
surface.prolongate(T_s, T_p)                        # surface -> parent, bit-exact at surface DOFs

# Symbolic expressions can be evaluated at surface points directly
vals = uw.function.evaluate(T_s.sym[0], surface.X.coords)   # to machine precision
```

Status (Phase 1): prototype + documented example +
contract test
(`docs/examples/submesh_investigation/{surface_submesh_prototype.py,
example_surface_extraction.py, test_surface_submesh_contract.py}`) all
passing.

### Surface submesh: solver path (Phase 2)

Phase 1 produces the *mesh*. Phase 2 makes a solver run on it. UW3's
solver stack was built when `dim == cdim` was implicit; a surface
submesh — `dim = parent.dim − 1`, `cdim = parent.cdim` — exercises
that path from a new angle.

Concrete deliverable: lateral surface diffusion (Laplace–Beltrami,
`∫_M ∇_M T · ∇_M w dA`) on the upper surface of a `SphericalShell`,
verified against the analytic spherical-harmonic decay
`exp(-l(l+1) κ t / R²)`. Either a `SurfaceDiffusion`/`LaplaceBeltrami`
sibling of `Poisson`, or a flag on `Poisson` — decide after the JIT
audit. The Phase 1 example
(`example_surface_extraction.py`) is the natural site to add this once
the solver path is operational.

The investigation needs to clear (at least) these knowns:

1. **JIT pointwise functions on `dim != cdim`.** `petsc_x[i]` indexing
   should run to `cdim`; gradient-of-basis indexing to `dim`
   (tangent-space). Audit `utilities/_jitextension.py` for any
   `range(self.dim)` that should be `range(self.cdim)` (or vice
   versa). Phase 1 status: navigation and `evaluate()` work via the
   standard volume-mesh code path on the stripped chart — no JIT
   change was needed for that. Assembly is the open question.
2. **FE assembly with non-square Jacobian.** PETSc DMPlex computes
   element Jacobians from the embedded coordinates and produces the
   correct surface metric automatically *when the FE machinery is
   exercised with* `dim != cdim`. Confirm this on a trivial bilinear
   form before assuming it.
3. **Coordinate-system symbols.** `mesh.X` on the surface submesh is a
   3-vector; `mesh.dim` is 2. Code that iterates `range(mesh.dim)`
   over `mesh.X` components silently misses the third — exactly the
   kind of cdim/dim conflation the audit needs to catch.
4. **Manifold in-cell test.** `_test_if_points_in_cells_internal` /
   `_mark_faces_inside_and_out` use a 2-D perpendicular construction
   to place face-relative control points and run the half-space test.
   The cd-1 path currently skips that step (centroid kd-tree only).
   A proper version projects the query into the cell's tangent plane
   then runs barycentric. Only needed if we ever support off-surface
   queries; on-surface queries already work via the centroid path.
5. **Surface outward normal.** For intrinsic surface diffusion the
   bilinear form is metric-only, no explicit normal needed. For
   coupled problems (surface advection driven by a 3D flow), we need
   the surface's outward unit normal in 3-space — distinct from
   `mesh.Gamma_N` which is the normal to the *boundary* of the mesh
   (a closed sphere has none). Possibly a new symbol
   (`mesh.surface_normal`); deferred until needed.
6. **Boundary conditions on a closed surface.** `SphericalShell.Upper`
   has no boundary curve, so no Dirichlet/Neumann is needed for the
   first pass. A partial-surface submesh would add another dim/cdim
   layer (BCs on a 1D curve in 3D); out of scope for the first pass.

## Design Principles

### 1. Separate meshes, separate variables, explicit copies

Each mesh has its own MeshVariables. The user decides when data moves between meshes. There are no hidden globals or auto-managed shared fields.

```python
# Each mesh owns its own variables
v_rock = MeshVariable("v", rock_mesh, ...)
v_full = MeshVariable("v", full_mesh, ...)

# Solver works on submesh variables directly
stokes = Stokes(rock_mesh, velocityField=v_rock, ...)
stokes.solve()

# Explicit copy to full mesh when needed (e.g., for visualisation or coupling)
rock_mesh.prolongate(v_rock, v_full)
```

### 2. Meshes know their lineage

Every mesh has a `parent` attribute and a `subpoint_is` mapping. Top-level meshes have `parent=None` and `subpoint_is=None`. Submeshes reference their parent and carry the IS.

```python
full_mesh.parent       # None
full_mesh.subpoint_is  # None

rock_mesh = full_mesh.extract_region("Inner")
rock_mesh.parent       # full_mesh
rock_mesh.subpoint_is  # IS mapping submesh points -> parent points
```

### 3. Restrict/prolongate as mesh operations

```python
mesh.restrict(var)    # parent -> submesh DOFs (no-op if parent is None)
mesh.prolongate(var)  # submesh DOFs -> parent (no-op if parent is None)
```

Solvers call these uniformly. On a top-level mesh they're no-ops. On a submesh they gather/scatter via the subpoint IS. The solver code doesn't branch.

### 4. One mesh per expression

An expression passed to a solver must only contain MeshVariable symbols from that solver's mesh. The JIT compiler evaluates all symbols against one DM's auxiliary vector and one coordinate system — mixing meshes is undefined.

The user must restrict cross-mesh data before building expressions:

```python
# T lives on full_mesh, but Stokes is on rock_mesh
rock_mesh.restrict(T_full, T_rock)

# Expression uses only rock_mesh variables — safe
stokes.bodyforce = rho_rock.sym * alpha * T_rock.sym * gravity
```

If meshes are mixed in an expression, detect it (check `var.mesh` for all MeshVariable atoms) and raise an error at solver setup.

### 5. Boundary mapping is automatic

When `extract_region("Inner")` creates a submesh, boundaries are remapped:
- Full mesh "Lower" (r=r_inner) → submesh "Lower"
- Full mesh "Internal" (r=r_internal) → submesh outer boundary
- Full mesh "Upper" (r=r_outer) → not present on submesh

The label names are preserved from the parent (they survive `DMPlexFilter`). The user refers to boundaries by the same names.

## PETSc Infrastructure Available

| API | What it does | Status |
|-----|-------------|--------|
| `DMPlexFilter(dm, label, value, ...)` | Extract cells by label → new DMPlex | **Works**, tested |
| `DMPlex.getSubpointIS()` | IS mapping submesh → parent points | Available in petsc4py |
| `DMSetRegionDS(dm, label, fields, ds, dsIn)` | Per-region discrete system | **Segfaults**, no examples |
| `DMGetCellDS(dm, point, &ds, &dsIn)` | Per-cell DS dispatch in assembly | Works but requires Region DS |
| `DMPlexCreateSubmesh(dm, label, value, ...)` | Co-dimension 1 submesh (boundaries) | **Works**, used by `extract_surface` (dim=parent.dim−1, cdim=parent.cdim) |
| `VecScatter` / `PetscSF` | Parallel data transfer | Standard PETSc |

### PETSc Alternatives Investigated (2026-04-05)

**DMComposite** — packs multiple DMs into one composite. Tested 2026-04-05.

- Accepts DMPlex sub-DMs from DMPlexFilter. Scatter/gather works correctly.
- Interface nodes appear in both sub-DMs (102 shared vertices + 102 shared edges confirmed).
- Composite Vec concatenates sub-DM DOFs — interface DOFs are **duplicated**, not shared. Synchronisation after each solve is still required.
- **Verdict**: Designed for **combining** separate problems (fluid + structure), not **subdividing** one mesh. Doesn't simplify our use case — the core challenge (interface DOF ownership, restrict/prolongate) remains the same either way. The direct subpoint IS approach is simpler and more natural.

**PCFIELDSPLIT with spatial IS** — split by region, not field.

- `PCFieldSplitSetIS()` accepts arbitrary IS — confirmed no restriction to field-based splits.
- Supports Schur complement strategies between spatial blocks.
- **Problem**: This is a preconditioner, not an assembly strategy. Both blocks still assemble from the same DS. Doesn't let you have different equations per region.
- **Verdict**: Useful for preconditioning variable-viscosity systems, but doesn't solve the core problem.

**DMCreateDomainDecomposition** — PETSc's native spatial decomposition.

- `DMCreateDomainDecomposition_Plex()` returns inner/outer IS with configurable overlap.
- `DMCreateDomainDecompositionScatters_Plex()` creates VecScatter for restrict/prolongate.
- **Problem**: Designed for PCASM/PCGASM where the *same* equations are solved on each subdomain. Not for different physics per region.
- **Verdict**: Scatter infrastructure is useful but intent doesn't match multi-physics.

### Assessment

None of the PETSc mechanisms directly solve "different equations on different subsets of the same mesh with shared fields." They each address adjacent problems:

| Mechanism | Different equations? | Shared fields? | Fits? |
|-----------|---------------------|----------------|-------|
| DMComposite | Yes | No (different vector layout) | Partial |
| PCFIELDSPLIT | No (same assembly) | Yes | No |
| DomainDecomp | No (same equations) | Yes | No |
| Region DS | Yes (in theory) | Yes | Segfaults |

The **DMPlexFilter + subpoint IS + UW3-level restrict/prolongate** approach remains the best fit. PETSc provides the building blocks (mesh filtering, IS mapping, parallel SF), UW3 handles the multi-physics orchestration.

## Open Questions

1. **DM lifecycle**: The solver currently clones DMs freely (`clone_dm_hierarchy`). If the submesh also clones, DMs proliferate with no clear ownership. Need a cleanup strategy.

2. **Mesh adaptation**: If the full mesh adapts (refinement, coarsening, surface deformation), the submesh must be re-extracted and the IS rebuilt. All in-flight MeshVariables need re-projection. How does this interact with the existing `refinement_callback` infrastructure?

3. **Parallel decomposition**: `DMPlexFilter` builds a new SF for the submesh. If the partition differs from the parent, restrict/prolongate need MPI communication. How expensive is this? Does it matter for the target use cases?

4. **Coupled solves**: If two solvers on different submeshes need to iterate (e.g., rock Stokes + air transport), the restrict/prolongate happens every outer iteration. Is the data copy overhead acceptable, or do we need shared vectors?

5. **Pressure space**: Discontinuous pressure (dP1) is required for viscosity contrasts at internal boundaries. Should this be the default for submesh solvers, or should the user choose?

## Implementation Plan

### Immediate: `Mesh.extract_region()`

The minimum viable feature. Everything else follows from existing UW3 patterns.

```python
rock_mesh = full_mesh.extract_region("Inner")
```

Wraps `DMPlexFilter`, returns a new `Mesh` with:
- `parent` reference to the full mesh
- `subpoint_is` from `getSubpointIS()` (stored for future optimisation)
- Boundaries inherited from parent labels (they survive DMPlexFilter)
- Coordinate system inherited from parent

The extracted mesh is fully independent — users create their own MeshVariables on it, set up solvers normally, and transfer data between parent and submesh via restrict/prolongate:

```python
# Separate variables on separate meshes
v_rock = MeshVariable("v", rock_mesh, ...)
rho_rock = MeshVariable("rho", rock_mesh, ...)
rho_full = MeshVariable("rho", full_mesh, ...)

# Transfer density from full mesh to rock submesh
rock_mesh.restrict(rho_full, rho_rock)

# Stokes on rock submesh — standard solver, nothing special
stokes = Stokes(rock_mesh, velocityField=v_rock, ...)
stokes.add_natural_bc(penalty * Gamma_N.dot(v_rock.sym) * Gamma_N, "Internal")
stokes.solve()

# Transfer rock velocity back to full mesh
rock_mesh.prolongate(v_rock, v_full)

# Gravity on full mesh using transferred data
gravity = Poisson(full_mesh, ...)
gravity.solve()
```

The restrict/prolongate use the subpoint IS from `DMPlexFilter` — a direct index mapping with exact point correspondence. No kd-tree search, no interpolation, no error. This is the preferred transfer mechanism between parent and submesh.

For transfer between unrelated meshes (no parent relationship), the existing `uw.function.evaluate(expr, coords)` path still works.

### Restrict / Prolongate

```python
rock_mesh.restrict(parent_var, sub_var)    # gather parent DOFs at subpoint IS
rock_mesh.prolongate(sub_var, parent_var)  # scatter submesh DOFs back to parent
```

- No-op when `parent is None` (top-level mesh)
- The subpoint IS maps submesh points → parent points
- Translation from point IS to DOF IS uses the PETSc section (offset lookup per point)
- Exact — same nodes, no interpolation

### Why not auto-managed globals?

We considered having MeshVariables live on the parent mesh with solvers auto-restricting/prolongating. This hides data flow, makes the solver more complex, and the user loses track of where data lives. The explicit approach is clearer: each mesh owns its variables, copies are visible.

### Mesh deformation and adaptation

Changes to the parent mesh must propagate to submeshes. Two cases:

**Coordinate deformation** (ALE, surface evolution): Parent node positions change but topology is unchanged. The subpoint IS remains valid — restrict the parent's coordinate Vec to update submesh node positions. The submesh DM's internal geometry (Jacobians, normals, quadrature) must then be rebuilt.

```python
# After deforming parent mesh coordinates
rock_mesh.sync_coordinates()  # restrict parent coords via subpoint IS, rebuild geometry
```

This should be automatic: if the submesh detects that its parent's coordinates have changed (version counter on the parent mesh, which we already have via `_mesh_version`), it updates on next access.

**Topology change** (adaptation, remeshing): The parent mesh gains/loses cells and vertices. The subpoint IS is invalidated — the submesh must be re-extracted from scratch. All submesh MeshVariables need re-projection onto the new submesh (interpolation from old to new via the usual adaptation path).

```python
# After parent mesh adapts
rock_mesh = full_mesh.extract_region("Inner")  # fresh extraction
# Old submesh variables are orphaned — user must re-create and re-project
```

This is the expensive case. The parent mesh already has `refinement_callback` infrastructure for post-adaptation fixups. The submesh re-extraction could hook into this: the parent notifies registered submeshes that topology has changed, and they invalidate themselves.

The parent `Mesh` should track its submeshes (weak references, like the existing `_registered_swarms` pattern) so it can notify them of coordinate or topology changes.

### Other items

- **Boundary remapping**: Document which parent labels map to submesh boundaries. DMPlexFilter preserves labels; "Internal" on the parent becomes an exterior boundary on the submesh.
- **DM lifecycle**: Audit clone/destroy patterns, ensure submesh DMs are cleaned up.
- **Parallel**: `DMPlexFilter` builds a new SF. Test in MPI before relying on it.

## Additional Findings

### Discontinuous pressure required for viscosity contrasts

Continuous P1 pressure cannot represent the pressure jump at a viscosity discontinuity (scales with viscosity ratio). With eta_rock/eta_air = 1000, the pressure smears across interface elements and corrupts velocity direction up to 177 degrees. Discontinuous P1 handles each side independently — velocity direction error drops to <5 degrees.

### Normalised boundary normal (Gamma_N)

`mesh.Gamma_N` now returns `Gamma / |Gamma|` — a unit normal regardless of element size. The raw `mesh.Gamma` magnitude scales with edge length (2D) / face area (3D). This affects penalty scaling: `penalty * Gamma.dot(v) * Gamma` has effective penalty ~ penalty * h², while `penalty * Gamma_N.dot(v) * Gamma_N` is mesh-independent. Nitsche's `gamma * mu / h` term now has correct 1/h scaling with normalised normals.
