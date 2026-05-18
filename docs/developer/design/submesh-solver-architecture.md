# Submesh Solver Architecture: Multi-Domain Equation Systems

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

## Two Submesh Flavours: Subdomain and Resolution Level

A *submesh* in UW3 is any mesh pulled out of a parent mesh that retains a
lineage link (`parent`, registration in `parent._registered_submeshes`)
and supports explicit field transfer back and forth. There are two
flavours, and they share one usage pattern:

> **get a submesh → build a solver on it → map fields back and forth**

| | Subdomain | Resolution level |
|---|---|---|
| Constructor | `mesh.extract_region("Inner")` | `coarsened_companion(fine, levels=1)` |
| PETSc mechanism | `DMPlexFilter` (cell subset) | `dm.refine()` nested hierarchy |
| Parent ↔ child map | `getSubpointIS()` (point-level) | nested `createInjection` / `createInterpolation` (DOF-level) |
| Transfer fidelity | exact (shared nodes) | exact (nested FE) |
| Example | `docs/examples/submesh_investigation/test_region_ds_submesh.py` | `docs/examples/submesh_investigation/example_refined_companion.py` |

Both examples solve the **same** annulus + radial-buoyancy Stokes problem
so the flavours are directly comparable: one solves on a *subdomain* of
the annulus, the other on a *coarser resolution* of the whole annulus,
and both map the solution back to the parent.

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
| `DMPlexCreateSubmesh(dm, label, value, ...)` | Co-dimension 1 submesh (boundaries) | Works but wrong dimension |
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
