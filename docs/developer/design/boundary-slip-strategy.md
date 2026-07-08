# Boundary-slip strategy: a mesh-owned tangent-slip + surface-restore contract

```{note}
**Status:** design proposal (2026-06-06). This document specifies a refactor;
no behaviour change is intended in the first (interface) step. It gates making
the MMPDE mover the production default — see
`docs/developer/design/anisotropic-mmpde-mover.md` and the mover-API
simplification work.
```

## Motivation

Every metric mover that moves *boundary* vertices needs **tangential boundary
slip**: a boundary node may slide *along* the domain surface (so the mesh can
concentrate cells where a feature meets the wall) but must not drift *off* it
(which would change the domain and, on a free-slip Stokes problem, leak
`v·n ≠ 0`). Slip is therefore two operations applied to the moved boundary
coordinates:

1. **tangent-project** — remove the boundary-normal component of a node's
   displacement, so it slides in the tangent plane;
2. **restore-to-surface** — snap the node back onto the true surface, because
   tangent-projecting a *finite* step off a *curved* surface leaves it a
   sagitta inside the chord; without restoration nodes creep inward over many
   iterations.

Today this logic is **duplicated and geometry-coupled**:

- The most evolved version lives in `meshing/_ot_adapt.py` as private helpers
  (`_resolve_slip`, `_build_slip_projector`, `_slip_normals`,
  `_is_radial_coords`, `_boundary_centre`, `_nearest_on_facets_2d/3d`). The
  `mmpde` and `ot` movers consume it through `_build_slip_projector`.
- The `spring` and `ma` movers carry their **own inline** per-ring radial snap
  (`meshing/smoothing.py`, the `boundary_slip and is_bnd.any()` blocks) — a
  special case of the same idea, written before the `_ot_adapt` unification.
- Whether a boundary is "radial" (snap to `|r|`) vs "flat" (no snap needed) is
  decided at **run time** by a `CoordinateSystemType` heuristic
  (`_is_radial_coords`), and the snap **centre** is recovered every call by a
  parallel `allreduce` over boundary coordinates — even though the constructor
  *knew* the exact centre and radii.
- Free / deforming surfaces (where the surface position is itself unknown) are
  handled by a `dict {label: snap_bool}` opt-out that disables restoration —
  a placeholder for a real kinematic restore that does not yet exist.

This is fragile (four code paths to keep parallel-consistent), guesses geometry
it could be told, and offers no clean seam for the free-surface case. Making
MMPDE the default mover means *every* analytic-boundary mesh must slip
correctly with no per-script `boundary_slip='ring'`/`'box'` fiddling. That
needs a **single, mesh-owned slip contract**.

## Proposal

Introduce a **first-class bounding-surface object** that owns a boundary's
geometry, state flags, and the methods to operate on it (tangent-project,
restore, normals). Slip is then a *bounding-surface-level* capability — the
surface knows whether it is radial, planar, free, or generic, and how to restore
a point to itself — and the **mesh is only the orchestrator** of the
cross-surface concerns the movers need (which vertices slip vs pin, including
junction vertices shared by two surfaces, and composing the per-surface
operations into one pass).

This is the key correction over a mesh-level provider dict: behaviour and state
live on the surface, not in a side table keyed by label. If one surface is
radial and another is free, each carries its own flags and the orchestrator
calls the correct per-surface restore for each.

```{important}
**`mesh.boundaries` is NOT repurposed.** That `Enum` carries the gmsh-style
boundary *labels* that persist all the way through: gmsh `.msh` → PETSc DMPlex
`DMLabel`s → the mesh hdf5 checkpoint. It is a persistence contract, not a
scratch object — silently turning it into rich objects would risk the
label round-trip. The bounding-surface objects live in a **separate**
collection (`mesh.bounding_surfaces`); each *references* a boundary label by
name but does not replace it. Reusing the `mesh.boundaries` name for the rich
objects would require a deliberate deprecation plan and is out of scope here.
```

### Bounding-surface objects

A `BoundingSurface` binds a boundary *label* (a name in `mesh.boundaries`) to
that surface's geometry, state, and slip methods:

```python
class BoundingSurface:
    label: str                # the gmsh/DMPlex boundary label ("Upper", "Lower", ...)
    kind: str                 # "radial" | "plane" | "facet" | "free"
    is_free: bool             # True ⇒ restore follows the live surface, not a fixed target
    # geometry it was told at construction (kind-dependent):
    #   radial: centre, radius     plane: point, normal     facet: reference_facets

    def normals(self, coords): ...          # outward unit normals on THIS surface (Gamma_P1 restricted)
    def tangent_project(self, coords): ...  # remove this surface's normal component
    def restore(self, coords): ...          # snap back onto THIS surface (kind-specific)
    def release(self): ...                  # flip rigid → free (free-surface activation)
```

The surface object *is* the restoration capability (the earlier
`TangentSlipProvider` folds into it). `restore` dispatches on `kind`:

| `kind` | meshes / origin | `restore` |
|--------|-----------------|-----------|
| `radial` | `Annulus`, `SphericalShell`, `CubedSphere`, cylinders | exact `|r|` snap about the **known** centre: `centre + r̂·radius`. Concave-safe (no chord sag). Inner/outer surfaces are separate objects with their own radius. |
| `plane` | `UnstructuredSimplexBox`, `StructuredQuadBox` faces | zero the off-plane coordinate (axis-aligned ⇒ tangent-project already keeps it on the face; supports non-axis-aligned planes too). |
| `facet` | loaded-from-file / internal boundaries with no analytic form | nearest point on the surface's reference facets (the current `_nearest_on_facets_*` fallback). Convex-safe; concave bias is a documented TODO. |
| `free` | free-surface module, OR any surface that has been `release()`-d | follow the **current deformed discrete surface** (generic surface tangent), not a fixed target. `is_free=True`. |

The object **encapsulates the geometry the constructor already knows**, so the
runtime `_is_radial_coords` guess and the per-call centre `allreduce` disappear:
`Annulus(radiusOuter, radiusInner, centre)` builds a `radial` surface on
`"Upper"` (radius `radiusOuter`) and on `"Lower"` (radius `radiusInner`)
directly.

### Public API

The mesh orchestrates the per-surface objects. The low-level contract is the
`(is_pinned, project)` pair the movers already consume, plus an in-place
convenience and a public registration hook (decisions confirmed in review,
2026-06-06):

```python
# Low-level: the projector tuple the movers expect. is_pinned drives the
# solve's pinned-DOF set; project() does tangent-project + restore per surface.
is_pinned, project = mesh.boundary_slip(slip_spec, reference_coords=X0)
Y = project(Y_moved)

# In-place convenience (built on the above) for callers that just want coords
# snapped back — e.g. a checkpoint reload, a diagnostic, the free-surface module.
mesh.project_to_slip_surface(coords, slip_spec, reference_coords=X0)

# Public registration so users can install a custom analytic surface object
# (e.g. an ellipsoid) that the constructors don't know about.
mesh.register_tangent_slip_provider(label, surface)
```

`slip_spec` keeps the back-compatible forms already accepted by
`_resolve_slip`: `True`/`"ring"`/`"box"`/`"all"` (all named codim-1
boundaries), a label name, a list of labels, or a `dict {label: snap_bool}`
(`False` = free surface, slip but do not restore).

The two primitives are surface-level, exposed at mesh level for convenience
over a label (and reusable by checkpoints, diagnostics, the free-surface
module):

```python
mesh.tangent_project(coords, labels)    # = surface.tangent_project per label
mesh.restore_to_surface(coords, label)  # = surface.restore for that label
```

`tangent_project` is **geometry-agnostic** — it uses the projected P1
boundary-normal field (`mesh.Gamma_P1`), which is already smooth and
consistently oriented on curved boundaries where raw face normals are noisy.
`restore` is **geometry-specific** and dispatches on the surface's `kind`.

### Restoration is a surface-*state* question, not just initial geometry

A subtlety the surface model must get right (raised in review, 2026-06-06):
**the correct restoration target depends on the surface's current state, not
only on how the mesh was built.** The motivating case is a free surface on a
sphere/annulus. At construction the outer surface is rigid at `|r| = radius`, so
the `radial` restore is exact. But once that surface is `release()`-d to deform
(free-surface dynamics), the analytic radius is *no longer the surface* —
snapping returned nodes to the original `|r|` would actively fight the surface
motion. The restore must fall back to a **generic surface tangent**: keep
returned nodes on the *current* discrete surface (the deformed boundary facets),
not on the frozen analytic shape.

So each surface object is **state-aware** along two axes:

1. **Mode (the `kind`/`is_free` flag on the object).** A surface is *rigid*
   (snap to an analytic target — `radial`/`plane`) or *free* (follow the current
   deformed surface — the generic surface-tangent / `facet`-style restore
   against the live boundary, `is_free=True`). `release()` flips a `radial`/
   `plane` surface to `free` **in place on the object** — no side-table to keep
   in sync. This is the same generic-surface mechanism as the loaded-from-file
   `facet` fallback, reached by a *state transition* rather than by mesh type.
2. **Reference.** A rigid surface restores relative to a captured reference
   (`reference_coords` per adapt) for idempotence; a free surface's "reference"
   is the *current* surface, re-read each call from live mesh state.

Implication for the contract: a surface's `restore` must be able to read
**current mesh state** (its live facets / the coordinate field), not just
constants frozen at construction. The analytic kinds are the constant-target
special case; `free`/`facet` are the live-target general case behind the same
method. The free-surface case is therefore a *mode of the same surface object*,
not a parallel code path — and "one surface radial, one free" is just two
objects with different flags, which the orchestrator handles per surface.

### Mesh-side data model

`mesh.bounding_surfaces` is a **new** collection of `BoundingSurface` objects,
keyed by boundary label — *separate from* and *additional to* `mesh.boundaries`
(the persistent gmsh/DMPlex label `Enum`, left untouched). The objects ARE the
slip registry — behaviour and state live on them, not in a side table:

```python
mesh.boundaries                          # unchanged: gmsh/DMPlex labels (persisted)
mesh.bounding_surfaces["Upper"].kind     # "radial"
mesh.bounding_surfaces["Upper"].release()  # → "free" (free-surface activation)
```

Constructors that know their geometry build `radial`/`plane` surfaces for their
labels. A label with no analytic object (mesh loaded from file, an internal
boundary, third-party mesh) defaults to a `facet` surface built from the current
boundary facets — exactly today's geometry-general path, so nothing regresses.
The surfaces are reconstructed on a checkpoint reload from the stored
construction parameters (a loaded mesh otherwise gets only the `facet` default);
because the labels themselves persist in `mesh.boundaries`/DMPlex, the
surface-to-label binding is always recoverable.
`register_tangent_slip_provider(label, surface)` lets a user install a custom
surface object (e.g. an ellipsoid) the constructors don't know about.

### Slip-vs-pin classification (unchanged, made canonical)

A boundary vertex **slips iff it belongs to exactly one slip surface**.
Vertices on a non-slip boundary (count 0), at a **junction** of two slip
surfaces (count ≥2 — a box corner, where the normal is ambiguous), or with a
degenerate/non-finite projected normal are **pinned**. This is the
label-driven rule already in `_build_slip_projector` (it fixed an older
topology classifier that spuriously pinned coarse-but-smooth curved rings); the
refactor promotes it to the one canonical implementation.

## Why this is the clean-land gate

- **Mmpde-as-default needs zero per-script slip config.** With surface objects
  built at construction, `boundary_slip=True` on any analytic-boundary mesh
  "just works" — no `'ring'` vs `'box'` choice, no coordinate-type guess.
- **One parallel-safe code path.** Today four movers must each keep the
  owned-vertex projection + halo sync + collective centre consistent; the
  refactor leaves exactly one.
- **A real seam for free surfaces.** A surface's `free` mode is where the
  kinematic restore lands later, behind the *same* `mesh.boundary_slip`
  interface the movers already call — no second integration.

## Migration plan (interface first, per the branching discipline)

Following `docs/developer/guides/branching-strategy.md` (extract the interface,
land it on `development`, keep the feature branch to implementation):

```{note}
**Implementation reality (2026-06-07).** The unified slip projector
(`_build_slip_projector`, `_resolve_slip`, `_gamma_p1_at_vertices`,
`_nearest_on_facets_*`) lives on the mover **feature branch**, not on
`development`. On `development` `_ot_adapt.py` has only the *primitive* helpers
(`_slip_normals`, `_boundary_centre`, `_is_radial_coords`) and the movers use
their older inline slip. So step 1 on `development` is implemented as a
**self-contained, additive** API (it does not depend on the feature-branch
projector), and step 2's bit-identical claim is interpreted as
*machine-precision identical* — the analytic centre differs from the feature
branch's boundary-COM `allreduce` only at round-off.
```

1. **Bounding-surface objects + mesh API, additive — no behaviour change**
   (lands on `development`):
   - Add `mesh.bounding_surfaces`: a **new** collection of `BoundingSurface`
     objects keyed by boundary label, with `kind`/`is_free` + the
     `normals`/`tangent_project`/`restore`/`release` methods. **Leave
     `mesh.boundaries` (the persisted gmsh/DMPlex label `Enum`) untouched.**
   - Add `mesh.boundary_slip` (orchestrator), `mesh.tangent_project`,
     `mesh.restore_to_surface`, `mesh.project_to_slip_surface`, and
     `register_tangent_slip_provider`.
   - Build `radial`/`plane` surfaces in the `Annulus`, `SphericalShell`,
     `CubedSphere`, and box constructors. **Self-contained**: `radial`/`plane`
     `restore` are direct (analytic centre/radius, plane projection); `normals`
     reuses the primitive `_slip_normals` (Gamma_P1); a label with **no**
     analytic surface is **pinned** (safe default) — `facet` restore is a
     follow-up. No dependence on the feature-branch projector.
   - Because the `development` movers do not call the new API, step 1 cannot
     change any existing trajectory (additive). Validate with **new tests**:
     constructors register the right `kind`/geometry; `radial.restore` lands
     points on `|r|`, `plane.restore` on the face; `release()` flips to `free`;
     `mesh.boundaries` is unchanged.
2. **Movers consume the public API** (on the mover feature branch, which has the
   unified projector): replace the `_build_slip_projector` call in `mmpde`/`ot`
   and the **inline** radial-snap blocks in `spring`/`ma` with
   `mesh.boundary_slip(...)`. Delete the private duplicates. Re-validate the
   convection harness (serial + np=5) for *machine-precision-identical*
   behaviour (the centre source changes COM → analytic).
3. **Follow-up (separate work)**: the `facet` restore + its concave-bias cure
   (mean-preserving / smoothness constraint — the documented TODO) and the
   `free`-surface restore mode (`release()` + live-surface follow) for the
   free-surface + adaptive-mesh case (cf. `project_freesurface_ale_design`).

## Invariants the refactor must preserve

- **Parallel safety.** Projection touches only owned vertices; the caller (the
  mover) halo-syncs. Any collective (e.g. a free-surface global reduction) must
  run unconditionally on every rank — the analytic surfaces remove the only
  current collective (the centre `allreduce`) because the centre is a known
  constant.
- **DM-stale safety.** `mesh.Gamma_P1` (the `_projected_normals` MeshVariable)
  must exist *before* a mover builds its solver DM (creating that MeshVariable
  mid-mover stales the DM handle — see `project_uw3_smoother_footguns`).
  Building surface objects / pre-touching `Gamma_P1` at construction makes this
  automatic instead of relying on `_resolve_slip`'s pre-touch.
- **Free surfaces** (`surface.is_free`, or a `dict` `False` value) slip without
  restoration.
- **Reference surface.** A rigid surface's restore is relative to a fixed
  reference (`reference_coords`, captured once per adapt) so repeated projection
  is idempotent and does not accumulate drift.
- **Units.** Surfaces store radii/points in the mesh's coordinate units.

## Decisions (review, 2026-06-06)

- **Locus.** Slip behaviour + state live on **bounding-surface objects** in a
  new `mesh.bounding_surfaces` collection, not a mesh-level provider table and
  **not** by repurposing `mesh.boundaries` (the persisted gmsh/DMPlex labels,
  left untouched). The mesh orchestrates cross-surface concerns (vertex
  slip-vs-pin classification, junctions, composition).
- **API surface.** Low-level `(is_pinned, project)` (the movers' contract) **plus**
  an in-place `mesh.project_to_slip_surface(coords, spec)` convenience built on
  it.
- **User-registerable surfaces.** Yes —
  `mesh.register_tangent_slip_provider(label, surface)` is public, so users can
  install a custom surface object (e.g. an ellipsoid).
- **Land plan.** Step 1 (bounding-surface objects + mesh API, internally
  delegating to the existing helpers, bit-identical) lands on `development` as
  its own PR; the mover-side swap (step 2) stays on the feature branch — clean
  API/impl split.

## Still open (for review)

1. **Naming.** Method `boundary_slip` (matches the existing kwarg) vs
   `slip_project` vs `tangent_slip`; the surface class `BoundingSurface` and its
   collection `mesh.bounding_surfaces` (chosen to keep `mesh.boundaries` safe);
   the registration method `register_tangent_slip_provider` (now installs a
   surface object — reconcile vs `register_bounding_surface`?); the `kind`
   values `radial`/`plane`/`facet`/`free`.
2. **Concave non-analytic restore.** Defer the `facet`-kind concave-bias cure to
   the follow-up, or block on it? (Radial surfaces take the exact analytic
   branch and are immune; the bias only bites a concave *non-analytic* surface,
   which no current production case hits.)

## Roadmap: from boundary slip to a mesh-owned surface contract (2026-06-09)

The tangent-slip contract above is the first instance of a more general idea: a
mesh keeps **declared surfaces** intact as it redistributes its nodes. This
section records the design we settled on for growing it from "the outer
boundary" to "any surface the mesh must preserve" — driven by the metric movers
(it is squarely *mesh-redistributor* work), with codim-1 **submesh extraction**
as the horizon we steer by rather than a separate effort. None of this is
implemented yet; it is the agreed direction and the constraints it must honour.

### Principles (load-bearing)

- **Declaration over topology, never an alternative topology.** DMPlex and its
  labels are authoritative. A `BoundingSurface` only *annotates* a label the
  mesh already owns ("this label of mine means a radial / plane / free
  surface"); it never *defines* topology. There is nothing to keep in sync —
  the same discipline that keeps `mesh.boundaries` (the persisted labelling)
  untouched, promoted to a rule. *The mesh decides what is important and what
  its declared objects represent.*
- **Geometry is per-surface, never per-mesh.** A spherical *regional* mesh is
  the decisive case: its caps are `radial` but its great-circle side cuts are
  `plane`, and the mesh's `SPHERICAL` `CoordinateSystem` is *wrong* for those
  sides. There is no single "mesh geometry" to inherit. Because each label
  carries its own `kind`, the heterogeneous case is correct by construction —
  **nothing reads the mesh's coordinate frame, only a surface's geometry.**
  This one rule disarms the r/θ/φ-on-a-plane trap, the deferred `geographic`
  case, and the dimension-drop ambiguity together.
- **A submesh declares its *own* surfaces; it does not inherit the parent's.**
  An internal interface becomes a bounding surface of an extracted submesh
  because *topologically it now is one* — the submesh, being a mesh, declares
  it. The connection is that both meshes annotate the *same persisted label*
  (and may reference the same geometry object): **borrow by reference, never
  re-home.** Re-deriving a surface's geometry under a dimension/coordinate
  change *is* the hard part — that is what stays deferred (geometry
  inheritance), and the per-surface reference is the seam that lets us tackle it
  later one `kind` at a time without re-plumbing extraction.

### Geometry-kind ⟂ capabilities

A surface has a **geometry kind** (`radial`/`plane`/`facet`/`free`) and a set of
**orthogonal capabilities**, declared independently:

- **`tangent_moving`** — the mover keeps nodes *on* this surface
  (`tangent_project + restore`). This is the broad, near-universal requirement:
  slip but stay on the surface to *preserve* it. It applies to outer
  boundaries, regional edge cuts, **internal interfaces**, and free surfaces
  alike. An internal interface *needs* it for the same reason an outer boundary
  does, turned inward — adapt the mesh without slip-constraining the interface
  and its nodes drift off it, destroying the surface you meant to preserve.
- **`extractable`** — a codim-1 submesh can be filtered from this surface. The
  narrower, opt-in capability; desirable but separate from preservation.

The build priority follows: `tangent_moving` for internal interfaces is the part
with *teeth* (correctness under adaptation); `extractable` is convenience on top.

**Concrete first extension.** Today the mover's slip gate is `is_bnd` — only
*outer* codim-1 labels are slip-eligible. To preserve an internal interface, its
label must enter the slip set even though those nodes are topologically interior,
and `mesh.boundary_slip` projects them onto the interface's `BoundingSurface`
exactly as it does an outer ring. The per-surface orchestration ("project nodes
on surface X back onto X, pin the junctions") already does the right thing; the
only change is that the eligible-vertex set becomes *"any vertex on a
`tangent_moving` surface"* rather than *"on the outer boundary."*

### Scope: interfaces yes, faults no

Bounding surfaces are the named codim-1 surfaces a mesh *declares* as
actual-or-potential boundaries — outer boundaries **and** internal interfaces
(including the free surface, which is just an internal-interface surface that has
been `release()`-d to `free`). A **fault is not** one of these: it is an
internal feature represented its own way (not a subdomain boundary; material is
~continuous across it, with slip), and the registry must not absorb it. Nothing
auto-classifies an internal surface — the interface-mesh constructor declares the
interface as a bounding surface; the fault machinery declares faults its own way.

### Declaration mechanism

- **Built-in meshes are the worked example.** The analytic constructors register
  at construction via helpers (`register_radial_surfaces`, a `plane` /
  internal-interface helper to add); that constructor code is the canonical
  template, because the helpers are *also* the public API a user calls by hand
  after loading their own gmsh. Keep them ergonomic and obvious.
- **User gmsh is the number→name→geometry sync.** gmsh gives numbers, DMPlex
  gives named-but-opaque labels, the geometry lives nowhere until UW3 declares
  it. The seam already isolates the hard part: `BoundingSurface` keys off the
  **label name**, never the gmsh number, so registration sits *after* the
  existing numbers→names mapping (`mesh.boundaries`), on stable names. Helpers to
  ease that chain are future work but bolt onto a name-based seam.

### Persistence (checkpoint roundtrip)

Surfaces are currently reconstructed only by re-running the constructor — a mesh
*loaded* from a checkpoint gets nothing but the `facet` default. Bounding-surface
metadata must therefore ride in the HDF5 next to the boundary-label metadata, and
reload must rebuild the objects. What is persisted is small and is *annotation,
not topology* (the DMPlex/labels roundtrip by their own mechanism; the surface
info is a sidecar keyed by label name), and it is kind-dependent:

- `radial` / `plane` — persist the few construction scalars (centre/radius,
  point/normal); exact reconstruction.
- `facet` — do not persist; it is derived from the current boundary facets, so
  regenerate on load.
- `free` — persist the mode flag (+ reference if any); the geometry is live.

A submesh roundtrips *its own* declared surfaces, consistent with "each mesh
declares its own."

### Discoverability

If the mesh *declares* its surfaces, the declarations must be *inspectable* —
and a checkpoint-loaded mesh must be *equally* self-describing (this is why
persistence matters, not just reconstruction-by-constructor). By examination the
mesh should answer:

- **What surfaces do I define?** — enumerate `mesh.bounding_surfaces`, with a
  human-readable summary of `label · kind · capabilities · geometry`.
- **By capability** — "which are `tangent_moving`? which are `extractable`?" — so
  the mover and the submesh extractor each ask the mesh for *their* set instead
  of hard-coding label names.
- **How do I access them** — the same objects carry both the operations
  (normals/restore) and the access path (slip via `mesh.boundary_slip`,
  extraction via `extract_surface(surface)`); discovery and use are one surface.

### Suggested build order (smallest-first)

1. **Registration helpers as template code** — mostly exists; add the
   `plane` / internal-interface helper and register regional edge cuts as
   `plane` (a correctness gap for boundary-slip on regional meshes *today*).
2. **`tangent_moving` for internal interfaces** — generalise the slip gate from
   `is_bnd` to "any `tangent_moving` surface." The part with teeth.
3. **HDF5 persistence** of the analytic surface metadata for checkpoint
   roundtrip; discoverability falls out of it.
4. **`extractable` + submesh re-declaration** — extraction accepts a surface and
   the child re-declares its surviving labels. Geometry inheritance stays parked.
5. **numbers→names→geometry helpers** for hand-rolled gmsh — later.

## Deferred cases (handle after the simple analytic geometries)

- **Geographic meshes are an odd case** (flagged in review, 2026-06-06). The
  `GEOGRAPHIC` coordinate system mixes a radial (depth) direction with
  lon/lat surface coordinates and a non-trivial metric; "tangent to the
  surface" and "the projected normal" are not the plain Cartesian operations the
  `radial`/`Gamma_P1` path assumes. **Do not** try to cover it in the first cut
  — get `Annulus`/`SphericalShell`/box working under the new contract first, then
  design a `geographic` surface `kind` against the settled interface. Until then
  geographic meshes keep the current `_is_radial_coords` → radial-snap behaviour
  via the fallback path (they classify as radial today).
- **Free-surface restoration** (the surface `free` mode and the `release()`
  rigid→free transition described above) is the primary follow-up
  (cf. `project_freesurface_ale_design`).
```
