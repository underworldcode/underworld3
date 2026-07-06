---
title: "Swarm Modernization Design (2026-07)"
---

# Swarm Modernization Design

**Status**: PROPOSED — awaiting maintainer review
**Date**: 2026-07 (quality campaign, dimension 5, deliverable FO-01)
**Provenance**: every current-behaviour claim in this document was verified against
`development` @ `3184a406` — by reading the code and, where marked *(probe-verified)*,
by runtime reproduction in the `amr-dev` environment built from that commit.

This document is the design blueprint for modernizing the swarm subsystem
(`src/underworld3/swarm.py` and its immediate clients). It builds on the audit
`docs/reviews/2026-07/SWARM-SUBSYSTEM-REVIEW.md` (findings SWARM-01…24) and the
remediation worklist (`docs/reviews/2026-07/REMEDIATION-WORKLIST.md`, row FO-01).
It is a **design document, not a refactor**: the maintainer ruled "audit +
modernization design doc now; deep refactor is a follow-on project". Follow-on
sessions implement the phases in {ref}`swarm-design-phasing`.

```{important}
Several audit findings have already been FIXED by the Track-0 remediation PRs
(#313, #323, #329). They are cited here as history and context only. A future
session that treats SWARM-01…08, SWARM-11, SWARM-16, SWARM-17 or SWARM-19 as
*open* problems is misreading the baseline — see the status table below.
```

## Finding status at `3184a406`

| Finding | Status | Where it went |
|---|---|---|
| SWARM-01/02/17 (stale cache / poisoned kd-tree after no-move migrate, particle addition, populate) | **FIXED** (#313, BF-02) | `Swarm.migrate` invalidates before its no-move early return; `add_particles_with_global_coordinates` and `populate` invalidate explicitly. *(probe-verified: fresh `.data` after both paths)* |
| SWARM-03 (modern coordinate writes never migrate) | **FIXED** (#313, BF-07) | Deferred-migration flag `_needs_migration`, consumed at collective points (see {ref}`swarm-design-migration-matrix`) |
| SWARM-04 (writes inside `migration_disabled()` discarded) | **FIXED** (#313, BF-05) | Deferred PETSc pack + `Swarm._flush_pending_petsc_sync()` at context exit |
| SWARM-05 / #215 Bug 3 / #289 (stale proxy consumed by solvers) | **FIXED** (#313, BF-08) | `Swarm._sync_before_assembly()` invoked from `Mesh.update_lvec()` |
| SWARM-06 (Lagrangian ddt history crash) | **FIXED** (#313, BF-04) | `systems/ddt.py` history writes now use `psi_star_0.data[:, psi_star_0._data_layout(i, j)]` |
| SWARM-07 (empty ranks: silent zeros / KDTree crash) | **FIXED** (#313, BF-06) | Starved-rank guards in `SwarmVariable._rbf_to_meshVar`, `rbf_interpolate`, `IndexSwarmVariable._update_proxy_variables` |
| SWARM-08/09 (dead recycle feature; dead `pic_swarm.py`) | **RESOLVED** (#323, ruling D4) | Feature excised, `recycle_rate > 1` raises `NotImplementedError`; `swarms/pic_swarm.py` deleted |
| SWARM-10 (no self-validating cache) | **OPEN** — this document, {ref}`swarm-design-cache` | |
| SWARM-11 (NodalPointSwarm positional-arg bug) | **FIXED** (#323, BF-14 + ruling D5) | Keyword-explicit super call; class deprecated, removal next cycle |
| SWARM-12 (IndexSwarmVariable eager update, `sym_1d` staleness) | **OPEN** — FO-02, see {ref}`swarm-design-phasing` | Lazification now *unblocked* by BF-08 |
| SWARM-13 (internal deprecated-pattern debt) | **IN FLIGHT** (Wave B, parallel branch) | Behaviour-neutral; this analysis remains valid |
| SWARM-14 (per-access view classes, triplicated `_data_layout`, dead `_array_cache`) | **OPEN** — this document, {ref}`swarm-design-shared-views` | |
| SWARM-15 (rank-local RBF seams) | **OPEN** — this document, {ref}`swarm-design-rbf-seams` | |
| SWARM-16 (substep advection wrong-rank evaluation) | **FIXED** (#323, BF-16) | Launch-point evaluation uses `global_evaluate`; `estimate_dt` shape bug fixed |
| SWARM-18 (migration-control exit gaps) | **PARTIALLY OPEN** — {ref}`swarm-design-migration-matrix` | |
| SWARM-19 (save() coordinate-system mismatch) | **FIXED** (#323, BF-17) | Both IO branches write model-unit coordinates |
| SWARM-20 (KDTree no-copy memoryview) | **OPEN** — this document, {ref}`swarm-design-kdtree` | |
| SWARM-21 (hot-path prints, dead `corrector`/`evalf` params) | **OPEN** (WA-16, not landed) | `Swarm.advection` still prints unconditionally per substep |
| SWARM-22 (dead API surface) | **PARTIALLY RESOLVED** | `sync=` kwarg now deprecates via `SwarmVariable._warn_deprecated_sync`; `old_data`, `use_legacy_array`/`use_enhanced_array`, commented pack/unpack blocks remain |
| SWARM-23 (`_get_map` stale-cache trap) | **OPEN** — this document, {ref}`swarm-design-getmap` | |
| SWARM-24 (misleading subsystem doc stub) | **INTERIM FIX** (WE-11, this PR) | Banner on `docs/developer/subsystems/swarm-system.md`; the real subsystem doc is written when this design is implemented |

Parallel restore duplication (#324, found after the audit) was **FIXED** by #329:
`Swarm.read_timestep` now uses a keep-local restore (see
{ref}`swarm-design-checkpoint`).

## The architecture today

The verified baseline, so the target designs below have a precise reference point.

**Data pipeline.** A particle field lives in a PETSc DMSwarm field. User access goes
through one cached *canonical* NumPy copy per variable — an `NDArray_With_Callback`
created by `SwarmVariable._create_canonical_data_array`, returned by the
`SwarmVariable.data` property in flat `(N, num_components)` shape. Writes fire
`canonical_data_callback`, which packs back to PETSc
(`SwarmVariable.pack_raw_data_to_petsc`), marks the proxy stale, and — for the
coordinate variable — marks the swarm for deferred migration. The `.array` property
wraps the same canonical data in a units/shape view built fresh on every access
(three-index shapes per Charter §7). Variables with `proxy_degree > 0` own a proxy
MeshVariable refreshed by rank-local RBF interpolation over a cached particle
kd-tree; `IndexSwarmVariable` instead owns one level-set MeshVariable per material
index.

**Consistency contracts.** Five hand-maintained contracts connect the layers:

1. *pack-on-write* — canonical writes reach PETSc via the callback (deferred, not
   discarded, while migration is suppressed — SWARM-04 fix);
2. *cache invalidation on layout change* — `Swarm._invalidate_canonical_data()`
   drops every canonical array, the kd-tree, and marks proxies stale; called from
   `migrate` (both exit paths), `populate`, both `add_particles_*` methods,
   `apply_snapshot_payload`, and `_route_by_nearest_centroid`;
3. *kd-tree invalidation* — same call path (`self._kdtree = None`);
4. *proxy staleness* — writes set `_proxy_stale`; freshness is enforced at
   consumption: the lazy `.sym` accessors and, since #313, the collective
   solve-entry refresh `Swarm._sync_before_assembly()` invoked from
   `Mesh.update_lvec()` (which the `evaluate`/`Integral` paths also call);
5. *migration on coordinate change* — deferred flag, consumed at collective points.

Contracts 4 and 5 now have consumption-side enforcement. Contracts 1–3 are still
**discipline-based**: nothing validates the canonical cache against the DMSwarm
(see {ref}`swarm-design-cache`).

**What Track 0 deliberately did not solve.** #313 fixed the forgotten-invalidation
*instances* and added the solve-entry choke point, but left the single-cache-
no-validation design in place; #323 excised dead features and fixed medium-severity
defects; #329 fixed parallel restore duplication and documented the honest contract
of `add_particles_with_global_coordinates`. The remaining work is architectural
convergence — the subject of this document.

**Wave B.** A parallel remediation branch is migrating the remaining internal
`with swarm.access(...)` sites to direct `.data` access (behaviour-neutral). After
it lands, internal call sites use direct `.data`/`.array`; the designs below assume
that state. At `3184a406` the only live `access()` shims left in `swarm.py` are in
dead code (`_rbf_reduce_to_meshVar`) and in the deprecated `NodalPointSwarm`.

(swarm-design-cache)=
## 1. Self-validating canonical cache (SWARM-10)

### Current behaviour

`SwarmVariable.data` returns the cached `_canonical_data` on a bare existence
check — there is no size, identity, or generation validation. Freshness relies
entirely on `Swarm._invalidate_canonical_data()` being called at every
layout-mutating site. The Track-0 fixes made those calls complete for the *known*
sites, but the design is unchanged: a bare PETSc-level mutation still poisons every
cached array. *(probe-verified at `3184a406`: after a bare `swarm.dm.addNPoints(2)`,
`dm.getLocalSize()` = 400 while `var.data.shape[0]` = 398; an explicit
`_invalidate_canonical_data()` heals it.)*

A `Swarm._population_generation` counter exists and is bumped at every wrapped
mutation site (`populate`, `migrate` — unconditionally, including the no-move
path — both `add_particles_*` methods, `apply_snapshot_payload`), but it is
consumed only by starved-rank warning guards and the snapshot payload — **no cache
validation reads it**. *(probe-verified: the counter does not move on a bare
`dm.addNPoints`.)* Bare-`dm` sites (`_route_by_nearest_centroid`, the deprecated
`NodalPointSwarm.__init__`) do not bump it.

The mesh side solved the same problem in 2025: `_BaseMeshVariable.data`
self-validates by comparing a recorded `id(self._lvec)` against the live vector
(`discretisation/discretisation_mesh_variables.py`, documented in
`docs/developer/subsystems/data-access.md`). The eager invalidation calls there are
a performance optimization; correctness does not depend on them.

### The problem

Discipline-based invalidation in this module has been broken at least five separate
times (#216 ×3, SWARM-01, SWARM-17). Each fix adds another call site to remember.
The swarm cache cannot reuse the mesh trick directly because a DMSwarm field has no
persistent PETSc vector whose identity can be tracked — `getField`/`restoreField`
hands out a transient view — and because the canonical array is a *copy* of the
field (`unpack_raw_data_from_petsc` returns `.copy()`), so even value changes that
preserve the local size (a balanced cross-rank swap during migration) invalidate it.

### Target design

Validate `SwarmVariable.data` against a two-part token recorded when the cache is
created:

```python
# recorded at cache creation
self._canonical_token = (swarm._population_generation, swarm.dm.getLocalSize())

# checked on every .data access
def _cache_valid(self):
    return (
        self._canonical_data is not None
        and self._canonical_token
            == (self.swarm._population_generation, self.swarm.dm.getLocalSize())
    )
```

- The **size** component catches every mutation that changes the local particle
  count, including bare `dm.addNPoints`/`removePoint` calls that bypass all
  wrappers — with no cooperation from the mutating site.
- The **generation** component catches same-size mutations (balanced migrations,
  restore-in-place) — but only at sites that bump the counter. `Swarm.migrate`
  already bumps unconditionally on both exit paths.

To narrow the remaining discipline to a single rule — *"population mutations go
through the Swarm wrappers"* — route the two known bare-`dm` sites through a small
private helper that owns the bump-and-invalidate pair:

```python
def _mutate_population(self):
    """Context manager: brackets any direct DMSwarm layout mutation.
    On exit: bump _population_generation, drop canonical caches and kd-tree."""
```

`_route_by_nearest_centroid` adopts it (it already invalidates, but does not bump);
`apply_snapshot_payload` adopts it (replacing its explicit pair);
`NodalPointSwarm.__init__` is deprecated and dies with the class.

Honest limits, carried over from the audit's refuted-claims appendix (R-4): the
token does **not** make `_invalidate_canonical_data()` optional. Same-size
mutations at non-bumping sites remain invisible to the token, so the existing
eager invalidation calls stay in place (as on the mesh side, they become
belt-and-braces plus a performance optimization — dropping a poisoned kd-tree
early is cheaper than validating it lazily). What the refactor buys is that a
*forgotten* call at a size-changing site — the entire observed bug class — becomes
harmless.

The kd-tree participates for free: `Swarm._get_kdtree` adopts the same token, so a
tree built over a superseded population is rebuilt on next use even if
`_kdtree = None` was missed. Value-only coordinate mutations (in-place writes with
no size change) are already handled: the canonical write callback packs and marks,
and `migrate()`'s unconditional bump covers post-advection state.

### Migration path and tests

1. Add the token check to `SwarmVariable.data` and `Swarm._get_kdtree`; keep all
   existing invalidation calls. Behaviour-identical for every currently-correct
   path (the token only *adds* recreation triggers).
2. Introduce `Swarm._mutate_population()` and route `_route_by_nearest_centroid`
   and `apply_snapshot_payload` through it.
3. Tests (serial, `level_1`/`tier_a`, extending
   `tests/test_0113_swarm_stale_cache_regression.py`):
   - bare `dm.addNPoints` after a `.data` touch → `.data` row count matches
     `dm.getLocalSize()` with **no** explicit invalidation (the probe in this
     document, inverted into an assertion);
   - `_route_by_nearest_centroid` bumps the generation.
4. np2/np4 (`tests/parallel/`): a balanced cross-rank exchange (equal counts both
   directions) followed by `.data` reads on both ranks returns migrated values —
   pins the generation component, which the size check alone cannot catch.

(swarm-design-migration-matrix)=
## 2. Migration trigger matrix (SWARM-03/18)

### Current behaviour — the matrix

#313 (BF-05/BF-07) established deferred migration. The verified matrix at
`3184a406`, with the guarantees each writer gets:

| Writer | PETSc pack | Migration |
|---|---|---|
| `swarm.coords` setter / `_particle_coordinates.data` write | immediate (canonical callback) | **deferred**: callback sets `_needs_migration`; consumed at the next collective point |
| same, inside `migration_control()` / `migration_disabled()` | **deferred**: recorded in `_pending_petsc_sync`, flushed by `_flush_pending_petsc_sync()` at context exit (or defensively at `migrate()` entry) | flush sets `_needs_migration`; see context rows below |
| deprecated `points` setter | immediate | **immediate**: the `_coords` callback (`swarm_update_callback` inside the `Swarm.points` property) calls `migrate()` per write unless `_migration_disabled` |
| `access()` compatibility shim | deferred to context exit (global callback delay), then as the first row | deferred (flag), **not** run at shim exit |
| `migration_control(disable=False)` exit | flush | `migrate()` runs at exit **only if `local_size` is unchanged** across the context; otherwise silently skipped (SWARM-18 residue) — the set flag is then consumed at the next collective point instead |
| `migration_control(disable=True)` / `migration_disabled()` exit | flush | not run at exit, by contract — but the flushed coordinate writes leave `_needs_migration` set, so the *next solve entry migrates anyway* |
| `advection()` | immediate per substep | suspended during the substep loop (`_deferred_migration_suspended`, so `Mesh.update_lvec` inside velocity evaluation cannot reorder particle rows mid-loop); explicit `migrate()` at the end |
| `populate` / `add_particles_with_coordinates` | direct field writes | `dm.migrate` (raw) in `add_particles_with_coordinates`; none needed in `populate` (points are constructed locally) |
| `add_particles_with_global_coordinates` | direct field writes | `Swarm.migrate()` iff `migrate=True` |
| `dont_clip_to_mesh()` exit | — | `migrate()` unconditionally |

**Collective points that consume the deferred flag**: `migration_control()` exit
(size-unchanged case), any explicit `Swarm.migrate()`, and solve entry —
`Swarm._sync_before_assembly()`, called from `Mesh.update_lvec()` for every
registered swarm in deterministic `_instance_number` order.

**Collective-safety rules** (all verified in code):

- `migrate()` is collective and must never run from a per-write callback — ranks
  write unevenly, so a per-write migrate deadlocks (maintainer ruling D12).
- `_sync_before_assembly` combines the rank-local `_needs_migration` and
  `_proxy_stale` flags with global MAX reductions before acting, so every rank
  takes the same sequence of collective actions even under uneven writes.
- Because the hook itself is collective, call sites reached by only a subset of
  ranks must pass `update_lvec(swarm_sync=False)` and rely on an earlier all-ranks
  refresh (the `petsc_interpolate` internals do exactly this — see the
  `Mesh.update_lvec` docstring).
- `_flush_pending_petsc_sync` raises `RuntimeError` if the particle layout changed
  while writes were pending — a deferred write cannot be applied across a layout
  change, and failing loudly beats silently corrupting rows.

### The problems

The matrix works, but it is the *residue of fixes*, not a designed state machine:

1. **SWARM-18 residue**: `migration_control(disable=False)` promises migration at
   exit but silently skips it when `local_size` changed inside the context. The
   deferred flag papers over this (solve entry catches it), but between context
   exit and the next collective point, particles sit on wrong ranks and rank-local
   operations (`rbf_interpolate`, kd-tree queries, `evaluate` without
   `global_evaluate`) act on them.
2. **`disable=True` semantics blur**: "completely disable migration" actually means
   "until the next solve entry", because `_flush_pending_petsc_sync` sets
   `_needs_migration` unconditionally for coordinate flushes. Defensible (a solve
   *requires* locality) but undocumented and surprising.
3. **The deprecated `points` setter still migrates per write** — the one remaining
   immediate-migration path, inconsistent with the deferred design and
   deadlock-prone under uneven writes (it is also the *documented* legacy pattern,
   so it is exercised).
4. **`migration_control` exit calls `var._update()` on every variable** — eager
   re-projection for every `IndexSwarmVariable` (see FO-02), duplicating what the
   stale flags already guarantee.
5. `add_particles_with_coordinates` runs a raw `dm.migrate` that ignores
   `_migration_disabled` entirely (second half of SWARM-18). Rarely harmful — the
   method filters to locally-owned points first, so the migrate is a near-no-op —
   but it is an undocumented exception to the context's contract.

### Target design

Make the deferred-migration state machine the *single* documented mechanism:

- **One writer contract**: every coordinate writer (modern, legacy, shim) sets
  `_needs_migration`; no writer migrates inline. The `points` setter callback
  drops its `migrate()` call and sets the flag (its docstring already steers users
  to `migration_control()` for partial writes).
- **One consumer contract**: collective points consume the flag —
  `migration_control()` exit (unconditionally, removing the size-changed skip:
  the flush has already reconciled the layout, so the guard's original purpose is
  gone), explicit `migrate()`, and solve entry. `dont_clip_to_mesh()` exit
  delegates to the same path.
- **`disable=True` gets explicit semantics** (maintainer decision, question 2 in
  {ref}`swarm-design-questions`): either (a) it *holds* the flag — no migration at
  any collective point until the context user migrates explicitly, accepting that
  a solve inside that window sees wrong-rank particles; or (b) current behaviour —
  it suppresses migration only until solve entry — is kept and documented as
  "migration cannot be suppressed across an assembly".
- **Drop the blanket `var._update()` loop** at context exit; staleness marking in
  the write callbacks plus consumption-side enforcement already cover it (and
  after FO-02, `IndexSwarmVariable` no longer needs the eager call).

### Migration path and tests

1. Document the matrix above in the subsystem doc (the FO-01 follow-on rewrite of
   `docs/developer/subsystems/swarm-system.md`).
2. Behavioural changes (points-setter deferral, exit-skip removal, `disable=True`
   ruling) ship as one PR after the maintainer decisions, each with an np2 test:
   - uneven cross-rank writes through each writer row, asserting re-ownership at
     the promised consumption point and global count preservation (extends
     `tests/parallel/test_0756_swarm_migration_semantics.py`);
   - `migration_control()` with particle addition inside the context, asserting
     migration *at exit* (the SWARM-18 case, currently skipped).
3. `tier_a` gate: the #313 regression suite already pins the deferred semantics;
   any change to consumption points must keep it green.

(swarm-design-shared-views)=
## 3. Shared array-view refactor (SWARM-14) — both variable families

### Current behaviour

- `SwarmVariable.array` builds a **fresh closure-defined view class and instance on
  every access**: `_create_array_view` dispatches to `_create_simple_array_view`
  (scalars/vectors) or `_create_tensor_array_view` (tensors) — together roughly 440
  lines in `swarm.py`, most of it duplicated between the two classes (the
  units/dimensionalisation pipeline appears twice, nearly verbatim).
- `MeshVariable.array` has the **identical** fresh-closure-per-access design in
  `discretisation/discretisation_mesh_variables.py` (audit appendix R-2: this is a
  *shared* defect, not swarm lag).
- Tensor swarm *reads* pay a full PETSc unpack copy per access
  (`TensorSwarmArrayView._get_array_data` calls `unpack_uw_data_from_petsc`), and
  tensor `__setitem__` re-unpacks, copies, packs per assignment.
- `_data_layout` exists in three near-copies: `SwarmVariable._data_layout`,
  `Swarm._data_layout` (annotated "cut'n'pasted from the MeshVariable class"; this
  copy lacks the negative-index storage-size convention the SwarmVariable copy
  supports), and `_BaseMeshVariable._data_layout`. All three encode the same
  component maps (including the Voigt/SYM_TENSOR map).
- `_array_cache` is dead on **both** sides: assigned `None` in
  `SwarmVariable.__init__` and twice in `discretisation_mesh_variables.py`, never
  assigned anything else. The one read —
  `Swarm._force_migration_after_mesh_change` guards on
  `var._array_cache is not None` — is a permanently-false branch.
- The reduction contract is already unified (#323, BF-11/ruling D6): both swarm
  view classes carry `_per_component_reduction` matching the MeshVariable
  per-component-tuple convention.

### The problem

Two ~400-line implementations of the same contract drift independently (the
reduction contract had already drifted once, BF-11). Per-access class creation is
avoidable overhead and defeats `isinstance`/identity reasoning. The layout map —
pure data — is copied into three methods, one of which is already divergent.

### Target design

One view implementation, one layout function, shared by both variable families:

- **`data_layout(vtype, dim, shape, i, j=None)`** — a module-level function (new
  module, e.g. `src/underworld3/discretisation/variable_layout.py`) that owns the
  component maps and the storage-size convention. `SwarmVariable._data_layout`,
  `_BaseMeshVariable._data_layout` become one-line delegates during a deprecation
  cycle; `Swarm._data_layout` (used only by the legacy access machinery) is
  deleted with `_legacy_access` (Wave B item WB-06).
- **One `VariableArrayView` class** (same module), defined once at module scope,
  parameterized by a small adapter protocol the variable supplies:
  `get_flat()` / `set_flat(arr)` (canonical data in `(N, components)`),
  `shape`, `num_components`, `units`, and the layout function. The class carries
  `__getitem__`/`__setitem__`, the units pipeline (one copy), the
  `_per_component_reduction` reductions, `__array__`/`__array_ufunc__`, and
  `delay_callback`. Charter §7 three-index shapes — scalars `(N, 1, 1)`, vectors
  `(N, 1, dim)`, tensors `(N, dim, dim)` — are the single contract for both
  families.
- **Views are cached per variable** and validated with the same token that
  validates the canonical cache ({ref}`swarm-design-cache` for swarm; `_lvec`
  identity for mesh). This finally gives `_array_cache` a real meaning — or,
  simpler, the view holds no state beyond its adapter and needs no invalidation at
  all (it reads through `get_flat()` on every operation, exactly as the closure
  classes do today). Preferred: the stateless form; delete `_array_cache`
  outright.
- Tensor read/write goes through the same pack/unpack helpers but against the
  canonical cache, not a fresh PETSc unpack per access — the canonical array is
  already synchronized, so the extra `getField` round-trip is pure waste.

### Migration path and tests

1. Extract `data_layout` + `VariableArrayView`; port **MeshVariable first** (it has
   the broader test coverage: `test_0850`–`test_0852` reduction suites and the
   mesh-variable access tests), asserting bit-identical behaviour.
2. Port SwarmVariable; delete the two closure classes and `_array_cache` (all
   sites). `test_0530` pins the `use_legacy_array`/`use_enhanced_array` no-op
   stubs — retire test and stubs together (SWARM-22 remnant, maintainer sign-off
   under the Wave-A deletion protocol).
3. Gate: full `tier_a` identical pre/post per port; the reduction suites and the
   units-pipeline tests (`.array` read/write with scaling active) are the
   behaviour net. No np-sensitivity expected (views are rank-local), but the
   standard np2 swarm suite runs anyway.

(swarm-design-rbf-seams)=
## 4. Rank-local RBF seam behaviour (SWARM-15)

### Current behaviour

Proxy refresh interpolates **strictly rank-locally**: `SwarmVariable._rbf_to_meshVar`
→ `rbf_interpolate` → `KDTree.rbf_interpolator_local` over `Swarm._get_kdtree()`,
which indexes only this rank's particles. There is no ghost-particle exchange —
no `DMSwarmCollect`, halo, or neighbour communication anywhere in `swarm.py` or
`ckdtree.pyx` (verified by search). A proxy node near a partition boundary is
therefore interpolated from a one-sided neighbourhood: with the default
`nnn = dim + 1` nearest particles, a node whose true nearest particles live on the
neighbouring rank gets values extrapolated from farther same-rank particles.
Results are deterministic but **np-dependent**. The degenerate end of the same
family (≤ 1 local particles) is now guarded — starved ranks leave their proxy
untouched and warn (#313, BF-06). This finding was evidence-read but not
adversarially verified in the audit; the mechanism is confirmed by the absence of
any exchange path, but the seam error magnitude has not been measured.

**Relation to #314**: `global_evaluate` hangs at np4 when a rank has zero interior
points (DMInterpolation empty-rank asymmetry, `tests/parallel/test_0760`). That is
a defect in the *evaluation* path, pre-existing and tracked separately. It is
related only as the other member of the "parallel edges" family — the seam design
below neither depends on nor fixes it, and a fix for #314 does not change seam
interpolation.

### The problem

Proxy fields (including material masks) differ between serial and parallel runs of
the same model near partition seams, by an amount set by the local particle
spacing. For convergence studies and bit-reproducibility across np this is a
silent, distributed error source with no diagnostic.

### Target design

Halo-particle exchange before proxy refresh (parallel-first, Charter §11):

- Each rank identifies its particles within a halo distance `h` of the partition
  boundary (using the mesh's existing domain kd-tree machinery:
  `mesh._get_domain_kdtree()` distances to neighbouring-domain centroids, or a
  facet-based bound; `h` = a small multiple of the local mean particle spacing,
  bounded below by the mesh cell size at the seam).
- Exchange those (coords, values) with neighbouring ranks (point-to-point, ranks
  known from the mesh partition adjacency), build the interpolation kd-tree over
  local + halo particles inside `_rbf_to_meshVar` / `_update_proxy_variables`.
- The proxy write itself remains rank-local (each rank writes only its own nodes),
  so the collective structure of the refresh (single symmetric write per level
  set, established by BF-06) is unchanged.
- The halo tree is transient (built inside the refresh); the cached
  `Swarm._get_kdtree()` stays local-only for its other consumers.

This makes proxy values np-independent up to halo truncation. The alternative —
documenting rank-locality as the contract and testing only bounds — costs nothing
but concedes np-dependence permanently; the recommendation is the exchange, sized
as its own workstream (it is the largest item in this document).

### Migration path and tests

1. First, a *measurement* test (np2 vs np1 proxy field difference on a fixed
   seeded swarm) to establish the baseline seam error — this is also the
   adversarial verification SWARM-15 still owes. Tier `tier_b` while
   tolerance-based.
2. Implement the exchange behind a `Swarm` option (default on once validated);
   np2/np4 tests assert the proxy field matches np1 to a tolerance that tightens
   with halo width, and exact equality for `nnn=1` when the true nearest particle
   is inside the halo.
3. Benchmark the refresh cost (the exchange adds one neighbour communication per
   refresh; refreshes are already collective, so latency rides existing sync
   points).

(swarm-design-getmap)=
## 5. `_get_map` stale-cache trap (SWARM-23)

### Current behaviour

`Swarm._get_map` caches nearest-particle index maps in `Swarm._nnmapdict`, keyed by
an `xxhash` of the *query* (proxy-node) coordinates only — nothing in the key or
the cache reflects particle state. The dictionary is cleared in exactly one place:
the `_legacy_access` exit manager — which has zero callers. It is **not** cleared
by `migrate()` or `_invalidate_canonical_data()`. The sole consumer chain is
`SwarmVariable._rbf_reduce_to_meshVar` → `_get_map`, and `_rbf_reduce_to_meshVar`
itself has zero callers (verified by search; it also still contains a
`with self.swarm.access():` block and a nearest-neighbour fallback pattern).
Related dead surface on the same method: `Swarm.migrate(remove_sent_points=...)`
accepts the parameter and ignores it — every internal `dm.migrate` call hardcodes
`remove_sent_points=True`.

### The problem

Dead code containing a live trap: if anything revives `_rbf_reduce_to_meshVar`
(or calls `_get_map` directly), it silently consumes particle indices from an
arbitrarily old population — the map is keyed only by where you *ask*, not what
the particles *are*. The kd-tree consulted inside is at least token-validated
after {ref}`swarm-design-cache`; the map dictionary is not.

### Target design

Delete the trio: `Swarm._get_map`, `Swarm._nnmapdict`,
`SwarmVariable._rbf_reduce_to_meshVar` (git history preserves them; a
nearest-particle *reduction* proxy mode, if ever wanted, is rebuilt against the
generation token from {ref}`swarm-design-cache` so the cache is
correct-by-construction). Also resolve `remove_sent_points`: honour it (thread
through to `dm.migrate`) or remove it with a one-cycle keyword shim — a dead
parameter is speculative generality (Charter §5). Recommendation: remove; no
caller passes `False` anywhere in `src/`, `tests/`, or `docs/`.

### Migration path and tests

Behaviour-neutral deletion (Wave-A protocol): full `tier_a` identical pre/post;
maintainer sign-off on the batch. The `remove_sent_points` shim, if removal is
chosen, ships with the standard two-test deprecation pattern.

(swarm-design-kdtree)=
## 6. KDTree copy-in-`__cinit__` (SWARM-20)

### Current behaviour

`KDTree.__cinit__` (`src/underworld3/ckdtree.pyx`) stores the points array as a
**no-copy memoryview** (`self.points = points`) and hands nanoflann a raw pointer
into that buffer; the index is built once. Any later in-place mutation of the
source array leaves the index topology (split planes frozen at build time) reading
*mutated* coordinate values: queries are internally inconsistent, agreeing with
neither the build-time snapshot nor the current data. *(probe-verified at
`3184a406`: after building a tree over 500 points and mutating the array in place,
1000 NN queries disagreed with the current-data truth in 940 cases and with the
build-time truth in 980 cases, with distance errors up to 0.21 on a unit square.)*

Every cached-tree consumer therefore depends on perfect invalidation discipline.
The known holes are closed (#313), and the token design in
{ref}`swarm-design-cache` narrows the exposure further, but the failure mode when
discipline *does* break is the worst available: silently wrong, non-reproducible
interpolation.

One nuance: when the input carries units, `__cinit__` already routes through
`np.ascontiguousarray` — which may or may not copy. So immutability currently
depends on the input's unit-wrapping and memory layout: accidental, not designed.

### The problem / the question

Should `KDTree` copy its points at construction, making every tree
immutable-by-construction? A stale tree would then return *coherent build-time*
answers — still stale, still a bug, but a reproducible and diagnosable one instead
of the inconsistent mixture measured above.

Cost analysis: one `O(N·dim)` float64 copy per construction, against an
`O(N log N)` index build over the same data — the copy is asymptotically and
practically noise (the audit's own churn work, "Eliminate KDTree construction
churn in solver loop", already made construction rare by caching trees; the copy
cost lands only where a tree is genuinely rebuilt). Several call sites already
copy defensively before constructing (e.g. `Swarm._route_by_nearest_centroid`
copies the coordinate field) — those copies become redundant and can be dropped,
recovering part of the cost.

### Target design

Copy unconditionally in `__cinit__`
(`points = np.ascontiguousarray(points_input, dtype=np.float64).copy()` semantics —
a single code path, which also deletes the units/no-units branch asymmetry).
No `copy=False` opt-out: an opt-out reintroduces the exact hazard for the one
caller who uses it (Charter §5, no speculative flags). Invalidation discipline
stays — staleness is still wrong — but its failure mode is defanged.

### Migration path and tests

1. One-line-scale change in `ckdtree.pyx` + rebuild (`rm -rf build/` per the
   stale-Cython rule).
2. The probe above becomes the regression test: build, mutate source in place,
   assert queries still agree with the **build-time** truth exactly.
3. Benchmark gate: populate/advect loop timing before/after (expected noise);
   `test_0006_memory_leak` (transient trees now own their memory — confirm no
   growth in the solver loop).

(swarm-design-checkpoint)=
## 7. Checkpoint/restore fidelity audit

The audit deferred round-trip fidelity to this document (its Known Limitations).
State of the save/read contract at `3184a406`, all verified in code:

### Current behaviour

**Coordinates — `Swarm.save`** (called by `Swarm.write_timestep`): both the
parallel-HDF5 branch and the sequential fallback write **model-unit** coordinates
from `_particle_coordinates.data` (BF-17 fixed the sequential branch, which
previously wrote physically-scaled `points` — checkpoints differed by the length
scale between IO branches and could not round-trip). Unit metadata (JSON attrs on
the `coordinates` dataset) is appended by rank 0 afterwards.

**Coordinates — `Swarm.read_timestep(migrate=True)`**: **keep-local restore**
(maintainer ruling on #329): every rank reads the full coordinate dataset with
plain h5py and `add_particles_with_coordinates` keeps only locally-owned points.
Zero communication, no rank-0 memory hotspot; the cost is np-fold read
amplification, which striped parallel filesystems absorb. The
`migrate=False` escape hatch gives every rank a full copy and preserves
out-of-domain points (debugging, mesh-adaptation transfers). The pre-#329
behaviour — read everywhere + `add_particles_with_global_coordinates` + migrate —
restored one copy of the swarm *per rank* (#324), because migration is a scatter
with no deduplication; the honest low-level contract is now stated in the
`add_particles_with_global_coordinates` docstring.

**Values — `SwarmVariable.read_timestep`**: **rank-0-routed restore** (design from
commit 9fb198d0): rank 0 reads saved coordinates *and* values, stages them on a
transient routing swarm (empty arrays elsewhere), migrates it with the same rule
the live swarm uses (so a saved point and the live particle at the same coordinate
land on the same rank), then transfers values by rank-local nearest-neighbour
(`rbf_interpolator_local` with `nnn=1`). A rank on which no saved points land
warns and leaves its particles unmodified.

**The #333 companion question** (open issue): the two restore paths now embody two
different philosophies — keep-local for the swarm, rank-0-routed for variables.
Unifying variables onto keep-local would give one documented strategy and remove
the all-to-one hop, at the price of transient per-rank memory equal to the full
dataset (values are wider than coordinates; chunked filtering may be needed at
scale). NFS-style filesystems remain the case where rank-0 reads win. This is a
maintainer decision ({ref}`swarm-design-questions`), not blocked on anything in
this document.

**Metadata race (#330)**: `Swarm.save` returns on non-zero ranks while rank 0 is
still appending the JSON metadata; an immediate reopen (write-then-read, exactly
what `write_timestep` → `read_timestep` sequences do) hits HDF5 file locking
(`BlockingIOError`, errno 35). A `TODO(BUG)` marks the site; the fix is a trailing
barrier (or moving the metadata append before the collective close).
`SwarmVariable.save` has the same shape (rank-0 metadata append after the
collective write) and needs the same barrier.

**In-memory snapshots** (`snapshot_payload` / `apply_snapshot_payload`,
design in `docs/developer/design/in_memory_checkpoint_design.md`): per-rank
capture of coordinates + user-variable arrays; restore clears and rebuilds the
local population via raw PETSc primitives, then writes values **through the
canonical array** so the PETSc pack fires (#313 fixed a restore that wrote into a
detached view and only appeared to work because of the SWARM-01 bug). The
variable-set contract is symmetric and loud: extra or missing variables raise
`SnapshotInvalidatedError`.

### Residual fidelity gaps (the audit part)

1. **#330** — the reopen race, both `save` methods. Small, decided, unshipped.
2. **Restore is nearest-neighbour, not identity** — `SwarmVariable.read_timestep`
   maps saved values onto the *current* particle layout via `nnn=1` interpolation.
   Restoring onto the exact saved swarm reproduces values through co-location
   (same migration rule ⇒ same rank ⇒ distance-zero neighbour); restoring onto a
   *different* population silently interpolates. That is a feature (it tolerates
   repopulation) but an undocumented one — the contract should be stated in the
   docstring and subsystem doc.
3. **Units metadata is written but never read** — neither `read_timestep` path
   checks the saved `coordinate_units` / `variable_units` attributes against the
   live model; a restore into a model with different reference quantities
   silently misinterprets model-unit values. Target: warn on mismatch (read side),
   as the metadata was designed to allow.
4. **No round-trip test matrix** — the pieces are individually tested
   (`test_0114_swarm_save_coordinate_units`, `tests/parallel/test_0757`,
   `test_0003_save_load`), but there is no systematic serial/np2/np4 ×
   `migrate=True/False` × scaled/unscaled matrix. The FO-01 implementation should
   add it (`level_2`, `tier_a` for the serial cells, np-gated for the rest).

(swarm-design-fossil)=
## 8. The fossil contract (#313)

### The designed behaviour

`SwarmVariable` holds its parent swarm by **weak reference** (`_swarm_ref`,
accessed through the `SwarmVariable.swarm` property). A strong back-reference
would cycle with the swarm's strong `_coord_var`/`_X0` members and defer DMSwarm
destruction from refcount-immediate to garbage-collector time — the
transient-evaluation-swarm leak guarded by `tests/test_0006_memory_leak.py`.

Consequently a variable can outlive its swarm. #313 made this a **stated
contract** rather than an accident:

- **Symbolic reads survive**: a variable whose swarm has been collected is a
  *symbolic fossil* — its proxy MeshVariable keeps the last projection.
  `_update_proxy_if_stale` (base and the `IndexSwarmVariable` override) checks the
  weakref *before* any swarm access, warns once ("the parent swarm no longer
  exists; the proxy retains its last projection"), and pins `_proxy_stale = False`
  so nothing tries again. `.sym` keeps returning the last-projected field.
- **Particle-data reads fail loudly**: `.data`, `rbf_interpolate`, and every other
  path that needs live particles goes through the `swarm` property, which raises
  `RuntimeError` ("Variables cannot outlive their parent swarm") on a dead ref.

This split is deliberate: solver expressions capture proxy syms at construction,
and a captured expression must remain *evaluable* (with honestly-stale data and a
warning) after a transient swarm dies — whereas particle data genuinely no longer
exists.

### How the new cache design preserves it

The validation token in {ref}`swarm-design-cache` reads
`self.swarm._population_generation` — a live-swarm access. The ordering rule,
stated here so the implementation cannot get it wrong:

1. `SwarmVariable.data` on a fossil must keep raising via the `swarm` property —
   the token check therefore performs its swarm access *through the property*
   (never through `_swarm_ref()` directly with a silent fallback), so the fossil
   behaviour of `.data` is unchanged by construction.
2. Proxy freshness checks must keep short-circuiting on the weakref **before**
   any token logic — `_update_proxy_if_stale` already does; any new solve-entry
   or view-level freshness hook added by this design must follow the same order.
3. `Swarm._sync_before_assembly` iterates `self._vars` (the swarm's own registry),
   so fossils are unreachable from the solve path — no change needed, noted as an
   invariant.

A regression test pins the contract: delete a swarm, assert `.sym` warns and
returns, assert `.data` raises, assert a captured `Projection` on the fossil proxy
still solves (with the stale field).

## Non-goals

Explicitly out of scope for the modernization, and why:

- **No PIC/cell-DM swarm**: `swarms/pic_swarm.py` was deleted and `recycle_rate > 1`
  excised (ruling D4). Streak/recycle functionality returns only as a fresh,
  tested feature proposal — not by resurrection.
- **No change to the `.data` flat shape** `(N, num_components)` or its
  units-bypass role — it is the sanctioned non-dimensional transfer path
  (Charter §7). `.array` remains the user interface.
- **No per-write migration, ever** — ruled out (D12); every design here keeps
  migration at collective points.
- **No change to the RBF weighting scheme itself** (inverse-distance, `nnn`
  neighbours) — {ref}`swarm-design-rbf-seams` changes only what data feeds it.
- **Solver access patterns unchanged**: solvers keep reading proxy DMs directly
  via `vec`/`update_lvec`; freshness stays consumption-enforced.
- **`NodalPointSwarm` is not modernized** — deprecated (#323, ruling D5), removal
  next cycle; its remaining internal `access()` uses die with it.
- **Units semantics in views are consolidated, not redesigned** — governed by
  `docs/developer/design/UNITS_SIMPLIFIED_DESIGN_2025-11.md`.
- **#314 (`global_evaluate` np4 empty-rank hang)** is tracked and fixed on the
  evaluation path, not here.

(swarm-design-phasing)=
## Phasing plan

Ordered by dependency; each phase is a separately reviewable PR (or small set).

**Phase 0 — FO-02, can ship now, independent of everything else**

- `IndexSwarmVariable.sym_1d` gains the staleness check (it currently returns
  `_MaskArray` with no `_update_proxy_if_stale()`, unlike `sym` — and unlike the
  *base* `SwarmVariable.sym_1d`, which does check).
- Vectorize the per-particle projection loop in
  `IndexSwarmVariable._update_proxy_variables` (`np.add.at` form), gated on
  bit-identical output against the Python loop on a random material distribution.
- **Lazification of `IndexSwarmVariable._update()` is now unblocked**: the audit
  blocked it on BF-08, which landed in #313 (`_sync_before_assembly` refreshes
  index proxies at solve entry — the `IndexSwarmVariable._update_proxy_if_stale`
  override exists precisely for this). Today every write to an index variable
  triggers a full eager re-projection *from inside the pack path*
  (`pack_raw_data_to_petsc` calls `self._update()`, which for the index class is
  eager) and the canonical callback then marks the proxy stale anyway — the work
  is discarded and repeated at the next `.sym` touch. Lazifying deletes that
  double cost and removes the special case in `_invalidate_canonical_data`.
  It is nevertheless a behavioural change for any consumer reading `_MaskArray`
  without `.sym` outside a solve (e.g. expressions built by `createMask` and
  evaluated through paths that skip `update_lvec`) — so it ships as its **own PR**
  with material-mask time-loop regression tests, per maintainer timing decision
  ({ref}`swarm-design-questions`).

**Phase 1 — cache self-validation (SWARM-10) + small deletions**

- The token check, `_mutate_population()` wrapper, tests
  ({ref}`swarm-design-cache`). Subsumes nothing that exists — all Track-0
  invalidations stay.
- Rides along (Wave-A protocol, maintainer sign-off): delete `_get_map` /
  `_nnmapdict` / `_rbf_reduce_to_meshVar` ({ref}`swarm-design-getmap`), the dead
  `_array_cache` sites, `SwarmVariable.old_data`, the commented-out pack/unpack
  blocks, and — decision pending — the ignored `remove_sent_points` parameter.
  Also the SWARM-21 leftovers (unconditional `advection()` prints gated on
  `self.verbose`; dead `corrector`/`evalf` parameters via deprecation shim).

**Phase 2 — shared array views (SWARM-14)**

- After Phase 1 (the view cache/validation story references the token). Mesh
  first, then swarm ({ref}`swarm-design-shared-views`). After Wave B lands
  (same-file churn discipline).

**Phase 3 — migration state machine (SWARM-03/18 consolidation)**

- After the maintainer rules on questions 2–4. Includes the `points`-setter
  deferral, the exit-skip removal, and the `disable=True` semantics
  ({ref}`swarm-design-migration-matrix`).

**Phase 4 — KDTree copy + RBF halo exchange**

- KDTree copy ({ref}`swarm-design-kdtree`) is small and independent — it can ship
  any time after its benchmark gate; listed here only because its regression test
  supersedes discipline assumptions from earlier phases.
- The halo exchange ({ref}`swarm-design-rbf-seams`) is the largest single item:
  measurement test first, then implementation, then default-on.

**Checkpoint items — independent track**

- #330 barrier fix: small, decided, ship immediately (both `save` methods).
- #333 restore unification: after the maintainer ruling; then the round-trip test
  matrix and the units-metadata read-side check
  ({ref}`swarm-design-checkpoint`).

**The subsystem doc** (`docs/developer/subsystems/swarm-system.md`) is rewritten
as the *final* FO-01 deliverable once Phases 1–3 have landed, replacing the WE-11
interim banner: pipeline, trigger matrix, cache contract, checkpoint contract,
fossil contract — the by-then-true versions of the sections above.

## Test strategy

Campaign ground rules apply to every phase: `pytest -m "level_1 and tier_a"`
green pre/post; `tier_a or tier_b` before merge; np2 **and** np4 runs of the
parallel swarm suite (`tests/parallel/test_0755`–`test_0757`, `test_0765`,
`test_0766`, plus each phase's new tests) for anything touching swarm, migration,
or partition-sensitive code. `test_0760` hangs at np4 on the current baseline
(#314, pre-existing) — exclude it from np4 gates until that issue closes, and do
not interpret its hang as a phase regression.

Per-phase regression tests are named in each section. Tier/level assignments:

| Test family | Level | Tier |
|---|---|---|
| Cache-token validation (serial; bare-mutation self-healing) | 1 | A |
| Cache-token np2/np4 (balanced-exchange generation case) | 1 | A |
| Migration-matrix writer/consumer np2 suite | 1 | A |
| Shared-view bit-identical port gates (reductions, units pipeline) | 1 | A |
| KDTree immutability probe-as-test | 1 | A |
| Fossil-contract pin (warn/raise/solve split) | 1 | A |
| Index-variable lazification time-loop masks | 2 | A |
| RBF seam np-consistency (tolerance-based) | 2 | B → A once the exchange is default-on |
| Checkpoint round-trip matrix (× migrate × scaling × np) | 2 | A serial cells; parallel cells gated |

Behaviour-neutral deletions (Phase 1 riders) carry no new tests; their gate is
`tier_a` identical pre/post plus retirement of the tests that pinned the deleted
stubs (`test_0530`).

(swarm-design-questions)=
## Open questions for the maintainer

1. **KDTree copy (SWARM-20)**: approve the unconditional copy in
   `KDTree.__cinit__` (recommended — no opt-out flag), accepting one `O(N·dim)`
   copy per tree construction?
2. **`migration_control(disable=True)` semantics**: when coordinates are written
   inside a disabled context, should the deferred migration (a) be *held* past
   solve entry until the user migrates explicitly (risking wrong-rank particles
   inside a solve), or (b) keep the current behaviour — suppressed only until the
   next solve entry — and document it as "migration cannot be suppressed across
   an assembly"?
3. **SWARM-18 exit skip**: approve removing the `local_size`-changed guard so
   `migration_control(disable=False)` always migrates at exit as its docstring
   promises?
4. **Deprecated `points` setter**: route its per-write migration onto the
   deferred flag now (a behaviour change inside a deprecated path, removing the
   last per-write collective), or leave it untouched until the setter is removed?
5. **`Swarm.migrate(remove_sent_points=...)`**: the parameter is accepted and
   ignored — remove it (one-cycle keyword shim, recommended) or honour it?
6. **Restore-strategy unification (#333)**: move `SwarmVariable.read_timestep`
   from rank-0-routed to keep-local to match `Swarm.read_timestep`, accepting
   transient full-dataset memory per rank (with chunking as the scale escape
   hatch), or keep two documented strategies?
7. **IndexSwarmVariable lazification timing**: ship it with Phase 0 (FO-02) now
   that BF-08 unblocks it, or hold it to Phase 1 so it lands behind the cache
   token?
8. **Shared-view module home**: `discretisation/variable_layout.py` +
   `variable_views.py` as proposed, or a single `utilities/` module? (Naming only,
   but it fixes the import direction between `swarm.py` and `discretisation/`.)
9. **`test_0530` retirement**: sign off deleting the
   `use_legacy_array`/`use_enhanced_array` no-op stubs together with the test that
   pins them (Phase 1 rider)?
10. **RBF seam target**: approve the halo-exchange design as the goal (vs
    documenting rank-local interpolation as the permanent contract), with the
    measurement test shipped first either way?

---

*Underworld development team with AI support from Claude Code.*
