# Swarm Subsystem Review — July 2026 Quality Campaign (Dimension 5)

**Status**: audit complete; findings adversarially verified 2026-07-03 (where marked)
**Base**: `development` @ `1d003481` (audit worktree, campaign index at `e848d131`)
**Scope**: `src/underworld3/swarm.py` (4,996 lines), `src/underworld3/swarms/pic_swarm.py`
(1,534 lines, uninstalled), the swarm-facing parts of `src/underworld3/systems/ddt.py`,
`src/underworld3/ckdtree.pyx`, and `docs/developer/subsystems/swarm-system.md`.

Every `file:line` cited below was read directly in this worktree; line numbers are
exact at `1d003481`. Findings SWARM-01 … SWARM-14 passed the full adversarial
verification pass (including two independent runtime reproductions in this
worktree's built environment); SWARM-15 … SWARM-24 had their evidence lines
personally read but did not go through the adversarial pass. Sub-claims refuted
during verification are recorded in the appendix so the same false leads are not
re-found.

## Overview

The swarm subsystem is the single densest cluster of correctness risk found by
this campaign, and the risk has one common shape: **caches and proxies that are
invalidated by call-site discipline rather than by self-validation**, in a module
where the discipline has already been broken at least four separate times.

The subsystem's data pipeline (PETSc DMSwarm field → cached canonical NumPy copy
→ per-access array view → RBF-projected proxy MeshVariable → solver JIT) has
*five* hand-maintained consistency contracts: pack-to-PETSc on write, cache
invalidation on layout change, kd-tree invalidation on coordinate change, proxy
staleness on data change, and migration on coordinate change. Each contract has
at least one confirmed hole at this baseline:

- `Swarm.migrate()` skips **all** invalidation when no particle changes rank —
  the *common* case in serial and for small in-domain moves — leaving stale
  canonical arrays, a stale kd-tree over a mutated buffer, and fresh-looking
  proxies (SWARM-01/02; runtime-reproduced: after adding 2 particles,
  `dm.getLocalSize()` = 398 but `var.data.shape[0]` = 396).
- Writing particle coordinates through the modern API (`swarm.coords` setter or
  `_particle_coordinates.data`) never migrates at all, despite the class
  docstring promising automatic migration (SWARM-03).
- Inside `migration_disabled()` — the context the docstrings *recommend* for
  batch writes — variable writes are silently discarded before ever reaching
  PETSc (SWARM-04), and the parallel test that covers this pattern
  (`ptest_0755`) passes vacuously.
- Solvers that consume a proxy without touching `.sym` read stale proxy data —
  the known issue #215, still open, `TODO(BUG)` at `swarm.py:1075` (SWARM-05).

Beyond the cache/staleness cluster: user-configured `Lagrangian`/`Lagrangian_Swarm`
history terms crash on first use (runtime-reproduced AttributeError, SWARM-06);
empty MPI ranks get silent zeros or a hard crash from RBF proxy updates
(SWARM-07); and the advertised streak-swarm feature (`recycle_rate > 1`) is dead
at two independent `NameError`s (SWARM-08), with the only working copy of that
logic stranded in a 1,534-line module that is never installed (SWARM-09).

The #216 fixes (commit `af537d56`) were correct but treated symptoms: they added
invalidation calls at three call sites while the underlying single-cache-no-
validation design remained. The highest-leverage remediation is architectural —
make the canonical cache self-validating the way mesh variables already are
(SWARM-10) — with the individual holes patched as fast, testable interim fixes.

## Changes Made

None — audit only. Proposed fixes are classified per finding below and feed the
campaign remediation waves: the SWARM-01/02/04 invalidation holes and SWARM-06
crash belong to the bug-fix track; SWARM-08/09/11 dead code to Wave A; the 13
internal `access()` sites (SWARM-13) to Wave B; SWARM-10/14 to the swarm
modernization design doc that is this dimension's follow-on deliverable.

## Findings — verified (adversarial pass complete)

| ID | Sev | Effort | Location | Finding |
|----|-----|--------|----------|---------|
| SWARM-01 | High | S | `swarm.py:3451` | `migrate()` early-returns when no particles are globally unclaimed *without* invalidating caches; `add_particles_with_global_coordinates` relies on migrate for invalidation → stale wrong-sized `.data` in serial (runtime-confirmed) |
| SWARM-02 | High | S | `swarm.py:3451` | Same early return leaves the cached kd-tree built over a mutated coordinate buffer → internally inconsistent NN queries feeding `rbf_interpolate` and proxy updates |
| SWARM-03 | High | M | `swarm.py:3114` | Coordinate writes via `swarm.coords` setter / `_particle_coordinates.data` never trigger migration, contradicting the class docstring → silent wrong-rank particles in parallel |
| SWARM-04 | High | M | `swarm.py:480` | Variable/coordinate writes inside `migration_disabled()` are silently discarded (callbacks early-return, nothing re-packs, invalidation destroys the only copy); `ptest_0755` asserts vacuously |
| SWARM-05 | High | M | `swarm.py:1075` | Stale-proxy hole (issue #215): `_update()` only sets `_proxy_stale`; refresh fires only via `.sym` — solvers reading the proxy DM directly consume stale data |
| SWARM-06 | High | M | `systems/ddt.py:3197` | `Lagrangian`/`Lagrangian_Swarm` history crashes with AttributeError on first update (runtime-confirmed): `psi_star_0[i, j].data` — `__getitem__` returns a sympy function with no `.data` |
| SWARM-07 | High | M | `swarm.py:1457` | Empty/starved ranks: `rbf_interpolate` silently writes zeros into proxies at ≤1 local particles; unguarded KDTree construction on 0 particles raises IndexError (empirically confirmed) → MPI abort/hang |
| SWARM-08 | Med | M | `swarm.py:3360` | Streak/recycle swarm (`recycle_rate > 1`, docstring-advertised) is dead: `populate()` NameError on `all_local_cells` (runtime-confirmed); `advection()` NameError on `cellid` (`getField` commented out at 4733, use at 4744). Zero tests |
| SWARM-09 | Med | S | `swarms/pic_swarm.py:22` | Entire 1,534-line module is dead: no `__init__.py`, never installed, broken relative import, sole importer commented out (`swarm.py:2450`); already diverging from `Swarm` (step_limit default, missing #216 fixes) |
| SWARM-10 | Med | M | `swarm.py:1546` | Architecture: `SwarmVariable.data` trusts the cached `_canonical_data` with no size/generation revalidation — the root cause behind #216 and SWARM-01; mesh variables already self-validate (`discretisation_mesh_variables.py:2721-2727`) |
| SWARM-11 | Med | S | `swarm.py:4885` | `NodalPointSwarm.__init__` passes `verbose` positionally into the `recycle_rate` slot (`Swarm.__init__` signature at 2523); class also has zero remaining instantiation sites (~130 lines dead public surface) |
| SWARM-12 | Med | M | `swarm.py:2176` | `IndexSwarmVariable` breaks the lazy-`_update()` contract (eager re-projection, special-cased in `_invalidate_canonical_data` at 2699-2707 to avoid an O(100 MiB) leak); `sym_1d` (2209-2226) returns `_MaskArray` with **no** staleness check. Eagerness is load-bearing — see Known Limitations |
| SWARM-13 | Med | L | `swarm.py:4458` | Deprecated-pattern debt: 13 live internal `with …access(…)` sites in `swarm.py` + 7 in `ddt.py`; `_legacy_access` (4317-4456) has zero callers; the style guide's "Preferred" swarm example doesn't run (getter-only property → AttributeError) |
| SWARM-14 | Med | L | `swarm.py:1592` | `.array` creates a fresh closure-defined view class + instance per access (~415 lines at 529-944; tensor reads do a full PETSc unpack copy each time); `_data_layout` exists in three near-copies (`swarm.py:983`, `swarm.py:4510` — "cut'n'pasted from the MeshVariable class" — `discretisation_mesh_variables.py:1654`). Mesh side shares the same design (see appendix, R-2) |

### SWARM-01 — `migrate()` no-move early return skips cache invalidation

`Swarm.migrate()` returns at `swarm.py:3451-3452` (`if global_unclaimed_points
== 0: return`) with the only `_invalidate_canonical_data()` call on the
fall-through path at 3525. `add_particles_with_global_coordinates` (def 3644)
calls `addNPoints` at 3706 — changing the DM local size — and its only
invalidation route is `if migrate: self.migrate(...)` at 3729-3730; with
`migrate=False` nothing ever invalidates. In serial (or whenever all points are
already claimed) the early return fires, so every SwarmVariable's cached `.data`
keeps the old particle count. Runtime-reproduced on this worktree's build:
`dm.getLocalSize()` = 398 vs `var.data.shape[0]` = 396 after adding 2 in-domain
points. New particles are invisible to reads; writes pack wrong-sized arrays.
The sibling `add_particles_with_coordinates` invalidates explicitly at 3635
(part of #216), confirming the asymmetry — the same bug class as the three #216
fixes, one path missed.

**Fix (S)**: move `_invalidate_canonical_data()` ahead of the early return; add
an explicit invalidation in `add_particles_with_global_coordinates` after
`addNPoints` (mirroring 3635). Serial regression test. Subsumed longer-term by
SWARM-10.

### SWARM-02 — same early return leaves a poisoned kd-tree

Advection writes `_particle_coordinates.data` in place (4674/4705); the
coordinate variable is a plain SwarmVariable (2588) whose canonical-data
callback (472-501) only syncs to PETSc — nothing drops `swarm._kdtree` (cleared
only at 2710, inside `_invalidate_canonical_data`). `KDTree` (`ckdtree.pyx:103-104`)
stores a **no-copy memoryview** and builds its nanoflann index once, so after an
in-place coordinate mutation followed by a no-move `migrate()` (serial, or any
parallel step where no particle changes rank — the common case), the cached tree's
topology reflects old positions while its stored points show new ones: queries are
internally inconsistent, not merely "frozen". `_get_kdtree` (2712-2723) returns it
unconditionally; `rbf_interpolate` consumes it at 1476. Silently wrong NN
interpolation into every proxy.

**Fix (S)**: same invalidation move as SWARM-01 (at minimum `self._kdtree = None`
+ proxy-stale marking before the early return). Regression test: build tree,
advect in-domain, assert `rbf_interpolate` reflects new positions.

### SWARM-03 — modern coordinate writes never migrate

The `Swarm` docstring (2510-2513) promises automatic migration when particles
move. Reality: `swarm.coords` setter (3114) routes to
`canonical_data_callback` (472-505) → `pack_raw_data_to_petsc` (1306-1349),
which packs the field and updates proxies but never migrates (its `sync=True`
branch is literally `pass  # TODO`). Only the *deprecated* `points` property
migrates (2948-2951); the `access()` shim (4458+) drops the migrate-on-exit that
`_legacy_access` performed (4427-4430); internal advection compensates with an
explicit `self.migrate()` at 4806. In parallel, user code that moves particles
through the supported API leaves them on the wrong rank — silent corruption.
Nuance: `migration_control()` (deferred mode) and `dont_clip_to_mesh()` do
migrate on exit (3275-3280 / 3209), but plain direct writes do not.

**Fix (M)**: post-write migrate hook on the coordinate variable — but
**deferred/context-exit form**, not per-write: `migrate()` is collective, so a
hook firing from a write callback risks MPI deadlock when ranks write unevenly.
Extend `migration_control` semantics into the coords setter and the `access()`
shim. Add an np2 test moving particles across the partition via
`._particle_coordinates.data`.

### SWARM-04 — writes inside `migration_disabled()` silently discarded

Both sync callbacks early-return while `_migration_disabled` is set
(`swarm.py:429-430`, `480-481`); every user write path funnels through them
(`.data` → canonical NDArray_With_Callback 1546-1550; `.array` setter → 1607;
`points` in-place → 2946). Nothing re-packs afterwards. `migrate()` reads
coordinates from the PETSc field (3428) — i.e. the *stale* ones — and its
trailing `_invalidate_canonical_data()` (3525) nulls the canonical copy,
destroying the only record of the writes. The docstrings (~3220-3253) explicitly
recommend writing inside the context; `tests/parallel/test_0755_swarm_global_stats.py:322-329`
does exactly that and asserts vacuously (its perturbation never reaches the
DMSwarm). Note: outright data destruction requires migrate to proceed past the
SWARM-01 early return; with stale in-domain coords the canonical copy survives
but PETSc silently never sees the writes — wrong either way.

**Fix (M)**: separate "suppress migration" from "suppress PETSc sync" — pack
while the flag is set, gate only the `migrate()` call; or accumulate a dirty
flag and flush pending canonical arrays in `_MigrationControlContext.__exit__`
before migrating. Fix `ptest_0755` to actually assert the values survive.

### SWARM-05 — stale proxy consumed by solvers (issue #215)

`_update()` (1060-1073) only sets `_proxy_stale`; `_update_proxy_if_stale()` is
invoked solely from `.sym`/`.sym_1d` (1627/1643; IndexSwarmVariable inline at
2203-2205) — a repo-wide grep finds no other callers. The solve path has no
enforcement: `mesh.update_lvec()` (`discretisation_mesh.py:2924-2960`) pulls the
proxy's vec directly into assembly; `_jitextension.py` has zero swarm/proxy
handling; `Projection` captures `uw_function` as a sympy Matrix at construction
(`solvers.py:2636/2667`) so `solve()` never re-touches `.sym`. Write
`material.data` → solve → stale proxy data. `write_proxy()` (1960-1966) has the
same hole. The unresolved `TODO(BUG)` at 1075-1080 documents it.

**Fix (M)**: enforce freshness at consumption — pre-solve/JIT walk for
swarm-proxy UnderworldFunctions calling `_update_proxy_if_stale()`, and/or have
the proxy MeshVariable's `.data`/vec access consult the owner's `_proxy_stale`.
Any hook must respect the `_updating_proxy` reentrancy guard
(`_rbf_to_meshVar` writes `meshVar.data[...]`, 1132). Regression test: write
`material.data`, solve a Projection without touching `.sym`, assert freshness.

### SWARM-06 — Lagrangian ddt history terms crash on first update

`ddt.py:3197, 3281, 3549, 3638-3639` all do `psi_star_0[i, j].data[:] = ...`
where `psi_star_0` is a SwarmVariable. `MathematicalMixin.__getitem__`
(`mathematical_mixin.py:74-109`) returns `sym[index]` — an
UnderworldAppliedFunction (`_function.pyx:51`) with no `.data`. Runtime-confirmed:
`Lagrangian.initialise_history()` raises AttributeError at exactly `ddt.py:3197`.
The default `DuDt` is SemiLagrangian, so only user-configured
`Lagrangian`/`Lagrangian_Swarm` histories hit it — but those are advertised
(`systems/solver_template.py:92-93`, `solvers.py:196-197`). Coverage gap: the
only test (`test_0007`) round-trips `.state` without ever advecting. The blocks
also use deprecated `swarm.access()` / `swarm.data` internally.

**Fix (M)**: rewrite the component update via the modern interface
(`psi_star_0.data[:, psi_star_0._data_layout(i, j)]`, coordinates from
`swarm._particle_coordinates.data`); add a minimal advecting-history test for
both classes.

### SWARM-07 — empty ranks: silent zeros or hard crash

`rbf_interpolate` (`swarm.py:1457-1458`) returns `np.zeros(...)` whenever the
rank holds ≤1 particles; `_rbf_to_meshVar` (1130-1132) writes that straight into
the proxy — silent wrong nodal values on starved ranks, no warning. Two verbatim
TODOs ("some form of global fall-back…", 1110 and 1148) confirm the known gap.
Separately, `IndexSwarmVariable._update_proxy_variables` reaches unguarded
KDTree construction (2374 via `_get_kdtree`; direct at 2409), and
`KDTree(np.zeros((0,2)))` raises `IndexError` (empirically confirmed;
boundscheck is on — a clean crash, not memory corruption, see appendix R-1) —
one rank raising during a collective proxy update means MPI abort/hang.

**Fix (M)**: explicit empty-rank strategy — guard tree construction on
`local_size == 0`; for starved ranks either leave the proxy untouched + warn or
implement the documented global fallback. At minimum, warn loudly instead of
writing zeros.

### SWARM-08 / SWARM-09 — dead recycle feature and its stranded working copy

`populate()`'s `recycle_rate > 1` branch (starts 3338) uses `all_local_cells`
(3360) — the name occurs exactly once in the file, at its use site → NameError
for any recycle swarm (runtime-confirmed). `advection()`'s cycle block indexes
`cellid[swarm_size::]` (4744) while the only binding is commented out (4733).
The feature is docstring-advertised (2483, streak example 2502-2503),
propagated by `adaptivity.py:710`, and has zero tests. The only *working* copy
of the recycle logic lives in `swarms/pic_swarm.py` — which is never installed
(no `__init__.py`; `find_packages()` excludes it; confirmed absent from
site-packages), has a broken import (`from .swarm` at line 22 — no such sibling),
and is already divergent (step_limit default True at 1086 vs False at
`swarm.py:4567`; #216 invalidation fixes absent; stray `print("Peace")` at 1169).

**Fix (M/S)**: decide once — either port the working recycle implementation into
`Swarm` (using `mesh.get_closest_local_cells()` instead of `DMSwarm_cellid`,
which doesn't exist on a BASIC swarm) and add a `level_1` streak test, or excise
`recycle_rate` machinery and the docstring claim. Either way delete
`swarms/pic_swarm.py` (git history preserves it) and the breadcrumbs at
`swarm.py:70, 2449-2455`. Do not leave it advertised and crashing.

### SWARM-10 — the architectural root: no self-validating cache

`SwarmVariable.data` (1546-1550) returns the cached `_canonical_data` on a bare
existence check. The mesh side already solved this: `_BaseMeshVariable.data`
self-validates against `id(self._lvec)` (`discretisation_mesh_variables.py:2721-2727`,
documented in `docs/developer/subsystems/data-access.md:131-150`). The swarm
equivalent — compare cached row count against `dm.getLocalSize()` plus a
`_population_generation` stamp (the counter exists at 2555 and is already bumped
at 3382, 3416, 3639, 3710, 4273, 4731; its own comment invites this use) — would
have prevented #216 and SWARM-01 by construction. Honest caveat: it cannot make
`_invalidate_canonical_data()` purely optional — same-local-size migrations swap
equal counts between ranks with values changed, and bare `dm.migrate` sites
(e.g. `_route_by_nearest_centroid`, 2754-2756) don't bump the generation — so
the refactor reduces the discipline to "bump the generation at every
layout-mutating site" rather than eliminating it. Still the single
highest-leverage swarm-modernization refactor.

### SWARM-11 — `NodalPointSwarm` positional-argument bug, dead class

`swarm.py:4885`: `super().__init__(mesh, verbose, clip_to_mesh=False)` against
`Swarm.__init__(self, mesh, recycle_rate=0, verbose=False, clip_to_mesh=True)`
(2523) — `verbose` lands in `recycle_rate`: silently discarded, and a truthy
value enables (broken, per SWARM-08) recycling. Zero instantiation sites remain
(SemiLagrangian dropped its nodal-swarm cache, `ddt.py:1683`); the constructor
also uses field arrays after `restoreField`/around `migrate` (~4946-4952).
**Fix (S)**: keyword-explicit super call now; decide keep-with-smoke-test vs
deprecate in the modernization doc.

### SWARM-12 — IndexSwarmVariable contract inconsistency

Base `_update()` is lazy; `IndexSwarmVariable._update()` (2176-2182) eagerly
runs a pure-Python O(N × nnn) re-projection (2348-2446), forcing the
special-case in `_invalidate_canonical_data` (2699-2707, documenting the
O(100 MiB) leak it avoids). `sym_1d` (2209-2226) returns `_MaskArray` with no
staleness check despite being documented as an alias for `sym` (which checks,
2203-2206). **Safe fixes only** (see Known Limitations): route `sym_1d` through
the staleness check; vectorize the per-particle loop (`np.add.at`). Do **not**
lazify `_update()` until a pre-solve refresh hook (SWARM-05's fix) exists —
the eager calls after migration (2951, 3280, 4443; `adaptivity.py:818`) are the
only mechanism refreshing material masks in a time loop.

### SWARM-13 / SWARM-14 — modernization debt (Wave B / design doc)

Thirteen live internal `with …access(…)` sites in `swarm.py` (1161, 2381 —
which stacks mesh and swarm access — 3339, 3345, 3376, 3628, 3725, 4719, 4767,
4791, 4949, 4972, 4975) plus seven in `ddt.py`; `_legacy_access` (4317-4456) has
zero callers. The style guide's "Preferred" swarm example
(`UW3_Style_and_Patterns_Guide.md:206`) is `swarm.data += displacement` — which
raises AttributeError (`swarm.data` is getter-only; only `points` has a setter,
3022-3024): the documented replacement pattern cannot run at all. On `.array`:
both SwarmVariable (1592, view classes 529-944) and MeshVariable
(`discretisation_mesh_variables.py:2012`) rebuild closure-defined view classes
per access — a *shared-view* refactor workstream, not swarm-parity; tensor swarm
reads additionally pay a full PETSc unpack copy per access (1280-1295). Unify
the three `_data_layout` near-copies; remove the dead `_array_cache`
(`discretisation_mesh_variables.py:446, 1888` — assigned None, never read).

## Findings — unverified (evidence read, no adversarial pass)

| ID | Sev | Effort | Location | Finding |
|----|-----|--------|----------|---------|
| SWARM-15 | Med | L | `swarm.py:1476` | Proxy RBF interpolation is strictly rank-local (`rbf_interpolator_local`; no ghost-particle exchange / DMSwarmCollect anywhere) → proxy nodes near partition seams interpolate from one-sided neighbourhoods; results deterministic but np-count dependent |
| SWARM-16 | Med | M | `swarm.py:4649` | `advection` with `substeps > 1`: launch-point velocity is evaluated **locally** each substep but no migration happens between substeps (only at 4806) → particles that crossed a partition are evaluated on the wrong rank from substep 2 on; only the midpoint uses `global_evaluate` (4666) |
| SWARM-17 | Med | S | `swarm.py:3384` | `populate()` adds particles via `addNPoints` + direct field writes but never calls `_invalidate_canonical_data()` before returning — same stale-cache class as #216/SWARM-01 |
| SWARM-18 | Med | S | `swarm.py:3277` | `migration_control(disable=False).__exit__` silently skips the promised deferred migration if `local_size` changed inside the context; `add_particles_with_coordinates` (3632) does a raw `dm.migrate` that bypasses `_migration_disabled` entirely while `Swarm.migrate` honours it (3410) |
| SWARM-19 | Med | S | `swarm.py:3810` | `save()` writes different coordinate systems per IO path: parallel-HDF5 branch saves `_particle_coordinates.data` (model units, 3779) but the sequential fallback saves deprecated `self.points` (physically scaled, plus DeprecationWarning and possible migration side-effect) → checkpoints differ by the length scale depending on h5py MPI support |
| SWARM-20 | Med | S | `ckdtree.pyx:103` | `KDTree` stores a no-copy memoryview and indexes it once; any later in-place mutation of the source array leaves index and data inconsistent — every cached-tree consumer depends on perfect invalidation discipline (which SWARM-01/02 show is not upheld). Copying in `__cinit__` makes trees immutable-by-construction |
| SWARM-21 | Low | S | `swarm.py:4641` | Unconditional hot-path prints: `"Advection (2nd): …"` every substep from every rank (flush=True, not gated on `self.verbose`; contrast 4582); three more per substep on the order-1 path (4682-4700); `read_timestep` prints too (1988, 1994). Also dead signature params: `corrector`'s implementation is commented out (4598-4631) and `evalf` is referenced only inside that block |
| SWARM-22 | Low | S | `swarm.py:1235` | Dead API surface: the `sync=True` kwarg on all four pack/unpack methods is a no-op with four identical TODO stubs (1235, 1285, 1342, 1391) — DMSwarm fields are rank-local, nothing to sync; `use_legacy_array`/`use_enhanced_array` pass-through stubs (961-967, pinned only by test_0530); `_rbf_reduce_to_meshVar` (1136) zero callers; `old_data` "TESTING" leftover (1499); large commented-out blocks (1184-1200, 1242-1263) |
| SWARM-23 | Low | S | `swarm.py:4539` | `_get_map`/`_nnmapdict` cache keys nearest-particle indices only by coordinate hash and is never cleared by `migrate()`/`_invalidate_canonical_data()` (only by the dead legacy access manager, 4434) — a stale-cache trap if `_rbf_reduce_to_meshVar` is ever revived. `migrate()`'s `remove_sent_points` parameter is accepted but ignored (hardcoded True at 3477) |
| SWARM-24 | Low | M | `docs/developer/subsystems/swarm-system.md:7` | The subsystem's only architecture doc is a 26-line stub claiming "Well-Documented Subsystem … Priority Low … Integration with NDArray system complete" and citing a nonexistent `swarm/` module of "4,484 total lines" (reality: 4,996 + 1,534 dead) — it actively misdirects reviewers away from the subsystem this audit found most in need of attention |

Two unverified submissions duplicated verified findings and are subsumed:
the recycle-NameError claim (into SWARM-08, adding the `adaptivity.py:710`
propagation detail) and the dead-`pic_swarm.py` claim (into SWARM-09, adding
the missing-#216-fixes and `print("Peace")` details). Two low-severity
duplicate submissions (debug prints; sync-TODO no-ops) are merged into
SWARM-21/22.

## System Architecture

What this dimension's survey established about the swarm subsystem, for the
maintainer:

**The data pipeline.** A particle field lives in the PETSc DMSwarm. User access
goes through a single cached *canonical* NumPy copy per variable
(`NDArray_With_Callback`, created lazily at `swarm.py:439-505`): reads return
the cache; writes fire callbacks that pack back to PETSc
(`pack_raw_data_to_petsc`, 1306-1349) and mark the proxy stale. `.array` wraps
the same cache in a units/shape view built fresh on every access. Variables
with `proxy_degree > 0` own a proxy MeshVariable refreshed by rank-local RBF
interpolation over a cached kd-tree; solvers consume the proxy's DM directly.

**The consistency model is discipline, not validation.** Unlike mesh variables
— whose `.data` self-validates against the live PETSc vec identity — the swarm
canonical cache, the kd-tree, and the proxy staleness flag are all invalidated
only where someone remembered to call `_invalidate_canonical_data()` (its own
docstring, 2683-2691, says bare `dm.migrate` callers must do so manually). The
verified findings are all instances of forgotten or unreachable invalidation:
the `migrate()` no-move early return (SWARM-01/02), `populate()` (SWARM-17),
and the `migration_disabled` write-discard (SWARM-04). A
`_population_generation` counter exists and is bumped at most mutation sites
but is consumed by nothing — the self-validating design (SWARM-10) is half
built.

**Migration is opt-in in practice.** Despite the docstring's promise of
automatic migration, the trigger matrix at this baseline is: deprecated
`points` writes → migrate; `advection()` → explicit internal migrate;
`migration_control()` / `dont_clip_to_mesh()` → migrate on exit (the former
only if local size is unchanged); modern `coords` / `_particle_coordinates.data`
writes and the `access()` shim → **never**. The one route that migrates
automatically is the one being deprecated.

**Proxies are lazy on the read side only.** Data writes reliably mark proxies
stale, but refresh happens only via `.sym` — nothing on the solve path enforces
it (issue #215). `IndexSwarmVariable` bypasses laziness entirely with an eager,
pure-Python re-projection that is simultaneously a performance problem, a
special case in the invalidation path, and — because nothing else refreshes
material masks after migration — load-bearing.

**Parallel edges are unhandled.** RBF proxy updates are strictly rank-local
(np-dependent seams, SWARM-15), empty ranks produce zeros or a crash
(SWARM-07), and multi-substep advection evaluates launch velocities locally
without inter-substep migration (SWARM-16).

**Dead mass distorts the picture.** A never-installed 1,534-line near-duplicate
of `Swarm` (`swarms/pic_swarm.py`) holds the only working copy of the
advertised-but-broken recycle feature; `NodalPointSwarm` (~130 lines) has no
callers and a positional-argument bug; `_legacy_access`, `_rbf_reduce_to_meshVar`,
no-op `sync` kwargs, and dead signature parameters (`corrector`, `evalf`) pad
the API surface. The subsystem doc claims all of this is in good shape.

## Testing Instructions

Per the campaign ground rules, anything touching swarm/migration code gates on
`tier_a` green pre/post, `tier_a or tier_b` before merge, and np2/np4 parallel
tests. Run from inside the fix worktree after `./uw build`:

```bash
pytest -m "level_1 and tier_a" tests/          # quick gate
pytest -m "tier_a or tier_b" tests/            # pre-merge gate
mpirun -np 2 pytest tests/parallel/            # swarm/migration changes
mpirun -np 4 pytest tests/parallel/
```

Finding-specific validation (each fix PR should add the named regression test):

- **SWARM-01/02/17 (stale cache/kdtree)** — serial test: create swarm +
  variable, touch `var.data` (populate the cache), call
  `add_particles_with_global_coordinates` (both `migrate=True` and `False`) /
  `populate()`, assert `var.data.shape[0] == swarm.dm.getLocalSize()`; and:
  build the kd-tree, advect particles in-domain, assert `rbf_interpolate`
  reflects the new positions.
- **SWARM-03 (migration on write)** — np2 test: move particles across the
  partition via `swarm._particle_coordinates.data`, then assert re-ownership
  (each rank's particles inside its local domain; global count preserved).
- **SWARM-04 (`migration_disabled` writes)** — np2 test: write `var.data`
  inside `migration_control()`/`migration_disabled()`, exit, migrate, assert
  values survive. Repair `ptest_0755:322-329` so its perturbation demonstrably
  reaches the DMSwarm (currently vacuous — its passing must not be taken as
  coverage).
- **SWARM-05 (stale proxy)** — write `material.data`, run a `Projection` solve
  without touching `.sym`, assert the solve consumed fresh values.
- **SWARM-06 (Lagrangian ddt)** — minimal advecting-history test for
  `Lagrangian` and `Lagrangian_Swarm` (currently `test_0007` sets
  `_history_initialised` manually and never advects — do not rely on it).
- **SWARM-07 (empty ranks)** — np4 test with all particles seeded in one
  rank's subdomain; assert no zeros written into proxies on starved ranks and
  no crash from `IndexSwarmVariable._update_proxy_variables`.
- **SWARM-08 (recycle)** — if repaired: `level_1` smoke test
  `Swarm(recycle_rate=2)` → populate → advect. If excised: assert constructor
  raises/warns on `recycle_rate > 1`.
- **SWARM-16 (substep advection)** — np2 trajectory test with
  `step_limit=True` forcing substeps > 1 across a partition boundary, compared
  against the np1 trajectory.
- **SWARM-19 (save)** — with coordinate scaling active, write via both IO
  branches (`force_sequential=True`/`False`) and assert byte-identical particle
  coordinates on read-back.

Behavioral guards for fixes that must NOT change results: SWARM-12's
vectorization should be verified bit-identical against the Python loop on a
random material distribution; SWARM-21's print removal and SWARM-22's dead-API
deletions are behavior-neutral by construction (verify test_0530 is retired
together with the stubs it pins).

## Known Limitations

- **Line numbers are exact at `1d003481` only.** The campaign index commit
  (`e848d131`) does not touch `src/`; `swarm.py` is byte-identical between the
  two.
- **SWARM-15 … SWARM-24 are not adversarially verified.** Their evidence lines
  were personally read, but no independent re-derivation or runtime
  reproduction was performed; treat severity/mechanism as provisional until the
  fix PR reproduces each.
- **SWARM-12's obvious fix is out of scope.** Lazifying
  `IndexSwarmVariable._update()` would silently freeze material masks across
  advection (the eager calls at `swarm.py:2951, 3280, 4443` and
  `adaptivity.py:818` are the only refresh mechanism on the solve path today) —
  a solver-numerics change. It becomes safe only *after* SWARM-05's pre-solve
  staleness hook lands. Until then, only the `sym_1d` staleness check and the
  loop vectorization should ship, and the `_invalidate_canonical_data`
  special-case (2699-2707) must be kept.
- **SWARM-10 reduces but does not eliminate invalidation discipline.**
  Same-local-size migrations defeat a size check, and bare `dm.migrate` sites
  do not bump `_population_generation`; the generation bump must be pushed into
  every layout-mutating site (or all migrate wrappers) as part of the refactor.
- **Fix-shape constraint for SWARM-03**: `migrate()` is collective. Any
  automatic-migration fix must be deferred (context-exit / explicit flush), not
  a per-write callback, or it will deadlock when ranks write coordinates
  unevenly.
- **Severity inflation risk in serial-only workflows**: several High findings
  (SWARM-03/04/07) manifest only under MPI; serial users see SWARM-01/02/06
  first.
- This review did not audit swarm **checkpoint/restore round-trip fidelity**
  beyond SWARM-19, nor the `uw.Model` serialization of swarms; both belong to
  the follow-on design doc.

## Appendix — refuted claims (do not re-find)

No whole finding was refuted, but four sub-claims failed adversarial
verification and are recorded so they are not resubmitted:

- **R-1: "`ckdtree.pyx:104` performs an out-of-bounds memory read on an empty
  points array."** Refuted: `setup.py` sets only `language_level`, so Cython's
  default `boundscheck=True` applies; `&points[0][0]` on a `(0, dim)` buffer
  raises a clean `IndexError: Out of bounds on buffer access (axis 0)`
  (empirically confirmed). Consequence is a hard crash/MPI hang (SWARM-07), not
  memory corruption or UB.
- **R-2: "SwarmVariable `.array` lags behind a cached single-view MeshVariable
  architecture."** Refuted: `MeshVariable.array`
  (`discretisation_mesh_variables.py:2012`) has the identical
  fresh-closure-class-per-access design, and its `_array_cache` attribute
  (446, 1888) is dead — assigned `None`, never read. The correct framing is a
  *shared* view refactor on both sides (SWARM-14), not swarm-parity.
- **R-3: "IndexSwarmVariable's eager `_update()` is a mere contract violation;
  make it lazy like the base."** Refuted as a safe fix: `createMask()` builds
  solver expressions from `self._MaskArray` directly (2277-2278) without the
  `.sym` refresh, and no solve-time code checks `_proxy_stale` — the eagerness
  is currently the only thing keeping material masks fresh in a time loop (see
  Known Limitations).
- **R-4: "A size+generation self-check makes `_invalidate_canonical_data()` a
  fast-path optimization only."** Overstated: same-local-size migrations change
  values without changing `dm.getLocalSize()`, and bare `dm.migrate` sites
  (e.g. `_route_by_nearest_centroid`, 2754-2756) do not bump the generation
  counter. The refactor (SWARM-10) narrows the discipline; it does not remove
  it.
- Also corrected en route: the style guide's "Preferred" swarm example is worse
  than "deprecated" — `swarm.data` is getter-only, so `swarm.data +=
  displacement` raises `AttributeError` and cannot run at all (folded into
  SWARM-13).

## Sign-Off

- **Audit dimension**: 5 — Swarm/particle subsystem architecture
- **Reviewer**: Claude (Fable 5), AI-assisted audit under the July 2026 quality
  campaign; adversarial verification pass completed 2026-07-03 for SWARM-01 …
  SWARM-14 (including runtime reproductions of SWARM-01 and SWARM-06 in this
  worktree's built environment).
- **Evidence standard**: every cited `file:line` read directly in the audit
  worktree at `development@1d003481`; no line numbers trusted from prior notes
  without re-reading.
- **Constraint compliance**: no proposed fix touches
  `petsc_generic_snes_solvers.pyx` numerics; all fixes are Python-level cache/
  staleness/dead-code changes or additive tests; API-affecting items (SWARM-08
  excise option, SWARM-22 `sync` kwarg removal) are routed through Wave C
  deprecation shims.
- **Follow-on deliverable**: swarm modernization design doc (self-validating
  cache SWARM-10, migration trigger matrix SWARM-03/18, shared array-view
  refactor SWARM-14, real subsystem doc replacing the SWARM-24 stub).

*Underworld development team with AI support from Claude Code.*
