# API Consistency Review — July 2026 Quality Campaign (Dimension 3)

**Status**: audit complete; findings adversarially verified 2026-07-03
**Base**: `development` @ `1d003481` (audit worktree, campaign index at `e848d131`)
**Scope**: public API surface — the boundary-condition family, solver
constructors, solver-scoped capabilities (rotated BC, boundary flux, custom
multigrid, update callbacks), mode attributes, quantity factories, meshing
constructors, and namespace exposure (`uw.*`, subpackage `__init__`/`__all__`).

Abbreviation used throughout: `pyx` = `src/underworld3/cython/petsc_generic_snes_solvers.pyx`.

## Overview

June 2026 added new boundary-condition entry points, three solver-scoped
capability families, and two new utility modules across many non-overlapping
AI-assisted sessions. Each addition is individually reasonable; collectively
they have split the public API into two argument-order camps, two
value-parameter names, two custom-multigrid entry points with different names
and different correctness envelopes, and a set of user-facing modules that
resolve only via deep import.

This review (a) proposes THE single convention set for the codebase and
(b) lists every verified deviation with a deprecation-shim design. Per the
maintainer's constraints, **every fix harmonizes via a shim** — old signatures
keep working for at least one release cycle with a `DeprecationWarning`;
nothing here is a hard break, and nothing touches solver numerics. All shims
are zero-cost when unused (one inline `isinstance`/`type` check or a
`None`-default keyword alias; no wrapper layers).

Every location cited below was read directly in this worktree; line numbers
are exact at `1d003481`. Nine findings are adversarially verified (API-01 …
API-09); three lower-severity observations were evidence-checked by the author
but did not go through the adversarial pass (API-10 … API-12) and are tabled
separately. Details refuted during verification are recorded in the appendix
so the same false leads are not re-found.

**Scale note (API-01)**: the migration surface for the BC argument-order fix
is far larger than early estimates. At this ref the legacy order appears at
roughly **920 call sites under `docs/`** (~40 in `.md`, ~695 in `.py`
examples, ~185 in `.ipynb`) **plus ~450 in `tests/`** — a ~1,370-site wave.
The shim makes this safe, but the deprecation-window length and the eventual
removal of the old order need explicit maintainer sign-off: this is the
most-used user-facing API in the codebase.

## Changes Made

None — audit only. Proposed changes are listed as findings and are scheduled
for Wave C (API harmonization) and Wave E (docs alignment).

## System Architecture

### The API landscape as found (all read at `1d003481`)

**Boundary-condition family** (methods on solver classes):

| Method | Signature (as found) | Location |
|---|---|---|
| `add_essential_bc` | `(conds, boundary, components=None)` | `pyx:1530` |
| `add_natural_bc` | `(conds, boundary, components=None)` | `pyx:1573` |
| `add_dirichlet_bc` | `(conds, boundary, components=None)` | `pyx:1631` |
| `add_nitsche_bc` (SNES_Vector) | `(boundary, g=None, direction=None, gamma=10.0, theta=1, local_h=True)` | `pyx:3317` |
| `add_rotated_freeslip_bc` (Stokes) | `(boundary, normal=None)` | `pyx:5170` |
| `add_nitsche_bc` (SNES_Stokes_SaddlePt) | `(boundary, g=None, direction=None, normal=None, gamma=10.0, theta=1, mask=None, local_h=True)` | `pyx:5293` |
| `add_constraint_bc` (Stokes_Constrained) | `(boundary, g=0.0, normal=None, screening=None, augmentation=None, augmentation_base=1.0e4, degree=None)` | `systems/solvers.py:2335` |

The legacy trio is value-first; every method added since is boundary-first.
These seven are the only BC-adder definitions in `src/` (an
`add_essential_p_bc` exists only as a commented-out stub near `pyx:5149`).
The prescribed datum is spelled `conds` in the trio and `g` in exactly three
newer methods (the two Nitsche overloads and `add_constraint_bc`);
`add_rotated_freeslip_bc` takes no value at all (free-slip `g=0` implicit) and
participates in the drift only through `normal=`. The `components=` selector
already carries a live `DeprecationWarning` (`pyx:1449-1455`) in favour of
`None`/`sympy.oo` entries in the value vector (`pyx:1393-1398, 1458-1464`);
`direction=` (defaulting to the outward surface normal, `pyx:3369-3376`) and
`normal=` (three per-method meanings: rotation source at `pyx:5170`, Nitsche
consistency normal at `pyx:5293`, `Gamma_P1` default at
`systems/solvers.py:2396`) overlap with it.

**Solver-scoped capabilities.** `add_update_callback` (`pyx:301`),
`boundary_flux` / `boundary_flux_field` (`pyx:2165`/`pyx:2183`, lazily
importing `utilities/boundary_flux.py` at `pyx:2180, 2190`), and the rotated
free-slip family (`pyx:5170/5253/5272`, delegating to
`utilities/rotated_bc.py`) all follow the good pattern: **a method on the
solver that lazily imports its implementation module**. Custom multigrid is
the outlier: the solver *method* `set_custom_mg(coarse_meshes, kind=...)`
(`pyx:259`) configures the **legacy serial-only, finest-only-reduction,
single-field-only** path (`_require_serial` called at
`utilities/custom_mg.py:640`; `NotImplementedError` for non-empty PC prefixes,
i.e. Stokes blocks, at `custom_mg.py:646-647`), while the correct
parallel-capable BC-per-level path is the free function
`set_custom_fmg(solver, coarse_meshes, *, builder=..., field_id=...)`
(`custom_mg.py:605`) reachable only by deep import. The method's own docstring
(`pyx:280-291`) advertises Stokes velocity-block support that only the
deep-import path delivers.

**Mode attributes.** `consistent_jacobian` is set as a bare attribute
(`pyx:92`) documented only by a code comment block (`pyx:71-91`). It is
dispatched by exact match in two places: `_jacobian_source` (`pyx:160-199`;
`if not mode` → Picard, `mode == "continuation"` → blended, **any other truthy
value falls through to full Newton**) and the solve path
(`pyx:1096`, `== "continuation"` selects the two-phase continuation solve).
`utilities/rotated_bc.py:396` branches on the same unvalidated tri-state.

**Namespace exposure.** `utilities/__init__.py` (86 lines) imports none of
`rotated_bc`, `boundary_flux`, `custom_mg` — all in-package uses are
method-local lazy imports, and docstring cross-references such as
`:mod:`underworld3.utilities.custom_mg`` (`pyx:290-291`) resolve only after a
deep import. `meshing/__init__.py` (`__all__` at line 68) neither imports nor
lists `bounding_surface.py`'s `BoundingSurface` (`:41`) and
`register_radial/plane/box_face_surfaces` (`:213/:224/:233`), so
`uw.meshing.BoundingSurface` raises `AttributeError` even though
`Mesh.register_tangent_slip_provider` requires a `BoundingSurface` instance
(`discretisation/discretisation_mesh.py:2719-2730`).

**Quantity factories.** `uw.quantity` (`__init__.py:223` →
`function/quantities.py:883`) returns a `UWQuantity`; `uw.create_quantity`
(`__init__.py:253` → `units.py:1115`, implemented as `value * ureg(units)` at
`units.py:60-64`) returns a **raw Pint Quantity**. Same conceptual purpose,
silently different objects downstream; `UWQuantity` itself is not exposed at
top level, so `isinstance` checks require a deep import.
`tests/test_0640_api_consistency_regression.py:140,254,274` currently pins
`create_quantity` as public.

**Constructors.** Base `SNES_Scalar` (`pyx:2265`) and every sibling —
`SNES_Vector` (`pyx:3200`), `SNES_Darcy`, `SNES_Stokes`, all projections —
order `(mesh, u_Field, degree, verbose, ...)`. `SNES_Poisson`
(`systems/solvers.py:190-198`) is uniquely inverted:
`(mesh, u_Field, verbose=False, degree=2, ...)`. Meshing constructors are
internally consistent in their grandfathered camelCase (`minCoords`,
`cellSize`); the one substantive drift is that the `units=` kwarg exists only
on the two Cartesian constructors (`meshing/cartesian.py:43, 970`).
`uw.function.evaluate` / `global_evaluate` were checked and found mutually
consistent — no finding.

### THE convention set (proposed)

| # | Convention | Rule |
|---|---|---|
| C1 | **BC argument order** | `add_<kind>_bc(boundary, value, ...)` — boundary label (str) first, prescribed datum second. Matches every method added since the trio; the trio is shimmed. |
| C2 | **Value-parameter name** | ONE name: **`value`** — the prescribed boundary datum in every `add_*_bc`. Self-documenting for non-FEM readers (founding readability rule). `conds=` and `g=` accepted as `None`-default deprecated keyword aliases for one cycle. |
| C3 | **Direction selection** | ONE mechanism: component masking via `None`/`sympy.oo` entries in `value` (as today); a scalar constraint along a vector uses `direction=` (defaults to outward normal). `normal=` is reserved strictly for overriding the *geometric surface-normal source* and is never a component mask. `components=` completes its already-warned deprecation. |
| C4 | **Method-on-solver rule** | Any capability that configures or reads one specific solver is a method on that solver, delegating via lazy import to a `utilities/*` implementation module (the `boundary_flux` pattern, `pyx:2180`). Free functions remain the implementation layer, not the documented entry point. |
| C5 | **Namespace exposure** | Every user-facing module is added to its subpackage `__init__`/`__all__` in the PR that creates it. Top-level `uw.*` keeps one canonical factory per concept. |
| C6 | **Constructor parameters** | `(mesh, <field(s)>, <discretisation: degree...>, <behaviour flags: verbose...>, DuDt, DFDt)` — discretisation before behaviour flags, everywhere. New parameters are `snake_case`; existing camelCase is grandfathered but never extended. |
| C7 | **Mode/state attributes** | Multi-state switches (e.g. `consistent_jacobian`) are validated properties with NumPy docstrings; unrecognized values raise `ValueError` at assignment, never silently select a behaviour. Falsy values normalize to the canonical `False`. |
| C8 | **Docstrings** | NumPy/Sphinx style with RST `:math:` (settled maintainer decision). Google-style `Args:` blocks are nonconforming. |
| C9 | **Deprecation shims** | Old signature detected inline (one type check on a positional, or a `None`-default keyword alias); `warnings.warn(..., DeprecationWarning, stacklevel=2)`; forward to the new path. No wrappers, no decorators — zero cost on the new signature. Removal only after an explicitly signed-off window (see API-01). |

### Shim designs (per deviation class)

**Arg-order shim (API-01)** — unambiguous because a boundary label is always a
`str` and a BC value never is (`add_condition` at `pyx:1359` raises
`ValueError` for string conds; all repo callers pass string boundaries, enum
users all use `.name`; no caller uses `conds=`/`boundary=` keywords):

```python
def add_essential_bc(self, boundary, value=None, components=None, **kw):
    if not isinstance(boundary, str):           # old order: (conds, boundary)
        boundary, value = value, boundary       # value slot holds the label
        warnings.warn("add_essential_bc(conds, boundary) is deprecated; "
                      "use add_essential_bc(boundary, value)",
                      DeprecationWarning, stacklevel=2)
    ...
```

One `isinstance` on the hot path; new-signature calls pay nothing else.

**Keyword-alias shim (API-02)** — `conds=None` / `g=None` kept in the
signature one cycle; if supplied and `value` is not, forward with a warning.
`None` is not a meaningful datum for these methods, so the sentinel is safe.

**Constructor-order shim (API-08)** — the swapped pair is `(verbose, degree)`
vs `(degree, verbose)`; the legacy detection must be `type(third) is bool`,
**not** `isinstance`, because `isinstance(True, int)` is `True`.

**Attribute-validation shim (API-04)** — `consistent_jacobian` becomes a
property; the setter accepts `{False, True, "continuation"}`, normalizes other
falsy values (`None`, `0`) to `False` (today `if not mode` treats them as
Picard), and raises `ValueError` for anything else. Invalid values currently
mis-select full Newton silently, so raising is a bug-fix, not a break.

## Findings — verified

All nine adversarially verified against this worktree; ranked
most-severe-first.

| ID | Location | Severity | Effort | Finding | Proposed fix |
|---|---|---|---|---|---|
| API-01 | `pyx:1530,1573,1631` vs `pyx:3317,5170,5293`, `systems/solvers.py:2335` | high | M | BC family split into two argument-order camps: the legacy trio takes `(conds, boundary)`, every later method takes `(boundary, ...)`. The migration surface is ~920 call sites in `docs/` plus ~450 in `tests/` (~1,370 total — an early "~101 docs sites" estimate was a ~13× undercount). This is the most-used user-facing API in the codebase. | Adopt boundary-first (C1) on the trio with the `isinstance(boundary, str)` arg-order shim; update docs/examples in the same wave (Wave C + Wave E). **Deprecation-window length and eventual removal of the old order require explicit maintainer sign-off** — do not hard-remove after one cycle by default. |
| API-02 | `pyx:1530/1573/1631` (`conds`) vs `pyx:3317,5293`, `solvers.py:2335` (`g`); `pyx:1449-1455` (`components`), `pyx:3369-3376` (`direction`), `pyx:5170/5293`, `solvers.py:2396` (`normal`) | medium | S | Value-parameter and direction vocabulary drift: the datum is `conds` in the trio and `g` in exactly three newer methods (`add_rotated_freeslip_bc` takes no value; there is no fifth `g` method). Direction/masking is spread across four overlapping mechanisms: `components=` (deprecated), `None`/`oo` masking, `direction=`, and `normal=` with three per-method meanings. | ONE datum name `value` (C2) with `conds=`/`g=` as `None`-default deprecated aliases; ONE direction convention (C3): masking in `value`, `direction=` for scalar-along-vector, `normal=` strictly for geometric-normal override; finish the `components=` deprecation. Lands in the same edit as API-01. |
| API-03 | `pyx:3317` vs `pyx:5293` | medium | S | Two same-named `add_nitsche_bc` methods diverge: the `SNES_Vector` version lacks `normal=` and `mask=`, so code written against the Stokes variant raises `TypeError` when moved to a vector solver. The vector docstring (`pyx:3336-3349`) does not itself document those parameters, but cross-references `SNES_Stokes_SaddlePt.add_nitsche_bc` twice ("See ... for details") with no note that the vector variant lacks them. | Align the `SNES_Vector` signature: accept `normal=` with the same geometric-normal-override semantics (Stokes `pyx:5391-5395`); accept `mask=` and raise a clear `NotImplementedError` naming the limitation if genuinely unsupported. Signature/docs only — no numerics. |
| API-04 | `pyx:92` (set), `pyx:71-91` (comment), `pyx:160-199` (`_jacobian_source`), `pyx:1096` (solve dispatch) | medium | S | `consistent_jacobian` is a bare, undocumented tri-state `False\|True\|"continuation"` with **no validation anywhere**. `_jacobian_source` falls through to the full-Newton tangent for any unrecognized truthy value — `"picard"`, `"Continuation"`, `1` all silently select Newton; the exact-string match at `pyx:1096` likewise silently skips the continuation solve on a typo. | Validated property (C7): setter accepts `{False, True, "continuation"}`, normalizes falsy to `False`, raises `ValueError` otherwise; NumPy docstring lifted from the `pyx:71-91` comment block. Validation covers both dispatch sites. Behaviour bit-identical for the three valid values (within the pyx no-numerics ground rule). |
| API-05 | `pyx:259` (`set_custom_mg`) vs `utilities/custom_mg.py:605` (`set_custom_fmg`), `:640` (`_require_serial` call), `:646-647` (single-field gate) | medium | S | Custom multigrid has two entry points with different names, parameter vocabularies (`kind=` vs `builder=`/`field_id=`), and correctness envelopes: the discoverable solver *method* is the legacy serial-only, finest-only-reduction, single-field-only path — and its docstring (`pyx:280-291`) promises Stokes velocity-block support that only the deep-import `set_custom_fmg` path delivers. | Add `SolverBaseClass.set_custom_fmg(coarse_meshes, *, builder=..., field_id=None, verbose=False)` delegating lazily to `utilities.custom_mg` (C4, `boundary_flux` pattern at `pyx:2180`); export `set_custom_fmg` from `utilities/__init__`; `set_custom_mg` gains a `DeprecationWarning` (behaviour preserved one cycle); unify on `builder=` with `kind=` as deprecated alias. |
| API-06 | `utilities/__init__.py` (86 lines); `meshing/__init__.py:68` (`__all__`); `meshing/bounding_surface.py:41,213,224,233` | medium | S | Exposure gaps: `utilities/__init__` imports none of `rotated_bc`/`boundary_flux`/`custom_mg` (docstring cross-refs like `pyx:290-291, 497` resolve only via deep import); `meshing/__init__` neither imports nor lists `BoundingSurface` and the three `register_*_surfaces` helpers, so `uw.meshing.BoundingSurface` raises `AttributeError` despite being required by `Mesh.register_tangent_slip_provider` (`discretisation_mesh.py:2719-2730`). | Add `from . import rotated_bc, boundary_flux, custom_mg` (+ `set_custom_fmg` re-export) to `utilities/__init__`; add the four `bounding_surface` names to `meshing/__init__` imports and `__all__` (C5). Pure additions, no shim; adopt C5 as the go-forward convention. |
| API-07 | `__init__.py:223,253`; `function/quantities.py:883`; `units.py:60-64,1115` | medium | S | Duplicate top-level quantity factories with different return types: `uw.quantity` → `UWQuantity`; `uw.create_quantity` → raw Pint `Quantity` (`value * ureg(units)`). Same purpose, silently different objects downstream; `UWQuantity` not exposed at top level (deep import needed for `isinstance`); `test_0640_api_consistency_regression.py:140,254,274` pins `create_quantity` as public. | `uw.quantity` is THE factory (matches CLAUDE.md units principles). `create_quantity` keeps exact behaviour/return type one cycle but emits `DeprecationWarning` naming `uw.quantity`; expose `uw.UWQuantity`; update test_0640 in the same PR. |
| API-08 | `systems/solvers.py:190-198` vs `pyx:2265` and siblings (`solvers.py:396,1114,2618,2933,3165,3316`) | medium | S | `SNES_Poisson.__init__` is `(mesh, u_Field, verbose=False, degree=2, ...)` while its base and every sibling put degree before verbose — SNES_Poisson is uniquely inverted. A positional `SNES_Poisson(mesh, None, 3)` intended as `degree=3` silently sets `verbose=3` and auto-creates a degree-2 field; with an existing field `T`, `SNES_Poisson(mesh, T, 3)` silently enables verbose (degree is unused then — see refuted-claims appendix). No in-repo caller is currently bitten (all keyword calls). | Reorder to `(mesh, u_Field, degree, verbose, ...)` (C6) with the constructor-order shim: `type(third) is bool` → legacy `verbose` + `DeprecationWarning`. Add a positional-call regression test. |
| API-09 | `pyx:2183` (`boundary_flux_field(..., scale=)`), `pyx:5272` (`dynamic_topography(..., buoyancy_scale=)`), `systems/solvers.py:2461` (`topography(..., buoyancy_scale=, reference=)`), `utilities/boundary_flux.py:256` (`boundary_flux_to_field`) | medium | S | Vocabulary drift across the three dynamic-topography/flux-recovery paths: field-write vs expression-return, `scale=` vs `buoyancy_scale=`, and a free function (`boundary_flux_to_field`) spelled differently from the method it backs (`boundary_flux_field`). **Do not alias `scale=` to `buoyancy_scale`**: they are not the same factor — `buoyancy_scale` is Δρg used as a divisor with the sign internal (`rotated_bc.py:775`, `-s/buoyancy_scale`), while `scale` is a generic multiplier whose topography value is `-1/(Δρg)` (negated reciprocal; primary tested use is heat flux/Nusselt, `test_1019:38`, `parallel/test_1065:44`). | Safe scope: (1) rename the free function `boundary_flux_to_field` → `boundary_flux_field` to match the method, old name aliased one cycle; (2) document the expression-return (`topography`) vs field-write (`dynamic_topography`/`boundary_flux_field`) distinction; (3) document in `boundary_flux_field`'s docstring that `scale = -1/buoyancy_scale` for topography — do not rename the parameter. |

## Findings — unverified

Evidence-checked by the author at the cited lines but **not** adversarially
verified; treat as candidates pending a verify pass before entering the
remediation worklist.

| ID | Location | Severity | Effort | Finding | Proposed fix |
|---|---|---|---|---|---|
| API-10 | `pyx:3200-3212` vs `pyx:2277-2281` (SNES_Scalar auto-create), `pyx:735-737` (`Unknowns.u` setter ignores `None`) | low | S | `SNES_Vector.__init__` advertises `u_Field=None` like `SNES_Scalar` but never auto-creates the variable (`self.Unknowns.u = u_Field` directly at `pyx:3212`, no `if u_Field is None` branch); the setter silently ignores `None`, so construction appears to succeed and the solver fails later with an obscure `AttributeError`. Same-named parameter, different contract between sibling bases. | Either auto-create the vector variable as `SNES_Scalar` does, or raise `ValueError("u_Field is required")` at construction. No shim needed — current behaviour is already a crash, just delayed and cryptic. |
| API-11 | `meshing/cartesian.py:43,378,970` (accept `units=`, thread it to `Mesh` at `:352,909,1367`) vs `meshing/annulus.py`/`spherical.py` (no `units=` anywhere; `Mesh` itself accepts it, `discretisation_mesh.py:274`) | low | M | The `units=` kwarg exists on **three** Cartesian constructors only (an earlier "two" was a miscount — `BoxInternalBoundary` at `:378` also has it), so the units system's mesh entry point is geometry-dependent: annulus/spherical auto-non-dimensionalise UWQuantity geometry args (e.g. `annulus.py:116-121`) but never pass coordinate units to `Mesh`. Worse, the three Cartesian docstrings disagree with themselves: two label `units=` "**Deprecated**" (`:83-85`, `:1005-1006`) while one documents it as live ("Coordinate units for unit-aware arrays", `:421-422`) and `StructuredQuadBox` actively auto-detects units from UWQuantity inputs (`:1106-1117`). | First decide the parameter's status (live vs deprecated) — the docstrings must stop contradicting the code; if live, thread `units=` through the remaining constructors (pure addition, default `None` preserves behaviour). Can be scheduled with a units wave rather than Wave C. |
| API-12 | `units.py:100,349,420,517,571,1119` (representative) | low | S | Top-level-exported units API uses Google-style `Args:` docstrings — including `uw.create_quantity` — against the settled NumPy/RST standard (C8); newer solver/BC docstrings already conform, making the exported units module the visible outlier. | Mechanically convert `units.py` public-function docstrings to NumPy style during Wave E (`scripts/docstring_sweep.py` exists). Full docstring census belongs to Dimension 6. |

## Testing Instructions

Validation plan for the eventual Wave C/E fixes (each PR cites its finding ID):

- **Baseline**: `./uw build` in the wave worktree, then
  `pytest -m "level_1 and tier_a"` green before and after; `tier_a or tier_b`
  before merge (campaign ground rule).
- **Every shim** lands with two tests: (1) the OLD signature produces
  identical results and emits exactly one `DeprecationWarning`
  (`pytest.warns(DeprecationWarning)`); (2) the NEW signature emits none
  (`warnings.simplefilter("error")` inside the test).
- **API-01/02**: run the arg-order shim tests on all three trio methods,
  including a mixed case (`add_dirichlet_bc("Top", 0.0)` new vs
  `add_dirichlet_bc(0.0, "Top")` old). Grep docs/tests after the Wave E sweep:
  `grep -rE "add_(essential|natural|dirichlet)_bc\(" docs tests` call sites
  must all be new-order.
- **API-03**: `vector_solver.add_nitsche_bc(bdy, normal=n)` must not raise
  `TypeError`; `mask=` on the vector solver raises `NotImplementedError` with
  a message naming the limitation (if that branch is taken). Existing Nitsche
  tier_a tests bit-identical.
- **API-04**: `solver.consistent_jacobian = "picard"` (and `"Continuation"`)
  raises `ValueError`; `False`/`True`/`"continuation"` round-trip; `None`/`0`
  normalize to `False`; existing users
  `tests/test_1018_rotated_freeslip.py:237` and
  `tests/parallel/test_1064_rotated_freeslip_parallel.py:191` (bare-attribute
  assignment of valid values) stay green; tier_a solver results bit-identical.
- **API-05/06**:
  `python -c "import underworld3 as uw; uw.utilities.custom_mg.set_custom_fmg; uw.meshing.BoundingSurface"`
  succeeds without deep imports; `set_custom_mg` warns but reproduces its
  current results; custom-MG tests (test_1015-1017 range) and rotated
  free-slip tests (`test_1018`, `parallel/test_1064`) green; **np2/np4
  parallel runs required** for anything touching `custom_mg`.
- **API-07**: update `tests/test_0640_api_consistency_regression.py` in the
  same PR that adds the `create_quantity` warning (it calls it bare at
  `:254`); assert `uw.UWQuantity` importable and
  `isinstance(uw.quantity(1, "m"), uw.UWQuantity)`.
- **API-08**: positional regression test — `SNES_Poisson(mesh, None, 3)` after
  the fix warns nothing and creates a degree-3 unknown;
  `SNES_Poisson(mesh, None, True)` (legacy verbose) warns and preserves
  degree=2.
- **API-09**: pure rename/docs — heat-flux tests
  `tests/test_1019_boundary_flux.py` and
  `tests/parallel/test_1065_boundary_flux_parallel.py` green; old
  free-function name importable with warning for one cycle.

## Known Limitations

- Line numbers are exact at `development@1d003481` and will drift as waves
  land; finding IDs, not line numbers, are the stable reference.
- This review covers the *public* API surface; internal call sites on
  deprecated data-access patterns (~41 sites, Wave B) are Dimension 1/4
  territory. After Wave B, internal code must not exercise the shims added
  here (shim-warning tests enforce this incidentally).
- The `value` canonical name (C2) is a proposal; the zero-churn alternative
  (standardize on `g`) flips only the alias direction — shim mechanics
  identical. Maintainer's call.
- API-01's eventual removal timeline is explicitly **not** decided here: with
  ~1,370 old-order call sites in docs and tests, "one release cycle then
  remove" would hard-break the most-used API; the shim can be kept
  indefinitely at negligible cost.
- `uw.function.evaluate`/`global_evaluate` and the meshing constructor family
  (beyond API-11 and grandfathered camelCase) were checked and found
  internally consistent — no findings; recorded so coverage is explicit.
- All `pyx` proposals are confined to naming/docs/validation/shims; no
  numerics change is proposed anywhere in this review (campaign ground rule).

## Appendix: refuted or corrected claims

No whole finding was refuted, but these sub-claims were corrected during
adversarial verification — recorded so they are not re-found:

1. **"~101 documentation call sites teach the old BC order"** — WRONG (~13×
   undercount). True surface at this ref: ~920 in `docs/` + ~450 in `tests/`
   (≈1,370 total; re-counted by this author:
   `grep -rEoh "add_(essential|natural|dirichlet)_bc\(" docs | wc -l` → 925
   across all file types, 920 restricted to `.md`/`.py`/`.ipynb`; same over
   `tests` → 449. No `docs/_build` exists at this ref to inflate the count).
2. **"The datum is spelled `g` in all five newer BC methods"** — WRONG.
   Exactly **three** methods take `g` (`pyx:3317`, `pyx:5293`,
   `solvers.py:2335`); `add_rotated_freeslip_bc` (`pyx:5170`) takes no value
   parameter, and there is no fifth method (`add_essential_p_bc` is
   commented out near `pyx:5149`).
3. **"The vector `add_nitsche_bc` docstring documents `normal`/`mask`
   parameters it does not accept"** — WRONG as stated. It never names them
   (`pyx:3318-3350` documents only its six actual parameters); the problem is
   the unqualified cross-references to the Stokes docstring (`pyx:3340,3344`)
   where they ARE documented.
4. **"`scale=` on `boundary_flux_field` is the same physical factor as
   `buoyancy_scale` and should be renamed/aliased"** — WRONG and dangerous:
   `buoyancy_scale` is Δρg used as a divisor with the minus sign internal
   (`rotated_bc.py:775`), `scale` a generic multiplier whose topography value
   is `-1/(Δρg)` — treating them as one factor invites sign/reciprocal errors.
   Document the relationship instead (see API-09).
5. **"`SNES_Poisson(mesh, T, 3)` silently changes the discretisation"** —
   WRONG for an existing field `T`: `degree` is only used when `u_Field is
   None` (auto-create at `pyx:2277-2281`; the `Unknowns.u` setter ignores a
   `None` reassignment at `pyx:735`), so `T`'s own degree governs; the call
   only silently sets `verbose=3`. The genuine wrong-discretisation case is
   `SNES_Poisson(mesh, None, 3)` (see API-08).
6. **"The `consistent_jacobian` setter should accept exactly
   `{False, True, "continuation"}`"** — needs one refinement: other falsy
   values (`None`, `0`) currently behave as Picard via `if not mode`
   (`pyx:179`), so the setter must normalize falsy to `False` rather than
   reject it (see API-04 fix).

## Sign-Off

| Reviewer | Role | Status |
|---|---|---|
| Louis Moresi | Maintainer | Pending review |
| Claude (audit session, Dimension 3 — api) | Author | Complete 2026-07-03 |
