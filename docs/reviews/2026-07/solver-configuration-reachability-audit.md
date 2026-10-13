# Solver configuration audit: reachability and silent fallback

**Date**: 2026-07-30
**Baseline**: `development` at `04379583` (post-#458). Diagnostic only — no code changes.
**Harness**: `~/+Simulations/solver_reachability_audit/` (`reachability_matrix.py`,
`mg_bundle_reachability.py`), `uw.Params` arguments, README alongside.

## Why this audit

Five defects of the same shape were found independently and by accident over the last
few weeks: #276/#478, #467, #468, #477, and arguably #425. Every one of them **degrades
to something that succeeds** — GAMG converges, richardson converges, the solve returns a
correct answer — so none presented as a failure. They presented months later as "this is
a bit slow", or as a parameter sweep returning identical rows, which reads as *"this
variable does not matter"* rather than *"this variable was never set"*. That cost a
twelve-cell sweep on #477 and a wrong premise in #471.

That rate says the next one is found by looking. This audit looks, in two directions:

- **reachability** — the user writes a setting and something else runs;
- **silent substitution** — the code deliberately chooses otherwise and nothing
  observable says so.

Findings are ranked by **how invisible the failure is**, not by the size of the
substitution. A knob that is unreachable but loudly so is a low row.

## Method

Per solver family: build a small solver, set `tolerance`, write a distinctive value for
each managed option, `solve()` **twice**, `pc.setUp()`, then read the **live PETSc
objects**. Two arms — a defaults arm and an overrides arm — because a "reachable" verdict
is meaningless if the override happened to match the default. Rows where the asked value
equals the default are reported as proving nothing.

Traps that would have made this lie, all of which have already produced a wrong
conclusion in a recent session: `getFieldSplitSubKSP()` on a PC that has not been set up
returns stale defaults; a single solve hides every solve-time clobber; the options
database shows nothing (a clobbered or never-written key leaves no trace there); and an
instrument must be calibrated on a known answer before it is trusted.

---

## Findings

### F-1 — The multigrid bundle silently overwrites user settings, and asking more explicitly makes it worse

**Invisibility: total.** No warning, no readable state, no trace in the options database.

Measured on Stokes, `refinement=1`, asking for `chebyshev` / `max_it 6` / `coarse svd`
on the velocity MG block:

| what the user set | result |
|---|---|
| the three bundle keys only | **overwritten** (gmres / 4 / redundant) |
| the three keys **+ `fieldsplit_velocity_pc_type=mg`** | **honoured** (chebyshev / 6 / svd) |
| the three keys + `preconditioner="fmg"` | **overwritten** |

The mechanism is `_apply_preconditioner_options`, which writes the whole bundle from
`_build` on every rebuild. It *has* a user-override latch — but the latch keys on
`pc_type` **alone**. So of the ten keys in the bundle, setting any of nine is silently
discarded, and setting the tenth rescues all ten. Which key is privileged is arbitrary
from the user's point of view, and the workaround ("to make your smoother stick, also set
`pc_type` to the value it already has") is undiscoverable.

Worse, it inverts: `preconditioner="fmg"` — the *more* deliberate request — bypasses the
latch entirely, because explicit mode "always applies". The more clearly a user states
their intent, the less control they have.

**Home: #471.** That PR makes the bundle a single owner applied unconditionally, which
entrenches this. The fix belongs there: the owner should not overwrite a key the user has
set, i.e. the latch should be per-key rather than keyed on `pc_type`. Doing it in #471
also means one place to change instead of three.

### F-2 — Geometric multigrid is unreachable for every single-field solver, and `auto` warns about nothing

Already filed as **#478**. Restated here because the *silence* is the part that belongs
in this audit: `preconditioner="auto"` (the default) produces no warning at all. Only an
explicit `"fmg"` warns. So the population most likely to be affected — anyone who built a
refined mesh and assumed the default would use it — gets no signal whatsoever.

**Home: #478**, its own PR.

### F-3 — Which options are reachable differs between two solvers of the same family, in both directions

| option | `Stokes` | `Stokes_Constrained` |
|---|---|---|
| `ksp_rtol` | **reachable** | **NOT reachable** (→ `tolerance × 0.1`) |
| `fieldsplit_velocity_ksp_rtol` | **NOT reachable** (→ `tolerance × 0.033`) | **reachable** |
| `fieldsplit_pressure_ksp_rtol` | **NOT reachable** (→ `tolerance × 0.1`) | **reachable** |
| `snes_max_it` | NOT reachable (→ 50) | NOT reachable (→ 50) |
| `snes_rtol` | NOT reachable (→ `tolerance`) | NOT reachable (→ `tolerance`) |

Both are saddle-point solvers. They disagree about the outer KSP tolerance *and* about
the inner ones, in opposite directions. Nothing documents the difference, and no user
could predict it.

Scalar (`Poisson`) and vector (`Vector_Projection`) families are **clean** — every option
tested is reachable. So the whole defect class is localised to the saddle-point path.

**Home: #475 / `feature/regime-diagram`** (see F-5).

### F-4 — `snes_rtol` is unreachable, and stays unreachable after the in-flight fix

The unpushed `_reassert_outer_tolerances()` deliberately re-writes `snes_rtol` and
`ksp_atol` at every solve, by design ("only the outer keys"). That is a defensible
ownership decision — `tolerance` owns the outer tolerances — but it leaves `snes_rtol`
looking like a settable knob that silently does nothing.

This is a **decide-and-document** row rather than a bug: either accept the user's value
or say in the `tolerance` docstring that these keys are owned and will be overwritten.
The current state is the worst of both.

**Home: #475**, with the fix it is adjacent to.

### F-5 — The #477 fix exists, is committed, and is invisible: three unpushed commits

`feature/regime-diagram` carries three commits that are **not on origin**:

```
954ff4f3 fix(solver): make Eisenstat-Walker switchable
5ad68b2f docs(solvers): the Stokes fieldsplit is two nested Krylov loops
f1555b3a fix(solver): make the Stokes solver settings reachable (#477, D-22/D2)
```

So PR #475 does not show the #477 fix, and anyone reading the PR list — or auditing
`development`, as this audit did — rediscovers a solved problem. Verified: the fix covers
`snes_max_it` and the two fieldsplit rtols; it does **not** cover the MG bundle keys
(F-1), which are clobbered at *build* time rather than solve time and are a different
mechanism.

**Also a merge collision to resolve deliberately.** #475 patches the hand-written
GAMG stale-key list in `_apply_preconditioner_options`:

```
-                        "mg_levels_pc_type", "mg_coarse_pc_type",
+                        "mg_levels_pc_type", "mg_levels_ksp_norm_type",
+                        "mg_coarse_pc_type",
```

#471 deletes that list entirely and derives it — and the derived GAMG stale set already
contains `mg_levels_ksp_norm_type`. So **#471 subsumes #475's patch**, but the two
conflict textually. Whichever lands second should take #471's derivation, not merge the
lists.

### F-6 — Only two of the twelve substitution points leave readable state, and both were added last week

Inventory of the config-surface fallbacks:

| site | substitutes | observable? |
|---|---|---|
| `_apply_preconditioner_options` single-field gate | geometric MG → GAMG | warns only on explicit `fmg` |
| `_apply_preconditioner_options` `n_levels <= 1` | geometric MG → GAMG | warns only on explicit `fmg` |
| `_apply_preconditioner_options` override latch | user's bundle → managed bundle | **nothing** (F-1) |
| `auto_inject_custom_mg` barycentric→RBF retry | sparse transfer → **dense** | warns (F-7) |
| `auto_inject_custom_mg` build failure | geometric MG → default PC | warns |
| `auto_inject_custom_mg` dimensional guard | skips the auto-pickup | warns |
| `_assert_finest_spans_operator` swallow | **disables a correctness guard** | **nothing** (F-8) |
| `CustomMGHierarchy.build` `setUp` swallow | possibly-stale DM section | **nothing** (F-8) |
| `_install_velocity_block_transfers` PETSc-error path | alternative assembly route | nothing (same outcome) |
| `_enforce_galerkin_for_geometric_mg` | forces `pc_mg_galerkin=both` | warns |
| `rotated_bc` no hierarchy | geometric MG → GAMG | **`velocity_pc`** |
| `rotated_bc` no pressure mass | 1/μ-mass Schur → `selfp`+jacobi | **`schur_pre`** |

The only two rows with readable state are `velocity_pc` and `schur_pre`, both added in
#465 — and they are the *only* reason #467 was findable at all. That is the pattern worth
generalising: **a warning is not observability.** Warnings are filtered, deduplicated by
Python's warning registry, and invisible in a long run; readable state can be asserted on
in a test. There is currently no public way to ask a solver "which preconditioner did I
actually get, and why" — `_pc_managed_value` and `_pc_user_override` are private.

A related distinction the current state cannot express: **"couldn't" and "chose not to"
are different.** GAMG-because-no-hierarchy and GAMG-because-we-distrust-injection are
indistinguishable to a user, and only one of them is a bug.

### F-7 — The RBF retry silently converts the transfer to dense

`auto_inject_custom_mg` retries a failed barycentric build with the RBF builder, whose
transfer is **dense** (`nnz/row == n_coarse`), making the Galerkin coarse operators dense
too. The code comment is explicit that this "rescues correctness but does not scale" and
that on a production-sized problem it should be treated as a performance cliff. It warns
once, and leaves no state — so a long run cannot tell you it happened, and no test can
assert which builder ran.

**Home:** the same reporting mechanism as F-6.

### F-8 — Two silent swallows disable correctness guards

- `_assert_finest_spans_operator`: `except Exception: return` — cannot read the operator,
  so the guard against a rectangular Galerkin product is skipped. Its own docstring calls
  it a guarantee.
- `CustomMGHierarchy.build`: swallows a failed `snes.setUp()` and proceeds with whatever
  DM section is current — documented rationale, but the failure it protects against
  (a stale section on an `adapt()` child) is exactly the case it then proceeds into.

Both are defensible as written. Both are places where "do not crash the solve" was chosen
over "say that the check did not run".

---

## What this says about the pattern

Every finding above is a locally reasonable five-minute decision. Nobody was careless.
The common cause is that **"keep working" was made the invariant instead of "do what was
asked, or say you didn't"** — and the second-order effect is that each safe fallback is
cheap for its author and expensive for whoever measures through it three months later.

The audit also found the pattern propagating: the #467 fix in #471 inherits the
opportunistic degrade-to-GAMG, because that is what the surrounding code does and what
the brief asked for. It is the right local call and one more silent fallback.

Three changes would have caught nearly all of it, and none requires giving up the safety:

1. **Report the choice as state, not a warning.** `velocity_pc` is the model. It is the
   only reason #467 was findable.
2. **Distinguish "couldn't" from "chose not to".** A capability we do not trust and a
   hierarchy that does not exist should not read the same.
3. **Do not silently overwrite what the user set** (F-1), or say plainly that the key is
   owned (F-4). The current middle ground — accept it, then discard it — is the worst
   option.

## Routing summary

| finding | home | status |
|---|---|---|
| F-1 MG bundle overwrites user settings | **#471** | **FIXED** in `5a390efe` |
| F-2 single-field MG gate is silent | **#478** | filed |
| F-3 Stokes vs Stokes_Constrained disagree | **#475** | filed as **#483** |
| F-4 `snes_rtol` owned but looks settable | **#475** | filed as **#483** |
| F-5 three unpushed commits; #471/#475 conflict | **coordination** | raised on #475 |
| F-6 fallbacks are not observable | own PR | filed as **#484** |
| F-7 RBF retry is a silent performance cliff | with F-6 | filed as **#484** |
| F-8 two guards can be skipped silently | with F-6 | filed as **#484** |

### F-1 as fixed (#471, `5a390efe`)

`MGBundle.apply` takes an ownership record and leaves alone any key whose current
value is not the one UW3 last wrote — including in the stale-key clear, since a
user-set key is not ours to remove. Ownership is **recorded, never inferred from the
value**: inference fails as soon as a second internal writer touches the key, which
is how the `tolerance` and `strategy` setters defeated the earlier attempt in #477.
All forty internal writes of a bundle key now go through
`SolverBaseClass._push_managed_option`.

Two things worth carrying forward from doing it:

- The record must be keyed by the **globally-qualified** option name. The solver
  writes through a prefixed view (`Solver_N_`) while `custom_mg._configure_pcmg`
  reads the global database using the live PC's full prefix; an unqualified record
  made every key look user-owned over there and the bundle **silently stopped
  applying**.
- That regression was caught by the **defaults arm**, not by any test. It is the
  second time in two sessions that the defaults arm caught an ownership regression
  (the first was #477's). Do not run an ownership change without one.

---

## Phase 3 (expanded 2026-07-30): are the solver's own PROPERTIES live?

Phases 1 and 2 checked option **keys** and fallback **branches**. They missed
`solver.strategy`, which validated three values and configured one. That is a third
axis, and the expansion covers it: for each solve-configuring property, set two
distinct values and see whether anything observable moves — round-trip, options under
the solver prefix, assembled `J` / `Pmat`, `|u|`, `|p|`, and iteration counts.

Harness: `property_inertness.py` (sweep) and `property_inertness_fair.py` (retest).

### The instrument was wrong three times before it was right

This is the most transferable part of the expansion. Each flaw produced a confident,
wrong answer:

1. **Options compared across different solver prefixes.** Every probe builds a fresh
   solver, so `Solver_7_` vs `Solver_8_` made every pair "differ" and *everything*
   read live — including the negative control. Fix: key by the suffix after
   `petsc_options_prefix`.
2. **Options read before the solve.** Most configuration is written at `_build`,
   during the first solve, so a pre-solve snapshot reported `strategy` and
   `preconditioner` inert when they are not.
3. **Blind observables.** A *pressure* nullspace need not move the velocity; a
   *preconditioner* flag need not move the answer at all. Three of four "still inert"
   verdicts were artefacts of measuring `|u|` alone. Adding `|p|` and iteration counts
   cleared them.

**The negative control caught (1).** A synthetic property that provably does nothing
must come back inert; when it came back live, the instrument was measuring noise. No
sweep of this kind is worth reading without one.

### Validation: it rediscovered two known defects unprompted

Run against `development`, the sound instrument independently reported inert:

- **`solver.strategy`** — accepted three values, configured one (filled in #471);
- **`preconditioner` on a single-field solver** — the #478 gate.

Both are known-true, which is what makes the negative results below trustworthy.

### Cleared — live once tested where they can matter

`tolerance`, `penalty`, `saddle_preconditioner`, `preconditioner` (Stokes),
`consistent_jacobian` (live on a *nonlinear* problem: nonlinear iterations 10 → 5),
`petsc_use_nullspace` (needs an actual velocity null mode — free slip),
`petsc_use_pressure_nullspace` (live on `|p|`, invisible in `|u|`),
`constant_nullspace`, `petsc_use_constant_nullspace`, `smoothing`,
`smoothing_length`.

### One unresolved: `multiplier_schur_pc` — verify or remove

It **is** read (it swaps the multiplier block's Pmat term to the 1/μ Schur mass), so
it is not dead code. But no observable moved for either value: `|u|`, `|p|`, outer
KSP iterations and the assembled `|Pmat|` were identical at constant viscosity and at
η contrast 1e4 and 1e6 — including in a case where the solve *failed* identically
both ways. And `|Pmat|` equalled `|J|` exactly, in both settings.

Not filed as a dead flag, because it is read and the negative could still be a
problem too easy to discriminate. Filed as **verify-or-remove**: either produce the
case where it earns its keep, or delete it. Note `Stokes_Constrained` deliberately
*rejects* `saddle_preconditioner` ("the Schur preconditioner is built
automatically"), so that class owns its Pmat by design — which is the context this
flag lives in.

### A property that gets it right

`constant_nullspace = True` on a Dirichlet problem raises:

> `constant_nullspace=True is only valid for pure-Neumann scalar problems, but
> essential (Dirichlet) boundary conditions are present on: Bottom, Left, Right, Top.
> Remove them or set constant_nullspace=False.`

Names the offending boundaries and the remedy. This is the shape the other flags
should have, and the opposite of the defect class this audit is about.

### Minor

`solver.penalty` returns a sympy expression rather than the number assigned, so
`float(solver.penalty)` raises. Consistent with UW3's mathematical-object pattern,
but worth a docstring note.

## Not covered

Parallel (np>1) — the sweep is serial. No finding here has a mechanism that looks
partition-dependent, but that is an argument for a targeted np2 pass, not a substitute
for one. `SNES_MultiComponent` was not probed. Options with no petsc4py getter
(e.g. `ksp_gmres_restart`) are **unauditable** by this method and are not claimed either
way.
