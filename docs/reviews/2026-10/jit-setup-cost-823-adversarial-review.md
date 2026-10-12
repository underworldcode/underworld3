# JIT setup cost (#823): adversarial review record

Branch `bugfix/jit-setup-cost-823`, stacked on `bugfix/jit-petsc-include-dirs` (#828).
Three adversarial rounds, each run against the installed build of the commits it
reviewed, plus an independent measurement session (the tier-2 codegen work) that
profiled the notch and confirmed generated-C identity by header hash.

## What the branch does

| Change | Cost removed (Spiegelman notch, Newton) | Output |
|---|---|---|
| Memoised complete unwrap (`_unwrap_expression_complete`) | 563 s → 0.11 s for the viscosity | srepr-identical to the fixed-point `subs` loop, except two cases where the loop was wrong (below) |
| Identity-walk symbol scan (`_unique_symbols`), memoised sqrt guard, debug text only when debugging | 35 s, 16 s, 23 s | identical |
| Field values, gradients, coordinates, constant slots and unit-carrying parameters declared real | complex-plane evaluation in `Pow` | solutions bit-identical on 18 models; C differs (`fabs` for `sqrt(x**2)`) |
| Every Jacobian derivative through `uw.function.diff_wrt_field` (a real stand-in) | 131 s → 3.5 s for `uu_G3` | C byte-identical to plain `sympy.diff` where both compile |
| Power-mean sharpness `s` held as its own constant | 40 s → 0.05 s Newton source | solutions to 6.8e-13, same iterations |
| Shared-graph constants substitution, printer `re()` pass only when needed | 4.3 s, 2.9 s | identical |

Notch setup on `development`: 616 s (sqrt law) and 1,352 s (power mean). On the
branch: 10 s and 13 s; 18 s and 25 s including the C build. The whole generated header
of the notch, VEP and TI fixtures is byte-identical across the last two commits
(md5 checked by the tier-2 session). Full serial suite 3,014 passed; parallel np=2
passes except #826; five reference solutions bit-identical to the build without the
realness declarations, except the power mean (6.8e-13, same 18 iterations).

## Findings and their state

Round 1 (on the memoised unwrap and the first realness commit):

- **Field realness broke the Newton Jacobian** (found by the tier-2 session first):
  `sqrt(g**2)` becomes `Abs(g)`, and `sympy.diff` with respect to a field goes through
  an assumption-free `Dummy`, leaving `Derivative(u, u)` unprintable. Drucker-Prager
  with a yield floor at softness 0 failed to compile. Fixed by differentiating through
  a real stand-in (`diff_wrt_field`), which also fixed an explicit `Abs` of the unknown
  in any Jacobian, broken on `development`.
- **Coordinate realness turns a twice-differentiated kink into a `DiracDelta`**,
  which C cannot print. Fixed: one rule (`_without_dirac_deltas`) for the JIT printer
  and every evaluate path, dropping the term (its value away from the zero) with a
  warning.
- Tests that could not fail (the unwrap equivalence passed on the old algorithm; the
  include test passed wherever the headers were on the default path): replaced by a
  constancy-call count (25 vs 103) and a check of the generated `setup.py`.
- Smaller: a guard around a hard dependency (Charter §5), the process-global
  coordinate patch documented, the identity claim corrected (the old loop returned
  `Subs(...)` for a held `Derivative` and a meaningless result for a cycle).

Round 2 (on the stand-in):

- `uw.function.evaluate` raised `NameError` on a `DiracDelta` the JIT compiled.
  Fixed by the shared rule, applied at each evaluator's entry so constants keep their
  broadcast shape.
- The governing Jacobian-layout doc and the plasticity guide taught plain
  `sympy.diff`, and pointed at private helpers. The helpers are public
  (`uw.function.diff_wrt_field`, `derive_by_array_wrt_field`) and both docs teach them.
- `UWQuantity` realness raised on a `Fraction` and said "real" for NaN; fixed.
- The helper mis-handled a held `Derivative` and a plain number; fixed.
- Tests: a loose accuracy gate (1e-4 for a 1.3e-9 error) tightened to 1e-7; a
  "compiles" test that never compiled now solves against the viscous answer; a source
  census fails if a solver differentiates with `sympy.diff`.

Round 3 (on the sharpness atom and the round-2 fixes):

- The Clement evaluation path still raised on a `DiracDelta`; fixed.
- The warning was attributed to library frames and repeated in a loop when the delta's
  argument changed; it now points at the user's call and warns once per call site, on
  every rank for a (rank-local) evaluation and on rank 0 for the (collective) JIT.
- A field `Abs` beside a held `Derivative` lost its derivative; fixed.
- The census test's exact count was a drift detector; dropped, and the scan covers
  `systems/`. The sharpness test now asserts a δ change does not recompile.
- `Float(0.001)` for an exact constant (Charter §6); now `Rational(1, 1000)`.

## Attacks that failed

- Memoised unwrap vs the fixed point: Piecewise, Subs, matrix-valued `.sym`, `None`
  sym, a diamond DAG (no false cycle), units with scaling on and off, `.sym` changed
  between calls, all four modes.
- `_unique_symbols` vs `atoms(Symbol)` on 18 awkward types; the sqrt guard vs
  `.replace(simultaneous=True)` on 161 cases.
- Determinism: generated C byte-identical across `PYTHONHASHSEED` 1, 2, 3 for 13
  models; no `Dummy` name reaches C.
- The sharpness atom tracks δ through the property, the homotopy control and the
  anchor reset; two models get distinct atoms; the JIT key is unchanged by a δ change.
- No spurious `DiracDelta` warning in any VP or VEP Newton solve at softness 0.

## Known and accepted

- For VEP and TI power-mean models δ no longer appears in the constants manifest
  (only `s` does), so the step transcript records `{s_{y}}`, from which δ follows.
- The `BaseScalar` realness patch applies to every `sympy.vector` coordinate system
  in the process (documented at the patch).
- Plain `sympy.diff` with respect to a field in user code now leaves
  `Derivative(u, u)` beside a field `Abs`; the guides direct users to
  `diff_wrt_field`.

## Out of scope, filed

- #826: `test_1063` [ti] topography 1.8 % off its anchor on the anchor's own mesh,
  identical on `development` and with every change here switched off.
- #827: every viscoelastic Stokes solve prints a PETSc error banner from
  `_check_velocity_preconditioner`.
