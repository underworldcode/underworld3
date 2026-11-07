# One convergence reference for every solver

> **Status**: design for review (2026-11-08). No code yet. It answers Louis's ruling on #911: "This is a persistent problem that we encounter in various solvers. PETSc does not respect the original residual of the solve when we warm start. We should try to find a DRY solution to this problem." The inventory it rests on is hyperion's, posted on #911 against development f31edcb1.

## The problem

Every UW3 solver hands PETSc a relative tolerance, and PETSc measures it against a quantity that a warm start shrinks.

- **The nonlinear test.** `SNESConvergedDefault` fixes its target at iteration 0 as `rtol * ||F(x0)||`, the residual of the guess this solve started from. A warm start begins near the answer, so it is asked for `rtol` of an already small residual. It either does needless iterations or cannot finish.
- **The linear test.** `KSPConvergedDefault` measures against `||B b||`, the preconditioned norm of its right-hand side. Inside Newton that right-hand side is the current residual, which shrinks at every step, so the linear target shrinks with it. In the last steps the target falls below round-off. The cause is the same in both: the reference is the current residual, not the problem's.

Measured on test_0022's rotated solve (power law, tolerance 1e-11, macOS, five increments): the Newton residuals ran 4.70, 1.97, 0.547, 0.0309, 9.8e-5, 1.15e-9. Newton's own target, `rtol * ref + atol`, is 5.7e-11 and reachable. The round-off floor of the rotated system, `eps * ||A||_F * ||u||_2`, is 2.9e-12. The next increment's outer linear solve was asked for 1e-11 of its 1.15e-9 right-hand side, which is 1e-20, and its pressure sub-solve for 3.3e-13 of what it was handed. Neither can get there. On CI hosts the file then took 18 to 52 minutes, nearly all of it in that one increment (#985).

UW3 already works around the nonlinear half in five separate places, each with its own reference (inventory, table 3):

| route | reference | survives a warm start |
|---|---|---|
| rotated loop | `max(r0, ||F(0)||)` | yes |
| continuation | `F_ref = ||F(0)||` | yes |
| difficulty-probe resume | `tolerance * fnorm0` of the first report | in part |
| Picard warm-up | `rtol * ||F||` at the solve's start | no |
| `SaddlePt` warm start | `atol = tolerance * ||F(guess)||` | no |

The Scalar, Vector and MultiComponent solvers have no warm-start handling, and their `solve()` is warm by default once a solve has succeeded. No route handles the linear half. The rotated bundle sets no `ksp_atol`, and the native routes have `ksp_atol = 1e-6 * tolerance` as their only floor.

## 1. The reference

Each `solve()` call measures one reference residual at its start and carries it through every Newton step, line search, retry and Picard stage of that call:

`F_ref = max(||F(x0)||, ||F(0)||)`

`F(0)` is the residual a cold start would have seen. It depends on the problem's forcing and boundary data, not on the guess, so a warm start and a cold start of the same problem get the same targets. The cost is one residual evaluation at `u = 0`, which the rotated loop already pays. The `max` keeps the cold-start reference when the guess is worse than zero.

Three cases need a rule of their own:

- **Forced only by a prescribed datum.** `F(0)` measures only that part of the forcing which reaches the unconstrained rows. When the datum is the only forcing it is zero, so the reference is the residual of the lifted state, `F(x_lift)`. Today that residual is taken one increment late (#883).
- **A null space.** The reference is measured in the same quotient space as the test. Each null mode is projected out of `F(0)` before its norm is taken, so a component that no iterate can reduce cannot enter the reference (#884).
- **A re-solve in a time loop.** The reference is recomputed at each call, because the forcing changes between steps. Within a call it does not change.

## 2. The targets

The nonlinear target is `T = max(rtol * F_ref, atol)`. `rtol`, `atol` and `stol` are the user's: `tolerance`, or `snes_rtol`, `snes_atol` and `snes_stol` when the user sets them (#924).

Each linear solve of the call stops once its residual is below `eta * T`, with `eta = 0.1`. This is the absolute floor of inexact Newton: a linear solve need only bring the next residual well inside the Newton target. The relative tolerances stay where they are, as the forcing term of the early steps; when the residual is large the floor never binds. The floor goes on the outer KSP and on every fieldsplit sub-KSP (pressure and velocity), because each sub-solve's relative tolerance is measured against what it is handed. It is bounded below by `10 * eps * ||A||_F * ||u||_2`, so no solve is asked to beat round-off.

Eisenstat-Walker, `SNESKSPSetUseEW`, stays on the saddle-point route. It sets the forcing term from the nonlinear convergence rate. Near convergence that rate is fast and the forcing term is small again, so EW alone does not prevent the last-step problem; the floor does. The two compose: EW sets the relative tolerance, and the floor bounds it from below.

## 3. One helper, in one place

`SolverBaseClass` gets two methods:

- `_reference_residual(x0)` returns `F_ref` with the datum and null-space rules above;
- `_apply_convergence_targets(F_ref)` sets the SNES atol to `T`, and the outer and sub-KSP atols to the floor.

They are called from two places:

- `_snes_solve_with_retries`, which already gates every native `snes.solve` (Scalar, Vector, MultiComponent and SaddlePt) and already saves and restores tolerances around the solve;
- the rotated loop, which bypasses that gate.

They replace the continuation's `F_ref`, the Picard warm-up's reference, the `SaddlePt` warm-start atol and the difficulty probe's resume anchor. The rotated vector route uses them like the rotated saddle route.

The adjoint solve is one linear solve from zero, so PETSc's own reference (`||b||`) is already the right one there and it is left alone. The projectors used by evaluation are warm-started on every call; with the helper their reference is `||F(0)||`, the right-hand side's scale.

## 4. PETSc mechanisms

- **SNES:** `SNESSetTolerances(atol=T)` once per call. `SNESConvergedDefault` tests `fnorm < abstol` before the relative test, and does so at iteration 0 unless `forceiteration` is set. A warm start already inside `T` therefore ends at iteration 0 with no linear solve, and PETSc's own relative test against `F(x0)` can only end a solve later. No custom convergence test is needed, and every SNES type, line search and FAS path keeps working.
- **KSP:** `KSPSetTolerances(atol=floor)` on the outer KSP and on each sub-KSP from `PCFieldSplitGetSubKSP`. The floor is constant within a call, so it is set once and needs no per-step hook.
- **`KSPConvergedDefaultSetUIRNorm` and `UMIRNorm`:** these choose the KSP's reference when the KSP itself starts from a nonzero guess. UW3 never warm-starts a KSP (`setInitialGuessNonzero` appears nowhere in `src`), so they are not needed. They become relevant if a KSP warm start is added later.
- **A custom `SNESSetConvergenceTest`:** not needed, because the atol route does the job without a callback.

## 5. Issues

| issue | settled by |
|---|---|
| #911 and #985 | the linear floor. The last increment stops at `eta * T` instead of chasing 1e-11 of 1e-9. |
| #924 | the helper reading the user's `snes_rtol`, `snes_atol` and `snes_stol` on the rotated route, as `_resolve_owned_option` already does on the native routes |
| #883 | the datum rule, which takes the reference from the lifted state at the first increment |
| #884 | the null-space rule, which needs the gauge modes collected before the reference is measured; on an uncached rotated solve today they are collected after the first assembly |
| the pressure rtol default (feature/rotated-pressure-rtol-default) | an overlap, not settled. The floor removes the last steps' pressure caps; the 1e-2 default cuts the pressure work of every step (kaiju measured 28-57 % of Stokes time). The measurement below decides whether both are kept. |

## 6. Evidence before merge

`petsc_generic_snes_solvers.pyx` is the protected solver file, so nothing merges on argument alone. Four arms on one build, each switchable at run time:

1. current;
2. the reference and floor;
3. the pressure rtol of 1e-2;
4. both.

The fixtures:
- test_0022 (laptop, plus hyperion with Nehalem and Zen forced);
- the rotated parallel set at np 2 and 4;
- the zg95 rotated drivers (kaiju);
- SolCx and the notch on the saddle-point route;
- a warm-started advection-diffusion time loop and an evaluation projector, for the Scalar and Vector routes.

Each arm records Newton iterations, linear iterations at every level (outer, velocity and pressure, as totals rather than last-application samples), the answer and its difference from the current arm, and wall time.

The bar to merge: answers agree to within the solve tolerance; Newton iterations do not increase; linear work does not increase on any fixture. A tier-B test pins the contract on every route: a warm re-solve of a converged problem stops at the same absolute residual as the cold solve, and takes no linear iterations it does not need.
