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
- **A solve with memory.** A time step whose residual carries the last step's answer through a history (`DuDt` or `DFDt`: advection-diffusion, a viscoelastic Stokes step) starts from that answer, and `||F(0)||` measures the whole field rather than the step's change. Under the `max`, a step short against the diffusion time stops near `rtol` of the whole field: on test_0008's advection-diffusion box at tolerance 1e-4, the error against a 1e-12 run after 300 steps is 9.1e-5, against 1.4e-6 under PETSc's own test, and test_0008 fails. Such an equation declares a zero reference (`_reference_residual` returns 0), so a step is measured against its own first residual, PETSc's default; stationary equations keep the `max`. The reference is still recomputed at each call and is fixed within it. This split awaits a ruling; `_convergence_reference_rule` (`"increment"` or `"stationary"`) overrides an equation's declaration, so both rules can be measured on the same model.

**Strain-rate-dependent rheology.** `F(0)` is the residual function evaluated at `u = 0` with the boundary data in place, not the linearised operator. At zero strain rate the viscous stress `2 eta(edot) edot` is zero for any finite `eta(0)`. That holds for a power law with a strain-rate floor, a viscosity cap, the δ-soft-min yield and Drucker-Prager alike. So the cap or regularisation that sets `eta(0)` does not enter the interior rows. What remains is the forcing: the body force, the natural tractions, the stress history of a viscoelastic step, and the residual of a strong Dirichlet lift. That last one is a real strain rate in the boundary cells, evaluated with the viscosity it implies. A cold start measures exactly this today, so for these rheologies `max(||F(x0)||, ||F(0)||)` is the cold start's reference, kept across warm starts.

One term does carry the cap. A nonzero velocity imposed weakly, through Nitsche or a penalty, contributes `gamma mu(0) / h * g` at `u = 0`, and `mu(0)` is the cap. On such a model `||F(0)||` can be set by the cap rather than the physics, and the targets would then be too loose or too tight. For that case the reference leaves the weak boundary terms out of `F(0)`, and so measures the forcing alone: body force, tractions, history and strong lifts. A zero-valued weak condition, such as free slip by Nitsche, contributes nothing at `u = 0` and needs no exception. The evidence below records `||F(x0)||`, `||F(0)||` and the forcing-only norm on each fixture, so the case is seen rather than argued.

## 2. The targets

The nonlinear target is `T = max(rtol * F_ref, atol)`. `rtol`, `atol` and `stol` are the user's: `tolerance`, or `snes_rtol`, `snes_atol` and `snes_stol` when the user sets them (#924).

Each linear solve of the call stops once its residual is below `eta * T`, with `eta = 0.1`. This is the floor of inexact Newton: a linear solve need only bring the next residual well inside the Newton target. It is applied as a reduction of the linear solve's own first residual, `eta * T / ||F(x_k)||`, not as an absolute tolerance, because the KSP need not measure `||F||`. GMRES with left preconditioning, the scalar route's default, measures the preconditioned residual, about 40 times smaller there; the floor as a KSP atol stopped a Newton step's linear solve at iteration 0, and the SNES ended on a zero step above its target (test_0028's time step). FGMRES, the saddle-point and rotated outer KSP, measures the true residual, where the two forms agree. The relative tolerances stay where they are, as the forcing term of the early steps; when the residual is large the floor never binds. The floor goes on the outer KSP and on every fieldsplit sub-KSP (pressure and velocity), because each sub-solve's relative tolerance is measured against what it is handed. It is bounded below by `10 * eps * ||A||_F * ||u||_2`, so no solve is asked to beat round-off.

**Thresholds that follow the target.** A rotated solve projects an admitted near-null rotation out of its velocity preconditioner when the mode's Jacobi-scaled residual is at or below `max(1e-10, min(1e-6, 100 tol))` (#964, round 4). What projecting costs scales with how far the velocity solve must go. At tol 1e-9 it paid at 62 tol and cost at 180 tol, and at tol 1e-11 a mode at 6.2e-8 ran the velocity solve into its cap. #964 keys the threshold on the solver tolerance. Under this policy the velocity solve stops at a floor set from `T`, so the threshold should be taken from the same per-call target, and there is one rule for both. It is then fixed per call, like the floor, and a re-solve whose target changes re-decides it. #964 rebuilds the whole workspace when the tolerance changes. `T` changes at every call of a time loop, so the helper should re-decide only the projection, which needs the modes' residuals (kept from the last assembly) and not a new hierarchy. The evidence arms record the threshold and the modes projected.

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
- **KSP:** `KSPSetPreSolve` on the outer KSP sets its rtol to `max(rtol_user, min(eta * T / ||b||, eta))` before each linear solve, where `b = F(x_k)`, and the call's end restores the tolerances and clears the hook. While the SNES has not converged `||F(x_k)|| > T`, so the ratio is under `eta`; the cap only guards a route that stops elsewhere. The sub-KSPs from `PCFieldSplitGetSubKSP` take the same form against what they are handed (still to do).
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
- SolCx and the notch (Drucker-Prager, δ-soft-min) on the saddle-point route, and a zg95 viscoplastic run on the rotated route, for the strain-rate-dependent case;
- a model with a nonzero weakly imposed velocity, for the one term that carries the viscosity cap;
- a warm-started advection-diffusion time loop and an evaluation projector, for the Scalar and Vector routes.

Each arm records Newton iterations, linear iterations at every level (outer, velocity and pressure, as totals rather than last-application samples), the answer and its difference from the current arm, and wall time.

The bar to merge: answers agree to within the solve tolerance; Newton iterations do not increase; linear work does not increase on any fixture. A tier-B test pins the contract on every route: a warm re-solve of a converged problem stops at the same absolute residual as the cold solve, and takes no linear iterations it does not need.
