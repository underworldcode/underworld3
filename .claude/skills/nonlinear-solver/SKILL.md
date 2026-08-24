---
name: nonlinear-solver
description: How to make a hard nonlinear Stokes solve (Drucker-Prager / yield-stress viscoplastic) CONVERGE reliably in Underworld3 the way the working recipe actually does it — automatic warm-start (one Picard step on a cold start) plus a MULTI-SOLVE δ-continuation (constant δ per solve, warm-start the next, sharper δ), the consistent-Newton tangent, and a non-symmetry-safe multigrid smoother. Reach for THIS when a viscoplastic solve stalls / diverges and you are about to hand-tune PETSc options, ramp δ, or "just add a monitor". It carries the CONFIG TRAP LIST — the setup mistakes that each produce a different failure a few steps in — and the one thing you must NOT do (ramp δ inside a single SNES solve). For the yield-law maths and which tangent per model, see `plasticity-solvers`.
---

# nonlinear-solver

The recipe that gets a **hard viscoplastic (Drucker–Prager) Stokes** problem to
converge, and — more importantly — the list of setup mistakes that stop it. The
central lesson from the Spiegelman hard-case study (`η_bg=1e26`, `V=10`): every
failure was a **solver-configuration** error, not a bad Jacobian. If the correct
setup is a minefield for an expert, that is an API regression — so the goal is to
make the correct path the default path.

Design of record: `docs/developer/design/nonlinear-solver-homotopy-warmstart.md`.
Yield-law maths, tangent-per-model, quadratic-convergence check: `plasticity-solvers`.

---

## The recipe (what actually converges)

1. **Warm start.** Start the continuation at **large δ**, where the yield surface is
   smooth and the problem is easy, and take **one Picard step** into the Newton
   basin. One Picard step is defect-correction iteration 1 — contractive, cheap. From
   a *warm* iterate, take **no** Picard step (it wastes the good quadratic start).
   A cold `v=0` start is safe on its own terms: `ε̇=0` makes `η_pl` infinite, which
   the soft-min carries to the viscous branch (see the trap list for the one form
   that must be written carefully).

2. **If a single solve at the sharp surface fails**, escalate in this order:
   **grid sequencing first** (solve coarse, transfer, re-solve fine — the
   measured 2-3x win at the notch), and only then a **multi-solve
   δ-continuation** as the rescue of last resort. The δ-discipline, when you do
   reach for it: hold δ **constant** for a full nonlinear solve to tolerance;
   warm-start the next, smaller δ from that converged state; march down to the
   sharp surface. δ is a `constants[]` atom, so each step is a recompile-free
   `PetscDSSetConstants` update.

   The packaged driver is `stokes.solve(homotopy=True)` (also callable directly
   as `underworld3.systems.yield_continuation`; tune with
   `homotopy_options=dict(delta0=…, down=…, dmin=…, entry_maxit=…,
   step_maxit=…)`). **Treat it as a rescue, not the default**: the evidence
   that once made a δ-march the recommended entry point was retracted (it
   rested on a unit-scaling error — `plasticity-solvers` carries the ruling
   and the surviving evidence), and the driver's documented cold-start
   guarantee does not currently hold (issue #473: entry can fail on a
   pressure-dependent yield, and the step control is effectively one-shot).
   Newton + the automatic Picard entry handles the standard cases without it.

3. **Consistent-Newton tangent** for non-elastic DP (`consistent_jacobian=True`);
   **Picard** for elastic VEP — see `plasticity-solvers` for the per-model table.

4. **`bt` line search** with the consistent tangent on a smooth (δ>0) surface.

---

## DO NOT ramp δ inside one SNES solve

Ramping δ **inside a single SNES solve** (a `SNESSetUpdate` callback that sharpens
the yield surface between Newton iterations) is **proven dead** — it diverges
`DIVERGED_LINEAR_SOLVE` after ~2 iterations and grinds for ~2 hours, **even on the
proven solver config**. Mechanism: the continuation only sharpens δ from a
**converged**, well-conditioned iterate; the in-SNES ramp sharpens δ **mid-solve**
at a far-from-solution iterate where the consistent-Newton Jacobian on a sharpening
surface is ill-conditioned and the linear solve fails. **Hide the *continuation*,
not the *ramp*.** (An in-SNES ramp API once shipped and has been removed from the
source entirely — use the multi-solve continuation above; `plasticity-solvers`
carries the yield-law substrate and the evidence on when a δ-march is worth it
at all.)

---

## CONFIG TRAP LIST

Each of these produces a *different* failure a few steps in — that is why the hard
case felt like whack-a-mole. Check them first.

| Trap | Symptom | Fix |
|---|---|---|
| Consistent Newton makes the velocity block **non-symmetric**; a Chebyshev/Richardson MG smoother assumes SPD | smoother diverges / stalls → `DIVERGED_LINEAR_SOLVE` or an endless grind | **Now the default** — the FMG bundle ships `mg_levels_ksp_type=gmres` + `pc_type=sor` + `norm_type=none`. Only an issue if you override it, or on GAMG (which uses PETSc's chebyshev default) |
| `preconditioner="fmg"` (vs explicit `pc_type=mg` + manual mg opts) | outer KSP "converges" in **1 iteration** → no real Newton correction → stall → `DIVERGED_LINE_SEARCH` | use explicit `pc_type=mg` with the smoother opts above; bound the outer KSP (`ksp_max_it`~80) so a hostile step fails fast |
| Cold plastic start `v=0`, or any rigid/unyielded point | `DIVERGED_FNORM_NAN` at iteration 0 | **Not** a div/0: `ε̇=0` gives `η_pl=+inf`, which `Min` and the sqrt soft-min carry correctly to the viscous branch. Only a soft-min form that computes `η_ve·η_pl/(η_ve+η_pl)` breaks (`inf/inf`). Fixed in the power-mean; if you hand-roll a blend, write the harmonic mean as `η_ve/(1+η_ve/η_pl)`. **Do not reach for a strain-rate floor** — it hides this rather than fixing it |
| LU velocity block with all-Dirichlet-ish BC | pressure nullspace singular | attach the Stokes nullspace / avoid a bare LU there |
| Hand-rolled `snes_monitor` to "see what's happening" | you read residuals but miss the tell | use `solve_with_diagnostics` / `get_snes_diagnostics` instead (below) |

**The diagnostic tell:** `solver.get_snes_diagnostics()["linear_iterations"] ≈ 1`
per Newton step means the linear solve is doing **no real work** (the FMG-1-iteration
trap). A healthy consistent-Newton solve does real Krylov work each step and
converges quadratically. Use `solve_with_diagnostics()`, not a hand-rolled monitor.

---

## Automatic warm-start (Layer 1 — landed)

`solver.has_solution` is a **public, read-only** status flag: `True` only after a
solve whose SNES converged; reset on a structural rebuild (remesh / adapt /
mesh-mover — the `is_setup=False` hook); kept through coefficient changes (viscosity,
δ, BC values, time step). A **diverged** solve leaves it `False`, so the next solve
auto-cold-starts rather than warming off a corrupted iterate.

On a **cold** (`zero_init_guess=True`) Stokes solve under the **consistent-Newton
tangent**, a single Picard step is now taken automatically (reusing the existing
`picard=1` machinery). The default (frozen) tangent path is bit-identical.

```python
stokes.consistent_jacobian = True
stokes.solve()                 # cold → one automatic Picard step, then Newton
if stokes.has_solution:
    ...
```

---

## Implementation status (this line of work)

- **Layer 1a — DONE:** `has_solution` + cold consistent-Newton Picard warm-up
  (`petsc_generic_snes_solvers.pyx`; test `test_0201`).
- **Layer 1b — DONE:** `zero_init_guess` is tri-state — `None` (default) auto-detects
  from `has_solution`, `True` forces fresh, `False` insists on warm. Note warm and cold
  agree only to the *convergence tolerance*, not bitwise.
- **Layer 3 — DONE:** the FMG velocity smoother defaults to `gmres`+`sor` with
  `mg_levels_ksp_norm_type=none` (fixed-cost V-cycle), unconditionally — see
  "Multigrid depth" below.
- **Layer 2 — SHIPPED, DEMOTED TO RESCUE:** the model advertises the homotopy
  (`supports_yield_homotopy` / `_yield_homotopy_control`) and
  `stokes.solve(homotopy=True, homotopy_options=...)` runs the residual-guided
  continuation, returning the march summary. The doctrine that made this the
  recommended entry point was retracted (unit-scaling error — see
  `plasticity-solvers`), and its cold-start guarantee is broken (issue #473);
  use it after Newton + Picard entry and grid sequencing have failed.

---

## Multigrid depth — how to measure a smoother honestly

**A two-level hierarchy is a coarse-grid correction, not a V-cycle.** Smoother
comparisons made on one are misleading: the gmres-over-richardson margin measured on
the Spiegelman notch is only 5 % at 3 levels but **25 % at 4** (ρ per V-cycle 0.746 →
0.560), because a deeper cycle applies the smoother on more coarse operators. Judge a
smoother at depth or not at all.

To get depth without a monster problem, refine a **deliberately ultra-coarse NESTED
base**: `make_notch_mesh.py 1` (492 cells) + uniform `refinement=N` gives 3 levels /
7,872 cells at `N=2` and 4 levels / 31,488 at `N=3` — deeper *and* smaller than the old
2-level 38,580-cell setup. In MG you want the coarsest grid as coarse as it can be
before the problem breaks down.

- **Never use a non-nested hierarchy** here — it does not give strong MG convergence
  (maintainer ruling). Uniform refinement nests by construction.
- Accepted tradeoff: uniform refinement does **not** snap new boundary nodes back to
  the analytic notch arcs (no CAD/EGADS model attached), so the corner geometry is
  frozen at the coarse mesh's chords on every level.

Measure with `fmg_contraction_probe.py` (ρ_MG per V-cycle; `<0.5` healthy, `0.8–0.95`
struggling, `≥0.98` hangs) or `smoother_depth_sweep.py` (pays the mesh build + viscous
seed once, sweeps smoothers in-process) in the Spiegelman study.

**`solve_report` cannot see the smoother.** It records the *Newton* contraction; the
outer KSP is Eisenstat–Walker-collapsed to ~1 iteration/step, so the smoother's work
hides inside the velocity sub-block. Probe the `fieldsplit_velocity_` sub-KSP directly.

**A smoother will not rescue small ξ.** At the hard corner the failure is operator
conditioning — the coarsest grid cannot represent the viscosity contrast — and at 4
levels *every* smoother fails there (richardson outright, gmres with ρ>1). Use the δ/ξ
continuation to stay in the solvable region.

## FMG on an ADAPT-ON-TOP child (locally refined meshes)

An `adapt()` child carries its **own custom-P geometric MG tail** — subsampled to
one level per **DOUBLING of h** (`mg_coarsening_ratio=2.0`, the `adapt()` default)
— on `child._custom_mg_coarse_meshes`, and solvers built on it pick it up
automatically. So the usual advice above ("never use a non-nested hierarchy") is
satisfied without you assembling anything:

```python
child = base.adapt(metric, max_levels=3, engine="edge_split")
stokes = uw.systems.Stokes(child, velocityField=v, pressureField=p)
stokes.solve()          # pc=mg auto-attached off the child's tail
```

Requirements and traps, all measured:

- **Build the base with `refinement>=1` for a deeper tail.** The custom-P tail
  always starts at the BASE mesh — with `refinement=0` it is
  `[base] + the intermediate doubling levels`, so there IS a coarse grid — but
  the uniform base levels extend it downward, and in MG you want the coarsest
  grid as coarse as it can be.
- **Keep the GRADED tail.** `_adapt_nested` stores one MG level per doubling of
  resolution (`_subsample_mg_levels`; per-bisection-pass levels were measured
  2.3–7.3× slower). Handing the solver a base-only tail instead — coarse base
  straight to the fully adapted mesh — **triples the V-cycle count**.
- **V-cycle counts are insensitive to element quality here, and that is a PASS not
  a failed measurement.** On a fault child the velocity block takes 2 iterations
  (iso) or 2–3 (TI) across meshes ranging from 156° to 105° max angle. The
  geometric hierarchy's coarse spaces come from the mesh hierarchy, not from the
  fine operator, so shape does not move it — which is exactly what makes
  adapt-on-top viable. **If you want a solver-side probe of mesh quality, use
  GAMG**, which does respond (iso 79 → 64 velocity iterations with `repair=True`).
  That is now actionable: `solver.preconditioner = "gamg"` is **respected** on an
  adapt child (#530) — before that guard the opportunistic pickup silently
  clobbered it back to `pc=mg`, so any FMG-vs-GAMG comparison was vacuous.
- **Single-field solvers get FMG too** (#478/#534): `preconditioner = "fmg"` on a
  Poisson/projection-class solver builds the custom-P tail over the mesh's own
  `dm_hierarchy` — the section is not Stokes-or-adapt-child only.
- **`relax()` can trip #424.** On a relaxed, unrepaired child the barycentric
  transfer hit 22 zero columns and fell back to the DENSE global RBF builder — a
  performance cliff, not just a warning.
- **Every PC degradation is recorded in `solver.pc_fallbacks`** (#534) — the
  requested/installed/reason record for the #424 barycentric→rbf retry, a
  collapsed hierarchy, a declined pickup. Read that, don't scrape warnings.
- **`repair=True` invalidates the any-degree nested transfer** (a flipped cell can
  straddle two coarse cells), so degree ≥ 2 falls back to the geometric builder.
  The exact ½,½ vertex prolongation survives, because flips move no vertex.
- Under **rotated free-slip** the mesh-owned adapt tail is picked up automatically
  too — the rotated KSP resolves hierarchies through the same
  `custom_mg.build_transfers` rule (#467 fixed the old silent GAMG fallback). See
  the `adapt-on-top-faults` skill for the plain-refined-mesh case, which still
  needs `set_custom_fmg`.

Companion skills: **`adapt-on-top-faults`** (building the child, engines, repair,
band sizing), **`adaptive-meshing`** (the mover, and `relax(pin_bands=...)` for
relaxing a mesh that was refined onto an interface).

## The Schur complement: pair the penalty with FMG, never with GAMG

**Symptom this is for**: the velocity block's iteration count is rock solid but
the pressure sub-solve wanders into the hundreds and eventually stops
converging.

**First: it is probably not the pressure block.** `S = -B A^-1 B^T` is applied
*through* the velocity solve, so a velocity solve that exits at its iteration
cap makes the Schur operator inconsistent between applications — and no Krylov
method converges against an operator that moves under it. The pressure block
then caps too, and the outer flounders. Measured on SolCx (eta 1e6, P2-P0disc,
h=1/30), changing **only** `fieldsplit_velocity_ksp_max_it`:

| velocity cap | sec | outer | pressure/app | velocity/app |
|---|---|---|---|---|
| 200 (default) | 976.0 | 44 | **200.0** | **200.0** |
| 5000 | **25.6** | **2** | **30.0** | 618.0 |

**38x from a number that is not in the pressure block**, and the velocity error
is identical in both rows. Before tuning the Schur solve, check whether either
block sat at exactly its cap — `solve_report.sub` gives iterations and
applications per block, and a per-application count equal to the cap to the
digit is the tell.

**Then: the penalty is the lever on the Schur count, and it needs FMG.**
`stokes.penalty = lambda` adds `lambda*mu*(div u)(div v)`, which makes the
eta-scaled mass matrix a better approximation to S. Matched on one mesh
(2592 cells), same discrete solve, only the velocity preconditioner differs:

| lambda | velocity PC | sec | outer | Schur/app | velocity/app | velocity total |
|---|---|---|---|---|---|---|
| 0  | GAMG | 15.49 | 2 | 125.5 | 94.7 | 24802 |
| 0  | **FMG** | **3.88** | 1 | **59.0** | **8.8** | **546** |
| 10 | GAMG | 20.68 | 7 | 22.3 | **199.9 capped** | 33976 |
| 10 | **FMG** | **3.06** | 1 | **18.0** | **13.5** | **270** |

- **With FMG, `penalty = 10` improves every axis at once**: 21% faster, Schur
  count 3.3x smaller, total velocity work halved. FMG absorbs grad-div
  augmentation (8.8 -> 13.5 iterations per application); GAMG does not
  (94.7 -> capped).
- **With GAMG, do not use it at all.** The same `penalty = 10` makes the solve
  *slower* (15.49 -> 20.68 s), because augmentation is exactly what drives GAMG
  into its cap. Uncapping rescues it to 11.03 s but it still needs **833**
  iterations per application, and FMG is 3.6x faster on the same mesh.
  Feasible is not competitive.

**The accuracy cost is consistent, so it is safe to pair by default.** The
penalty is grad-div, not a true augmented Lagrangian — `div(P2)` is not inside
`P0`, so the term does not vanish at the discrete solution and it does perturb
the answer. But the perturbation converges away: same rate, and the gap shrinks
under refinement.

| cells | lambda=0 v err | rate | lambda=10 v err | rate | gap |
|---|---|---|---|---|---|
| 648   | 2.112e-1 | —    | 2.327e-1 | —    | 1.102 |
| 2592  | 1.266e-1 | 1.67 | 1.376e-1 | 1.69 | 1.087 |
| 10368 | 8.727e-2 | 1.45 | 9.305e-2 | 1.48 | **1.066** |

For a pressure-dependent constitutive law use the mechanical pressure,
`p_mech = p - lambda*mu*(div u)`; the raw `p` is the multiplier.

**Traps.**

- **FMG needs a refined base or you silently get GAMG.** Measured:
  `refinement=0` -> one hierarchy level -> default velocity PC is `gamg`;
  `refinement=2` -> `mg`. So `penalty` set "with FMG" on an unrefined mesh is
  actually the harmful GAMG pairing. Check
  `snes.getKSP().getPC().getFieldSplitSubKSP()[0].getPC().getType()`, or read
  `solver.pc_fallbacks`.
- **Scaling `saddle_preconditioner` by a constant does nothing** — it does not
  change the Krylov subspace. `1/eta` and `101/eta` both give 28 iterations,
  identical to every digit, so an "AL-matched" `1/(eta*(1+lambda))` cannot help.
  The 1/eta *weighting* itself is worth 1.9x (28 vs 52 with a flat `1`).
- **Eisenstat-Walker is inert under `snes_type=ksponly`** — identical iterations
  and error on or off. And `outer 1` is not an EW artefact: it is what a full
  Schur factorisation gives when the Schur complement is solved well.
- Measurements: `~/+Simulations/pressure_schur_625/` (#625).

## Gotchas

- **`./uw build` → `amr-dev` env**; verify `uw.__file__` is the worktree site-packages.
- **Run VEP/consistent-Newton tests UNFORKED** — `pytest --forked` SIGABRTs (fork of
  multithreaded PETSc).
- Benchmark **every** default change — "Solver Stability is Paramount".
- ξ (rate-strengthening) is a **non-homotopic** regularisation: put a user loop
  *around* `solve()`, never inside the δ-march.

## Reference

- Design: `docs/developer/design/nonlinear-solver-homotopy-warmstart.md`,
  `jacobian-consistent-tangent.md`, `solver-strategies-catalogue.md`.
- Continuation driver: `underworld3.systems.yield_continuation`.
- Diagnostics: `SNES_*.get_snes_diagnostics()` / `solve_with_diagnostics()`.
- Related skills: `plasticity-solvers` (yield law + tangent per model),
  `free-surface-convection`, `adaptive-meshing` (mover + `relax(pin_bands=...)`),
  `adapt-on-top-faults` (locally refined children and their MG tail).
- Reconnection / refinement engines:
  `docs/developer/design/mesh-reconnection-and-delaunay-adapt.md`.
