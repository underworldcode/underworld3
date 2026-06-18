---
title: "Multilevel solvers for nonlinear Stokes — strategy, findings, and a decision to park"
status: PARKED (2026-06-16)
---

# Multilevel solvers for nonlinear Stokes — strategy note + decision

**Decision (2026-06-16): park this. Capture the understanding, ship nothing yet.**
The conceptual payoff of the investigation was large; the *implementation* payoff
*right now* is small relative to the engineering load, and the codebase wants a
cleanup before features of this size land. This note records the landscape, the
findings, and the conditions under which it is worth coming back. Companion docs
with the detailed experiments: `snesfas-feasibility.md` and
`snesfas-vanka-feasibility-study.md`. **The working prototypes are preserved in-repo
at `docs/examples/snesfas_investigation/`** (curated set + README; the full scratch
set is in `~/+Simulations/snesfas_spike/`). No `src/` changes were made — this was a
spike.

## What we were chasing

A robust, scalable solver for **nonlinear Stokes** in the regime that actually
matters for geodynamics: **strong nonlinearity (plasticity, T-dependent viscosity)
*and* high viscosity contrast — which co-occur, because the nonlinearity is what
creates the contrast.** FMG (the geometric-MG-on-the-velocity-block preconditioner,
#231) is the current robust default; the question was whether SNESFAS / Vanka /
grid-sequencing could do better where FMG-inside-Newton struggles.

## The conceptual map (the durable result)

Three multilevel strategies, and the key distinctions that took a while to get
straight:

- **FMG (Galerkin).** Coarse operator `A_H = R A_h P`, built *algebraically from the
  fine matrix* — it never re-visits the PDE. The multigrid acts on the **velocity
  block only**, inside the Schur fieldsplit; the saddle/pressure coupling and the
  viscosity jump are handled by the Schur, *not* by a coarse-grid correction. That is
  why FMG is robust to high contrast: it keeps the hard, indefinite, high-contrast
  part **out of the multigrid**. Transfers can be approximate (barycentric / general);
  they only move a *correction* with homogeneous boundary values, and the smoother +
  Krylov clean up any imperfection — so it tolerates UW3's non-nested (curved /
  mover-deformed) hierarchies.

- **SNESFAS (re-discretisation + τ-correction).** Nonlinear multigrid on the **full
  coupled saddle**. Each level uses the *genuine re-discretised* coarse problem (not
  Galerkin — `-snes_view` confirms "Not using Galerkin… function evaluation"), plus
  the **τ-correction** that couples levels (the coarse grid solves the full problem
  with the restricted fine residual as forcing, and prolongs only the *change*). The
  smoother is a separate choice — **Vanka** (patch-local mini-Stokes solves) is one
  option for the saddle, *not* part of FAS itself. The τ-correction is the extra
  nonlinear machinery — **and it is exactly the part that is fragile to high
  contrast**, because it is a repeated coarse-grid *correction* across a jump the
  coarse grid cannot resolve.

- **Grid sequencing / nested iteration (Brandt).** Solve the re-discretised coarse
  problem (cheap, well-conditioned) to **find the basin**, interpolate up as the
  initial guess, solve the next level, repeat. Coarse grids are used for
  *initialisation*, never *correction* — so there is **no coarse-correction-across-
  the-jump**, hence it is robust to contrast. It lacks FAS's τ-correction (so it
  doesn't get the textbook MG convergence *rate*), but for problems that become
  *more ill-conditioned with refinement* it is the effective, robust tool: each
  level is only a bounded correction to the one below.

The unifying insight: **nonlinearity wants a coarse-grid correction (FAS); high
contrast forbids one (the coarse grid can't represent the jump).** Those are
opposite demands, which is why no single monolithic cycle covers both, and why the
robust workhorse stays "split the saddle off (FMG) + handle the nonlinearity by
Newton/Picard + continuation (timestepping)."

## What was actually demonstrated (iteration counts — timings were unreliable)

(Wall-clock numbers in the companion docs are *soft* — JIT, machine noise,
resolution/tolerance-sensitive. Trust the iteration counts.)

- **SNESFAS runs through UW3 with zero `src` changes** (petsc_options only) — `copyDS`
  + PETSc's FAS setup give coarse residuals. Mesh-independent on scalar nonlinear
  problems; beats Newton on stiff nonlinear diffusion.
- **Vanka works** on UW3's simplex P2–P1 once patches are built as the *support of
  each pressure basis function* (custom-IS `PCASM`, exact local LU) — the stock
  `PCPATCH` constructs are the wrong tool and fail. A geometric MG with that Vanka
  smoother is **mesh-independent (5/5/6 outer iters across 16× DOFs)** — needs a GMRES
  Krylov smoother (additive Schwarz alone diverges).
- **FAS-Vanka wins on plasticity** — viscoplastic τ_y=1: **2 nonlinear iterations vs
  10** for Newton+FMG and GAMG (the coarse problem is better conditioned, as
  predicted). But with *additive* Vanka it **fails at extreme contrast (10⁶)** no
  matter how hard you smooth — needs multiplicative Vanka + interface-aware coarsening.
- **The cascade helps iff the per-level solve is iteration-limited.** On SolCx it does
  nothing for FMG (already 1–3 iters → cascade is pure overhead) but rescues a
  struggling GAMG (**8→2 iters**). Its payoff is exactly proportional to how badly the
  cold fine solve struggles.
- **PETSc's `-snes_grid_sequence` does not work on UW3 meshes** — the built-in
  interpolator hard-fails (point location) on UW3's non-nested (boundary-snapped /
  mover-deformed) hierarchies and on the BC-constrained boundary DOFs. A hand-rolled
  cascade using `uw.function.evaluate` (robust, never hard-fails; BCs re-imposed per
  level) sidesteps this entirely.

## Why park it now

- **The marginal win is small in the current regime.** FMG already handles the
  linear / moderate / high-contrast-*linear* cases well; for nonlinear, Picard/Newton
  + FMG + timestepping-as-continuation is robust if unspectacular. FAS-Vanka's clear
  advantage (plasticity iteration count) does **not** yet translate to wall-clock
  (LU/Vanka smoother cost), and it breaks on exactly the high-contrast that real
  nonlinear problems also carry.
- **The engineering load is high.** Production FAS-Vanka needs: a multiplicative /
  interface-aware Vanka smoother (the extreme-contrast gap), per-level pressure
  nullspace, parallel patch construction, and a UW3 API. Production grid-sequencing
  needs the problem **re-bindable to coarser meshes** (re-create variables, re-bind
  expressions/BCs, and **restrict auxiliary/material fields** — the recurring "get the
  coefficient data onto the coarse grid" problem).
- **The tree needs a cleanup first.** Landing a feature of this weight onto a working
  tree with many outstanding changes is the wrong order.

## When to come back (what would change the calculus)

- **3-D and large-parallel.** This is where the argument flips: the direct solves
  that FMG's coarse grid and the LU paths lean on scale ~O(N²) in 3-D and MUMPS gets
  unreliable at high core counts, while Vanka's local solves scale ~O(N) and
  parallelise. Re-measure there.
- **A concrete problem where FMG genuinely fails at production resolution** — e.g. a
  notched/localising plasticity benchmark whose cold fine solve collapses. The spike
  could not tune a *simplified* version into the "easy-coarse / hard-fine" knife-edge
  (UW3's Picard+FMG was robust on smooth viscoplastic across resolutions); the real
  pathology needs the singular features (geometric notch tip, Drucker–Prager) and is
  the right target if/when it bites in practice.

## Cheapest future entry point

If a real need appears, the lowest-cost first step is **Tier-1 grid sequencing**: a
~30-line `grid_sequence_solve(build_fn, resolutions)` helper that loops coarse→fine,
solves each level with FMG, and interpolates up via `uw.function.evaluate`. It needs
no solver-internals changes and is robust to UW3's non-nested meshes. It only helps
when the fine solve is iteration-limited, so it should be opt-in. Prototype:
`~/+Simulations/snesfas_spike/notched_beam_cascade.py`. The callback-free
`solve(grid_sequence=True)` and the FAS-Vanka path are the larger follow-ons, gated
on the cleanup and on 3-D/parallel demand.
