# Monge–Ampère mesh redistribution: Newton/cofactor linearisation

> **Status**: exploration (Phase 0), `feature/winslow-mesh-smoother`,
> 2026-05-17. Companion to
> `docs/developer/subsystems/mesh-metric-redistribution.md` (the
> shipped BFO-Picard + direct-solver work) and the project memory
> `project-ma-efficiency-direct-solver`.

## Motivation

The shipped MA path (`_winslow_elliptic`) is a damped
**Benamou–Froese–Oberman Picard** iteration: each iteration solves a
*constant-coefficient* Poisson `Δφ = √((φxx−φyy)²+4φxy²+4g)−2` with
the recovered Hessian of the previous iterate, ~20–25 iterations,
under-relaxation `ω=0.4`. The constant operator is what made the
factor-once-reuse direct-solver speedup (~10×) possible — but that is
a **serial** expedient (sparse direct factorisation does not scale to
large-3D parallel per-timestep use; this build has only MUMPS + GAMG,
no hypre/SuperLU_DIST).

A **Newton / quasi-Newton** linearisation is the textbook approach for
smooth MA / mesh redistribution / OT. Linearising
`R(φ)=det(I+D²φ)−g`:

$$ \operatorname{cof}(I+D^2\varphi_k) : D^2\,\delta\varphi
   \;=\; g-\det(I+D^2\varphi_k), \qquad
   \varphi_{k+1}=\varphi_k+\lambda\,\delta\varphi $$

Using the Jacobi (Piola) identity `∂_i cof(M)_{ij}=0`, the weak form
is the **symmetric variable-coefficient elliptic** problem

$$ a(\delta\varphi,v)=\int (C_k\nabla\delta\varphi)\cdot\nabla v,
   \qquad C_k=\operatorname{cof}(I+D^2\varphi_k), $$

with `C_k` SPD iff `φ_k` is convex (Brenier branch). In 2D
`C_k = [[1+φyy, −φxy],[−φxy, 1+φxx]] = det(M_k)·M_k⁻ᵀ`. Only **first
derivatives of the unknown** appear (in the flux `F1=C_k∇δφ`); all
2nd-derivative content is in the *coefficient* `C_k`, read from the
existing recovered-Hessian field (`_hessian_recovery_class`,
first-derivatives-only — UW3-legal).

It slots into the existing `uw.systems.Poisson` (`SNES_Scalar`):
`F1 = constitutive_model.flux = c·∇u`, so a `DiffusionModel` subclass
with `_c = C_k` *is* the Newton operator; `f = det(I+H_k)−g` is the
source; `constant_nullspace` handles the pure-Neumann singularity
exactly as the BFO path does. Single-field scalar SNES — **not** the
rejected fully-coupled (φ,H) SNES.

### What it can and cannot change

- **Cannot** change the fixed-node grading ceiling (≈1.5–1.8× for an
  8–20× target). Same equation, same recovered Hessian ⇒ same fixed
  point. The OT ~10× needs *more nodes* (settled — see
  `project-ma-recovered-hessian-picard-inadequate`). Newton is **not a
  grading lever**; `ma_cost_grading.py` (1.02/1.43/1.71/1.54) is the
  regression guard.
- **Can** change convergence: few Newton iterations vs ~20–25 Picard
  ⇒ insensitive to per-iteration setup cost (the GAMG-resetup failure
  mode), and the per-step operator is SPD variable-coefficient
  elliptic ⇒ **AMG-friendly** ⇒ the right structure for the parallel
  rework.

## Phase 0 — residual-contraction quantification

**Goal**: confirm Newton contracts the MA residual
`r_k = det(I+H_k) − g` in a handful of iterations vs the BFO-Picard's
~20–25, on the canonical res-16 Annulus, *before* any source changes.
Both schemes share the φ field, the recovered-Hessian solver, the `c`
normalisation, `g`, the constant nullspace and pinned BCs — the **only
difference is the inner potential update**. Geometry is held fixed
(no node move) to isolate solver contraction.

Script: `scripts/ma_newton_phase0.py` (no `src/` changes; uses
`smoothing._hessian_recovery_class`, `_use_direct_solver`,
`_auto_pinned_labels`).

### Results

**Run 1 (AMP=8, RES=16) — a methodological finding.** Measuring
contraction of `r_k = det(I+H_k) − g` (H recovered) was the *wrong
yardstick*: it has a large **irreducible floor** that *neither*
scheme reduces — BFO plateaus at `‖r‖≈0.29` (from 0.46), Newton at
`≈0.34`. That floor is precisely the recovered-Hessian
under-estimation of `det(D²φ)` that the project memory identifies as
the root cause of the ≈1.5–1.8× single-solve cap (the FE-MA fixed
point is *self-consistently under-deformed*; `det(I+H_rec)−g` is O(0.3)
even at the exact FE solution). Strong confirmation that **Newton on
the cofactor operator cannot beat the grading ceiling** (same
recovered Hessian, same floor) — exactly as predicted; it is not a
grading lever.

Consequence for the experiment: a residual-decrease line search on
`‖det(I+H_rec)−g‖` is meaningless here (it rejected almost every
Newton step, collapsing `λ→0.008` and freezing the iteration — *not*
a fair Newton test). The correct Phase-0 question is the
*efficiency* one: **does Newton reach the same fixed-node transport
map in far fewer iterations than BFO's ~20–25?** The valid metric is
the **transport-map increment** `Δ_k = ‖∇φ_k − ∇φ_{k-1}‖∞` (→0 as
the map converges) and the realised `max|∇φ|` / honest grading
(must match BFO — the regression guard). Run 2 uses that, with
fixed damping (no residual-rejection; keep only a `det(I+H)>0`
convexity backtrack).

**Run 2 (AMP=8, RES=16) — transport-map contraction.** Metric:
`d_k=max|∇φ_k|`, increment `Δ_k=max|∇φ_k−∇φ_{k-1}|`; final honest
`d/n` after one signed-area-backtracked move. Three Newton
convexity-safeguard variants, all vs the shipped BFO-Picard.

| scheme | converges? | iters (Δ<1e-3·d₀) | final d/n | note |
|---|---|---|---|---|
| **BFO-Picard** (`+√` branch, ω=0.4) | yes | **16** | **1.713** | shipped; reference |
| Newton, residual line-search | no | — (frozen) | — | λ→0.008; the `det(I+H_rec)−g` floor (Run 1) makes the search objective meaningless |
| Newton, `det>0` backtrack only | no | — (stalls) | 1.58 | recovered-H noise breaks convexity under a finite step ⇒ λ→0.002, under-deforms |
| Newton, **PD-projected H** (eps=0.05) | no | — (creeps) | 1.49 | no λ collapse, but `Δ_k` plateaus ≈2e-3 (never contracts), overshoots `max|∇φ|` past BFO, map inverts cells (move scale→0.5) |

### Verdict — Newton/cofactor is NOT the efficiency/parallel path

Decisive negative result, consistent with and extending the settled
memory:

1. **It cannot beat the grading cap** (predicted): same recovered
   Hessian ⇒ same `det(I+H_rec)−g` floor (Run 1). Not a grading lever.
2. **It is *less robust* than BFO at the same recovered-Hessian
   quality** (new): all three convexity safeguards from the standard
   remedy list fail to reach BFO's fixed point — the iteration either
   freezes, stalls under-deformed (1.58), or creeps past it into a
   cell-inverting state (1.49). BFO reaches d/n 1.713 in 16 iters.
3. **Root cause**: BFO's `Δφ=√((φxx−φyy)²+4φxy²+4g)−2` is not "just a
   linearisation" — it is a *closed-form convex-branch solve* that
   expresses the new Laplacian via `g` and only the **deviatoric**
   part of the recovered Hessian, side-stepping the noisy/
   under-estimated full `det`. The cofactor-Newton operator feeds the
   full noisy recovered Hessian into *both* the variable coefficient
   `C_k` *and* the residual `det(I+H_k)−g`; at this recovery quality
   that is fragile (non-convex repulsion) or, once convexity is
   forced by projection, no longer the true MA equation (drifts
   instead of contracting). UW3 forbids 2nd derivatives of mesh-var
   functions, so a genuinely sharp `D²φ` (which Newton needs) is not
   available — the original footgun. Newton would only pay off with a
   fundamentally better Hessian / a wide-stencil MA discretisation:
   research effort, **no expected grading gain (settled) and now a
   demonstrated robustness loss**. Do not pursue.

### Implication for the parallel requirement

The validated efficiency lever stays the **factor/setup-once-reuse on
the constant BFO Laplacian** (shipped, ~10× serial via MUMPS). For
**parallel**, port that exact pattern to GAMG (the only AMG in this
build): build the GAMG hierarchy **once per `_winslow_elliptic` call**
(the operator is constant across the ~25 BFO iters) via
`snes_lag_jacobian=-2` / `KSPSetReusePreconditioner` with the constant
near-nullspace already wired, and warm-start the Krylov from the
previous Picard φ. Parallel-scalable, keeps BFO's robust convex-branch
structure, preserves grading. This — not Newton — is the parallel
work item. Script: `scripts/ma_newton_phase0.py`; data
`/tmp/metric_mesh/ma_newton_phase0.npz`.

## BFO + GAMG-reuse parallel prototype — tested, fragile (2026-05-17)

Wired as a *selectable* path: `_winslow_elliptic(...,
linear_solver="gamg")` (default stays `"direct"`).
`_use_iterative_solver`: FGMRES + GAMG(SOR smoother) for the elliptic
φ-Poisson — CG was *not* justified there (UW3 DMPlex-FEM assembly +
Neumann/nullspace gives no exact symmetry guarantee, and the SOR
smoother is non-symmetric ⇒ non-SPD preconditioner; FGMRES tolerates
both); CG + Jacobi for the provably-SPD mass systems.
`snes_lag_jacobian=-2` / `snes_lag_preconditioner=-2` so the GAMG
hierarchy is built **once per call** and reused across the ~25 Picard
iters (verified: φ-KSP iter count flat ≈75 once warm), Krylov
warm-started from the previous Picard φ.

The reuse mechanism works and **grading is bit-for-bit preserved
where it converges**. But the path is **not robust and does not
scale here** (`scripts/ma_solver_scaling.py`, AMP=8, direct = serial
MUMPS):

| RES | nodes | direct cold/warm | gamg cold/warm | d/n dir/gmg |
|----|------|------------------|----------------|-------------|
| 24 | 1748 | 3.1 / 3.8 s | 27.7 / 27.6 s | 1.712 / **1.007** ⚠ |
| 32 | 3059 | 6.9 / 8.7 s | 7.2 / 15.1 s | 1.722 / 1.722 |
| 48 | 6655 | 11.5 / 23.2 s | 16.3 / **69.2** s | 1.729 / 1.729 |

- **res-24 fails outright** — `DIVERGED_LINEAR_SOLVE` after 0 iters,
  φ≈0, d/n 1.007 (no-op). A *correctness* failure at one resolution
  while 32/48 converge: the hallmark of the documented
  GAMG-on-pure-Neumann + `constant_nullspace` + warm-resolve
  fragility (see the `_attach_constant_nullspace` code comment and
  `project-ma-efficiency-direct-solver`).
- Where it converges it is **2–3× slower than direct** and the
  **warm≫cold degradation returns** (res-48: gamg warm 69 s vs cold
  16 s) — the precise pathology the direct path *eliminated*. The
  gamg/direct ratio is erratic (7.3 / 1.75 / 3.0), **not** shrinking
  with N: no scalability signal at feasible 2D sizes.

### Two challenges that reshaped the verdict

**(a) "Did you wire the nullspace in?"** Verified at runtime: yes —
on the gamg path `ps.constant_nullspace=True` attaches the constant
`MatNullSpace` to the operator, the near-nullspace *and* the KSP
operator, cold *and* warm. The divergence is **not** a missing/
unprojected nullspace; the warm KSP runs to `its=10000`,
`reason=-3` (DIVERGED_ITS) — a GAMG *convergence* failure. The
direct path masks this entirely (MUMPS `icntl_24` null-pivot
detection solves the singular system irrespective of the PETSc
nullspace), which is why the iterative path is the first place a
conditioning problem surfaces.

**(b) "Why P3?"** No good reason — inherited from the original BFO
implementation. Sweeping φ∈{P1,P2,P3} × {direct,gamg}
(`scripts/ma_phi_order.py`):

| effect | finding |
|---|---|
| grading is set by φ **order**, not the solver | P2 ≡ P3 (≈1.71); **P1 is ~18 % weaker** (≈1.40) — P1 is *not* grading-equivalent, P2 is the floor |
| P3 is a **major GAMG confound** | res-24: P2+gamg converges (its=77, d/n 1.709 ✓) exactly where P3+gamg catastrophically fails (10000 its, d/n 1.007 ✗) |
| P2 does **not** fully cure GAMG | res-32 P2 *warm* still diverges — GAMG remains erratic across (res, cold/warm) even at P2 |

### Bankable win, independent of the parallel question

φ=P2 ≡ P3 grading to ~3 dp across AMP 0/2/8/20 on the **direct**
path (1.022/1.434/1.707/1.542 vs the recorded 1.02/1.43/1.71/1.54;
AMP=0 no-op exact; no tangle) at **~2× lower cost** (smaller
matrices — which also *helps* the direct factorisation scale, the
exact opposite of a scaling concern). **`phi_degree` default is now
2.** Canonical `cost_compare.py` at P2: MA cold ≈0.7–0.9 s (vs ~12–18 s
original), grading bit-for-bit. Combined with the factor-once-reuse
work this is ~15–20× over the original GAMG baseline.

### Verdict & recommendation

GAMG's failure was *partly* an own-goal (P3) — at P2 it converges in
many more cases — but P2 still leaves it **erratic on the warm
(post-`_deform_mesh`) re-solve**, so it is not a robust parallel
path yet. Combined with: no alternative AMG in this build (hypre/ML
absent), 2D sparse-direct being near-optimal at every feasible size,
and (decisively) the user's accepted position — **MUMPS direct is
fine for now; smaller matrices (P2) only help its scaling.** Keep
`linear_solver="direct"` (MUMPS — itself MPI-parallel) as the
validated path; retain `"gamg"` as experimental/documented-fragile
(do not delete — lag/reuse machinery is correct). A robust iterative
path would still need the pure-Neumann operator de-fragilised
(single Dirichlet pin, not the constant nullspace — ∇φ is unaffected
by the additive constant) and/or hypre, and is **gated behind**
parallel-exact assembly + 3D (the smoother is 2D-triangle-only,
serial-exact-assembly-only — the linear solver is *not* the parallel
bottleneck yet). Scripts: `ma_gamg_vs_direct.py`,
`ma_solver_scaling.py`, `ma_phi_order.py`, `ma_phi2_validate.py`.

### Spring as the MA initial guess — settled (do not re-run)

Asked whether seeding MA from the cheap `_winslow_spring` result
helps convergence. This is **settled-rejected** in
`project-ma-recovered-hessian-picard-inadequate`: spring-as-MA-
preconditioner is dead — at full AMP the spring drives a cell to
near-degeneracy and MA's signed-area backtrack *prevents* inversion
but cannot *cure* an already-degenerate start (it freezes); a
mild-spring→MA does converge but is **net slower than MA-only**
(the spring pass costs without cutting MA's ~25 Picard iters enough
to pay for itself). The mechanism is geometric — independent of
φ-order or solver speed — so the conclusion stands, and with MA now
~0.8 s the spring complexity is even less attractive. Not pursued.

### P1 vs P2 × GAMG, scaling with #triangles (check, 2026-05-17)

`scripts/ma_p1_gamg_scaling.py`, AMP=8, RES 16→64 (1.5k→22.7k tris):

- **P1 does not rescue GAMG.** When P1+GAMG converges it is
  textbook-good — **18–22 iters, N-independent** (vs P2's
  77→99→103, slowly growing) — confirming P1 is genuinely more
  AMG-friendly. *But it still fails erratically*: P1+GAMG diverges
  at res-32 (10000 its) and res-64 (r=-4, d/n collapses to 1.021
  no-op). P2+GAMG fails at 16 and 32. Neither order is reliable
  across the sweep — the pure-Neumann + warm-resolve breakdown is
  **order-independent and resolution-erratic**. Direct (MUMPS) is
  `✓` at every (res, order).
- Grading holds at every resolution: P1 ≈1.40 (1.397–1.421), P2
  ≈1.71–1.75 — P1 is ~18 % weaker *everywhere*, not a grading
  option regardless of solver.
- **More important side-finding (direct path):** the *warm* cost
  scales badly with N. P2-direct warm: 1.3 s (res-16) → 17.8 s
  (res-48) → **46.4 s (res-64)**, far above cold (9.5 s at res-64).
  The per-call post-`_deform_mesh` rebuild + MUMPS refactorisation +
  cache-invalidated `evaluate()` re-interpolation is O(N)-growing
  and re-opens a warm≫cold gap at realistic resolution. This — not
  the GAMG question — is the next per-timestep-scaling work item
  (the res-16 warm≈cold result does not extrapolate). Scripts add
  `ma_p1_gamg_scaling.py`.

### d/n is anisotropy/sliver-blind — rim over-collapse (2026-05-17)

User flagged the P2 rim cells as far tighter than the nominal 1/3.
`scripts/ma_radial_anisotropy.py` (res-16, AMP=8, vs undeformed):

| | band-mean radial (rim) | **min radial** | minA/meanA |
|---|---|---|---|
| undeformed | 1.00 | 1.00 | 0.575 |
| P1 | 0.65 | 0.43 | 0.240 |
| P2 | 0.38 (~1/3) | **0.14 (~1/7)** | **0.019** |
| P3 | 0.38 | 0.13 | 0.026 |

The reported deep/near ≈1.71 is a **per-node mean of all incident
edges** — it averages the collapsed *radial* edges with the
frozen/expanded *tangential* ones (tangential edges actually grow in
the interior; see the figure) and so hides a near-degenerate radial
sliver layer. Band-mean radial ≈0.38× matches the isotropic edge
criterion, but the **thinnest layer is ≈0.14× (~1/7)** and the
smallest cell is ~1/52 of the mean area.

**Mechanism:** the outer ring is *pinned* (it is the boundary) and
the metric peaks *exactly at* r=R_O — equidistribution demands
maximal density where nodes cannot move, so it jams the next
ring(s) against the fixed wall into one sliver layer, **independent
of AMP**. The isotropic `AMP = 1/s² − 1` design rule is wrong here:
in an annulus all transport is radial (tangential node count
frozen) *and* a boundary-peaked metric against a pinned boundary
over-collapses the wall layer.

**Consequences:** (1) d/n is fine as a *regression/consistency*
guard but does **not** certify mesh quality near a boundary-peaked
feature — use `minA/meanA` or a radial/tangential split. (2) Levers:
offset the Gaussian peak inward (`r=R_O−k·W`, k≈2–3) so the band
sits where nodes can redistribute on both sides; or cap AMP to a
quality floor (`minA/meanA ≥ 0.1` ⇒ AMP ≲ 3); or design the metric
from the *pinned-boundary 1-D radial OT*, not the isotropic rule.
Fig `/tmp/metric_mesh/ma_radial_profile.png`; script
`ma_radial_anisotropy.py`.

### Localised features: GAMG is robust + the "snuggle" metric fix (2026-05-17)

User: nodes should "snuggle up close to the feature"; the rim
example was "too local" (bulk has no metric gradient → doesn't
move). Interior blob (0.78,0), AMP=8, `ma_localised_reach_gamg.py`
+ `ma_heavytail_metric.py`:

| metric | far/near (resolution) | inward (distant→feature) | minA | GAMG |
|---|---|---|---|---|
| Gaussian W=0.12 | 2.42 | +0.008 | 0.105 | ✓ ~30 it |
| Gaussian W=0.30 | 1.55 | +0.010 | 0.267 | ✓ ~30 it |
| **Lorentzian (core 0.12 + 1/d² tail)** | **2.74** | **+0.025** | 0.089 | ✓ ~31 it |

- **A wider Gaussian is the WRONG fix.** One Gaussian width sets
  *both* the resolution scale and the reach: narrow ⇒ sharp but
  isolated pucker (bulk idle); broad ⇒ global motion but the
  feature washes out (far/near→1.5). The fix is a **heavy-tailed
  (Lorentzian) monitor**: a sharp core (best feature resolution,
  far/near 2.74) + a slow `1/d²` tail (∇ρ≠0 everywhere ⇒ distant
  nodes migrate IN ~3× more). The whole mesh rakes coherently
  toward the feature (`/tmp/metric_mesh/ma_heavytail.png`). Mild
  quality cost (minA 0.089 vs 0.105), no tangle. This is the
  standard r-adaptation lesson (monitor needs global reach — heavy
  tail or post-smoothing — not a narrow bump).
- **GAMG is ROBUST for localised interior cases — revises the
  earlier verdict.** Every metric shape × width × resolution
  converged in ~27–54 iters, cost competitive with direct, *zero*
  failures. The earlier GAMG fragility was **specifically** the
  boundary-peaked-metric-against-pinned-boundary pathology (metric
  spiking where the operator is pinned/singular). For the realistic
  localised-feature use case the parallel GAMG path is viable —
  the blanket "GAMG fragile" should be read as "fragile only for a
  metric peaked on the pinned boundary". Scripts:
  `ma_localised_reach_gamg.py`, `ma_heavytail_metric.py`.

### Polar metric + boundary slip — settled negative (2026-05-18)

Tested "define the metric in (r,θ) so it pulls in θ" + boundary
slip. `ma_polar_lorentzian_slip{,_v2}.py`, `ma_lorentzian_slip_final.py`,
interior/near-rim feature, AMP=8 res-24 (compact Cartesian
Lorentzian at an *interior* point gave far/near 2.74 — the
reference):

| variant | far/near | rim drift | GAMG |
|---|---|---|---|
| polar, chord 2(1−cosΔθ) | 1.38 | — | ✓ |
| polar, true wrapped angle, balanced cores | 1.12 | 1e-16 | ✓ |
| compact Cartesian Lorentzian near rim, slip off | 1.21 | 1e-16 | ✓ |
| …slip on | 1.12 (minA 0.32→0.48) | 3e-16 | ✓ |

1. **Separable (r,θ) Lorentzian is the wrong shape** — an
   anisotropic spoke, not a blob: the chord `2(1−cosΔθ)` saturates
   at the antipode (no angular reach); the balanced/true-angle
   version is a low-gradient radial ridge the smoother washes out
   (far/near≈1.1, ≈ no-op). Use a **compact `|X−P|²` Lorentzian
   about the feature point** — it has the correct combined
   radial+angular extent and pulls in θ automatically (far/near
   2.74 at an interior point).
2. **Slip works mechanically, is not a concentrator.** Rim radial
   drift ~1e-16 (nodes provably stay on the ring); GAMG robust
   (~31 it) throughout. But slip ON near a boundary feature
   *relaxes* the mesh (far/near 1.21→1.12, minA 0.32→0.48) — it
   removes the hard pin so the rim equalises; it does NOT drag rim
   nodes tangentially toward θ₀ (rim count near θ₀ 16→18). Slip
   buys boundary *quality*, not feature *concentration*.
3. **Boundary-proximal features are choked.** The same compact
   Lorentzian gives far/near 2.74 at r₀=0.78 (interior) but only
   1.21 at r₀=0.88 (near rim) — no node room between feature and
   pinned wall; slip relaxes rather than fills. Same fixed-node +
   pinned-boundary limit, feature side.

Net: compact Cartesian `|X−P|²` Lorentzian about the feature point
(pulls in θ inherently); keep features with interior room; slip is
safe and good for boundary *quality* but is not the lever for a
tangential pull. Drop the polar-separable formulation. Figures
`/tmp/metric_mesh/ma_polar_slip{,_v2}.png`, `ma_lorentzian_slip.png`.

### Angular OT target vs anisotropic scalar (2026-05-18) — (2) is a dead end

User: the metric should exploit the *abundant tangential* node
budget (slide spare angular nodes toward the feature) rather than
the *scarce pinned radial* one. Built (1) the exact 1-D angular OT
as the target for (2) a new opt-in `move_anisotropy=(w_r,w_θ)`
that rescales the realised displacement in the local
radial/tangential frame. Angle-only feature ρ(θ)=1+AMP/(1+(Δθ/Wθ)²),
AMP=8, res-24:

| | far/near | frac@θ₀ | minA | radial drift |
|---|---|---|---|---|
| undeformed | 1.00 | 0.159 | 0.547 | 0 |
| **(1) exact angular OT [TARGET]** | **2.21** | **0.415** | 0.209 | 1e-16 |
| (2) winslow isotropic | 0.98 | 0.158 | 0.356 | 6.8e-2 |
| (2) winslow tangential-preferred | 0.99 | 0.158 | 0.392 | 7.9e-3 |

- **(1) is exactly right** — rakes spare angular nodes into the θ₀
  sector (frac 0.16→0.42, far/near 2.2), radius untouched (drift
  1e-16), no tangle. For separable/structured features the explicit
  1-D OT is the correct tool, used *directly*.
- **(2) is a structural dead end.** Scalar BFO on the same metric
  produces ≈zero angular concentration (far/near 0.98, frac 0.158 ≈
  uniform) for *any* weighting. `move_anisotropy` works as designed
  — it suppresses *spurious radial* drift (6.8e-2→8e-3) — but there
  is no angular concentration to preserve: the scalar potential
  never generates the coherent tangential transport. Reweighting
  can shape transport the solver produces, not manufacture
  transport it does not.
- **Root cause = the foundational cap, both directions.** A scalar
  equidistribution potential with fixed topology cannot deliver
  large coherent *bulk* transport — radial (the ~1.7 cap) *or*
  tangential (here). Hoop/fixed-topology stiffness cuts both ways.

Verdict: "(1) as a target for (2)" *proves (2) cannot reach it*.
Use the explicit 1-D OT directly for separable features
(directional / dimensional-split redistribution); the generalisable
heavy route is a true anisotropic metric-*tensor* adaptation — not
anisotropic diffusivity / move-weighting on the scalar potential.
`move_anisotropy` is kept as an opt-in *quality* knob (suppresses
off-direction drift), not a concentrator. Script
`ma_angular_ot_target.py`; fig `/tmp/metric_mesh/ma_angular_ot.png`.

### (3) metric-tensor machinery — construction verified (2026-05-18)

`ma_metric_tensor_viz.py`: scalar density ρ(x) → `M = (1/h0²)[I +
β ĝĝᵀ(|∇ρ|/∇ρ_ref)²]`, eigen-clamped to spacing ∈ [H_MIN,H_MAX]
(≤8:1). Desired-cell ellipses drawn on a clean polar sample grid for
a radial feature ρ(r) and an angular feature ρ(θ). Result is
correct and confirms the design:

- Radial feature → ellipses **tangentially elongated** (short ⟂ r,
  long along the ring); circular where ∇ρ→0 (crest, far field).
- Angular feature → ellipses **radially elongated** (short ⟂ θ,
  long in r), concentrated in the θ₀ sector.
- **The eigenframe auto-aligns to r̂ / θ̂ with no (r,θ) frame
  specified anywhere** — M was fed only the Cartesian ∇ρ. This is
  the resolution of the user's (r,θ) puzzle: scalar density in,
  tensor alignment emergent from its gradient; API stays scalar.
- Max anisotropy = the eigen-clamp band (8.3:1), as designed.

Honest nuance (visible in the figure): a *gradient*-based metric
refines where ρ **changes** (the flanks) and is isotropic at a
smooth peak (∇ρ=0) and far away. Correct for "resolve the feature's
structure"; for small cells at the feature *core* use smoothed
`|∇ρ|` or the Hessian-based `M=|H(ρ)|` (curvature-aligned; needs the
recovered-Hessian path, extra cost). Gradient form is the
first-derivative, UW3-clean first cut.

Status: the metric *construction* (the ~1-day half) is verified and
cheap. Remaining for (3): the anisotropic **mover** (metric-Winslow
/ M-weighted displacement solve — the medium-effort half), with the
standing caveat that it improves cell alignment/quality, not the
fixed-node-count cap. Fig `/tmp/metric_mesh/ma_metric_tensor.png`.

---

## NEXT-PHASE KICKOFF BRIEF (read this first in a new session)

**Goal:** build the anisotropic *mover* for approach (3). The metric
*construction* is done & verified (`ma_metric_tensor_viz.py`,
`M = (1/h0²)[I + β ĝĝᵀ(|∇ρ|/ref)²]`, eigen-clamped). What remains is
the solver that moves nodes to satisfy a tensor metric M(x).

**Read before starting (do NOT re-derive / re-explore):**
- Memory `project-ma-efficiency-direct-solver` — the settled
  dead-ends. Do not retry: Newton/cofactor; GAMG on a
  boundary-peaked/pinned metric; polar-separable metrics; boundary
  slip as a *concentrator*; anisotropic *reweighting of the scalar
  BFO* (`move_anisotropy`) as a concentrator. All proven dead.
- This design doc, the "(3) metric-tensor machinery" + the angular-
  OT section (why scalar BFO can't do coherent bulk transport — the
  fixed-topology cap, both directions).
- `src/underworld3/meshing/smoothing.py`: the cache/lag/MUMPS infra,
  `_use_direct_solver` / `_use_iterative_solver`, `linear_solver`,
  `phi_degree=2` default, `move_anisotropy` (keep as a quality knob),
  and the Phase-0 `_CofDiff` pattern (script
  `ma_newton_phase0.py`) — the working example of a variable
  *tensor*-coefficient `SNES_Scalar` in UW3 (reuse this for M).

**Concrete plan:** a metric-Winslow / MMPDE M-weighted displacement
solve — `∇·(M ∇ξ)=0`-type vector system (or the M-weighted Laplace
smooth of the coordinate map), M the gradient-derived tensor field
above, move = the solved displacement, with the existing signed-area
backtrack + `boundary_slip`. Reuse: the tensor-constitutive pattern
(`_CofDiff`-style `DiffusionModel` subclass with `_c = M`), the
factor-once-reuse solver options, the cache. Validate on the SAME
model problems with the SAME honest, anisotropy-aware diagnostics
(`ma_radial_anisotropy.py`: minA + radial/tangential split, NOT
d/n) and against the explicit 1-D OT target (`ma_angular_ot_target.py`,
`ma_analytic_check.py`).

**Standing caveat (accepted by the user):** (3) improves cell
alignment/quality and removes the slivers/wasted-isotropic-resolution
— it does **not** beat the fixed node-count cap (that needs
`mesh.adapt`). For separable features the explicit 1-D OT (method 1)
stays exact and strictly cheaper; (3) earns its keep only for the
general non-separable case. Gradient-based M refines feature *edges*;
Hessian-based `M=|H(ρ)|` (curvature-aligned, needs the recovered-
Hessian path) is the follow-up if core-resolution is needed.

**Scope estimate:** ~1–2 weeks to a validated prototype on the
Annulus model problems. New feature branch off
`feature/winslow-mesh-smoother`. Effort is the solver + its
validation arc, not the metric (done).

---

## (3) anisotropic mover — IMPLEMENTED & VALIDATED (2026-05-18)

Branch `feature/anisotropic-metric-mover` (off
`feature/winslow-mesh-smoother`). `_winslow_anisotropic` in
`smoothing.py`; `smooth_mesh_interior(..., method="anisotropic")`.

### Formulation (as built)

Displacement form of the **decoupled direct** M-weighted Laplace
(Winslow) coordinate map. Per physical component `c`:

$$ \nabla\!\cdot(D\nabla u_c) = -\textstyle\sum_j\partial_j D_{jc},
   \qquad u_c=0 \text{ on the pinned boundary}, $$

so `ψ_c = x_c + u_c` solves `∇·(D∇ψ_c)=0`, `ψ=x` on the boundary
(the direct Winslow smoother — clusters nodes where `D` is large).
`D = M` (the verified eigen-clamped `M = (1/h0²)[I + β ĝĝᵀ
(|∇ρ|/gref)²]`). The two components share the *same* tensor
operator `_c = D` via a `_CofDiff`-style `DiffusionModel`
subclass; reuses `_use_direct_solver` (factor-once), the cache,
the signed-area backtrack, `boundary_slip`, `move_anisotropy`.
**Linear** — one solve/component/step, no Picard (cheaper than the
BFO `_winslow_elliptic`). Homogeneous Dirichlet ⇒ non-singular ⇒
**no `constant_nullspace`**, side-stepping the GAMG-pure-Neumann
fragility entirely.

### Two formulation findings (do NOT re-derive)

1. **The metric must be built ONCE and held fixed & Lagrangian**
   (like `_winslow_spring`'s rest-lengths/A0). Re-projecting ∇ρ on
   the progressively distorted mesh inside the outer loop is a
   *positive feedback* — `D` blows up on squashed cells →
   catastrophic over-collapse (minA/meanA → 1e-3). With `D` fixed,
   the outer loop is a stable damped fixed-point iteration of one
   linear operator toward the M-harmonic map.
2. **The decoupled direct Winslow form has no
   Rado–Kneser–Choquet non-folding guarantee**, so its stable
   regime is bounded by the metric anisotropy/contrast. A single
   un-damped elliptic jump folds; under-relaxation (`relax`) +
   `n_outer` damped steps is required (the analogue of the BFO
   `picard_relax=0.4`). Characterised Pareto frontier
   (`scripts/aniso_param_sweep.py`, interior radial feature): `β`
   is *not* the binding lever — the **eigen-clamp `aniso_cap`** is.

   | `aniso_cap` | needs | minA/meanA | note |
   |---|---|---|---|
   | 2 | `relax≈0.1–0.2` | **≈0.47–0.50** | robust default |
   | 4 | `relax≈0.05`, `n_outer≳25` | ≈0.35 | sharper, still clean |
   | ≳6 | — | ≲0.02 (folds) | needs coupled/inverse Winslow |

   Defaults shipped: `aniso_cap=2`, `relax=0.2`, `n_outer=12`,
   `β=200`. AMP=0 is an **exact isotropic no-op** (a scale-aware
   `g_eps=1e-9` floor rejects the ~1e-18 projection round-off of a
   uniform-ρ zero gradient — without it the noisy `gref` fabricated
   O(1) anisotropy).

### Validation arc (anisotropy-aware: radial/tangential split +
minA/meanA, NOT the anisotropy-blind d/n; grids rendered)

| problem (res, AMP=8) | metric | (3) minA/meanA | isotropic MA | spring |
|---|---|---|---|---|
| radial @R_O (pathology) | — | **0.240** | 0.019 | 0.177 |
| radial interior r=0.70 | — | **0.466** | 0.182 | 0.253 |
| angular-only (separable) | — | **0.243** | 0.144 | — |
| non-separable blob | — | **0.295** | 0.109 | 0.119 |

- **(3) is the cleanest method everywhere** — 2.6–12× better
  minA/meanA than the isotropic MA, never slivers, linear/cheap
  (~3 s res-16, no Picard).
- **Concentration is milder** than MA (radial interior far/near ≈
  MA; non-separable far/near 1.10 vs MA 1.37; angular ≈ uniform).
  (3) trades grading *magnitude* for clean anisotropic *cell
  alignment* — exactly its intended role.
- **Separable features confirm the settled cap**: angular-only
  (3) ≈ uniform concentration (far/near 1.02, frac@θ0 0.160) — it
  CANNOT beat the explicit 1-D OT (`ma_angular_ot_target.py`
  target far/near 2.21), same fixed-topology limit as the scalar
  paths. (3) is for the **non-separable** case + quality, not
  separable concentration.
- Figures: `/tmp/metric_mesh/aniso_radial_peak{1p00,0p70}.png`,
  `aniso_angular.png`, `aniso_nonsep.png` (the non-separable zoom
  is the clearest: MA/spring pull a degenerate slivered knot into
  the blob; (3) gives a clean, well-shaped, blob-aligned
  densification).

### Verdict

A **validated prototype matching the brief**: (3) improves cell
alignment/quality and removes the slivers/wasted isotropic
resolution; it does **not** beat the fixed node-count cap (the
explicit 1-D OT stays exact + cheaper for separable features).
Open follow-ups (out of prototype scope): the **coupled/inverse**
Winslow (RKC-non-folding) to admit `aniso_cap ≳ 6`; Hessian-based
`M=|H(ρ)|` for feature-core resolution; parallel-exact assembly.
Scripts: `aniso_smoke.py`, `aniso_param_sweep.py`,
`aniso_validate_{radial,angular,nonsep}.py`,
`aniso_blob_metric.py` (target-vs-realised), `aniso_convection_demo.py`
(Ra=1e5 → refine on ∇T).

### Architecture (pipeline & components)

`_winslow_anisotropic` in `src/underworld3/meshing/smoothing.py`;
reached via `smooth_mesh_interior(mesh, metric=ρ,
method="anisotropic")`. `ρ` is a target *density* (larger ⇒ finer)
— typically a Lagrangian `f(frozen_field.sym)`.

**Cache build (once per mesh/topology/params key):**

1. `grho` — projected `∇ρ`: a `Vector_Projection` with
   `uw_function = [ρ.diff(Xᵢ)]`, `smoothing=0`. A *first* derivative
   of the Lagrangian density only (UW3-legal).
2. `Df` — a `TENSOR` MeshVariable holding the metric tensor;
   initialised to the identity.
3. `_TensorDiff(DiffusionModel)` — `_build_c_tensor` sets
   `_c = Df.sym` (the `_CofDiff` pattern from `ma_newton_phase0.py`:
   a variable tensor-coefficient `SNES_Scalar`).
4. Per coordinate component `c`: a scalar `uw.systems.Poisson` with
   that constitutive tensor, source
   `f_c = Σⱼ ∂D_{jc}/∂xⱼ`, **homogeneous Dirichlet `u_c=0`** on the
   pinned boundary (non-singular → no `constant_nullspace` → no
   GAMG-pure-Neumann fragility), wired to `_use_direct_solver`
   (MUMPS, factor-once-reuse) or the `_use_iterative_solver` GAMG
   path. (`boundary_slip=True` ⇒ pure-Neumann + `constant_nullspace`
   + ring-projection instead, as in `_winslow_elliptic`.)

**Per call:**

5. **Build `D` ONCE on the undeformed mesh.** `gproj.solve()`;
   per node `M = (1/h₀²)[I + β ĝĝᵀ(|∇ρ|/gref)²]`; eigen-decompose;
   **clamp eigenvalues** to `[1/h_max², 1/h_min²]` (the `aniso_cap`
   band); reassemble → write `Df`. A scale-aware `g_eps=1e-9` floor
   makes uniform ρ an exact no-op (rejects the ~1e-18 projection
   round-off of a zero gradient). `D` is thereafter **fixed and
   Lagrangian** — it rides material points through `_deform_mesh`;
   re-projecting it each step is the positive-feedback collapse
   (settled).
6. **Damped MMPDE outer loop** (`n_outer` steps): solve the `cdim`
   displacement Poissons `∇·(D∇u_c) = −Σⱼ∂ⱼD_{jc}` (so `ψ=x+u` is
   the M-harmonic coordinate map); optional `move_anisotropy`
   reweight; `step = relax·disp`; **coherent global signed-area
   backtrack** (halve the scale until no triangle inverts) + slip
   ring-projection; `mesh._deform_mesh`; stop when
   `max|Δx| < outer_tol`.

Reuses `_winslow_elliptic`'s backtrack, `boundary_slip`,
`move_anisotropy`, the solver cache and the MUMPS
factor-once-reuse wiring verbatim. **Linear** — one solve per
component per outer step, no Picard (cheaper than the BFO MA).

### GAMG parity + cost per step (2026-05-18 — measured)

`scripts/aniso_cost_and_gamg.py`, interior radial feature, res
16/24/32/48 (1.5k–12.9k tris), `direct` vs `gamg`. Times: **cold**
(fresh mesh — MeshVariable+solver creation + 1st factorisation,
one-off per remesh), **warm** (same mesh object, cache hit — the
genuine per-timestep cost in a dynamic loop), per-outer-step, and
the D-build.

| res | ntri | warm direct | warm gamg | warm/outer | D-build | minA/meanA |
|----|------|------|------|------|------|------|
| 16 | 1522 | 3.08 s | 3.26 s | 0.25 s | 0.34 s | 0.4657 |
| 24 | 3268 | 6.29 s | 6.30 s | 0.51 s | 0.64 s | 0.4256 |
| 32 | 5814 | 10.94 s | 10.94 s | 0.89 s | 1.11 s | 0.3938 |
| 48 | 12856 | 23.72 s | 23.98 s | 1.94 s | 2.41 s | 0.4452 |

- **GAMG is robust here — bit-parity with direct**
  (`|minA_g−minA_d| ≤ 5e-5` at every resolution). The mover is
  **non-singular** (homogeneous Dirichlet, no constant nullspace),
  so it does **not** hit the pure-Neumann + warm-resolve fragility
  that made the MA `gamg` path erratic. This is the **first** of
  the three metric methods with a working parity-preserving
  parallel-capable solver path. (At feasible 2D sizes MUMPS is
  near-optimal so `gamg` is not *faster* — the point is it *works
  and matches*, so the parallel route is real.)
- **cold ≈ warm at every resolution** — no warm-≫-cold
  degradation (the MA path's O(N) post-deform rebuild pathology is
  absent here; the cache reuses the MeshVariables/solvers, only the
  operator is refactorised because `D`+geometry change each call).
- **Cost is ~O(N) (linear in #cells).** warm 3.1→23.7 s for
  ntri 1522→12856 (≈7.7× for ≈8.4× cells); per-outer-step and
  D-build likewise ~O(N). No superlinear blow-up — the per-step
  work is a fixed number of **sparse SPD-ish elliptic solves**
  (the part GAMG parallelises with optimal O(N/P) complexity and
  good weak scaling) plus embarrassingly-local per-node /
  backtrack work.
- **The cost lever is `n_outer`.** Default 12 ⇒ ~12 scalar
  elliptic solves of the mesh size. The damped MMPDE converges
  (most displacement is in the first few steps; `max|Δx|` decays),
  so an `outer_tol` early-exit / a small `n_outer` cuts the warm
  cost to ≈ `D-build + 3–5 · warm/outer` (≈1.5–2 s at res-16). The
  per-step adaptation is then ≈ *a handful of pressure-solve-class
  SPD solves* — genuinely cheap for an r-adaptation scheme (most
  need nonlinear solves or global transport; this does not).
- Honest hotspot: the per-node eigen-clamp is a Python loop
  (`np.linalg.eigh` per node) — vectorisable to a batched
  `eigh` on a stacked `(N,d,d)` array (a cheap win, matters more in
  3D / at scale); currently dominated by the solves anyway.

**Parallel verdict (the user's hypothesis, now evidenced):** the
per-step cost is `1 ∇ρ projection + a vectorisable eigen-clamp +
n_outer × (cdim non-singular SPD elliptic solves + a local
backtrack)`, all O(N) and GAMG-parallelisable with proven
2D parity. This is one of the few r-adaptation strategies with
**no nonlinear solve and no global transport** — structurally
inexpensive in parallel. (Caveat: the *assembly* — ∇ρ projection /
D-build / backtrack — is still serial-exact; the parallel-exact
cross-rank version is the remaining piece, not the solver.)

### Solver limitations

- **2D triangle meshes only** (hard `NotImplementedError`).
- **Decoupled direct Winslow form → no Rado–Kneser–Choquet
  non-folding guarantee.** Stable only for modest anisotropy:
  `aniso_cap≈2` (robust default), `≈4` with gentler `relax` + more
  `n_outer`, **`≳6` folds regardless**. The backtrack prevents
  *inversion*, not extreme squashing — a property of the
  formulation, not a tuning miss.
- **Fixed node budget** — relative redistribution only; cannot
  beat the node-count cap. For *separable* features the explicit
  1-D OT is exact and strictly cheaper.
- **Gradient metric resolves edges/fronts, not cores** —
  isotropic-coarse (de-refined) where `∇ρ=0` (a smooth peak). Right
  tool for boundary layers / interfaces / fronts; wrong tool for
  resolving a smooth peak's centre (→ Hessian metric).
- **Metric is Lagrangian-fixed** (built once). A tensor metric
  should co-rotate with large deformation; we don't — fine for
  modest moves, not large-strain.
- **Serial-exact assembly only** — the ∇ρ projection / `D` build /
  backtrack under-count at rank-partition boundaries (same caveat
  as spring/MA). The *solver* is no longer the parallel blocker
  (GAMG validated, see the cost section); the cross-rank
  parallel-exact assembly is the remaining piece. MUMPS scales to
  modest sizes; GAMG is the route beyond.
- **Linear, component-decoupled** — an anisotropic Laplacian
  smoother, not the full nonlinear (Jacobian-coupled) Winslow
  generator.

### Corners still unexplored

- **Solution-accuracy proof.** Validated mesh *quality + alignment*
  only — NOT yet that it *helps the PDE* (lower T-discretisation
  error / better Nu at fixed node count vs a uniform mesh). That
  accuracy/cost study is the real payoff and is untested.
- **Dynamic-adaptive loop.** The demo is static ("20 steps then
  refine once", `aniso_convection_demo.py`). Re-refining every N
  steps with the metric riding the flow (ALE-style, interacting
  with SLCN advection / the free-surface ALE) — the production use
  case — is unexplored.
- **Coupled / inverse Winslow** (computational ξ harmonic in
  physical space → RKC-non-folding) to safely admit `aniso_cap ≳ 6`
  and stronger alignment. The heavy MMPDE (map inversion /
  resampling).
- **Hessian metric `M=|H(ρ)|`** (curvature-aligned) for feature-
  *core* resolution — reuse the recovered-Hessian path
  (`_hessian_recovery_class`; first-derivative L2 recovery, since
  UW3 forbids 2nd derivatives of mesh-var functions).
- **A `metric_from_gradient`-style ρ helper** unifying the metric
  API across `mesh.adapt` (absolute `h`, MMG re-meshes) and the
  mover (relative `ρ`, fixed budget) — discussed, not built.
- **GAMG path — VALIDATED (2026-05-18), see the cost section.**
  Bit-parity with direct at res 16–48 (non-singular ⇒ no
  pure-Neumann fragility); the parallel-scalable route is real.
  *Remaining*: cross-rank **parallel-exact assembly** (the ∇ρ
  projection / D-build / backtrack are serial-exact — the solver
  is not the blocker), and a true MPI weak-scaling study.
- **3D extensibility — concrete scope.** Already
  dimension-general: the metric formula
  `M=base[I+β ĝĝᵀ(|∇ρ|/gref)²]`, the eigen-clamp
  (`np.linalg.eigh` works for 3×3), the `TENSOR` MeshVariable
  (`dim²` comps), the displacement form `∇·(D∇u_c)=−Σⱼ∂ⱼD_{jc}`
  over `c=0..cdim−1`, the per-component `Poisson` + `_TensorDiff`
  (3×3 `_c`), and the solver wiring — and GAMG (now proven for
  this operator) is exactly what makes 3D viable (3D sparse-direct
  does not scale). 2D-specific work to remove: the
  `cdim!=2` guard; `_tri_cells`/`_signed_areas` →
  `_tet_cells`/`_signed_volumes` for the inversion backtrack (the
  main piece — a shared limitation with spring/MA); ~5 lines of
  the eigen-clamp / `Df.array[:,i,j]` writes generalised to
  `cdim`; `boundary_slip`/`move_anisotropy` stay 2D (default
  off/None). Modest, well-scoped (~1–2 days) — the solver core is
  already dim-general; the careful step is validating the tet
  signed-volume backtrack before it lands in the shared smoother.
- **Auto-tuning** `aniso_cap`/`relax`/`n_outer` (largest cap that
  keeps `minA/meanA` above a floor — the Pareto frontier is
  characterised but not automated).
- **Free-surface / deformed-boundary slip** (polyline projection —
  shared open item with spring/MA).

---

## NEXT-PHASE KICKOFF BRIEF — dynamic adaptive convection (read first)

**Phase just closed (2026-05-18):** the anisotropic mover is a
validated 2D prototype, GAMG-parity, ~O(N), and the **API is
locked in**:

- `uw.meshing.smooth_mesh_interior(mesh, metric=ρ,
  method="anisotropic", method_kwargs=dict(aniso_cap=2.0,
  relax=0.2, n_outer=12, linear_solver="direct"))`
- `uw.meshing.metric_density_from_gradient(mesh, field, amp=8.0,
  lo_percentile=50, hi_percentile=97)` → the Lagrangian
  `ρ = 1+amp·t` density (the relative analogue of
  `adaptivity.metric_from_gradient`; cached for per-step use).
- Docs: `docs/advanced/mesh-adaptation.md` (peer to `mesh.adapt`),
  `docs/developer/subsystems/mesh-metric-redistribution.md`,
  this design note.
- Test harness: `scripts/adaptive_convection_harness.py`.

**Goal of the next phase:** a *correct* dynamic-adaptive
convection solve — coarse adaptively-snuggled mesh reproducing a
fine uniform reference. The harness already runs the comparison
(Ra=1e5, uniform res-24 reference vs res-16 adaptive,
`Nu(t)`/`vrms(t)` rms error, figure).

**THE open piece — the node-update / ALE correction.** When the
mover displaces nodes by `Δx` over the step interval `Δt`, the
mesh has velocity `v_mesh = Δx/Δt`. The SLCN advection–diffusion
must transport along the material velocity *relative to the moving
mesh*: `V_fn = v_fluid − v_mesh` for the post-adapt step (ALE), or
T must be conservatively remapped onto the moved nodes. Without it
the pure coordinate move is read as a spurious advection of T.
**Precedent is settled in this codebase:** the free-surface ALE
finding (memory `project_freesurface_ale_design` — a Lagrangian
mesh move needs `V_fn = v − v_mesh` or convection is
non-physically damped, Nu ~57 vs 143). The hook is
`apply_adaptation_correction` in the harness: `--correction none`
is the uncorrected baseline (expected to drift — it *quantifies*
the error the correction must remove); `--correction ale` raises
with the spec. **Acceptance test:** harness `rms ΔNu(adaptive
res-16 vs uniform res-24)` small with the correction, large
without.

**Other follow-ups, priority order:**
1. ALE correction + harness acceptance (above) — the headline.
2. **3D port** — scoped ~1–2 days. The solver core is already
   dimension-general; the 2D-specific work is the tet
   signed-volume inversion backtrack (`_tri_cells`/`_signed_areas`
   → tet) + dropping the `cdim!=2` guard + ~5 generalised lines.
   **The metric stays `1/h²` per principal direction in 3D — it is
   NOT `1/h³`** (a Riemannian metric measures *edge length*, which
   is 1-D regardless of embedding dimension: `eᵀMe=1` ⇒ eigenvalue
   `1/h²`; dimension enters only the complexity integral
   `∫√(det M)` via `det M = ∏1/hᵢ²`). For the *mover* the overall
   `D` scale is moreover irrelevant (the displacement PDE is
   invariant under `D→αD`) — only the anisotropy/contrast ratios
   matter, so 3D needs no scaling change at all.
3. Parallel-exact cross-rank assembly + MPI weak-scaling (GAMG
   solver path already validated bit-parity).
4. Hessian metric `M=|H(ρ)|` for feature-*core* resolution.
5. `aniso_cap`/`relax`/`n_outer` auto-tuning to a `minA/meanA`
   floor.

**How to resume:** run
`python scripts/adaptive_convection_harness.py --correction none`
for the baseline error, then implement `apply_adaptation_correction`
(`--correction ale`) and re-run to show the gap closes.
