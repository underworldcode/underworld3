# Boundary stress recovery & projection post-processing

Two related rules for getting accurate, affordable derived quantities (boundary
tractions, radial stress, deviatoric-stress components) out of a solved Stokes
field. Both came out of the spherical-benchmark post-processing work
(issues [#156], [#157], [#158]).

## 1. Use a *linear* solver for projections

`uw.systems.Projection` (and the vector/tensor/multi-component variants) performs
an **L2 projection** — a linear, symmetric-positive-definite mass-matrix solve
(or a screened-Poisson smooth when `smoothing > 0`, still linear SPD). It
inherits the general `SNES_Scalar` default stack:

```text
snes_type = newtonls    # Newton — but the problem is linear
ksp_type  = gmres
pc_type   = gamg        # AMG setup + repartition: the expensive part
```

That stack is robust for general nonlinear scalar PDEs, but for projection it is
**unnecessarily heavy**: GAMG setup/repartition dominates cost and memory, and
when you project many quantities in sequence (e.g. six deviatoric-stress
components at large MPI scale) it becomes the bottleneck — runs have been
OOM-killed mid-sequence on Gadi.

Switch projectors used purely for output/post-processing to a lightweight linear
solve with the opt-in helper:

```python
proj = uw.systems.Projection(mesh, target)
proj.uw_function = expr
proj.linear_solver()          # ksponly + CG + jacobi, drops the GAMG options
proj.solve()
```

`linear_solver(pc="jacobi", rtol=1e-10)` sets `snes_type=ksponly`,
`ksp_type=cg`, `pc_type=<pc>` and removes the now-unused GAMG options. It is
**opt-in** — the global default is unchanged, so internal paths that rely on it
(proxy variables, derivative evaluation, …) are unaffected. `jacobi` is fine for
a well-conditioned mass matrix; use `bjacobi` or `icc` if CG iteration counts
climb on distorted or high-degree meshes.

## 2. Project components, compose analytically

When recovering a **boundary stress** such as the radial normal stress
`σ_rr = nᵀ σ n`, do **not** project the composed expression directly:

```python
# DON'T: project the fully-composed scalar
sigma_rr = uw.systems.Projection(mesh, scalar)
sigma_rr.uw_function = n.T * stokes.stress * n      # = nᵀ(τ − pI)n
```

The composite mixes quantities of **different FE character** — the deviatoric
stress `τ ~ C:ε̇(u)` (built from velocity gradients, one order below the P2
velocity) and the pressure `p` (often discontinuous P1) — so a single scalar
projection cannot represent it as accurately as the parts. On the spherical
Thieulot benchmark the direct path gave ≈2× the `σ_rr` error of the reference.

Instead, **project the low-level components and compose analytically**:

```python
# DO: project the deviatoric-stress components, then form σ_rr symbolically
tau_proj = <project the τ_ij components, e.g. with a tensor/multi-component projection>
sigma_rr = n.T * tau_proj * n - p        # p used directly, not re-projected
```

This reproduces the `stokes.tau`-based benchmark values. The general principle:
**project the smoothest available primitives and combine them in closed form** —
projecting a derived composite throws away accuracy the primitives still have.

Combine the two rules — project the τ components with `linear_solver()` for the
cheap linear solve, then compose `σ_rr` analytically — for accurate boundary
stress recovery that also scales.

## 3. Zhong spherical-shell geoid response

Rotated free slip already exposes normal traction through
`Stokes.boundary_normal_traction()`. The Zhong adapter projects that recovered
traction onto the unnormalised `P_l^0` benchmark harmonic and applies the
Appendix A geoid operator:

```python
stokes.solve()

response = uw.postprocessing.zhong2008_response_from_rotated_stokes(
    stokes=stokes,
    radius_inner=0.55,
    radius_outer=1.0,
    radius_internal=0.775,
    harmonic_degree=2,
    load_scale=1.0,
    include_self_gravity=True,
)
```

The adapter delegates stress recovery to the existing rotated-free-slip API;
it does not implement a second CBF, constrained-multiplier, or topography
recovery path. `load_scale` must match the coefficient multiplying the model's
internal `P_l^0` load.

When surface and CMB topography coefficients are already available, call
`zhong2008_geoid_response()` or `zhong2008_self_gravity_response()` directly.
These functions are pure post-processing and do not require a Stokes object.

The Zhong Appendix A calculation is expressed as one linear operator:

```text
N = G h + n_load
(I - Q G) h_self_gravity = h + Q n_load
```

where `h` contains surface and CMB topography, `N` contains their geoid
responses, and `Q` contains the two self-gravity density factors. Sharing `G`
and `n_load` between the no-self-gravity and self-gravity paths avoids separate
scalar implementations of the same coefficients.

The rotated harmonic projector gathers boundary samples to rank zero and
reconstructs their spherical triangulation. A future boundary-reaction
functional could replace this step with a direct distributed finite-element
projection without changing the Zhong operator API.

## See also

- Issues [#156] (projection solver settings), [#157] (projection memory),
  [#158] (direct vs τ-based `σ_rr`).
- [`Projection`](../../api/) — the projection solver family.

[#156]: https://github.com/underworldcode/underworld3/issues/156
[#157]: https://github.com/underworldcode/underworld3/issues/157
[#158]: https://github.com/underworldcode/underworld3/issues/158
