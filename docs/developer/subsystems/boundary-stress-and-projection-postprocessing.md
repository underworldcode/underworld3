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

## 3. Spherical-shell geoid and self-gravity response

Rotated free slip already exposes normal traction through
`Stokes.boundary_normal_traction()`. The convenience adapter projects that
recovered traction onto the unnormalised axisymmetric `P_l^0` harmonic and
applies the spherical-shell geoid operator:

```python
stokes.solve()

response = uw.postprocessing.geoid.spherical_shell_response_from_rotated_stokes(
    stokes=stokes,
    radius_inner=0.55,
    radius_outer=1.0,
    harmonic_degree=2,
    internal_load_radius=0.775,
    internal_load_coefficient=1.0,
    include_self_gravity=True,
    surface_density_contrast=3300.0,
    cmb_density_contrast=5400.0,
    planet_radius=6370000.0,
    gravity=9.8,
    gravitational_constant=6.67e-11,
    projection="reaction",
)
```

The adapter supports two projection paths. `projection="centroid"` retains the
original pointwise-recovery workflow: recover `sigma_nn`, gather the samples to
rank zero, reconstruct a spherical triangulation, and integrate centroid
values. `projection="reaction"` contracts the assembled normal-reaction load
directly with the harmonic test function through
`Stokes.boundary_normal_traction_integral()`. The latter is distributed, avoids
the rank-zero surface reconstruction, and is an integral/fitted quantity rather
than a consumer of the slowly converging P2 vertex values on curved boundaries
(#414). Its fitted coefficient uses the matching discrete boundary norm, not an
analytical spherical norm, so the numerator and denominator share the same
faceted geometry.

Both paths reuse the existing rotated-free-slip reaction; neither implements a
second CBF, constrained-multiplier, or topography recovery. `centroid` remains
the compatibility default while the direct reaction path accumulates benchmark
coverage. `internal_load_coefficient` must use the same harmonic normalisation
and sign convention as the model's internal load.

When surface and CMB topography coefficients are already available, call
`uw.postprocessing.geoid.spherical_shell_geoid_response()` or
`uw.postprocessing.geoid.spherical_shell_self_gravity_response()` directly.
These functions are pure post-processing, work for any spherical-harmonic
order with a consistent coefficient normalisation, and do not require a Stokes
object. They support non-negative harmonic degrees, and the internal load is
optional. The rotated-Stokes adapter requires degree one or greater because
normal-traction recovery removes the degree-zero mean.

The density contrasts, dimensional outer-radius scale, and gravity are required
when self-gravity is enabled. They deliberately have no Earth- or
benchmark-specific defaults. The universal gravitational constant defaults to
the current CODATA value and can be overridden when reproducing a paper's
rounded constant.

The calculation is expressed as one linear operator:

```text
N = G h + n_load
(I - Q G) h_self_gravity = h + Q n_load
```

where `h` contains surface and CMB topography, `N` contains their geoid
responses, and `Q` contains the two self-gravity density factors. Sharing `G`
and `n_load` between the no-self-gravity and self-gravity paths avoids separate
scalar implementations of the same coefficients.

This module does not compute a benchmark's semi-analytical Stokes solution.
Published reference solvers, such as the Zhong et al. propagator-matrix method,
belong in `uw.analytic`; their computed topography coefficients can be passed to
the pure post-processing functions above.

`Stokes.boundary_normal_traction_integral(boundary, fn)` is useful beyond the
geoid adapter whenever only an integrated or fitted normal-traction diagnostic
is required. Pointwise consumers should continue to call
`boundary_normal_traction()` or `dynamic_topography()` and follow their curved
P2 midpoint/field guidance.

## See also

- Issues [#156] (projection solver settings), [#157] (projection memory),
  [#158] (direct vs τ-based `σ_rr`).
- [`Projection`](../../api/) — the projection solver family.

[#156]: https://github.com/underworldcode/underworld3/issues/156
[#157]: https://github.com/underworldcode/underworld3/issues/157
[#158]: https://github.com/underworldcode/underworld3/issues/158
