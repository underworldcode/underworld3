---
title: "Local Scattered-Point Interpolation"
---

## Purpose

Underworld needs to move field values between point sets that have no shared
mesh topology: particles to proxy mesh nodes, nodes to arbitrary evaluation
points, one mesh to another after adaptation. The tool for that is a *local*
interpolant — for each target point, take its `nnn` nearest source points from
a kd-tree and form a weighted average.

This is general-purpose numerical machinery, and the goal is straightforward:
**the interpolation should be accurate**. The single question that determines
that is **which fields the weights reproduce exactly**. A scheme that cannot
reproduce a uniform gradient will smear every field that has one, everywhere it
is used, and no amount of tuning elsewhere recovers what it threw away.

That is what `order` selects on
{py:meth}`underworld3.ckdtree.KDTree.rbf_interpolator_local`.

## The two schemes

### `order=0` — inverse distance (Shepard)

$$ w_j = \frac{d_j^{-p}}{\sum_k d_k^{-p}} $$

Weights are positive and sum to one. Consequences, both of them important:

- A **constant** field is reproduced exactly.
- The result is a **convex combination** of the neighbouring values, so it can
  never overshoot them. The interpolant is intrinsically bounded.
- A field with a **gradient is smeared**, and the error does *not* vanish as
  the stencil tightens. Adding neighbours does not help; refining the point
  spacing helps only at first order.

### `order=1` — polyharmonic RBF with an affine tail

For a target $x^*$ with neighbours $x_j$, solve the small saddle-point system

$$
\begin{bmatrix} A & P \\ P^{T} & 0 \end{bmatrix}
\begin{bmatrix} w \\ \lambda \end{bmatrix}
=
\begin{bmatrix} \varphi(|x^* - x_j|) \\ 1,\; x^* \end{bmatrix}
$$

with $A_{ij} = \varphi(|x_i - x_j|)$, the polyharmonic (thin-plate) kernel
$\varphi(r) = r^2 \log r$, and the affine tail $P = [1,\; x_j]$.

The lower block *is* the reproduction constraint $P^{T} w = [1,\; x^*]$, so

$$ \sum_j w_j = 1, \qquad \sum_j w_j x_j = x^* $$

hold to round-off by construction: **constants and linear fields are exact**.

Weights are signed, so the interpolant can overshoot the neighbouring values —
the price of giving up the convex combination.

Cost is one dense $(nnn + d + 1)^3$ solve per target point, and the result is
still sparse: `nnn` non-zeros per row. The weights depend only on geometry, so
one solve serves every data component.

## Measured

Relative error of a swarm proxy variable against the analytic field, at the
proxy's own nodal coordinates. A **linear** field lies exactly inside both the
P1 and P2 proxy spaces, so the discretisation contributes nothing and every bit
of the measured error is particle-to-node transfer error.

2D simplex box, `cellSize=1/8`, `fill_param=4`, proxy degree 1:

| field | `order=0`, `nnn=3` | `order=1`, `nnn=6` |
|---|---|---|
| linear, max | 4.6e-3 | **5.1e-16** |
| quadratic, max | 9.2e-3 | 1.0e-4 |
| quadratic, rms | 2.1e-3 | 2.1e-5 |

3D simplex box, `cellSize=1/8`, `fill_param=4`, proxy degree 1:

| field | `order=0`, `nnn=4` | `order=1`, `nnn=8` |
|---|---|---|
| linear, max | 4.8e-3 | **5.5e-16** |
| quadratic, max | 8.5e-3 | 2.4e-4 |
| quadratic, rms | 2.0e-3 | 2.3e-5 |

Two things to read off. First, the linear-field column is the clean statement:
one scheme is exact, the other is not. Second — and this is the part that makes
it a general accuracy improvement rather than a special case — the gain carries
over to a field with curvature, by roughly two orders of magnitude in rms.

Widening the `order=0` stencil does **not** close the gap. At `nnn=12` the
inverse-distance error is unchanged to two significant figures, because the
error is set by the asymmetry of the neighbourhood, not by how many points are
in it.

## Choosing `nnn`

`order=1` requires **`nnn >= dim + 2`** and raises `ValueError` below that.

The reason is worth knowing. At exactly `nnn == dim + 1` the affine block $P$
is square, so the constraint $P^{T} w = [1, x^*]$ alone determines $w$: the RBF
block is inert and the scheme collapses to bare **barycentric interpolation on
the neighbour simplex**. That is singular whenever those `dim + 1` points are
collinear (2D) or coplanar (3D) — common near boundaries and on graded meshes.

The swarm proxy default is `nnn = 2 * (dim + 1)` (6 in 2D, 8 in 3D), which
leaves enough slack that degenerate neighbourhoods are rare.

## Degenerate stencils

If the neighbours cannot support an affine fit, the saddle system is singular.
This is not hypothetical: it happens at boundaries and in graded meshes.

The implementation (`underworld3.utilities.rbf_stencil.linear_exact_weights`):

1. **Pre-screens** the affine block by SVD — degenerate if
   $\sigma_{\min} < 10^{-8}\,\sigma_{\max}$.
2. **Solves** only the healthy stencils, batched and chunked over targets.
3. **Validates the answer** rather than guessing a condition number: it checks
   the two reproduction identities and finiteness on the computed weights, and
   demotes anything that fails.
4. **Falls back** to the inverse-distance weights for those points — less
   accurate there, but finite and bounded. It never returns NaN.
5. **Warns once** with the count and fraction that fell back. A silent
   geometric fallback is the failure mode behind issue #424; a per-point
   warning would be unusable, a single counted one is not.

## Limiting: `monotone`

`order=1` weights are signed, so the interpolant can oscillate where the data
is rough. `monotone=True` (or `"clamp"`) bounds that.

The important part is *what* it bounds. The obvious limiter — clip the result
to the min/max of the stencil's source values — is **wrong for a linear-exact
scheme**, and wrong in a way that is easy to miss:

```{warning}
A target that lies outside the convex hull of its own `nnn` neighbours has a
value outside their range **even for an exactly linear field**. Proxy nodes are
routinely in that position, and not only at domain boundaries. A raw min/max
clip therefore cannot distinguish legitimate extrapolation from ringing, and
fires on the linear part — destroying the one guarantee the scheme exists to
provide.
```

So the limiter here follows the slope-limiter discipline instead: **never limit
the linear reconstruction, limit only the correction on top of it.**

1. Fit an affine function to the stencil data by least squares.
2. Keep that trend at the target untouched.
3. Bound the remaining RBF correction to the range of the non-affine residual
   the stencil actually exhibits.

Consequences, both measured:

- On any field the scheme already reproduces exactly, the limiter is a **no-op**
  (it moves the answer by ~1e-15). Linear reproduction survives it.
- On a rough field it does bite — and, like any limiter, it trades accuracy for
  boundedness where the correction genuinely exceeds the observed residual. On
  a `sin(6x) + |x|²` test field it moved the answer by 2e-3 (2D) and 5e-2 (3D);
  in 3D that made the max error slightly worse (9.0e-2 → 1.3e-1), still well
  below inverse distance (2.7e-1).

Note what this does and does not promise: it bounds *new oscillation relative
to the local trend*, not absolute range. A quantity that must stay inside hard
physical bounds (a fraction in $[0,1]$) needs its own clip on top.

### Material level sets stay on `order=0` — measured, not assumed

`IndexSwarmVariable` builds one level-set MeshVariable per material index and
keeps its own inverse-distance weighting. It was tested against `order=1` and
**deliberately not changed**.

The reason is structural. A material indicator is **piecewise constant**, not
smooth. Away from an interface both schemes reproduce it exactly, because both
reproduce constants — linear exactness has nothing to add. At the interface the
field is *discontinuous*, so no polynomial-reproducing scheme is exact either;
signed weights simply add overshoot where the data has a jump.

Measured on a straight interface at `x = 0.5` (exactly representable, so any
displacement of the recovered 0.5 contour is scheme error):

| scheme | interface error, median | level-set range |
|---|---|---|
| inverse distance, `nnn=5` | 5.2e-3 – 1.1e-2 | `[0, 1]` exactly |
| `order=1`, `nnn=6` | 6.9e-3 – 7.8e-3 | `[0, 1]` exactly |
| `order=1`, `nnn=8` | 5.2e-3 – 7.9e-3 | **`[-0.038, 1.038]`** |

The accuracy result is a wash — `order=1` is better at the coarse resolution
and equal or worse at the fine one, with the ordering flipping between cases —
while `nnn=8` violates the `[0, 1]` bound by ~3.8%.

Partition of unity survives either way (all indices share one weight set, and
the indicator flags sum to one per particle, so the level sets sum to
`Σ w_j = 1` regardless of sign). But a *negative* material fraction is still
physically wrong, and `constitutive_models.py` consumes these directly.

```{note}
The interface metric groups nodes into rows by `y` and interpolates the 0.5
crossing, which is crude on an unstructured simplex mesh — the *maximum* error
is identical across all schemes because it is set by node spacing, not by the
weights. Only the median is informative, and it is the median that shows no
consistent gain.
```

So the swarm story is deliberately split: the plain `SwarmVariable` proxy takes
`order=1` because its fields are smooth and the gain is two orders of magnitude;
`IndexSwarmVariable` keeps inverse distance because its field is a jump, where
there is nothing to gain and a bound to lose.

Consumers that depend on absolute boundedness, and are therefore deliberately
left on `order=0`:

- The **RBF rung of the point-location fallback ladder** in
  `uw.function.evaluate` — `docs/developer/design/point-location-capability.md`
  states its contract as "bounded, topology-free, honest".
  `MeshVariable.rbf_interpolate` therefore keeps `order=0` as its default.
- `IndexSwarmVariable` level sets, which carry material fractions in $[0,1]$ and
  are divided through by their sum in `constitutive_models.py`. That path
  hand-rolls its own inverse-distance weighting and is untouched.

## Where each is used

| Caller | Default | Why |
|---|---|---|
| `SwarmVariable.rbf_interpolate` / `_rbf_to_meshVar` (proxy refresh) | `order=1`, `nnn=2*(dim+1)` | Accuracy; the proxy feeds integration, derivatives and the Lagrangian time derivative |
| `MeshVariable.rbf_interpolate` | `order=0` | Documented bounded contract (see above) |
| `SwarmVariable.read_timestep`, `MeshVariable.read_timestep` | `nnn=1` | Exact checkpoint round-trip, no averaging wanted |
| `IndexSwarmVariable._update_proxy_variables` | its own IDW | Level-set fractions must stay in $[0,1]$ |

A rank holding fewer particles than the affine tail needs drops to `order=0`
automatically rather than failing the refresh.

## Parallel behaviour

Proxy refresh is **rank-local**: the kd-tree indexes only this rank's
particles and there is no halo exchange (SWARM-15, see
`docs/developer/design/SWARM_MODERNIZATION_DESIGN_2026-07.md` §4). A proxy node
near a partition seam therefore gathers from a one-sided neighbourhood.

Linear exactness improves this but does not fix it. A linear-exact stencil
reproduces a linear field exactly from *any* neighbourhood, one-sided or not,
so for linear fields the seam error is zero and the proxy is np-independent
(pinned by `tests/parallel/test_0776_linear_rbf_proxy_parallel.py`). For a
field with curvature a one-sided stencil still differs from a centred one, so
np-dependence remains. The halo exchange in SWARM-15 is still the real fix.

## Related

- `docs/developer/subsystems/data-access.md` — lazy proxy updates
- `docs/developer/design/SWARM_MODERNIZATION_DESIGN_2026-07.md` §4 — rank-local seams
- `docs/developer/design/point-location-capability.md` — the evaluate fallback ladder
- `utilities/custom_mg.py` — a *global, dense* polyharmonic prolongation for
  multigrid transfers. Same kernel and tail, but every coarse point enters every
  row, so `PᵀAP` is dense. The local scheme documented here is the sparse
  equivalent.
