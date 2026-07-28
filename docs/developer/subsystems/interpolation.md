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

$d_j$ is the **actual** distance to the neighbour, and `p` defaults to 1 — so
the default really is inverse distance. (Until #427 the kd-tree's squared
distances were used directly, making the decay $r^{-2p}$: the documented
default of `p=2` was in fact $1/r^4$.)

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

## Getting the operator, not just the values

When the same transfer is applied more than once — or when the operator *is*
the product, as for a multigrid prolongation — build it once:

```python
T = kdt.interpolation_matrix(target_coords, order=1)   # scipy CSR
values = T @ data                                      # identical to the value API
```

The weights depend only on geometry, so one build serves every field and every
component; the value API re-solves each call. `T @ data` is asserted equal to
`rbf_interpolator_local(...)` in the test suite so the two cannot drift.

```{warning}
`rbf_stencil.linear_exact_weights` is the raw kernel underneath, and it returns
**zeros** for degenerate rows plus a mask. It has no kd-tree, so it can neither
widen a stencil nor fall back. A caller that ignores the mask silently
interpolates to zero. Use `interpolation_matrix` unless you specifically need
the bare weights — its rows are never empty.
```

### Row-wise construction carries no column guarantee

Every row has `nnn` non-zeros. **Nothing guarantees every column is non-empty.**
A source point that is not among the `nnn` nearest neighbours of any target
produces an empty column, and a consumer forming a Galerkin coarse operator
$P^{T} A P$ then gets a singular matrix — the failure mode of issue #424.

This is a property of *any* row-wise kNN construction, not of this kernel, and
it is not fixed by the scheme being linear-exact. Consumers that need full
column rank must check for and repair empty columns themselves.

## Choosing `nnn`

`order=1` requires **`nnn >= dim + 2`** and raises `ValueError` below that.

The reason is worth knowing. At exactly `nnn == dim + 1` the affine block $P$
is square, so the constraint $P^{T} w = [1, x^*]$ alone determines $w$: the RBF
block is inert and the scheme collapses to bare **barycentric interpolation on
the neighbour simplex**. That is singular whenever those `dim + 1` points are
collinear (2D) or coplanar (3D) — common near boundaries and on graded meshes.

The swarm proxy default is `nnn = 2 * (dim + 1)` (6 in 2D, 8 in 3D), which
leaves enough slack that degenerate neighbourhoods are rare. That is also what
`nnn=None` resolves to at `order=1`; `order=0` keeps its historical default of
4. The default has to depend on `order` *and* dimension — a fixed default of 4
is below the enforced minimum in 3D and would simply raise.

## Things that fail silently if you get them wrong

- **Build kd-trees from `.coords_nd`, never `.coords`.** `MeshVariable.coords`
  dimensionalises once the model has reference quantities set, while particle
  coordinates and `MeshVariable._get_kdtree` are non-dimensional. Mixing them
  raises from `_convert_coords_to_tree_units` (issue #426).
- **The output carries no units.** Coordinates are converted into the tree's
  frame, but `data` passes through untouched and the return is a plain array.
  Re-attach units at the boundary if the caller needs them.
- **`order=1` costs about 5x `order=0`** — a dense $(nnn+d+1)^3$ solve per
  target point. Irrelevant if you build an operator once; think twice in a hot
  loop.
- **Do not validate against `uw.function.evaluate`.** It returns wrong values
  for P1 fields at points lying exactly on cell edges in 3D (issue #432), and
  proxy nodes sit on cell boundaries routinely. Compare against an analytic
  field, or against an independent implementation, instead.

## Determinism

The weights are a pure function of the geometry: no global state, no random
numbers, no accumulation across calls. Repeated calls, and separate `KDTree`
instances built from equal point sets, return bit-identical results.

Where several source points are exactly equidistant from a target, *which* of
them the kNN search selects is unspecified — but the choice is deterministic,
so results reproduce run to run. Both properties are pinned by tests.

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

### Material level sets stay on `order=0` — variance, not bias

`IndexSwarmVariable` builds one level-set MeshVariable per material index and
keeps its own inverse-distance weighting. It was tested against `order=1` and
**deliberately not changed** — but not for the reason one might expect.

The tempting argument, *a material indicator is piecewise constant so linear
exactness has nothing to gain*, is **wrong**. The indicator is the *particle*
data; the quantity estimated at a node is the local material **fraction**,
which near an interface is a smooth ramp. Reproducing a ramp is exactly what a
constants-only scheme cannot do.

The real reason is that a level-set node estimates that fraction from a handful
of **integer** samples, so its error has a variance term as well as a bias term:

$$ \mathrm{Var}(\text{estimate}) = \sum_j w_j^2 \,\mathrm{Var}(f_j) $$

For weights summing to one, uniform weighting minimises $\sum_j w_j^2$ at
$1/nnn$. Inverse-distance weights are positive and stay near that floor;
linear-exact weights are signed and sit far above it:

| dim | `nnn` | $\sum w^2$ inverse distance | $\sum w^2$ `order=1` | $1/nnn$ | amplification |
|---|---|---|---|---|---|
| 2 | 6 | 0.223 | 2.314 | 0.167 | **10.4x** |
| 2 | 20 | 0.079 | 0.922 | 0.050 | **11.7x** |
| 3 | 6 | 0.188 | 1.290 | 0.167 | **6.9x** |
| 3 | 20 | 0.061 | 0.657 | 0.050 | **10.9x** |

Note the second row of each pair: inverse distance averages the noise *down* as
the stencil grows, while `order=1` barely moves. So the noise cannot be bought
off with more neighbours.

Measured end to end, with particles assigned material 1 with probability
$p(x)=x$ so that the exact nodal fraction is a known linear function
(three seeds, `cellSize` 1/16 and 1/32):

| scheme | bias | rms | level-set range |
|---|---|---|---|
| shipped `update_type=0` (scatter) | ~1e-3 | **0.09 – 0.11** | `[0, 1]` |
| shipped `update_type=1` (gather) | ~1e-3 | 0.18 | `[0, 1]` |
| gather, `order=1` | ~1e-3 | 0.18 – 0.20 | **`[-0.35, 1.32]`** |

The bias that linear exactness would remove is already ~1e-3 for every scheme,
because a roughly symmetric stencil reproduces a linear ramp in expectation
anyway. What is left is variance — and `order=1` amplifies it and breaks the
`[0, 1]` bound that `constitutive_models.py` relies on.

Partition of unity survives regardless of weight sign (all indices share one
denominator, and the per-particle indicator flags sum to one, so the level sets
sum to $\sum_j w_j = 1$). It was never the objection.

```{note}
The two branches are different algorithms, and only one has a stencil to
re-weight. `update_type=0`, the **default**, is a *scatter*: it queries `nnn`
nodes but masks with `is_nearest`, so each particle accumulates $1/d$ into its
nearest node only. `update_type=1` is the gather form. That the scatter has the
lowest rms is consistent with the variance argument — a node aggregates every
particle that is nearest to it, typically many more than `nnn`.
```

So the lever for material-fraction accuracy is **more samples per node**, not
better polynomial reproduction. The swarm story splits accordingly: the plain
`SwarmVariable` proxy takes `order=1` because it carries exact real values and
only bias matters; `IndexSwarmVariable` keeps inverse distance because it
carries quantised values and variance dominates.

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
