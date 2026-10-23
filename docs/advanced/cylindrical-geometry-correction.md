# Circular geometry correction on affine annulus meshes

These explicitly selected Stokes methods support two-dimensional, affine
triangular annuli with continuous P2 velocity, continuous P1 regular pressure
and constant viscosity. Use nondimensional mesh coordinates, fields and input
parameters throughout; active reference-unit scaling is unsupported.
They correct source integration, wall data and reaction
recovery; they do not replace straight elements with curved elements. Numerical
validation currently targets the isoviscous annulus benchmarks. They do not
establish support for spherical shells, variable viscosity or other FE pairs.

## Pressure extraction for a prescribed circular sheet

For a radial sheet load at $r=R$, write

$$
p=q+p_J,\qquad p_J=-A H(r-R)b(r)\cos(n\theta),\qquad
b(r)=\frac{r_o-r}{r_o-R}.
$$

In $-\nabla\cdot\tau+\nabla p=f$, the distributional gradient of $p_J$
cancels $f=-A\cos(n\theta)e_r\delta(r-R)$. The regular equation uses
$A H(r-R)\nabla[b\cos(n\theta)]$. Circular-cut integration constructs six
P2 load moments on each outer triangle. A DG P2 field represents these moments
under the native quadrature; it is a force representation, not a new velocity
or pressure element pair.

```python
lift = stokes.cylindrical_pressure_lifting(
    interface_radius=2.0, outer_radius=2.22, wavenumber=2,
    amplitude=1.0, quadrature_points=12,
)
stokes.bodyforce = lift.regular_force.sym
stokes.add_natural_bc(lift.natural_datum, "Upper")
physical_pressure = stokes.p.sym[0] + lift.pressure_jump
```

The natural datum is $-p_J\Gamma$ on the polygonal outer wall. Although
$b(r_o)=0$, $b$ is nonzero inside each outer chord. Dropping this term changes
the regular problem. `pressure_jump_integral` permits physical pressure-gauge
calibration; it is the owned-cell integral of the prescribed extraction.
Smooth density problems use their original volume force and physical P1 pressure;
they do not require a pressure lift.

## Numerical transfer of circular free-slip conditions

At a polygon wall radius $r_h$, let $d=R-r_h$. Lagged numerical traces impose
$u_r(r_h)=-d\partial_r u_r$ and replace shear stress by
$\sigma_{rt}(r_h)=-d\partial_r\sigma_{rt}$. Diagonal stresses remain numerical.
The force moments are projected tangentially before inversion of the consistent
P2 line mass. Discrete normal flux and rigid-rotation torque are projected to zero.
No analytical Stokes solution supplies the wall data.

```python
stokes.petsc_use_pressure_nullspace = True
walls = {}
for boundary, radius, sign in (("Lower", 1.22, -1), ("Upper", 2.22, 1)):
    walls[boundary] = stokes.add_cylindrical_wall_transfer(
        boundary=boundary, radius=radius, sign=sign, viscosity=1.0,
        interface_radius=2.0, outer_radius=2.22, wavenumber=2,
        amplitude=1.0, quadrature=12,
    )
history = stokes.solve_cylindrical_wall_updates(walls=walls, updates=4)
```

Create wall fields before the first solve and before another free-slip condition
is registered on these boundaries. Use `amplitude=0` for smooth physical pressure.
Four updates produce five accepted solves, with final fields retained from the
last solve. A converged, zero-increment warm start is accepted. Failed histories
remain on `stokes.cylindrical_wall_history`; file output belongs to the caller.
Enable `stokes.petsc_use_pressure_nullspace = True` before the first solve.
The wall history reads the projected pressure-gauge residual produced by that
configuration; other pressure-gauge configurations are outside this helper's
supported contract. The helper does not install the pressure projection or
velocity nullspace itself. Configure the rigid-rotation velocity modes separately
when required by the enclosing free-slip system.

## Spatial circular reaction and topography

The circular reaction moments are

$$
R_{\rm circle}=R_{\rm CBF}+L_{\rm circle}(\sigma_h)-L_{\rm polygon}(\sigma_h).
$$

The circular stress uses the first-order radial Taylor transfer. Each geometric
load has its own mean removed; the circular consistent line mass determines the
spatial trace. The original assembled CBF contribution is preserved, including
its numerical residual, and its solver cache is not overwritten.

```python
checks = stokes.recover_cylindrical_topography(
    wall=walls["Upper"], field=outer_topography, buoyancy_scale=-1.0,
)
```

The target must be a continuous scalar P2 field created before solving. The
method writes only this wall and returns mass-moment, mean and length checks.
Direct-field Poisson consumes the spatial field. Spectral gravity subsequently
extracts a mode using

```python
coefficient = stokes.cylindrical_boundary_coefficient(
    wall=walls["Upper"], field=outer_topography, wavenumber=2,
)
```

Coefficient extraction removes the circle-weighted mean and uses the exact
radial-projection line Jacobian. It is not part of spatial reaction recovery.

## Parallel scope and reproducibility

Call the methods collectively. Only owned boundary-cell traces and boundary
nodes are replicated; full volume fields are never gathered. Small SciPy sparse
solves invert the replicated one-dimensional boundary mass systems. The Stokes
operator and volume solves remain distributed PETSc. Wall handles cache geometry
and are valid only for their original mesh, fields and fixed parameters.

Record pressure representation, quadrature, source amplitude, wall-update history,
reaction checks and the target actually supplied to gravity. Keep complete spatial
errors distinct from modal errors. A correction changes the numerical problem
on an unchanged affine mesh; both baseline and corrected cases must be solved
independently for a scientific before/after comparison.

## References and attribution boundaries

- Chang, K. S., and Kwak, D. Y. (2009), *Extraction method for Stokes Flow with
  jumps in the pressure*, [arXiv:0911.4769](https://arxiv.org/abs/0911.4769).
  Section 2, equations (4)-(6), motivates regular/singular pressure splitting.
  UW3 uses its own prescribed radial profile, exact-circle cap integration,
  DG P2 load moments and outer-chord natural datum, rather than the paper's
  harmonic extension and CR discretisation.
- Atallah, N. M., Canuto, C., and Scovazzi, G. (2020), *The second-generation
  Shifted Boundary Method and its numerical analysis*, Computer Methods in
  Applied Mechanics and Engineering 372, 113341,
  [doi:10.1016/j.cma.2020.113341](https://doi.org/10.1016/j.cma.2020.113341),
  [author preprint](https://arxiv.org/abs/2004.10584). Section 2.2, equations
  (8)-(13), supplies the Taylor-shift motivation. Its Section 4.2 uses a
  Nitsche formulation and assumes body-fitted Neumann boundaries. UW3's lagged
  strong free-slip, tangential load projection and compatibility corrections
  are separate constructions; that analysis provides no stability proof here.
- Zhong, S., Gurnis, M., and Hulbert, G. (1993), *Accurate determination of
  surface normal stress in viscous flow from a consistent boundary flux
  method*, Physics of the Earth and Planetary Interiors 78(1-2), 1-8,
  [doi:10.1016/0031-9201(93)90078-N](https://doi.org/10.1016/0031-9201(93)90078-N).
  This is the CBF stress-recovery reference. UW3's circular moment transfer,
  Taylor reaction correction and weighted coefficient integration extend the
  calculation and are not results established by that paper.
