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
# Register only on a natural/free-slip wall; omit for full velocity Dirichlet.
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

## Full-vector zero-slip wall transfer

The separate zero-slip API registers one vector P2 essential datum per wall.
For each boundary node $x_h$, project to $P(x_h)=R x_h/|x_h|$ and update

$$g_h(x_h)=u_h(x_h)-u_h(P(x_h)).$$

This evaluates the complete one-sided P2 polynomial, including the quadratic
Taylor term. The radial shift has the same coordinate sign at both walls;
the inner/outer normal sign enters only polygon-flux integration. No exact
Stokes field, pressure trace or harmonic coefficient supplies the update.

```python
walls = {
    "Lower": stokes.add_cylindrical_zero_slip_transfer(
        boundary="Lower", radius=1.22, sign=-1, quadrature=12),
    "Upper": stokes.add_cylindrical_zero_slip_transfer(
        boundary="Upper", radius=2.22, sign=1, quadrature=12),
}
stokes.petsc_use_pressure_nullspace = True
history = stokes.solve_cylindrical_zero_slip_updates(walls=walls, updates=4)
```

Use standard Stokes with both velocity components prescribed; there is no
rigid-rotation nullspace. Creating the handles registers the essential fields
once. Later updates change their values before each standard solve. Pressure
extraction is independent: zero-slip case3 uses the same regular force and
physical $q+p_J$, with no outer natural datum because its test trace is zero.

Before each solve, remove the combined polygon flux using the normal P2
midpoint bubbles on both walls. For total chord length $L$ and raw net flux $F$,
subtract $c n_h$ at every midpoint, where $c=F/(2L/3)$. Vertex and midpoint
tangential values remain unchanged. This is the minimum integrated-L2 change
within that bubble span. It enforces one global compatibility condition;
individual wall fluxes are recorded rather than fitted independently.

`cylindrical_zero_slip_history` retains accepted standard-solver reports,
flux before/after projection, its L2 change, and true-circle nodal/full-Gauss
velocity residuals. At a fixed point a nonzero $c$ perturbs the true-circle
constraint; verify that defect and its refinement behaviour. Four updates are
an explicit experiment schedule, not a convergence guarantee.

Ruas (2019), DOI10.1016/j.camwa.2018.05.029, sections2/4 and equations29–30,
provides the close P2 true-boundary trial/canonical polygon-test precedent.
UW3 uses symmetric stress and a global flux projection instead of that exact
formulation. Atallah et al. (2020), section4, and Burman et al. (2018),
https://arxiv.org/abs/1801.07463, motivate shifted Dirichlet data through different
weak formulations. None of these results is claimed as a proof of this strong
numerical fixed point, especially for the delta case's piecewise regularity.
Independent polynomial/flux/parallel tests pass; model acceptance remains pending
in the benchmark investigation.

### Full Dirichlet traction recovery

Standard Stokes can recover the full Cartesian traction from essential-boundary
reactions with `boundary_flux(..., mass="consistent", remove_mean=False)`.
Its fresh volume-residual assembly uses the current pressure gauge and registered
essential data. For saved-state replay, restore both the solved velocity/pressure
and the last `ZeroSlipShiftLower/Upper` auxiliary fields; replacing those data by
zero or by another wall update changes the state being measured.

`recover_cylindrical_zero_slip_traction` transfers the regular-pressure reaction
to the physical circle using one-sided numerical stress:

$$R_c=R_q+L_c(\sigma_{q+p_J})-L_h(\sigma_q).$$

The polygon subtraction must use **regular** stress when the solved unknown is
$q$. A prescribed lift can vanish on the true wall while remaining nonzero on
the chords. Using physical stress in that subtraction without first converting
the regular reaction leaves a spurious load. Circle continuation uses the complete
affine-cell P2/P1 polynomial, rather than an analytical Stokes solution.

The line system has Cartesian P2 test components and rotating trial vectors
$[n,e_\theta]$ multiplying scalar P2 normal/tangential coefficients. Thus it
preserves every transferred Cartesian weak moment and represents a constant
normal pressure exactly. This avoids contracting reactions with nodal normals.
The public shear target is $\sigma_{r\theta}$; the outward tangential traction
has the additional inner/outer `sign`. No wall mean is removed.

```python
checks = stokes.recover_cylindrical_zero_slip_traction(
    wall=walls["Upper"], normal_field=normal_circle, shear_field=shear_circle,
    recovered_normal_field=normal_recovered, recovered_shear_field=shear_recovered,
    viscosity=1.0, pressure_jump=pressure_jump, degree=6,
    halfwidth_edges=5, quadrature=24,
)
```

All four targets are distinct continuous scalar P2 fields on the Stokes mesh.
Call after the global pressure calibration. The optional spatial reconstruction
is applied separately to the two rotating scalar traces, preserving each mean
and first angular moments and therefore their combined Cartesian resultant.
It changes finer transferred moments; the circular CBF fields remain available.
Returned checks distinguish the original polygon resultant, geometry-transfer
change, transferred resultant, and subsequent reconstruction changes. The
kernel's physical bandwidth limits described below still apply to shear.

For an exact stationary incompressible zero-slip solution, $u=0$ along the
circle implies $\partial_r u_r=0$, so $p=-\sigma_{nn}$. This gives a derived
**boundary pressure estimate**, not an improved FE pressure field. Keep the
original pressure trace and the defect $2\mu\partial_r u_r$ in verification;
weak incompressibility does not make that identity exact for the discrete field.

Brunátová et al. (2025), Section 2.3.2, equations (9)-(10), gives a direct
variational precedent for full Dirichlet traction and its tangential decomposition.
Its Stokes example uses gradient pseudostress, while UW3 uses symmetric stress.
Neither it nor the CBF reference proves this circular-transfer/rotating-trial/
weighted-reconstruction adaptation stable or superconvergent.

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

## Spatial trace reconstruction

Retain the circular CBF trace and reconstruct into a **distinct** continuous
scalar P2 field created alongside the other mesh variables before solving:

```python
checks = stokes.reconstruct_cylindrical_trace(
    wall=walls["Upper"], source=cbf_topography, target=reconstructed_topography,
    degree=6, halfwidth_edges=5,
)
```

This is an explicitly selected spatial postprocessor. It does not change the
Stokes solution or the cached assembled reaction. The compact local-L2 kernel
uses the endpoint weight $w(z)=(1-z^2)^2$ and reproduces evaluation at zero for
polynomials through the requested even degree (4, 6 or 8). Degree 6 has the
normalized kernel

$$
K_6(z)=\frac{(1-z^2)^2}{8192}
(24255-315315z^2+945945z^4-765765z^6),\qquad |z|\leq1,
$$

and is zero outside its support. The physical support halfwidth is
`halfwidth_edges` times the mean circular edge arc length, not a prescribed
source wavelength. The currently measured $n=2$ annulus choice is degree 6 and
halfwidth 5. The degree describes the kernel moments; the output remains P2.

Convolution integrates the existing P2 trace using its radial projection onto
the true circle. Integration is split at original and shifted source vertices,
including on nonuniform walls, before projection onto the circular consistent
P2 mass. A minimum circular-L2 correction preserves
$\int f\,ds$, $\int f\cos\theta\,ds$ and $\int f\sin\theta\,ds$.
The latter two are Cartesian components of the radial resultant, defined by
geometry rather than a fitted benchmark wavenumber.

The reconstructed field **changes fine-scale nodal CBF moments**. Keeping every
such moment in the same P2 space would uniquely retain the original CBF trace.
Keep both fields, record which is supplied to later calculations, and assess
complete spatial errors as well as coefficients. Polynomial reproduction of the
kernel is not a proof of superconvergence of the complete annular Stokes chain.
The spline/DG SIAC results discussed by Docampo-Sanchez et al. (2020) do not
establish this weighted polynomial operator's accuracy on circular or graded
CBF traces.

The finite support imposes a bandwidth limit: genuine rapid angular variations
within that support can be attenuated or amplified. For example, the degree-6 kernel with
halfwidth 5 has continuous response about 0.8213 for an eighth harmonic on a
32-edge circle. No automatic tuning to a source wavenumber is performed. The
caller must assess supported physical scales as well as unwanted spatial errors.
Returned checks record actual fine-scale nodal-moment changes, circular-L2 change
and physical-moment conservation, alongside the kernel degree and support.

### Pressure calibration and the reaction cache

Pressure calibration may change the solved pressure by a constant while the
cached Cartesian reaction retains the solve's earlier gauge. Circular recovery
checks that the difference is constant and evaluates its numerical Taylor trace
in the cached-reaction gauge. It leaves the calibrated pressure unchanged.
A nonconstant pressure change makes the cache stale and is rejected; solve
Stokes again before recovery. No analytical pressure or wall
reference is used for this synchronization.

## Parallel scope and reproducibility

Call the methods collectively. Only owned boundary-cell traces and boundary
nodes are replicated; full volume fields are never gathered. Small SciPy sparse
solves invert the replicated one-dimensional boundary mass systems. The Stokes
operator and volume solves remain distributed PETSc. Wall handles cache geometry
and are valid only for their original mesh, fields and fixed parameters.

Record pressure representation, quadrature, source amplitude, wall-update history,
reaction checks, reconstruction degree/support and the target actually supplied to gravity. Keep complete spatial
errors distinct from modal errors. A correction changes the numerical problem
on an unchanged affine mesh; both baseline and corrected cases must be solved
independently for a scientific before/after comparison.

## References and attribution boundaries

- Brunátová, J., Dokken, J. S., Valen-Sendstad, K., and Hron, J. (2025),
  *On the Numerical Evaluation of Wall Shear Stress Using the Finite Element
  Method*, International Journal for Numerical Methods in Biomedical Engineering
  41(9), e70086, [doi:10.1002/cnm.70086](https://doi.org/10.1002/cnm.70086).
  Section 2.3.2 supplies the full Dirichlet variational recovery class and
  normal/tangential decomposition. UW3's circular moments, symmetric Stokes
  form, rotating trial basis and spatial reconstruction are separate adaptations.
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

- Docampo-Sanchez, J., Jacobs, G. B., Li, X., and Ryan, J. K. (2020),
  *Enhancing accuracy with a convolution filter: What works and why!*,
  Computers & Fluids 213, 104727,
  [doi:10.1016/j.compfluid.2020.104727](https://doi.org/10.1016/j.compfluid.2020.104727).
  This supplies the accuracy-enhancing convolution context. UW3's endpoint-weighted
  polynomial kernel, circular P2 projection and physical-moment restoration are
  distinct from its spline/DG operator; its convergence results are not inherited.
