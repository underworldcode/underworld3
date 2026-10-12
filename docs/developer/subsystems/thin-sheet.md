# Thin sheet: `uw.systems.ThinSheet`

A vertically averaged (plane-stress) sheet solver for the lithosphere, on a flat
mesh or on the surface of a sphere, with any deviatoric constitutive model —
Newtonian, Maxwell viscoelastic, or yielding — and the split-node fault machinery.

Design record: `docs/developer/design/THIN_SHEET_VISCOELASTIC_DESIGN.md`.
Tests: `tests/test_1075`–`test_1084`. Session drivers: `scripts/sessions/thin_sheet_*.py`.

## The equations

With thickness $L$, tangential velocity $\mathbf u$ and the plane-stress closure
$\sigma_{nn} = 0$ on the dynamic stress, the pressure of an incompressible material is
$p = \tau_{nn} = -\operatorname{tr}\boldsymbol\tau$ and

$$
-\nabla_s\cdot\big(L\,\boldsymbol\sigma\big) + c\,(\mathbf u - \mathbf u_\mathrm{ref})
= L\,\mathbf f + \mathbf t_b,
\qquad
\boldsymbol\sigma = \boldsymbol\tau + \operatorname{tr}(\boldsymbol\tau)\,\mathbf P,
\qquad
\dot{\boldsymbol\varepsilon}_s = \mathbf P\,\mathrm{sym}(\nabla_s\mathbf u)\,\mathbf P .
$$

$\boldsymbol\tau$ is the deviatoric stress the constitutive model returns for the
in-plane strain rate $\dot{\boldsymbol\varepsilon}_s$; $c$ is a basal drag
coefficient ($\eta_a/h_a$ for a viscous channel) against the velocity of what lies
beneath; $\mathbf f$ a body force per unit **volume** (as for `Stokes`) and
$\mathbf t_b$ a basal traction. There is no pressure unknown: the solver is an
`SNES_Vector`, and $p = -\operatorname{tr}\boldsymbol\tau$ is available as the
diagnostic `sheet.p`. Every invariant a constitutive model forms includes the
out-of-plane component $M_{nn} = -\operatorname{tr}M$ (`Unknowns.second_invariant`),
so a yielding sheet yields at the von Mises stress.

Closed forms the tests pin: uniaxial stretch with free edges gives the Trouton
stress $3\eta\dot\varepsilon$; a rigid motion carries no stress; the Maxwell
uniaxial response is $3\eta\dot\varepsilon(1-e^{-t\mu/\eta})$; Elsasser stress
diffusion into an elastic sheet over a viscous substrate has $D = \mu L/c$; a
frictionless fault under remote shear slips as the plane-stress Griffith crack
with $E' = 3\eta$.

```python
sheet = uw.systems.ThinSheet(mesh)
sheet.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel(
    sheet.Unknowns, order=1, integrator="etd")
sheet.constitutive_model.Parameters.shear_viscosity_0 = eta
sheet.constitutive_model.Parameters.shear_modulus = mu
sheet.thickness = L
sheet.add_basal_drag(c, "Base", reference_velocity=u_substrate)
sheet.add_fault_bc(0, "Flt")             # after mesh.add_fault(("Flt", points))
sheet.solve(timestep=dt)
```

## One model, thin or thick

The same model function runs as a thin sheet and as a thick one — `Stokes` on a
thin 3-D layer — by changing the mesh and the solver class:

```python
def model(mesh, Solver):
    solver = Solver(mesh)
    solver.constitutive_model = uw.constitutive_models.ViscousFlowModel
    solver.constitutive_model.Parameters.shear_viscosity_0 = eta
    solver.add_basal_drag(c, "Base", reference_velocity=u_base)
    solver.add_fault_bc(0, "Flt")
    return solver

sheet = model(sheet_mesh, uw.systems.ThinSheet)             # sheet.thickness = L
layer = model(sheet_mesh.extrude(L, layers=2), uw.systems.Stokes)
```

`mesh.extrude(L, layers)` (`uw.meshing.extrude_sheet`) builds the layer from the
sheet mesh: the sheet's boundaries become its walls under the same names, its faces
are `Base` and `Surface`, and every split-node fault of the sheet becomes a vertical
fault through the whole layer — open at the surface and the base, its tips shared
vertical edges — with the side labels and the pairing rebuilt, so
`Stokes.add_fault_bc` works on it. The thin sheet accepts the boundary name of
`add_basal_drag` / `add_basal_traction` and ignores it (it is its own base).
`uw.maths.tensor.resolved_traction` and `coulomb_failure_stress` act on either
solver's `stress`. A constitutive model's tangent flux (`flux_jacobian`) is
composed with each solver's own closure (`_flux_closure`).

Measured (`test_1083`, $L = 0.04$, $cL/\eta = 0.04$): the layer's mid-plane
velocity matches the sheet's to 0.7 % unfaulted and 0.5 % faulted, the slip to
1.6 %. Limits: serial only; a faulted layer gets no geometric-multigrid tail yet
(the rotated 3-D pair operator fails under custom multigrid, marked TODO(BUG) in
`meshing/extrude.py`), so it solves with algebraic multigrid; the radial extrusion of a spherical
manifold is implemented but not yet tested.

## On a sphere

The velocity on a `SphericalManifold` (`dim=2`, `cdim=3`) is carried in the
embedding space. Two normals are involved, and they are different on purpose:

- the **in-plane projector** uses the cell's facet normal (a cell-wise constant
  field built once per mesh): the assembler differentiates along the flat facets,
  and only with the facet normal is a rigid rotation — a Killing field —
  strain-free in the discrete form. With the analytic normal it keeps a spurious
  strain of order $h$;
- the **tangency constraint** $\mathbf u\cdot\hat{\mathbf n} = 0$ uses the analytic
  surface normal at every node, imposed strongly through the per-node rotation of
  `utilities/rotated_bc.py` (`SHEET_TANGENCY`), so a rigid rotation is exactly
  admissible. Faults compose with it because the split-node pair blocks use the
  same machinery.

The rigid-rotation gauge offers the three Killing fields; with basal drag they are
not null modes and are rejected by the operator test, as they should be.

The stress history is carried on the grid (`stress_transport = "eulerian"`, the
streamline-upwind transport the Eulerian solvers use) by default. On a manifold it
is the only choice; the semi-Lagrangian trace-back is refused there. On a flat
mesh it can be chosen, but next to a split-node fault its point evaluation reads
across the slit: with a sliding Coulomb fault it produced a period-2-step
oscillation of the near-tip slip rate and patchy stress along the trace, both of
which the grid history removes, at a sixth of the Newton work.

## Faults on a sphere

`mesh.add_fault(("name", points))` with three-coordinate points reads the polyline
as great-circle arcs about the origin. The coincident pair at a fault node carries
the frame $\{\hat\nu, \hat t, \hat r\}$ — the in-surface conormal of the fault edge
(orthogonalised against the node's surface normal), the trace tangent, the surface
normal — and pins the jump along $\hat\nu$ (no opening) and both sides' tangency
(the jump and the mean along $\hat r$); the slip row stays free, or carries the
interface law. Diagnostics (`fault_slip`, `fault_pair_jumps`,
`fault_normal_traction`) read the same frame.

Known limits: faults may not reach the mesh boundary (2-D rule); on a split mesh
`uw.function.evaluate` within about $h/2$ of the slit is unreliable — read nodal or
pair data there; `mesh.adapt` runs on a manifold but does not yet snap the refined
vertices back to the sphere.

## The quadratic benchmark

On the sphere a divergence-free toroidal drag thins a corotational Maxwell sheet at
second order (the Weissenberg effect): $-\nabla_s\cdot\mathbf u/A^2 =
(24/91)P_2 - (24/287)P_4$ in steady state, scaled by ETD-1's effective relaxation
time $\alpha\Delta t/(1-\alpha)$. `test_1084`; the closed form is derived by
`scripts/sessions/thin_sheet_quadratic_closed_form.py`.

## Round 2 (not yet)

Thickness evolution with the gravitational potential energy term (England &
McKenzie's Argand number) — on the sphere with the Eulerian advection–diffusion —
and topography from it.
