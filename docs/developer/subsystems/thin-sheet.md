# Thin sheet: `uw.systems.ThinSheet`

A vertically averaged (plane-stress) sheet solver for the lithosphere, on a flat
mesh or on the surface of a sphere, with any deviatoric constitutive model —
Newtonian, Maxwell viscoelastic, or yielding — and the split-node fault machinery.

Design record: `docs/developer/design/THIN_SHEET_VISCOELASTIC_DESIGN.md`.
Tests: `tests/test_1075`–`test_1079`. Session drivers: `scripts/sessions/thin_sheet_*.py`.

## The equations

With thickness $L$, tangential velocity $\mathbf u$ and the plane-stress closure
$\sigma_{nn} = 0$ on the dynamic stress, the pressure of an incompressible material is
$p = \tau_{nn} = -\operatorname{tr}\boldsymbol\tau$ and

$$
-\nabla_s\cdot\big(L\,\boldsymbol\sigma\big) + c\,\mathbf u = \mathbf f,
\qquad
\boldsymbol\sigma = \boldsymbol\tau + \operatorname{tr}(\boldsymbol\tau)\,\mathbf P,
\qquad
\dot{\boldsymbol\varepsilon}_s = \mathbf P\,\mathrm{sym}(\nabla_s\mathbf u)\,\mathbf P .
$$

$\boldsymbol\tau$ is the deviatoric stress the constitutive model returns for the
in-plane strain rate $\dot{\boldsymbol\varepsilon}_s$; $c$ is a basal drag
coefficient ($\eta_a/h_a$ for a viscous channel); $\mathbf f$ a force per unit
area, which for a sheet dragged by a moving substrate is $c\,\mathbf u_s$. There is
no pressure unknown: the solver is an `SNES_Vector`.

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
sheet.basal_drag = c
sheet.bodyforce = list(c * u_substrate)
sheet.add_fault_bc(0, "Flt")             # after mesh.add_fault(("Flt", points))
sheet.solve(timestep=dt)
```

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

A stress history on a manifold is carried on the grid (`stress_transport =
"eulerian"`, the streamline-upwind transport the Eulerian solvers use); the
semi-Lagrangian trace-back is refused there.

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

## Round 2 (not yet)

Thickness evolution with the gravitational potential energy term (England &
McKenzie's Argand number) — on the sphere with the Eulerian advection–diffusion —
and yield in the sheet with the invariant that includes $\tau_{nn}$.
