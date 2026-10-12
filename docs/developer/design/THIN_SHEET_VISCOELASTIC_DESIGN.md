# Thin viscoelastic sheet with faults — flat and on the sphere

**Status**: In progress (started 2026-10-08). Branch `feature/thin-sheet`.

## Goal

A vertically averaged (thin-sheet) model of the lithosphere carrying faults, on a
flat 2-D mesh and on the surface of a sphere, with a Maxwell viscoelastic
rheology and a viscous substrate. Headline result: the loading, slip and
postseismic relaxation of a strike-slip fault in a viscoelastic sheet, flat first
and then on the sphere.

## The sheet

With thickness $L$, tangential velocity $\mathbf u$ and the plane-stress closure
$\sigma_{nn} = 0$ on the dynamic stress, the pressure of an incompressible
material is eliminated: $p = \tau_{nn} = -\operatorname{tr}_s\boldsymbol\tau$, so

$$
\boldsymbol\sigma = \boldsymbol\tau + \operatorname{tr}(\boldsymbol\tau)\,\mathbf P,
\qquad \mathbf P = \mathbf I - \hat{\mathbf n}\hat{\mathbf n}^T,
\qquad
-\nabla_s\cdot(L\boldsymbol\sigma) + c\,\mathbf u = \mathbf f .
$$

$\boldsymbol\tau$ is any deviatoric constitutive law of the in-plane strain rate
$\dot{\boldsymbol\varepsilon}_s = \mathbf P\,\mathrm{sym}(\nabla_s\mathbf u)\,\mathbf P$
— Newtonian, Maxwell (the stress history is the in-plane deviatoric stress;
$\tau_{nn}$ follows from the trace), or yielding. $c$ is a basal drag
($\eta_a/h_a$ for a viscous substrate); $\mathbf f$ a force per unit area.

Consequences used as hard baselines:

- uniaxial stretch with free lateral edges: $\sigma_{xx} = 3\eta\dot\varepsilon$ (Trouton);
- rigid motion: $\boldsymbol\sigma = 0$ exactly;
- Maxwell uniaxial loading: $\sigma = 3\eta\dot\varepsilon\,(1 - e^{-t\mu/\eta})$;
- a fault in a thin sheet: Barr & Houseman (1996) Appendix (A10)–(A13) — the
  plane-strain (A1)–(A9) solution is already in `underworld3.analytic`; the
  thin-sheet one is to be added beside it;
- stress diffusion with basal drag after imposed slip (Elsasser; Lehner, Li &
  Rice 1981).

Not in the first round (recorded so they are not forgotten): thickness
evolution $\dot L = -L\,\nabla_s\cdot\mathbf u$ with the gravitational potential
energy gradient (England & McKenzie's Argand number); yield in the sheet, whose
invariant must include $\tau_{nn} = -\operatorname{tr}\boldsymbol\tau$ (the
plane-strain code omits it).

## Implementation

`uw.systems.ThinSheet` (`systems/thin_sheet.py`) is an `SNES_Vector`: there is
no pressure unknown, so no saddle point. Its inner `_Unknowns` hands the
constitutive model the projected strain rate through `Unknowns.E` (the VEP model
already reads it; `ViscousFlowModel.grad_u` now does too — identical on a volume
mesh). Templates: `F1 = L (τ + tr τ P)`, `F0 = -f + c u`.

### On the sphere

The velocity is carried in the embedding space (3 components on a
`SphericalManifold`, `dim=2`, `cdim=3`). PETSc already supplies the surface
gradient along the flat facets in the derivative index; the outer projection in
$\dot{\boldsymbol\varepsilon}_s$ supplies the curvature term
$-\mathbf u\cdot\nabla_s\hat{\mathbf n}$. Two normals are involved and they
are deliberately different:

- the **projector** uses the facet's own normal (a cell-wise constant field
  built once per mesh). The gradient is taken along the flat facet, so a rigid
  rotation — a Killing field, strain-free on the sphere — is strain-free in the
  discrete form only with the facet normal; with the analytic normal it keeps a
  spurious strain of order $h$ (measured 0.7% of the velocity at $h = 0.25$);
- the **tangency constraint** $\mathbf u\cdot\hat{\mathbf n}=0$ uses the
  analytic normal of the surface, imposed strongly at every node through the
  per-node rotation of `utilities/rotated_bc.py` (`SHEET_TANGENCY`, the same
  machinery as rotated free-slip and the split-node fault constraint, so faults
  compose with it). The nodes lie on the sphere, so a rigid rotation is exactly
  admissible and is reproduced to solver tolerance.

A penalty was used only to probe feasibility and is not an API. The toroidal
manufactured solution converges at second order (the flat-facet geometry limit
for P2) with tangency at machine precision.

Vector solves on a manifold needed four `dim → cdim` corrections (the manifold
vector-FE gap): the `SNES_Vector` FE component count, its PC block size and
default unknown, `maths/vector_calculus` (component count), and the
`sympy.vector` truncation in `_jitextension`.

### Faults on the sphere (Phase 5)

The split-node pipeline is topological and survives; what has to generalise is
geometry: great-circle crossings in `line_cut`, cell orientation by the outward
normal in `reconnect`, the contact frame $\{\hat\nu = \hat r\times\hat t,\ \hat t,\
\hat r\}$ in `fault_contact` (PETSc's FVM edge normal is wrong for `cdim=3`), and
the `mesh.dim` strides in `rotated_bc`, whose rigid-rotation gauge must offer the
three Killing fields. `mesh.adapt` (NVB) runs on the manifold already; refined
vertices need snapping back to the sphere.

## Phases

0. Feasibility probes (sheet operator on S² with penalty and with the strong
   rotation; NVB adapt on S²; splitting an arc) — scratch, not committed.
1. Flat viscous `ThinSheet` + tests (`tests/test_1075_thin_sheet.py`).
2. Viscoelastic sheet (Maxwell; small-strain non-advected history for fault
   problems; Elsasser check).
3. Faults in the flat sheet (rotated/fault BCs hoisted off the saddle class; a
   vector-only branch of the rotated solve; BH96 thin-sheet solution).
4. Sheet on the sphere with strong tangency and the Killing null space.
5. Faults on the sphere.
6. Results and examples.

## One model, thin or thick

Principle (maintainer, 2026-10-10): the same physics is the same model script. A model
must move between the thin sheet (`ThinSheet` on a surface mesh) and a thick sheet
(`Stokes` on a thin 3-D mesh) by changing the mesh and the solver class, nothing else.
The solver owns the kinematics, the closure and the geometry; the model owns the
rheology. What is already shared: the constitutive models, the fault interface API, the
boundary-condition methods, the stress history and `solve(timestep=)`.

State of the alignment (2026-10-12):

| | thin sheet | thick sheet | aligned form | state |
|---|---|---|---|---|
| construction | `ThinSheet(mesh, v, p, degree, p_continuous)` | `Stokes(mesh, v, p, ...)` | same signature; the sheet's `p` is the diagnostic $-\operatorname{tr}\tau$, refreshed after each solve | done |
| mesh | the sheet mesh | `mesh.extrude(L, layers)` | the thick twin keeps the sheet's boundaries as walls (same names) and adds `Base` / `Surface` | done (serial); MG tail coarsens through the thickness first |
| forcing | body force per unit volume, times $L$ | per unit volume | `bodyforce` means the same on both | done |
| basal coupling | `add_basal_drag(c, "Base", reference_velocity=)`, `add_basal_traction(t, "Base")` | the same, as natural conditions on `Base` | one line serves both; the sheet ignores the boundary name | done |
| fault geometry | `mesh.add_fault` | extruded with the mesh: a vertical fault through the layer, split at surface and base, tips shared | `add_fault_bc` on either | done flat (test_1083: velocity 0.5 %, slip 1.6 %); sphere implemented, untested; faulted layer has no MG tail yet (TODO(BUG)) |
| stress | `solver.stress` (in-plane, tension positive) | `solver.stress` (3-D) | `uw.maths.tensor.resolved_traction` / `coulomb_failure_stress` on `solver.stress` | done |
| nonlinear invariants | `Unknowns.second_invariant` adds $M_{nn} = -\operatorname{tr}M$ | plain contraction | every model invariant goes through the solver | done |
| Newton tangent | `_flux_closure` | `_flux_closure` | the model's `flux_jacobian` is composed with the solver's closure (it used to replace the whole flux, and was ignored on vector solvers) | done |
| topography | implied isostatic rate | free surface | with thickness transport (round 2) | open |

Mesh-pairing notes. A thin 3-D fault cannot come from the 3-D splitter: it keeps
the patch rim welded (the tip rule one dimension up), so a fault through a layer
would be locked at the surface and the base, and it needs about nine cells of
depth. Extruding the already-split sheet avoids both: the slit is duplicated at
every level, so the fault is open where it meets the faces, and the two sides are
cut into tetrahedra identically (the minus copy of a vertex takes its plus copy's
place in the vertex order), so the pairing is rebuilt by joining mapped vertex
tuples.
