# Micromorphic plasticity as a first-class constitutive choice

Status: proposal, 2026-10-17. Branch `feature/shear-band-length-scale`.

## What it is

The micromorphic (gradient-plasticity) regularisation gives a yield law a length
scale. An extra field $\chi$, the strain-rate invariant smoothed over $\ell$, is
solved in the same Newton system as the flow:

$$
H\,(\chi - \dot\varepsilon_{II}) - \nabla\cdot\left(H\ell^2\,\nabla\chi\right) = 0,
\qquad
\tau_y \;\rightarrow\; \tau_y + H\,(\dot\varepsilon_{II} - \chi).
$$

Modes longer than $\ell$ see no extra stiffness; modes shorter than $\ell$ (the
collapsing band) see about $H$. The shear-band campaign uses it on the Spiegelman
notch in 2-D and 3-D (`uw3-campaigns/notch_length_scale`, REPORT.md).

Today it is written by hand in each script: a `MeshVariable`, a
`solver.add_coupled_field(chi, F0, F1)` call, and the shift added inside the yield
stress before its floor. About ten lines, easy to get subtly wrong (the floor order,
the region mask, units of $H$).

## The proposal: the model declares, the solver provides

The same pattern as the stress history of the viscoelastic models: the constitutive
model states what it needs, and the solver builds it during set-up.

**User side.** Two parameters on every yielding model (ViscoPlastic,
ViscoElasticPlastic, TransverseIsotropicVEP), next to `plastic_rate_strengthening`:

```python
cm = uw.constitutive_models.ViscoPlasticFlowModel
cm.Parameters.yield_stress = C + sin_phi * p
cm.Parameters.plastic_length_scale = uw.quantity(1, "km")       # ell; 0 = off (default)
cm.Parameters.plastic_gradient_modulus = uw.quantity(1e20, "Pa*s")  # H; see default below
stokes.solve()
```

Setting a non-zero length scale is the whole switch. Nothing else in the script
changes: no field to create, no coupled field to register, no solver options.

**Model side.**
- Owns $\chi$: a P1 scalar `MeshVariable` (units 1/s), created lazily on the
  solver's mesh, named after the model so two models cannot collide.
- Adds the shift $H(\dot\varepsilon_{II} - \chi)$ to the yield stress *before* the
  floor (`yield_stress_min`), in one helper shared by the three yielding families.
- Declares its coupled-field request: $(\chi, F_0, F_1)$ with
  $F_0 = H(\chi - m\,\dot\varepsilon_{II})$, $F_1 = H\ell^2\nabla\chi$.

**Solver side** (all automatic once the request exists):
- registers the request in `_setup_pointwise_functions` (idempotent; a changed
  request is a rewire, as a tangent change is);
- groups $u$ and $\chi$ in split 0 of the fieldsplit (on this branch already);
- geometric multigrid on split 0: native nested FMG, and custom/non-nested FMG
  through the multi-field transfer (`CustomMGHierarchy._build_multifield`, on this
  branch);
- rotated free-slip and fault contact: needs the rotated driver to carry coupled
  fields (below; today refused by `_reject_rotated_with_coupled_fields`);
- Newton only, plus the automatic cold Picard warm-up step (ruling 2026-10-11,
  `docs/reviews/2026-10/coupled-volume-fields-adversarial-review.md`). An explicit
  `consistent_jacobian = False` with the length scale set is refused with a message,
  not silently changed.

**Everything else that must come along.**
- Checkpoint and restart write and read $\chi$ with the solution.
- Grid sequencing / prolongation carries $\chi$.
- `describe()` / `view()` and the transcript show the length scale, $H$, and the
  shifted yield stress; `uw.capabilities()` lists the option under the yielding
  families.
- Capability guides `plasticity-solvers` and `nonlinear-solver` gain a section.

## The rotated driver (needed in any order)

`rotated_bc._solve_rotated_iterative` hand-builds its Schur split from two index sets,
velocity and pressure, because the rotated operator $R^T J R$ carries no DM field
information. A coupled field is a third DM field in neither set, so the rotated solve
covers only part of the operator. To carry coupled fields:
1. split 0 = velocity plus the coupled fields (as the native grouped split);
2. the rotation $Q$ is the identity on coupled-field rows (walls constrain $u$ only);
3. the split-0 prolongation is the multi-field transfer, with the normal-row zeroing
   on velocity rows only;
4. null space: micromorphic $\chi$ is invariant under a rigid rotation (no change);
   a Cosserat micro-rotation carries $\omega$ = the rotation rate.

Then the guard goes, with a regression test: the same problem with a rotated wall and
with a Dirichlet (or Nitsche) wall.

## Decisions for the maintainer

1. **Names.** `plastic_length_scale` and `plastic_gradient_modulus`, alongside
   `plastic_rate_strengthening`? Or one object,
   `cm.Parameters.plastic_regularisation = Micromorphic(length, modulus)`, which would
   also take Cosserat later?
2. **Default $H$.** The campaign uses $H = 0.1\,\eta_{bg}$ ($\beta = 0.1$, the
   recommendation still open). A default of $0.1 \times$ `shear_viscosity_0`, or no
   default (the length scale alone refused)?
3. **Region mask $m$.** The campaign confines the source to the strong layer, so the
   weak layer's large strain rate does not smear into $\chi$ across the interface. In
   a `MultiMaterialConstitutiveModel` the material's own indicator is the natural $m$;
   on a single model, 1 everywhere, with an optional override?
4. **Order.** Rotated driver first (useful on its own for the fault models), then the
   model parameters; both on this branch before its PR.

## Tests

- The first-class option reproduces the hand-built campaign formulation to round-off
  (2-D notch, small mesh), serial and np 2 / 4.
- Rotated wall against Dirichlet wall with a coupled field.
- Restart: solve, checkpoint, restore, re-solve converges at iteration 0.
- Each yielding family (VP, VEP, TI-VEP) gives the shift; `length = 0` is
  bit-identical to the model without it.

Underworld development team with AI support from Claude Code
