# Micromorphic plasticity as a first-class constitutive choice

Status: implemented 2026-10-17 on `feature/shear-band-length-scale` (`753e1417` rotated driver,
`fa97582a` the option). Tests: `test_1070` (rotated), `test_1078` (the option).

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

**User side.** One regularisation object on every yielding model (ViscoPlastic,
ViscoElasticPlastic, TransverseIsotropicVEP), next to `plastic_rate_strengthening`
(decision 1 below; the proposal had two loose parameters):

```python
cm = uw.constitutive_models.ViscoPlasticFlowModel
cm.Parameters.yield_stress = C + sin_phi * p
cm.Parameters.plastic_regularisation = uw.constitutive_models.Micromorphic(
    length=uw.quantity(1, "km"))          # modulus default 0.1 x shear_viscosity_0
stokes.solve()
```

Setting the regularisation is the whole switch. Nothing else in the script
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
- rotated free-slip and fault contact: the rotated driver carries coupled fields
  (below; the pair was refused before 753e1417);
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

**Done.** Items 1-3 in `753e1417`; item 4 in `73ca4744`
(`_complete_mode_on_coupled_fields`: each mode's coupled part solves the coupled rows
of the operator). Tests in `test_1070`. Noted on the way: a driven lid meeting a
rotated wall at a corner gives a different solution (~30% of the velocity) from the
same wall as a Dirichlet condition, with or without a coupled field; the two impose
conflicting corner values and resolve them differently. Pre-existing.

## Decisions (maintainer, 2026-10-17)

1. **One regularisation object**, not loose parameters:
   `cm.Parameters.plastic_regularisation = Micromorphic(length=..., modulus=...)`. The
   slot is built to take other regularisations later (Cosserat, and the rate
   strengthening that is a bare parameter today). The user-side example above becomes
   `cm.Parameters.plastic_regularisation = uw.constitutive_models.Micromorphic(
   length=uw.quantity(1, "km"))`.
2. **Default $H = 0.1 \times$ `shear_viscosity_0`** ($\beta = 0.1$) when only the
   length is given.
3. **Region mask: the material indicator.** 1 everywhere on a single model; in a
   `MultiMaterialConstitutiveModel` each yielding material's own indicator. An optional
   override on the object.
4. **Order: the rotated driver first**, then the regularisation object; both on this
   branch before its PR.

## Tests

- The first-class option reproduces the hand-built campaign formulation to round-off
  (2-D notch, small mesh), serial and np 2 / 4.
- Rotated wall against Dirichlet wall with a coupled field.
- Restart: solve, checkpoint, restore, re-solve converges at iteration 0.
- Each yielding family (VP, VEP, TI-VEP) carries the shift in its own flux.
- Without a regularisation the model's residual and Jacobian are unchanged
  (compared bit for bit against development in the adversarial review).
- Replacing or removing the regularisation; a changed mask refreshes the equation.
- The adjoint of a coupled solve is the transpose of the whole operator.
