---
title: "Saddle-point field groups: coupled fields and multipliers in Stokes"
---

# Saddle-point field groups: coupled fields and multipliers in Stokes

Every Stokes solver is one monolithic Newton system over the fields of one DM, and
one two-way Schur fieldsplit. Extra fields do not need extra solvers: each joins one
of the two groups of that split. This page is the reference for how that works, what
conditions each group, and the paths (rotated constraints, custom multigrid) that
build the split by hand.

Read it before adding a field to a Stokes-family solver, changing the Stokes
fieldsplit options, or touching `rotated_bc` or `custom_mg` on a multi-field problem.

## The two groups

`SNES_Stokes_SaddlePt._setup_block_fieldsplit_options`
(`cython/petsc_generic_snes_solvers.pyx`) groups by DM field index:

| Split | Fields | Added by |
|---|---|---|
| 0 | velocity, then every **coupled volume field** | `add_coupled_field` (and a constitutive model's request, e.g. `Micromorphic`) |
| 1 | pressure, then every **multiplier** | `add_constraint_bc` (`SNES_Stokes_Constrained`) |

as `pc_fieldsplit_0_fields = 0,<coupled ids>` and
`pc_fieldsplit_1_fields = 1,<multiplier ids>`. The outer solve is the usual Schur
factorisation with fgmres, the split-0 solve is the velocity multigrid, and the
split-1 solve the Schur complement. Nothing else changes when a group gains a field:

$$
\begin{pmatrix} A & B^T \\ B & C \end{pmatrix},\qquad
A = \begin{pmatrix} A_{uu} & A_{uc} \\ A_{cu} & A_{cc} \end{pmatrix},\qquad
C = \begin{pmatrix} 0 & 0 \\ 0 & C_{\lambda\lambda} \end{pmatrix},
$$

with $c$ the coupled fields; the split-1 rows of the multipliers $\lambda$ hold their
boundary coupling to $\mathbf{u}$, and $C_{\lambda\lambda}$ only their interior screening
mass (below). The field kinds differ in support and terms:

| | Coupled field | Multiplier |
|---|---|---|
| Support | volume (a `MeshVariable` on the mesh) | a boundary; interior DOFs constrained out |
| Its residual | $\int_\Omega F_0\psi + F_1\cdot\nabla\psi$, user/model given | $\int_\Gamma \psi\,(\mathbf{n}\cdot\mathbf{u} - g)$ |
| Into the momentum row | through the flux (the flux may depend on it), or an added `momentum_flux` | $\int_\Gamma (h + r(\mathbf{n}\cdot\mathbf{u}-g))\,\mathbf{n}\cdot\mathbf{v}$ |
| Group | split 0, with velocity | split 1, with pressure |
| Tangent | Newton only (the cross blocks need it) | linear in $h$ |

## Coupled fields (split 0)

`add_coupled_field(field, F0, F1=None, momentum_flux=None)` adds the field to the DM
and its equation to the same residual. The Jacobian blocks against every other field
are derived symbolically from the residuals by `_coupled_jacobian_block`, in PETSc's
pointwise layout ([petsc-jacobian-layout.md](petsc-jacobian-layout.md)); a block that
is identically zero is registered as NULL, not compiled. The blocks follow
`consistent_jacobian` like the Stokes ones. For the micromorphic regularisation
(`Micromorphic`, [design/micromorphic-plasticity.md](../design/micromorphic-plasticity.md)):

| Block | Content | Scale |
|---|---|---|
| $A_{cc}$ | $H(\psi\chi + \ell^2\nabla\psi\cdot\nabla\chi)$, a screened Poisson operator | $H$ |
| $A_{cu}$ | $-H\,\partial\dot\varepsilon_{II}/\partial\nabla\mathbf{u}$ (the source) | $H$ |
| $A_{uc}$ | the yield shift $\tau_y + H(\dot\varepsilon_{II}-\chi)$ through the plastic viscosity | $H$ |
| $B$ against $c$ | zero: the $\chi$ equation does not see the pressure | |

The cross blocks scale with $H$ (default $0.1\,\eta_0$), not with the viscosity
contrast, which is why the coupled split 0 is no harder to precondition than the
velocity block alone: it adds stiffness of order $H$ to the short-wavelength mode (a
collapsing band) that makes the yielded velocity block near-singular. Coupled problems
are Newton only: the frozen tangent drops $A_{uc}$, and the notch campaign measured it
stalling (review record `docs/reviews/2026-10/coupled-volume-fields-adversarial-review.md`).

**Split 0 is not node-blocked.** The velocity block size (`mat_block_size = dim`) is
not mirrored onto split 0 when coupled fields share it: a block would group a coupled
DOF with a velocity component. Mirroring it caused an np = 3 hang (PETSc refused the
block size only on the ranks whose split-0 share was odd).

## Multipliers (split 1)

`add_constraint_bc` (on `SNES_Stokes_Constrained`) adds a scalar multiplier $h$ per
constrained boundary, at the velocity degree by default. The base class registers the
boundary pair: $C^T$ into the momentum row ($h\,\mathbf{n}$), $C$ into the multiplier
row ($\mathbf{n}\cdot\mathbf{u} - g$), and the augmented-Lagrangian stiffness
$r(\mathbf{n}\otimes\mathbf{n})$ on the velocity, with $r$ viscosity-weighted. The
interior multiplier DOFs are constrained out of the section
(`_constrain_interior_multipliers_in_section`), so the solved $h$ block is
boundary-sized, and a small interior screening mass keeps the interior block
non-singular. At convergence $h + r(\mathbf{n}\cdot\mathbf{u}-g)$ is the boundary
normal traction (dynamic topography: `traction()`, `topography()`).

What `SNES_Stokes_Constrained` sets beyond the base, because split 1 is $[p, h]$:

- Schur preconditioning `selfp`, not the base `a11`. Its $hh$ corner is
  $-C\,\mathrm{diag}(A)^{-1}C^T$, the true constraint Schur complement, which the
  pressure mass cannot supply.
- `fgmres` outer Krylov: the grouped preconditioner runs iterative inner solves, so it
  varies between outer iterations (a non-flexible method false-converges on it).
- Unpreconditioned convergence norm, for the same reason.
- The automatic pressure gauge on an enclosed problem (`auto_pressure_gauge`): the
  constant pressure and constant multiplier are gauge-free and would otherwise land on
  partition-dependent levels.

These are what a plain `Stokes` would need to turn on when its first multiplier is
added; the split itself is already the base class's. See
[design/CONSTRAINED_FREESLIP_MULTIPLIER.md](../design/CONSTRAINED_FREESLIP_MULTIPLIER.md).

## Multigrid on split 0

The velocity multigrid acts on all of split 0, smoothing $u$ and the coupled fields
together. Transfers are per field and block-diagonal; the coupling is restored on
each level by the Galerkin coarse operator.

- **Native nested hierarchy** (a refined base mesh): PETSc interpolates each field of
  the split sub-DM with its own element (P2 velocity, P1 $\chi$).
- **Custom transfers** (`set_custom_fmg`, `mesh.adapt` children, non-nested gmsh
  levels): `CustomMGHierarchy._build_multifield` (`utilities/custom_mg.py`) builds one
  transfer per field through per-field sub-hierarchies and assembles them, in the
  split's numbering (`_split_index_maps`: the index sets of
  `dm.createSubDM([0, coupled ids])`), into one CSR matrix. `field_id=0` on a solver
  with coupled fields means the whole of split 0 (`_split_fields`). In parallel the
  non-nested coarse levels must be co-located with the fine partition
  (`_colocate_level`) or coarse DOFs are left with no fine image.

## The rotated path builds the split by hand

Rotated free-slip and fault contact (`utilities/rotated_bc.py`,
[rotated-freeslip.md](rotated-freeslip.md)) solve $R^T J R$, which carries no DM
field information, so `_solve_rotated_iterative` builds its two-way split from
explicit index sets. Split 0 is `_velocity_split_is`: the velocity, plus the coupled
fields when there are any, as the index set of `dm.createSubDM([0, coupled ids])` (the
numbering the multi-field transfer is built in). The rotation $Q$ is the identity on
coupled rows (walls constrain $\mathbf{u}$ only), and the constrained normal rows are
zeroed in the split-0 prolongation, velocity rows only. Each rigid-rotation null mode is completed on the coupled fields from
the operator's own rows, $c = -A_{cc}^{-1}A_{cu}\mathbf{u}_{rot}$
(`_complete_mode_on_coupled_fields`), in both the null space and the gauge removal:
zero for a field a rotation leaves unchanged (the micromorphic $\chi$), the rotation
rate for a Cosserat micro-rotation. A solver cannot carry rotated constraints and
multipliers together (`_reject_mixed_constraint_mechanisms`, #464): they are
alternative ways to impose the same wall-normal condition, and the rotated split has no
place for a multiplier.

## Adding a field: checklist

1. Which group? A volume field with its own equation joins split 0; a boundary
   constraint's multiplier joins split 1.
2. Its Jacobian blocks come from `_coupled_jacobian_block`; check the zero blocks are
   really zero (they are not compiled).
3. Every path that splits by field number must know the group: the native split
   options, the rotated split (`_velocity_split_is`), the custom-MG split
   (`_split_fields`), and the post-solve copy-back (`_ensure_local_field_index_sets`).
4. Do not mirror a node block size onto a mixed group.
5. Test serially and at np 2, 3 and 4 (odd rank counts find block-size and
   partition asymmetries), with the native and a custom hierarchy.

## Open: bugs, not constraints

These are unfinished work (maintainer ruling, 2026-10-17: a refusal of an unfinished
case is a bug to fix, never a constraint to design or advise around). Each is marked
`TODO(BUG)` at its site.

- Rotated constraints refuse an adjoint (`_adjoint_support`). Fix: assemble the
  transpose by kernels (the `by_kernels` route of `adjoint_solve`) and apply the same
  rotation and normal-row constraint.
- Rotated constraints and multipliers are refused together
  (`_reject_mixed_constraint_mechanisms`, #464). Fix: the rotated split takes the
  multipliers into split 1, as it takes coupled fields into split 0.
- A wall-clock guard is refused on the rotated path. Fix: instrument the rotated KSP.
- `adjoint_solve` returns no adjoint field for a coupled field or a multiplier (its
  `rhs` / `target` handle velocity and pressure only), and no test covers either.
- Coupled fields and multipliers on one solver: no test of the combination.
- `add_constraint_bc` on `Stokes` itself (switching on the split-1 defaults with its
  first multiplier), with `SNES_Stokes_Constrained` an alias.
- `add_coupled_field` covers volume terms only; a split-0 field with boundary terms
  has no API.

## Where the code is

| Piece | Where |
|---|---|
| Group options | `SNES_Stokes_SaddlePt._setup_block_fieldsplit_options` |
| Coupled field API, Jacobian blocks | `add_coupled_field`, `_coupled_jacobian_block` (`petsc_generic_snes_solvers.pyx`) |
| A model's request | `Constitutive_Model._coupled_field_requests`, `SNES_Stokes._register_model_coupled_fields` |
| Multiplier API and accessors | `SNES_Stokes_Constrained` (`systems/solvers.py`) |
| Multiplier boundary terms, interior reduction | `_setup_pointwise_functions`, `_constrain_interior_multipliers_in_section` (`.pyx`) |
| Multi-field transfers | `CustomMGHierarchy._build_multifield`, `_split_index_maps`, `_split_fields` (`utilities/custom_mg.py`) |
| Rotated split | `_velocity_split_is`, `_build_rotated_custom_Pl`, `_rotated_nullspace` (`utilities/rotated_bc.py`) |
| Tests | `test_1070` (coupled fields, rotated), `test_1075` (multi-field FMG), `test_1078` (Micromorphic), `test_1063` (constrained free-slip, parallel) |
