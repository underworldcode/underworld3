# Rotated Free-Slip Linear Solver Reuse

## Scope

This note documents the repeated-solve workspace introduced for linear Stokes
problems using rotated strong free slip. It addresses
[issue #417](https://github.com/underworldcode/underworld3/issues/417).

The change does not alter the rotated boundary condition or its discrete
equations. It changes PETSc object ownership and determines when the assembled
operator and its preconditioner can be reused.

## Previous Lifecycle

Every linear rotated solve previously rebuilt:

1. the nodal rotation matrices `Q` and `Q.T`;
2. the transformed saddle operator `Ahat = Q A Q.T`;
3. the pressure mass block;
4. the coupled and pressure nullspaces;
5. the fieldsplit Schur KSP/PC;
6. the velocity GAMG hierarchy.

The linear/nonlinear detector also assembled and copied two trial Jacobians on
every call. Long transient runs therefore accumulated a large PETSc and system
allocator high-water mark even when only the body-force field changed.

## Reused Workspace

The Stokes solver now owns one `_rotated_linear_cache` containing:

- `Q`, `Q.T`, and constrained normal rows;
- the transformed matrix `Ahat`;
- two deterministic operator-probe vector triples;
- operator coefficient mesh variables and their UW data-version counters;
- the pressure mass block, nullspaces, fieldsplit KSP/PC, and GAMG hierarchy;
- the constraint-row diagonal scale.

For an unchanged linear operator, a subsequent solve:

1. assembles only `F(0)` and forms the new right-hand side;
2. rotates and constrains that right-hand side;
3. solves with the existing KSP/PC hierarchy;
4. rotates the solution back;
5. forms and retains the Cartesian reaction needed by topography recovery.

The prior per-solve solution and reaction vectors are explicitly destroyed
before replacement.

## Invalidation

The cache is destroyed before a solver or DM rebuild. The cached
linear/nonlinear classification is invalidated at the same time.

Mutable operator coefficients are collected from the constitutive parameters,
constraint, penalty, and saddle preconditioner. Their base
`MeshVariable._state` counters decide whether Jacobian assembly is required.

- A body-force field change does not refresh the operator.
- A viscosity field change refreshes `J`, `Jp`, and the rotated operator.
- An explicit `solve(time=...)` refreshes and compares the operator, covering
  expressions that depend on `mesh.t`.
- Mesh, field-layout, boundary-condition, constitutive, or forced setup changes
  destroy the full workspace.

After a coefficient-triggered assembly, two deterministic matrix-vector
products decide whether matrix values actually changed. This avoids exact
`Mat.equal`, which rejects harmless distributed-assembly roundoff, and avoids
copying CSR arrays, which recreated the memory high-water problem.

The high-level `uw.systems.Stokes.solve()` wrapper exposes and forwards
`time=` for this purpose. It remains distinct from the viscoelastic integration
`timestep=`.

## PETSc Ownership

`_destroy_rotated_ksp_ctx` explicitly destroys the KSP, pressure mass matrix,
and owned nullspaces. `_destroy_rotated_linear_cache` additionally destroys the
operator probes, transformed operator, and rotation matrices.

Linear topography no longer requires retaining a copy of the native operator
and right-hand side. The Cartesian reaction `J U - b` is computed immediately
after the solve and stored in the solve result.

## Validation

The focused regression verifies that:

- changing only a temperature/body-force field preserves the `Q`, `Ahat`, and
  KSP handles;
- the velocity scales with the changed right-hand side;
- changing a viscosity mesh variable destroys the old KSP and refreshes the
  operator;
- the refreshed velocity has the expected inverse-viscosity scaling.

Additional serial and eight-rank tests cover spherical velocity/leakage,
reaction-derived topography, boundary traction, and dynamic-topography field
recovery.

Always launch MPI tests with the worktree MPI executable:

```bash
.pixi/envs/amr-dev/bin/mpirun -np 8 \
  .pixi/envs/amr-dev/bin/python <script.py>
```

Using a system `mpirun` from a different Open MPI installation can stall during
initialization and is not a solver failure.
