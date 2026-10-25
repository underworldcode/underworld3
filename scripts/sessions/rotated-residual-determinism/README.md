# Does a rotated solve's starting residual depend on solver history? (2026-10-25)

Kaiju's restart tests: a rotated free-slip fault solve, given an initial guess
identical to 6e-16 and coefficient fields identical to 1e-11, started from
|F0| = 3646.2 on a solver with 10 prior solves and 3644.1 on a fresh one (6e-4).

`probe_rotated_residual_state.py`: free-slip annulus (rotation null mode),
power-law viscosity, consistent_jacobian="continuation". Solver A solves ten
times with a slowly varying forcing; solver B is fresh and solves once. Both
are then set to one snapshot of (u, p) and the same forcing, 50% above A's
last, and solve once; the starting residual each reports is compared, then the
cross-solve cache (`_rotated_linear_cache`) and the continuation alpha are
cleared on both and the comparison repeated.

| np | state | A | B | rel diff |
|---|---|---|---|---|
| 2 | as is | 4.2202104319e-02 | 4.2202104319e-02 | 0 |
| 2 | cache cleared | 4.2202104319e-02 | 4.2202104319e-02 | 0 |
| 2 | alpha reset | 4.2202104319e-02 | 4.2202104319e-02 | 0 |
| 4 | as is | 4.2202104319e-02 | 4.2202104319e-02 | 0 |
| 4 | cache cleared | 4.2202104319e-02 | 4.2202104319e-02 | 0 |
| 4 | alpha reset | 4.2202104319e-02 | 4.2202104319e-02 | 0 |

On the rotated free-slip path the starting residual carries no history: not
the cached operator, the KSP context and its null space, the gauge projection,
nor the continuation state. The 6e-4 is therefore in fault-specific state.

What the fault path carries between solves:
- The interface assembler (`fault_contact._InterfaceAssembler`) is rebuilt
  every solve; its reaction-fed normal stress starts at zero and is filled
  from this solve's reaction before the first residual. Not carried.
- The rate-and-state state theta is carried: `solver._fault_state`, advanced
  only by `update_fault_state` between steps, read into the assembler at each
  solve. A solver that has stepped holds evolved theta; a fresh one holds
  theta0. Nothing in `checkpoint/` saves `_fault_state`, so a restart starts
  from theta0. If kaiju's model uses a rate-and-state law, that is the 6e-4
  and it is a restart defect (fault state not checkpointed), not a solver one.
- Not covered here: plastic yield histories, the Stokes stress history (DDt)
  and any per-solve JIT constant; kaiju reports those fields identical.

Next step, if the model is rate-and-state: compare `solver._fault_state` on
the two solvers in kaiju's test, and add `_fault_state` to the snapshot.
