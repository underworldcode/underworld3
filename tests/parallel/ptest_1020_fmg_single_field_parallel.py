"""Parallel (MPI) test: the single-field explicit-fmg unlock (#478) at np>1.

A distribute-then-refine hierarchy is co-partitioned (refine() never moves
points across the decomposition), so the requested-native custom-P transfers
are rank-local and the route must work unchanged in parallel:

  * explicit ``preconditioner="fmg"`` on a scalar solver -> LIVE pc_type "mg"
    over every hierarchy level, converged;
  * ``"auto"`` keeps GAMG with the decline recorded (route parity with serial).

Run:

    cd tests/parallel
    mpirun -np 2 python ./ptest_1020_fmg_single_field_parallel.py
"""

import underworld3 as uw

rank = uw.mpi.rank

mesh = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
    cellSize=0.2, refinement=2, qdegree=2,
)
assert len(mesh.dm_hierarchy) == 3


def poisson_on(name):
    T = uw.discretisation.MeshVariable(name, mesh, 1, degree=2)
    p = uw.systems.Poisson(mesh, T)
    p.constitutive_model = uw.constitutive_models.DiffusionModel
    p.constitutive_model.Parameters.diffusivity = 1.0
    p.add_dirichlet_bc(0.0, "Bottom")
    p.add_dirichlet_bc(1.0, "Top")
    return p


# --- explicit fmg: custom-P geometric MG on the live PC ---------------------
p_fmg = poisson_on("Tfmg")
p_fmg.preconditioner = "fmg"
p_fmg.solve()
assert p_fmg.snes.getConvergedReason() > 0, "fmg-routed solve did not converge"
pc = p_fmg.snes.getKSP().getPC()
assert pc.getType() == "mg", f"expected live pc 'mg', got {pc.getType()!r}"
assert pc.getMGLevels() == len(mesh.dm_hierarchy)
assert "custom_mg.build" not in p_fmg.pc_fallbacks, "the transfer build degraded"

# --- auto: GAMG + the recorded decline (route parity with serial) -----------
p_auto = poisson_on("Tauto")
p_auto.solve()
assert p_auto.snes.getConvergedReason() > 0
assert p_auto.snes.getKSP().getPC().getType() == "gamg"
rec = p_auto.pc_fallbacks["single_field_gate"]
assert rec["reason"] == "declined" and rec["installed"] == "gamg"

uw.pprint("ptest_1020_fmg_single_field_parallel: PASS")
