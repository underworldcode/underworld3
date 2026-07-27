"""Parallel (MPI) test: the nonlinear warm-start / yield-homotopy layers at np > 1.

Charter §11 — no feature is complete if it only works in serial. All four layers of
``docs/developer/design/nonlinear-solver-homotopy-warmstart.md`` make decisions that
would deadlock or diverge between ranks if any of them were driven by a rank-LOCAL
quantity:

  * Layer 1a/1b — ``has_solution`` and the tri-state ``zero_init_guess`` gate the
    ``dm.localToGlobal`` / ``snes.solve`` collectives. If ranks disagreed on
    cold-vs-warm they would take different branches: a hang, not a wrong answer.
  * Layer 2 — every branch of the δ-march (converged? retry? settle?) is taken from
    ``snes.getConvergedReason()`` / ``getIterationNumber()``, which are collective and
    therefore rank-identical. The revert also writes fields inside
    ``synchronised_array_update`` on every rank.
  * Layer 3 — the FMG velocity smoother (gmres + sor, fixed-cost V-cycle) has to work
    with the redundant-LU coarse solve at np > 1, which the serial option-string test
    in ``test_1014`` cannot exercise.

Run:

    cd tests/parallel
    mpirun -np 2 python ./ptest_0202_warmstart_homotopy_parallel.py

Asserts (rank-collectively; any rank disagreeing shows up as a hang or a mismatch):
  1. ``has_solution`` is False before, True after, and identical on every rank.
  2. The auto-detected cold/warm resolution is identical on every rank.
  3. A geometric-FMG Stokes solve converges with the new default smoother.
  4. ``solve(homotopy=True)`` marches and converges, with the same settled δ and step
     count on every rank.
"""

import sympy

import underworld3 as uw
from underworld3 import mpi

comm = uw.mpi.comm
rank = uw.mpi.rank


def _all_ranks_agree(value):
    """True when `value` is identical on every rank."""
    gathered = comm.allgather(value)
    return all(v == gathered[0] for v in gathered)


# --- 1/2/3: warm-start bookkeeping and the FMG smoother, on a refined (hierarchy) mesh
mesh = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.25, refinement=1, qdegree=3
)
v = uw.discretisation.MeshVariable("Vp", mesh, mesh.dim, degree=2)
p = uw.discretisation.MeshVariable("Pp", mesh, 1, degree=1, continuous=True)

stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
stokes.bodyforce = sympy.Matrix([0.0, -1.0])
stokes.add_essential_bc((0.0, 0.0), "Bottom")
stokes.add_essential_bc((0.0, None), "Left")
stokes.add_essential_bc((0.0, None), "Right")
stokes.preconditioner = "fmg"          # exercise the new gmres/sor smoother in parallel
stokes.tolerance = 1.0e-6

assert stokes.has_solution is False
assert _all_ranks_agree(stokes.has_solution), "has_solution disagrees between ranks"
assert _all_ranks_agree(stokes._resolve_zero_init_guess(None)), \
    "the cold/warm decision disagrees between ranks — the solve collectives would split"

stokes.solve()
assert stokes.snes.getConvergedReason() > 0, "FMG Stokes solve failed at np>1"
assert stokes.has_solution is True
assert _all_ranks_agree(stokes.has_solution)
# With a solution in hand every rank must now independently decide "warm".
assert stokes._resolve_zero_init_guess(None) is False
assert _all_ranks_agree(stokes._resolve_zero_init_guess(None))

n_levels = len(getattr(mesh, "dm_hierarchy", []) or [])

# --- 4: the yield homotopy march in parallel
mesh2 = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.3
)
x, y = mesh2.X
v2 = uw.discretisation.MeshVariable("Vh", mesh2, mesh2.dim, degree=2)
p2 = uw.discretisation.MeshVariable("Ph", mesh2, 1, degree=1, continuous=True)
vp = uw.systems.Stokes(mesh2, velocityField=v2, pressureField=p2)
vp.constitutive_model = uw.constitutive_models.ViscoPlasticFlowModel
vp.constitutive_model.Parameters.shear_viscosity_0 = 1.0
vp.constitutive_model.Parameters.yield_stress = 0.35     # genuinely yielding
vp.bodyforce = sympy.Matrix([[0.0, -2.0 * sympy.cos(sympy.pi * x)]])
vp.add_essential_bc((sympy.oo, 0.0), "Top")
vp.add_essential_bc((sympy.oo, 0.0), "Bottom")
vp.add_essential_bc((0.0, sympy.oo), "Left")
vp.add_essential_bc((0.0, sympy.oo), "Right")
vp.petsc_use_pressure_nullspace = True
vp.tolerance = 1.0e-8

report = vp.solve(homotopy=True,
                  homotopy_options=dict(delta0=1.0, dmin=1.0e-3, verbose=False))

assert report["converged"] is True, f"homotopy march failed at np>1: {report}"
assert _all_ranks_agree(report["steps"]), "march step count disagrees between ranks"
assert _all_ranks_agree(report["settled_delta"]), "settled delta disagrees between ranks"
assert vp.has_solution is True
assert _all_ranks_agree(vp.has_solution)

uw.pprint(
    f"ptest_0202 OK (np={uw.mpi.size}): FMG hierarchy {n_levels} levels converged with the "
    f"gmres/sor smoother; homotopy settled delta={report['settled_delta']:.3e} in "
    f"{report['steps']} steps, rank-consistent"
)
