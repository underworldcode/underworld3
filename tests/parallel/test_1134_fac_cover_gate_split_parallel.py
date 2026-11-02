"""A split finest level takes its FAC patch however large it is (#918).

The finest level's FAC patch is the rows its transfer from the coarse level
cannot represent, plus a one-cell halo. The cover gate declines a patch whose
subdomain (patch plus halo) exceeds 75 % of the level's rows, so that a uniform
refinement keeps whole-level smoothing, and it said nothing when it did. A split
fault level crosses 75 % by size alone: on UWTN 2026-022's linear fixture at
h_b/S = 48 and 97 the patch was declined, and the velocity solve took 909 and
1487 iterations against 66 with the patch. A split's structural rows are rows
the coarse level cannot represent, and whole-level SOR does not converge them,
so a split level now takes its patch whatever its size. The verdict is recorded
on ``hierarchy.fac_gate``.

The fixture is the same linear fault on a base of 1/8 refined once (h_b = 1/16),
at 160 rungs (h_b/S = 24): the finest level's patch and halo are 85 % of its
rows, measured on macOS (9470 of 11120). The placed fault's fill depends on the
host by a few rows (test_1106), so the test asserts the fraction against the
old gate, not the counts. The verdict comes from the global counts, so it is
the same at any rank count.

This file carries no ``mpi(min_size=2)`` mark: at np 1 it checks the serial
branch, in the parallel batches the parallel one.

    python -m pytest tests/parallel/test_1134_fac_cover_gate_split_parallel.py
    mpirun -n 2 python -m pytest --with-mpi tests/parallel/test_1134_fac_cover_gate_split_parallel.py
"""
import json

import numpy as np
import pytest

import underworld3 as uw
from underworld3.utilities.custom_mg import set_custom_fmg
from underworld3.utilities.place_surface import place_fault_ribbon_2d

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]

A, B = np.array([0.30, 0.45]), np.array([0.70, 0.55])
N_RUNGS = 160
OLD_GATE = 0.75


def test_a_split_level_takes_its_patch_over_the_old_gate():
    length = float(np.linalg.norm(B - A))
    spacing = length / N_RUNGS
    trace = A + (np.arange(N_RUNGS + 1) * spacing)[:, None] * (B - A) / length
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0), maxCoords=(1, 1), cellSize=1 / 8, regular=True,
        qdegree=2, refinement=1)
    levels = base._coarse_level_meshes()[:-1] + [base]
    mesh = place_fault_ribbon_2d(base, [("Fault", trace)], 2 * spacing, margin_rings=1,
                                 clearance=0.7, split=True, mesher="ladder")[0]
    x, y = mesh.X
    v = uw.discretisation.MeshVariable("v", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("p", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    for wall in ("Bottom", "Top", "Left", "Right"):
        stokes.add_dirichlet_bc((y - 0.5, 0.0), wall)
    stokes.add_fault_bc(0, boundary="Fault")
    stokes.petsc_use_pressure_nullspace = True
    stokes.tolerance = 1e-6
    set_custom_fmg(stokes, levels, field_id=0)
    stokes.solve()
    assert stokes._rotated_freeslip_info["converged"]

    h = stokes._custom_mg["hierarchy"]
    installed = uw.mpi.comm.allreduce(h.level_patch_rows[-1] is not None)
    assert installed == uw.mpi.size, (
        f"np {uw.mpi.size}: the split level's patch was declined (#918)")
    gate = h.fac_gate[-1]
    assert gate is not None and gate["taken"], gate
    assert gate["subdomain_rows"] > OLD_GATE * gate["rows"], (
        f"{gate}: the fixture no longer exceeds the old 75 % gate, so the test "
        f"cannot fail")
    text = json.dumps(gate, sort_keys=True)
    assert len(set(uw.mpi.comm.allgather(text))) == 1, f"rank {uw.mpi.rank}: {gate}"
