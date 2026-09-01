"""The finite-width 3-D fault network in parallel: build the band,
split every patch, contact solve at np=2 — the serial answer, on a
distributed mesh.

The band build gathers only for the rank-0 CAD assembly
(place_thin_volume) and the network split redistributes ONCE keyed on
the union of the network's facets (split_faults); everything else runs
distributed. Run with:
    mpirun -np 2 python -m pytest tests/parallel/\
ptest_0863_fault_network_3d_width_parallel.py --with-mpi
"""
import numpy as np
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.parallel_safe, pytest.mark.level_2,
              pytest.mark.tier_b]

H = 0.08
WIDTH = 0.04
P_A = np.array([[0.30, 0.50, 0.30], [0.70, 0.50, 0.30],
                [0.70, 0.50, 0.70], [0.30, 0.50, 0.70]])
P_B = np.array([[0.30, 0.62, 0.32], [0.62, 0.30, 0.32],
                [0.62, 0.30, 0.68], [0.30, 0.62, 0.68]])
# the serial run of this exact case (test_0863's end-to-end fixture):
# peak tangential pair slip per prepared piece
SERIAL = {"Main": 0.17567, "Cross_1": 0.00335, "Cross_2": 0.00295}


def test_network_3d_width_split_solve_np2():
    fsA = uw.meshing.FaultSurface("Main", P_A)
    fsA.triangulate()
    fsB = uw.meshing.FaultSurface("Cross", P_B)
    fsB.triangulate()
    net = uw.meshing.FaultNetwork([fsA, fsB],
                                  hierarchy=["Main", "Cross"])
    net.prepare(h=H, ligament=1.0, verbose=False)
    net.build(width=WIDTH, realisation="split", h_far=0.24,
              margin_rings=0.5)
    mesh = net.mesh

    # the mesh is DISTRIBUTED (the far field balanced; only the CAD
    # assembly is gathered), not serial-on-one-rank
    comm = mesh.dm.comm.tompi4py()
    local = int(mesh.dm.getHeightStratum(0)[1])
    assert comm.allreduce(local, op=min) > 0, "a rank holds no cells"

    x, y, z = mesh.X
    v = uw.discretisation.MeshVariable("v3W", mesh, 3, degree=2)
    p = uw.discretisation.MeshVariable("p3W", mesh, 1, degree=0,
                                       continuous=False)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.bodyforce = [0.0, 0.0, 0.0]
    for wall in ("Bottom", "Top", "Left", "Right", "Front", "Back"):
        stokes.add_dirichlet_bc((y - 0.5, 0.0, 0.0), wall)
    net.apply(stokes)
    stokes.petsc_use_pressure_nullspace = True
    stokes.tolerance = 1e-5
    info = net.solve(stokes)
    assert info.get("converged")

    slips = net.slips(stokes)                    # rank-local pairs
    for name, expected in SERIAL.items():
        peak = comm.allreduce(float(slips.get(name, 0.0)), op=max)
        assert peak == pytest.approx(expected, rel=2e-2), (
            f"{name}: parallel peak {peak:.4f} vs serial {expected}")
