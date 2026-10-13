"""The 3-D fault network in parallel: prepare, embed, split ALL patches,
contact solve at np=2 — the serial answer, on a distributed mesh.

Sequential split_fault calls used to refuse multi-fault networks in
parallel (a prior pairing cannot migrate through the redistribution each
split performed). split_faults redistributes ONCE, keyed on the union of
the network's facets, and every split then runs with serial topology.
Run with:
    mpirun -np 2 python -m pytest tests/parallel/\
ptest_0851_fault_network_3d_parallel.py --with-mpi
"""
import numpy as np
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.parallel_safe, pytest.mark.level_2,
              pytest.mark.tier_b]

P_A = np.array([[0.30, 0.50, 0.30], [0.70, 0.50, 0.30],
                [0.70, 0.50, 0.70], [0.30, 0.50, 0.70]])
P_B = np.array([[0.30, 0.62, 0.32], [0.62, 0.30, 0.32],
                [0.62, 0.30, 0.68], [0.30, 0.62, 0.68]])
# the serial run of this exact case (test_0851's fixture, h=0.08):
# peak tangential pair slip per prepared piece
SERIAL = {"Main": 0.1517, "Cross_1": 0.0142, "Cross_2": 0.0064}


def test_network_3d_split_solve_np2():
    fsA = uw.meshing.FaultSurface("Main", P_A)
    fsA.triangulate()
    fsB = uw.meshing.FaultSurface("Cross", P_B)
    fsB.triangulate()
    net = uw.meshing.FaultNetwork([fsA, fsB], hierarchy=["Main", "Cross"])
    mesh = net.prepare(h=0.08, ligament=1.0, verbose=False).build(
        h_far=0.24)

    # the mesh is DISTRIBUTED (the far field balanced; only the fault
    # star is gathered), not serial-on-one-rank
    comm = mesh.dm.comm.tompi4py()
    local = int(mesh.dm.getHeightStratum(0)[1])
    assert comm.allreduce(local, op=min) > 0, "a rank holds no cells"

    x, y, z = mesh.X
    v = uw.discretisation.MeshVariable("v3P", mesh, 3, degree=2)
    p = uw.discretisation.MeshVariable("p3P", mesh, 1, degree=0,
                                       continuous=False)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscoPlasticFlowModel
    cm = stokes.constitutive_model
    cm.yield_mode = "min"
    cm.Parameters.shear_viscosity_0 = 1.0
    cm.Parameters.yield_stress = net.damage_yield(v, dial=0.05)
    stokes.consistent_jacobian = True
    stokes.bodyforce = [0.0, 0.0, 0.0]
    for wall in ("Bottom", "Top", "Left", "Right", "Front", "Back"):
        stokes.add_dirichlet_bc((y - 0.5, 0.0, 0.0), wall)
    net.apply_contact(stokes)
    stokes.petsc_use_pressure_nullspace = True
    stokes.tolerance = 1e-5
    info = net.solve(stokes)
    assert info.get("converged")

    slips = net.slips(stokes)                    # rank-local pairs
    for name, expected in SERIAL.items():
        peak = comm.allreduce(float(slips.get(name, 0.0)), op=max)
        assert peak == pytest.approx(expected, rel=2e-2), (
            f"{name}: parallel peak {peak:.4f} vs serial {expected}")
