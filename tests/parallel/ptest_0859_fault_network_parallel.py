"""The fault network in parallel: build, junction glue, contact solve at
np=2 — the serial answer, and geometric FMG with NO preconditioner
fallback.

The placed mesh is gathered onto its surgery rank while the multigrid
tail stays load-balanced; unless the tail is co-located with the finest
level (custom_mg.adopt_hierarchy), the transfer pairs a fine node with a
coarse cell on another rank, coarse DOFs lose every fine image and the
build degrades to the local-RBF rescue (measured on the S-fault rig:
488 orphan coarse DOFs on one level at np=2). This test is the guard.
Run with:
    mpirun -np 2 python -m pytest tests/parallel/\
ptest_0859_fault_network_parallel.py --with-mpi
"""
import numpy as np
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.parallel_safe, pytest.mark.level_2,
              pytest.mark.tier_b]

H = 0.03
WIDTH = 0.04
# the serial answer of the same network (tests/test_0859, glued): the
# peak tangential jump per piece, read on every rank after an all-gather
SERIAL = {"Main": 0.4373, "Cont": 0.3603, "Splay": 0.1044}


def _pieces():
    main = np.column_stack([np.linspace(0.25, 0.50, 12), np.full(12, 0.5)])
    cont = np.column_stack([np.linspace(0.55, 0.75, 9), np.full(9, 0.5)])
    s = np.linspace(0.0, 1.0, 8)
    splay = np.column_stack([0.38 + 0.12 * s, 0.5 + 0.18 * s])
    return [("Main", main), ("Cont", cont), ("Splay", splay)]


def test_network_glue_solve_np2():
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=8 * H,
        regular=False, refinement=1, qdegree=2)
    pieces = _pieces()
    net = uw.meshing.FaultNetwork(pieces, hierarchy=[n for n, _p in pieces])
    net.prepare(h=H, ligament=1.0, verbose=False)
    net.build(base=base, width=WIDTH, realisation="split", max_levels=1)

    mesh = net.mesh
    x, y = mesh.X
    v = uw.discretisation.MeshVariable("U", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1,
                                       continuous=True)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = \
        net.junction_patch(eta_0=1.0)
    for wall in ("Bottom", "Top", "Left", "Right"):
        stokes.add_dirichlet_bc((2.0 * (y - 0.5), 0.0), wall)
    stokes.petsc_use_pressure_nullspace = True
    stokes.tolerance = 1e-5
    net.apply(stokes)
    info = net.solve(stokes)
    assert info.get("converged"), "the contact solve did not converge"

    # geometric FMG survived the partition: nothing was swapped for a
    # rescue builder or the default preconditioner
    fallbacks = getattr(stokes, "pc_fallbacks", {}) or {}
    assert not fallbacks, f"preconditioner fallback recorded: {fallbacks}"

    # the pairs are rank-local; the peak per piece is a global max
    local = net.slips(stokes)
    comm = mesh.dm.comm.tompi4py()
    for name, expected in SERIAL.items():
        peak = comm.allreduce(float(local.get(name, 0.0)), op=max)
        assert peak == pytest.approx(expected, rel=2e-2), (
            f"{name}: parallel peak slip {peak:.4f} vs serial {expected}")
