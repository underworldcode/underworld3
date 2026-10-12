"""The forward stress history in parallel: np >= 2 equals serial.

A Maxwell fluid whose shear modulus varies in x, driven by a body force that
turns over two counter-rotating cells between no-slip walls, on a gmsh
(file-read) mesh. The stress is then non-uniform in both directions, so an
arrival handed across a seam with the wrong value, fitted into the wrong
cell, or dropped shows in the answer; and the flow crosses a seam of either
orientation. The stress at two points after ten steps must be the serial
value, and arrivals must actually have changed rank.
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b, pytest.mark.mpi(min_size=2), pytest.mark.timeout(600)]

# BASELINES: the serial values (see the ledger)
XY_AT_A, XY_AT_B = -0.1190789, 0.0702257
POINTS = np.array([[0.5, 0.2], [-0.3, -0.35]])


def turned_over_maxwell_box(transport="forward", steps=10, dt=0.1):
    Lx, H = 1.0, 0.5
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(-Lx, -H), maxCoords=(Lx, H),
                                             cellSize=0.125, qdegree=3, regular=False)
    x, _y = mesh.X
    v = uw.discretisation.MeshVariable("U_to", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("P_to", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.stress_transport = transport
    stokes.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel(stokes.Unknowns, order=1)
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.constitutive_model.Parameters.shear_modulus = 1.0 + 0.5 * sympy.sin(sympy.pi * x / Lx)
    stokes.constitutive_model.Parameters.dt_elastic = dt
    for wall in ("Top", "Bottom", "Left", "Right"):
        stokes.add_dirichlet_bc((0.0, 0.0), wall)
    stokes.bodyforce = sympy.Matrix([[0.0, 4.0 * sympy.sin(sympy.pi * x / Lx)]])
    stokes.tolerance = 1.0e-8
    relocated = 0
    for _ in range(steps):
        stokes.solve(timestep=dt, zero_init_guess=False)
        relocated += getattr(stokes.DFDt, "_n_relocated", 0)
    values = np.asarray(uw.function.global_evaluate(stokes.DFDt.psi_star[0].sym[0, 1], POINTS)).reshape(-1)
    return type(stokes.DFDt).__name__, values, uw.mpi.comm.allreduce(relocated, op=uw.MPI.SUM)


def test_the_forward_history_gives_the_serial_stress_on_every_rank():
    kind, values, relocated = turned_over_maxwell_box()
    assert kind == "ForwardSemiLagrangian"
    # The serial values are the hard baseline (they pin the physics; regenerate
    # them if a default changes). Parallel matches them to 1e-4, not to
    # round-off: a cell's arrivals are summed into its least-squares fit in an
    # order the partition sets, and ten steps of that reordering reach ~2e-5 on
    # a mildly conditioned fit. A dropped or misrouted arrival would be far larger.
    assert abs(values[0] - XY_AT_A) < 1.0e-4 and abs(values[1] - XY_AT_B) < 1.0e-4, values
    # the seam must actually have been crossed for this to test the parallel path
    assert relocated > 0
