"""FaultNetwork: the user-facing network toolkit.

Hierarchy: at an X crossing the SENIOR trace runs through and only the
junior is cut (pairwise — reversing the hierarchy flips who is
severed). End to end: prepare -> build -> contact -> damage glue ->
solve transmits slip on every piece under pure Newton from cold.
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]

MAIN = np.array([[0.2, 0.45], [0.8, 0.55]])
CROSS = np.array([[0.55, 0.25], [0.45, 0.75]])


def test_hierarchy_decides_who_is_severed():
    net = uw.meshing.FaultNetwork(
        [("Main", MAIN), ("Cross", CROSS)],
        hierarchy=["Main", "Cross"])
    net.prepare(h=0.02, verbose=False)
    names = [n for n, _ in net.prepared]
    assert names.count("Main") == 1          # senior: continuous
    assert sum(n.startswith("Cross_") for n in names) == 2

    rev = uw.meshing.FaultNetwork(
        [("Main", MAIN), ("Cross", CROSS)],
        hierarchy=["Cross", "Main"])
    rev.prepare(h=0.02, verbose=False)
    names = [n for n, _ in rev.prepared]
    assert names.count("Cross") == 1          # flipped seniority
    assert sum(n.startswith("Main_") for n in names) == 2


def test_hierarchy_names_validated():
    with pytest.raises(ValueError, match="hierarchy names"):
        uw.meshing.FaultNetwork([("Main", MAIN)], hierarchy=["Nope"])
    with pytest.raises(ValueError, match="duplicate"):
        uw.meshing.FaultNetwork([("A", MAIN), ("A", CROSS)])


def test_network_end_to_end_transmits_slip():
    h = 0.02
    net = uw.meshing.FaultNetwork(
        [("Main", MAIN), ("Cross", CROSS)],
        hierarchy=["Main", "Cross"])
    mesh = net.prepare(h=h, verbose=False).build(max_levels=1)

    x, y = mesh.X
    v = uw.discretisation.MeshVariable("vNT", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("pNT", mesh, 1, degree=0,
                                       continuous=False)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = \
        uw.constitutive_models.ViscoPlasticFlowModel
    cm = stokes.constitutive_model
    cm.yield_mode = "min"
    cm.Parameters.shear_viscosity_0 = 1.0
    cm.Parameters.yield_stress = net.damage_yield(v, dial=0.05)
    stokes.consistent_jacobian = True
    stokes.bodyforce = [0.0, 0.0]
    trend = float(np.degrees(np.arctan2(MAIN[1][1] - MAIN[0][1],
                                        MAIN[1][0] - MAIN[0][0])))
    t = np.radians(trend)
    tx, ty = np.cos(t), np.sin(t)
    xr = (x - 0.5) * tx + (y - 0.5) * ty
    yr = -(x - 0.5) * ty + (y - 0.5) * tx
    drive = (yr * tx, yr * ty)               # simple shear along Main
    del xr
    for wall in ("Bottom", "Top", "Left", "Right"):
        stokes.add_dirichlet_bc(drive, wall)
    net.apply_contact(stokes)
    stokes.petsc_use_pressure_nullspace = True
    stokes.tolerance = 1e-5

    info = net.solve(stokes)
    assert info["converged"]

    slips = net.slips(stokes)
    assert set(slips) == {"Main", "Cross_1", "Cross_2"}
    # the aligned master slips substantially; every piece moves
    assert slips["Main"] > 0.05
    assert all(s > 0.0 for s in slips.values())
    assert all(np.isfinite(s) for s in slips.values())

    # the glue expression carries a plug per junction (one X crossing)
    expr = net.damage_yield(v, dial=0.05)
    assert isinstance(expr, sympy.Expr)
    assert len(net.junctions) == 1
