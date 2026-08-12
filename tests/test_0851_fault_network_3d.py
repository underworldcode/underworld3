"""The 3-D fault-network path: planar-patch preparation, multi-patch
embed, sequential split, tube glue, end-to-end solve.

Geometry oracles are analytic: the vertical plane y=0.5 and the tilted
plane x+y=0.92 intersect along x=0.42, y=0.5, with the z-span set by
the polygon overlap.
"""
import numpy as np
import pytest

import underworld3 as uw
from underworld3.meshing.fault_network_3d import (crossing_segment,
                                                  prepare_fault_surfaces)

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]

P_A = np.array([[0.30, 0.50, 0.30], [0.70, 0.50, 0.30],
                [0.70, 0.50, 0.70], [0.30, 0.50, 0.70]])
P_B = np.array([[0.30, 0.62, 0.32], [0.62, 0.30, 0.32],
                [0.62, 0.30, 0.68], [0.30, 0.62, 0.68]])
P_FAR = np.array([[0.72, 0.60, 0.30], [0.90, 0.42, 0.30],
                  [0.90, 0.42, 0.62], [0.72, 0.60, 0.62]])


def _surfaces(*pairs):
    out = []
    for name, pts in pairs:
        fs = uw.meshing.FaultSurface(name, pts)
        fs.triangulate()
        out.append(fs)
    return out


def test_crossing_segment_analytic():
    seg = crossing_segment(P_A, P_B)
    assert seg is not None
    P0, P1 = seg
    for P in (P0, P1):
        assert abs(P[0] - 0.42) < 1e-9 and abs(P[1] - 0.5) < 1e-9
    zs = sorted([P0[2], P1[2]])
    assert abs(zs[0] - 0.32) < 1e-9 and abs(zs[1] - 0.68) < 1e-9

    assert crossing_segment(P_A, P_FAR) is None          # line misses
    assert crossing_segment(P_A, P_A + [0, 0.1, 0]) is None  # parallel


def test_preparer_trims_junior_and_records_segment():
    fsA, fsB = _surfaces(("Main", P_A), ("Cross", P_B))
    prepared, report, junctions = prepare_fault_surfaces(
        [fsA, fsB], spacing=0.03, ligament=1.5,
        hierarchy=["Main", "Cross"], verbose=False)
    names = [n for n, _ in prepared]
    assert names.count("Main") == 1                      # senior intact
    assert sum(n.startswith("Cross") for n in names) == 2
    assert len(junctions) == 1
    P0, P1 = junctions[0]["segment"]
    assert abs(P0[0] - 0.42) < 1e-9 and abs(P1[0] - 0.42) < 1e-9
    # every prepared piece keeps clearance from the junction line: the
    # trimmed rims must not come within the band of x=0.42 in-plane
    for n, poly in prepared:
        if n.startswith("Cross"):
            d = np.abs(poly[:, 0] + poly[:, 1] - 0.92) / np.sqrt(2)
            assert (np.abs(poly[:, :2] @ [1, 1] - 0.92) < 1e-9).all()
            # in-plane distance from the junction line
            dline = np.abs(np.sqrt((poly[:, 0] - 0.42) ** 2
                                   + (poly[:, 1] - 0.5) ** 2))
            assert dline.min() > 0.045 - 1e-6            # >= ligament


def test_preparer_refusals():
    fsA, fsB = _surfaces(("Main", P_A), ("Near", P_FAR))
    # P_FAR does not cross P_A and sits farther than a tiny ligament —
    # fine; but with a HUGE ligament it becomes a near-miss -> refused
    with pytest.raises(NotImplementedError, match="without crossing"):
        prepare_fault_surfaces([fsA, fsB], spacing=0.10, ligament=2.0,
                               hierarchy=["Main", "Near"], verbose=False)
    # non-convex rim refused
    bowtie = np.array([[0.3, 0.5, 0.3], [0.7, 0.5, 0.7],
                       [0.7, 0.5, 0.3], [0.3, 0.5, 0.7]])
    with pytest.raises(NotImplementedError, match="non-convex"):
        prepare_fault_surfaces([("Bow", bowtie)], spacing=0.03,
                               verbose=False)


def test_embed_refuses_crossing_patches():
    fsA, fsB = _surfaces(("Main", P_A), ("Cross", P_B))
    with pytest.raises(ValueError, match="cross"):
        uw.meshing.BoxInternalPatch(cellSize=0.15,
                                    patch_points=[fsA, fsB])


def test_network_3d_end_to_end():
    fsA, fsB = _surfaces(("Main", P_A), ("Cross", P_B))
    net = uw.meshing.FaultNetwork([fsA, fsB],
                                  hierarchy=["Main", "Cross"])
    # ligament 1 element (the measured floor): at h=0.08 the default 2
    # would swallow these small test patches via the min-area gate
    mesh = net.prepare(h=0.08, ligament=1.0, verbose=False).build(
        h_far=0.24)
    assert mesh.dim == 3
    bnames = [b.name for b in mesh.boundaries]
    for n, _p in net.prepared:
        assert f"{n}Plus" in bnames and f"{n}Minus" in bnames

    x, y, z = mesh.X
    v = uw.discretisation.MeshVariable("v3N", mesh, 3, degree=2)
    p = uw.discretisation.MeshVariable("p3N", mesh, 1, degree=0,
                                       continuous=False)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = \
        uw.constitutive_models.ViscoPlasticFlowModel
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
    assert info["converged"]
    slips = net.slips(stokes)
    assert all(np.isfinite(s) for s in slips.values())
    # the drive-aligned senior dominates
    assert slips["Main"] > 0.05
    assert slips["Main"] > 3 * max(
        s for n, s in slips.items() if n != "Main")
