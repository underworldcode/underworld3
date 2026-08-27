"""The junction glue is read off the mesh: ribbon minus cut.

A split network is welded wherever one cut stops short of another —
a kissing branch, an abutting pair. The band (the ribbon, with its
extrapolated margins) covers those places; the cut chains do not. The
difference, restricted to where two ribbons meet, is the set of cells
the glue belongs in (:meth:`FaultNetwork.junction_cells`), and a weak
isotropic patch on them (:meth:`FaultNetwork.junction_patch`) lets the
cut carry slip through the joint. Measured on the S-fault rig
(2026-08-27); here on the smallest network that has both joint kinds.

The negative controls: free tips are NOT junctions (damage there would
lengthen the fault rather than join it), and the patch must leave the
strand away from the joints as the split alone had it.
"""
import numpy as np
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b,
              pytest.mark.skipif(uw.mpi.size > 1,
                                 reason="serial suite")]

H = 0.03
WIDTH = 0.04          # two rungs across: a band, not a line of cells


def _pieces():
    """A main strand, a collinear continuation abutting it across a
    gap of one and a half cells (wider than the ligament, so prepare()
    leaves it as it is), and a splay ending on the main (a T, trimmed
    back by prepare() into a kissing junction)."""
    main = np.column_stack([np.linspace(0.25, 0.50, 12),
                            np.full(12, 0.5)])
    cont = np.column_stack([np.linspace(0.55, 0.75, 9),
                            np.full(9, 0.5)])
    s = np.linspace(0.0, 1.0, 8)
    splay = np.column_stack([0.38 + 0.12 * s, 0.5 + 0.18 * s])
    return [("Main", main), ("Cont", cont), ("Splay", splay)]


def _longer():
    """The other end member: the same fault, just LONGER — Main and its
    continuation as one trace, no joint to weld."""
    main, cont, splay = (p for _n, p in _pieces())
    line = np.column_stack([np.linspace(main[0, 0], cont[-1, 0], 22),
                            np.full(22, 0.5)])
    return [("Main", line), ("Splay", splay)]


def _network(pieces=None):
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=8 * H,
        regular=False, refinement=1, qdegree=2)
    pieces = _pieces() if pieces is None else pieces
    net = uw.meshing.FaultNetwork(pieces,
                                  hierarchy=[n for n, _p in pieces])
    net.prepare(h=H, ligament=1.0, verbose=False)
    net.build(base=base, width=WIDTH, realisation="split", max_levels=1)
    return net


def _centroids(net):
    from underworld3.utilities.place_surface import _cell_centroids_of
    band = net.info["band"]
    ids, cen = _cell_centroids_of(net.mesh.dm, np.ones_like(band))
    out = np.zeros((len(band), 2))
    out[ids] = cen
    return out


def test_junction_cells_sit_at_the_joints_and_not_at_the_free_tips():
    net = _network()
    cells = net.junction_cells()
    cen = _centroids(net)

    assert cells.any()
    assert not (cells & ~net.info["band"]).any(), "glue outside the ribbon"

    joints = [np.array([0.525, 0.5]), np.array([0.38, 0.5])]   # gap, T
    near_joint = np.min([np.linalg.norm(cen[cells] - j, axis=1)
                         for j in joints], axis=0)
    assert near_joint.max() < 5 * H, "a junction cell far from any joint"
    for j in joints:
        assert (np.linalg.norm(cen[cells] - j, axis=1) < 3 * H).any(), (
            f"no junction cells at the joint {j}")

    free_tips = [np.array([0.25, 0.5]), np.array([0.75, 0.5]),
                 np.array([0.5, 0.68])]
    for tip in free_tips:
        assert not (np.linalg.norm(cen[cells] - tip, axis=1) < 2 * H).any(), (
            f"a free tip {tip} was treated as a junction")


def test_a_near_miss_is_pulled_back_to_one_ligament_not_three():
    """Two collinear pieces closer than the ligament are an offset
    junction: the join opens to ONE ligament, shared between the two
    ends — not a ligament on each side on top of the gap it had."""
    main = np.column_stack([np.linspace(0.25, 0.50, 12), np.full(12, 0.5)])
    cont = np.column_stack([np.linspace(0.53, 0.75, 9), np.full(9, 0.5)])
    net = uw.meshing.FaultNetwork([("Main", main), ("Cont", cont)])
    net.prepare(h=H, ligament=2.0, verbose=False)
    ends = dict((n, P) for n, P in net.prepared)
    gap = ends["Cont"][0, 0] - ends["Main"][-1, 0]
    assert gap == pytest.approx(2.0 * H, rel=0.15), (
        f"the join opened to {gap:.3f}, not the ligament {2 * H:.3f}")
    assert [j["kind"] for j in net.junctions] == ["near-miss", "near-miss"]


def test_a_collinear_pair_shares_one_spine_and_makes_no_slivers():
    """Two ribbons placed along one line interleave their vertices into
    sliver cells (measured: 7800 cells below 1e-6 in area on the rig).
    The pieces of a collinear abutting pair are placed on ONE spine, cut
    at their own ends, with the gap as spine the split does not cut."""
    net = _network()
    assert [n for n, _S, _i in net.spines] == ["Main+Cont", "Splay"]
    main_cont = net.spines[0]
    assert (main_cont[2] == -1).sum() >= 1, "no gap vertex on the shared spine"

    dm = net.mesh.dm
    cS, _cE = dm.getHeightStratum(0)
    ids = np.flatnonzero(net.info["band"])
    areas = np.array([dm.computeCellGeometryFVM(int(c) + cS)[0] for c in ids])
    assert areas.min() > 1e-3 * np.median(areas), (
        f"sliver cells in the band: min area {areas.min():.2e} against a "
        f"median of {np.median(areas):.2e}")


def test_the_weak_plane_has_no_junction_cells():
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=8 * H,
        regular=False, refinement=1, qdegree=2)
    net = uw.meshing.FaultNetwork(_pieces())
    net.prepare(h=H, ligament=1.0, verbose=False)
    net.build(base=base, width=WIDTH, realisation="ti", max_levels=1)
    with pytest.raises(RuntimeError, match="SPLIT"):
        net.junction_cells()


def _slip_at(solver, name, x0):
    """The cut's tangential jump at the pair nearest ``x = x0``."""
    from underworld3.utilities.fault_contact import fault_pair_jumps
    coords, jumps, normals = fault_pair_jumps(
        solver, name, solver._rotated_freeslip_info)
    k = np.argmin(np.abs(coords[:, 0] - x0))
    jn = float(jumps[k] @ normals[k])
    return float(np.linalg.norm(jumps[k] - jn * normals[k]))


def _shear_solve(net, glue, tag):
    mesh = net.mesh
    x, y = mesh.X
    v = uw.discretisation.MeshVariable(f"U{tag}", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable(f"P{tag}", mesh, 1, degree=1,
                                       continuous=True)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = (
        net.junction_patch(eta_0=1.0) if glue else 1.0)
    for wall in ("Bottom", "Top", "Left", "Right"):
        stokes.add_dirichlet_bc((2.0 * (y - 0.5), 0.0), wall)
    stokes.petsc_use_pressure_nullspace = True
    stokes.tolerance = 1e-5
    net.apply(stokes)
    net.solve(stokes)
    return stokes


def test_the_patch_lands_between_the_cut_and_the_longer_fault():
    """The two end members: the fault CUT (abutting pieces, welded by
    the intact bridge) and the fault LONGER (one continuous trace). The
    glue must move the continuation's slip from the first toward the
    second and not beyond it. Read at one station on the continuation,
    under a shear along the fault."""
    X0 = 0.70
    cut = _network()
    welded = _slip_at(_shear_solve(cut, glue=False, tag="w"), "Cont", X0)
    glued = _slip_at(_shear_solve(cut, glue=True, tag="g"), "Cont", X0)
    longer = _slip_at(_shear_solve(_network(_longer()), glue=False,
                                   tag="l"), "Main", X0)

    assert longer > 1.2 * welded, (
        f"no weld to repair: cut {welded:.3f} vs longer {longer:.3f}")
    assert glued > welded + 0.5 * (longer - welded), (
        f"the glue repaired less than half the weld: {welded:.3f} -> "
        f"{glued:.3f} against {longer:.3f}")
    assert glued < 1.1 * longer, (
        f"the glue overshot the longer fault: {glued:.3f} vs {longer:.3f}")
