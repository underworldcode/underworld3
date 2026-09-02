"""The finite-width 3-D fault network: ONE band, both realisations.

test_0858's property one dimension up: ``build(width=...)`` thickens
the margin-expanded patches into one fused band with each un-expanded
patch embedded as a conforming mid-surface, so the same mesh is cut
and split (``realisation="split"``) or left whole for the volumetric
weak plane (``"ti"``). Geometry oracles are analytic: the patches are
planar, so the honoured-footprint rule and the weak-plane director
(the patch normal) are exact.

The parallel form is ``tests/parallel/ptest_0863``.
"""
import numpy as np
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b,
              pytest.mark.skipif(uw.mpi.size > 1,
                                 reason="serial suite; the parallel "
                                        "form is ptest_0863")]

H = 0.08
WIDTH = 0.04

P_A = np.array([[0.30, 0.50, 0.30], [0.70, 0.50, 0.30],
                [0.70, 0.50, 0.70], [0.30, 0.50, 0.70]])
P_B = np.array([[0.30, 0.62, 0.32], [0.62, 0.30, 0.32],
                [0.62, 0.30, 0.68], [0.30, 0.62, 0.68]])


def _network(realisation, width=WIDTH):
    fsA = uw.meshing.FaultSurface("Main", P_A)
    fsA.triangulate()
    fsB = uw.meshing.FaultSurface("Cross", P_B)
    fsB.triangulate()
    net = uw.meshing.FaultNetwork([fsA, fsB],
                                  hierarchy=["Main", "Cross"])
    net.prepare(h=H, ligament=1.0, verbose=False)
    # margin_rings=0.5: the margin must stay half a cell clear of the
    # ligament=1.0 junction gap (the build refuses a welding margin),
    # and these small test patches in a unit box also need the
    # expanded band to clear the walls for the carve
    net.build(width=width, realisation=realisation, h_far=0.24,
              margin_rings=0.5)
    return net


def test_the_two_realisations_share_one_mesh():
    split = _network("split")
    ti = _network("ti")

    assert split.realisation == "split" and ti.realisation == "ti"
    assert [n for n, _p in split.prepared] == \
        [n for n, _p in ti.prepared]
    assert split.info["n_cells"] == ti.info["n_cells"], (
        "the realisations no longer share a mesh; the comparison "
        "between them is then confounded by the discretisation")
    # the ONLY difference: the split duplicated the cut's nodes
    assert (split.mesh.dm.getDepthStratum(0)[1]
            > ti.mesh.dm.getDepthStratum(0)[1])


def test_the_weak_plane_director_is_the_patch_normal():
    ti = _network("ti")
    eta1, ndir, foot = ti.ti_fields(eta_1=0.01, eta_0=1.0)

    assert foot.sum() > 0, "no footprint cells: nothing would be weak"
    assert foot.sum() <= ti.info["band"].sum()
    vals = eta1.array[:, 0, 0]
    assert np.allclose(vals[foot], 0.01)
    assert np.allclose(vals[~foot], 1.0)

    # planar patches make the director exact: y-hat on Main's plane
    # (y = 0.5), (1, 1, 0)/sqrt(2) on the Cross pieces' (x + y = 0.92)
    d = np.asarray(ndir.array).reshape(-1, 3)
    foots = ti.footprints
    assert foots["Main"].any()
    assert np.allclose(np.abs(d[foots["Main"]] @ [0.0, 1.0, 0.0]), 1.0)
    nB = np.array([1.0, 1.0, 0.0]) / np.sqrt(2)
    for name, m in foots.items():
        if name.startswith("Cross") and m.any():
            assert np.allclose(np.abs(d[m] @ nB), 1.0)


def test_the_band_is_damage_material_in_either_realisation():
    for realisation in ("split", "ti"):
        net = _network(realisation)
        assert net.band.sum() > 0
        assert set(net.footprints) == {n for n, _p in net.prepared}
        assert net.footprints["Main"].sum() > 0
        assert net.footprints["Main"].sum() <= net.band.sum()

        tau = net.band_yield(tau_y=4.0)
        painted = net._band_yield_var.array[:, 0, 0]
        assert np.allclose(painted[net.band], 4.0)
        assert np.allclose(painted[~net.band], 1.0e8)
        assert tau.free_symbols                    # a usable expression

        # the junction glue in 3-D is damage_yield's tubes; the 2-D
        # ribbon rule refuses rather than guessing
        if realisation == "split":
            with pytest.raises(NotImplementedError, match="damage_yield"):
                net.junction_cells()


def test_the_build_refusals_are_loud():
    fsA = uw.meshing.FaultSurface("Main", P_A)
    fsA.triangulate()
    net = uw.meshing.FaultNetwork([fsA])
    net.prepare(h=H, verbose=False)
    with pytest.raises(ValueError, match="width"):
        net.build(realisation="ti")
    with pytest.raises(ValueError, match="realisation"):
        net.build(width=WIDTH, realisation="smeared")
    # the band has ONE mesher; the no-band meshers are not it
    with pytest.raises(ValueError, match="network"):
        net.build(width=WIDTH, mesher="embed")
    with pytest.raises(NotImplementedError, match="base box"):
        net.build(width=WIDTH, base="a mesh")

    # a junction network refuses a margin that welds the junction gap:
    # the gap is physics (the intact ligament), the margin convenience,
    # and a welded gap degenerates the split (self-paired nodes)
    fsB = uw.meshing.FaultSurface("Cross", P_B)
    fsB.triangulate()
    fsA2 = uw.meshing.FaultSurface("Main", P_A)
    fsA2.triangulate()
    netj = uw.meshing.FaultNetwork([fsA2, fsB],
                                   hierarchy=["Main", "Cross"])
    netj.prepare(h=H, ligament=1.0, verbose=False)
    with pytest.raises(ValueError, match="junction gap"):
        netj.build(width=WIDTH, margin_rings=1)


def test_an_outcropping_band_is_refused():
    """A band that reaches the domain boundary cannot yet carry its
    embedded mid-surface: refused loudly, before any meshing."""
    P_out = np.array([[0.30, 0.50, 0.30], [0.70, 0.50, 0.30],
                      [0.70, 0.50, 0.95], [0.30, 0.50, 0.95]])
    fs = uw.meshing.FaultSurface("Daylight", P_out)
    fs.triangulate()
    net = uw.meshing.FaultNetwork([fs])
    net.prepare(h=H, verbose=False)
    with pytest.raises(NotImplementedError, match="boundary"):
        net.build(width=WIDTH)


def test_embed_belongs_to_the_network_mesher():
    """embed= means mid-surfaces fragmented into the fused band; any
    other mesher would silently ignore it, so it is refused."""
    from underworld3.utilities.place_surface import place_thin_volume

    box = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0, 0.0), maxCoords=(1.0, 1.0, 1.0),
        cellSize=0.5, qdegree=1)
    with pytest.raises(ValueError, match="network"):
        place_thin_volume(box.dm, [P_A], 0.05, embed=[P_A])


def test_the_expansion_survives_collinear_rim_points():
    """A clipped rim may carry collinear consecutive points (the
    convexity gate passes them); the in-plane offset must not divide
    by zero there — on a straight rim the corner is the edge's own
    offset — and the plane normal must not come from a collinear
    leading triple (Newell's method)."""
    from underworld3.meshing.fault_network import _expand_convex_polygon

    P = np.array([[0.30, 0.50, 0.30], [0.50, 0.50, 0.30],
                  [0.70, 0.50, 0.30],
                  [0.70, 0.50, 0.70], [0.30, 0.50, 0.70]])
    E = _expand_convex_polygon(P, 0.08)
    assert np.isfinite(E).all()
    # the straight rim's midpoint moves exactly one offset outward,
    # staying in the patch plane
    assert E[1] == pytest.approx([0.50, 0.50, 0.22])
    assert np.allclose(E[:, 1], 0.50)


def test_the_weak_plane_gauges_the_jump_across_its_layer():
    """The 3-D weak plane's slip is the in-plane velocity jump across
    the layer. Read on a PRESCRIBED linear shear, where the jump is
    known exactly: no solve, just the gauge's own arithmetic."""
    import types

    P = np.array([[0.30, 0.30, 0.50], [0.70, 0.30, 0.50],
                  [0.70, 0.70, 0.50], [0.30, 0.70, 0.50]])
    fs = uw.meshing.FaultSurface("Flat", P)
    fs.triangulate()
    net = uw.meshing.FaultNetwork([fs])
    net.prepare(h=H, verbose=False)
    net.build(width=WIDTH, realisation="ti", h_far=0.24,
              margin_rings=1)

    a = 3.0                  # v = (a z, 0, 0): in-plane jump = 2 a skirt
    v = uw.discretisation.MeshVariable("Ug", net.mesh, 3, degree=2)
    v.array[:, 0, 0] = a * np.asarray(v.coords)[:, 2]
    v.array[:, 0, 1] = 0.0
    v.array[:, 0, 2] = 0.0

    skirt = 0.5 * WIDTH + float(net.info["spacing"][0])
    got = net.slips(types.SimpleNamespace(u=v))
    assert got["Flat"] == pytest.approx(2 * a * skirt, rel=1e-6)


def test_the_realisations_solve_on_the_shared_band():
    """Both realisations run on the one band under the same shear
    drive: the split's contact releases the drive-aligned senior, and
    the weak plane's TI solve converges on the same cells."""
    slips = {}
    for realisation in ("split", "ti"):
        net = _network(realisation)
        mesh = net.mesh
        x, y, z = mesh.X
        v = uw.discretisation.MeshVariable(f"v3W_{realisation}", mesh, 3,
                                           degree=2)
        p = uw.discretisation.MeshVariable(f"p3W_{realisation}", mesh, 1,
                                           degree=0, continuous=False)
        stokes = uw.systems.Stokes(mesh, velocityField=v,
                                   pressureField=p)
        stokes.bodyforce = [0.0, 0.0, 0.0]
        for wall in ("Bottom", "Top", "Left", "Right", "Front", "Back"):
            stokes.add_dirichlet_bc((y - 0.5, 0.0, 0.0), wall)
        if realisation == "split":
            stokes.constitutive_model = \
                uw.constitutive_models.ViscousFlowModel
            stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
            net.apply(stokes)
            stokes.petsc_use_pressure_nullspace = True
            stokes.tolerance = 1e-5
            info = net.solve(stokes)
            assert info.get("converged")
            # the band's base owns a geometric tail; the split child must
            # inherit it (it silently fell back to GAMG before)
            assert info.get("velocity_pc") == "custom-FMG", info
        else:
            net.apply(stokes, eta_1=0.01)
            stokes.petsc_use_pressure_nullspace = True
            stokes.tolerance = 1e-5
            stokes.solve()
        slips[realisation] = net.slips(stokes)

    for realisation, got in slips.items():
        assert all(np.isfinite(s) for s in got.values()), (
            realisation, got)
    # the drive-aligned senior dominates the split
    s = slips["split"]
    assert s["Main"] > 0.05
    assert s["Main"] > 3 * max(v for n, v in s.items() if n != "Main")
    # the two gauges are different quantities on different physics, but
    # both are the layer's own throughput: same order of magnitude
    assert 0.1 * s["Main"] < slips["ti"]["Main"] < 10 * s["Main"]
