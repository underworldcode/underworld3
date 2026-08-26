"""One fault specification, two realisations (:class:`FaultNetwork`).

A fault is specified once — trace, hierarchy, properties — and the
realisation is a keyword. The split cuts the band and constrains the
node pairs; the weak plane leaves the same band whole and paints a
transversely-isotropic rheology on it. The property under test is that
these are realisations of ONE specification: the same call, the same
prepared pieces, and a mesh whose CELLS are identical (only the split's
duplicated vertices differ), which is what makes the two comparable.

The negative controls are the two ways the pair could be faked: a weak
plane without a meshed layer (there is nothing to be weak), and a
directorless painted zone (an isotropic blob, not a fault).
"""
import numpy as np
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b,
              pytest.mark.skipif(uw.mpi.size > 1,
                                 reason="serial suite")]

H = 0.03
WIDTH = 0.02


def _trace():
    """A gently curved trace, well inside a unit box."""
    s = np.linspace(0.3, 0.7, 15)
    return np.column_stack([s, 0.5 + 0.15 * (s - 0.5)])


def _base():
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=8 * H,
        regular=False, refinement=1, qdegree=2)


def _network(realisation, base, width=WIDTH):
    net = uw.meshing.FaultNetwork([("Main", _trace())])
    net.prepare(h=H, verbose=False)
    net.build(base=base, width=width, realisation=realisation,
              max_levels=1)
    return net


def test_the_two_realisations_share_one_mesh():
    base = _base()
    split = _network("split", base)
    ti = _network("ti", base)

    assert split.realisation == "split" and ti.realisation == "ti"
    assert [n for n, _p in split.prepared] == [n for n, _p in ti.prepared]
    assert split.info["n_cells"] == ti.info["n_cells"], (
        "the realisations no longer share a mesh; the comparison between "
        "them is then confounded by the discretisation")
    # the ONLY difference: the split duplicated the cut's nodes
    assert (split.mesh.dm.getDepthStratum(0)[1]
            > ti.mesh.dm.getDepthStratum(0)[1])


def test_the_weak_plane_is_painted_on_the_fault_footprint():
    ti = _network("ti", _base())
    eta1, ndir, foot = ti.ti_fields(eta_1=0.01, eta_0=1.0)

    assert foot.sum() > 0, "no footprint cells: nothing would be weak"
    assert foot.sum() < ti.info["band"].sum() + 1
    vals = eta1.array[:, 0, 0]
    assert np.allclose(vals[foot], 0.01)
    assert np.allclose(vals[~foot], 1.0)

    # the director is a unit normal of the trace, cell by cell — a
    # painted zone WITHOUT this is an isotropic blob, not a fault
    d = np.asarray(ndir.array).reshape(-1, 2)[foot]
    assert np.allclose(np.linalg.norm(d, axis=1), 1.0)
    P = _trace()
    t = (P[-1] - P[0]) / np.linalg.norm(P[-1] - P[0])
    assert np.abs(d @ t).max() < 0.05, (
        "the directors are not perpendicular to the trace")


def test_a_weak_plane_needs_a_layer_to_be_weak_in():
    net = uw.meshing.FaultNetwork([("Main", _trace())])
    net.prepare(h=H, verbose=False)
    with pytest.raises(ValueError, match="width"):
        net.build(base=_base(), realisation="ti", max_levels=1)
    with pytest.raises(ValueError, match="realisation"):
        net.build(base=_base(), width=WIDTH, realisation="smeared",
                  max_levels=1)
    # the mesher choices are the ones THIS dimension offers
    with pytest.raises(ValueError, match="2-D"):
        net.build(base=_base(), width=WIDTH, mesher="embed", max_levels=1)


def test_the_solver_is_given_whichever_realisation_was_built():
    """One call imposes the network; what it imposes follows the
    realisation, and asking for the other one is refused rather than
    silently ignored."""
    for realisation in ("split", "ti"):
        net = _network(realisation, _base())
        mesh = net.mesh
        v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
        q = uw.discretisation.MeshVariable("P", mesh, 1, degree=1,
                                           continuous=True)
        stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=q)
        net.apply(stokes, eta_1=0.01)
        if realisation == "split":
            assert getattr(stokes, "_fault_contact_faults", []) == \
                [n for n, _p in net.prepared]
        else:
            assert net.ti["footprint"].sum() > 0
            params = stokes.constitutive_model.Parameters
            assert params.director.shape == (1, mesh.dim)
            with pytest.raises(RuntimeError, match="no fault pairs"):
                net.apply_contact(stokes)


def test_a_weak_plane_cannot_be_gauged_without_a_layer():
    net = uw.meshing.FaultNetwork([("Main", _trace())])
    net.prepare(h=H, verbose=False)
    net.build(base=_base(), max_levels=1)          # no width: no band
    assert net.info is None
    with pytest.raises(RuntimeError, match="width"):
        net.ti_fields(eta_1=0.01)


def test_the_weak_plane_gauges_the_jump_across_its_layer():
    """The weak plane has no node pair, so its slip is the tangential
    velocity jump across the layer. Read on a PRESCRIBED linear shear,
    where the jump is known exactly: no solve, no solver behaviour, just
    the gauge's own arithmetic and sampling.
    """
    import types

    s = np.linspace(0.3, 0.7, 15)
    trace = np.column_stack([s, np.full_like(s, 0.5)])     # horizontal
    net = uw.meshing.FaultNetwork([("Main", trace)])
    net.prepare(h=H, verbose=False)
    net.build(base=_base(), width=WIDTH, realisation="ti", max_levels=1)

    a = 3.0                                    # v = (a y, 0): t = x-hat
    v = uw.discretisation.MeshVariable("Ug", net.mesh, net.mesh.dim,
                                       degree=2)
    v.array[:, 0, 0] = a * np.asarray(v.coords)[:, 1]
    v.array[:, 0, 1] = 0.0

    skirt = 0.5 * WIDTH + float(net.info["spacing"][0])
    got = net.slips(types.SimpleNamespace(u=v))
    assert got["Main"] == pytest.approx(a * 2 * skirt, rel=1e-6)


def test_the_fault_carries_its_own_properties():
    """The surface object survives the realisation and is where
    fault-attached (not mesh-attached) properties live."""
    for realisation in ("split", "ti"):
        net = _network(realisation, _base())
        name = net.prepared[0][0]
        surf = net.surface(name)
        friction = surf.add_variable("mu", size=1)
        friction.data[:] = 0.6
        assert np.allclose(np.asarray(friction.data), 0.6)
        with pytest.raises(KeyError):
            net.surface("NotAFault")
