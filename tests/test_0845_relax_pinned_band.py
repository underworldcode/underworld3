"""Relaxation with an interface band held fixed.

Relaxation and interface-tracking refinement work against each other: the mover
optimises element shape against an equilateral reference and knows nothing about
where the material changes, so it slides the small cells that refinement placed
on an interface *off* it. ``pin_bands`` is the fix, and these are the properties
that make it a fix rather than just a way to switch the mover off:

* the pinned vertices do not move **at all** — exactly, not approximately;
* vertices away from the band **do** move, so the mover is still working;
* the domain boundary stays pinned. That one is a real trap: passing
  ``pinned_labels`` explicitly REPLACES the auto default of "pin every named
  boundary", so a naive implementation that substituted the band label would
  silently let the mover deform the box.
"""
import numpy as np
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]


def _coords(mesh):
    return np.asarray(mesh.dm.getCoordinatesLocal().array).reshape(-1, mesh.dim)


def _fixture(cell_size=0.2):
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=cell_size,
        regular=False, qdegree=2)
    points = np.array([[0.12, 0.10, 0.0], [0.50, 0.52, 0.0], [0.88, 0.92, 0.0]])
    surface = uw.meshing.Surface("pinflt", mesh, points, symbol="Pf")
    surface.discretize()
    return mesh, surface


def test_pinned_vertices_do_not_move_and_others_do():
    mesh, surface = _fixture()
    before = _coords(mesh).copy()

    name = mesh.label_interface_band(surface, offset=0.0, halo=1)
    label = mesh.dm.getLabel(name)
    vS, vE = mesh.dm.getDepthStratum(0)
    pinned = np.array(sorted(
        int(p) for p in label.getStratumIS(1).getIndices())) - vS
    assert len(pinned) > 0, "no band was labelled; the fixture is not exercising it"

    mesh.relax(pin_bands=[surface], pin_halo=1)
    after = _coords(mesh)

    moved = np.linalg.norm(after - before, axis=1)
    assert moved[pinned].max() == 0.0, (
        f"{int((moved[pinned] > 0).sum())} pinned vertices moved")

    free = np.setdiff1d(np.arange(len(before)), pinned)
    assert moved[free].max() > 0.0, (
        "nothing moved anywhere — pinning switched the mover off rather than "
        "steering it")


def test_domain_boundary_stays_pinned():
    """``pin_bands`` must MERGE with the auto-pinned boundaries, not replace them."""
    mesh, surface = _fixture()
    before = _coords(mesh).copy()
    on_boundary = (np.isclose(before[:, 0], 0.0) | np.isclose(before[:, 0], 1.0)
                   | np.isclose(before[:, 1], 0.0) | np.isclose(before[:, 1], 1.0))

    mesh.relax(pin_bands=[surface])
    after = _coords(mesh)

    assert np.allclose(after[on_boundary], before[on_boundary], atol=0.0), (
        "the domain boundary moved; pin_bands replaced the auto-pinned labels "
        "instead of adding to them")


def test_offset_selects_the_weak_zone_margin():
    """An offset band tracks the level set, not the surface."""
    mesh, surface = _fixture()
    at_surface = mesh.label_interface_band(surface, offset=0.0, halo=0,
                                           name="band_zero")
    at_margin = mesh.label_interface_band(surface, offset=0.15, halo=0,
                                          name="band_margin")
    vS, _vE = mesh.dm.getDepthStratum(0)
    X = _coords(mesh)
    d = surface.unsigned_distance(X)

    for name, offset in ((at_surface, 0.0), (at_margin, 0.15)):
        idx = np.array(sorted(int(p) for p in
                              mesh.dm.getLabel(name).getStratumIS(1).getIndices()))
        assert len(idx) > 0, f"{name} labelled nothing"
        # Every pinned vertex belongs to a cell the level set cuts, so it must lie
        # within a cell diameter of that level set.
        assert np.abs(d[idx - vS] - offset).min() < 0.2

    zero = {int(p) for p in mesh.dm.getLabel(at_surface).getStratumIS(1).getIndices()}
    margin = {int(p) for p in mesh.dm.getLabel(at_margin).getStratumIS(1).getIndices()}
    assert zero != margin, "the offset had no effect on which band was labelled"


def test_halo_grows_the_pinned_set():
    mesh, surface = _fixture()
    sizes = []
    for halo in (0, 1, 2):
        name = mesh.label_interface_band(surface, halo=halo, name=f"h{halo}")
        sizes.append(len(mesh.dm.getLabel(name).getStratumIS(1).getIndices()))
    assert sizes[0] < sizes[1] < sizes[2], sizes
