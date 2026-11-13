"""A boundary name the mesh holds no label for is an empty boundary (#1163).

A mesh's ``boundaries`` enum can name a boundary its DM holds no label for: a
solid disc, ``Annulus(radiusInner=0)``, lists ``Lower`` and has no inner wall.
petsc4py's ``dm.getLabel(name)`` does not return ``None`` for such a name. It
returns a label object with a null handle, and any query on that object
(``getNumValues``, ``getStratumSize``, ``getStratumIS``, ``getValueIS``) is a
segmentation fault. ``dm.hasLabel(name)`` is the test that answers.

Each function here took the ``None`` test for the missing-label case and
crashed the interpreter on the solid disc. The baselines are closed-form: a
boundary with no label has no vertices and no cells, the disc's rim is the
circle of its radius, and the default slip and smoothing leave the rim on it.

Whether a label exists is a property of the distributed DM, the same on every
rank (a rank holding no point of a label still holds the label), so the
answers here do not depend on the partition. This file carries no
``mpi(min_size=2)`` mark: it runs serially and in the parallel batches.
"""
import numpy as np
import pytest
from mpi4py import MPI

import underworld3 as uw
from underworld3.meshing.smoothing.graph import (_auto_pinned_labels, _owned_vertex_mask,
                                                 _pinned_mask)

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b, pytest.mark.timeout(300)]

RADIUS = 1.0
# Distance of a rim vertex from its circle, relative to the radius.
ON_RIM = 64 * np.finfo(float).eps


def _disc():
    return uw.meshing.Annulus(radiusOuter=RADIUS, radiusInner=0.0, cellSize=0.2)


def _global_sum(local_count):
    return uw.mpi.comm.allreduce(int(local_count))


def _global_max(local_value):
    return uw.mpi.comm.allreduce(float(local_value), op=MPI.MAX)


def test_the_solid_disc_names_a_wall_it_holds_no_label_for():
    """The fixture is the case of the issue: ``Lower`` is in the enum and the
    DM has no such label, on every rank."""
    mesh = _disc()
    assert "Lower" in [b.name for b in mesh.boundaries]
    assert not mesh.dm.hasLabel("Lower")
    assert mesh.dm.hasLabel("Upper")


def test_a_label_the_mesh_does_not_hold_pins_no_vertex():
    mesh = _disc()
    for name in ("Lower", "Centre", "no_such_boundary"):
        assert not _pinned_mask(mesh.dm, (name,)).any(), name
    # A missing name beside a held one does not hide the held one.
    with_missing = _pinned_mask(mesh.dm, ("Lower", "Upper"))
    assert np.array_equal(with_missing, _pinned_mask(mesh.dm, ("Upper",)))
    assert _global_sum(with_missing.sum()) > 0


def test_the_default_pinned_labels_are_the_ones_the_mesh_holds():
    assert _auto_pinned_labels(_disc()) == ("Upper",)
    annulus = uw.meshing.Annulus(radiusOuter=RADIUS, radiusInner=0.5, cellSize=0.2)
    assert _auto_pinned_labels(annulus) == ("Lower", "Upper")


def test_default_boundary_slip_on_the_solid_disc_keeps_the_rim():
    """The default projector slides the rim along its circle and restores it:
    after a displacement of every vertex each rank's owned rim vertices are
    back on the circle (the projector leaves ghost copies to their owners)."""
    mesh = _disc()
    _, project = mesh.boundary_slip()
    coords = np.asarray(mesh.X.coords).copy()
    rim = _pinned_mask(mesh.dm, ("Upper",)) & _owned_vertex_mask(mesh.dm)
    shifted = coords + 0.01 * np.array([1.0, 0.5])
    restored = np.asarray(project(shifted))
    radii = np.linalg.norm(restored[rim], axis=1)
    worst = _global_max(np.abs(radii - RADIUS).max() if radii.size else 0.0)
    assert worst / RADIUS < ON_RIM


def test_default_smoothing_on_the_solid_disc_keeps_the_rim():
    """``smooth_mesh_interior`` with its defaults pins every wall the mesh
    holds: the rim vertices do not move at all."""
    mesh = _disc()
    rim = _pinned_mask(mesh.dm, ("Upper",))
    before = np.asarray(mesh.X.coords)[rim].copy()
    uw.meshing.smooth_mesh_interior(mesh)
    after = np.asarray(mesh.X.coords)[rim]
    assert before.shape == after.shape
    moved = _global_max(np.abs(after - before).max() if before.size else 0.0)
    assert moved == 0.0


def test_no_cell_supports_a_wall_the_mesh_holds_no_label_for():
    mesh = _disc()
    assert not mesh.cells_supporting("Lower").any()
    assert _global_sum(mesh.cells_supporting("Upper").sum()) > 0


def test_the_facets_of_a_wall_the_mesh_holds_no_label_for_are_an_empty_surface():
    pytest.importorskip("pyvista")
    from underworld3.visualisation.visualisation import labelled_facets_to_pv_mesh

    mesh = _disc()
    assert labelled_facets_to_pv_mesh(mesh, "Lower").n_points == 0


def test_a_dm_without_the_consolidated_label_has_no_boundary_stratum():
    """The boundary-flux reader takes a boundary's points from the consolidated
    ``UW_Boundaries`` label of the DM it is handed; a DM without that label
    (one rebuilt outside the Mesh constructor) holds none of the boundary."""
    from underworld3.utilities.boundary_flux import _boundary_stratum_is

    mesh = _disc()
    bare = mesh.dm.clone()
    if bare.hasLabel("UW_Boundaries"):
        bare.removeLabel("UW_Boundaries")
    assert _boundary_stratum_is(bare, mesh, "Upper") is None
