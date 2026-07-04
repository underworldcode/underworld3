"""Regression test for the mesh-coordinate-mutation capability gate (PR #246).

`Mesh._deform_mesh` is the raw internal primitive: it moves nodes WITHOUT
transferring fields or SL/DDt history. On a *live* mesh (one carrying variables
or remesh hooks) a bare call corrupts the solution, so it is gated: it must be
reached through a sanctioned path —

  * ``mesh.deform(new_coords, dt=...)``  — arbitrary displacement (transfer-aware)
  * ``with mesh._coord_mutation(): mesh._deform_mesh(...)``  — trusted internal
  * ``with mesh.ephemeral_coords(): ...``  — trial deform, restored on exit

This test LOCKS that behaviour so the user-facing protection can't silently
regress. See docs/developer/design/REMESH_FIELD_TRANSFER_DESIGN.md and
docs/developer/design/lagged-clone-sl-history.md.
"""

import numpy as np
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


def _live_mesh():
    """A small mesh carrying a variable (so the gate is armed)."""
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(6, 6), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0))
    s = uw.discretisation.MeshVariable("S_gate", mesh, 1, degree=1, continuous=True)
    s.data[:, 0] = 1.0
    return mesh, s


def _perturbed(mesh):
    coords = np.asarray(mesh.X.coords)
    p = coords.copy()
    p[:, 0] += 1.0e-3 * coords[:, 1]
    return p


def test_bare_deform_mesh_on_live_mesh_raises():
    """The gate fires: a bare _deform_mesh on a live mesh raises, naming the
    sanctioned public entry point."""
    mesh, _ = _live_mesh()
    with pytest.raises(RuntimeError, match="deform"):
        mesh._deform_mesh(_perturbed(mesh))


def test_public_deform_succeeds_and_moves_nodes():
    """mesh.deform(...) is the sanctioned path and actually moves the nodes."""
    mesh, _ = _live_mesh()
    before = np.asarray(mesh.X.coords).copy()
    mesh.deform(_perturbed(mesh), dt=1.0)
    after = np.asarray(mesh.X.coords)
    assert not np.allclose(before, after), "deform() should move the coordinates"


def test_coord_mutation_scope_sanctions_direct_call():
    """The internal _coord_mutation() scope lets a deliberate direct call pass
    the gate."""
    mesh, _ = _live_mesh()
    with mesh._coord_mutation():
        mesh._deform_mesh(_perturbed(mesh))  # must not raise


def test_ephemeral_coords_restores_on_exit():
    """ephemeral_coords() allows a trial deform and restores coords on exit."""
    mesh, _ = _live_mesh()
    before = np.asarray(mesh.X.coords).copy()
    with mesh.ephemeral_coords():
        mesh._deform_mesh(_perturbed(mesh))
        moved = np.asarray(mesh.X.coords)
        assert not np.allclose(before, moved), "trial deform should move coords"
    after = np.asarray(mesh.X.coords)
    assert np.allclose(before, after), "ephemeral_coords must restore coords on exit"
