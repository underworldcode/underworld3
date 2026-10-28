"""Checkpoint round-trip on a SPLIT (``add_fault``) mesh — issue #640.

A cut duplicates a node at exactly the same coordinate, one copy per
side. ``read_timestep``'s nearest-neighbour remap cannot choose between
them, so before the fix both sides were handed the same saved value and
part of the slip discontinuity was smeared into the first element ring
(near-fault stress came back ~200x the in-memory answer).

The tests below stamp each duplicate pair with values that differ by
construction, so a collapse is exact and visible without a solve.
"""
import numpy as np
import pytest

import underworld3 as uw

TRACE = np.array([[0.30, 0.50], [0.50, 0.52], [0.70, 0.50]])


def _duplicate_groups(coords):
    """{coordinate: [row indices]} for coordinates carrying >1 DOF."""
    groups = {}
    for row, point in enumerate(coords):
        groups.setdefault(tuple(point), []).append(row)
    return {k: v for k, v in groups.items() if len(v) > 1}


def _stamped_variable(mesh, name="vRT"):
    """A variable whose coincident DOFs hold deliberately different values."""
    var = uw.discretisation.MeshVariable(name, mesh, 2, degree=2)
    coords = np.asarray(var.coords_nd)
    stamp = np.zeros(coords.shape[0])
    for rows in _duplicate_groups(coords).values():
        for side, row in enumerate(rows):
            stamp[row] = 1.0 + side
    var.data[:, 0] = 100.0 * coords[:, 0] + 10.0 * coords[:, 1] + stamp
    var.data[:, 1] = stamp
    return var


def _split_mesh():
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
        cellSize=1 / 12, regular=False, qdegree=2,
    )
    return base.add_fault([("F", TRACE)])


def _write(tmp_path, mesh, var, petsc_reload):
    mesh.write_timestep(
        "rt", 0, outputPath=str(tmp_path), meshVars=[var],
        petsc_reload=petsc_reload,
    )
    reloaded = uw.discretisation.Mesh(
        str(tmp_path / "rt.mesh.00000.h5"), simplex=True, qdegree=2)
    return reloaded, uw.discretisation.MeshVariable(
        var.clean_name, reloaded, 2, degree=2)


def _collapsed_groups(saved_var, loaded_var):
    """Coincident groups whose two sides no longer hold distinct values."""
    written = {k: sorted(saved_var.data[r, 1] for r in rows)
               for k, rows in
               _duplicate_groups(np.asarray(saved_var.coords_nd)).items()}
    wrong = []
    for key, rows in _duplicate_groups(
            np.asarray(loaded_var.coords_nd)).items():
        want = written.get(key)
        if want is None:
            continue
        got = sorted(loaded_var.data[r, 1] for r in rows)
        if not np.allclose(got, want):
            wrong.append(key)
    return wrong


@pytest.mark.level_1
@pytest.mark.tier_a
def test_split_mesh_has_coincident_dofs():
    """The premise: a cut really does duplicate coordinates."""
    mesh = _split_mesh()
    var = uw.discretisation.MeshVariable("vPremise", mesh, 2, degree=2)
    groups = _duplicate_groups(np.asarray(var.coords_nd))
    assert len(groups) > 0
    assert max(len(rows) for rows in groups.values()) == 2


@pytest.mark.level_1
@pytest.mark.tier_a
def test_roundtrip_keeps_the_two_sides_distinct(tmp_path):
    """#640: both sides of the cut survive a write/read round-trip."""
    mesh = _split_mesh()
    var = _stamped_variable(mesh)
    _, loaded = _write(tmp_path, mesh, var, petsc_reload=True)

    loaded.data[...] = 0.0
    loaded.read_timestep("rt", var.clean_name, 0, outputPath=str(tmp_path))

    assert _collapsed_groups(var, loaded) == []


@pytest.mark.level_1
@pytest.mark.tier_a
def test_roundtrip_refuses_without_a_native_payload(tmp_path):
    """No section in the file means the sides cannot be told apart."""
    mesh = _split_mesh()
    var = _stamped_variable(mesh)
    _, loaded = _write(tmp_path, mesh, var, petsc_reload=False)

    with pytest.raises(RuntimeError, match="more than one DOF"):
        loaded.read_timestep("rt", var.clean_name, 0,
                             outputPath=str(tmp_path))


@pytest.mark.level_1
@pytest.mark.tier_a
def test_ambiguous_remap_is_available_on_request(tmp_path):
    """The escape hatch still reads — far-field use only, and it collapses."""
    mesh = _split_mesh()
    var = _stamped_variable(mesh)
    _, loaded = _write(tmp_path, mesh, var, petsc_reload=False)

    loaded.read_timestep("rt", var.clean_name, 0, outputPath=str(tmp_path),
                         allow_ambiguous_duplicates=True)

    # it reads (no raise), the smooth part is right away from the cut ...
    coords = np.asarray(loaded.coords_nd)
    off_cut = np.abs(coords[:, 1] - 0.51) > 0.1
    expected = 100.0 * coords[off_cut, 0] + 10.0 * coords[off_cut, 1]
    assert np.allclose(loaded.data[off_cut, 0], expected, atol=1e-8)
    # ... and the cut is exactly the damage this flag admits to
    assert len(_collapsed_groups(var, loaded)) > 0


@pytest.mark.level_1
@pytest.mark.tier_a
def test_unsplit_mesh_still_uses_the_coordinate_remap(tmp_path):
    """No duplicates, no change: the flexible remap path is untouched."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
        cellSize=1 / 12, regular=False, qdegree=2,
    )
    var = _stamped_variable(mesh, name="vPlain")
    assert _duplicate_groups(np.asarray(var.coords_nd)) == {}

    _, loaded = _write(tmp_path, mesh, var, petsc_reload=False)
    loaded.read_timestep("rt", var.clean_name, 0, outputPath=str(tmp_path))

    coords = np.asarray(loaded.coords_nd)
    assert np.allclose(loaded.data[:, 0],
                       100.0 * coords[:, 0] + 10.0 * coords[:, 1], atol=1e-8)


@pytest.mark.level_1
@pytest.mark.tier_b
def test_the_payload_is_found_wherever_the_writer_put_it(tmp_path):
    """The existence check must not be tied to one file layout.

    #756 moved the reload payload from the file root into /restart/petsc.
    The writer still emitted the per-field section and its vector, one
    group deeper, but `_petsc_payload_name` looked only at the root, so it
    reported "no PETSc-native payload" for a file written WITH
    petsc_reload=True and the #640 guard refused the good path as well as
    the bad. The round-trip test above catches the symptom; this pins the
    cause, so a future relocation fails here with the reason rather than
    as a confusing refusal.
    """
    import h5py

    mesh = _split_mesh()
    var = _stamped_variable(mesh)
    _write(tmp_path, mesh, var, petsc_reload=True)
    written = tmp_path / f"rt.mesh.{var.clean_name}.00000.h5"

    # The section and its global vector are both on disk, somewhere.
    with h5py.File(written, "r") as h5f:
        groups = []
        h5f.visititems(
            lambda name, obj: groups.append(name)
            if isinstance(obj, h5py.Group) else None)
    assert any(g.endswith(f"dms/{var.clean_name}/section") for g in groups), groups
    assert any(g.endswith(f"dms/{var.clean_name}/vecs/{var.clean_name}")
               for g in groups), groups

    # ... and the guard's existence check finds it there. The method lives on
    # _BaseMeshVariable; EnhancedMeshVariable is a proxy and does not inherit
    # from it, so reach for the defining class rather than type(var).
    from underworld3.discretisation.discretisation_mesh_variables import (
        _BaseMeshVariable,
    )

    assert _BaseMeshVariable._petsc_payload_name(
        str(written), var.clean_name) == var.clean_name
