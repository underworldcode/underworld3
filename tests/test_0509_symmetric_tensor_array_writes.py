"""Writing one off-diagonal of a symmetric tensor through ``.array``.

A symmetric tensor is ``(dim, dim)`` in the ``.array`` view and stores only
its independent components, so ``[i, j]`` and ``[j, i]`` share ONE stored
value. The pack-back loop visits every ``(i, j)`` and writes that shared
column twice, so the pair's second visit overwrote the first:

    var.array[:, 0, 1] = 30.0     ->  stored 0.0     silently discarded
    var.array[:, 1, 0] = 30.0     ->  stored 30.0    worked

Lower beat upper purely because it came last in the loop. In 3-D all three
upper off-diagonals vanished. Nothing raised, and reading ``.array`` back
showed the write had not happened -- so a stress or strain-rate history
assembled component by component through the documented interface lost every
shear term.

The charter (section 7) makes ``.array`` the interface for new code, which is
what makes this worth a guard rather than a note.
"""

import numpy as np
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]


def _box(dim=2):
    if dim == 2:
        return uw.meshing.UnstructuredSimplexBox(cellSize=0.4, qdegree=3)
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0, 0.0), maxCoords=(1.0, 1.0, 1.0),
        cellSize=0.5, qdegree=3,
    )


@pytest.mark.parametrize("corner", [(0, 1), (1, 0)])
def test_either_off_diagonal_write_lands(corner):
    """Both halves of the pair must work, and both must mirror."""
    mesh = _box()
    i, j = corner
    var = uw.discretisation.MeshVariable(
        f"sym{i}{j}", mesh, vtype=uw.VarType.SYM_TENSOR, degree=1)

    with uw.synchronised_array_update():
        var.array[:, i, j] = 30.0

    read_back = np.asarray(var.array)
    assert np.allclose(read_back[:, i, j], 30.0), read_back[0]
    assert np.allclose(read_back[:, j, i], 30.0), "the write did not mirror"
    assert np.allclose(np.asarray(var.data)[:, 2], 30.0), "it never reached storage"


def test_an_off_diagonal_write_lands_without_the_context_manager():
    mesh = _box()
    var = uw.discretisation.MeshVariable(
        "symplain", mesh, vtype=uw.VarType.SYM_TENSOR, degree=1)
    var.array[:, 0, 1] = 30.0
    assert np.allclose(np.asarray(var.array)[:, 0, 1], 30.0)


def test_every_upper_off_diagonal_lands_in_three_dimensions():
    """2-D has one pair and 3-D has three; all of them were lost."""
    mesh = _box(dim=3)
    var = uw.discretisation.MeshVariable(
        "sym3d", mesh, vtype=uw.VarType.SYM_TENSOR, degree=1)

    with uw.synchronised_array_update():
        var.array[:, 0, 0] = 1.0
        var.array[:, 1, 1] = 2.0
        var.array[:, 2, 2] = 3.0
        var.array[:, 0, 1] = 4.0
        var.array[:, 0, 2] = 5.0
        var.array[:, 1, 2] = 6.0

    read_back = np.asarray(var.array)[0]
    expected = np.array([[1.0, 4.0, 5.0], [4.0, 2.0, 6.0], [5.0, 6.0, 3.0]])
    assert np.allclose(read_back, expected), read_back


def test_asking_for_an_asymmetric_value_is_refused():
    """[i, j] and [j, i] are one stored number. Setting them to different
    values in one assignment cannot be honoured, and used to be resolved
    silently by whichever the loop wrote last."""
    mesh = _box()
    var = uw.discretisation.MeshVariable(
        "symbad", mesh, vtype=uw.VarType.SYM_TENSOR, degree=1)
    values = np.zeros((np.asarray(var.array).shape[0], 2, 2))
    values[:, 0, 1] = 7.0
    values[:, 1, 0] = 9.0

    with pytest.raises(ValueError, match="symmetric tensor"):
        with uw.synchronised_array_update():
            var.array[...] = values


def test_a_full_tensor_still_holds_both_corners():
    """The mirroring must not touch VarType.TENSOR, which stores all four."""
    mesh = _box()
    var = uw.discretisation.MeshVariable(
        "fullt", mesh, vtype=uw.VarType.TENSOR, degree=1)

    with uw.synchronised_array_update():
        var.array[:, 0, 1] = 7.0
        var.array[:, 1, 0] = 9.0

    read_back = np.asarray(var.array)[0]
    assert read_back[0, 1] == pytest.approx(7.0)
    assert read_back[1, 0] == pytest.approx(9.0)


def test_a_whole_symmetric_assignment_is_unchanged():
    """The path that always worked must keep working."""
    mesh = _box()
    var = uw.discretisation.MeshVariable(
        "symwhole", mesh, vtype=uw.VarType.SYM_TENSOR, degree=1)
    values = np.zeros((np.asarray(var.array).shape[0], 2, 2))
    values[:, 0, 0], values[:, 1, 1] = 1.0, 2.0
    values[:, 0, 1] = values[:, 1, 0] = 3.0

    with uw.synchronised_array_update():
        var.array[...] = values

    assert np.allclose(np.asarray(var.data)[0], [1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# The same guarantee on the other two carriers. The swarm variant failed
# differently -- loudly rather than silently -- because its pack was a flat
# reshape that ignored symmetric storage entirely:
#
#     ValueError: could not broadcast input array from shape (156,4)
#                 into shape (156,3)
#
# so a symmetric tensor on a swarm could not be written through .array at all.
# ---------------------------------------------------------------------------


def test_a_swarm_symmetric_tensor_takes_an_off_diagonal_write():
    mesh = _box()
    swarm = uw.swarm.Swarm(mesh)
    stress = uw.swarm.SwarmVariable("tau", swarm, vtype=uw.VarType.SYM_TENSOR)
    swarm.populate(fill_param=2)

    with uw.synchronised_array_update():
        stress.array[:, 0, 1] = 30.0

    read_back = np.asarray(stress.array)
    assert np.asarray(stress.data).shape[1] == 3, "symmetric storage is 3 wide in 2-D"
    assert np.allclose(read_back[:, 0, 1], 30.0)
    assert np.allclose(read_back[:, 1, 0], 30.0)


def test_an_integration_point_symmetric_tensor_takes_an_off_diagonal_write():
    mesh = _box()
    stress = uw.discretisation.IntegrationPointVariable(
        "tauq", mesh, vtype=uw.VarType.SYM_TENSOR)

    with uw.synchronised_array_update():
        stress.array[:, 0, 1] = 30.0

    read_back = np.asarray(stress.array)
    assert np.allclose(read_back[:, 0, 1], 30.0)
    assert np.allclose(read_back[:, 1, 0], 30.0)


def test_a_swarm_vector_round_trips_unchanged():
    """The pack change must not disturb the shapes that already worked."""
    mesh = _box()
    swarm = uw.swarm.Swarm(mesh)
    velocity = uw.swarm.SwarmVariable("vel", swarm, vtype=uw.VarType.VECTOR)
    swarm.populate(fill_param=2)

    with uw.synchronised_array_update():
        velocity.array[:, 0, 0] = 1.0
        velocity.array[:, 0, 1] = 2.0

    assert np.allclose(np.asarray(velocity.data)[0], [1.0, 2.0])


# ---------------------------------------------------------------------------
# Deferred writes.
#
# `synchronised_array_update()` defers the PETSc pack, and the swarm view read
# its "current" values straight from the PETSc field -- so inside that context
# every write started from the pre-context state and overwrote the one before
# it. Only the last survived, silently. Mesh variables were immune, because
# their view reads and writes the canonical array.
#
# Writing a vector or tensor one component at a time inside that context is
# exactly what docs/developer/subsystems/data-access.md recommends.
# ---------------------------------------------------------------------------


def _carrier(mesh, vtype, tag, carrier):
    """The same variable on a mesh, a swarm, or the integration points.

    Returns the swarm alongside it: a SwarmVariable holds only a weak
    reference to its swarm, so a local one is collected when the helper
    returns and the variable then refuses to be read.
    """
    if carrier == "mesh":
        return uw.discretisation.MeshVariable(
            f"m{tag}", mesh, vtype=vtype, degree=1), None
    if carrier == "integration_point":
        return uw.discretisation.IntegrationPointVariable(
            f"q{tag}", mesh, vtype=vtype), None
    swarm = uw.swarm.Swarm(mesh)
    variable = uw.swarm.SwarmVariable(f"s{tag}", swarm, vtype=vtype)
    swarm.populate(fill_param=2)
    return variable, swarm


@pytest.mark.parametrize("carrier", ["mesh", "swarm", "integration_point"])
def test_component_writes_in_one_context_all_survive(carrier):
    """Every component written inside a single deferred context must land."""
    mesh = _box()
    var, _swarm = _carrier(mesh, uw.VarType.VECTOR, f"v{carrier[:2]}", carrier)

    with uw.synchronised_array_update():
        var.array[:, 0, 0] = 1.0
        var.array[:, 0, 1] = 2.0

    assert np.allclose(np.asarray(var.data)[0], [1.0, 2.0]), (
        f"{carrier}: an earlier write in the context was overwritten")


@pytest.mark.parametrize("carrier", ["mesh", "swarm", "integration_point"])
@pytest.mark.parametrize("corner", [(0, 1), (1, 0)])
def test_a_symmetric_off_diagonal_write_lands_on_every_carrier(carrier, corner):
    """The product of shape, carrier and index — the cell of this matrix that
    nobody had filled is where both defects lived."""
    mesh = _box()
    i, j = corner
    var, _swarm = _carrier(
        mesh, uw.VarType.SYM_TENSOR, f"t{carrier[:2]}{i}{j}", carrier)

    with uw.synchronised_array_update():
        var.array[:, i, j] = 30.0

    read_back = np.asarray(var.array)
    assert np.allclose(read_back[:, i, j], 30.0), f"{carrier} {corner}: lost"
    assert np.allclose(read_back[:, j, i], 30.0), f"{carrier} {corner}: not mirrored"
