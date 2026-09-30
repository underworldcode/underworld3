"""Legacy-call guard for the mesh.adapt() semantics change.

Before 2026-07 ``mesh.adapt(metric)`` was the IN-PLACE MMG remesher (now
``mesh.remesh``). The nested adapt-on-top ``adapt`` returns a NEW child mesh
instead, which a legacy in-place caller would silently discard (or, on an
unrefined base, hit a confusing "needs refinement>=1" RuntimeError). A bare
call with no new-API keyword must therefore raise a TypeError pointing at
``mesh.remesh(metric)`` and at the opt-in keywords — never silently redirect
(the return semantics differ).
"""

import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


@pytest.fixture(scope="module")
def mesh_and_metric():
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.5, qdegree=2
    )
    metric = uw.discretisation.MeshVariable("Hguard", mesh, 1, degree=1)
    metric.data[:, 0] = 100.0
    return mesh, metric


def test_bare_adapt_call_raises_typeerror(mesh_and_metric):
    """mesh.adapt(metric) with no new-API keyword is the legacy in-place call
    shape and must refuse with guidance, not run either behaviour."""
    mesh, metric = mesh_and_metric
    with pytest.raises(TypeError, match=r"mesh\.remesh\(metric\)"):
        mesh.adapt(metric)


def test_bare_adapt_with_verbose_still_raises(mesh_and_metric):
    """verbose= exists in both signatures, so it does not disambiguate."""
    mesh, metric = mesh_and_metric
    with pytest.raises(TypeError, match="child mesh"):
        mesh.adapt(metric, verbose=True)


def test_error_message_names_both_paths(mesh_and_metric):
    """The message must state the in-place replacement AND the opt-in shape."""
    mesh, metric = mesh_and_metric
    with pytest.raises(TypeError) as excinfo:
        mesh.adapt(metric)
    message = str(excinfo.value)
    assert "mesh.remesh(metric)" in message
    assert "child mesh" in message
    assert "max_levels" in message


def test_explicit_keyword_opts_in(mesh_and_metric):
    """Passing any new-API keyword bypasses the guard and reaches the nested
    path (which then reports the refinement>=1 requirement for this base)."""
    mesh, metric = mesh_and_metric
    with pytest.raises(RuntimeError, match="refinement>=1"):
        mesh.adapt(metric, max_levels=1)
