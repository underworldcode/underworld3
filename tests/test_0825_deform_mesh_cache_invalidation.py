"""Regression test for cache invalidation on Mesh._deform_mesh.

After a coordinate change, three mesh-level caches need to be marked
stale so subsequent lookups recompute against the new geometry:

  - self._evaluation_hash + self._evaluation_interpolated_results
    (the legacy uw.function.evaluate fast-path cache)
  - self._dminterpolation_cache (DMInterpolation structures keyed on
    coord-hash; encode cell residency and reference->physical maps)
  - self._topology_version (a counter that downstream caches consult)

mesh.adapt() and _legacy_access already invalidate these. Before this
fix, _deform_mesh did not, leaving a gap where user code that re-uses
the same coord array across a deformation could hit a stale cache
built on the pre-deform mesh.

This test warms each cache (by performing an evaluate that populates
them), then calls _deform_mesh and verifies the caches are cleared
and the topology version has been bumped.
"""

import numpy as np
import pytest

import underworld3 as uw


pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


def _build_mesh_with_field():
    """A small structured-quad mesh plus a scalar field to drive evals."""
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(8, 8),
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
    )
    s = uw.discretisation.MeshVariable(
        "S_cache_test", mesh, 1, degree=2, continuous=True)
    s.data[:, 0] = 1.0  # constant — exact value irrelevant
    return mesh, s


def _warm_caches(mesh, var):
    """Trigger an evaluate that populates _evaluation_hash and
    _dminterpolation_cache."""
    coords = np.asarray(mesh.X.coords, dtype=np.double)
    # uw.function.evaluate goes through the petsc_interpolate path,
    # which populates _dminterpolation_cache.
    _ = uw.function.evaluate(var.sym, coords)


def test_topology_version_bumps_on_deform():
    mesh, s = _build_mesh_with_field()
    _warm_caches(mesh, s)
    before = mesh._topology_version
    coords = np.asarray(mesh.X.coords)
    perturbed = coords.copy()
    perturbed[:, 0] += 1.0e-3 * coords[:, 1]
    # _deform_mesh is the internal primitive (gated against bare use on a
    # live mesh); this test deliberately exercises it, so open the
    # sanctioned _coord_mutation scope.
    with mesh._coord_mutation():
        mesh._deform_mesh(perturbed)
    assert mesh._topology_version > before, (
        "_topology_version should be incremented by _deform_mesh")


def test_evaluation_hash_invalidated_on_deform():
    mesh, s = _build_mesh_with_field()
    _warm_caches(mesh, s)
    # uw.function.evaluate may have populated _evaluation_hash
    coords = np.asarray(mesh.X.coords)
    perturbed = coords.copy()
    perturbed[:, 1] += 5.0e-4 * coords[:, 0]
    with mesh._coord_mutation():
        mesh._deform_mesh(perturbed)
    assert mesh._evaluation_hash is None
    assert mesh._evaluation_interpolated_results is None


def test_dminterpolation_cache_invalidated_on_deform():
    mesh, s = _build_mesh_with_field()
    _warm_caches(mesh, s)
    # The cache should have at least one entry after the warm-up evaluate.
    n_before = len(getattr(
        mesh._dminterpolation_cache, "_cache", {}))
    coords = np.asarray(mesh.X.coords)
    perturbed = coords.copy()
    perturbed[:, 1] += 1.0e-3 * coords[:, 0]
    with mesh._coord_mutation():
        mesh._deform_mesh(perturbed)
    n_after = len(getattr(
        mesh._dminterpolation_cache, "_cache", {}))
    assert n_after == 0, (
        f"Expected DMInterpolation cache to be empty after deform; "
        f"got {n_after} entries (before deform: {n_before})")
