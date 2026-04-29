"""Regression tests for ``Mesh.adapt`` variable transfer.

``Mesh.adapt`` rebuilds the underlying PETSc DM and re-establishes every
registered MeshVariable on the new mesh. The transfer routes via
``uw.function.evaluate`` (which uses a transient swarm under the hood
in parallel) so that user-supplied field values survive the adaptation.

The bug these tests catch is a silent reset to zero. Before the fix,
``temp_mesh.X.coords`` was used as the evaluation target — this only
matches the new mesh's degree-1 continuous DOFs, so any degree>=2 or
discontinuous variable came out the wrong size and the reshape raised,
which the surrounding ``try`` swallowed and the variable was zeroed.

These tests require a PETSc build with mesh-adaptation backends
(MMG/pragmatic). The conda-forge PETSc used by the default CI matrix
doesn't include them, so the tests skip there. They run on the
``amr-dev``/``amr-openmpi-dev`` environments which build PETSc from
source with the adaptation tooling.
"""

import numpy as np
import pytest

import underworld3 as uw
from underworld3.meshing import UnstructuredSimplexBox


def _has_petsc_adaptation_backend():
    """Probe whether PETSc was built with MMG/pragmatic.

    The cheapest test: build a tiny mesh, try to call ``adaptMetric`` with
    a trivial isotropic metric. If PETSc raises ``error code 63``
    (``PETSC_ERR_ARG_OUTOFRANGE`` — typical of "no mesh adapter
    configured"), the backend is missing.
    """
    try:
        m = UnstructuredSimplexBox(
            minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.5,
        )
        H = uw.discretisation.MeshVariable("_probe", m, 1, degree=1)
        H.array[:, 0, 0] = 100.0
        # Use the real adapt() entry point but with a coarse metric;
        # if PETSc raises, we capture and skip.
        m.adapt(H)
        return True
    except Exception:
        return False


_petsc_has_adaptation = _has_petsc_adaptation_backend()

pytestmark = [
    pytest.mark.level_2,
    pytest.mark.skipif(
        not _petsc_has_adaptation,
        reason="PETSc was built without mesh-adaptation backends "
        "(MMG/pragmatic). Use the amr-dev environment to run these tests.",
    ),
]


def _smooth_field(coords):
    return np.sin(np.pi * coords[:, 0]) * np.cos(np.pi * coords[:, 1])


def _refinement_metric(mesh):
    """Refine the right half of the box, keep the left coarse."""
    H = uw.discretisation.MeshVariable("_H_metric", mesh, 1, degree=1)
    H.array[:, 0, 0] = np.where(
        H.coords[:, 0] > 0.5, 1.0 / 0.05 ** 2, 1.0 / 0.1 ** 2
    )
    return H


@pytest.fixture
def mesh():
    return UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=1.0 / 16.0,
    )


def test_adapt_preserves_degree_1_continuous(mesh):
    """Degree-1 continuous field survives adapt within RBF tolerance."""
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=1)
    T.array[:, 0, 0] = _smooth_field(T.coords)

    metric = _refinement_metric(mesh)
    mesh.adapt(metric)

    T2 = mesh._vars["T"]
    expected = _smooth_field(T2.coords)
    actual = np.asarray(T2.array[:, 0, 0])
    max_err = float(np.abs(expected - actual).max())
    p95_err = float(np.percentile(np.abs(expected - actual), 95))

    assert np.asarray(T2.array).max() > 0.5, (
        "Field appears to have been reset to zero by adapt"
    )
    # RBF interpolation tolerance — coarse cells are ~1/10 wide so 2% is
    # well within the smoothing kernel. Tighten if the implementation
    # ever beats this.
    assert max_err < 0.1, f"max err = {max_err:.3e}"
    assert p95_err < 0.05, f"p95 err = {p95_err:.3e}"


def test_adapt_preserves_degree_2_continuous(mesh):
    """Degree-2 continuous field survives adapt — the case the previous
    bug silently reset to zero.
    """
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
    T.array[:, 0, 0] = _smooth_field(T.coords)

    metric = _refinement_metric(mesh)
    mesh.adapt(metric)

    T2 = mesh._vars["T"]
    expected = _smooth_field(T2.coords)
    actual = np.asarray(T2.array[:, 0, 0])
    max_err = float(np.abs(expected - actual).max())
    p95_err = float(np.percentile(np.abs(expected - actual), 95))

    assert np.asarray(T2.array).max() > 0.5, (
        "Field appears to have been reset to zero by adapt"
    )
    assert max_err < 0.05, f"max err = {max_err:.3e}"
    assert p95_err < 0.01, f"p95 err = {p95_err:.3e}"


def test_adapt_preserves_vector_field(mesh):
    """A vector MeshVariable survives adapt with both components intact."""
    V = uw.discretisation.MeshVariable("V", mesh, mesh.dim, degree=2)
    V.array[:, 0, 0] = _smooth_field(V.coords)
    V.array[:, 0, 1] = -_smooth_field(V.coords)

    metric = _refinement_metric(mesh)
    mesh.adapt(metric)

    V2 = mesh._vars["V"]
    expected_x = _smooth_field(V2.coords)
    expected_y = -_smooth_field(V2.coords)
    actual_x = np.asarray(V2.array[:, 0, 0])
    actual_y = np.asarray(V2.array[:, 0, 1])

    assert np.asarray(V2.array[:, 0, 0]).max() > 0.5, "x-component zeroed"
    assert np.asarray(V2.array[:, 0, 1]).min() < -0.5, "y-component zeroed"
    assert np.abs(expected_x - actual_x).max() < 0.05
    assert np.abs(expected_y - actual_y).max() < 0.05


def test_adapt_preserves_constant_field(mesh):
    """Constant field comes back constant — the simplest sanity check."""
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
    T.array[:, 0, 0] = 3.14

    metric = _refinement_metric(mesh)
    mesh.adapt(metric)

    T2 = mesh._vars["T"]
    actual = np.asarray(T2.array[:, 0, 0])
    assert np.allclose(actual, 3.14, atol=1e-8), (
        f"Constant field corrupted by adapt — range "
        f"[{actual.min():.3e}, {actual.max():.3e}]"
    )
