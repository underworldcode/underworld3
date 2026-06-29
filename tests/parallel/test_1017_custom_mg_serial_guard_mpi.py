"""Parallel guard for custom-P geometric MG (custom_mg).

Custom-P transfers are serial-only (experimental): the reduced maps use rank-local
DOF indices and the prolongations assemble as serial AIJ, so at np>1 they would
silently build wrong P. set_custom_fmg / inject must therefore FAIL LOUDLY in
parallel rather than produce incorrect results. Parallel (np>1) support is a
designed fast-follow (nested co-partitioned, rank-local P + MPIAIJ).

Run:
    mpirun -n 2 python -m pytest --with-mpi tests/parallel/test_1017_custom_mg_serial_guard_mpi.py
"""
import pytest
import sympy
import underworld3 as uw
from underworld3.utilities import custom_mg

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.timeout(120)]


def _box(cellSize, refinement=None):
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
        cellSize=cellSize, refinement=refinement, qdegree=2)


def _poisson(mesh):
    p = uw.systems.Poisson(mesh)
    p.constitutive_model = uw.constitutive_models.DiffusionModel
    p.constitutive_model.Parameters.diffusivity = 1
    p.f = 0.0
    p.add_dirichlet_bc(0.0, "Bottom"); p.add_dirichlet_bc(1.0, "Top")
    return p


@pytest.mark.mpi(min_size=2)
def test_set_custom_fmg_raises_in_parallel():
    """set_custom_fmg must raise NotImplementedError (not silently mis-build) np>1."""
    assert uw.mpi.size > 1
    s = _poisson(_box(0.25, refinement=2))
    with pytest.raises(NotImplementedError):
        custom_mg.set_custom_fmg(s, [_box(0.285), _box(0.142)], builder="barycentric")


@pytest.mark.mpi(min_size=2)
def test_inject_guard_blocks_legacy_path_in_parallel():
    """The legacy set_custom_mg path must also fail loudly at solve() in parallel."""
    assert uw.mpi.size > 1
    s = _poisson(_box(0.25, refinement=2))
    # legacy setter writes _custom_mg directly (no module guard); inject must catch it
    s.set_custom_mg([_box(0.285), _box(0.142)], kind="barycentric")
    with pytest.raises(NotImplementedError):
        s.solve()
