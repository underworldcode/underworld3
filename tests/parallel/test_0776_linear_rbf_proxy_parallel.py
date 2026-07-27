"""Linear-exact swarm proxy transfer under MPI.

Proxy refresh is strictly rank-local: the kd-tree indexes only this rank's
particles, and there is no halo exchange (SWARM-15, recorded in
``docs/developer/design/SWARM_MODERNIZATION_DESIGN_2026-07.md`` §4). A proxy
node near a partition seam therefore gathers from a one-sided neighbourhood,
which is why proxy values have historically been np-dependent.

A linear-exact stencil reproduces a linear field exactly from *any*
neighbourhood, one-sided or not. So for a linear field the seam error is
zero, and the proxy becomes np-independent. That is what this test pins.

It does **not** claim the seam problem is fixed in general: for a field with
curvature the one-sided stencil still differs from a centred one, so np
dependence remains. The test asserts the linear case only, which is the part
that is genuinely exact.

Run with:
    mpirun -n 2 python -m pytest --with-mpi tests/parallel/test_0776_linear_rbf_proxy_parallel.py
    mpirun -n 4 python -m pytest --with-mpi tests/parallel/test_0776_linear_rbf_proxy_parallel.py
"""

import numpy as np
import pytest

import underworld3 as uw
from underworld3.meshing import UnstructuredSimplexBox

pytestmark = [
    pytest.mark.level_1,
    pytest.mark.tier_b,
    pytest.mark.mpi(min_size=2),
    pytest.mark.timeout(120),
]


def _linear(coords):
    return 0.5 + coords @ np.arange(1, coords.shape[1] + 1, dtype=float)


def _build():
    mesh = UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 12.0
    )
    swarm = uw.swarm.Swarm(mesh)
    var = swarm.add_variable(name="f", size=1, proxy_degree=1)
    swarm.populate(fill_param=4)
    return mesh, swarm, var


@pytest.mark.mpi(min_size=2)
def test_proxy_is_linear_exact_on_every_rank():
    """Every rank's owned proxy nodes carry the analytic linear field."""
    mesh, swarm, var = _build()

    proxy = var._meshVar
    var.data[:, 0] = _linear(swarm._particle_coordinates.data)
    var._rbf_to_meshVar(proxy)

    node_coords = np.asarray(proxy.coords)
    expected = _linear(node_coords)
    local_error = (
        np.abs(np.asarray(proxy.data[:, 0]) - expected).max() / np.abs(expected).max()
        if node_coords.shape[0]
        else 0.0
    )

    # Collective: a rank-local assertion would let one bad rank pass silently
    # while the others reported success (the house failure mode).
    global_error = uw.mpi.comm.allreduce(local_error, uw.MPI.MAX)

    assert global_error < 1.0e-12, (
        f"rank-local linear-exact transfer left a global max error of "
        f"{global_error:.3e}; seam nodes should still be exact for a linear field"
    )

    del swarm
    del mesh


@pytest.mark.mpi(min_size=2)
def test_proxy_constant_is_exact_and_collective_refresh_completes():
    """A constant is exact under either scheme; this checks the refresh itself
    stays collective (every rank performs the same read-then-write)."""
    mesh, swarm, var = _build()

    proxy = var._meshVar
    var.data[:, 0] = 2.75
    var._rbf_to_meshVar(proxy)

    local_error = (
        np.abs(np.asarray(proxy.data[:, 0]) - 2.75).max()
        if proxy.coords.shape[0]
        else 0.0
    )
    global_error = uw.mpi.comm.allreduce(local_error, uw.MPI.MAX)

    assert global_error < 1.0e-12, f"constant field error {global_error:.3e}"

    del swarm
    del mesh
