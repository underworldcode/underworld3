"""Bounded transport-only memory regression on tiny volume simplices.

No Stokes, reaction diagnostics, checkpoints, or forced garbage collection
occur in the measured loop. RSS is current resident memory, not peak RSS.
This catches repeated allocation regressions; it cannot certify every
production mesh or long coupled trajectory as leak-free.
"""

import time

import numpy as np
import pytest

import underworld3 as uw
from underworld3.utilities import memprobe

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]


def _workspace(thermal):
    """Record identities, not contents that should change during transport."""
    identity = [thermal.snes.handle, thermal.dm.handle,
                tuple((name, field.vec.handle) for name, field in thermal.mesh.vars.items())]
    if thermal.time_integrator == "citcoms":
        identity.extend([
            thermal._lumped_mass.handle,
            tuple(vector.handle for vector in thermal._citcoms_work_vectors),
            tuple(id(array) for array in thermal._simplex_data_cache),
            tuple(id(array) for array in thermal._directional_rate_work),
        ])
    return identity


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("method", ["pc2", "cn", "bdf2"])
def test_repeated_transport_memory_and_workspace_reuse(dim, method):
    pytest.importorskip("psutil", reason="This test requires current RSS, not peak RSS.")
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0,) * dim, maxCoords=(1.0,) * dim,
        cellSize=0.25, qdegree=4, regular=False,
    )
    temperature = uw.discretisation.MeshVariable("T", mesh, 1, degree=1)
    velocity = uw.discretisation.MeshVariable("U", mesh, dim, degree=1)
    temperature.array[:, 0, 0] = np.prod(np.sin(np.pi * np.asarray(temperature.coords)), axis=1)
    velocity.array[...] = 0.0
    velocity.array[:, 0, 0] = 0.2
    settings = ({"time_integrator": "citcoms"} if method == "pc2"
                else {"order": 1, "theta": 0.5} if method == "cn"
                else {"order": 2})
    thermal = uw.systems.AdvDiffusionSUPG(mesh, temperature, velocity.sym, **settings)
    thermal.constitutive_model.Parameters.diffusivity = 0.01
    for boundary in mesh.boundaries:
        if boundary.name not in ("All_Boundaries", "Null_Boundary"):
            thermal.add_dirichlet_bc(0.0, boundary.name)
    samples = []
    start = time.perf_counter()
    for step in range(1, 201):
        velocity.array[:, 0, 0] = 0.2 * (1.0 + 0.1 * np.sin(step))
        dt = (0.001, 0.0015, 0.002, 0.0025)[step % 4]
        thermal.estimate_dt()
        thermal.solve(timestep=dt)
        if step == 40:
            workspace = _workspace(thermal)
        if step >= 40 and step % 10 == 0:
            unchanged = _workspace(thermal) == workspace
            assert all(uw.mpi.comm.allgather(unchanged)), "Solver workspace was reallocated"
            rss = uw.mpi.comm.allgather(memprobe.snapshot()["rss_mb"])
            samples.append((step, *rss))
    elapsed = max(uw.mpi.comm.allgather(time.perf_counter() - start))
    samples = np.asarray(samples)
    late = samples[samples[:, 0] >= 120]
    slopes = np.polyfit(late[:, 0], late[:, 1:], 1)[0]
    growth = samples[-1, 1:] - samples[0, 1:]
    nodal = np.asarray(temperature.array)
    assert all(uw.mpi.comm.allgather(bool(np.isfinite(nodal).all())))
    minimum = min(uw.mpi.comm.allgather(float(np.min(nodal))))
    maximum = max(uw.mpi.comm.allgather(float(np.max(nodal))))
    uw.pprint(
        f"SUPG_MEMORY method={method} dim={dim} ranks={uw.mpi.size} seconds={elapsed:.6f} "
        f"rss_start_mib={samples[0, 1:].sum():.6f} rss_end_mib={samples[-1, 1:].sum():.6f} "
        f"growth_mib={growth.sum():.6f} late_slope_mib_per_step={slopes.sum():.9f} "
        f"max_rank_growth_mib={growth.max():.6f} max_rank_slope={slopes.max():.9f} "
        f"Tmin={minimum:.9g} Tmax={maximum:.9g}")
    uw.pprint(f"SUPG_MEMORY_SAMPLES method={method} dim={dim} values={samples.tolist()}")
    assert np.isfinite(samples).all()
    assert -0.05 < minimum and maximum < 1.05
    # Fixed pre-run bounds allow allocator noise but reject sustained growth.
    assert growth.max() < 16.0, growth
    assert slopes.max() < 0.05, slopes
