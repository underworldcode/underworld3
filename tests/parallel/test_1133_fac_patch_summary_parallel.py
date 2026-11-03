"""The FAC patch summary is collective-safe and the same on every rank.

``CustomMGHierarchy.level_patch_rows`` is rank-local, and its form differs
between ranks: a level is None, one ``(owned, subdomain)`` pair, or a list of
pairs, and a rank holding no part of a fault zone carries an empty pair. The
add-on matrix's row 5 driver looped one allreduce over its own rank's blocks.
At np 4 on kaiju the ranks made different numbers of calls, and the run hung
after its solve had finished.

``hierarchy.patch_summary`` is now made once, where the patch is installed, with
one allgather (``custom_mg._fac_patch_summary``). Reading it afterwards makes no
collective call. The tests check two routes:

- the hierarchy's own install: a Stokes box with a fault-zone strip near one
  wall, which at np 2 and 4 leaves ranks with no zone cells;
- the rotated path: UWTN 2026-022's linear fault fixture at 80 rungs (the
  fixture of test_1106), whose structural patch rows do not depend on the
  partition.

At every rank count they assert that every rank holds the identical summary and
that the coarse levels carry no patch. Against the serial baselines, at every rank
count (the finest level's blocks are cut from the rows gathered over the ranks,
each held by one rank, #925):

- the zone strip's patch is one block of exactly 1510 rows;
- the fault's structural patch is one block, its rows test_1106's recorded serial
  counts, and they equal the rank-local record summed over the ranks.

This file carries no ``mpi(min_size=2)`` mark: at np 1 it checks the serial
baseline, in the parallel batches the partition.

    python -m pytest tests/parallel/test_1133_fac_patch_summary_parallel.py
    mpirun -n 2 python -m pytest --with-mpi tests/parallel/test_1133_fac_patch_summary_parallel.py
"""
import json

import numpy as np
import pytest

import underworld3 as uw
from underworld3.utilities.custom_mg import set_custom_fmg
from underworld3.utilities.place_surface import place_fault_ribbon_2d

pytestmark = [pytest.mark.level_2]

# the zone strip's patch rows at np 1 (P2 velocity, 2 components per node)
ZONE_SERIAL_ROWS = 1510
# test_1106's serial counts of the fixture's structural patch rows, per host
SERIAL_PATCH_ROWS = {"macOS arm64, custom PETSc 3.25.0": 4780,
                     "CI Linux x86_64, conda-forge PETSc 3.25.3": 4772}


def _stokes(mesh):
    x, y = mesh.X
    v = uw.discretisation.MeshVariable("v", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("p", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    for wall in ("Bottom", "Top", "Left", "Right"):
        stokes.add_dirichlet_bc((y - 0.5, 0.0), wall)
    stokes.petsc_use_pressure_nullspace = True
    stokes.tolerance = 1e-6
    return stokes


def _same_on_every_rank(summary):
    text = json.dumps(summary, sort_keys=True)
    return len(set(uw.mpi.comm.allgather(text))) == 1


@pytest.mark.tier_b
def test_the_zone_patch_summary_is_the_same_on_every_rank():
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0), maxCoords=(1, 1), cellSize=1 / 8, regular=True,
        qdegree=2, refinement=2)
    levels = base._coarse_level_meshes()[:-1] + [base]
    stokes = _stokes(base)
    cS, cE = base.dm.getHeightStratum(0)
    centroids = np.array([base.dm.computeCellGeometryFVM(c)[1][:2]
                          for c in range(cS, cE)]).reshape(-1, 2)
    zone = (centroids[:, 0] > 0.05) & (centroids[:, 0] < 0.2)
    set_custom_fmg(stokes, levels, field_id=0, fac_zone=zone)
    stokes.solve()
    assert stokes.snes.getConvergedReason() > 0

    h = stokes._custom_mg["hierarchy"]
    summary = h.patch_summary
    assert summary is not None, "the install did not record the patch summary"
    assert _same_on_every_rank(summary), f"rank {uw.mpi.rank}: {summary}"
    assert len(summary) == len(h.level_patch_rows)        # one entry per PCMG level
    assert all(entry is None for entry in summary[:-1]), summary
    finest = summary[-1]
    assert finest is not None, "the zone strip installed no patch"
    held = [rank for rank in finest["block_rows_per_rank"] if rank]
    assert held == [[ZONE_SERIAL_ROWS]], (
        f"np {uw.mpi.size}: zone blocks per rank {finest['block_rows_per_rank']}; "
        f"serial has one block of {ZONE_SERIAL_ROWS} rows (#925)")
    assert finest["owned_rows"] == ZONE_SERIAL_ROWS, finest


@pytest.mark.tier_b
def test_the_structural_patch_summary_is_the_serial_rows_at_any_rank_count():
    a, b = np.array([0.30, 0.45]), np.array([0.70, 0.55])
    n_rungs = 80
    length = float(np.linalg.norm(b - a))
    spacing = length / n_rungs
    trace = a + (np.arange(n_rungs + 1) * spacing)[:, None] * (b - a) / length
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0), maxCoords=(1, 1), cellSize=1 / 8, regular=True,
        qdegree=2, refinement=2)
    levels = base._coarse_level_meshes()[:-1] + [base]
    mesh = place_fault_ribbon_2d(base, [("Fault", trace)], 2 * spacing, margin_rings=1,
                                 clearance=0.7, split=True, mesher="ladder")[0]
    stokes = _stokes(mesh)
    stokes.add_fault_bc(0, boundary="Fault")
    set_custom_fmg(stokes, levels, field_id=0)
    stokes.solve()
    assert stokes._rotated_freeslip_info["converged"]

    h = stokes._custom_mg["hierarchy"]
    summary = h.patch_summary
    assert summary is not None, "the rotated install did not record the patch summary"
    assert _same_on_every_rank(summary), f"rank {uw.mpi.rank}: {summary}"
    assert all(entry is None for entry in summary[:-1]), summary
    finest = summary[-1]
    assert finest is not None, "the finest level has no band patch"
    assert finest["owned_rows"] in SERIAL_PATCH_ROWS.values(), (
        f"np {uw.mpi.size}: {finest['owned_rows']} structural patch rows; serial "
        f"counts {SERIAL_PATCH_ROWS} (a new host records its serial count in test_1106)")
    held = [rank for rank in finest["block_rows_per_rank"] if rank]
    assert len(held) == 1 and len(held[0]) == 1, (
        f"np {uw.mpi.size}: structural blocks per rank {finest['block_rows_per_rank']}; "
        f"serial has one block (#925)")
    # the summary is the rank-local record summed over the ranks
    entry = h.level_patch_rows[-1]
    blocks = entry if isinstance(entry, list) else [entry]
    owned = uw.mpi.comm.allreduce(int(sum(len(own) for own, _sub in blocks)))
    assert finest["owned_rows"] == owned, (finest["owned_rows"], owned)
