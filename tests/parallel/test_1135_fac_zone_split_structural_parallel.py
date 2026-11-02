"""A split finest level keeps its structural rows in the FAC patch when a zone is keyed (#921).

The finest level's FAC patch is keyed either on the structural rows (the rows
its transfer cannot represent, found from the transfer) or, when ``fac_zone``
masks are passed, on the masks' cells. With a zone keyed, the structural rows
joined the zone's blocks only under ``UW_FAC_STRUCTURAL=1``, against the
documentation, which said they were always added. On a placed split fault those
rows are the band, the split duplicates and the fill around them, which the
coarse level cannot represent, and whole-level SOR converges them slowly: on the
add-on matrix's row 5 (a TI band keyed beside a split fault, h_b/S = 24) the
velocity solve took 820 iterations, summed over 17 applications, against 94 with
the union.

Now a level that carries a split fault always has its structural rows in the
patch, whatever the zone. A level without a split (a painted or placed band)
keeps #670's default, the zone's blocks alone; that default was measured on a
placed weak plane with no split.

Both tests key a zone strip near the left wall, away from the fault, on UWTN
2026-022's linear fault fixture at 80 rungs (the fixture of test_1106), and
compare with the structural patch the same level gets without a zone:

- split: every structural row is in the zone-keyed patch, and so are the
  strip's rows;
- placed but not split: no structural row is in it, and the structural patch
  is not empty, so a union would have shown.

In parallel each rank compares its own rows: both hierarchies are built on the
one solver, so the global numbering is the same. This file carries no
``mpi(min_size=2)`` mark: at np 1 it checks the serial branch, in the parallel
batches the parallel one.

    python -m pytest tests/parallel/test_1135_fac_zone_split_structural_parallel.py
    mpirun -n 2 python -m pytest --with-mpi tests/parallel/test_1135_fac_zone_split_structural_parallel.py
"""
import numpy as np
import pytest

import underworld3 as uw
from underworld3.utilities.custom_mg import CustomMGHierarchy, set_custom_fmg
from underworld3.utilities.place_surface import place_fault_ribbon_2d

pytestmark = [pytest.mark.level_2]

A, B = np.array([0.30, 0.45]), np.array([0.70, 0.55])
N_RUNGS = 80
# the zone: cells whose centroid lies in this strip, clear of the fault's
# refinement and its halo
STRIP = (0.03, 0.12)


def _patch_rows(entry):
    """Every row of one level's patch on this rank (owned and halo)."""
    if entry is None:
        return np.zeros(0, dtype=np.int64)
    blocks = entry if isinstance(entry, list) else [entry]
    return np.unique(np.concatenate([np.asarray(sub, dtype=np.int64)
                                     for _own, sub in blocks]))


def _structural_and_zone_patches(split):
    """Solve the fixture with the structural patch (no zone), then build the
    zone-keyed hierarchy on the same solver; return this rank's rows of both
    finest-level patches."""
    length = float(np.linalg.norm(B - A))
    spacing = length / N_RUNGS
    trace = A + (np.arange(N_RUNGS + 1) * spacing)[:, None] * (B - A) / length
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0), maxCoords=(1, 1), cellSize=1 / 8, regular=True,
        qdegree=2, refinement=2)
    levels = base._coarse_level_meshes()[:-1] + [base]
    mesh = place_fault_ribbon_2d(base, [("Fault", trace)], 2 * spacing, margin_rings=1,
                                 clearance=0.7, split=split, mesher="ladder")[0]
    x, y = mesh.X
    v = uw.discretisation.MeshVariable("v", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("p", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    for wall in ("Bottom", "Top", "Left", "Right"):
        stokes.add_dirichlet_bc((y - 0.5, 0.0), wall)
    if split:
        stokes.add_fault_bc(0, boundary="Fault")
    stokes.petsc_use_pressure_nullspace = True
    stokes.tolerance = 1e-6
    set_custom_fmg(stokes, levels, field_id=0)
    stokes.solve()
    structural = _patch_rows(stokes._custom_mg["hierarchy"].level_patch_rows[-1])

    cS, cE = mesh.dm.getHeightStratum(0)
    centroids = np.array([mesh.dm.computeCellGeometryFVM(c)[1][:2]
                          for c in range(cS, cE)]).reshape(-1, 2)
    strip = (centroids[:, 0] > STRIP[0]) & (centroids[:, 0] < STRIP[1])
    zoned = CustomMGHierarchy(levels + [mesh], field_id=0, fac_zone=strip)
    zoned.build(stokes)
    return structural, _patch_rows(zoned.level_patch_rows[-1])


@pytest.mark.tier_b
def test_a_split_level_keeps_its_structural_rows_with_a_zone_keyed():
    structural, zoned = _structural_and_zone_patches(split=True)
    assert uw.mpi.comm.allreduce(structural.size) > 0, "the split level has no structural patch"
    missing = np.setdiff1d(structural, zoned).size
    assert uw.mpi.comm.allreduce(missing) == 0, (
        f"rank {uw.mpi.rank}: {missing} of {structural.size} structural rows are not in "
        f"the zone-keyed patch (#921)")
    # and the zone's own rows are there too: a dropped zone would leave the
    # structural patch alone, which also contains every structural row
    zone_only = np.setdiff1d(zoned, structural).size
    assert uw.mpi.comm.allreduce(zone_only) > 0, "the zone strip is not in the patch"


@pytest.mark.tier_b
def test_an_unsplit_level_keys_its_patch_on_the_zone_alone():
    structural, zoned = _structural_and_zone_patches(split=False)
    assert uw.mpi.comm.allreduce(structural.size) > 0, (
        "the placed level has no structural patch, so a union could not show")
    assert uw.mpi.comm.allreduce(zoned.size) > 0, "the zone strip keyed no patch"
    shared = np.intersect1d(structural, zoned).size
    assert uw.mpi.comm.allreduce(shared) == 0, (
        f"rank {uw.mpi.rank}: {shared} structural rows are in the zone-keyed patch of "
        f"a level with no split (#670's default is the zone alone)")
