"""The FAC band patch is the same rows at any rank count (#896).

The finest level's band patch is found by classifying each fine row of the
transfer: an identity row onto a coincident coarse node is background, anything
else is patch. In parallel (``custom_mg._fac_patch_split_parallel``) a rank
could only test coincidence against the coarse nodes it holds, and a row whose
coarse node lived on another rank was counted as patch. A placed fault mesh is
partitioned by the add-on partition, which gathers the fault onto one rank, while
the coarse levels keep the base partition, so most other ranks' rows point at
coarse nodes held elsewhere. Measured on UWTN 2026-022's linear fixture at 320
rungs: 4474 background rows of rank 1's 9410 counted as patch at np 2, a cover of
0.797 against the 0.75 gate, so the patch was declined and the solve took 640
velocity iterations instead of 68. At np 4 the same error added 2742 rows and
the gate passed at 0.721.

The band does not depend on the partition, so this test asserts that the global
count of patch rows on the finest level equals the serial count, and that the
patch is installed. It runs the same fixture at 80 rungs.

This file carries no ``mpi(min_size=2)`` mark: at np 1 it checks the serial
baseline, in the parallel batches the partition.

    python -m pytest tests/parallel/test_1106_fac_cover_partition_parallel.py
    mpirun -n 2 python -m pytest --with-mpi tests/parallel/test_1106_fac_cover_partition_parallel.py
"""
import numpy as np
import pytest

import underworld3 as uw
from underworld3.utilities.custom_mg import set_custom_fmg
from underworld3.utilities.place_surface import place_fault_ribbon_2d

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]

A, B = np.array([0.30, 0.45]), np.array([0.70, 0.55])
N_RUNGS = 80
# Patch rows of the finest level in serial (the band), measured on the fixture
# below; the parallel split must find the same rows. Before the fix: 5798 at np 2.
SERIAL_PATCH_ROWS = 4248


def _finest_patch_rows():
    """Solve the linear fault fixture (H0: welded 1/8, 1/16, 1/32 base levels under
    the placed ladder) and return the finest level's patch rows, summed over the
    ranks, or None when no patch was installed."""
    length = float(np.linalg.norm(B - A))
    spacing = length / N_RUNGS
    trace = A + (np.arange(N_RUNGS + 1) * spacing)[:, None] * (B - A) / length
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0), maxCoords=(1, 1), cellSize=1 / 8, regular=False,
        qdegree=2, refinement=2)
    levels = base._coarse_level_meshes()[:-1] + [base]
    mesh = place_fault_ribbon_2d(base, [("Fault", trace)], 2 * spacing, margin_rings=1,
                                 clearance=0.7, split=True, mesher="ladder")[0]
    x, y = mesh.X
    v = uw.discretisation.MeshVariable("v", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("p", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    for wall in ("Bottom", "Top", "Left", "Right"):
        stokes.add_dirichlet_bc((y - 0.5, 0.0), wall)
    stokes.add_fault_bc(0, boundary="Fault")
    stokes.petsc_use_pressure_nullspace = True
    stokes.tolerance = 1e-6
    set_custom_fmg(stokes, levels, field_id=0)
    stokes.solve()
    assert stokes._rotated_freeslip_info["converged"]
    patch = stokes._custom_mg["hierarchy"].level_patch_rows[-1]
    installed = uw.mpi.comm.allreduce(patch is not None, op=__import__("mpi4py").MPI.LAND)
    if not installed:
        return None
    owned = patch[0] if isinstance(patch, tuple) else np.concatenate([b[0] for b in patch])
    return uw.mpi.comm.allreduce(int(len(owned)))


def test_the_band_patch_is_the_same_rows_at_any_rank_count():
    rows = _finest_patch_rows()
    assert rows is not None, f"np {uw.mpi.size}: the finest level has no band patch (#896)"
    assert rows == SERIAL_PATCH_ROWS, (
        f"np {uw.mpi.size}: {rows} patch rows on the finest level, {SERIAL_PATCH_ROWS} "
        f"in serial (#896: background rows whose coarse node is on another rank)")
