"""
MPI regression test for VE_Stokes first-solve deadlock (issue #130).

The bug: the first ``VE_Stokes.solve()`` call on a fresh in-memory mesh
deadlocked at specific ``(np, mesh)`` partition geometries (e.g. np=4
with a 16x8 StructuredQuadBox → 4x2 rank partition). Root cause was
lazy invocation of ``mesh._get_coords_for_basis`` (DMClone +
createInterpolation + globalToLocal collectives) from rank-local code
paths in ``global_evaluate_nd``: ranks whose migrated particles were
all interior skipped the RBF path and never entered the collective,
while ranks with exterior particles did — deadlocking forever.

The fix pre-populates each variable's coordinate cache at the end of
``_BaseMeshVariable.__init__`` (a collective context), so subsequent
rank-local lookups always hit the cache.

This test runs the canonical failure case at np=4, 16x8 and fails fast
via the pytest timeout rather than blocking indefinitely.
"""

import sympy
import pytest
import underworld3 as uw
from underworld3.function import expression


pytestmark = [
    pytest.mark.level_2,
    pytest.mark.tier_a,
    pytest.mark.mpi(min_size=4),
    pytest.mark.timeout(60),
]


@pytest.mark.mpi(min_size=4)
def test_ve_stokes_first_solve_does_not_deadlock():
    """
    First VE_Stokes.solve() must complete under MPI on a partition-sensitive
    mesh (16x8 -> 4x2 rank partition at np=4). Before the fix for issue #130,
    this hung at the first solve indefinitely.
    """
    mesh = uw.meshing.StructuredQuadBox(elementRes=(16, 8))
    v = uw.discretisation.MeshVariable("V", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)

    stokes = uw.systems.Stokes(
        mesh, velocityField=v, pressureField=p,
    )
    stokes.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel(
        stokes.Unknowns, order=2,
    )
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.constitutive_model.Parameters.shear_modulus = 1.0
    stokes.constitutive_model.Parameters.dt_elastic = 0.02

    V_top = expression("V_top", 0.5, "Top BC")
    stokes.add_dirichlet_bc((V_top, 0.0), "Top")
    stokes.add_dirichlet_bc((0.0, 0.0), "Bottom")
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Left")
    stokes.add_dirichlet_bc((sympy.oo, 0.0), "Right")

    stokes.solve(timestep=0.02)

    # Velocity must carry the top BC. If the solve returned without diverging
    # (pre-fix deadlock would hit the pytest timeout) but with zero velocity,
    # something else went wrong.
    import numpy as np
    v_max = float(np.abs(v.data).max()) if v.data.size else 0.0
    gathered = uw.mpi.comm.allgather(v_max)
    assert max(gathered) > 1.0e-6, (
        f"Velocity field is effectively zero after solve; max|v|={gathered}"
    )
