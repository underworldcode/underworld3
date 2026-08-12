"""The consistent tangent must be finite at a state of rest.

The regression (measured, issue #507): differentiating an unwrapped
strain-rate invariant produces half-integer powers of
(grad v : grad v) whose value or derivative is 0/0 at v = 0, so EVERY
consistent-tangent assembly at a cold start filled the operator with
NaN — surfacing as GAMG's "Computed maximum singular value as zero"
(error 77) on the standard path, the rotated free-slip path, and the
split-node fault path alike. The alpha-blended continuation kernel
inherited it even in its Picard phase (IEEE 0*NaN = NaN). The guard in
``_jacobian_unwrap`` regularises exactly the half-integer-power family
(+1e-36 under the root), leaving the residual and the default Picard
tangent bit-identical.
"""
import numpy as np
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]


def _vp_box(tag, tangent):
    mesh = uw.meshing.UnstructuredSimplexBox(cellSize=0.1)
    x, y = mesh.X
    v = uw.discretisation.MeshVariable(f"V{tag}", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable(f"P{tag}", mesh, 1, degree=0,
                                       continuous=False)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscoPlasticFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    # 1e4 never yields: the problem is secretly linear, which is exactly
    # the corner the yield-homotopy campaigns never visited
    stokes.constitutive_model.Parameters.yield_stress = 1.0e4
    stokes.consistent_jacobian = tangent
    stokes.bodyforce = [0.0, 0.0]
    for wall in ("Left", "Right", "Top", "Bottom"):
        stokes.add_dirichlet_bc((y - 0.5, 0.0), wall)
    stokes.petsc_use_pressure_nullspace = True
    stokes.tolerance = 1e-6
    return stokes


@pytest.mark.parametrize("tangent", [True, "continuation"])
def test_cold_consistent_tangent_is_finite_and_solves(tangent):
    stokes = _vp_box("a" if tangent is True else "b", tangent)
    stokes.solve(verbose=False)          # cold start — used to raise 77

    # and the assembled Jacobian at the rest state is finite
    snes = stokes.snes
    J = snes.getJacobian()[0].copy()
    U = stokes.dm.getGlobalVec()
    U.set(0.0)
    snes.computeJacobian(U, J)
    assert np.isfinite(J.norm()), "cold-state Jacobian contains NaN/Inf"
