"""The adjoint operator assembled from the transposed kernels IS K^T.

``adjoint_solve`` no longer transposes the assembled Jacobian: it rewires the
solver to the pointwise kernels of the transposed bilinear form (g0 and g3
transposed on their paired indices, g1 and g2 exchanged) and lets the SNES
assemble. This checks that matrix against the explicit transpose to machine
precision, on a non-symmetric operator and on a nonlinear saddle point with
pressure in the viscosity, so every block and every index order is exercised.
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw
from petsc4py import PETSc


def _adjoint_versus_transpose(solver):
    def assemble():
        dm = solver.dm
        g = dm.getGlobalVec()
        if hasattr(solver, "_gather_fields_to_global"):
            solver._gather_fields_to_global(g)
        else:
            dm.localToGlobal(solver.u.vec, g)
        solver.mesh.update_lvec()
        dm.setAuxiliaryVec(solver.mesh.lvec, None)
        J, P = solver.snes.getJacobian()[:2]
        solver.snes.computeJacobian(g, J, P)
        out = J.copy()
        dm.restoreGlobalVec(g)
        return out

    K = assemble()
    token = solver._install_adjoint_kernels()
    K_adj = assemble()
    solver._uninstall_adjoint_kernels(token)
    Kt = PETSc.Mat()
    K.transpose(Kt)          # a new matrix; the no-argument form transposes in place
    D = K_adj.copy()
    D.axpy(-1.0, Kt)
    S = K.copy()
    S.axpy(-1.0, Kt)
    return D.norm() / K.norm(), S.norm() / K.norm()


@pytest.mark.level_1
@pytest.mark.tier_a
def test_supg_step_adjoint_kernels_give_the_transpose():
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1),
                                             cellSize=1 / 8, qdegree=3)
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
    adv = uw.systems.AdvDiffusion(mesh, u_Field=T, V_fn=sympy.Matrix([[1.0, 0.3]]), theta=1.0)
    adv.constitutive_model = uw.constitutive_models.DiffusionModel
    adv.constitutive_model.Parameters.diffusivity = 0.05
    adv.add_essential_bc(1.0, "Bottom")
    adv.add_essential_bc(0.0, "Top")
    adv.solve(timestep=0.1, zero_init_guess=True)
    error, asymmetry = _adjoint_versus_transpose(adv)
    assert asymmetry > 1e-3, "the operator must be non-symmetric for this to test anything"
    assert error < 1e-12


@pytest.mark.level_1
@pytest.mark.tier_a
def test_nonlinear_stokes_adjoint_kernels_give_the_transpose():
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1),
                                             cellSize=1 / 6, qdegree=3)
    x, y = mesh.X
    v = uw.discretisation.MeshVariable("v", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("p", mesh, 1, degree=1, continuous=True)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    # viscosity depending on the strain rate AND the pressure: every block of
    # the Jacobian is non-trivial, and (u,p) and (p,u) are not transposes of
    # each other
    edot = stokes.constitutive_model.Parameters.strainrate_inv_II if hasattr(
        stokes.constitutive_model.Parameters, "strainrate_inv_II") else None
    E = mesh.vector.strain_tensor(v.sym)
    e2 = (E[0, 0] ** 2 + E[1, 1] ** 2 + 2 * E[0, 1] ** 2) / 2 + uw.maths.functions.vanishing
    stokes.constitutive_model.Parameters.shear_viscosity_0 = (1 + sympy.Max(p.sym[0], 0)) / (1 + sympy.sqrt(e2))
    stokes.bodyforce = sympy.Matrix([0, -sympy.sin(3 * x)])
    stokes.add_essential_bc((0.0, 0.0), "Bottom")
    stokes.add_essential_bc((0.5, None), "Left")
    stokes.add_essential_bc((-0.5, None), "Right")
    stokes.solve(zero_init_guess=True)
    error, asymmetry = _adjoint_versus_transpose(stokes)
    assert asymmetry > 1e-3
    assert error < 1e-12
