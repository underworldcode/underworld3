"""The preconditioning matrix from the constitutive model's preconditioner flux.

With an upper-convected viscoelastic model the tangent of the stretch terms
carries the stored stress into the velocity block of the Jacobian: the block is
not symmetric, and where dt |sigma*| / eta ~ 1 its symmetric part is
indefinite, which the velocity multigrid cannot be built from.
``preconditioner_flux = "model"`` builds the velocity block of the
preconditioning matrix from the model's stretch-free flux instead, and leaves
the Jacobian -- so the Newton step and the converged answer -- unchanged.

One step of planar extension from a stressed state, every Dirichlet value the
extension field: the default preconditioning matrix is the Jacobian; with the
model's flux its velocity block is symmetric while the Jacobian's is not, and
the solution is the default's.
"""

import numpy as np
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]

DT, RATE, SIGMA = 0.1, 1.0, 2.0


def _extension(preconditioner_flux):
    uw.reset_default_model()
    mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4), minCoords=(-1, -1), maxCoords=(1, 1))
    x, y = mesh.X
    tag = preconditioner_flux[:3]
    v = uw.discretisation.MeshVariable(f"U_pcf{tag}", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable(f"P_pcf{tag}", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.stress_transport = "eulerian"
    stokes.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel(
        stokes.Unknowns, order=1, integrator="etd", objective_rate="upper_convected",
        convected_step="deformation")
    parameters = stokes.constitutive_model.Parameters
    parameters.shear_viscosity_0 = 1.0
    parameters.shear_modulus = 1.0
    parameters.dt_elastic = DT
    extension = (RATE * x, -RATE * y)
    for boundary in ("Left", "Top", "Bottom"):
        stokes.add_dirichlet_bc(extension, boundary)
    stokes.tolerance = 1.0e-8
    stokes.petsc_options["snes_type"] = "newtonls"
    stokes.preconditioner_flux = preconditioner_flux
    X = np.asarray(v.coords)
    v.data[:, 0], v.data[:, 1] = RATE * X[:, 0], -RATE * X[:, 1]
    # a uniform stored stress, so the stretch tangent is not small
    stokes.DFDt.set_initial_history([SIGMA])
    stokes.solve(timestep=DT, zero_init_guess=False)
    assert stokes.snes.getConvergedReason() > 0

    J, P = stokes.snes.getJacobian()[:2]
    _, isets, _ = stokes.dm.createFieldDecomposition()
    Ju = J.createSubMatrix(isets[0], isets[0])
    Pu = P.createSubMatrix(isets[0], isets[0])
    velocity = np.asarray(v.array[...]).copy()
    return stokes, Ju, Pu, velocity


def _relative_difference(a, b):
    d = a.copy()
    d.axpy(-1.0, b)
    return d.norm() / a.norm()


def _asymmetry(a):
    return _relative_difference(a, a.copy().transpose())


def test_default_preconditioner_is_the_jacobian():
    stokes, Ju, Pu, _ = _extension("jacobian")
    assert stokes.preconditioner_flux == "jacobian"
    assert _relative_difference(Ju, Pu) == 0.0


def test_model_flux_preconditions_with_a_symmetric_block_and_keeps_the_answer():
    _, Ju0, _, v0 = _extension("jacobian")
    stokes, Ju, Pu, v = _extension("model")
    # the stretch tangent: the Jacobian's velocity block is not symmetric ...
    assert _asymmetry(Ju) > 1.0e-3, _asymmetry(Ju)
    # ... and is the default Jacobian still, up to the two solves' converged
    # iterates (tolerance 1e-8 on different Krylov paths)
    assert _relative_difference(Ju0, Ju) < 1.0e-6, _relative_difference(Ju0, Ju)
    # the preconditioner's is the relaxed viscous block: symmetric
    assert _asymmetry(Pu) < 1.0e-12, _asymmetry(Pu)
    assert _relative_difference(Ju, Pu) > 1.0e-3
    # the preconditioner does not move the converged answer
    dv = np.abs(v - v0).max() / np.abs(v0).max()
    assert dv < 1.0e-6, dv


def test_preconditioner_flux_is_validated_and_recompiles():
    stokes, _, _, _ = _extension("jacobian")
    with pytest.raises(ValueError):
        stokes.preconditioner_flux = "viscous"
    assert stokes.is_setup
    stokes.preconditioner_flux = "model"
    assert not stokes.is_setup
