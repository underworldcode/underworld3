"""No PETSc error banner from asking an un-set-up field split for its blocks (#827).

A field split has block solvers only once ``PCSetUp`` has run. Asking for them
earlier raises, and PETSc prints a ~38-line "Object is in wrong state" banner to
stderr before the exception reaches Python, so ``except`` cannot hide it and a
healthy solve looks broken. Three paths reached it:

- a viscoelastic solve checked the velocity preconditioner before solving;
- the same check after a plain solve that converged without a linear solve, on
  a solver rebuilt by a ``consistent_jacobian`` change, so the new split was
  never set up;
- the custom-multigrid "still installed?" test, run before each solve, on a
  solver rebuilt by a ``consistent_jacobian`` change (the notch runs' path).

The banner is C-level output on file descriptor 2, so ``capfd`` sees it.
"""
import pytest
import sympy

import underworld3 as uw
from underworld3.utilities._petsc_tools import _fieldsplit_sub_ksps

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]

BANNER = "PETSC ERROR"


def _stokes(cellSize=0.25, refinement=None):
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=cellSize,
        refinement=refinement)
    v = uw.discretisation.MeshVariable("V", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.add_dirichlet_bc((0.0, 0.0), "Bottom")
    stokes.add_dirichlet_bc((1.0, 0.0), "Top")
    return stokes


def test_helper_is_quiet_on_an_unsetup_split(capfd):
    """The helper: [] before set-up, with nothing printed; the blocks after."""
    stokes = _stokes()
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes._setup_pointwise_functions()
    stokes._setup_discretisation()
    stokes._setup_solver()
    pc = stokes.snes.getKSP().getPC()
    assert pc.getType() == "fieldsplit"
    assert _fieldsplit_sub_ksps(pc) == []
    assert BANNER not in capfd.readouterr().err
    stokes.solve()
    assert len(_fieldsplit_sub_ksps(stokes.snes.getKSP().getPC())) == 2


def test_viscoelastic_solve_prints_no_banner(capfd):
    """The #827 reproducer."""
    stokes = _stokes()
    stokes.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel(
        stokes.Unknowns, order=1)
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.constitutive_model.Parameters.shear_modulus = 1.0
    stokes.constitutive_model.Parameters.dt_elastic = sympy.Rational(1, 10)
    stokes.solve(timestep=0.1)
    assert BANNER not in capfd.readouterr().err


def test_warm_solve_without_a_linear_solve_prints_no_banner(capfd):
    """A tangent change rebuilds the solver; the warm solve that follows meets its
    (loose) tolerance at once, so no linear solve sets the new split up, and the
    post-solve velocity-preconditioner check asked it for its blocks."""
    stokes = _stokes()
    stokes.constitutive_model = uw.constitutive_models.ViscoPlasticFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.constitutive_model.Parameters.yield_stress = 0.3
    stokes.consistent_jacobian = False
    stokes.solve()
    capfd.readouterr()
    stokes.petsc_options["snes_atol"] = 1.0e6
    stokes.consistent_jacobian = True
    stokes.solve(zero_init_guess=False)
    assert stokes.snes.getIterationNumber() == 0
    assert BANNER not in capfd.readouterr().err


def test_custom_fmg_rewire_prints_no_banner(capfd):
    """A custom multigrid on the velocity block (an independent, non-nested coarsest
    mesh: the shape of the notch runs' gmsh-tail FMG), then a tangent change. The
    change rebuilds the solver, and the next solve's "is the multigrid still
    installed?" test asked the new, un-set-up split for its blocks."""
    stokes = _stokes(refinement=1)
    stokes.constitutive_model = uw.constitutive_models.ViscoPlasticFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.constitutive_model.Parameters.yield_stress = 0.3
    fine = stokes.mesh
    tail = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.6)
    stokes.set_custom_fmg([tail] + fine._coarse_level_meshes()[:-1], field_id=0)
    stokes.consistent_jacobian = False
    stokes.solve()
    capfd.readouterr()
    stokes.consistent_jacobian = True
    stokes.solve(zero_init_guess=False)
    assert stokes.snes.getConvergedReason() > 0
    assert BANNER not in capfd.readouterr().err
    assert stokes.snes.getKSP().getPC().getFieldSplitSubKSP()[0].getPC().getType() == "mg"
