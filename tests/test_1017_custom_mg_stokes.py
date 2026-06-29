"""Layer-1 generalized FMG hierarchy on the STOKES velocity block (Phase-1 Step 1).

Custom-built prolongations (barycentric / RBF) drive geometric multigrid on the
velocity sub-block of the saddle-point solver, via set_custom_fmg(field_id=0).
Validated on SolCx (eta_B=1e6): the velocity block converges in a handful of MG
iterations (vs ~200 for GAMG) and the solution matches a GAMG reference.

The hard part (see custom_mg._install_velocity_block_transfers): the velocity
sub-PC is unreachable until the monolithic Jacobian is assembled, so the install
forces a Jacobian assembly, reaches the velocity sub-PC, and rebuilds a fresh PCMG
from our supplied prolongations.
"""
import numpy as np
import pytest
import underworld3 as uw
from underworld3.function import analytic as A
from underworld3.utilities import custom_mg

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]


def _wrap(dm, m0):
    return uw.discretisation.Mesh(
        dm.clone(), simplex=True,
        coordinate_system_type=m0.CoordinateSystem.coordinate_type,
        qdegree=3, boundaries=m0.boundaries)


def _hierarchy():
    """cellSize 0.25 base, two uniform refinements -> 3 nested levels."""
    m0 = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.25, regular=True, qdegree=3)
    dm0 = m0.dm
    dm1 = dm0.refine()
    dm2 = dm1.refine()
    meshes = [_wrap(dm0, m0), _wrap(dm1, m0), _wrap(dm2, m0)]
    return meshes[:-1], meshes[-1]


def _walls(s):
    s.add_dirichlet_bc((0.0, None), "Left")
    s.add_dirichlet_bc((0.0, None), "Right")
    s.add_dirichlet_bc((None, 0.0), "Bottom")
    s.add_dirichlet_bc((None, 0.0), "Top")
    s.petsc_use_pressure_nullspace = True
    s.petsc_options["snes_type"] = "ksponly"


def _coarse_factory(mesh):
    """Same-discretisation coarse-level Stokes (only its constrained section is
    read by CustomMGHierarchy.build to derive the BC-reduced DOF map)."""
    s = uw.systems.Stokes(mesh)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    _walls(s)
    return s


def _solcx(mesh, eta_B=1e6):
    sol = A.SolCx(mesh, eta_A=1.0, eta_B=eta_B, x_c=0.5, n=1)
    s = uw.systems.Stokes(mesh)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = sol.fn_viscosity
    s.saddle_preconditioner = 1.0 / sol.fn_viscosity
    s.bodyforce = sol.fn_bodyforce
    _walls(s)
    s.tolerance = 1e-8
    return s, sol


def _vel_ksp(s):
    return s.snes.getKSP().getPC().getFieldSplitSubKSP()[0]


def test_custom_fmg_velocity_block_barycentric():
    """Custom barycentric P drives geometric MG on the SolCx velocity block,
    converging in few iterations and matching a GAMG reference solution."""
    coarse, fine = _hierarchy()

    # GAMG reference
    sg, solg = _solcx(fine)
    sg.preconditioner = "gamg"
    sg.solve()
    verr_g = solg.velocity_error(sg.u)
    iters_g = _vel_ksp(sg).getIterationNumber()

    # custom-P geometric MG on the velocity block
    s, sol = _solcx(fine)
    custom_mg.set_custom_fmg(s, coarse, level_solver_factory=_coarse_factory,
                             builder="barycentric", field_id=0)
    s.solve()
    vksp = _vel_ksp(s)

    assert s.snes.getConvergedReason() > 0
    assert vksp.getPC().getType() == "mg"
    assert vksp.getPC().getMGLevels() == len(coarse) + 1
    # geometric MG crushes GAMG on the eta-jump velocity block
    assert vksp.getIterationNumber() <= 15
    assert vksp.getIterationNumber() < iters_g
    # correct solution: matches analytic to GAMG's accuracy, and matches GAMG itself
    verr = sol.velocity_error(s.u)
    assert verr < 2.0 * verr_g + 1e-6
    rel = np.linalg.norm(s.u.data - sg.u.data) / (np.linalg.norm(sg.u.data) + 1e-30)
    assert rel < 1e-4


def test_custom_fmg_velocity_block_rbf():
    """The RBF builder must also drive a converging velocity-block MG solve."""
    coarse, fine = _hierarchy()
    s, sol = _solcx(fine)
    custom_mg.set_custom_fmg(s, coarse, level_solver_factory=_coarse_factory,
                             builder="rbf", field_id=0)
    s.solve()
    vksp = _vel_ksp(s)
    assert s.snes.getConvergedReason() > 0
    assert vksp.getPC().getType() == "mg"
    assert vksp.getPC().getMGLevels() == len(coarse) + 1
    assert vksp.getIterationNumber() <= 25
    assert sol.velocity_error(s.u) < 1e-2
