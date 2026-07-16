"""Layer-1 generalized FMG hierarchy across the solver families that consume a mesh.

Custom-built prolongations (barycentric / RBF) drive geometric multigrid via
set_custom_fmg, validated here on each solver-PC topology so the same adapted-mesh
hierarchy works for every solver running on it:
  - STOKES velocity sub-block (set_custom_fmg field_id=0) — SolCx eta_B=1e6;
  - SNES_Vector (Vector_Projection) — top-level vector PC (field_id=None);
  - Stokes_Constrained (free-slip via in-saddle multipliers) — velocity block under
    the grouped [p, h] fieldsplit.

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
    custom_mg.set_custom_fmg(s, coarse, builder="barycentric", field_id=0)
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
    custom_mg.set_custom_fmg(s, coarse, builder="rbf", field_id=0)
    s.solve()
    vksp = _vel_ksp(s)
    assert s.snes.getConvergedReason() > 0
    assert vksp.getPC().getType() == "mg"
    assert vksp.getPC().getMGLevels() == len(coarse) + 1
    assert vksp.getIterationNumber() <= 25
    assert sol.velocity_error(s.u) < 1e-2


def test_custom_fmg_vector_solver():
    """SNES_Vector (Vector_Projection): custom-P on the top-level vector PC
    (field_id=None) — the same scalar/single-field-vector path, ncomp=dim."""
    import sympy
    coarse, fine = _hierarchy()
    x, y = fine.X
    v = uw.discretisation.MeshVariable("Ucp", fine, fine.dim, degree=2)
    proj = uw.systems.Vector_Projection(fine, v)
    proj.uw_function = sympy.Matrix([[sympy.sin(sympy.pi * x) * sympy.cos(sympy.pi * y),
                                      sympy.cos(sympy.pi * x) * sympy.sin(sympy.pi * y)]])
    proj.smoothing = 1.0e-3
    proj.add_dirichlet_bc((0.0, 0.0), "Bottom")
    proj.add_dirichlet_bc((0.0, 0.0), "Top")
    custom_mg.set_custom_fmg(proj, coarse, builder="barycentric")   # field_id=None
    proj.solve()
    assert proj.snes.getKSP().getPC().getType() == "mg"
    assert proj.snes.getKSP().getPC().getMGLevels() == len(coarse) + 1
    assert proj.snes.getConvergedReason() > 0


def test_custom_fmg_stokes_constrained():
    """Stokes_Constrained (free-slip via in-saddle multipliers): custom-P on the
    velocity block under the grouped [p, h] fieldsplit. Velocity must still match
    the analytic SolCx free-slip solution."""
    import sympy
    coarse, fine = _hierarchy()
    sol = A.SolCx(fine, eta_A=1.0, eta_B=1.0e6, x_c=0.5, n=1)
    s = uw.systems.Stokes_Constrained(fine)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = sol.fn_viscosity
    s.bodyforce = sol.fn_bodyforce
    s.add_constraint_bc(0.0, "Left",   normal=sympy.Matrix([[-1.0, 0.0]]))
    s.add_constraint_bc(0.0, "Right",  normal=sympy.Matrix([[1.0, 0.0]]))
    s.add_constraint_bc(0.0, "Bottom", normal=sympy.Matrix([[0.0, -1.0]]))
    s.add_constraint_bc(0.0, "Top",    normal=sympy.Matrix([[0.0, 1.0]]))
    s.petsc_use_pressure_nullspace = True
    s.tolerance = 1.0e-9
    s.petsc_options["snes_type"] = "ksponly"
    custom_mg.set_custom_fmg(s, coarse, builder="barycentric", field_id=0)
    s.solve()
    vksp = _vel_ksp(s)
    assert s.snes.getConvergedReason() > 0
    assert vksp.getPC().getType() == "mg"
    assert vksp.getPC().getMGLevels() == len(coarse) + 1
    assert vksp.getIterationNumber() <= 15
    assert sol.velocity_error(s.u) < 5.0e-3


def test_custom_fmg_stokes_on_sbr_child():
    """Stokes velocity-block custom-P on an SBR-refined CHILD mesh whose bodyforce
    depends on a mesh variable (the Layer-2 adapt-on-top scenario). Regression for
    the operator-not-wired bug: forcing the Jacobian before the fieldsplit left the
    KSP carrying an unassembled operator -> PCSetUp 'Matrix must be set first'
    (err73). Fixed by wiring the assembled Jacobian into the KSP before reaching
    the velocity sub-PC."""
    import sympy
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.25, regular=True,
        refinement=2, qdegree=3)
    child = _wrap(custom_mg.sbr_refine_where(
        base.dm_hierarchy[-1], lambda c: abs(c[0] - 0.5) < 0.15), base)
    coarse = [_wrap(d, base) for d in base.dm_hierarchy]   # static base levels

    Tp = uw.discretisation.MeshVariable("Tp", child, 1, degree=2)   # proxy field
    Tp.data[:, 0] = 1.0
    s = uw.systems.Stokes(child)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    s.saddle_preconditioner = 1.0
    s.bodyforce = sympy.Matrix([[0.0], [1.0e4 * Tp.sym[0]]])   # mesh-variable bodyforce
    s.add_dirichlet_bc((0.0, 0.0), "Bottom"); s.add_dirichlet_bc((0.0, 0.0), "Top")
    s.add_dirichlet_bc((0.0, None), "Left"); s.add_dirichlet_bc((0.0, None), "Right")
    s.petsc_use_pressure_nullspace = True
    s.petsc_options["snes_type"] = "ksponly"
    custom_mg.set_custom_fmg(s, coarse, builder="barycentric", field_id=0)
    s.solve()
    vksp = _vel_ksp(s)
    assert s.snes.getConvergedReason() > 0
    assert vksp.getPC().getType() == "mg"
    assert vksp.getPC().getMGLevels() == len(coarse) + 1
