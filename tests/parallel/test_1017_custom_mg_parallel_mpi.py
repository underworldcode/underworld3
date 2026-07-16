"""Parallel (np>1) correctness for custom-P geometric MG (custom_mg).

The nested co-partitioned hierarchy path is parallel-capable: rank-local point
location (ghost-inclusive coarse coords) + global-section reduced numbering +
MPIAIJ transfers. These tests assert that, on >1 ranks, custom-P drives a
converging geometric-MG solve whose result matches a GAMG reference — for both a
scalar Poisson and the Stokes (SolCx) velocity block. The legacy finest-only
path remains serial-only and must still fail loudly.

Run:
    mpirun -n 2 python -m pytest --with-mpi tests/parallel/test_1017_custom_mg_parallel_mpi.py
"""
import numpy as np
import pytest
import underworld3 as uw
from underworld3.function import analytic as A
from underworld3.utilities import custom_mg

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.timeout(180)]


def _wrap(dm, m0, q=2):
    return uw.discretisation.Mesh(
        dm.clone(), simplex=True,
        coordinate_system_type=m0.CoordinateSystem.coordinate_type,
        qdegree=q, boundaries=m0.boundaries)


def _box(cellSize, refinement=None, q=2):
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0., 0.), maxCoords=(1., 1.),
        cellSize=cellSize, regular=True, refinement=refinement, qdegree=q)


def _rel_l2(a, b):
    """Global relative L2 difference of two field-data arrays (rank-reduced)."""
    da = np.asarray(a) - np.asarray(b)
    num = uw.mpi.comm.allreduce(float(np.sum(da**2)))
    den = uw.mpi.comm.allreduce(float(np.sum(np.asarray(b)**2)))
    return (num**0.5) / (den**0.5 + 1e-30)


def _poisson(mesh):
    p = uw.systems.Poisson(mesh)
    p.constitutive_model = uw.constitutive_models.DiffusionModel
    p.constitutive_model.Parameters.diffusivity = 1
    p.f = 0.0
    p.add_dirichlet_bc(0.0, "Bottom"); p.add_dirichlet_bc(1.0, "Top")
    p.petsc_options["ksp_rtol"] = 1e-8; p.petsc_options["ksp_type"] = "cg"
    return p


@pytest.mark.mpi(min_size=2)
def test_parallel_custom_fmg_scalar():
    """Scalar custom-P drives a converging geometric MG solve in parallel and
    matches a GAMG solve of the same problem."""
    assert uw.mpi.size > 1
    m0 = _box(0.25); dm1 = m0.dm.refine(); dm2 = dm1.refine()
    coarse = [_wrap(m0.dm, m0), _wrap(dm1, m0)]; fine = _wrap(dm2, m0)

    g = _poisson(_wrap(dm2, m0)); g.preconditioner = "gamg"; g.solve()
    c = _poisson(fine)
    custom_mg.set_custom_fmg(c, coarse, builder="barycentric")
    c.solve()

    assert c.snes.getKSP().getPC().getType() == "mg"
    assert c.snes.getConvergedReason() > 0
    assert _rel_l2(c.Unknowns.u.data, g.Unknowns.u.data) < 1e-4


@pytest.mark.mpi(min_size=2)
def test_parallel_custom_fmg_nonnested():
    """NON-NESTED coarse tail in parallel: the coarse mesh is an INDEPENDENT box at
    a different cellSize (partitioned independently of the fine mesh), so a fine
    leaf on rank r can sit in a coarse cell owned by rank s. The co-partitioned
    rank-local builder misses those (zero columns); the "auto" cross-partition path
    all-gathers the coarse cloud so every rank locates its fine nodes against the
    full coarse mesh. Result must converge and match a GAMG reference."""
    assert uw.mpi.size > 1
    fine   = _box(0.07)                       # NO refine hierarchy
    coarse = _box(0.25)                        # independent, non-nested
    g = _poisson(fine); g.preconditioner = "gamg"; g.solve()
    c = _poisson(fine)
    custom_mg.set_custom_fmg(c, [coarse], builder="barycentric")   # cross_partition="auto"
    c.solve()
    assert c.snes.getKSP().getPC().getType() == "mg"
    assert c.snes.getConvergedReason() > 0
    assert _rel_l2(c.Unknowns.u.data, g.Unknowns.u.data) < 1e-4


@pytest.mark.mpi(min_size=2)
def test_parallel_custom_fmg_stokes_velocity_block():
    """Custom-P on the SolCx velocity block converges in few MG iters in parallel
    and matches the analytic velocity to the GAMG reference's accuracy."""
    assert uw.mpi.size > 1
    fine = _box(0.25, refinement=2, q=3)
    coarse = [_wrap(d, fine, q=3) for d in fine.dm_hierarchy[:-1]]

    def solcx(mesh):
        sol = A.SolCx(mesh, eta_A=1.0, eta_B=1e6, x_c=0.5, n=1)
        s = uw.systems.Stokes(mesh)
        s.constitutive_model = uw.constitutive_models.ViscousFlowModel
        s.constitutive_model.Parameters.shear_viscosity_0 = sol.fn_viscosity
        s.saddle_preconditioner = 1.0 / sol.fn_viscosity
        s.bodyforce = sol.fn_bodyforce
        s.add_dirichlet_bc((0.0, None), "Left"); s.add_dirichlet_bc((0.0, None), "Right")
        s.add_dirichlet_bc((None, 0.0), "Bottom"); s.add_dirichlet_bc((None, 0.0), "Top")
        s.petsc_use_pressure_nullspace = True; s.tolerance = 1e-8
        s.petsc_options["snes_type"] = "ksponly"
        return s, sol

    sg, solg = solcx(fine); sg.preconditioner = "gamg"; sg.solve()
    sc, sol = solcx(fine)
    custom_mg.set_custom_fmg(sc, coarse, builder="barycentric", field_id=0)
    sc.solve()
    vksp = sc.snes.getKSP().getPC().getFieldSplitSubKSP()[0]

    assert sc.snes.getConvergedReason() > 0
    assert vksp.getPC().getType() == "mg"
    assert vksp.getPC().getMGLevels() == len(coarse) + 1
    assert vksp.getIterationNumber() <= 15           # geometric MG, not GAMG (~200)
    assert sol.velocity_error(sc.u) < 2.0 * solg.velocity_error(sg.u) + 1e-6


@pytest.mark.mpi(min_size=2)
def test_parallel_custom_fmg_vector():
    """SNES_Vector (Vector_Projection) custom-P drives a top-level vector PCMG in
    parallel (field_id=None, ncomp=dim). Fine mesh has NO native hierarchy, so the
    'mg' PC can only come from our custom-P."""
    import sympy
    assert uw.mpi.size > 1
    m0 = _box(0.25); dm2 = m0.dm.refine().refine()
    coarse = [_wrap(m0.dm, m0), _wrap(m0.dm.refine(), m0)]
    fine = _wrap(dm2, m0)
    x, y = fine.X
    v = uw.discretisation.MeshVariable("Ucpp", fine, fine.dim, degree=2)
    proj = uw.systems.Vector_Projection(fine, v)
    proj.uw_function = sympy.Matrix([[sympy.sin(sympy.pi * x) * sympy.cos(sympy.pi * y),
                                      sympy.cos(sympy.pi * x) * sympy.sin(sympy.pi * y)]])
    proj.smoothing = 1.0e-3
    proj.add_dirichlet_bc((0.0, 0.0), "Bottom")
    proj.add_dirichlet_bc((0.0, 0.0), "Top")
    custom_mg.set_custom_fmg(proj, coarse, builder="barycentric")
    proj.solve()
    assert proj.snes.getKSP().getPC().getType() == "mg"
    assert proj.snes.getConvergedReason() > 0


@pytest.mark.mpi(min_size=2)
@pytest.mark.slow
def test_parallel_custom_fmg_stokes_constrained():
    """Stokes_Constrained (free-slip multipliers, grouped [p,h] split) custom-P on
    the velocity block in parallel — velocity must match the analytic SolCx."""
    import sympy
    assert uw.mpi.size > 1
    m0 = _box(0.25, q=3); dm2 = m0.dm.refine().refine()
    coarse = [_wrap(m0.dm, m0, q=3), _wrap(m0.dm.refine(), m0, q=3)]
    fine = _wrap(dm2, m0, q=3)
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
    vksp = s.snes.getKSP().getPC().getFieldSplitSubKSP()[0]
    assert s.snes.getConvergedReason() > 0
    assert vksp.getPC().getType() == "mg"
    assert sol.velocity_error(s.u) < 5.0e-3


@pytest.mark.mpi(min_size=2)
def test_legacy_path_serial_guard():
    """The legacy finest-only set_custom_mg path is serial-only and must raise
    loudly in parallel (it has no parallel reduced map)."""
    assert uw.mpi.size > 1
    s = _poisson(_box(0.25, refinement=2))
    s.set_custom_mg([_box(0.285), _box(0.142)], kind="barycentric")
    with pytest.raises(NotImplementedError):
        s.solve()
