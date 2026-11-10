"""An exact rotation is projected out of the rotated velocity preconditioner (#963).

With the analytic radial normal on both walls of an annulus, the rotated constraint
leaves the rigid rotation an exact null vector of the rotated velocity block: its
Jacobi-scaled residual, max |(D^-1 A m)_i| / max |m_i|, is at round-off. The custom-FMG cycle returns a rounding-level component
along it in each preconditioned direction. The KSP's null space removes the mode from
the right-hand side and the solution, not from those directions, so the inner FGMRES
floored near 1e-7 against its 3.3e-11 and the outer FGMRES restarted it until the CI
supervisor killed the job (test_1064's annulus_fmg case, every CI run from eddab707).
Whether rounding put the floor above the tolerance depended on the OpenBLAS kernel
(hyperion, Linux): SkylakeX finished, Zen and Haswell stalled at np 2, Nehalem and
Sandybridge at np 1, Core2 and Prescott at both.

The velocity preconditioner is now ``z = P M (P r)`` with ``P`` projecting out the exact
modes, and the answer is unchanged to 13 digits.

- The analytic-normal annulus projects one mode, its inner velocity solve converges in
  a few cycles, and it gives the recorded answer. Without the projection the inner solve
  ran to its cap of 200 on macOS arm64, where the outer solve happened to converge anyway,
  and on most x86 kernels; under SkylakeX it took 4 (hyperion). So this bound catches a
  regression on arm64 and those kernels, and the forced-kernel child below catches it on
  any x86 host.
- These wrapped meshes declare no rotation, so under the default (facet-averaged)
  normal the rotation takes the constraint test and fails it: nothing is admitted or
  projected.
- Whether a mode is exact is a local measure, each row scaled by its own diagonal
  (#964, round 3). The scale it replaced, |A m| / (|A|_F |m|), took its size from the
  stiffest cells: on an annulus whose walls slide along their circles, a viscosity
  contrast of 1e5 classed the faceted rotation as exact, projected it, and the solve
  took 2 to 4 times as long. The slid annulus pins the threshold from both sides, at a
  contrast of 1 and of 1e6.
- The threshold follows the tolerance, max(1e-10, min(1e-6, 100 tol)): what projecting
  a near-null mode costs grows as the tolerance tightens (#964, round 4). At tol 1e-9 a
  mode at 6.2e-8 is projected and one at 1.8e-7 is not, and a re-solve at a new
  tolerance re-decides.
- The projection acts on both sides of the multigrid and removes every mode, after
  orthonormalising them.
- Destroying the solve's workspace destroys the multigrid the projection wraps.
- On x86, the same solve under ``OPENBLAS_CORETYPE=Nehalem``, a kernel that stalled at
  np 1 before the fix, finishes in a child process. macOS arm64 cannot force an x86
  kernel and skips it.
"""
import os
import platform
import subprocess
import sys

import numpy as np
import pytest
import sympy
from petsc4py import PETSc

import underworld3 as uw
from underworld3.utilities import custom_mg, rotated_bc
from _rotated_residual_reference import _assert_original_constrained_equations

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]

RI, RO = 0.5, 1.0
# the velocity L2 of the analytic-normal solve: 0.0190696175990617 on macOS arm64 and
# 0.01906961745108 on Linux (hyperion, every kernel), whose gmsh builds a slightly
# different annulus; the band is centred between them
L2_ANALYTIC = 0.019069617525


def _wrap(dm, m0):
    return uw.discretisation.Mesh(dm.clone(), simplex=True, qdegree=3,
                                  coordinate_system_type=m0.CoordinateSystem.coordinate_type,
                                  boundaries=m0.boundaries)


def _annulus_fmg(name, radial):
    """test_1064's annulus_fmg case: a nested three-level annulus, rotated free slip on
    both walls, the custom FMG on the velocity block, tolerance 1e-9."""
    m0 = uw.meshing.Annulus(radiusInner=RI, radiusOuter=RO, cellSize=0.2, qdegree=3)
    dm1 = m0.dm.refine()
    dm2 = dm1.refine()
    coarse = [m0, _wrap(dm1, m0)]
    fine = _wrap(dm2, m0)
    x, y = fine.X
    r = sympy.sqrt(x ** 2 + y ** 2)
    th = sympy.atan2(y, x)
    v = uw.discretisation.MeshVariable(f"V{name}", fine, 2, degree=2, continuous=True)
    p = uw.discretisation.MeshVariable(f"P{name}", fine, 1, degree=1, continuous=True)
    s = uw.systems.Stokes(fine, velocityField=v, pressureField=p)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    s.bodyforce = sympy.Matrix([[x / r * sympy.cos(4 * th) * (r - RI) * (RO - r) * 40.0,
                                 y / r * sympy.cos(4 * th) * (r - RI) * (RO - r) * 40.0]])
    nhat = sympy.Matrix([[x / r, y / r]]) if radial else None
    s.add_rotated_freeslip_bc(0, "Lower", normal=nhat)
    s.add_rotated_freeslip_bc(0, "Upper", normal=nhat)
    s.tolerance = 1e-9
    s.saddle_preconditioner = 1.0
    s.petsc_use_pressure_nullspace = True
    s.petsc_options["snes_type"] = "ksponly"
    custom_mg.set_custom_fmg(s, coarse, builder="barycentric", field_id=0)
    s.solve()
    L2 = float(np.sqrt(uw.maths.Integral(fine, v.sym.dot(v.sym)).evaluate()))
    return s._rotated_freeslip_info, L2, s


def test_an_exact_rotation_is_projected_out_of_the_velocity_preconditioner():
    info, L2, _s = _annulus_fmg("1142a", radial=True)
    assert info["ksp_reason"] > 0 and info["velocity_pc"] == "custom-FMG", info
    assert info["velocity_pc_type"] == "mg", info
    # the inner velocity solve converges in a few cycles; without the projection it ran
    # to its cap of 200 on macOS arm64 and most x86 kernels (4 under SkylakeX), and only
    # whether the outer solve then converged depended on the kernel
    assert max(info["vel_its_last"]) <= 10, info["vel_its_last"]
    assert info["projected_null_modes"] == 1, info
    # exact to round-off, seven orders below the 1e-6 at which a mode is projected
    assert max(info["null_mode_ratios"]) < 1.0e-13, info["null_mode_ratios"]
    assert L2 == pytest.approx(L2_ANALYTIC, rel=1.0e-8)


def test_an_undeclared_rotation_under_the_default_normal_is_neither_admitted_nor_projected():
    # the wrapped meshes declare no rotation, so the facet-averaged normal is tested and
    # the rotation fails the 1e-8 constraint test: nothing is admitted or projected. The
    # factory annulus declares its rotation, and under this normal it is admitted and,
    # its edges being equal, exact and projected (the next test's slide 0)
    info, _L2, _s = _annulus_fmg("1142d", radial=False)
    assert info["ksp_reason"] > 0, info
    assert info["projected_null_modes"] == 0 and info["null_mode_ratios"] == [], info


def test_the_projection_releases_the_multigrid_it_wraps():
    # the sub-KSP gives up the multigrid when the projection takes its place, so the
    # projection must destroy it: at np 2, where petsc4py defers destruction, the whole
    # hierarchy outlived every solve, 19 MB an adjoint solve (hyperion)
    _info, _L2, s = _annulus_fmg("1142r", radial=True)
    ctx = s._rotated_linear_cache["ctx"]
    inner = ctx["pc"].getFieldSplitSubKSP()[0].getPC().getPythonContext().inner
    assert inner.handle
    rotated_bc._destroy_rotated_ksp_ctx(ctx)
    assert not inner.handle


def _slid_annulus_fmg(name, slide, contrast, tolerance=1.0e-5, solve=True):
    """The factory annulus on its two refinements with the custom FMG, every vertex slid
    along its circle by ``slide * sin(3 theta)`` as an MMPDE mover slides it, and the
    viscosity 1 at both walls and ``contrast`` mid-shell. The walls stay circles, so the
    declared rotation stays exempt under the default normal; the uneven wall edges leave
    it near-null, by an amount set by ``slide``."""
    mesh = uw.meshing.Annulus(radiusInner=RI, radiusOuter=RO, cellSize=0.2, refinement=2,
                              qdegree=3)
    if slide:
        X = np.array(mesh.X.coords)
        r = np.linalg.norm(X, axis=1)
        th = np.arctan2(X[:, 1], X[:, 0])
        th = th + slide * np.sin(3.0 * th)
        mesh.deform(np.column_stack([r * np.cos(th), r * np.sin(th)]))
    x, y = mesh.X
    r = sympy.sqrt(x ** 2 + y ** 2)
    v = uw.discretisation.MeshVariable(f"V{name}", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable(f"P{name}", mesh, 1, degree=1)
    s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = sympy.exp(
        sympy.log(contrast) * sympy.sin(sympy.pi * (r - RI) / (RO - RI)) ** 2)
    s.bodyforce = (sympy.cos(3 * sympy.atan2(y, x)) * sympy.sin(sympy.pi * (r - RI) / (RO - RI))
                   * sympy.Matrix([x / r, y / r]).T)
    for wall in ("Lower", "Upper"):
        s.add_rotated_freeslip_bc(0, wall)
    s.petsc_use_pressure_nullspace = True
    s.tolerance = tolerance
    custom_mg.set_custom_fmg(s, mesh._coarse_level_meshes()[:-1], field_id=0)
    if not solve:
        return s
    s.solve()
    info = s._rotated_freeslip_info
    assert info["converged"], info
    _assert_original_constrained_equations(s, info)
    return info


# Physical weak rotations must remain in the equations, including under a
# large viscosity contrast. The original residual is checked independently.
SLID = [(0.0, 1.0), (1.0e-4, 1.0), (3.0e-4, 1.0),
        (1.0e-4, 1.0e6), (3.0e-4, 1.0e6), (0.1, 1.0e6)]


@pytest.mark.parametrize("slide, contrast", SLID,
                         ids=[f"slide{a:g}-contrast{c:g}" for a, c in SLID])
def test_physical_weak_rotations_remain_under_viscosity_contrast(slide, contrast):
    info = _slid_annulus_fmg(f"1142v{SLID.index((slide, contrast))}", slide, contrast)
    assert info["ksp_reason"] > 0 and info["velocity_pc"] == "custom-FMG", info
    assert info["declared_modes_exempt"] == [True], info
    assert info["rotation_gauge"]["orthonormal"] == 0
    assert info["null_mode_ratios"] == []
    assert info["projected_null_modes"] == 0


def test_the_threshold_follows_the_tolerance():
    threshold = rotated_bc._projection_threshold
    assert threshold(1.0e-5) == 1.0e-6          # capped
    assert threshold(1.0e-9) == pytest.approx(1.0e-7, rel=1.0e-12)
    assert threshold(1.0e-11) == pytest.approx(1.0e-9, rel=1.0e-12)
    assert threshold(1.0e-14) == 1.0e-10        # floored: a mode exact to rounding (4e-11)


@pytest.mark.parametrize("slide", [1.0e-5, 3.0e-5])
def test_a_tight_tolerance_retains_physical_weak_rotations(slide):
    info = _slid_annulus_fmg(f"1142t{slide:g}", slide, 1.0, tolerance=1.0e-9)
    assert info["ksp_reason"] > 0, info
    assert info["rotation_gauge"]["orthonormal"] == 0
    assert info["null_mode_ratios"] == []
    assert info["projected_null_modes"] == 0


def test_a_re_solve_at_a_new_tolerance_re_decides_the_projection():
    # tight first: a workspace kept from a loose solve would project this mode at tol
    # 1e-9, where projecting it ran past ten minutes on one rank
    s = _slid_annulus_fmg("1142c", 3.0e-5, 1.0, tolerance=1.0e-9, solve=False)
    s.solve()
    first = dict(s._rotated_freeslip_info)
    assert first["projected_null_modes"] == 0, first["null_mode_ratios"]   # 1.8e-7 > 1e-7
    s.tolerance = 1.0e-5
    # cold: warm from the tight answer it would stop at its first residual check
    s.solve(zero_init_guess=True)
    second = s._rotated_freeslip_info
    # the KSP context is rebuilt for the new tolerance; the geometry is kept
    assert second["rotation_reused"], second
    assert second["null_mode_threshold"] == 1.0e-6
    assert second["projected_null_modes"] == 0, second["null_mode_ratios"]
    _assert_original_constrained_equations(s, second)
    assert second["ksp_reason"] > 0, second


def test_a_converged_re_solve_at_a_new_tolerance_keeps_the_geometry():
    # warm and already inside the looser tolerance, the re-solve builds no KSP. The
    # rotation, the operator and the prolongation stay for the next solve: dropping
    # them and rebuilding the prolongation grew RSS by about 10 MB a re-solve at np 2
    # (hyperion, round 5)
    # A slid wall's approximate rotation can fail the finalized original-
    # residual guard at 1e-9. Such a solve correctly starts cold next time;
    # use the exact geometry to test a genuinely converged automatic warm start.
    s = _slid_annulus_fmg("1142k", 0.0, 1.0, tolerance=1.0e-9, solve=False)
    s.solve()
    assert s._rotated_freeslip_info["converged"] and s.has_solution
    Q = s._rotated_linear_cache["Q"]
    s.tolerance = 1.0e-5
    s.solve()
    early = s._rotated_freeslip_info
    assert early["converged"] and early["nonlinear_iterations"] == 0, early
    assert s._rotated_linear_cache is not None and s._rotated_linear_cache["Q"] is Q
    s.solve(zero_init_guess=True)
    third = s._rotated_freeslip_info
    assert third["rotation_reused"], third
    assert third["null_mode_threshold"] == 1.0e-6
    assert third["projected_null_modes"] == 0, third["null_mode_ratios"]
    _assert_original_constrained_equations(s, third)


class _AddsAMode:
    """An inner preconditioner that returns its input plus the first basis vector, a
    component along a mode, and keeps a copy of what it was given."""

    def __init__(self):
        self.given = None

    def apply(self, pc, x, y):
        self.given = x.copy()
        x.copy(y)
        y.setValue(0, y.getValue(0) + 1.0)
        y.assemble()


def test_the_projection_acts_on_both_sides_and_removes_every_mode():
    n = 6
    A = PETSc.Mat().createAIJ([n, n], nnz=1, comm=PETSc.COMM_SELF)
    for i in range(n):
        A.setValue(i, i, 0.0 if i < 2 else float(i))
    A.assemble()
    basis = []
    for i in range(3):
        e = A.createVecLeft()
        e.setValue(i, 1.0)
        e.assemble()
        basis.append(e)
    # two exact modes of A, not orthogonal: e0 and e0 + e1
    m2 = basis[0].copy()
    m2.axpy(1.0, basis[1])
    modes, ratios = rotated_bc._exact_velocity_null_modes(A, [basis[0].copy(), m2])
    assert len(modes) == 2 and max(ratios) == 0.0, ratios
    gram = np.array([[a.dot(b) for b in modes] for a in modes])
    assert np.allclose(gram, np.eye(2), rtol=0.0, atol=1.0e-15), gram
    spy = _AddsAMode()
    inner = PETSc.PC().createPython(spy, comm=PETSc.COMM_SELF)
    inner.setOperators(A, A)
    shell = PETSc.PC().createPython(rotated_bc._NullModeProjectedPC(inner, modes),
                                    comm=PETSc.COMM_SELF)
    shell.setOperators(A, A)
    shell.setUp()
    r = basis[0].copy()
    r.axpy(2.0, basis[1])
    r.axpy(1.0, basis[2])
    z = r.duplicate()
    shell.apply(r, z)
    # the multigrid is handed r without its mode components ...
    assert np.allclose(spy.given.getArray(), [0, 0, 1, 0, 0, 0], rtol=0.0, atol=1.0e-15)
    # ... and what it adds along a mode is removed from what it returns
    assert np.allclose(z.getArray(), [0, 0, 1, 0, 0, 0], rtol=0.0, atol=1.0e-15)
    shell.destroy()
    for vec in basis + [r, z, spy.given]:
        vec.destroy()
    A.destroy()


@pytest.mark.skipif(platform.machine().lower() not in ("x86_64", "amd64"),
                    reason="forcing an OpenBLAS kernel needs an x86 build")
def test_the_solve_finishes_under_a_kernel_that_stalled():
    env = {k: v for k, v in os.environ.items() if not k.startswith(("OMPI_", "PMI", "PMIX"))}
    env.update(OPENBLAS_CORETYPE="Nehalem", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    try:
        out = subprocess.run([sys.executable, __file__, "stall_probe"], env=env,
                             capture_output=True, text=True, timeout=240)
    except subprocess.TimeoutExpired:
        pytest.fail("the analytic-normal annulus did not finish in 240 s under "
                    "OPENBLAS_CORETYPE=Nehalem: the inner velocity solve has stalled (#963)")
    lines = [ln for ln in out.stdout.splitlines() if ln.startswith("STALLPROBE ")]
    assert out.returncode == 0 and lines, out.stdout[-2000:] + out.stderr[-2000:]
    reason, projected, L2 = lines[-1].split()[1:]
    assert int(reason) > 0 and int(projected) == 1, lines[-1]
    assert float(L2) == pytest.approx(L2_ANALYTIC, rel=1.0e-8)


if __name__ == "__main__":
    # the child of test_the_solve_finishes_under_a_kernel_that_stalled
    assert sys.argv[1:] == ["stall_probe"], sys.argv
    _info, _L2, _s = _annulus_fmg("1142s", radial=True)
    print(f"STALLPROBE {_info['ksp_reason']} {_info['projected_null_modes']} {_L2!r}", flush=True)
