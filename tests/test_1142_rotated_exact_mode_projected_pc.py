"""An exact rotation is projected out of the rotated velocity preconditioner (#963).

With the analytic radial normal on both walls of an annulus, the rotated constraint
leaves the rigid rotation an exact null vector of the rotated velocity block
(|A m| / (|A| |m|) = 1.7e-18). The custom-FMG cycle returns a rounding-level component
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
  a few cycles (without the projection it ran to its cap of 200 on macOS arm64 too, where
  the outer solve happened to converge anyway), and it gives the recorded answer.
- Under the default (facet-averaged) normal the rotation is near-null, not null, and
  nothing is projected.
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

import underworld3 as uw
from underworld3.utilities import custom_mg

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]

RI, RO = 0.5, 1.0
# the velocity L2 of the analytic-normal solve, recorded serially on macOS arm64 before
# the fix (0.019069617599505716) and on hyperion under SkylakeX after it
# (0.01906961745108); the projection changes the answer by less than the solve tolerance
L2_ANALYTIC = 0.0190696176


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
    return s._rotated_freeslip_info, L2


def test_an_exact_rotation_is_projected_out_of_the_velocity_preconditioner():
    info, L2 = _annulus_fmg("1142a", radial=True)
    assert info["ksp_reason"] > 0 and info["velocity_pc"] == "custom-FMG", info
    assert info["velocity_pc_type"] == "mg", info
    # the inner velocity solve converges in a few cycles; without the projection it ran
    # to its cap of 200 on every host (macOS arm64 included), and only whether the outer
    # solve then converged depended on the kernel
    assert max(info["vel_its_last"]) <= 10, info["vel_its_last"]
    assert info["projected_null_modes"] == 1, info
    assert L2 == pytest.approx(L2_ANALYTIC, rel=1.0e-8)


def test_a_near_null_rotation_is_left_to_the_multigrid():
    info, _L2 = _annulus_fmg("1142d", radial=False)
    assert info["ksp_reason"] > 0, info
    assert info["projected_null_modes"] == 0, info


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
    _info, _L2 = _annulus_fmg("1142s", radial=True)
    print(f"STALLPROBE {_info['ksp_reason']} {_info['projected_null_modes']} {_L2!r}", flush=True)
