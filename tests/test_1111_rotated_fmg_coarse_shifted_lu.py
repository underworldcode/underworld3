"""The coarse solve of the rotated custom-P FMG is redundant LU with a zero-pivot shift (#901).

A closed free-slip wall leaves rigid rotations: one on an annulus, and three on
a spherical shell with the radial normal. The Galerkin-coarsened rotated
velocity block inherits them, and the coarse level used to be solved by a dense
SVD of the gathered matrix (#306), whose cost grew with the rank count. On a
614-cell shell bottom level it took 64 s to factor at np 1 and 102 s at np 4.

On every hierarchy measured, the inherited rotations make the coarse operator
near-singular, not singular (lambda_min/lambda_max 2e-5 to 2e-8), and plain
redundant LU gives the SVD's answer at its iteration counts. The rotated route
therefore uses the shared redundant LU, with a nonzero factor shift for the case
that is singular. Contracts:

- the live coarse PC carries the NONZERO shift, and the solve records the
  redundant coarse solve;
- on the annulus and the shell, the answer is the SVD path's, at the
  measured iterations;
- on #306's case, a coarse level that is the fine mesh itself, the rotation is
  an exact null vector of the coarse operator (an exactness property, checked
  on any host), and the solve gives the SVD path's answer to 1e-7. Plain LU
  does not stop at that pivot on this build: it converges 5e-7 off, inside the
  solver tolerance. The shift's effect here is the accuracy, hence the
  live-object check of the shift itself.

The answers and the iteration counts were recorded on gmsh meshes, which
triangulate differently on other hosts. Those assertions are gated on the
mesh's fingerprint (global cells, volume); the rest hold anywhere.

The shell's walls are triangulated as the convex hull of their nodes, so its
refinement cannot fold a cell (#946, test_0001_sphere_refinement_folds). Before,
gmsh could put all four vertices of a coarse cell on the inner wall at this cell
size, and the velocity multigrid converged on one host's mesh and ran to its
iteration cap on every solve on another's.
"""
import contextlib
import faulthandler
import os

import numpy as np
import pytest
import sympy
from petsc4py import PETSc

import underworld3 as uw
from underworld3.utilities import custom_mg, rotated_bc

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]

# The SVD path's answer on each fixture (np 1): on the annulus from development
# fc67b650, on the shell from a dense SVD coarse solve set by hand, which gave
# the shifted LU's value and iterations in 91 s against 10 s. Then the velocity
# / Schur iterations of the last outer iteration, and the fingerprint (global
# cells, volume) of the mesh they were recorded on.
CASES = {
    "annulus": {"vrms": 4.7900648439, "rtol": 1.0e-9, "modes": 1, "its": (4, 7),
                "fingerprint": (640, 2.3561872025)},
    "shell": {"vrms": 1.7753740524, "rtol": 1.0e-8, "modes": 3, "its": (5, 17),
              "fingerprint": (4904, 3.6466889256)},
}
# The forced-singular annulus (cell 0.25, the coarse level a copy of the mesh):
# the SVD path's answer (development fc67b650, np 1).
SINGULAR_VRMS = 4.7839629565
SINGULAR_FINGERPRINT = (116, 2.3518070923)


def _fingerprint(mesh):
    s, e = mesh.dm.getHeightStratum(0)
    return (uw.mpi.comm.allreduce(e - s),
            float(uw.maths.Integral(mesh, sympy.Integer(1)).evaluate()))


@pytest.fixture(autouse=True)
def _watchdog(request):
    """A solve that does not converge stays inside one KSPSolve, which holds the
    GIL, so pytest-timeout cannot stop it (test_1111[shell] ran for hours on a
    folded mesh). faulthandler's watchdog does not need the GIL: after 10 min it
    prints every thread's stack to the terminal's stderr (the captured one is
    lost on exit) and exits."""
    capture = request.config.pluginmanager.getplugin("capturemanager")
    with capture.global_and_fixture_disabled() if capture else contextlib.nullcontext():
        stderr = os.fdopen(os.dup(2), "w")
    faulthandler.dump_traceback_later(600, exit=True, file=stderr)
    yield
    faulthandler.cancel_dump_traceback_later()
    stderr.close()


def _skip_unless_recorded(mesh, recorded):
    cells, volume = _fingerprint(mesh)
    if cells != recorded[0] or abs(volume - recorded[1]) > 1.0e-9:
        pytest.skip(f"this host's gmsh built a different mesh ({cells} cells, volume "
                    f"{volume:.10f}; recorded {recorded}): the recorded answer does not apply")


def _shift_is_live(mg_pc, tmp_path):
    """The NONZERO factor shift on the coarse LU, read off the live KSP's view."""
    path = tmp_path / "coarse_ksp.txt"
    viewer = PETSc.Viewer().createASCII(str(path), comm=mg_pc.getComm())
    mg_pc.getMGCoarseSolve().view(viewer)
    viewer.destroy()
    return "[NONZERO]" in path.read_text()


def _seen_pcs(monkeypatch):
    """Records the velocity multigrid PC that _configure_pcmg sets up."""
    seen = {}
    configure = custom_mg._configure_pcmg

    def configure_seen(pc, Ps, **kw):
        seen["pc"] = pc
        return configure(pc, Ps, **kw)

    monkeypatch.setattr(custom_mg, "_configure_pcmg", configure_seen)
    return seen


def _stokes(mesh, name):
    """Rotated free slip on both walls, isoviscous P2/P1."""
    dim = mesh.dim
    v = uw.discretisation.MeshVariable(f"V{name}", mesh, dim, degree=2)
    p = uw.discretisation.MeshVariable(f"P{name}", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    X = mesh.X
    r = sympy.sqrt(sum(xi ** 2 for xi in X))
    th = sympy.atan2(X[1], X[0])
    radial = sympy.sin(sympy.pi * (r - 0.5) / 0.5)
    shape = sympy.cos(4 * th) * (1 if dim == 2 else X[2] / r)
    stokes.bodyforce = 1.0e3 * shape * radial * sympy.Matrix([xi / r for xi in X])
    # Both fixtures require genuine singular coarse operators, rather than
    # weak rotations produced by facet-averaged wall constraints.
    normal = mesh.CoordinateSystem.unit_e_0
    for wall in ("Lower", "Upper"):
        stokes.add_rotated_freeslip_bc(0, wall, normal=normal)
    stokes.petsc_use_pressure_nullspace = True
    stokes.tolerance = 1.0e-6
    return v, stokes


def _vrms(mesh, v):
    vol = float(uw.maths.Integral(mesh, sympy.Integer(1)).evaluate())
    return float(np.sqrt(uw.maths.Integral(mesh, v.sym.dot(v.sym)).evaluate() / vol))


@pytest.mark.parametrize("kind", ["annulus", "shell"])
def test_the_coarse_solve_is_shifted_lu(kind, monkeypatch, tmp_path):
    case = CASES[kind]
    seen = _seen_pcs(monkeypatch)
    if kind == "annulus":
        mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.4,
                                  refinement=2, qdegree=3)
    else:
        mesh = uw.meshing.SphericalShell(radiusOuter=1.0, radiusInner=0.5, cellSize=0.5,
                                         refinement=1, qdegree=3)
    v, stokes = _stokes(mesh, f"1111{kind}")
    custom_mg.set_custom_fmg(stokes, mesh._coarse_level_meshes()[:-1], field_id=0)
    stokes.solve(zero_init_guess=True)
    info = stokes._rotated_freeslip_info
    assert info["converged"] and info["velocity_pc"] == "custom-FMG", info
    assert stokes._rotated_velocity_null_modes == case["modes"]
    assert info["coarse_solve"] == "redundant", info.get("coarse_solve")
    assert _shift_is_live(seen["pc"], tmp_path)
    vrms = _vrms(mesh, v)
    _skip_unless_recorded(mesh, case["fingerprint"])
    assert abs(vrms - case["vrms"]) <= case["rtol"] * case["vrms"], vrms
    assert (info["vel_its_last"][-1], info["pres_its_last"][-1]) == case["its"], info


def test_a_singular_coarse_level_converges_to_the_svd_answer(monkeypatch, tmp_path):
    """#306's case: the coarse level is a copy of the mesh, so the transfer is
    the identity and the Galerkin coarse operator is the fine rotated operator,
    whose rotation is an exact null vector. The test checks that, the shift on
    the live coarse PC, and that the solve gives the SVD path's answer."""
    seen = _seen_pcs(monkeypatch)
    nullspace = rotated_bc._rotated_nullspace

    def nullspace_seen(solver, Q, normal_rows, *, pressure_null_verified):
        seen["nsp"] = nullspace(solver, Q, normal_rows,
                                pressure_null_verified=pressure_null_verified)
        seen["vel_is"] = solver._subdict["velocity"][0]
        return seen["nsp"]

    monkeypatch.setattr(rotated_bc, "_rotated_nullspace", nullspace_seen)
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.25, qdegree=3)
    twin = uw.discretisation.Mesh(mesh.dm.clone(), simplex=True,
                                  coordinate_system_type=mesh.CoordinateSystem.coordinate_type,
                                  qdegree=3, boundaries=mesh.boundaries)
    v, stokes = _stokes(mesh, "1111twin")
    custom_mg.set_custom_fmg(stokes, [twin], field_id=0)
    stokes.solve(zero_init_guess=True)

    # the case under test: the coarse operator annihilates the rotation
    pc = seen["pc"]
    A = pc.getMGCoarseSolve().getOperators()[0]
    (q,) = custom_mg._restrict_null_modes(
        pc, rotated_bc._velocity_null_modes(seen["nsp"], seen["vel_is"]))
    Aq = A.createVecLeft()
    A.mult(q, Aq)
    assert Aq.norm() <= 1.0e-10 * A.norm(), Aq.norm() / A.norm()

    info = stokes._rotated_freeslip_info
    assert info["converged"] and info["coarse_solve"] == "redundant", info
    assert _shift_is_live(pc, tmp_path)
    vrms = _vrms(mesh, v)
    _skip_unless_recorded(mesh, SINGULAR_FINGERPRINT)
    assert abs(vrms - SINGULAR_VRMS) <= 1.0e-7 * SINGULAR_VRMS, vrms
