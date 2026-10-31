"""GAMG as the coarse-level solve of the geometric multigrid, asked for (#901).

For a coarsest mesh too large for every rank to factor, the coarse level can be
solved by GAMG instead of the gathered LU: ``set_custom_fmg(...,
coarse_solve="gamg")``. It is never a default and never automatic, so nothing
else exercises it. This test keeps it from rotting, on the rotated route (an
annulus with rotated free slip on both walls, where the rigid-body near-null
space is carried into the wall frame) and on the standard route (a no-slip
Stokes box). Contract: the live coarse PC is the GAMG solve, the answer is the
gathered LU's, and the velocity and Schur iterations stay within those of the
same solve with the automatic coarse solve (they move with the partition, so
the bound is taken from the run, not recorded).

    mpirun -n 2 python -m pytest --with-mpi tests/parallel/test_1114_mg_coarse_gamg_parallel.py
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw
from underworld3.utilities import custom_mg

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_2, pytest.mark.tier_b,
              pytest.mark.timeout(600)]

# The gathered-LU answer at np 2 (redundant LU; on the rotated route the SVD
# coarse solve gave it too).
VRMS = {"rotated": 4.7900648436, "standard": 1.597852740266e-01}


def _stokes(route, coarse_solve):
    if route == "rotated":
        mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.4,
                                  refinement=2, qdegree=3)
    else:
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.25, refinement=2,
            regular=True, qdegree=3)
    tag = f"{route}{coarse_solve}"
    v = uw.discretisation.MeshVariable(f"V1114{tag}", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable(f"P1114{tag}", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    X = mesh.X
    if route == "rotated":
        r = sympy.sqrt(X[0] ** 2 + X[1] ** 2)
        th = sympy.atan2(X[1], X[0])
        stokes.bodyforce = (1.0e3 * sympy.cos(4 * th) * sympy.sin(sympy.pi * (r - 0.5) / 0.5)
                            * sympy.Matrix([X[0] / r, X[1] / r]))
        for wall in ("Lower", "Upper"):
            stokes.add_rotated_freeslip_bc(0, wall)
    else:
        stokes.bodyforce = sympy.Matrix(
            [0, 1.0e2 * sympy.sin(sympy.pi * X[0]) * sympy.sin(sympy.pi * X[1])])
        for wall in ("Bottom", "Top", "Left", "Right"):
            stokes.add_dirichlet_bc((0.0, 0.0), wall)
    stokes.petsc_use_pressure_nullspace = True
    stokes.tolerance = 1.0e-6
    custom_mg.set_custom_fmg(stokes, mesh._coarse_level_meshes()[:-1], field_id=0,
                             coarse_solve=coarse_solve)
    return mesh, v, stokes


def _solved(route, coarse_solve, seen):
    """Solve; return the coarse PC, the converged flag, the last outer
    iteration's (velocity, Schur) iterations and v_rms."""
    mesh, v, stokes = _stokes(route, coarse_solve)
    stokes.solve(zero_init_guess=True)
    coarse_pc = seen["pc"].getMGCoarseSolve().getPC()
    if route == "rotated":
        info = stokes._rotated_freeslip_info
        converged = info["converged"]
        its = (info["vel_its_last"][-1], info["pres_its_last"][-1])
        assert info["coarse_solve"] == (coarse_solve or info["coarse_solve"]), info
    else:
        converged = stokes.snes.getConvergedReason() > 0
        vel_ksp, pres_ksp = stokes.snes.getKSP().getPC().getFieldSplitSubKSP()
        its = (vel_ksp.getIterationNumber(), pres_ksp.getIterationNumber())
    vol = float(uw.maths.Integral(mesh, sympy.Integer(1)).evaluate())
    vrms = float(np.sqrt(uw.maths.Integral(mesh, v.sym.dot(v.sym)).evaluate() / vol))
    return coarse_pc, converged, its, vrms


@pytest.mark.parametrize("route", ["rotated", "standard"])
def test_gamg_on_the_coarse_level_when_asked_for(monkeypatch, route):
    seen = {}
    configure = custom_mg._configure_pcmg

    def configure_seen(pc, Ps, **kw):
        out = configure(pc, Ps, **kw)
        seen["pc"] = pc
        return out

    monkeypatch.setattr(custom_mg, "_configure_pcmg", configure_seen)
    _, converged_auto, its_auto, vrms_auto = _solved(route, None, seen)
    coarse_pc, converged, its, vrms = _solved(route, "gamg", seen)

    assert converged_auto and converged
    assert coarse_pc.getType() == "python"
    assert isinstance(coarse_pc.getPythonContext(), custom_mg.GamgCoarseSolve)
    assert its[0] <= its_auto[0] and its[1] <= its_auto[1], (its, its_auto)
    for value in (vrms_auto, vrms):
        assert abs(value - VRMS[route]) <= 1.0e-8 * VRMS[route], (vrms_auto, vrms)
