"""A warm rotated re-solve whose pressure equations are already satisfied (#910).

Started from ``v = 1.05 v_solved`` with the solved pressure, a warm rotated free-slip
solve has a residual in the velocity rows only. The Schur complement's right-hand
side is then round-off (4e-15 against 7e-2 cold), and the pressure sub-solve was
asked to cut that by its relative tolerance. It ran to its 200-iteration cap, each
iteration a velocity solve; or the noise grew past the divergence tolerance and the
next velocity solve saw a NaN, failing the solve at iteration 0 (PC_FAILED, -11).
Which of the two happened depended on the rounding, so the failure looked random:
16 of 26 serial runs had at least one.

- every warm re-solve converges;
- the pressure sub-solve stops at round-off, nowhere near its cap;
- a warm solve that carries a genuine pressure correction still solves for it, at
  mu 1 (1.8e8 times the round-off of its terms) and at mu 1e-4 (1.45e6, the
  smallest genuine ratio measured, just above the 1e6 threshold);
- a cold solve at a large forcing is untouched: the round-off is measured in the
  Schur solve's own terms, not in the outer residual's units. A first fix scaled
  it by the convergence reference, and on this annulus at forcing 1e8 it cut
  genuine sub-solves short, 9.3e-7 off the direct solve where development is
  1.7e-8 (hyperion's review).
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw
from underworld3.utilities import custom_mg

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]

WARM_SOLVES = 4


def _annulus(tag="", forcing=1.0e3, lu=False, mu=1.0):
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.4,
                              refinement=2, qdegree=3)
    v = uw.discretisation.MeshVariable(f"V1157{tag}", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable(f"P1157{tag}", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = mu
    X = mesh.X
    r = sympy.sqrt(X[0] ** 2 + X[1] ** 2)
    th = sympy.atan2(X[1], X[0])
    stokes.bodyforce = (forcing * sympy.cos(4 * th) * sympy.sin(sympy.pi * (r - 0.5) / 0.5)
                        * sympy.Matrix([X[0] / r, X[1] / r]))
    stokes.add_rotated_freeslip_bc(0, "Lower")
    stokes.add_rotated_freeslip_bc(0, "Upper")
    stokes.petsc_use_pressure_nullspace = True
    if lu:
        stokes._rotated_use_lu = True
        stokes.tolerance = 1.0e-11
    else:
        stokes.tolerance = 1.0e-6
        custom_mg.set_custom_fmg(stokes, mesh._coarse_level_meshes()[:-1], field_id=0)
    return v, stokes


def test_warm_resolves_with_satisfied_pressure_converge_without_thrashing():
    v, stokes = _annulus()
    stokes.tolerate_linear_failure = True      # count failures rather than stop at one
    stokes.solve(zero_init_guess=True)
    assert stokes._rotated_freeslip_info["converged"]
    v_solved = np.asarray(v.array).copy()
    records = []
    for _ in range(WARM_SOLVES):
        v.array[...] = 1.05 * v_solved
        stokes.solve(zero_init_guess=False)
        info = stokes._rotated_freeslip_info
        records.append((bool(info["converged"]), int(info["ksp_reason"]),
                        max(its for its in info["pres_its_last"] if its is not None)))
    # development: a converged warm solve ran the pressure sub-solve to 200, and one
    # solve in four or so failed with -11
    assert all(converged for converged, _reason, _its in records), records
    assert max(its for _c, _r, its in records) < 50, records
    # the first warm solve carries a real pressure correction (7 iterations here and
    # on development, 1.8e8 times the round-off of its terms) and must still solve
    # for it; a test that fires after iteration 0 stops it at 2
    assert records[0][2] >= 6, records


def test_a_genuine_warm_correction_at_low_viscosity_is_solved():
    # mu 1e-4: the warm pressure correction is 1.45e6 times the round-off of its
    # terms, the narrowest genuine case measured; a threshold above it skips it
    v, stokes = _annulus("mu", mu=1.0e-4)
    stokes.solve(zero_init_guess=True)
    assert stokes._rotated_freeslip_info["converged"]
    v.array[...] = 1.05 * np.asarray(v.array).copy()
    stokes.solve(zero_init_guess=False)
    info = stokes._rotated_freeslip_info
    assert info["converged"]
    assert stokes._rotated_linear_cache["ctx"]["schur_roundoff"].fired == 0
    assert max(its for its in info["pres_its_last"] if its is not None) >= 5, info["pres_its_last"]


def test_a_cold_solve_at_a_large_forcing_is_untouched():
    v, stokes = _annulus("big", forcing=1.0e8)
    stokes.solve(zero_init_guess=True)
    assert stokes._rotated_freeslip_info["converged"]
    v_lu, direct = _annulus("biglu", forcing=1.0e8, lu=True)
    direct.solve(zero_init_guess=True)
    a, b = np.asarray(v.array).ravel(), np.asarray(v_lu.array).ravel()
    err = np.sqrt(uw.mpi.comm.allreduce(float(((a - b) ** 2).sum()))
                  / uw.mpi.comm.allreduce(float((b ** 2).sum())))
    # the reference-scaled floor: 9.3e-7; development and this test: 1.2e-8 (wombat)
    assert err < 5.0e-8, err
    assert stokes._rotated_linear_cache["ctx"]["schur_roundoff"].fired == 0
