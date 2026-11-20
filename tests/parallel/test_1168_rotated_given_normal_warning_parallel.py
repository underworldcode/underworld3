"""A curved given normal on a closed rotated free-slip domain is reported (#1168).

When every exterior boundary facet has its normal velocity fixed (a rotated
free-slip wall, or an essential condition on every velocity component), the
continuity row tested with a constant pressure is a sum over the boundary of
the facet-weighted normal flux. A given normal that varies along a faceted
wall differs from the facet-weighted normal at each node, so that row holds a
weighted sum of the wall slip fixed, with the mean pressure as its
multiplier. On a manufactured free-slip solution on an elliptical annulus
(#1168) the velocity error with the exact ellipse normal stayed at 63 to 67 %
under refinement, while the default normal converged at rate 2.1.

This is reported, not changed: the solve's numbers are not touched. The
verdict is recorded on the solver, the same on every rank
(``solver._given_normal_extra_constraint``), and the warning is printed once,
on rank 0.

The cases: a curved given normal on a closed annulus warns, also with a
no-slip inner wall (it fixes the normal velocity too); the default normal does
not; a constant normal does not; a curved given normal with an open (natural)
boundary does not, because the flux there is free.
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b,
              pytest.mark.timeout(300, method="thread")]

WARNING = "constant-pressure continuity row"


def _stokes(mesh, tag, nullspace):
    v = uw.discretisation.MeshVariable(f"V1168{tag}", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable(f"P1168{tag}", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1
    stokes.petsc_use_pressure_nullspace = nullspace
    stokes.tolerance = 1.0e-6
    return stokes


def _annulus():
    return uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.25)


def _radial(mesh):
    x, y = mesh.X
    r = sympy.sqrt(x**2 + y**2)
    return sympy.Matrix([[x / r, y / r]])


def _load(mesh):
    x, y = mesh.X
    r = sympy.sqrt(x**2 + y**2)
    return sympy.Matrix([x / r, y / r]) * sympy.cos(4 * sympy.atan2(y, x))


def _solve_and_read(stokes, capsys):
    stokes.solve()
    out = capsys.readouterr().out
    verdicts = uw.mpi.comm.allgather(bool(getattr(stokes, "_given_normal_extra_constraint",
                                                  False)))
    assert len(set(verdicts)) == 1, f"the verdict differs between ranks: {verdicts}"
    return verdicts[0], out


def test_a_curved_given_normal_on_a_closed_annulus_warns(capsys):
    mesh = _annulus()
    stokes = _stokes(mesh, "a", nullspace=True)
    stokes.bodyforce = _load(mesh)
    stokes.add_rotated_freeslip_bc(0, "Upper", normal=_radial(mesh))
    stokes.add_rotated_freeslip_bc(0, "Lower", normal=_radial(mesh))
    flagged, out = _solve_and_read(stokes, capsys)
    assert flagged
    if uw.mpi.rank == 0:
        assert WARNING in out and "#1168" in out and "normal=None" in out
        assert out.count(WARNING) == 1


def test_a_no_slip_inner_wall_still_fixes_the_flux_and_warns(capsys):
    mesh = _annulus()
    stokes = _stokes(mesh, "b", nullspace=True)
    stokes.bodyforce = _load(mesh)
    stokes.add_rotated_freeslip_bc(0, "Upper", normal=_radial(mesh))
    stokes.add_dirichlet_bc((0.0, 0.0), "Lower")
    flagged, _ = _solve_and_read(stokes, capsys)
    assert flagged


def test_the_default_normal_does_not_warn(capsys):
    mesh = _annulus()
    stokes = _stokes(mesh, "c", nullspace=True)
    stokes.bodyforce = _load(mesh)
    stokes.add_rotated_freeslip_bc(0, "Upper")
    stokes.add_rotated_freeslip_bc(0, "Lower")
    flagged, out = _solve_and_read(stokes, capsys)
    assert not flagged
    assert WARNING not in out


def test_a_constant_normal_does_not_warn(capsys):
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=0.25)
    stokes = _stokes(mesh, "d", nullspace=True)
    x, y = mesh.X
    stokes.bodyforce = sympy.Matrix([0, sympy.cos(sympy.pi * x)])
    stokes.add_rotated_freeslip_bc(0, "Top", normal=np.array([0.0, 1.0]))
    stokes.add_rotated_freeslip_bc(0, "Bottom", normal=np.array([0.0, -1.0]))
    stokes.add_rotated_freeslip_bc(0, "Left", normal=np.array([-1.0, 0.0]))
    stokes.add_rotated_freeslip_bc(0, "Right", normal=np.array([1.0, 0.0]))
    flagged, out = _solve_and_read(stokes, capsys)
    assert not flagged
    assert WARNING not in out


def test_a_curved_given_normal_with_an_open_boundary_does_not_warn(capsys):
    mesh = _annulus()
    stokes = _stokes(mesh, "e", nullspace=False)
    stokes.bodyforce = _load(mesh)
    stokes.add_rotated_freeslip_bc(0, "Upper", normal=_radial(mesh))
    # Lower has no condition: a natural (traction-free) boundary, its flux free.
    flagged, out = _solve_and_read(stokes, capsys)
    assert not flagged
    assert WARNING not in out
