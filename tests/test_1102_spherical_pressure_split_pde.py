"""Exact hydrostatic pressure jumps on a fitted polyhedral spherical shell.

The independent discontinuous pressure is constant on each labelled volume,
plus an optional Cartesian affine pressure. It follows the fitted interface,
not a radial Piecewise expression that would cut some affine tetrahedra.
These gates validate pressure splitting and gauges, not true-sphere traction.
"""

import numpy as np
import pytest
import sympy as sp
import underworld3 as uw

pytestmark = pytest.mark.level_2


@pytest.mark.tier_b
@pytest.mark.parametrize("gradient", [0, 1])
def test_spherical_fitted_pressure_jump_hydrostatic_exactness(gradient):
    """A prescribed jump and regular pressure gradient drive exactly zero flow."""
    mesh = uw.meshing.SphericalShellInternalBoundary(
        radiusInner=0.55, radiusInternal=0.775, radiusOuter=1,
        cellSize=1 / 8, degree=1, qdegree=3,
    )
    velocity = uw.discretisation.MeshVariable("HydrostaticU", mesh, 3, degree=2)
    pressure = uw.discretisation.MeshVariable("HydrostaticQ", mesh, 1, degree=1)
    exact_step = uw.discretisation.MeshVariable(
        "HydrostaticStep", mesh, 1, degree=0, continuous=False,
    )
    stokes = uw.systems.Stokes(mesh, velocityField=velocity, pressureField=pressure)
    x, y, z = mesh.X
    radius = sp.sqrt(x*x + y*y + z*z)
    split = stokes.pressure_split(
        jumps={"Internal": sp.Integer(2)},
        normals={"Internal": (x / radius, y / radius, z / radius)},
    )

    # The independently labelled volume is constant on each complete cell.
    outer = mesh.cells_labelled("Outer", mesh.regions.Outer.value)
    inner = mesh.cells_labelled("Inner", mesh.regions.Inner.value)
    assert np.all(outer != inner)
    exact_step.array[:, 0, 0] = np.where(outer, 2, 0)

    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.viscosity = 1
    stokes.bodyforce = sp.Matrix([gradient, 0, 0]) + split.regular_force.sym.T
    for boundary in ("Lower", "Upper"):
        stokes.add_essential_bc(sp.zeros(3, 1), boundary)
    stokes.tolerance = 1e-10
    stokes.solve(verbose=False, debug=False)
    assert stokes.solve_report is not None and stokes.solve_report.converged

    volume = uw.maths.Integral(mesh, sp.S.One).evaluate()
    target = exact_step.sym[0] + gradient * x
    target_mean = uw.maths.Integral(mesh, target).evaluate() / volume
    difference = split.physical_pressure - target
    gauge = uw.maths.Integral(mesh, difference).evaluate() / volume
    pressure_rms = np.sqrt(abs(uw.maths.Integral(
        mesh, (difference - gauge)**2,
    ).evaluate()) / volume)
    reference_rms = np.sqrt(abs(uw.maths.Integral(
        mesh, (target - target_mean)**2,
    ).evaluate()) / volume)
    velocity_rms = np.sqrt(abs(uw.maths.Integral(
        mesh, velocity.sym.dot(velocity.sym),
    ).evaluate()) / volume)
    if uw.mpi.rank == 0:
        print("Spherical fitted hydrostatic RMS:", dict(
            gradient=gradient, pressure=pressure_rms, velocity=velocity_rms,
            reference=reference_rms, gauge=gauge,
        ))
    assert reference_rms > 0
    assert pressure_rms / reference_rms < 2e-8
    assert velocity_rms / 2 < 2e-9
    # The lift integral used by pressure gauges must agree with native FE integration.
    integral = uw.maths.Integral(mesh, split.pressure_jump).evaluate()
    assert integral == pytest.approx(split.pressure_jump_integral, abs=volume * 1e-11)
