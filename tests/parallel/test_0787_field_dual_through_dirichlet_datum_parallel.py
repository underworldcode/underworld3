"""The adjoint dual on a field that enters only through a Dirichlet datum (#787).

``gradient(..., fields=[f])`` differentiated the volume residual only, so a field
read only in an essential datum, ``T|Top = f``, had a dual of zero; it was later
refused instead. Its dual is the reaction of the adjoint constraint plus the
misfit's own dual on the constrained rows, mapped onto the field's nodes, the same
term a parameter in a datum carries (#762). Where the field lives on the unknown's
own space its nodes are the constrained nodes, and the map is exact.

The dual is checked against a central difference of J along a direction, paired
over the owned degrees of freedom (``uw.adjoint.inner``). Serial and parallel.

    python scripts/mpi_supervisor.py --silence 300 -- mpirun -n 2 python -m pytest --with-mpi tests/parallel/test_0787_field_dual_through_dirichlet_datum_parallel.py
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b, pytest.mark.timeout(300)]


def _poisson(datum_of_f, misfit_of_T):
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1), cellSize=1 / 8)
    T = uw.discretisation.MeshVariable("T0787", mesh, 1, degree=2)
    f = uw.discretisation.MeshVariable("f0787", mesh, 1, degree=2)
    X = np.asarray(f.coords)
    f.array[:, 0, 0] = 1.0 + np.sin(np.pi * X[:, 0])
    poisson = uw.systems.Poisson(mesh, u_Field=T)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1.0
    poisson.f = 1.0
    poisson.tolerance = 1e-11
    poisson.petsc_options.delValue("ksp_monitor")
    poisson.add_dirichlet_bc(0.0, "Bottom")
    poisson.add_dirichlet_bc(sympy.Matrix([datum_of_f(f.sym[0])]), "Top")
    return mesh, T, f, poisson, misfit_of_T(T.sym[0])


@pytest.mark.parametrize("datum_of_f", [lambda s: s, lambda s: s ** 2 / 2],
                         ids=["T = f", "T = f^2 / 2"])
def test_the_dual_on_a_field_in_a_dirichlet_datum(datum_of_f):
    mesh, T, f, poisson, misfit = _poisson(datum_of_f, lambda t: t ** 2 / 2)
    F0 = np.array(np.asarray(f.array)[:, 0, 0], copy=True)

    def J():
        poisson.solve(zero_init_guess=True)
        return float(uw.maths.Integral(mesh, misfit).evaluate())

    J()
    out = poisson.gradient(misfit, fields=[f])
    X = np.asarray(f.coords)
    d = np.cos(2 * X[:, 0]) * (0.5 + X[:, 1])
    adjoint = uw.adjoint.inner(f, out["fields"][f], d)

    h = 1e-5
    f.array[:, 0, 0] = F0 + h * d
    Jp = J()
    f.array[:, 0, 0] = F0 - h * d
    Jm = J()
    f.array[:, 0, 0] = F0
    fd = (Jp - Jm) / (2 * h)
    assert abs(fd) > 1e-3, f"premise: the datum moves J ({fd})"
    assert adjoint == pytest.approx(fd, rel=1e-6), (adjoint, fd)


def test_the_dual_on_a_field_in_a_stokes_velocity_datum():
    """A saddle point: the lid velocity on Top is (f, 0), f a P2 scalar on the
    velocity's own nodes. The reaction carries the pressure adjoint. The
    direction takes equal values at the two top corners (see below)."""
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1), cellSize=1 / 6)
    v = uw.discretisation.MeshVariable("V0787", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("P0787", mesh, 1, degree=1)
    f = uw.discretisation.MeshVariable("g0787", mesh, 1, degree=2)
    X = np.asarray(f.coords)
    f.array[:, 0, 0] = np.sin(np.pi * X[:, 0])
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.saddle_preconditioner = 1.0
    stokes.tolerance = 1e-11
    stokes.petsc_options["ksp_type"] = "fgmres"
    stokes.petsc_use_pressure_nullspace = True
    stokes.petsc_options.delValue("ksp_monitor")
    for wall in ("Left", "Right", "Bottom"):
        stokes.add_dirichlet_bc((0.0, 0.0), wall)
    stokes.add_dirichlet_bc(sympy.Matrix([f.sym[0], 0.0]).T, "Top")
    x, y = mesh.X
    misfit = x * y * v.sym[0] + v.sym[1] ** 2
    F0 = np.array(np.asarray(f.array)[:, 0, 0], copy=True)

    def J():
        stokes.solve(zero_init_guess=True)
        return float(uw.maths.Integral(mesh, misfit).evaluate())

    J()
    out = stokes.gradient(misfit, fields=[f])
    # Equal at the two top corners: there the x-velocity is the side walls'
    # normal velocity, so unequal corner values push net flux through the walls
    # of a closed incompressible box, and the finite difference then measures
    # the solve's handling of an incompatible problem, not the derivative.
    d = (1.0 + 0.3 * np.cos(2 * np.pi * X[:, 0])) * (0.5 + X[:, 1])
    adjoint = uw.adjoint.inner(f, out["fields"][f], d)
    h = 1e-5
    f.array[:, 0, 0] = F0 + h * d
    Jp = J()
    f.array[:, 0, 0] = F0 - h * d
    Jm = J()
    f.array[:, 0, 0] = F0
    fd = (Jp - Jm) / (2 * h)
    assert abs(fd) > 1e-4, f"premise: the datum moves J ({fd})"
    assert adjoint == pytest.approx(fd, rel=1e-6), (adjoint, fd)


def test_a_field_on_another_space_in_a_datum_is_refused():
    """A P1 field in a datum on a P2 unknown: its nodes are not the constrained
    nodes, and the dual needs a projection that is not built. Refused, on every
    rank alike, rather than given a wrong or zero dual."""
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1), cellSize=1 / 6)
    T = uw.discretisation.MeshVariable("T0787s", mesh, 1, degree=2)
    f = uw.discretisation.MeshVariable("f0787s", mesh, 1, degree=1)
    f.array[:, 0, 0] = 1.0
    poisson = uw.systems.Poisson(mesh, u_Field=T)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1.0
    poisson.f = 1.0
    poisson.petsc_options.delValue("ksp_monitor")
    poisson.add_dirichlet_bc(0.0, "Bottom")
    poisson.add_dirichlet_bc(sympy.Matrix([f.sym[0]]), "Top")
    poisson.solve()
    message = ""
    try:
        poisson.gradient(T.sym[0] ** 2 / 2, fields=[f])
    except NotImplementedError as error:
        message = str(error)
    messages = uw.mpi.comm.allgather(message)
    assert len(set(messages)) == 1 and "needs a projection" in messages[0], messages
