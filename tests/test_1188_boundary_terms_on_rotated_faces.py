"""Boundary terms read the coordinates at the point where they read the fields (#1188).

In 3-D, PETSc evaluates a boundary integrand at the physical point of face
quadrature point q and the basis at the cell's tabulation point q, reversed for a
negative orientation. That pairs the two correctly only on faces at orientation 0
in their cell: a triangle has six orientations. A refined mesh puts about a fifth
of its wall faces at other orientations, and there a load or a normal written in
the coordinates is evaluated at one point and spread with the basis of another.
The total of a load is right; its moments are not.

Each check is an exact moment of a term on the Top wall of the unit cube (a plane
where x runs from 0 to 1), so it holds to round-off whatever the mesh:

- the residual of the natural condition f = x: (b . x) / (b . 1) = (1/3) / (1/2);
- the Jacobian of the natural condition f = x T: (x . M x) / (1 . M 1) = (1/4) / (1/2);
- the boundary integral of T x with T = x: 1/3;
- Nitsche free slip with the radial unit vector as the normal, on a spherical
  shell, with a rigid rotation, whose radial component is zero at every point:
  the residual is zero;
- a load on an internal surface is the same whichever of each face's two cells
  assembles it (in parallel the partition decides which).

The cube and the shell are refined once, which puts wall faces at orientations 1
and 2; swapping the internal faces' cells puts them at -1, -2 and -3. P1 holds 1
and x, and qdegree 3 integrates every moment exactly.

The file runs serially and under mpirun (scripts/test.sh, np 4): counts and
maxima are taken over all ranks, and a rank may hold none of a boundary's faces.
"""
import numpy as np
import pytest
import sympy
from mpi4py import MPI

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]


def _faces(mesh, boundary):
    """This rank's faces of ``boundary``; none on a rank that holds none of them."""
    dm = mesh.dm
    label = dm.getLabel("UW_Boundaries")
    value = mesh.boundaries[boundary].value
    if label.getStratumSize(value) == 0:
        return []
    f_start, f_end = dm.getHeightStratum(1)
    return [face for face in label.getStratumIS(value).getIndices() if f_start <= face < f_end]


def _orientations(mesh, boundary):
    """The orientation of each of this rank's faces of ``boundary`` in its first support cell."""
    dm = mesh.dm
    out = []
    for face in _faces(mesh, boundary):
        cell = dm.getSupport(face)[0]
        out.append(int(dm.getConeOrientation(cell)[list(dm.getCone(cell)).index(face)]))
    return np.array(out, dtype=int)


def _rotated_faces(mesh, boundary):
    """The number of faces of ``boundary`` not at orientation 0, over all ranks."""
    return uw.mpi.comm.allreduce(int(np.count_nonzero(_orientations(mesh, boundary))), op=MPI.SUM)


def _largest(values):
    """The largest magnitude in ``values`` over all ranks."""
    return uw.mpi.comm.allreduce(float(np.max(np.abs(values), initial=0.0)), op=MPI.MAX)


def _refined_cube():
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0, 0.0), maxCoords=(1.0, 1.0, 1.0), cellSize=0.5,
        refinement=1, qdegree=3)


def _poisson_on_the_top_wall(mesh, tag, natural):
    """A built Poisson solver whose only boundary term is ``natural(x, T)`` on Top.
    Returns the SNES and the solver's global vectors of x and of 1."""
    T = uw.discretisation.MeshVariable(f"T1188{tag}", mesh, 1, degree=1)
    s = uw.systems.Poisson(mesh, u_Field=T)
    s.constitutive_model = uw.constitutive_models.DiffusionModel
    s.constitutive_model.Parameters.diffusivity = 1.0
    s.f = 0.0
    x = mesh.CoordinateSystem.X[0]
    if natural is not None:
        s.add_natural_bc([natural(x, T.sym[0])], "Top")
    s._build()
    T.array[:, 0, 0] = np.asarray(T.coords)[:, 0]
    g_x = s.dm.getGlobalVec()
    s.dm.localToGlobal(T.vec, g_x)
    g_1 = g_x.duplicate()
    g_1.set(1.0)
    return s.snes, g_x, g_1


def test_the_natural_load_has_its_exact_first_moment():
    mesh = _refined_cube()
    assert _rotated_faces(mesh, "Top") > 0     # the fixture has rotated faces
    snes, g_x, g_1 = _poisson_on_the_top_wall(mesh, "r", lambda x, T: x)
    zero, b = g_x.duplicate(), g_x.duplicate()
    zero.set(0.0)
    snes.computeFunction(zero, b)

    # int x dS and int x^2 dS over the wall, whatever the residual's sign convention
    assert abs(b.dot(g_1)) > 0.4
    ratio = b.dot(g_x) / b.dot(g_1)
    # development: 0.66567 (-1.0e-3 from 2/3)
    assert abs(ratio - 2.0 / 3.0) < 1.0e-13, ratio


def test_the_natural_jacobian_has_its_exact_moments():
    def jacobian(tag, natural):
        snes, g_x, g_1 = _poisson_on_the_top_wall(_refined_cube(), tag, natural)
        snes.setUp()                         # a direct computeJacobian needs it
        J = snes.getJacobian()[0]
        snes.computeJacobian(g_x, J, J)
        return J, g_x, g_1

    assert _rotated_faces(_refined_cube(), "Top") > 0
    J1, g_x, g_1 = jacobian("j1", lambda x, T: x * T)
    J0, _g, _h = jacobian("j0", None)
    J1.axpy(-1.0, J0)                        # the boundary term alone
    M_x, M_1 = g_x.duplicate(), g_x.duplicate()
    J1.mult(g_x, M_x)
    J1.mult(g_1, M_1)

    assert abs(g_1.dot(M_1)) > 0.4
    ratio = g_x.dot(M_x) / g_1.dot(M_1)
    # development: 0.49899 (-1.0e-3 from 1/2)
    assert abs(ratio - 0.5) < 1.0e-13, ratio


def test_the_boundary_integral_has_its_exact_moment():
    mesh = _refined_cube()
    assert _rotated_faces(mesh, "Top") > 0
    T = uw.discretisation.MeshVariable("T1188i", mesh, 1, degree=1)
    T.array[:, 0, 0] = np.asarray(T.coords)[:, 0]
    x = mesh.CoordinateSystem.X[0]

    value = uw.maths.BdIntegral(mesh, fn=T.sym[0] * x, boundary="Top").evaluate()

    # development: 0.33283 (-5.0e-4 from 1/3)
    assert abs(value - 1.0 / 3.0) < 1.0e-13, value


def test_nitsche_with_a_radial_normal_leaves_a_rigid_rotation_alone():
    mesh = uw.meshing.SphericalShell(radiusInner=0.55, radiusOuter=1.0, cellSize=0.5,
                                     qdegree=2, refinement=1)
    assert _rotated_faces(mesh, "Upper") > 0
    velocity = uw.discretisation.MeshVariable("U1188n", mesh, 3, degree=2, vtype=uw.VarType.VECTOR)
    pressure = uw.discretisation.MeshVariable("P1188n", mesh, 1, degree=1, continuous=True)
    stokes = uw.systems.Stokes(mesh, velocityField=velocity, pressureField=pressure)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.bodyforce = 0.0 * mesh.CoordinateSystem.unit_e_0
    stokes.add_nitsche_bc(0.0, "Upper", normal=mesh.CoordinateSystem.unit_e_0)
    stokes._build()

    X = np.asarray(velocity.coords)

    def largest_residual(field):
        for var in stokes.fields.values():
            var.array[...] = 0.0
        velocity.array[:, 0, :] = field
        rows = stokes.compute_volume_residual_fields()["velocity"]
        return _largest(np.asarray(rows))

    # a radial velocity through the same term gives the scale
    scale = largest_residual(X)
    rotation = largest_residual(np.stack([-X[:, 1], X[:, 0], 0.0 * X[:, 0]], axis=1))

    assert scale > 0.0
    # development: 1.5e-2 of the scale
    assert rotation < 1.0e-12 * scale, rotation / scale


def test_an_internal_load_does_not_depend_on_which_cell_assembles_it():
    """An interior face's boundary terms are assembled in the frame of its first
    support cell, which in parallel is whichever cell the partition leaves first.
    Swapping the support order of every Internal face puts them at negative
    orientations in that cell; the load must not change."""
    def internal_load(mesh, tag):
        velocity = uw.discretisation.MeshVariable(f"U1188s{tag}", mesh, 3, degree=2,
                                                  vtype=uw.VarType.VECTOR)
        pressure = uw.discretisation.MeshVariable(f"P1188s{tag}", mesh, 1, degree=1)
        stokes = uw.systems.Stokes(mesh, velocityField=velocity, pressureField=pressure)
        stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
        stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
        stokes.bodyforce = sympy.Matrix([0.0, 0.0, 0.0]).T
        theta = mesh.CoordinateSystem.xR[1]
        shape = sympy.assoc_legendre(2, 0, sympy.cos(theta))
        stokes.add_natural_bc(shape * mesh.CoordinateSystem.unit_e_0, "Internal")
        stokes._build()
        for var in stokes.fields.values():
            var.array[...] = 0.0
        rows = stokes.compute_volume_residual_fields()["velocity"]
        return np.asarray(velocity.coords).copy(), np.asarray(rows).reshape(-1, 3).copy()

    mesh = uw.meshing.SphericalShellInternalBoundary(
        radiusOuter=1.0, radiusInternal=0.775, radiusInner=0.55, cellSize=0.3, qdegree=2)
    X0, load0 = internal_load(mesh, "a")
    dm = mesh.dm
    # a face whose two cells are both on this rank is swapped; one at the edge of the
    # rank's cells has a single local cell and keeps it
    swapped = [face for face in _faces(mesh, "Internal") if dm.getSupportSize(face) == 2]
    for face in swapped:
        support = dm.getSupport(face)
        dm.setSupport(face, np.array(support[::-1], dtype=support.dtype))
    orientations = _orientations(mesh, "Internal")
    two_cells = np.array([dm.getSupportSize(face) == 2 for face in _faces(mesh, "Internal")], dtype=bool)
    # each swapped face is now seen from its other cell
    assert uw.mpi.comm.allreduce(int(np.count_nonzero(orientations[two_cells] >= 0)), op=MPI.SUM) == 0
    assert uw.mpi.comm.allreduce(len(swapped), op=MPI.SUM) > 0
    X1, load1 = internal_load(mesh, "b")

    # one layout, so rows compare by position
    assert uw.mpi.comm.allreduce(int(not np.array_equal(X0, X1)), op=MPI.SUM) == 0
    assert _largest(load0) > 1.0e-3
    # development: 534 of 534 loaded nodes change, by up to half the largest load
    assert _largest(load1 - load0) < 1.0e-14 * _largest(load0)
