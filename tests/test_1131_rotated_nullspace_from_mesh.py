"""The rotated free-slip path takes its rigid-rotation modes from the mesh (#906).

A spherical shell with rotated free slip on both walls and the default normal
(the facet-averaged mesh normal) used to solve with an arbitrary rigid rotation
in its answer. The rotated path built every Cartesian rotation and kept those
that passed a 1e-8 test against the discrete constraints; on the faceted sphere
the facet-averaged normal is not radial at the nodes, so all three rotations
failed (violations 0.0140, 0.0164 and 0.0075) and none was removed. Measured on
the shell below: rotation rates (-5.98, -1.00, -1.15) in an answer of v_rms 4.47.

The modes are now declared by the geometry, ``mesh.nullspace_rotations``, as the
standard Stokes path already reads them. A declared mode is made exactly
compatible with v.n = 0 by zeroing its constrained rows, and is still tested
against the assembled operator, so a no-slip wall that pins it rejects it. A mesh
that declares nothing keeps the detection (a box finds no mode, as before), and
a mesh without the metadata declares its modes through the public setter.
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]


def _rotation_rates(v):
    """Least-squares rigid rotation of the nodal velocity: the omega minimising
    sum |v_i - omega x r_i|^2 over the velocity nodes, summed over the ranks."""
    X = np.asarray(v.coords)
    V = np.asarray(v.array).reshape(X.shape)
    A = np.einsum("ij,ij->", X, X) * np.eye(3) - X.T @ X
    b = np.cross(X, V).sum(axis=0)
    A = uw.mpi.comm.allreduce(A)
    b = uw.mpi.comm.allreduce(b)
    return np.linalg.solve(A, b)


def _shell_stokes(name, normal=None):
    """#906's shell: P2/P1, isoviscous, a cos(4 theta) buoyancy, rotated free slip
    on both walls."""
    mesh = uw.meshing.SphericalShell(radiusOuter=1.0, radiusInner=0.5, cellSize=0.5,
                                     refinement=1, qdegree=3)
    v = uw.discretisation.MeshVariable(f"V{name}", mesh, 3, degree=2)
    p = uw.discretisation.MeshVariable(f"P{name}", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    x, y, z = mesh.X
    r = sympy.sqrt(x ** 2 + y ** 2 + z ** 2)
    th = sympy.atan2(y, x)
    stokes.bodyforce = (sympy.cos(4 * th) * (z / r) * sympy.sin(sympy.pi * (r - 0.5) / 0.5)
                        * sympy.Matrix([x / r, y / r, z / r]).T)
    for wall in ("Lower", "Upper"):
        stokes.add_rotated_freeslip_bc(0, wall, normal=normal)
    stokes.petsc_use_pressure_nullspace = True
    stokes.tolerance = 1.0e-6
    return mesh, v, stokes


def _annulus_stokes(name, inner="Lower"):
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.2, qdegree=3)
    v = uw.discretisation.MeshVariable(f"V{name}", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable(f"P{name}", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    x, y = mesh.X
    r = sympy.sqrt(x ** 2 + y ** 2)
    th = sympy.atan2(y, x)
    stokes.bodyforce = (sympy.cos(3 * th) * sympy.sin(sympy.pi * (r - 0.5) / 0.5)
                        * sympy.Matrix([x / r, y / r]).T)
    stokes.tolerance = 1.0e-6
    return mesh, v, stokes


def test_a_shell_with_the_default_normal_returns_no_rigid_rotation():
    mesh, v, stokes = _shell_stokes("1131a")
    assert len(mesh.nullspace_rotations) == 3
    stokes.solve()
    info = stokes._rotated_freeslip_info
    assert info["converged"], info
    assert info["rotation_gauge"]["orthonormal"] == 3, info["rotation_gauge"]
    omega = _rotation_rates(v)
    # radial-normal reference on this shell: |omega| <= 0.012 (#906)
    assert np.abs(omega).max() < 0.05, omega


def test_a_box_declares_no_mode_and_finds_none():
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=0.2, qdegree=3)
    assert mesh.nullspace_rotations == []
    v = uw.discretisation.MeshVariable("V1131b", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("P1131b", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    x, y = mesh.X
    stokes.bodyforce = (0.0, sympy.cos(sympy.pi * x) * sympy.sin(sympy.pi * y))
    for wall in ("Bottom", "Top", "Left", "Right"):
        stokes.add_rotated_freeslip_bc(0, wall)
    stokes.petsc_use_pressure_nullspace = True
    stokes.solve()
    assert stokes._rotated_freeslip_info["rotation_gauge"]["orthonormal"] == 0


def test_a_declared_rotation_pinned_by_a_no_slip_wall_is_rejected():
    mesh, v, stokes = _annulus_stokes("1131c")
    stokes.add_dirichlet_bc((0.0, 0.0), "Lower")
    stokes.add_rotated_freeslip_bc(0, "Upper")
    stokes.solve()
    gauge = stokes._rotated_freeslip_info["rotation_gauge"]
    assert gauge["offered"] == 1 and gauge["orthonormal"] == 0, gauge


def test_a_mesh_without_the_metadata_declares_it_through_the_setter():
    source = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.2, qdegree=3)
    mesh = uw.discretisation.Mesh(source.dm.clone(), simplex=True, qdegree=3,
                                  boundaries=source.boundaries,
                                  coordinate_system_type=source.CoordinateSystem.coordinate_type)
    assert mesh.nullspace_rotations == []
    x, y = mesh.X
    with pytest.raises(ValueError):
        mesh.nullspace_rotations = [sympy.Matrix([-y, x, 0])]   # three components in 2-D
    mesh.nullspace_rotations = [sympy.Matrix([-y, x])]
    assert len(mesh.nullspace_rotations) == 1
    mesh.nullspace_rotations = []
    assert mesh.nullspace_rotations == []
