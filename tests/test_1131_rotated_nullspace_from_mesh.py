"""The rotated free-slip path takes its rigid-rotation modes from the mesh (#906).

A spherical shell with rotated free slip on both walls and the default normal
(the facet-averaged mesh normal) used to solve with an arbitrary rigid rotation
in its answer. The rotated path built every Cartesian rotation and kept those
that passed a 1e-8 test against the discrete constraints; on the faceted sphere
the facet-averaged normal is not radial at the nodes, so all three rotations
failed (violations 0.0140, 0.0164 and 0.0075) and none was removed. Measured on
the shell below with FMG: a rigid rotation of rate 6.2e-3 in an answer of v_rms
4.62e-3, against 1.81e-3 under the analytic radial normal.

The modes are now declared by the geometry, ``mesh.nullspace_rotations``, as the
standard Stokes path already reads them. A declared mode is made exactly
compatible with v.n = 0 by zeroing its constrained rows, and is still tested
against the assembled operator, so a no-slip wall that pins it rejects it. Only an
exact mode selects the SVD coarse solve: the declared modes are near-null under
the facet-averaged normal, and counting them ran a dense SVD for 40 minutes on a
shell that LU solves in 40 seconds. A mesh
that declares nothing keeps the detection (a box finds no mode, as before), and
a mesh without the metadata declares its modes through the public setter.
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]

# v_rms of #906's shell under the default normal over v_rms under the analytic
# one, recorded on macOS arm64: 1.88497e-3 / 1.81167e-3. Development gave 2.55,
# the spurious rotation included. A ratio, because both solves share the mesh
# and CI's gmsh need not build this one.
VRMS_RATIO_MESH_OVER_RADIAL = 1.04046


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


def _shell_stokes(name, radial=False, torque=0.0):
    """#906's shell: P2/P1, isoviscous, a cos(4 theta) buoyancy, rotated free slip
    on both walls, with the default (facet-averaged) normal or the analytic one;
    ``torque`` adds an azimuthal push about z of that size, a net torque."""
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
                        * sympy.Matrix([x / r, y / r, z / r]).T
                        + torque * sympy.Matrix([-y, x, 0]).T)
    normal = mesh.CoordinateSystem.unit_e_0 if radial else None
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


def _v_rms(v):
    V = np.asarray(v.array).reshape(-1, v.mesh.cdim)
    return np.sqrt(uw.mpi.comm.allreduce(float(np.sum(V * V)))
                   / uw.mpi.comm.allreduce(V.shape[0]))


@pytest.fixture(scope="module")
def radial_shell():
    mesh, v, stokes = _shell_stokes("1131r", radial=True)
    stokes.solve()
    return stokes, _v_rms(v)


def test_a_shell_with_the_default_normal_returns_no_rigid_rotation(radial_shell, capsys):
    mesh, v, stokes = _shell_stokes("1131a")
    assert len(mesh.nullspace_rotations) == 3
    capsys.readouterr()
    stokes.solve()
    # the forcing is radial: the gauge component is faceting, 8.8e-5 of the
    # forcing scale, and the net-torque guard stays quiet
    assert "net torque" not in capsys.readouterr().out
    info = stokes._rotated_freeslip_info
    assert info["converged"], info
    assert info["rotation_gauge"]["orthonormal"] == 3, info["rotation_gauge"]
    vrms = _v_rms(v)
    # the gauge removes the rotation to round-off; development left 6.2e-3
    assert np.abs(_rotation_rates(v)).max() < 1e-10 * vrms
    # near-null, not null (0.0164 here): the coarse solve stays LU
    assert stokes._rotated_velocity_null_modes == 0
    assert 1e-3 < stokes._rotated_null_mode_slack < 0.05
    assert vrms / radial_shell[1] == pytest.approx(VRMS_RATIO_MESH_OVER_RADIAL, rel=0.01)


def test_a_net_torque_still_warns_under_the_default_normal(capsys):
    # 1% of the forcing: the gauge component is 1.7e-2 of the forcing scale,
    # against 8.8e-5 from the faceting alone
    mesh, v, stokes = _shell_stokes("1131t", torque=0.01)
    capsys.readouterr()
    stokes.solve()
    assert "net torque" in capsys.readouterr().out


def test_the_analytic_normal_makes_the_declared_modes_exact(radial_shell):
    stokes = radial_shell[0]
    info = stokes._rotated_freeslip_info
    assert info["converged"], info
    assert info["rotation_gauge"]["orthonormal"] == 3, info["rotation_gauge"]
    assert max(m["viol"] for m in info["rotation_gauge"]["modes"]) < 1e-14
    # exact modes: the coarse block is singular, and the count says so
    assert stokes._rotated_velocity_null_modes == 3
    assert stokes._rotated_null_mode_slack == 0.0


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
