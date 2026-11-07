"""The rotated free-slip path takes its rigid-rotation modes from the mesh (#906).

A spherical shell with rotated free slip on both walls and the default normal (the
facet-averaged mesh normal) used to solve with an arbitrary rigid rotation in its
answer. The rotated path built every Cartesian rotation and kept those that passed a
1e-8 test against the discrete constraints. On the faceted sphere the facet-averaged
normal is not radial at the nodes, so all three rotations failed (violations
0.0075-0.0164), none was removed, and the answer carried a rotation: rate 4.66 in a
v_rms of 3.93 on the shell below, against 1.81 under the analytic radial normal.

The modes are now the ones the mesh declares, ``mesh.nullspace_rotations``, as the
standard Stokes path already reads them. A declared mode skips the constraint test
only while it leaves every rotated wall in place: the wall's vertices must be
equidistant from the rotation's centre, to 1e-9 relative. Faceting keeps them on the
sphere; a deformed wall (a free surface) does not, and the mode then has to pass the
constraint test like an undeclared one, with a warning. An admitted declared mode
has its constrained rows zeroed, so v.n = 0 stays exact. It is still tested against
the assembled operator, so a no-slip wall that pins it rejects it. A mesh that
declares nothing keeps the detection (a box finds no mode). A mesh without the
metadata declares its modes through the public setter, and meshes cut from a
declaring mesh inherit them.

Contracts, all exactness properties:
- on the shell, under either normal, three modes are removed, the answer's rigid
  rotation is at round-off, and v.n is zero on every constrained row;
- an annulus as built keeps its declared mode; one whose outer wall is pushed off its
  circle by 1e-6, as a free surface moves it, loses it, with a warning, and the
  constraint test then rejects it; one moved the way an MMPDE mover moves it (wall
  vertices sliding along their circles, the interior moving) keeps it, and its
  answer carries no rotation;
- a no-slip wall rejects a declared mode; a box declares none and finds none;
- the setter refuses a field that is not a rigid rotation, and an add_fault child
  inherits the declaration, written in its own coordinates;
- a shell in a units model solves and keeps its three modes (#877: the modes were
  built from dimensional coordinates, and development raised TypeError).
"""
import warnings

import numpy as np
import pytest
import sympy
from mpi4py import MPI
from petsc4py import PETSc

import underworld3 as uw
from underworld3.adjoint import misfit_duals
from underworld3.utilities import custom_mg, rotated_bc

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]


# The nodal helpers below read local arrays, which carry ghost copies at np > 1:
# this file runs on one rank. Its parallel twin asserts only verdicts that do not
# depend on the partition.


def _rotation_rates(v):
    """Least-squares rigid rotation of the nodal velocity: the omega minimising
    sum |v_i - omega x r_i|^2 over the velocity nodes. The gauge removes this
    nodal component exactly; an area-weighted rate is not zero."""
    X = np.asarray(v.coords_nd)
    V = np.asarray(v.array).reshape(X.shape)
    A = np.einsum("ij,ij->", X, X) * np.eye(3) - X.T @ X
    b = np.cross(X, V).sum(axis=0)
    return np.linalg.solve(A, b)


def _normal_velocity(info):
    """The largest |v.n| on the constrained rows (rows of Q·U), over |U|, over ranks."""
    Q, U, rows = info["Q"], info["U"], info["normal_rows"]
    Uhat = U.duplicate()
    Q.mult(U, Uhat)
    rs, re = Uhat.getOwnershipRange()
    a = Uhat.getArray()
    local = [g - rs for g in rows if rs <= g < re]
    vn = float(np.abs(a[local]).max()) if local else 0.0
    Uhat.destroy()
    return uw.mpi.comm.allreduce(vn, op=MPI.MAX) / U.norm(PETSc.NormType.NORM_INFINITY)


def _shell_stokes(name, radial=False, torque=0.0):
    """P2/P1, isoviscous, a cos(4 theta) buoyancy, rotated free slip on both walls
    with the default normal or the analytic one, and the custom FMG on the shell's
    levels; ``torque`` adds an azimuthal push about z of that size, a net torque."""
    mesh = uw.meshing.SphericalShell(radiusOuter=1.0, radiusInner=0.5, cellSize=0.3,
                                     refinement=1, qdegree=3)
    v = uw.discretisation.MeshVariable(f"V{name}", mesh, 3, degree=2)
    p = uw.discretisation.MeshVariable(f"P{name}", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    x, y, z = mesh.X
    r = sympy.sqrt(x ** 2 + y ** 2 + z ** 2)
    th = sympy.atan2(y, x)
    stokes.bodyforce = (1.0e3 * sympy.cos(4 * th) * (z / r) * sympy.sin(sympy.pi * (r - 0.5) / 0.5)
                        * sympy.Matrix([x / r, y / r, z / r]).T
                        + torque * 1.0e3 * sympy.Matrix([-y, x, 0]).T)
    normal = mesh.CoordinateSystem.unit_e_0 if radial else None
    for wall in ("Lower", "Upper"):
        stokes.add_rotated_freeslip_bc(0, wall, normal=normal)
    stokes.petsc_use_pressure_nullspace = True
    stokes.tolerance = 1.0e-6
    custom_mg.set_custom_fmg(stokes, mesh._coarse_level_meshes()[:-1], field_id=0)
    return mesh, v, stokes


def _annulus_stokes(name, mesh=None):
    mesh = mesh or uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.2, qdegree=3)
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
    return np.sqrt(float(np.sum(V * V)) / V.shape[0])


@pytest.mark.parametrize("radial", [False, True], ids=["default_normal", "analytic_normal"])
def test_the_shell_returns_no_rigid_rotation(radial, capsys):
    mesh, v, stokes = _shell_stokes(f"1131{int(radial)}", radial=radial)
    assert len(mesh.nullspace_rotations) == 3
    capsys.readouterr()
    stokes.solve()
    # the forcing is radial: the net-torque guard stays quiet
    assert "net torque" not in capsys.readouterr().out
    info = stokes._rotated_freeslip_info
    assert info["converged"] and info["velocity_pc"] == "custom-FMG", info
    # exempt from the constraint test under the facet-averaged normal; a given normal
    # is tested, and the exact radial one passes
    assert info["declared_modes_exempt"] == [not radial] * 3
    assert info["rotation_gauge"]["orthonormal"] == 3, info["rotation_gauge"]
    assert stokes._rotated_velocity_null_modes == 3
    # the gauge removes the rotation to round-off; without the declared modes the
    # default normal left a rate of 4.66 here
    assert np.abs(_rotation_rates(v)).max() < 1.0e-10 * _v_rms(v)
    # the admitted modes were zeroed on the constrained rows, so v.n stays exact
    assert _normal_velocity(info) < 1.0e-12
    if radial:
        assert max(m["viol"] for m in info["rotation_gauge"]["modes"]) < 1.0e-14
        assert stokes._rotated_null_mode_slack == 0.0
        # exact modes of the velocity block: projected out of the multigrid (#963)
        assert info["projected_null_modes"] == 3, info["null_mode_ratios"]
        assert max(info["null_mode_ratios"]) < 1.0e-14
    else:
        assert 1.0e-3 < stokes._rotated_null_mode_slack < 0.05    # near-null, not null
        # near-null on the faceted walls: left to the multigrid, well above the 1e-10
        # at which a mode is projected (#963)
        assert info["projected_null_modes"] == 0, info["null_mode_ratios"]
        assert min(info["null_mode_ratios"]) > 1.0e-8, info["null_mode_ratios"]


def test_a_net_torque_still_warns_under_the_default_normal(capsys):
    mesh, v, stokes = _shell_stokes("1131t", torque=0.01)
    capsys.readouterr()
    stokes.solve()
    assert "net torque" in capsys.readouterr().out


def test_an_annulus_as_built_keeps_its_declared_mode():
    mesh, v, stokes = _annulus_stokes("1131k")
    for wall in ("Lower", "Upper"):
        stokes.add_rotated_freeslip_bc(0, wall)
    stokes.petsc_use_pressure_nullspace = True
    stokes.solve()
    assert stokes._rotated_freeslip_info["declared_modes_exempt"] == [True]
    assert stokes._rotated_freeslip_info["rotation_gauge"]["orthonormal"] == 1
    # a re-solve reuses the cached workspace, and reports the verdict it was built with
    stokes.solve()
    assert stokes._rotated_freeslip_info["declared_modes_exempt"] == [True]


@pytest.mark.parametrize("amplitude, exempt", [(1.0e-6, False), (1.0e-8, False), (1.0e-10, True)],
                         ids=["1e-6", "1e-8", "1e-10"])
def test_a_wall_moved_off_its_circle_loses_the_exemption(amplitude, exempt):
    # the outer wall pushed off its circle by `amplitude` of its radius, against the
    # 1e-9 tolerance on the spread of the wall's vertices
    deformed = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.2, qdegree=3)
    X = np.array(deformed.X.coords)
    r = np.linalg.norm(X, axis=1)
    outer = np.isclose(r, 1.0, atol=1.0e-9)
    theta = np.arctan2(X[:, 1], X[:, 0])
    X[outer] *= (1.0 + amplitude * np.cos(3.0 * theta[outer]))[:, None]
    deformed.deform(X)
    mesh, v, stokes = _annulus_stokes(f"1131d{int(-np.log10(amplitude))}", mesh=deformed)
    for wall in ("Lower", "Upper"):
        stokes.add_rotated_freeslip_bc(0, wall)
    stokes.petsc_use_pressure_nullspace = True
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        stokes.solve()
    info = stokes._rotated_freeslip_info
    assert info["declared_modes_exempt"] == [exempt]
    warned = any("'Upper' is not a sphere about the rotation centre" in str(w.message)
                 for w in record)
    with uw.selective_ranks(0) as selected:
        if selected:
            assert warned != exempt, [str(w.message) for w in record]
    # refused, the facet-averaged normal leaves the rotation a violation well above 1e-8
    assert info["rotation_gauge"]["orthonormal"] == (1 if exempt else 0), info["rotation_gauge"]
    assert _normal_velocity(info) < 1.0e-12


def test_mixed_given_and_default_normals_warn_that_the_rotation_stays():
    # the exemption needs every wall on the default normal; a mix is refused, loudly,
    # because the default-normal wall then fails the constraint test by its faceting
    mesh, v, stokes = _annulus_stokes("1131x")
    x, y = mesh.X
    r = sympy.sqrt(x ** 2 + y ** 2)
    stokes.add_rotated_freeslip_bc(0, "Lower", normal=sympy.Matrix([[x / r, y / r]]))
    stokes.add_rotated_freeslip_bc(0, "Upper")
    stokes.petsc_use_pressure_nullspace = True
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        stokes.solve()
    assert stokes._rotated_freeslip_info["declared_modes_exempt"] == [False]
    with uw.selective_ranks(0) as selected:
        if selected:
            assert any("take a given normal and the others the facet-averaged one"
                       in str(w.message) for w in record), [str(w.message) for w in record]


def _rotation_rate_2d(v):
    """Least-squares rigid rotation rate of a 2-D nodal velocity."""
    X = np.asarray(v.coords_nd)
    V = np.asarray(v.array).reshape(X.shape)
    return float(np.sum(X[:, 0] * V[:, 1] - X[:, 1] * V[:, 0])) / float(np.sum(X * X))


def test_a_move_that_keeps_the_walls_keeps_the_declared_mode():
    # an MMPDE-style move: every vertex slides along its circle, and the interior
    # vertices also move radially by an amount that vanishes at both walls. The
    # detection route drops the mode here (violation 0.0037 against 1e-8) and leaves
    # a rotation of rate 2.5e-6 in a v_rms of 5.0e-3
    moved = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.2, qdegree=3)
    X = np.array(moved.X.coords)
    r = np.linalg.norm(X, axis=1)
    theta = np.arctan2(X[:, 1], X[:, 0])
    theta = theta + 0.05 * np.sin(3.0 * theta)
    r = r + 0.02 * np.sin(np.pi * (r - 0.5) / 0.5) * np.cos(2.0 * theta)
    moved.deform(np.column_stack([r * np.cos(theta), r * np.sin(theta)]))
    mesh, v, stokes = _annulus_stokes("1131m", mesh=moved)
    for wall in ("Lower", "Upper"):
        stokes.add_rotated_freeslip_bc(0, wall)
    stokes.petsc_use_pressure_nullspace = True
    stokes.solve()
    info = stokes._rotated_freeslip_info
    assert info["converged"], info
    assert info["declared_modes_exempt"] == [True]
    assert info["rotation_gauge"]["orthonormal"] == 1, info["rotation_gauge"]
    assert abs(_rotation_rate_2d(v)) < 1.0e-10 * _v_rms(v)
    assert _normal_velocity(info) < 1.0e-12


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
    assert stokes._rotated_freeslip_info["declared_modes_exempt"] is None
    assert stokes._rotated_freeslip_info["rotation_gauge"]["orthonormal"] == 0


def test_a_declared_rotation_pinned_by_a_no_slip_wall_is_rejected():
    mesh, v, stokes = _annulus_stokes("1131c")
    stokes.add_dirichlet_bc((0.0, 0.0), "Lower")
    stokes.add_rotated_freeslip_bc(0, "Upper")
    stokes.solve()
    gauge = stokes._rotated_freeslip_info["rotation_gauge"]
    assert gauge["offered"] == 1 and gauge["orthonormal"] == 0, gauge


def test_the_setter_declares_only_rigid_rotations():
    source = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.2, qdegree=3)
    mesh = uw.discretisation.Mesh(source.dm.clone(), simplex=True, qdegree=3,
                                  boundaries=source.boundaries,
                                  coordinate_system_type=source.CoordinateSystem.coordinate_type)
    assert mesh.nullspace_rotations == []
    x, y = mesh.X
    with pytest.raises(ValueError):
        mesh.nullspace_rotations = [sympy.Matrix([-y, x, 0])]   # three components in 2-D
    with pytest.raises(ValueError):
        mesh.nullspace_rotations = [sympy.Matrix([1, 0])]       # a translation
    with pytest.raises(ValueError):
        mesh.nullspace_rotations = [sympy.Matrix([x, y])]       # a dilation
    mesh.nullspace_rotations = [sympy.Matrix([-(y - 0.1), x - 0.2])]   # about (0.2, 0.1)
    assert len(mesh.nullspace_rotations) == 1
    # round-off asymmetry and an unsimplified constant are still rotations
    mesh.nullspace_rotations = [sympy.Matrix([-0.1 * 3 * y, 0.3 * x])]
    mesh.nullspace_rotations = [sympy.Matrix([-(sympy.sin(1) ** 2 + sympy.cos(1) ** 2) * y, x])]
    box = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0, 0.0),
                                            maxCoords=(1.0, 1.0, 1.0), cellSize=0.5)
    bx, by, bz = box.X
    with pytest.raises(ValueError):
        box.nullspace_rotations = [sympy.Matrix([-by, bx, 1])]   # a screw about z
    box.nullspace_rotations = [sympy.Matrix([-by, bx, 0])]
    mesh.nullspace_rotations = [sympy.Matrix([-y, x])]
    assert len(mesh.nullspace_rotations) == 1
    mesh.nullspace_rotations = []
    assert mesh.nullspace_rotations == []


def test_an_add_fault_child_inherits_the_declaration():
    base = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.2, refinement=1,
                              qdegree=2)
    child = base.add_fault([("Flt", np.array([[0.62, 0.10], [0.85, 0.30]]))])
    assert len(child.nullspace_rotations) == 1
    # written in the child's own coordinate symbols, so it evaluates on the child
    assert child.nullspace_rotations[0].free_symbols <= set(child.X)
    c = np.asarray(child.X.coords)
    field = rotated_bc._declared_rotation_fields(child, c)[0]
    assert np.allclose(field, np.column_stack([-c[:, 1], c[:, 0]]), rtol=0, atol=1.0e-14)
    # and a solve on the child, with the fault a traction-free crack, keeps it exempt
    mesh, v, stokes = _annulus_stokes("1131f", mesh=child)
    for wall in ("Lower", "Upper"):
        stokes.add_rotated_freeslip_bc(0, wall)
    stokes.petsc_use_pressure_nullspace = True
    stokes.solve()
    info = stokes._rotated_freeslip_info
    assert info["converged"], info
    assert info["declared_modes_exempt"] == [True]
    assert info["rotation_gauge"]["orthonormal"] == 1, info["rotation_gauge"]
    assert abs(_rotation_rate_2d(v)) < 1.0e-10 * _v_rms(v)


@pytest.mark.parametrize("velocity_units", [None, "cm/year"], ids=["plain_velocity", "unit_velocity"])
def test_a_units_model_shell_solves_with_its_declared_modes(velocity_units):
    # #877: the modes were built from the dimensional coordinates, a UnitAwareArray in a
    # units model, on which np.zeros_like raised TypeError in 3-D (rotated_bc.py:3023 on
    # development eddab707); they are built from coords_nd now. With a velocity that
    # carries units they were then written through its .array, which refuses a plain
    # array ("Cannot assign plain array", hyperion on d992064d)
    uw.reset_default_model()
    uw.get_default_model().set_reference_quantities(
        length=uw.quantity(1000, "km"), viscosity=uw.quantity(1e21, "Pa*s"),
        time=uw.quantity(1, "Myr"))
    try:
        mesh = uw.meshing.SphericalShell(radiusOuter=2.0, radiusInner=1.0, cellSize=0.5,
                                         refinement=1, qdegree=3)
        v = uw.discretisation.MeshVariable(f"V1131u{velocity_units is not None:d}", mesh, 3,
                                           degree=2, units=velocity_units)
        p = uw.discretisation.MeshVariable(f"P1131u{velocity_units is not None:d}", mesh, 1,
                                           degree=1)
        stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
        stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
        stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
        x, y, z = mesh.X
        r = sympy.sqrt(x ** 2 + y ** 2 + z ** 2)
        stokes.bodyforce = (sympy.cos(4 * sympy.atan2(y, x)) * (z / r)
                            * sympy.Matrix([x / r, y / r, z / r]).T)
        for wall in ("Lower", "Upper"):
            stokes.add_rotated_freeslip_bc(0, wall)
        stokes.petsc_use_pressure_nullspace = True
        stokes.solve()
        info = stokes._rotated_freeslip_info
        assert info["converged"], info
        assert info["declared_modes_exempt"] == [True, True, True]
        assert info["rotation_gauge"]["orthonormal"] == 3, info["rotation_gauge"]
        assert _normal_velocity(info) < 1.0e-12
    finally:
        uw.reset_default_model()


def test_a_given_normal_is_tested_not_exempt():
    # a circular wall with a given normal that is not radial: the rotation is not
    # tangent to it, so it must take the constraint test and fail it. Exempted, the
    # solve diverged to a v_rms of 2e7 (hyperion, on c6c2668a)
    mesh, v, stokes = _annulus_stokes("1131n")
    x, y = mesh.X
    r = sympy.sqrt(x ** 2 + y ** 2)
    tilted = sympy.Matrix([[x / r + 0.1, y / r]]) / sympy.sqrt((x / r + 0.1) ** 2 + (y / r) ** 2)
    for wall in ("Lower", "Upper"):
        stokes.add_rotated_freeslip_bc(0, wall, normal=tilted)
    stokes.petsc_use_pressure_nullspace = True
    stokes.solve()
    info = stokes._rotated_freeslip_info
    assert info["converged"], info
    assert info["declared_modes_exempt"] == [False]
    assert info["rotation_gauge"]["orthonormal"] == 0, info["rotation_gauge"]
    assert _normal_velocity(info) < 1.0e-12


def test_a_segmented_shell_keeps_its_modes():
    # its Upper label also holds interior vertices (r down to 0.62); the walls are
    # read from the closure of the labelled facets, which lie on the sphere
    mesh = uw.meshing.SegmentedSphericalShell(radiusOuter=1.0, radiusInner=0.5,
                                              cellSize=0.4, numSegments=4, qdegree=3)
    v = uw.discretisation.MeshVariable("V1131s", mesh, 3, degree=2)
    p = uw.discretisation.MeshVariable("P1131s", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    x, y, z = mesh.X
    r = sympy.sqrt(x ** 2 + y ** 2 + z ** 2)
    stokes.bodyforce = (sympy.cos(4 * sympy.atan2(y, x)) * (z / r)
                        * sympy.Matrix([x / r, y / r, z / r]).T)
    for wall in ("Lower", "Upper"):
        stokes.add_rotated_freeslip_bc(0, wall)
    stokes.petsc_use_pressure_nullspace = True
    stokes.solve()
    info = stokes._rotated_freeslip_info
    assert info["converged"], info
    assert info["declared_modes_exempt"] == [True, True, True]
    assert info["rotation_gauge"]["orthonormal"] == 3, info["rotation_gauge"]
    assert np.abs(_rotation_rates(v)).max() < 1.0e-10 * _v_rms(v)


def test_a_sphere_off_the_origin_is_exempt_about_its_own_centre():
    mesh = uw.meshing.SphericalShell(radiusOuter=1.0, radiusInner=0.5, cellSize=0.5,
                                     qdegree=2)
    mesh.deform(np.array(mesh.X.coords) + np.array([0.0, 0.0, 0.5]))
    x, y, z = mesh.X
    zc = z - 0.5
    mesh.nullspace_rotations = [sympy.Matrix([0, -zc, y]), sympy.Matrix([zc, 0, -x]),
                                sympy.Matrix([-y, x, 0])]
    assert np.allclose(rotated_bc._declared_centre(mesh), [0.0, 0.0, 0.5], atol=1.0e-12)
    v = uw.discretisation.MeshVariable("V1131o", mesh, 3, degree=2)
    p = uw.discretisation.MeshVariable("P1131o", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    for wall in ("Lower", "Upper"):
        stokes.add_rotated_freeslip_bc(0, wall)
    assert rotated_bc._declared_mode_flags(stokes) == ([True, True, True], None)
    # a single declared rotation in 3-D leaves its centre a line: no exemption
    mesh.nullspace_rotations = [sympy.Matrix([-y, x, 0])]
    assert rotated_bc._declared_centre(mesh) is None


def test_the_declaration_is_part_of_the_cached_geometry_and_the_adjoint_record():
    mesh, v, stokes = _annulus_stokes("1131g")
    for wall in ("Lower", "Upper"):
        stokes.add_rotated_freeslip_bc(0, wall)
    stokes.petsc_use_pressure_nullspace = True
    walls = stokes._rotated_freeslip_bcs
    stokes.solve()
    before = rotated_bc._rotated_geometry_signature(stokes, walls)
    names = ("_rotated_velocity_null_modes", "_rotated_null_mode_slack",
             "_rotated_declared_modes_exempt")
    forward = {name: getattr(stokes, name) for name in names}
    assert forward["_rotated_declared_modes_exempt"] == [True]
    # a changed declaration changes the geometry the workspace was built from
    mesh.nullspace_rotations = []
    assert rotated_bc._rotated_geometry_signature(stokes, walls) != before
    # the adjoint builds its own null space (none, now) and puts the forward's
    # record back, which a cached forward solve reads again
    dual = misfit_duals(sympy.Rational(1, 2) * (v.sym[0] ** 2 + v.sym[1] ** 2), [v])[v]
    mu = uw.discretisation.MeshVariable("mu1131g", mesh, 2, degree=2)
    lam = uw.discretisation.MeshVariable("lam1131g", mesh, 1, degree=1)
    stokes.adjoint_solve((dual, None), target=(mu, lam))
    assert {name: getattr(stokes, name) for name in names} == forward


def test_a_wall_is_read_from_its_facets_not_every_point_its_label_holds():
    # a label can hold points off the wall: Linux's gmsh labels interior vertices
    # SegmentedSphericalShell's "Upper" directly (r down to 0.62; hyperion), while the
    # labelled facets lie on the sphere. Label an interior vertex by hand, so the case
    # is the same on every host
    mesh, v, stokes = _annulus_stokes("1131l")
    for wall in ("Lower", "Upper"):
        stokes.add_rotated_freeslip_bc(0, wall)
    dm = mesh.dm
    vs, ve = dm.getDepthStratum(0)
    csec = dm.getCoordinateSection()
    X = dm.getCoordinatesLocal().array.reshape(-1, 2)
    r = np.array([np.linalg.norm(X[csec.getOffset(q) // 2]) for q in range(vs, ve)])
    interior = vs + int(np.argmin(np.abs(r - 0.75)))
    label = dm.getLabel("Upper")
    label.setValue(interior, int(label.getValueIS().getIndices()[0]))
    assert rotated_bc._declared_mode_flags(stokes) == ([True], None)


def test_an_undeclared_box_in_a_units_model_offers_the_cartesian_rotations():
    # #877's other path: a mesh that declares nothing is offered the Cartesian
    # rotations, built with np.zeros_like on the velocity coordinates. In a units model
    # the dimensional ones are a UnitAwareArray, on which that raised TypeError
    uw.reset_default_model()
    uw.get_default_model().set_reference_quantities(
        length=uw.quantity(1000, "km"), viscosity=uw.quantity(1e21, "Pa*s"),
        time=uw.quantity(1, "Myr"))
    try:
        mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0, 0.0),
                                                 maxCoords=(1.0, 1.0, 1.0), cellSize=0.5,
                                                 qdegree=3)
        mesh._nullspace_rotations = None        # undeclared: detection, as before #906
        v = uw.discretisation.MeshVariable("V1131x", mesh, 3, degree=2)
        p = uw.discretisation.MeshVariable("P1131x", mesh, 1, degree=1)
        stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
        stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
        stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
        x, y, z = mesh.X
        stokes.bodyforce = sympy.Matrix([[0, 0, sympy.sin(sympy.pi * x) * sympy.sin(sympy.pi * y)]])
        for wall in ("Bottom", "Top", "Right", "Left", "Front", "Back"):
            stokes.add_rotated_freeslip_bc(0, wall)
        stokes.petsc_use_pressure_nullspace = True
        stokes.solve()
        info = stokes._rotated_freeslip_info
        assert info["converged"], info
        assert info["declared_modes_exempt"] is None
        gauge = info["rotation_gauge"]
        assert gauge["offered"] == 3 and gauge["orthonormal"] == 0, gauge
    finally:
        uw.reset_default_model()
