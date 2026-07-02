"""Local-h scaling for the Nitsche free-slip penalty.

``add_nitsche_bc`` stabilises the constraint with a term ``gamma*mu/h``.
Historically ``h`` was a single GLOBAL scalar (``mesh.get_min_radius()`` — the
smallest cell anywhere). On a non-uniform / adaptive / deforming mesh that
mis-scales the boundary penalty: the *global minimum* cell size is applied on
every facet, even where the boundary cells are much coarser, and it drifts as
refinement (or distortion) changes the global minimum. ``local_h=True`` (the
default) replaces it with a per-cell, deformation-tracking size field
(``mesh.cell_size()``).

What these tests assert — and why they are framed around the *mechanism*
rather than a boundary-``v.n`` leak comparison:

  On a STATIC graded mesh the global-h penalty is the global *minimum*, so it
  is never weaker than the correct local penalty — it *over*-stiffens the
  coarse boundary cells. Over-stiffening gives a SMALLER ``v.n`` leak there
  (it approaches a hard constraint), so a naive "local enforces v.n better
  than global" leak test is actually backwards on a static mesh. The real
  costs of global-h are (a) the wrong asymptotic conditioning of the velocity
  block on graded meshes and (b) spurious boundary-velocity spikes on a
  *deforming* mesh once a distorted/refined cell drags the global minimum far
  below the boundary cell size. (b) is exercised by the held-lid free-surface
  integration in ``~/+Simulations/fs_convection_goal4`` (vhmax ~3000 -> ~300).

So here we assert the things that are both TRUE and decisive on a static mesh:
  1. the penalty size is genuinely LOCAL (matches each cell's size, not the
     global minimum) — and at a coarse free-slip boundary it is many times the
     global minimum that global-h would wrongly use;
  2. the field TRACKS deformation (it is not stale after ``mesh.deform``) —
     this is the property whose absence caused the free-surface spike;
  3. local-h still solves free-slip correctly (matches the essential-BC
     reference and keeps ``v.n`` small on both the coarse and fine ends).

Run: pixi run python -m pytest tests/test_1065_nitsche_local_h.py -v
"""

import os
from enum import Enum

import numpy as np
import pytest
import sympy
from mpi4py import MPI

import underworld3 as uw
from underworld3.discretisation import Mesh
from underworld3.coordinates import CoordinateSystemType

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]


# Parallel-safe global reductions. Every assertion below is made on a value
# that is IDENTICAL on all ranks (a global reduction), so the ranks never
# diverge on a pass/fail decision — a rank-local assertion that passes on one
# rank and fails on another would leave the passing rank spinning forever in
# the next collective.
def _gmin(a):
    a = np.asarray(a).reshape(-1)
    loc = float(a.min()) if a.size else float("inf")
    return uw.mpi.comm.allreduce(loc, op=MPI.MIN)


def _gmax(a):
    a = np.asarray(a).reshape(-1)
    loc = float(a.max()) if a.size else float("-inf")
    return uw.mpi.comm.allreduce(loc, op=MPI.MAX)


def _any(flag):
    return bool(uw.mpi.comm.allreduce(bool(flag), op=MPI.LOR))


class _boundaries_2D(Enum):
    Bottom = 11
    Top = 12
    Right = 13
    Left = 14


class _boundary_normals_2D(Enum):
    Bottom = sympy.Matrix([0, 1])
    Top = sympy.Matrix([0, 1])
    Right = sympy.Matrix([1, 0])
    Left = sympy.Matrix([1, 0])


def _graded_box(h_fine=0.04, h_coarse=0.12):
    """Unit box graded FINE at the Bottom edge, COARSE at the Top edge.

    The Top free-slip boundary therefore sits on cells ~``h_coarse`` while the
    global minimum cell size is ~``h_fine`` (down at the Bottom), so global-h
    would over-stiffen the Top penalty by ~``h_coarse/h_fine``.
    """
    os.makedirs(".meshes", exist_ok=True)
    fname = f".meshes/uw_test_graded_box_{h_fine}_{h_coarse}.msh"

    if uw.mpi.rank == 0:
        import gmsh

        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.model.add("GradedBox")
        p1 = gmsh.model.geo.add_point(0.0, 0.0, 0.0, meshSize=h_fine)
        p2 = gmsh.model.geo.add_point(1.0, 0.0, 0.0, meshSize=h_fine)
        p3 = gmsh.model.geo.add_point(0.0, 1.0, 0.0, meshSize=h_coarse)
        p4 = gmsh.model.geo.add_point(1.0, 1.0, 0.0, meshSize=h_coarse)
        l1 = gmsh.model.geo.add_line(p1, p2, tag=_boundaries_2D.Bottom.value)
        l2 = gmsh.model.geo.add_line(p2, p4, tag=_boundaries_2D.Right.value)
        l3 = gmsh.model.geo.add_line(p4, p3, tag=_boundaries_2D.Top.value)
        l4 = gmsh.model.geo.add_line(p3, p1, tag=_boundaries_2D.Left.value)
        cl = gmsh.model.geo.add_curve_loop((l1, l2, l3, l4))
        surface = gmsh.model.geo.add_plane_surface([cl])
        gmsh.model.geo.synchronize()
        for l, b in ((l1, "Bottom"), (l2, "Right"), (l3, "Top"), (l4, "Left")):
            gmsh.model.add_physical_group(1, [l], l)
            gmsh.model.set_physical_name(1, l, b)
        gmsh.model.addPhysicalGroup(2, [surface], 99999)
        gmsh.model.setPhysicalName(2, 99999, "Elements")
        gmsh.model.mesh.generate(2)
        gmsh.write(fname)
        gmsh.finalize()

    # ensure the file is on disk before non-root ranks read it (cold cache)
    uw.mpi.barrier()

    return Mesh(
        fname,
        degree=1,
        qdegree=3,
        boundaries=_boundaries_2D,
        boundary_normals=_boundary_normals_2D,
        coordinate_system_type=CoordinateSystemType.CARTESIAN,
        useMultipleTags=True,
        useRegions=True,
        markVertices=True,
    )


def _box_wobble(X0, amp):
    """Smooth interior perturbation tapering to zero on all four box edges,
    so boundaries stay put and the mesh stays valid for small ``amp``."""
    c = np.asarray(X0).copy()
    bump = np.sin(np.pi * c[:, 0]) * np.sin(np.pi * c[:, 1])
    c[:, 0] += amp * bump
    c[:, 1] += amp * bump
    return c


# --------------------------------------------------------------------------
# 1. The penalty size is LOCAL, not the global minimum
# --------------------------------------------------------------------------
def test_cell_size_is_local_per_cell():
    """``mesh.cell_size()`` is a per-cell field equal to each cell's
    characteristic size (``mesh._radii``), not the single global minimum."""
    mesh = _graded_box()
    h = mesh.cell_size()  # sympy symbol -> backed by a P0 field
    field = np.asarray(mesh._cell_size_variable.data[:, 0]).reshape(-1)
    radii = np.asarray(mesh._radii).reshape(-1)

    # field exactly mirrors the per-cell characteristic size (rank-local check,
    # reduced to a single global pass/fail so all ranks agree)
    n = min(field.shape[0], radii.shape[0])
    assert not _any(not np.allclose(field[:n], radii[:n]))

    # the mesh is genuinely graded (so local != global is meaningful) — use
    # GLOBAL min/max so the full fine-to-coarse range is seen in parallel too
    gfmin, gfmax = _gmin(field), _gmax(field)
    assert gfmax / gfmin > 3.0

    # the global scalar that global-h would use is just the minimum cell size
    assert np.isclose(mesh.get_min_radius(), gfmin, rtol=1e-6)


def test_local_h_at_coarse_freeslip_boundary_exceeds_global_min():
    """At the COARSE Top free-slip boundary the LOCAL penalty size is many
    times the global minimum. global-h would over-stiffen that penalty by
    exactly this factor; local-h scales it correctly."""
    mesh = _graded_box(h_fine=0.04, h_coarse=0.12)
    # build/exercise the field; its data equals mesh._radii (asserted in
    # test_cell_size_is_local_per_cell), so we read the per-cell sizes directly
    # from _radii / _centroids — a rank-local lookup, avoiding the collective
    # arbitrary-point uw.function.evaluate (which deadlocks in parallel).
    _ = mesh.cell_size()
    cen = np.asarray(mesh._centroids)
    radii = np.asarray(mesh._radii).reshape(-1)

    near_top = cen[:, 1] > 0.85          # cells adjacent to the Top free-slip edge
    h_top = radii[near_top]

    g = mesh.get_min_radius()            # global minimum (the value global-h uses)
    h_top_min = _gmin(h_top)             # smallest LOCAL size along the Top (global)
    # The Top free-slip cells are clearly coarser than the global minimum, so
    # global-h would over-stiffen the Top penalty (gamma*mu/g) by ~ h_top/g.
    assert h_top_min > 2.0 * g


# --------------------------------------------------------------------------
# 2. The field tracks deformation (it is not stale) — the free-surface bug
# --------------------------------------------------------------------------
def test_cell_size_tracks_deformation():
    """After ``mesh.deform`` the cell-size field is refreshed to the new
    geometry — a stale size on a deformed mesh is exactly what re-introduced
    the Nitsche mis-scaling on the free surface."""
    mesh = _graded_box()
    _ = mesh.cell_size()
    h_before = mesh._cell_size_variable.data[:, 0].copy()

    X = np.asarray(mesh.X.coords).copy()
    moved = mesh.deform(_box_wobble(X, amp=0.04))
    assert moved  # geometry actually changed

    h_after = mesh._cell_size_variable.data[:, 0].copy()
    radii_after = np.asarray(mesh._radii).reshape(-1)

    # not stale: the field changed with the geometry SOMEWHERE (global OR) ...
    nb = min(h_after.shape[0], h_before.shape[0])
    assert _any(not np.allclose(h_after[:nb], h_before[:nb]))
    # ... and on EVERY rank it equals the freshly recomputed per-cell sizes.
    na = min(h_after.shape[0], radii_after.shape[0])
    assert not _any(not np.allclose(h_after[:na], radii_after[:na]))


# --------------------------------------------------------------------------
# 3. local-h still solves free-slip correctly (back-compat / correctness)
# --------------------------------------------------------------------------
def _solve_freeslip(mesh, method, gamma=10.0):
    v = uw.discretisation.MeshVariable(
        "U", mesh, mesh.dim, degree=2, vtype=uw.VarType.VECTOR)
    p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.saddle_preconditioner = 1.0
    x, y = mesh.X
    stokes.bodyforce = sympy.Matrix([0, sympy.cos(sympy.pi * x)])
    stokes.add_dirichlet_bc((0.0, 0.0), "Left")
    stokes.add_dirichlet_bc((0.0, 0.0), "Right")
    if method == "essential":
        stokes.add_dirichlet_bc((sympy.oo, 0.0), "Top")
        stokes.add_dirichlet_bc((sympy.oo, 0.0), "Bottom")
    else:
        local = method == "nitsche_local"
        stokes.add_nitsche_bc("Top", gamma=gamma, local_h=local)
        stokes.add_nitsche_bc("Bottom", gamma=gamma, local_h=local)
    stokes.tolerance = 1e-8
    stokes.petsc_options["snes_type"] = "ksponly"
    stokes.petsc_options["ksp_type"] = "fgmres"
    stokes.solve()
    return v


def _vn_rms(mesh, v, boundary):
    Gamma = mesh.Gamma
    vn = v.sym.dot(Gamma)
    num = float(uw.maths.BdIntegral(mesh, fn=vn ** 2, boundary=boundary).evaluate())
    length = float(uw.maths.BdIntegral(mesh, fn=1.0, boundary=boundary).evaluate())
    return (num / length) ** 0.5


def test_local_h_freeslip_solution_is_correct():
    """On the graded mesh, local-h free-slip Nitsche reproduces the
    essential-BC reference and keeps v.n small on BOTH the coarse (Top) and
    fine (Bottom) ends — the constraint holds everywhere, independent of the
    local refinement."""
    mesh_ref = _graded_box()
    v_ref = _solve_freeslip(mesh_ref, "essential")
    vrms_ref = float(np.sqrt(uw.maths.Integral(
        mesh_ref, v_ref.sym.dot(v_ref.sym)).evaluate()))

    mesh = _graded_box()
    v = _solve_freeslip(mesh, "nitsche_local")
    vrms = float(np.sqrt(uw.maths.Integral(mesh, v.sym.dot(v.sym)).evaluate()))

    # matches the exact free-slip (hard-constraint) solution
    assert abs(vrms - vrms_ref) / vrms_ref < 0.02

    # constraint enforced on both the coarse and the fine boundary
    top = _vn_rms(mesh, v, "Top")
    bot = _vn_rms(mesh, v, "Bottom")
    assert top < 1.0e-3
    assert bot < 1.0e-3
