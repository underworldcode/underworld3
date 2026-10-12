"""One model, thin or thick: the same model function runs as a thin sheet
(``ThinSheet`` on a surface mesh) and as a thick one (``Stokes`` on the extruded
layer, ``mesh.extrude``), changing only the mesh and the solver class.

The thin sheet is the depth-averaged, plane-stress limit of the layer, so the
two agree to the thin-sheet error: O(c L / eta) for the shear the basal drag
drives through the layer and O((L / ell)^2) for the vertical averaging, with
ell the drag coupling length sqrt(eta L / c). Here L = 0.04, eta = 1, c = 1:
ell = 0.2, c L / eta = 0.04. Measured 2026-10-12 (h = 0.05, two layers): the
mid-plane velocity of the layer differs from the sheet's by 0.72 % unfaulted and
0.53 % faulted (away from the slit), the slip along the fault by 1.6 %.
"""

import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_2]

L, ETA, C = 0.04, 1, 1


def _sheet_mesh(faulted):
    # refinement=1: the sheet's multigrid level is extruded with it, so the
    # layer's Stokes solve gets geometric multigrid (h = 0.05 on the fine level)
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.1, refinement=1, qdegree=3)
    if faulted:
        mesh = mesh.add_fault([("Flt", np.array([[0.25, 0.5], [0.75, 0.5]]))])
    return mesh


def model(mesh, Solver):
    """The model, written once: a viscous plate dragged by a base moving in
    simple shear, free at its edges, optionally cut by a frictionless fault."""
    solver = Solver(mesh)
    solver.constitutive_model = uw.constitutive_models.ViscousFlowModel
    solver.constitutive_model.Parameters.shear_viscosity_0 = ETA
    y = mesh.X[1]
    base_velocity = [y - sympy.Rational(1, 2), 0] + [0] * (mesh.cdim - 2)
    solver.add_basal_drag(C, "Base", reference_velocity=base_velocity)
    if "Flt" in [b.name for b in mesh.boundaries]:
        solver.add_fault_bc(0, "Flt")
    # the contract is the thin-sheet error (percent), not solver round-off
    solver.tolerance = 1.0e-5
    return solver


def _run(faulted):
    sheet_mesh = _sheet_mesh(faulted)
    sheet = model(sheet_mesh, uw.systems.ThinSheet)
    sheet.thickness = L
    sheet.solve()
    layer = model(sheet_mesh.extrude(L, layers=2), uw.systems.Stokes)
    layer.solve()
    return sheet, layer


def _mid_plane(layer, xy):
    """The layer's in-plane velocity on its mid-plane, at the sheet's nodes."""
    pts = np.column_stack([xy, np.zeros(len(xy))])
    return np.asarray(uw.function.evaluate(layer.u.sym, pts)).reshape(-1, 3)[:, :2]


@pytest.mark.tier_b
def test_extruded_layer_geometry_and_pairing():
    """The layer: positive volume (area x L, every cell oriented as PETSc
    expects — inverted cells flip every volume integral against the facet ones,
    and a basal drag pushed), the sheet's walls under their names plus Base and
    Surface, and the fault pairing coincident and open at the surface and the
    base (paired vertices at both faces)."""
    sheet = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.1, qdegree=3)
    split = sheet.add_fault([("Flt", np.array([[0.3, 0.5], [0.7, 0.5]]))])
    layer = split.extrude(L, layers=2)
    uw.discretisation.MeshVariable("probe", layer, 1, degree=1)   # Integral needs a field
    assert abs(uw.maths.Integral(layer, 1).evaluate() - L) < 1.0e-12
    names = {b.name for b in layer.boundaries}
    assert {"Left", "Right", "Top", "Bottom", "Base", "Surface",
            "Flt", "FltPlus", "FltMinus"} <= names

    dm = layer.dm
    v_start, v_end = dm.getDepthStratum(0)
    coords = np.asarray(dm.getCoordinatesLocal().array).reshape(-1, 3)

    def centroid(q):
        verts = [int(r) - v_start for r in dm.getTransitiveClosure(q)[0]
                 if v_start <= r < v_end]
        return coords[verts].mean(axis=0)

    pairs = layer._fault_point_pairs["Flt"]
    assert len(pairs) > 0
    gap = max(np.linalg.norm(centroid(a) - centroid(b)) for a, b in pairs.items())
    assert gap < 1.0e-12
    paired_vertex_z = {round(float(coords[q - v_start, 2]), 12)
                       for q in pairs if v_start <= q < v_end}
    assert round(-L / 2, 12) in paired_vertex_z and round(L / 2, 12) in paired_vertex_z


@pytest.mark.tier_b
def test_dragged_plate_thin_and_thick_agree():
    sheet, layer = _run(faulted=False)
    xy = sheet.u.coords
    u_sheet = np.asarray(sheet.u.array).reshape(-1, 2)
    u_layer = _mid_plane(layer, xy)
    error = np.linalg.norm(u_layer - u_sheet) / np.linalg.norm(u_sheet)
    assert error < 2.0e-2, f"relative difference {error:.3e}"


@pytest.mark.tier_b
def test_faulted_plate_thin_and_thick_agree():
    """The same plate cut by a frictionless fault: on the layer the fault is the
    sheet's, extruded through the thickness (open at the surface and base,
    shared at its tip columns), with ``Stokes.add_fault_bc``. Compared away from
    the slit (point evaluation within a cell of a slit reads across it); the
    fault must slip, or the comparison is vacuous."""
    sheet, layer = _run(faulted=True)
    xy = sheet.u.coords
    u_sheet = np.asarray(sheet.u.array).reshape(-1, 2)
    away = np.abs(xy[:, 1] - 0.5) > 0.06
    u_layer = _mid_plane(layer, xy[away])
    error = np.linalg.norm(u_layer - u_sheet[away]) / np.linalg.norm(u_sheet[away])
    assert error < 2.0e-2, f"relative difference {error:.3e}"

    # the slip: coincident nodes on the slit carry two velocities each; their
    # spread in the fault-parallel component is the slip
    s_sheet = np.asarray(sheet.u.array).reshape(-1, 2)
    on_fault = (np.abs(xy[:, 1] - 0.5) < 1.0e-10) & (np.abs(xy[:, 0] - 0.5) < 0.2)
    assert on_fault.sum() >= 4
    # coincident nodes: two velocities per position, their spread is the slip
    positions = np.round(xy[on_fault, 0], 10)
    spread = [np.ptp(s_sheet[on_fault][positions == p, 0]) for p in np.unique(positions)]
    assert max(spread) > 1.0e-2, f"the sheet's fault does not slip ({max(spread):.2e})"
    xyz = layer.u.coords
    layer_on_fault = ((np.abs(xyz[:, 1] - 0.5) < 1.0e-10) & (np.abs(xyz[:, 0] - 0.5) < 0.2)
                      & (np.abs(xyz[:, 2]) < 1.0e-10))
    u_l = np.asarray(layer.u.array).reshape(-1, 3)[layer_on_fault]
    pos_l = np.round(xyz[layer_on_fault, 0], 10)
    spread_l = {p: np.ptp(u_l[pos_l == p, 0]) for p in np.unique(pos_l)}
    spread_s = {p: np.ptp(s_sheet[on_fault][positions == p, 0]) for p in np.unique(positions)}
    common = sorted(set(spread_l) & set(spread_s))
    assert len(common) >= 4
    slip_l = np.array([spread_l[p] for p in common])
    slip_s = np.array([spread_s[p] for p in common])
    assert np.linalg.norm(slip_l - slip_s) < 0.05 * np.linalg.norm(slip_s), (
        f"slip thick {slip_l} thin {slip_s}")


@pytest.mark.tier_b
def test_dragged_shell_thin_and_thick_agree_on_the_sphere():
    """The same model on the sphere: a sheet on the unit-sphere manifold and the
    shell extruded radially from it, dragged by a base in the toroidal flow
    rhat x grad(z^2). The difference is first order in L — the curvature the
    thin sheet neglects (the base has 1 - L/R of the mid-surface's area) and the
    drag-driven shear c L / eta both scale with L. Measured 2026-10-12
    (h = 0.15, two layers): 3.32 % at L = 0.04, 1.35 % at L = 0.02."""
    sphere = uw.meshing.SphericalManifold(radius=1.0, cellSize=0.15, degree=1, qdegree=3)

    def shell_model(mesh, Solver):
        solver = Solver(mesh)
        solver.constitutive_model = uw.constitutive_models.ViscousFlowModel
        solver.constitutive_model.Parameters.shear_viscosity_0 = 1
        x, y, z = mesh.X
        solver.add_basal_drag(1, "Base", reference_velocity=[2 * y * z, -2 * x * z, 0])
        solver.tolerance = 1.0e-5
        return solver

    thickness = 0.02
    sheet = shell_model(sphere, uw.systems.ThinSheet)
    sheet.thickness = thickness
    sheet.solve()
    shell = shell_model(sphere.extrude(thickness, layers=2), uw.systems.Stokes)
    shell.solve()
    pts = sheet.u.coords
    u_sheet = np.asarray(sheet.u.array).reshape(-1, 3)
    u_shell = np.asarray(uw.function.evaluate(shell.u.sym, pts)).reshape(-1, 3)
    error = np.linalg.norm(u_shell - u_sheet) / np.linalg.norm(u_sheet)
    assert error < 2.5e-2, f"relative difference {error:.3e}"
