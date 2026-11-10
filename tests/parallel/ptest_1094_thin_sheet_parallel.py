"""ThinSheet and extrude_sheet in parallel (#878).

Neither had a test at np > 1. This file covers the parallel path of
``extrude_sheet`` (the gather onto one rank, ``stamp_point_pairs``, the
distribution, the fault-interior redistribution, ``collect_point_pairs``) and
ThinSheet solves on both of its routes, against closed forms:

* the extruded layer has volume ``area x L``, carries the sheet's walls plus
  ``Base`` and ``Surface``, and its fault pairs are coincident and present at
  both faces, gathered over the ranks;
* a uniform force against a basal drag on a flat sheet gives ``u = L f / c``
  exactly (no strain), on the standard path and with a split-node fault, which
  takes the rotated path;
* on a sphere, a force ``c (omega x X)`` against a drag ``c`` gives the rigid
  rotation ``omega x X`` exactly (a Killing field: no surface strain rate,
  carried exactly by P2), with machine-exact tangency.

Run with:
    mpirun -n 2 python -m pytest --with-mpi tests/parallel/ptest_1094_thin_sheet_parallel.py
    mpirun -n 4 python -m pytest --with-mpi tests/parallel/ptest_1094_thin_sheet_parallel.py
"""
import numpy as np
import pytest
import sympy
from mpi4py import MPI

import underworld3 as uw

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_2, pytest.mark.timeout(900)]

L = 0.04


def _gmax(value):
    return uw.mpi.comm.allreduce(float(value), op=MPI.MAX)


def _flat_sheet(faulted):
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.1, qdegree=3)
    if faulted:
        mesh = mesh.add_fault([("Flt", np.array([[0.3, 0.5], [0.7, 0.5]]))])
    return mesh


@pytest.mark.tier_b
def test_the_extruded_layer_is_the_sheet_times_its_thickness_with_its_pairing():
    layer = _flat_sheet(faulted=True).extrude(L, layers=2)
    uw.discretisation.MeshVariable("probe1094", layer, 1, degree=1)   # Integral needs a field
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

    pairs = layer._fault_point_pairs.get("Flt", {})
    gap = max([np.linalg.norm(centroid(a) - centroid(b)) for a, b in pairs.items()] or [0.0])
    assert _gmax(gap) < 1.0e-12
    assert uw.mpi.comm.allreduce(len(pairs), op=MPI.SUM) > 0
    z = {round(float(coords[q - v_start, 2]), 12) for q in pairs if v_start <= q < v_end}
    z = set().union(*uw.mpi.comm.allgather(z))
    assert round(-L / 2, 12) in z and round(L / 2, 12) in z


@pytest.mark.tier_b
@pytest.mark.parametrize("faulted", [False, True])
def test_a_uniform_force_against_a_drag_gives_u_equal_to_L_f_over_c(faulted):
    mesh = _flat_sheet(faulted)
    sheet = uw.systems.ThinSheet(mesh)
    sheet.constitutive_model = uw.constitutive_models.ViscousFlowModel
    sheet.constitutive_model.Parameters.shear_viscosity_0 = 1
    sheet.tolerance = 1.0e-10
    sheet.thickness = 2
    sheet.basal_drag = 4
    sheet.bodyforce = [1, 0]
    if faulted:
        sheet.add_fault_bc(0, "Flt")
    sheet.solve()
    u = np.asarray(sheet.u.array)[:, 0, :]
    error = np.abs(u - np.array([0.5, 0.0])).max() if len(u) else 0.0
    assert _gmax(error) < 1.0e-8


@pytest.mark.tier_b
def test_a_rigid_rotation_of_the_sphere_is_returned_exactly():
    sphere = uw.meshing.SphericalManifold(radius=1.0, cellSize=0.25, degree=1, qdegree=5)
    x, y, z = sphere.X
    omega = (sympy.Rational(1, 3), sympy.Rational(-1, 2), sympy.Rational(2, 7))
    rot = sympy.Matrix([[omega[1] * z - omega[2] * y,
                         omega[2] * x - omega[0] * z,
                         omega[0] * y - omega[1] * x]])
    sheet = uw.systems.ThinSheet(sphere)
    sheet.constitutive_model = uw.constitutive_models.ViscousFlowModel
    sheet.constitutive_model.Parameters.shear_viscosity_0 = 1
    sheet.tolerance = 1.0e-9
    sheet.basal_drag = 2
    sheet.bodyforce = list(2 * rot)
    sheet.solve()
    assert sheet._rotated_freeslip_info["converged"]
    c = np.asarray(sheet.u.coords)
    u = np.asarray(sheet.u.array)[:, 0, :]
    if len(c):
        exact = np.cross(np.array([float(w) for w in omega]), c)
        error = np.abs(u - exact).max()
        rhat = c / np.linalg.norm(c, axis=1)[:, None]
        leak = np.abs((u * rhat).sum(axis=1)).max()
    else:
        error = leak = 0.0
    assert _gmax(error) < 1.0e-7
    assert _gmax(leak) < 1.0e-13
