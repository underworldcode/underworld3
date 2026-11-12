"""The remap of a smooth field in parallel: the bounds that hold in serial, on the wall
and inside, and a ghost holds its owner's value. See
tests/test_0768_remap_without_a_limiter.py.
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b, pytest.mark.timeout(120)]


@pytest.mark.mpi(min_size=2)
def test_the_remap_holds_its_serial_bounds():
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=1 / 8, qdegree=3)
    x, y = mesh.X
    r = sympy.sqrt(x ** 2 + y ** 2)
    field = (r - 0.5) * 2 + 0.5 * sympy.cos(4 * sympy.atan2(y, x)) * r
    T = uw.discretisation.MeshVariable("T0768", mesh, 1, degree=2)
    T.array[:, 0, 0] = uw.function.evaluate(field, T.coords).reshape(-1)

    coords = np.array(mesh.X.coords)
    radius = np.linalg.norm(coords, axis=1)
    wall = (np.abs(radius - 1.0) < 1.0e-9) | (np.abs(radius - 0.5) < 1.0e-9)
    angle = 0.2 * (1 / 8) / 0.5
    rotation = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    coords[wall] = coords[wall] @ rotation
    mesh.deform(coords)

    largest = lambda values: uw.mpi.comm.allreduce(
        float(values.max()) if values.size else 0.0, op=uw.MPI.MAX)
    dof = np.asarray(T.coords).reshape(-1, 2)
    value = np.asarray(T.array).reshape(-1)
    error = np.abs(value - uw.function.evaluate(field, dof).reshape(-1))
    dof_radius = np.linalg.norm(dof, axis=1)
    interior = (dof_radius < 0.99) & (dof_radius > 0.51)
    assert largest(error[interior]) < 2.0e-3     # serial 8.3e-4, under the clamp 5e-2
    assert largest(error[~interior]) < 4.0e-2    # serial 2.7e-2, under the clamp 1e-1

    iset, subdm = mesh.dm.createSubDM(T.field_id)
    gvec = subdm.createGlobalVector()
    lvec = subdm.createLocalVector()
    subdm.localToGlobal(T.vec, gvec, addv=False)
    subdm.globalToLocal(gvec, lvec, addv=False)
    assert largest(np.abs(value - lvec.array)) < 1.0e-12     # 8e-2 under the clamp
