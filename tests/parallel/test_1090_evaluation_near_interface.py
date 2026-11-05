"""Exact-cell interpolation must preserve physical points near shared faces.

A wide ownership tolerance can select a neighbouring cell, after which the
reference-coordinate clamp projects a near-face query onto that face. Both a
continuous kink and a global polynomial are exactly represented here, so their
point errors must be roundoff rather than a finite positional error.
"""

import numpy as np
import pytest
import underworld3 as uw


def near_interface_geometry():
    mesh = uw.meshing.BoxInternalBoundary(
        elementRes=(32,32), zelementRes=(16,16), zintCoord=.5,
        minCoords=(0.,0.), maxCoords=(1.,1.), simplex=True, qdegree=2,
    )
    x = np.random.default_rng(19).uniform(.02,.98,500)
    points = np.concatenate([
        np.column_stack((x,np.full(len(x),.5+delta)))
        for delta in (-1e-4,-1e-6,0.,1e-6,1e-4)
    ])
    return mesh, points


@pytest.mark.parametrize("degree", (1,2))
def test_continuous_gradient_jump_is_evaluated_in_the_containing_cell(degree):
    mesh, points = near_interface_geometry()
    field = uw.discretisation.MeshVariable("Kink",mesh,1,degree=degree)
    field.data[:,0] = np.abs(np.asarray(field.coords)[:,1]-.5)
    values, outside = uw.function.global_evaluate(field.sym[0],points,check_extrapolated=True)
    assert not np.any(outside)
    np.testing.assert_allclose(np.asarray(values).reshape(-1),np.abs(points[:,1]-.5),rtol=0,atol=1e-12)


@pytest.mark.parametrize("degree", (1,2))
def test_polynomial_queries_are_not_projected_onto_nearby_faces(degree):
    mesh, points = near_interface_geometry()
    field = uw.discretisation.MeshVariable("Polynomial",mesh,1,degree=degree)
    nodes = np.asarray(field.coords)
    nodal = 1+nodes[:,0]+2*nodes[:,1]
    expected = 1+points[:,0]+2*points[:,1]
    if degree == 2:
        nodal += nodes[:,0]**2-nodes[:,0]*nodes[:,1]
        expected += points[:,0]**2-points[:,0]*points[:,1]
    field.data[:,0] = nodal
    values, outside = uw.function.global_evaluate(field.sym[0],points,check_extrapolated=True)
    assert not np.any(outside)
    np.testing.assert_allclose(np.asarray(values).reshape(-1),expected,rtol=0,atol=1e-12)
