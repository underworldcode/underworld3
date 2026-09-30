"""
Regression tests for EnhancedMeshVariable.clone(name, varsymbol).

Issue #498: the enhanced wrapper declared clone(self) and forwarded no
arguments, while the base MeshVariable.clone requires (name, varsymbol).
Every in-tree caller uses the two-argument form, so clone was broken in
both directions and six shipped examples aborted on it.

These tests assert each documented argument takes effect and that the
clone is structurally identical but has independent data.
"""

import numpy as np
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


@pytest.fixture(scope="module")
def mesh():
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.5
    )


def test_clone_accepts_two_arguments_scalar(mesh):
    """The exact call pattern the examples use: var.clone(name, symbol)."""
    p = uw.discretisation.MeshVariable(
        "P_clone_src", mesh, vtype=uw.VarType.SCALAR, degree=2, varsymbol=r"{p}"
    )
    p2 = p.clone("P_clone_dst", r"{p_2}")

    # name argument takes effect
    assert "P_clone_dst" in p2.name
    # varsymbol argument takes effect
    assert "p_2" in str(p2.symbol)
    # structure is copied
    assert p2.num_components == p.num_components
    assert p2.degree == p.degree
    assert p2.continuous == p.continuous
    assert p2.vtype == p.vtype
    assert p2.mesh is p.mesh


def test_clone_of_clone_vector(mesh):
    """Ex_Stokes_Cartesian_SolC clones a clone: v0 = v.clone(...); v1 = v0.clone(...)."""
    v = uw.discretisation.MeshVariable(
        "V_clone_src", mesh, vtype=uw.VarType.VECTOR, degree=2, varsymbol=r"{v}"
    )
    v0 = v.clone("v0_clone", r"{v_0}")
    v1 = v0.clone("v1_clone", r"{v_1}")

    for cloned, nm in ((v0, "v0_clone"), (v1, "v1_clone")):
        assert nm in cloned.name
        assert cloned.num_components == v.num_components
        assert cloned.degree == v.degree
        assert cloned.vtype == v.vtype

    # clones stay the enhanced type so chained clone/units/math ops work
    assert isinstance(v1, uw.discretisation.MeshVariable)


def test_clone_data_is_independent(mesh):
    """Writing to the clone must not touch the original."""
    t = uw.discretisation.MeshVariable(
        "T_clone_src", mesh, vtype=uw.VarType.SCALAR, degree=1
    )
    t.array[...] = 1.0
    t2 = t.clone("T_clone_dst", r"{T_2}")
    t2.array[...] = 2.0

    assert np.allclose(np.asarray(t.array), 1.0)
    assert np.allclose(np.asarray(t2.array), 2.0)


def test_clone_preserves_discontinuous(mesh):
    """continuous=False must survive the clone (P0/P-disc pressure spaces)."""
    p0 = uw.discretisation.MeshVariable(
        "P0_clone_src", mesh, vtype=uw.VarType.SCALAR, degree=0, continuous=False
    )
    p0c = p0.clone("P0_clone_dst", r"{p_0}")
    assert p0c.continuous is False
    assert p0c.degree == 0
