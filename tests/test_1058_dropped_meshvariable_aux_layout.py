"""A dropped MeshVariable must not corrupt the auxiliary data of later solves.

`mesh.vars` holds variables weakly, but a DMPlex cannot shed a field: a
variable that is dropped and garbage-collected leaves its PETSc field in
the DM. Two places used to assume the registry and the DM field list line
up by position:

- `Mesh.update_lvec` zipped `mesh.vars.values()` against the DM's field
  decomposition, so every later variable was packed into the wrong field
  (the orphan's slot) and its own slot stayed at whatever it held;
- the JIT's `petsc_a[]` offsets were a running count over the live
  variables, skipping the orphan's components.

Measured before the fix: a cell-size (P0) field landing in a P2 slot as
garbage, NaN residuals (`DIVERGED_FUNCTION_NANORINF`) in one run and a
subtly wrong answer in the next, depending on when the collector ran.

A mesh owns its variables (since 2026-10-27; before, the default Model held
the only strong reference and a model reset released them), so a variable
is orphaned when it is taken off the mesh: the variable-statistics helpers
delete their temporaries from `mesh.vars` on purpose, and that leaves the
field. The orphan is an ordinary state, not a misuse; ``_orphan`` makes one
the same way.

Run: pixi run python -m pytest tests/test_1058_dropped_meshvariable_aux_layout.py -v
"""
import gc

import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


def _mesh():
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.25, qdegree=3)


def _orphan(mesh, name, components, degree):
    """A variable taken off the mesh and collected: its DM field stays."""
    variable = uw.discretisation.MeshVariable(name, mesh, components, degree=degree)
    del mesh.vars[variable.clean_name]
    del variable
    gc.collect()


def _poisson_with_field_coefficient(mesh, tag):
    """A Poisson solve whose answer depends on an auxiliary field (the
    diffusivity is a MeshVariable), so mis-packed aux data changes it."""
    x, y = mesh.X
    kappa = uw.discretisation.MeshVariable(f"kappa_{tag}", mesh, 1, degree=1)
    kappa.array[:, 0, 0] = uw.function.evaluate(1.0 + 4.0 * x * y, kappa.coords).reshape(-1)
    u = uw.discretisation.MeshVariable(f"u_{tag}", mesh, 1, degree=2)
    poisson = uw.systems.Poisson(mesh, u)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = kappa.sym[0]
    poisson.f = 1.0
    for b in ("Left", "Right", "Top", "Bottom"):
        poisson.add_dirichlet_bc(0.0, b)
    poisson.solve()
    return np.array(u.array), kappa, u


def test_dropped_variable_leaves_an_orphaned_field():
    """The premise: dropping a variable does not shrink the DM."""
    mesh = _mesh()
    n_fields = mesh.dm.getNumFields()
    _orphan(mesh, "temporary", 2, 2)
    assert "temporary" not in mesh.vars
    assert mesh.dm.getNumFields() == n_fields + 1


def test_solve_after_a_dropped_variable_matches_a_clean_mesh():
    reference, _k, _u = _poisson_with_field_coefficient(_mesh(), "ref")

    mesh = _mesh()
    _orphan(mesh, "dropped_vector", 2, 2)
    _orphan(mesh, "dropped_scalar", 1, 1)
    assert mesh.dm.getNumFields() > len(mesh.vars)

    answer, _k, _u = _poisson_with_field_coefficient(mesh, "orphan")
    assert np.allclose(answer, reference, rtol=0, atol=1e-10)


def test_packed_aux_vector_lands_in_the_named_fields():
    mesh = _mesh()
    _orphan(mesh, "dropped", 2, 1)
    assert "dropped" not in mesh.vars
    x, y = mesh.X
    a = uw.discretisation.MeshVariable("a_live", mesh, 1, degree=1)
    a.array[:, 0, 0] = uw.function.evaluate(x + 2 * y, a.coords).reshape(-1)

    mesh.update_lvec()
    names, isets, _dms = mesh.dm.createFieldDecomposition()
    g = mesh.dm.getGlobalVec()
    mesh.dm.localToGlobal(mesh.lvec, g)
    packed = {}
    for name, iset in zip(names, isets):
        sub = g.getSubVector(iset)
        packed[name] = (sub.min()[1], sub.max()[1])
        g.restoreSubVector(iset, sub)
    mesh.dm.restoreGlobalVec(g)

    assert packed["dropped"] == (0.0, 0.0)
    lo, hi = packed["a_live"]
    assert lo == pytest.approx(0.0) and hi == pytest.approx(3.0)
