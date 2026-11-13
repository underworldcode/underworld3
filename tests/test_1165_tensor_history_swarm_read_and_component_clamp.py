"""Three tensor-shaped paths that raised (probe pass, development 01dec8d2).

* #995: ``SemiLagrangian(..., vtype=MATRIX)`` built its history stores without
  the MATRIX shape, so their constructor failed on ``None``.
* #984: ``SwarmVariable.read_timestep`` put the file's values into
  ``self.array[:, 0, :]``, which is the stored layout only for scalars and
  vectors: a tensor raised "could not broadcast".
* #1000: ``global_evaluate`` of ONE component with ``monotone="clamp"`` built
  the bound for every component of the variable and failed to reshape.
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw

# Serial: each rank would get its own tmp_path for the swarm files, and the clamp
# is rank-local by design (its parallel behaviour is a separate defect).
pytestmark = [pytest.mark.level_1,
              pytest.mark.skipif(uw.mpi.size > 1, reason="serial: per-rank tmp_path; rank-local clamp")]


def _mesh():
    return uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1), cellSize=0.25)


@pytest.mark.tier_b
def test_a_matrix_semi_lagrangian_history_holds_the_field_it_tracks():
    mesh = _mesh()
    M = uw.discretisation.MeshVariable("M1165", mesh, (2, 3), vtype=uw.VarType.MATRIX, degree=1)
    v = uw.discretisation.MeshVariable("v1165", mesh, 2, degree=2)
    X = np.asarray(M.coords)
    for i in range(2):
        for j in range(3):
            M.array[:, i, j] = (1 + i) * X[:, 0] + (1 + j) * X[:, 1]
    history = uw.systems.ddt.SemiLagrangian(mesh, M.sym, v.sym, vtype=uw.VarType.MATRIX,
                                            degree=1, continuous=True)
    assert history.psi_star[0].shape == (2, 3)
    history.initialise_history()
    assert np.abs(np.asarray(history.psi_star[0].array) - np.asarray(M.array)).max() < 1e-12
    # Two steps at zero velocity: the trace-back stays at the nodes, so each
    # step stores the field it tracks, exactly.
    for step in range(2):
        history.update_pre_solve(0.1)
        history.update_post_solve(0.1)
        assert np.abs(np.asarray(history.psi_star[0].array) - np.asarray(M.array)).max() < 1e-12, step


@pytest.mark.tier_b
@pytest.mark.parametrize("vtype", [uw.VarType.SYM_TENSOR, uw.VarType.TENSOR])
def test_a_tensor_swarm_variable_round_trips_through_write_and_read_timestep(vtype, tmp_path):
    mesh = _mesh()
    swarm = uw.swarm.Swarm(mesh)
    t = uw.swarm.SwarmVariable(f"t1165_{vtype.name}", swarm, vtype=vtype, proxy_degree=1)
    swarm.populate(fill_param=2)
    stored = np.asarray(t.data)
    values = np.arange(stored.size, dtype=float).reshape(stored.shape) / stored.size
    t.data[...] = values
    swarm.write_timestep("sw1165", "sw", swarmVars=[t], outputPath=str(tmp_path), index=0)
    t.data[...] = 0.0
    t.read_timestep("sw1165", "sw", t.name, 0, outputPath=str(tmp_path))
    # the same swarm, so every particle finds its own saved point
    assert np.abs(np.asarray(t.data) - values).max() == 0.0


def _linear_field(var):
    """Every stored entry a different linear function of x and y."""
    X = np.asarray(var.coords)
    rows, cols = var.sym.shape
    for i in range(rows):
        for j in range(cols):
            if var.vtype == uw.VarType.SYM_TENSOR and j < i:
                continue
            var.array[:, i, j] = (1 + i + 2 * j) * X[:, 0] + (2 - i + j) * X[:, 1]
            if var.vtype == uw.VarType.SYM_TENSOR:
                var.array[:, j, i] = var.array[:, i, j]


@pytest.mark.tier_b
@pytest.mark.parametrize("vtype, shape", [(uw.VarType.TENSOR, None), (uw.VarType.SYM_TENSOR, None),
                                          (uw.VarType.VECTOR, None), (uw.VarType.MATRIX, (2, 3))])
def test_clamping_one_component_equals_that_entry_of_the_whole_clamp(vtype, shape):
    """The clamp of one component is that entry of the whole-variable clamp,
    at every point. This holds whether or not the clamp bites, so the baseline
    does not depend on the mesh. A MATRIX stores entries out of row order
    (a 2 x 3 stores entries 0..5 in columns 0, 2, 4, 1, 3, 5), so it is the
    case where the flat entry index and the stored column differ."""
    mesh = _mesh()
    if shape is None:
        var = uw.discretisation.MeshVariable(f"c1165_{vtype.name}", mesh, vtype=vtype, degree=1)
    else:
        var = uw.discretisation.MeshVariable(f"c1165_{vtype.name}", mesh, shape, vtype=vtype, degree=1)
    _linear_field(var)
    rng = np.random.default_rng(1165)
    pts = 0.02 + 0.96 * rng.random((200, 2))
    whole = np.asarray(uw.function.global_evaluate(var.sym, pts, monotone="clamp"))
    whole = whole.reshape(len(pts), *var.sym.shape)
    rows, cols = var.sym.shape
    for i in range(rows):
        for j in range(cols):
            one = np.asarray(uw.function.global_evaluate(var.sym[i, j], pts, monotone="clamp")).ravel()
            assert np.abs(one - whole[:, i, j]).max() == 0.0, (i, j)


@pytest.mark.tier_b
def test_a_tensor_swarm_variable_round_trips_in_a_units_model(tmp_path):
    """read_timestep in a model with reference quantities. Development raised
    here for every swarm variable shape; the storage-layout read fixes it."""
    uw.reset_default_model()
    model = uw.get_default_model()
    model.set_reference_quantities(domain_depth=uw.quantity(1000, "km"),
                                   plate_velocity=uw.quantity(5, "cm/year"))
    try:
        mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4))
        swarm = uw.swarm.Swarm(mesh)
        t = uw.swarm.SwarmVariable("t1165_units", swarm, vtype=uw.VarType.SYM_TENSOR, proxy_degree=1)
        swarm.populate(fill_param=2)
        stored = np.asarray(t.data)
        values = np.arange(stored.size, dtype=float).reshape(stored.shape) / stored.size
        t.data[...] = values
        swarm.write_timestep("sw1165u", "sw", swarmVars=[t], outputPath=str(tmp_path), index=0)
        t.data[...] = 0.0
        t.read_timestep("sw1165u", "sw", t.name, 0, outputPath=str(tmp_path))
        assert np.abs(np.asarray(t.data) - values).max() == 0.0
    finally:
        uw.reset_default_model()
