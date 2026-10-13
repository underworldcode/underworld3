"""Tests for adding particles to a swarm after initial population.

Verifies that `Swarm.add_particles_with_coordinates()` works correctly
after `swarm.populate()`, including variable data assignment and proxy
mesh variable interpolation.
"""

import numpy as np
import pytest
import sympy as sp

import underworld3 as uw
from underworld3.meshing import UnstructuredSimplexBox

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


@pytest.fixture
def mesh():
    return UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=1.0 / 16.0,
    )


def test_add_particles_after_populate(mesh):
    """Populate swarm, add particles, verify count increases."""
    swarm = uw.swarm.Swarm(mesh)
    swarm.add_variable(name="material", size=1, dtype=int)
    swarm.populate(fill_param=3)

    initial_count = swarm.local_size

    new_coords = np.array([
        [0.25, 0.25],
        [0.5, 0.5],
        [0.75, 0.75],
        [0.1, 0.9],
        [0.9, 0.1],
        [0.3, 0.7],
        [0.7, 0.3],
        [0.4, 0.6],
        [0.6, 0.4],
        [0.5, 0.1],
    ])
    added = swarm.add_particles_with_coordinates(new_coords)

    total_added = uw.mpi.comm.allreduce(added, uw.MPI.SUM)
    assert total_added > 0, "No particles were added"
    assert swarm.local_size > initial_count

    del swarm
    del mesh


def test_add_particles_set_variable_data(mesh):
    """Add particles and set their variable data, verify values."""
    swarm = uw.swarm.Swarm(mesh)
    var = swarm.add_variable(name="temperature", size=1)
    swarm.populate(fill_param=2)

    initial_count = swarm.local_size

    new_coords = np.array([
        [0.25, 0.25],
        [0.5, 0.5],
        [0.75, 0.75],
    ])
    added = swarm.add_particles_with_coordinates(new_coords)
    total_added = uw.mpi.comm.allreduce(added, uw.MPI.SUM)
    assert total_added > 0, f"Expected >0 particles added across all ranks, got {total_added}"

    with swarm.access(var):
        # Set new particles to a known value (e.g., 999.0)
        var.data[initial_count:, 0] = 999.0

    with swarm.access(var):
        assert np.all(var.data[initial_count:, 0] == 999.0)
        # Original particles should still have their initial values (0.0)
        assert np.all(var.data[:initial_count, 0] == 0.0)

    del swarm
    del mesh


def test_proxy_mesh_variable_after_add(mesh):
    """Add particles, verify proxy mesh variable interpolation works."""
    swarm = uw.swarm.Swarm(mesh)
    var = swarm.add_variable(name="velocity", size=2, proxy_degree=1)
    swarm.populate(fill_param=2)

    initial_count = swarm.local_size

    new_coords = np.array([
        [0.5, 0.5],
        [0.25, 0.75],
        [0.75, 0.25],
    ])
    added = swarm.add_particles_with_coordinates(new_coords)
    total_added = uw.mpi.comm.allreduce(added, uw.MPI.SUM)
    assert total_added > 0, f"Expected >0 total particles added across all ranks, got {total_added}"

    with swarm.access(var):
        # Set all particles to a constant value
        var.data[:, 0] = 1.0
        var.data[:, 1] = 2.0

    # Force proxy update from swarm data
    var._update_proxy_if_stale()

    # Access proxy mesh variable — this triggers interpolation from swarm to mesh
    proxy = var._meshVar
    assert proxy is not None

    # Evaluate proxy at particle coordinates — should return values close to 1.0/2.0
    with swarm.access(swarm._particle_coordinates):
        coords = swarm._particle_coordinates.data.copy()

    # proxy is a vector MeshVariable with 2 components; evaluate each
    for i in range(2):
        proxy_values = uw.function.evaluate(proxy.sym[i], coords)
        assert proxy_values is not None
        # With constant field, proxy should recover the constant everywhere
        expected = 1.0 if i == 0 else 2.0
        assert np.allclose(proxy_values[:, 0], expected, atol=1e-10), (
            f"Component {i}: expected {expected}, got {proxy_values[:, 0]}"
        )

    del swarm
    del mesh


def test_add_particles_outside_domain_ignored(mesh):
    """Particles outside the mesh domain should be silently ignored."""
    swarm = uw.swarm.Swarm(mesh)
    swarm.add_variable(name="tag", size=1, dtype=int)
    swarm.populate(fill_param=2)

    initial_count = swarm.local_size

    outside_coords = np.array([
        [-1.0, -1.0],
        [2.0, 2.0],
        [5.0, 5.0],
    ])
    added = swarm.add_particles_with_coordinates(outside_coords)

    assert added == 0, f"Expected 0 particles added, got {added}"
    assert swarm.local_size == initial_count

    del swarm
    del mesh


def test_add_particles_wrong_shape_raises(mesh):
    """Wrong-shaped coordinate arrays should raise ValueError."""
    swarm = uw.swarm.Swarm(mesh)
    swarm.add_variable(name="tag", size=1, dtype=int)
    swarm.populate(fill_param=2)

    # 1D array
    with pytest.raises(ValueError):
        swarm.add_particles_with_coordinates(np.array([0.5, 0.5]))

    # Wrong number of columns
    with pytest.raises(ValueError):
        swarm.add_particles_with_coordinates(np.array([[0.5, 0.5, 0.5]]))

    del swarm
    del mesh


def test_populate_after_add_raises(mesh):
    """Calling populate() after particles exist should raise RuntimeError."""
    swarm = uw.swarm.Swarm(mesh)
    swarm.add_variable(name="tag", size=1, dtype=int)
    swarm.populate(fill_param=2)

    swarm.add_particles_with_coordinates(np.array([[0.5, 0.5]]))

    with pytest.raises(RuntimeError, match="Cannot populate swarm"):
        swarm.populate(fill_param=3)

    del swarm
    del mesh


def test_re_add_after_particles_leave_domain(mesh):
    """Simulate update_swarm(): particles leave domain, re-add from initial snapshot.

    This is the pattern from sandbox_wedge_model.py where return_coords_to_bounds=None
    allows particles to escape. After advection, some cells become empty and need
    re-population from the initial particle snapshot.
    """
    mesh.return_coords_to_bounds = None  # allow particles to escape

    swarm = uw.swarm.Swarm(mesh)
    material = swarm.add_variable(name="material", size=1, dtype=int)
    swarm.populate(fill_param=3)

    # Snapshot initial state (like the model does)
    coords_init = swarm.data.copy()
    material_init = material.data.copy()

    # Set material values
    with swarm.access(material):
        material.data[:, 0] = 1

    # Simulate advection by moving ALL particles out of domain
    with swarm.access(swarm._particle_coordinates):
        swarm._particle_coordinates.data[:, 0] = -999.0
        swarm._particle_coordinates.data[:, 1] = -999.0

    # Now call update_swarm pattern: find empty cells, re-add from snapshot
    nstart, nend = mesh.dm.getHeightStratum(0)
    mesh_cells = np.arange(nstart, nend)

    swarm_cell_ids = mesh.get_closest_local_cells(swarm.data)
    valid_ids = swarm_cell_ids[swarm_cell_ids >= 0]
    cells_with_particles = np.unique(valid_ids)
    empty_cells = mesh_cells[~np.isin(mesh_cells, cells_with_particles)]

    # All cells should be empty since we moved everything out
    assert len(empty_cells) > 0, "Expected some empty cells after moving particles out"

    # Find original particles that were in empty cells
    original_cell_ids = mesh.get_closest_local_cells(coords_init)
    re_add_mask = np.isin(original_cell_ids, empty_cells)
    assert np.any(re_add_mask), "Expected some original particles to match empty cells"

    # Re-add — this is where the model hangs in parallel
    coords_to_add = coords_init[re_add_mask]
    old_size = swarm.local_size
    n_added = swarm.add_particles_with_coordinates(coords_to_add)

    total_added = uw.mpi.comm.allreduce(n_added, uw.MPI.SUM)
    assert total_added > 0, f"Expected particles to be added, got {total_added} total"

    # Set material for newly added particles.
    # NOTE: ALL ranks must enter swarm.access() because it has an MPI barrier.
    with swarm.access(material):
        if n_added > 0:
            material.data[old_size:old_size + n_added, 0] = material_init[re_add_mask, 0]

    # Verify re-addition worked
    assert swarm.local_size == old_size + n_added

    del swarm
    del mesh


def test_re_add_multiple_cycles(mesh):
    """Multiple advection + re-add cycles, matching the model's step loop.

    The model calls update_swarm() every 30 steps. This test simulates
    that pattern: move particles out, re-add, move out again, re-add again.
    """
    mesh.return_coords_to_bounds = None

    swarm = uw.swarm.Swarm(mesh)
    material = swarm.add_variable(name="material", size=1, dtype=int)
    swarm.populate(fill_param=3)

    coords_init = swarm.data.copy()
    material_init = material.data.copy()

    with swarm.access(material):
        material.data[:, 0] = 1

    nstart, nend = mesh.dm.getHeightStratum(0)
    mesh_cells = np.arange(nstart, nend)

    for cycle in range(5):
        # Move some particles out of domain (simulate advection)
        with swarm.access(swarm._particle_coordinates):
            # Move bottom 30% out
            n_local = swarm._particle_coordinates.data.shape[0]
            n_move = max(1, n_local // 3)
            swarm._particle_coordinates.data[:n_move, 0] = -999.0
            swarm._particle_coordinates.data[:n_move, 1] = -999.0

        # update_swarm pattern
        swarm_cell_ids = mesh.get_closest_local_cells(swarm.data)
        valid_ids = swarm_cell_ids[swarm_cell_ids >= 0]
        cells_with_particles = np.unique(valid_ids)
        empty_cells = mesh_cells[~np.isin(mesh_cells, cells_with_particles)]

        if len(empty_cells) == 0:
            continue

        original_cell_ids = mesh.get_closest_local_cells(coords_init)
        re_add_mask = np.isin(original_cell_ids, empty_cells)

        if not np.any(re_add_mask):
            continue

        coords_to_add = coords_init[re_add_mask]
        old_size = swarm.local_size
        n_added = swarm.add_particles_with_coordinates(coords_to_add)

        # NOTE: ALL ranks must enter swarm.access() because it has an MPI barrier.
        with swarm.access(material):
            if n_added > 0:
                material.data[old_size:old_size + n_added, 0] = material_init[re_add_mask, 0]

    del swarm
    del mesh


def test_update_swarm_pattern_parallel(mesh):
    """Exact update_swarm() pattern from the model, with barriers to catch rank desync.

    Simulates: populate, snapshot, partial particle loss, then the exact
    update_swarm() logic with collective checks and add_particles_with_coordinates.
    """
    mesh.return_coords_to_bounds = None

    swarm = uw.swarm.Swarm(mesh)
    material = swarm.add_variable(name="material", size=1, dtype=int)
    swarm.populate(fill_param=3)

    coords_init = swarm.data.copy()
    material_init = material.data.copy()

    with swarm.access(material):
        material.data[:, 0] = 1

    # Simulate advection: move RIGHT half of particles out of domain
    # (like the sandbox model where material is pushed right and escapes)
    with swarm.access(swarm._particle_coordinates):
        mid_x = 0.5  # mesh goes from 0.0 to 1.0
        right_mask = swarm._particle_coordinates.data[:, 0] > mid_x
        swarm._particle_coordinates.data[right_mask, 0] = 999.0
        swarm._particle_coordinates.data[right_mask, 1] = 999.0

    # --- Exact update_swarm() pattern ---
    uw.mpi.comm.barrier()

    swarm_cell_ids = mesh.get_closest_local_cells(swarm.data)
    uw.mpi.comm.barrier()

    nstart, nend = mesh.dm.getHeightStratum(0)
    n_cells = int(nend - nstart)
    mesh_cells = np.arange(nstart, nend)
    uw.mpi.comm.barrier()

    original_swarm_cell_ids = mesh.get_closest_local_cells(coords_init)
    uw.mpi.comm.barrier()

    valid_ids = swarm_cell_ids[swarm_cell_ids >= 0]
    cells_with_particles = np.unique(valid_ids)
    empty_cells = mesh_cells[~np.isin(mesh_cells, cells_with_particles)]

    local_has_work = 1 if len(empty_cells) > 0 else 0
    global_has_work = uw.mpi.comm.allreduce(local_has_work, uw.MPI.MAX)
    uw.mpi.comm.barrier()

    if global_has_work == 0:
        return

    re_add_mask = np.isin(original_swarm_cell_ids, empty_cells)
    local_has_readd = 1 if np.any(re_add_mask) else 0
    global_has_readd = uw.mpi.comm.allreduce(local_has_readd, uw.MPI.MAX)
    uw.mpi.comm.barrier()

    if global_has_readd == 0:
        return

    coords_to_add = coords_init[re_add_mask] if np.any(re_add_mask) else np.empty((0, mesh.dim))
    old_size = swarm.local_size
    uw.mpi.comm.barrier()

    n_added = swarm.add_particles_with_coordinates(coords_to_add)
    uw.mpi.comm.barrier()

    # NOTE: ALL ranks must enter swarm.access() because it has an MPI barrier.
    with swarm.access(material):
        if n_added > 0:
            material.data[old_size:old_size + n_added, 0] = material_init[re_add_mask, 0]

    uw.mpi.comm.barrier()

    del swarm
    del mesh


def test_kdtree_and_save_after_particle_loss(mesh):
    """Test KDTree build + swarm save after particles leave domain."""
    mesh.return_coords_to_bounds = None

    swarm = uw.swarm.Swarm(mesh)
    material = swarm.add_variable(name="material", size=1, dtype=int)
    strain = swarm.add_variable(name="strain", size=1)
    swarm.populate(fill_param=3)

    with swarm.access(material):
        material.data[:, 0] = 1

    # Move right half out
    with swarm.access(swarm._particle_coordinates):
        right_mask = swarm._particle_coordinates.data[:, 0] > 0.5
        swarm._particle_coordinates.data[right_mask, 0] = 999.0
        swarm._particle_coordinates.data[right_mask, 1] = 999.0

    # KDTree build + query
    uw.mpi.comm.barrier()
    kd = swarm._get_kdtree()
    uw.mpi.comm.barrier()

    n_nodes = 100
    query_pts = np.random.rand(n_nodes, mesh.dim) * 0.5
    k = min(5, swarm.local_size)
    if k > 0:
        distances, indices = kd.query(query_pts, k=k, sqr_dists=False)
    uw.mpi.comm.barrier()

    # Swarm save — use a single temp dir (all ranks must access the same file)
    import os, tempfile, shutil
    if uw.mpi.rank == 0:
        outdir = tempfile.mkdtemp()
    else:
        outdir = None
    outdir = uw.mpi.comm.bcast(outdir, root=0)
    swarm.write_timestep("test_kdtree_save", swarmname="swarm", index=0,
                         outputPath=outdir, swarmVars=[material, strain])
    uw.mpi.comm.barrier()

    if uw.mpi.rank == 0:
        shutil.rmtree(outdir, ignore_errors=True)

    del swarm
    del mesh


def test_kdtree_after_particle_loss_only(mesh):
    """Test only KDTree build + query after particles leave domain."""
    mesh.return_coords_to_bounds = None

    swarm = uw.swarm.Swarm(mesh)
    material = swarm.add_variable(name="material", size=1, dtype=int)
    swarm.populate(fill_param=3)

    with swarm.access(material):
        material.data[:, 0] = 1

    with swarm.access(swarm._particle_coordinates):
        right_mask = swarm._particle_coordinates.data[:, 0] > 0.5
        swarm._particle_coordinates.data[right_mask, 0] = 999.0
        swarm._particle_coordinates.data[right_mask, 1] = 999.0

    uw.mpi.comm.barrier()
    kd = swarm._get_kdtree()
    uw.mpi.comm.barrier()

    n_nodes = 100
    query_pts = np.random.rand(n_nodes, mesh.dim) * 0.5
    k = min(5, swarm.local_size)
    if k > 0:
        distances, indices = kd.query(query_pts, k=k, sqr_dists=False)
    uw.mpi.comm.barrier()

    del swarm
    del mesh


def test_swarm_save_after_particle_loss_only(mesh):
    """Test only swarm save after particles leave domain."""
    mesh.return_coords_to_bounds = None

    swarm = uw.swarm.Swarm(mesh)
    material = swarm.add_variable(name="material", size=1, dtype=int)
    strain = swarm.add_variable(name="strain", size=1)
    swarm.populate(fill_param=3)

    with swarm.access(material):
        material.data[:, 0] = 1

    with swarm.access(swarm._particle_coordinates):
        right_mask = swarm._particle_coordinates.data[:, 0] > 0.5
        swarm._particle_coordinates.data[right_mask, 0] = 999.0
        swarm._particle_coordinates.data[right_mask, 1] = 999.0

    import tempfile, shutil, os
    if uw.mpi.rank == 0:
        outdir = tempfile.mkdtemp()
    else:
        outdir = None
    outdir = uw.mpi.comm.bcast(outdir, root=0)
    uw.mpi.comm.barrier()
    swarm.write_timestep("test_save_only", swarmname="swarm", index=0,
                         outputPath=outdir, swarmVars=[material, strain])
    uw.mpi.comm.barrier()

    if uw.mpi.rank == 0:
        shutil.rmtree(outdir, ignore_errors=True)

    del swarm
    del mesh


def test_mesh_access_after_add_particles_hang(mesh):
    """Reproduces the HPC hang: update_swarm + calculate_kd_weights + mesh.access(mat_kd).

    On HPC with many ranks, after add_particles_with_coordinates() with
    particles to re-add, the mesh.access(mat_kd) call hangs.
    This test reproduces the exact pattern from sandbox_wedge_model.py.
    """
    mesh.return_coords_to_bounds = None

    swarm = uw.swarm.Swarm(mesh)
    material = swarm.add_variable(name="material", size=1, dtype=int)
    strain = swarm.add_variable(name="strain", size=1)
    swarm.populate(fill_param=3)

    n_materials = 2
    mat_kd = uw.discretisation.MeshVariable("M_kd", mesh, [1, n_materials], degree=1,
                                             vtype=uw.VarType.MATRIX, continuous=True)

    with swarm.access(material):
        material.data[:, 0] = 1

    uw.mpi.comm.barrier()
    if uw.mpi.rank == 0:
        print(f"  populated: {swarm.local_size} particles per rank", flush=True)

    # Snapshot initial state (like the model)
    coords_init = swarm.data.copy()
    material_init = material.data.copy()

    # --- simulate advection: move right-half particles out of domain ---
    with swarm.access(swarm._particle_coordinates):
        right_mask = swarm._particle_coordinates.data[:, 0] > 0.5
        swarm._particle_coordinates.data[right_mask, 0] = 999.0
        swarm._particle_coordinates.data[right_mask, 1] = 999.0

    uw.mpi.comm.barrier()
    if uw.mpi.rank == 0:
        print(f"  moved all particles out, now re-adding", flush=True)

    # --- update_swarm pattern: find empty cells, re-add from snapshot ---
    nstart, nend = mesh.dm.getHeightStratum(0)
    mesh_cells = np.arange(nstart, nend)

    swarm_cell_ids = mesh.get_closest_local_cells(swarm.data)
    valid_ids = swarm_cell_ids[swarm_cell_ids >= 0]
    cells_with_particles = np.unique(valid_ids)
    empty_cells = mesh_cells[~np.isin(mesh_cells, cells_with_particles)]

    local_has_work = 1 if len(empty_cells) > 0 else 0
    global_has_work = uw.mpi.comm.allreduce(local_has_work, uw.MPI.MAX)
    n_added = 0
    if global_has_work > 0:
        original_cell_ids = mesh.get_closest_local_cells(coords_init)
        re_add_mask = np.isin(original_cell_ids, empty_cells)

        local_has_readd = 1 if np.any(re_add_mask) else 0
        global_has_readd = uw.mpi.comm.allreduce(local_has_readd, uw.MPI.MAX)

        if global_has_readd > 0:
            coords_to_add = coords_init[re_add_mask] if np.any(re_add_mask) else np.empty((0, mesh.dim))
            old_size = swarm.local_size
            n_added = swarm.add_particles_with_coordinates(coords_to_add)

            # Sync material data for newly added particles
            # NOTE: n_added may differ across ranks — DO NOT guard with if n_added > 0
            # because swarm.access(material) has an MPI barrier inside delay_callbacks_global
            with swarm.access(material):
                if n_added > 0:
                    material.data[old_size:old_size + n_added, 0] = material_init[re_add_mask, 0]

    uw.mpi.comm.barrier()
    if uw.mpi.rank == 0:
        print(f"  update_swarm: added {n_added} particles, total now {swarm.local_size}", flush=True)

    # --- calculate_kd_weights (matches model) ---
    k_val = 8
    n_nodes = mat_kd.coords.shape[0]
    n_local = swarm.data.shape[0]
    mat_weights = np.zeros((n_nodes, n_materials))

    if n_local > 0:
        effective_k = min(k_val, n_local)
        kd = swarm._get_kdtree()
        distances, indices = kd.query(mat_kd.coords, k=effective_k, sqr_dists=False)

        material_data = material.data.copy()
        for i in range(n_nodes):
            mat_vals = material_data[indices[i]].flatten()
            for j in range(n_materials):
                mat_weights[i, j] = np.sum(mat_vals == j)
            total = mat_weights[i].sum()
            if total > 0:
                mat_weights[i] /= total
            else:
                mat_weights[i] = 1.0 / n_materials

    uw.mpi.comm.barrier()
    if uw.mpi.rank == 0:
        print(f"  calculate_kd_weights done, about to mesh.access", flush=True)

    # --- mesh.access(mat_kd) — THIS IS WHERE IT HANGS ON HPC ---
    with mesh.access(mat_kd):
        mat_kd.data[...] = mat_weights

    uw.mpi.comm.barrier()
    if uw.mpi.rank == 0:
        print(f"  mesh.access done!", flush=True)

    del swarm
    del mesh


_log = []
dbg = uw.mpi.rank == 0 and (lambda s: (_log.append(s), print(s, flush=True))[-1]) or (lambda s: None)


def test_swarm_access_write_after_add_particles():
    """Reproduces HPC hang: move half particles out, add back, Stokes solve,
    projection solve, then write to a swarm variable inside swarm.access().
    Matches the model's solve_solvers + update_strain pattern.

    Requires MUMPS (available on Setonix). On Mac without MUMPS, the Stokes
    solve will abort — run only on HPC with 16+ ranks."""
    dbg("Creating mesh...")
    mesh = UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=1.0 / 8.0,
    )
    dbg("Mesh created")
    mesh.return_coords_to_bounds = None

    x = mesh.X

    # Stokes system (like model)
    dbg("Creating stokes vars...")
    v_soln = uw.discretisation.MeshVariable("V", mesh, mesh.dim, degree=2)
    p_soln = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)

    dbg("Setting up Stokes...")
    stokes = uw.systems.Stokes(mesh, velocityField=v_soln, pressureField=p_soln)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.saddle_preconditioner = 1.0
    stokes.tolerance = 1e-4
    stokes.petsc_options.delValue("ksp_monitor")
    stokes.petsc_options.delValue("snes_monitor")
    stokes.petsc_options["fieldsplit_velocity_pc_type"] = "lu"
    stokes.petsc_options["fieldsplit_pressure_pc_type"] = "lu"
    stokes.add_dirichlet_bc([0.0, 0.0], "Bottom")
    stokes.add_dirichlet_bc([1.0, 0.0], "Top")
    stokes.add_dirichlet_bc([0.0, None], "Left")
    stokes.add_dirichlet_bc([0.0, None], "Right")
    stokes.add_condition(
        p_soln.field_id, "dirichlet", sp.Matrix([0]), "Top", components=(0)
    )

    # Projection system (like model's updateSR)
    dbg("Creating projection...")
    proj_var = uw.discretisation.MeshVariable("proj", mesh, 1, degree=2)
    edot_expr = stokes.Unknowns.Einv2
    proj = uw.systems.Projection(mesh, proj_var)
    proj.uw_function = edot_expr
    proj.smoothing = 1.0e-6
    proj.petsc_options.delValue("ksp_monitor")

    def update_proj():
        proj.solve(_force_setup=True, zero_init_guess=True)

    dbg("Creating swarm...")
    swarm = uw.swarm.Swarm(mesh)
    material = swarm.add_variable(name="material", size=1, dtype=int)
    strain = swarm.add_variable(name="strain", size=1)
    dbg("Populating swarm...")
    swarm.populate(fill_param=3)

    dbg("Writing material data...")
    with swarm.access(material):
        material.data[:, 0] = 1

    # Save initial state (like model's swarm_coords_init, material_data_init)
    dbg("Saving init state...")
    swarm_coords_init = swarm.data.copy()
    material_data_init = material.data.copy()
    dbg("Getting closest local cells...")
    original_cell_ids = mesh.get_closest_local_cells(swarm_coords_init)
    dbg(f"Init state saved, {len(swarm_coords_init)} coords, {len(original_cell_ids)} cell ids")

    def update_swarm():
        """Model's update_swarm pattern: re-add original particles to empty cells."""
        nstart, nend = mesh.dm.getHeightStratum(0)
        mesh_cells = np.arange(nstart, nend)

        swarm_cell_ids = mesh.get_closest_local_cells(swarm.data)
        valid_ids = swarm_cell_ids[swarm_cell_ids >= 0]
        cells_with_particles = np.unique(valid_ids)
        empty_cells = mesh_cells[~np.isin(mesh_cells, cells_with_particles)]

        local_has_work = 1 if len(empty_cells) > 0 else 0
        global_has_work = uw.mpi.comm.allreduce(local_has_work, uw.MPI.MAX)
        if global_has_work == 0:
            return

        re_add_mask = np.isin(original_cell_ids, empty_cells)
        local_has_readd = 1 if np.any(re_add_mask) else 0
        global_has_readd = uw.mpi.comm.allreduce(local_has_readd, uw.MPI.MAX)
        if global_has_readd == 0:
            return

        # ALWAYS call add_particles_with_coordinates collectively (it calls dm.migrate
        # which is COLLECTIVE). Pass empty array on ranks with no local work.
        coords_to_add = swarm_coords_init[re_add_mask] if np.any(re_add_mask) else np.empty((0, mesh.dim))
        n_added = swarm.add_particles_with_coordinates(coords_to_add)

        # MATERIAL WRITE: entered by ALL ranks (swarm.access has MPI barrier).
        # Only ranks with added particles actually write.
        with swarm.access(material):
            if n_added > 0:
                material.data[-n_added:, 0] = material_data_init[re_add_mask, 0]

    for cycle in range(2):
        dbg(f"=== Cycle {cycle} ===")
        # Move particles out of domain (simulate advection)
        dbg("  Moving particles out...")
        with swarm.access(swarm._particle_coordinates):
            right_mask = swarm._particle_coordinates.data[:, 0] > 0.5
            swarm._particle_coordinates.data[right_mask, 0] = 1.5
            swarm._particle_coordinates.data[right_mask, 1] = 1.5

        uw.mpi.comm.barrier()
        dbg("  Particles moved out")

        # Re-add original particles to empty cells (like model's update_swarm)
        dbg("  update_swarm...")
        update_swarm()
        uw.mpi.comm.barrier()
        dbg("  update_swarm done")

        # Iterative Stokes solve + projection + evaluate + write
        dbg("  Stokes solve...")
        stokes.solve(zero_init_guess=True, picard=0)
        dbg("  First Stokes done, iterative...")
        for it in range(10):
            with mesh.access(v_soln):
                v_old = v_soln.data.copy()
            stokes.solve(zero_init_guess=False, picard=0)
            with mesh.access(v_soln):
                v_new = v_soln.data.copy()
        uw.mpi.comm.barrier()
        dbg("  Iterative Stokes done")

        dbg("  Projection solve...")
        update_proj()
        uw.mpi.comm.barrier()
        dbg("  Projection done")

        dbg("  evaluate...")
        evaluated = uw.function.evaluate(edot_expr, swarm.data).flatten()
        dbg(f"  evaluate done, shape={evaluated.shape}")
        uw.mpi.comm.barrier()
        dbg("  evaluate done")

        dbg("  swarm.access(strain) write...")
        with swarm.access(strain):
            strain.data[:, 0] = strain.data[:, 0] + 0.5 * evaluated
            material_data = material.data[:, 0]
            strain.data[material_data == 0] = 0.

        uw.mpi.comm.barrier()
        dbg("  strain write done")

        with swarm.access(strain):
            assert np.all(strain.data[:, 0] >= 0.0)

    dbg("Cleanup...")
    del swarm
    del proj_var, proj
    del v_soln, p_soln, stokes
    del mesh
    dbg("Done!")

def test_proxy_updates_after_add_particles(mesh):
    """Proxy mesh variable reflects swarm changes after add_particles_with_coordinates.

    The proxy (created via proxy_degree) does RBF interpolation from swarm to
    mesh on demand. After adding particles and changing their values, the proxy
    must be updated via _update_proxy_if_stale() before its data is current.
    """
    swarm = uw.swarm.Swarm(mesh)
    material = swarm.add_variable(name="material", size=1, dtype=int, proxy_degree=2)
    swarm.populate(fill_param=3)

    # Set all particles to material 0 initially
    with swarm.access(material):
        material.data[:, 0] = 0

    # First proxy update (via material.sym access)
    _ = material.sym
    proxy = material._meshVar
    assert proxy is not None
    p1 = proxy.data.copy()

    # Add new particles in the right half with material=1
    xs = np.linspace(0.55, 0.95, 10)
    ys = np.linspace(0.05, 0.95, 10)
    xx, yy = np.meshgrid(xs, ys)
    new_coords = np.column_stack((xx.ravel(), yy.ravel()))

    initial_local_size = max(swarm.dm.getLocalSize(), 0)
    added = swarm.add_particles_with_coordinates(new_coords)

    # Set new particles to material 1
    current_local_size = max(swarm.dm.getLocalSize(), 0)
    n_new_local = current_local_size - initial_local_size
    if n_new_local > 0:
        with swarm.access(material):
            material.data[-n_new_local:, 0] = 1

    # Update proxy from (now-modified) swarm data
    material._update_proxy_if_stale()
    p2 = proxy.data.copy()

    # Verify the proxy updated
    diff = np.max(np.abs(p2 - p1))
    total_added = uw.mpi.comm.allreduce(added, uw.MPI.SUM)
    assert total_added > 0, "No particles were added by any rank"
    assert diff > 1e-6, (
        f"Proxy did not update after swarm change: "
        f"max|p2-p1| = {diff}, total_added={total_added}"
    )

    del proxy, swarm, material

    # Print all accumulated debug messages
    import sys
    for msg in _log:
        print(msg, flush=True)
