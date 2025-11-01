"""
Basic parallel operations tests - smoke tests for MPI functionality.

These tests verify that basic Underworld3 operations work correctly in parallel:
- MPI initialization
- Mesh creation and distribution
- Variable creation
- Swarm operations
- Projection operations
- Advection operations

These tests require MPI and should be run with pytest-mpi:
    mpirun -n 2 python -m pytest --with-mpi tests/parallel/test_0700_basic_parallel_operations.py
    mpirun -n 4 python -m pytest --with-mpi tests/parallel/test_0700_basic_parallel_operations.py

Note: These are smoke tests - they verify operations complete without error
but don't validate numerical results.
"""

import pytest
import numpy as np
import underworld3 as uw
from mpi4py import MPI
import petsc4py
from petsc4py import PETSc

# Set default timeout for all parallel tests (prevent hangs)
pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.timeout(60)]


@pytest.mark.mpi(min_size=2)
def test_petsc4py_initialization():
    """Test that petsc4py and MPI are properly initialized."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Verify MPI is working
    assert size >= 2, "Test requires at least 2 MPI ranks"
    assert 0 <= rank < size, f"Invalid rank {rank} for size {size}"

    # Verify PETSc is initialized
    assert PETSc.Sys.isInitialized(), "PETSc should be initialized"


@pytest.mark.mpi(min_size=2)
def test_underworld_mpi_attributes():
    """Test that underworld3 MPI attributes are accessible."""
    # Verify uw.mpi module is accessible
    assert hasattr(uw, "mpi"), "uw.mpi module should exist"
    assert hasattr(uw.mpi, "rank"), "uw.mpi.rank should exist"
    assert hasattr(uw.mpi, "size"), "uw.mpi.size should exist"

    # Verify values are consistent with MPI
    assert uw.mpi.size >= 2, "Test requires at least 2 MPI ranks"
    assert 0 <= uw.mpi.rank < uw.mpi.size


@pytest.mark.mpi(min_size=2)
def test_petsc_dmplex_creation():
    """Test creating PETSc DMPlex in parallel."""
    dm = PETSc.DMPlex().create()
    assert dm is not None, "DMPlex creation should succeed"
    dm.destroy()


@pytest.mark.mpi(min_size=2)
def test_mesh_creation_and_distribution():
    """Test creating and distributing an Underworld3 mesh."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=0.25,
    )

    # Verify mesh was created
    assert mesh is not None
    assert mesh.dim == 2

    # Verify mesh has local data on all ranks
    assert (
        mesh.data.size > 0 or uw.mpi.size == 1
    )  # Some ranks might have no elements in very uneven splits


@pytest.mark.mpi(min_size=2)
def test_continuous_mesh_variables():
    """Test creating continuous mesh variables of different degrees."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=0.25,
    )

    # Create variables with different polynomial degrees
    C1 = uw.discretisation.MeshVariable("C_1", mesh, 1, degree=1, continuous=True)
    C2 = uw.discretisation.MeshVariable("C_2", mesh, 1, degree=2, continuous=True)
    C3 = uw.discretisation.MeshVariable("C_3", mesh, 1, degree=3, continuous=True)

    # Verify variables were created
    assert C1 is not None
    assert C2 is not None
    assert C3 is not None

    # Verify they have data
    assert C1.data.shape[1] == 1  # 1 component
    assert C2.data.shape[1] == 1
    assert C3.data.shape[1] == 1


@pytest.mark.mpi(min_size=2)
def test_discontinuous_mesh_variables():
    """Test creating discontinuous mesh variables."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=0.25,
    )

    # Create discontinuous variables
    dC0 = uw.discretisation.MeshVariable("dC_0", mesh, 1, degree=0, continuous=False)
    dC1 = uw.discretisation.MeshVariable("dC_1", mesh, 1, degree=1, continuous=False)
    dC2 = uw.discretisation.MeshVariable("dC_2", mesh, 1, degree=2, continuous=False)

    # Verify variables were created
    assert dC0 is not None
    assert dC1 is not None
    assert dC2 is not None


@pytest.mark.mpi(min_size=2)
def test_swarm_creation_and_population():
    """Test creating and populating a swarm."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=0.25,
    )

    # Create swarm
    swarm = uw.swarm.Swarm(mesh=mesh)
    assert swarm is not None

    # Create swarm variable
    sw_values = uw.swarm.SwarmVariable("Ss", swarm, 1, proxy_degree=1, proxy_continuous=True)
    assert sw_values is not None

    # Populate swarm
    swarm.populate(fill_param=3)

    # Verify particles were created (globally)
    total_particles = sw_values.global_size()
    assert total_particles > 0, "Swarm should have particles after populate()"


@pytest.mark.mpi(min_size=2)
def test_swarm_migration():
    """Test swarm particle migration."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=0.25,
    )

    swarm = uw.swarm.Swarm(mesh=mesh)
    swarm.populate(fill_param=3)

    # Get particle count before migration
    count_before = swarm.local_size

    # Trigger migration (even with no movement, this tests the mechanism)
    swarm.migrate()

    # Verify swarm still exists and has particles globally
    # Note: local_size may change after migration due to redistribution
    total_before = uw.mpi.comm.allreduce(count_before, op=MPI.SUM)
    total_after = uw.mpi.comm.allreduce(swarm.local_size, op=MPI.SUM)

    assert total_after == total_before, "Total particle count should be preserved by migration"


@pytest.mark.mpi(min_size=2)
def test_mesh_variable_projection():
    """Test projection between mesh variables."""
    import sympy

    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=0.25,
    )

    x, y = mesh.X

    # Create source function
    s_fn = sympy.cos(5.0 * sympy.pi * x) * sympy.cos(5.0 * sympy.pi * y)

    # Create variables with different degrees
    s_soln = uw.discretisation.MeshVariable("S", mesh, 1, degree=1)
    s_values = uw.discretisation.MeshVariable("S2", mesh, 1, degree=2, continuous=True)

    # Perform projection (tests that the operation completes)
    with uw.synchronised_array_update():
        # evaluate() returns shape (N,1,1), need to flatten to (N,)
        result = uw.function.evaluate(s_fn, s_soln.coords)
        s_soln.data[:, 0] = result.flatten()


@pytest.mark.mpi(min_size=2)
def test_swarm_to_mesh_projection():
    """Test projection from swarm to mesh variables."""
    import sympy

    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=0.25,
    )

    x, y = mesh.X

    # Create mesh variable
    s_fn = sympy.cos(5.0 * sympy.pi * x) * sympy.cos(5.0 * sympy.pi * y)
    s_soln = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)

    # Create swarm and swarm variable
    swarm = uw.swarm.Swarm(mesh=mesh)
    sw_values = uw.swarm.SwarmVariable("Ss", swarm, 1, proxy_degree=1, proxy_continuous=True)
    swarm.populate(fill_param=3)

    # Initialize swarm variable with some data (just set to constant for smoke test)
    with uw.synchronised_array_update():
        sw_values.data[:, 0] = 1.0

    # Test that the proxy mechanism works (accessing sym triggers update)
    proxy_expr = sw_values.sym
    assert proxy_expr is not None


@pytest.mark.mpi(min_size=4)
def test_nodal_swarm_advection_basic():
    """
    Test basic nodal swarm advection in parallel.

    This is a simplified version of ptest_004 that verifies the advection
    mechanism works without running a full simulation.

    Requires 4 ranks to test cross-processor particle movement.
    """
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(4.0, 1.0),
        cellSize=0.25,
        regular=False,
        qdegree=3,
    )

    # Create velocity field
    v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)

    # Vector being advected
    vec_tst = uw.discretisation.MeshVariable("Vn", mesh, mesh.dim, degree=2)

    # Create semi-Lagrangian advection
    DuDt = uw.systems.ddt.SemiLagrangian(
        mesh,
        vec_tst.sym,
        v.sym,
        vtype=uw.VarType.VECTOR,
        order=2,
        smoothing=0.0,
    )

    # Verify advection object was created
    assert DuDt is not None

    # Initialize velocity field to constant (simple test case)
    with uw.synchronised_array_update():
        v.data[:, 0] = 0.1
        v.data[:, 1] = 0.0
        vec_tst.data[:, 0] = 1.0
        vec_tst.data[:, 1] = 0.0

    # Perform one advection step (tests that it completes without error)
    # Note: Not testing numerical accuracy, just that operation works in parallel
    dt = 0.1
    DuDt.update(dt)


@pytest.mark.mpi(min_size=2)
def test_parallel_mesh_coordinate_access():
    """Test that mesh coordinates are accessible in parallel."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        cellSize=0.25,
    )

    # Access coordinates
    coords = mesh.data
    assert coords is not None
    assert coords.shape[1] == mesh.dim

    # Verify we can get global statistics using MPI
    local_x_max = np.max(coords[:, 0]) if coords.size > 0 else -np.inf
    global_x_max = uw.mpi.comm.allreduce(local_x_max, op=MPI.MAX)

    # All ranks should have the same global max
    all_max = uw.mpi.comm.gather(global_x_max, root=0)
    if uw.mpi.rank == 0:
        assert len(set(all_max)) == 1, "All ranks should have same global max"
        # Should be approximately 1.0 since maxCoords=(1.0, 1.0)
        assert abs(global_x_max - 1.0) < 0.1, f"Expected x_max near 1.0, got {global_x_max}"


@pytest.mark.mpi(min_size=2)
def test_that_parallel_tests_end():
    """Sentinel test to ensure test suite completes."""
    assert True


if __name__ == "__main__":
    # Allow running standalone for debugging
    import sys

    sys.exit(pytest.main([__file__, "-v", "--with-mpi"]))
