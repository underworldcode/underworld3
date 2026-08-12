"""The composed 3-D chain in parallel: place_sheet -> split_fault ->
frictionless contact solve, np=2.

This is the acceptance that caught BOTH #518-era defects (mixed cell
handedness; unreconciled interface cones, #520/#521): topological
gates pass on meshes whose FE operator is wrong, so the gate here is
the SOLVE — converged at serial speed, serial answer, machine-zero
no-opening leak. Run with:

    mpirun -np 2 python -m pytest tests/parallel/\
ptest_0852_place_split_contact.py --with-mpi
"""
import numpy as np
import pytest

import underworld3 as uw
from underworld3.utilities.place_surface import place_sheet
from underworld3.utilities.fault_split import split_fault
from underworld3.utilities import fault_contact

pytestmark = [pytest.mark.parallel_safe, pytest.mark.level_2,
              pytest.mark.tier_b]


def _conditioned_sheet(n=5, half=0.2, tilt=0.25):
    """Tilted structured sheet with corner diagonals flipped so no
    triangle is all-rim (the split's precondition)."""
    u = np.array([1.0, 0.0, tilt])
    u = u / np.linalg.norm(u)
    v = np.array([0.0, 1.0, 0.0])
    s = np.linspace(-half, half, n)
    pts = np.array([np.array([0.5, 0.5, 0.5]) + a * u + b * v
                    for a in s for b in s])
    rim = lambda k: (k // n in (0, n - 1)) or (k % n in (0, n - 1))
    tris = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b = i * n + j, i * n + j + 1
            c, d = (i + 1) * n + j, (i + 1) * n + j + 1
            t1, t2 = (a, b, d), (a, d, c)
            if all(map(rim, t1)) or all(map(rim, t2)):
                t1, t2 = (a, b, c), (b, d, c)
            tris += [t1, t2]
    return pts, np.array(tris, dtype=np.int64)


def test_place_split_contact_np2():
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0, 0.0), maxCoords=(1.0, 1.0, 1.0),
        cellSize=0.11, regular=False, qdegree=2)
    pts, tris = _conditioned_sheet()
    boundaries = base._boundaries_with("Rupture")
    placed, info = place_sheet(base.dm, pts, tris, label="Rupture",
                               label_value=boundaries["Rupture"].value)
    assert info["n_surface_facets"] == len(tris)

    mesh = uw.discretisation.Mesh(
        placed, simplex=True,
        coordinate_system_type=base.CoordinateSystem.coordinate_type,
        qdegree=base.qdegree, boundaries=boundaries, verbose=False)
    child = split_fault(mesh, "Rupture")

    x, y, z = child.X
    v = uw.discretisation.MeshVariable("vPC", child, 3, degree=2)
    p = uw.discretisation.MeshVariable("pPC", child, 1, degree=0,
                                       continuous=False)
    stokes = uw.systems.Stokes(child, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.bodyforce = [0.0, 0.0, 0.0]
    for wall in ("Bottom", "Top", "Left", "Right", "Front", "Back"):
        stokes.add_dirichlet_bc((z - 0.5, 0.0, 0.0), wall)
    stokes.add_fault_bc(0, boundary="Rupture")
    stokes.petsc_use_pressure_nullspace = True
    stokes.tolerance = 1e-5

    fault_info = fault_contact.solve_with_fault(stokes)
    assert fault_info["converged"]

    # rank-local pairs (the fault region lives on one rank; an empty
    # rank is the NORMAL parallel case)
    coords, jumps, normals = fault_contact.fault_pair_jumps(
        stokes, "Rupture", stokes._rotated_freeslip_info)
    if len(jumps):
        jn = np.einsum("ij,ij->i", jumps, normals)
        tang = jumps - jn[:, None] * normals
        peak = float(np.linalg.norm(tang, axis=1).max())
        leak = float(np.abs(jn).max())
    else:
        peak, leak = 0.0, 0.0
    peaks = uw.mpi.comm.allgather(peak)
    leaks = uw.mpi.comm.allgather(leak)
    n_pairs = uw.mpi.comm.allgather(len(jumps))

    assert sum(n_pairs) > 0
    # the serial answer for this configuration: peak 0.1462
    assert 0.10 < max(peaks) < 0.20
    assert max(leaks) < 1e-12
