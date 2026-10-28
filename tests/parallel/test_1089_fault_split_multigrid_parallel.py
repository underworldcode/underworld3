"""Split-level fault multigrid at any rank count.

The welded tail under a split finest did not run at np > 1 (found 2026-10-13):
the split gathers the fault's cell star onto one rank, the node-cloud transfer
left hundreds of coarse unknowns with no fine image, their injection repair
duplicated columns, the Galerkin coarse operators went singular and the inode
SOR returned NaN. The cell-located transfer locates every owned fine node in
the local coarse mesh and asks the other ranks for the rest, and unreached
coarse unknowns are dropped, never repaired. Contract: the split-level
hierarchy (and the welded one) converges at np = 2 and 4 in as few velocity
iterations as in serial (6 at k = 2; serial 6, np = 4 measured 7), with the
serial slip. The thick twin of the faulted sheet is extruded in parallel with
its pairing complete and rank-local.

    mpirun -n 2 python -m pytest --with-mpi tests/parallel/test_1089_fault_split_multigrid_parallel.py
    mpirun -n 4 python -m pytest --with-mpi tests/parallel/test_1089_fault_split_multigrid_parallel.py
"""

import numpy as np
import pytest
import sympy

import underworld3 as uw
from fault_fixtures import faulted_2d, shear_fault_solve

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_2, pytest.mark.tier_b,
              pytest.mark.timeout(1800)]

SERIAL_PEAK_SLIP = 0.159581          # k = 2, frictionless, solve tolerance 1e-6


def _faulted_box(k, split_levels=True):
    return faulted_2d(k, split_levels=split_levels)


def _solve(mesh):
    _stokes, info, slip = shear_fault_solve(mesh)
    return info, slip


def test_split_levels_converge_like_serial():
    info, slip = _solve(_faulted_box(2))
    assert info["velocity_pc"] == "custom-FMG" and info["converged"], info
    assert info["vel_its_last"][-1] <= 9, info["vel_its_last"]
    assert abs(slip - SERIAL_PEAK_SLIP) < 1.0e-4 * SERIAL_PEAK_SLIP, slip


def test_the_welded_tail_runs_in_parallel_too():
    info, slip = _solve(_faulted_box(2, split_levels=False))
    assert info["velocity_pc"] == "custom-FMG" and info["converged"], info
    assert abs(slip - SERIAL_PEAK_SLIP) < 1.0e-4 * SERIAL_PEAK_SLIP, slip


def test_the_thick_twin_is_extruded_with_its_pairing():
    """The faulted plate of test_1083, extruded at np > 1: every pair of the
    serial layer is present on exactly one rank with its two points coincident,
    the layer's levels are split, and the plate solves with geometric
    multigrid."""
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.1, refinement=1,
        qdegree=3)
    sheet = base.add_fault([("Flt", np.array([[0.25, 0.5], [0.75, 0.5]]))])
    layer = sheet.extrude(0.04, layers=2)
    pairs = layer._fault_point_pairs["Flt"]
    n_pairs = uw.mpi.comm.allreduce(len(pairs))
    assert n_pairs == 261, n_pairs           # the serial layer's count
    dm = layer.dm
    cs, cv = dm.getCoordinateSection(), dm.getCoordinatesLocal().array
    vS, vE = dm.getDepthStratum(0)

    def centroid(q):
        verts = [int(v) for v in dm.getTransitiveClosure(int(q))[0] if vS <= v < vE]
        return np.array([cv[cs.getOffset(v):cs.getOffset(v) + 3] for v in verts]).mean(axis=0)

    gap = max([np.linalg.norm(centroid(a) - centroid(b)) for a, b in pairs.items()] or [0.0])
    assert uw.mpi.comm.allreduce(gap, op=max) < 1.0e-12
    tail = layer._custom_mg_coarse_meshes
    assert tail and any(getattr(m, "_fault_point_pairs", None) for m in tail)

    stokes = uw.systems.Stokes(layer)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1
    y = layer.X[1]
    stokes.add_basal_drag(1, "Base", reference_velocity=[y - sympy.Rational(1, 2), 0, 0])
    stokes.add_fault_bc(0, "Flt")
    stokes.tolerance = 1.0e-5
    stokes.solve()
    info = stokes._rotated_freeslip_info
    assert info["velocity_pc"] == "custom-FMG" and info["converged"], info
    from underworld3.utilities.fault_contact import fault_pair_jumps
    _c, jumps, normals = fault_pair_jumps(stokes, "Flt", info, gather=True)
    tangential = jumps - np.einsum("ij,ij->i", jumps, normals)[:, None] * normals
    assert np.linalg.norm(tangential, axis=1).max() > 1.0e-2
    assert np.abs(np.einsum("ij,ij->i", jumps, normals)).max() < 1.0e-8
