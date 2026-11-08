"""Custom-P multigrid transfers into a SPLIT finest level, in parallel.

A refined patch box (a cuboid with a conforming fault patch, refined twice so
its coarse levels are the multigrid tail) is split along the fault. The split
moves the fault's cell star onto one rank, so the finest partition is no
longer the coarse partition: on that rank thousands of fine nodes lie outside
the LOCAL coarse hull. Two defects followed (found 2026-10-17 on the box fault
model, where the velocity multigrid needed 4 iterations in serial and 22, 37,
55 on 2, 4, 8 ranks):

* the co-partitioned builder snapped those nodes to the nearest local coarse
  node (a wrong row no zero-column count can see), and the automatic rule only
  rebuilt a level across partitions on empty COLUMNS;
* the cross-partition builder ran the barycentric builder's serial ORPHAN
  REPAIR on each rank's slice of the all-gathered coarse cloud, where most
  columns look empty, and overwrote thousands of fine rows with weight-one
  injections onto coarse nodes far away (measured: 20272 of 27785 x-rows wrong
  on 8 ranks, errors up to 2.9 on a box of length 3).

The checks: the installed finest transfer reproduces the coordinate field in
the reduced numbering (exact for a Lagrange interpolation), and the velocity
solve takes no more iterations than a few above serial.
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_2, pytest.mark.tier_b,
              pytest.mark.timeout(1800)]

SERIAL_VEL_ITS = 4          # measured, tolerance 1e-4, this geometry


def _faulted_box():
    from underworld3.utilities.fault_split import split_fault
    Lx, Ly, H = 3.0, 2.0, 1.0
    h, hf, f = 0.2, 0.1, 4
    yf, xa, xb, zb = 1.0, 1.0, 2.0, 0.5
    ztop = H - 2 * hf
    patch = np.array([[xa, yf, zb], [xb, yf, zb], [xb, yf, ztop], [xa, yf, ztop]])
    fine = uw.meshing.BoxInternalPatch(cellSize=f * h, minCoords=(0, 0, 0), maxCoords=(Lx, Ly, H),
                                       patch_points=patch, patch_name="Fault",
                                       patch_cellSize=f * hf, grading_distance=8 * hf,
                                       refinement=2, qdegree=3)
    return split_fault(fine, "Fault")


def _solve(mesh):
    v = uw.discretisation.MeshVariable("v", mesh, 3, degree=2)
    p = uw.discretisation.MeshVariable("p", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.bodyforce = sympy.Matrix([0, 0, 0])
    stokes.tolerance = 1e-4
    stokes.add_dirichlet_bc((0.0, 0.0, 0.0), "Left")
    stokes.add_natural_bc(sympy.Matrix([[-0.01, -0.01, 0.0]]), "Right")
    stokes.add_dirichlet_bc((None, None, 0.0), "Bottom")
    stokes.add_fault_bc(0.0, "Fault")
    stokes.solve(zero_init_guess=True)
    return stokes


def _x_reproduction(stokes, coarse_mesh, P):
    """Max |P x_c - x_f| over the x-component rows, in the reduced numbering."""
    from mpi4py import MPI
    from underworld3.utilities.custom_mg import _coarse_dof_layout, _level_dof_layout
    comm = MPI.COMM_WORLD
    Xc = np.asarray(coarse_mesh._get_coords_for_basis(2, True))
    Xf = np.asarray(stokes.mesh._get_coords_for_basis(2, True))
    lay_c = _coarse_dof_layout(stokes, coarse_mesh, 0)
    lay_f = _level_dof_layout(stokes.dm, 0)
    xc, xf = P.createVecRight(), P.createVecLeft()
    for i in range(Xc.shape[0]):
        g = int(lay_c.l2g[3 * i])
        if lay_c.rstart <= g < lay_c.rend:
            xc.setValue(g, float(Xc[i, 0]))
    for i in range(Xf.shape[0]):
        g = int(lay_f.l2g[3 * i])
        if lay_f.rstart <= g < lay_f.rend:
            xf.setValue(g, float(Xf[i, 0]))
    xc.assemble()
    xf.assemble()
    y = P.createVecLeft()
    P.mult(xc, y)
    d = np.abs(y.getArray() - xf.getArray())
    mask = np.abs(xf.getArray()) > 0          # the x-rows set above (x = 0 is the fixed face)
    return comm.allreduce(float(d[mask].max()) if mask.any() else 0.0, op=MPI.MAX)


def test_split_finest_transfer_reproduces_coordinates_and_converges():
    mesh = _faulted_box()
    assert len(mesh._custom_mg_coarse_meshes) == 2
    stokes = _solve(mesh)
    info = stokes._rotated_freeslip_info
    assert info["velocity_pc"] == "custom-FMG" and info["converged"], info
    hierarchy = stokes._custom_mg["hierarchy"]
    raw = hierarchy.transfers
    coarse = hierarchy.level_meshes[-2]
    err = _x_reproduction(stokes, coarse, raw[-1])
    assert err < 1e-8, f"finest transfer does not reproduce x: max error {err:.3e}"
    assert info["vel_its_last"][-1] <= SERIAL_VEL_ITS + 4, info["vel_its_last"]
