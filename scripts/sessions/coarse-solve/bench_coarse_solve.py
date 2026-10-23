"""Coarse-solve benchmark for the rotated free-slip custom FMG on an annulus.

Free slip on both annulus boundaries, so the rotated velocity block carries the
rigid-rotation null mode. The custom-P FMG's coarse solve is swapped after
custom_mg._configure_pcmg installs it, by ``-uw_coarse``:

  svd            the current default (dense SVD on the gathered coarse matrix)
  redundant_lu   redundant + PETSc LU (expected to hit the zero pivot)
  redundant_mumps redundant + MUMPS LU with null-pivot detection (ICNTL(24))
  mumps          distributed MUMPS LU on the whole communicator, ICNTL(24)
  telescope      PCTELESCOPE onto ~4 ranks, then MUMPS LU with ICNTL(24)

Reports, max over ranks: solve wall time, the coarse solve's total time and
count (PETSc event "MGSmooth Level 0"), velocity-block iterations per outer
iteration, nonlinear/outer convergence. Run under mpirun.
"""
import time
import numpy as np
import sympy
from petsc4py import PETSc
import underworld3 as uw
from underworld3.utilities import custom_mg

PETSc.Log.begin()
params = uw.Params(uw_cell_size=0.2, uw_refinement=3, uw_coarse="svd", uw_telescope_ranks=4, uw_logview="")
VARIANT = str(params.uw_coarse)


def _swap_coarse(pc):
    ksp_c = pc.getMGCoarseSolve()
    pc_c = ksp_c.getPC()
    ksp_c.setType("preonly")
    if VARIANT == "svd":
        return
    if VARIANT == "redundant_lu":
        pc_c.setType("redundant")
        pc_c.setFromOptions()
        return
    opts = PETSc.Options()
    if VARIANT == "redundant_mumps":
        pc_c.setType("redundant")
        prefix = pc_c.getOptionsPrefix() or ""
        opts[prefix + "redundant_pc_type"] = "lu"
        opts[prefix + "redundant_pc_factor_mat_solver_type"] = "mumps"
        opts[prefix + "redundant_mat_mumps_icntl_24"] = 1
        pc_c.setFromOptions()
    elif VARIANT == "mumps":
        pc_c.setType("lu")
        pc_c.setFactorSolverType("mumps")
        prefix = pc_c.getOptionsPrefix() or ""
        opts[prefix + "mat_mumps_icntl_24"] = 1
        pc_c.setFromOptions()
    elif VARIANT == "telescope":
        pc_c.setType("telescope")
        prefix = pc_c.getOptionsPrefix() or ""
        size = uw.mpi.size
        opts[prefix + "pc_telescope_reduction_factor"] = max(1, size // int(params.uw_telescope_ranks))
        opts[prefix + "telescope_ksp_type"] = "preonly"
        opts[prefix + "telescope_pc_type"] = "lu"
        opts[prefix + "telescope_pc_factor_mat_solver_type"] = "mumps"
        opts[prefix + "telescope_mat_mumps_icntl_24"] = 1
        pc_c.setFromOptions()
    else:
        raise ValueError(VARIANT)


import os
from underworld3.utilities import rotated_bc

_orig_configure = custom_mg._configure_pcmg


def _configure_logged(pc, *args, **kwargs):
    PETSc.Options()[(pc.getOptionsPrefix() or "") + "pc_mg_log"] = True
    return _orig_configure(pc, *args, **kwargs)


custom_mg._configure_pcmg = _configure_logged
if VARIANT != "svd":
    os.environ["UW_ROTATED_COARSE"] = VARIANT

from underworld3.utilities.place_surface import place_fault_ribbon_2d
base = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=params.uw_cell_size,
                          refinement=int(params.uw_refinement), qdegree=3)
levels = base._coarse_level_meshes()
_w = 0.03
_r = np.arange(0.62, 0.88 + 1e-12, _w / 2)
_trace = np.column_stack([_r * np.cos(np.pi / 6), _r * np.sin(np.pi / 6)])
mesh, _info = place_fault_ribbon_2d(base, [("F", _trace)], _w)
s, e = levels[0].dm.getHeightStratum(0)
coarse_cells = uw.mpi.comm.allreduce(e - s)
s, e = mesh.dm.getHeightStratum(0)
fine_cells = uw.mpi.comm.allreduce(e - s)

v = uw.discretisation.MeshVariable("V", mesh, 2, degree=2)
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)
stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
x, y = mesh.X
r = sympy.sqrt(x ** 2 + y ** 2)
th = sympy.atan2(y, x)
buoyancy = 1.0e3 * sympy.cos(4 * th) * sympy.sin(sympy.pi * (r - 0.5) / 0.5)
stokes.bodyforce = buoyancy * sympy.Matrix([x / r, y / r])
stokes.add_rotated_freeslip_bc(0, "Lower")
stokes.add_rotated_freeslip_bc(0, "Upper")
stokes.add_fault_bc(0.0, "F")
stokes.petsc_use_pressure_nullspace = True
stokes.tolerance = 1.0e-6

t0 = time.perf_counter()
status = "ok"
try:
    stokes.solve(zero_init_guess=True)
except Exception as exc:
    status = f"{type(exc).__name__}: {str(exc)[:160]}"
wall = time.perf_counter() - t0
comm = uw.mpi.comm
from mpi4py import MPI
wall = comm.allreduce(wall, op=MPI.MAX)
info = getattr(stokes, "_rotated_freeslip_info", {}) or {}
uw.pprint(f"BENCH np {uw.mpi.size} coarse_cells {coarse_cells} fine_cells {fine_cells} "
          f"variant {VARIANT}: status {status}; converged {info.get('converged')}; "
          f"velocity_pc {info.get('velocity_pc')}; outer its {info.get('nonlinear_iterations')}; "
          f"ksp_its {info.get('ksp_its')}; wall {wall:.2f}s; "
          f"null modes {getattr(stokes, '_rotated_velocity_null_modes', None)}; "
          f"prototype {rotated_bc.last_coarse_prototype}")

if str(params.uw_logview):
    _viewer = PETSc.Viewer().createASCII(str(params.uw_logview), comm=PETSc.COMM_WORLD)
    PETSc.Log.view(_viewer)
    _viewer.destroy()
