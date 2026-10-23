"""Bug B probe: custom-P FMG build on a small faulted annulus, in parallel.

Records which velocity preconditioner the Stokes solve installed, whether the
custom-P build fell back, and the solve outcome. Run under mpirun.
"""
import warnings
import numpy as np
import sympy
import underworld3 as uw
from underworld3.utilities.place_surface import place_fault_ribbon_2d

params = uw.Params(
    uw_cell_size=0.08,
    uw_refinement=2,
    uw_width=0.03,
)

base = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=params.uw_cell_size,
                          refinement=int(params.uw_refinement), qdegree=3)
levels = base._coarse_level_meshes()
counts = [uw.mpi.comm.allreduce(l.dm.getHeightStratum(0)[1] - l.dm.getHeightStratum(0)[0])
          for l in levels]
uw.pprint(f"PROBE ranks {uw.mpi.size}; coarse level cells {counts}")

w = float(params.uw_width)
r = np.arange(0.62, 0.88 + 1e-12, w / 2)
th = np.deg2rad(30.0)
trace = np.column_stack([r * np.cos(th), r * np.sin(th)])
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    mesh, info = place_fault_ribbon_2d(base, [("F", trace)], w)
    tail = getattr(mesh, "_custom_mg_coarse_meshes", None)
    uw.pprint(f"PROBE placed: tail levels {None if tail is None else len(tail)}; "
              f"addon_partition {getattr(base, '_addon_partition', None) is not None}")

    v = uw.discretisation.MeshVariable("V", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    x, y = mesh.X
    stokes.bodyforce = sympy.Matrix([0, 0]) + 1.0 * sympy.Matrix([-y, x])
    stokes.add_dirichlet_bc((0.0, 0.0), "Lower")
    stokes.add_dirichlet_bc((0.0, 0.0), "Upper")
    stokes.add_fault_bc(0.0, "F")
    stokes.tolerance = 1.0e-5
    stokes.solve(zero_init_guess=True)

info = getattr(stokes, "_rotated_freeslip_info", {}) or {}
pc = stokes.snes.getKSP().getPC().getType()
fallbacks = getattr(stokes, "_pc_fallbacks", None)
uw.pprint(f"PROBE result: outer pc {pc}; velocity_pc {info.get('velocity_pc')}; "
          f"converged {stokes.snes.getConvergedReason()}; fallbacks {fallbacks}")
for wmsg in caught:
    if "custom" in str(wmsg.message).lower() or "fallback" in str(wmsg.message).lower():
        uw.pprint(f"PROBE warning: {str(wmsg.message)[:300]}")
