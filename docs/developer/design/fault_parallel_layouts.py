"""Throughput of the parallel fault network by LAYOUT, everything else
fixed: the same four faults (one alone, a junction-connected pair, one
more) on a 6 x 1 x 1 box whose np=3 partition is three slabs along x
(seams near x = 2 and 4), split realisation, contact solve.

    mpirun -np 3 python -u fault_parallel_layouts.py -uw_layout local|straddle|gathered [-uw_tail 0]

local     : one fault per slab, the pair in the middle slab -> nothing moves
straddle  : the fourth fault shifted onto the seam near x = 4 -> gathered
gathered  : every fault forced into ONE region (the pre-#672 behaviour)
-uw_tail 0: the geometric tail dropped -> the velocity block on GAMG

Reports: build time, regions / cells moved, per-rank cells, Newton and
Krylov counts, cold and warm solve wall time, and the slips (which must
agree across layouts: the answer is partition-independent, the cost is
not). Design note: docs/developer/design/fault-parallel-placement-2026-09.md
"""
import time
import numpy as np
import underworld3 as uw

params = uw.Params(layout=uw.Param("local", "local | straddle | gathered"),
                   tail=uw.Param(1, "1 = geometric tail (custom-FMG), 0 = GAMG"),
                   solve=uw.Param(1, "0 = build only"))
layout = str(params.layout)
if layout == "gathered":
    # one region for the whole network: the old behaviour, for reference
    from underworld3.utilities import place_surface as ps
    _orig = ps._gather_regions
    def one_region(dm, ids, verbose=False, layers=1):
        ids = np.asarray(ids)
        n_ids = dm.getComm().tompi4py().allreduce(int(ids.max()) if ids.size else 0, op=max)
        work, n_region, n_moved, owner, _canon = _orig(
            dm, (ids > 0).astype(np.int32), verbose=verbose, layers=layers)
        return work, n_region, n_moved, owner, {k: 1 for k in range(1, n_ids + 1)}
    ps._gather_regions = one_region

from underworld3.utilities import place_surface as _ps
_g = _ps._gather_regions
_ps._gather_regions = lambda dm, ids, verbose=False, layers=1: _g(dm, ids, verbose=True, layers=layers)
H, W = 0.08, 0.04
def patch(x0, x1, y, z0=0.3, z1=0.7):
    return np.array([[x0, y, z0], [x1, y, z0], [x1, y, z1], [x0, y, z1]])
# a 6 x 1 x 1 box: np=3 slabs of length 2 with seams near x = 2 and 4;
# faults at least 1.5 apart so no two shells touch, whichever layout
# patches 0.2 tall (z 0.4..0.6): the 6:1 box meshes with cells up to 0.4
# across, and the carve refuses a cavity that reaches a wall
A = patch(0.40, 0.70, 0.50, 0.40, 0.60)           # slab 1
B = patch(2.20, 2.60, 0.50, 0.40, 0.60)           # slab 2, senior of the pair
C = np.array([[2.20, 0.62, 0.42], [2.52, 0.30, 0.42],
              [2.52, 0.30, 0.58], [2.20, 0.62, 0.58]])   # crosses B
xD = 5.20 if layout != "straddle" else 3.85       # slab 0, or ON the seam near 4
D = patch(xD, xD + 0.30, 0.50, 0.40, 0.60)

comm = uw.mpi.comm
t0 = time.perf_counter()
faults = []
for name, P in (("A", A), ("B", B), ("C", C), ("D", D)):
    f = uw.meshing.FaultSurface(name, P); f.triangulate(); faults.append(f)
net = uw.meshing.FaultNetwork(faults, hierarchy=["A", "B", "C", "D"])
net.prepare(h=H, ligament=1.5, verbose=False)   # 1.0 degenerates the junction cut on this mesh
net.realisation, net.width = "split", W
net._build_3d_band(h_far=0.24, realisation="split", margin_rings=0.5,
                   carve_clearance=0.3, minCoords=(0.0, 0.0, 0.0),
                   maxCoords=(6.0, 1.0, 1.0))
t_build = time.perf_counter() - t0
mesh = net.mesh
if not int(params.solve):
    cells = comm.gather(int(mesh.dm.getHeightStratum(0)[1]), root=0)
    if comm.rank == 0:
        print(f"[layout] {layout} build-only np={comm.size}: regions {net.info['n_regions']} gathered {net.info['n_gathered']} moved {net.info['n_moved']} cells/rank {cells}", flush=True)
    raise SystemExit(0)
if not int(params.tail):
    mesh._custom_mg_coarse_meshes = None
cells = comm.gather(int(mesh.dm.getHeightStratum(0)[1]), root=0)
band = comm.gather(int(np.count_nonzero(mesh.cells_labelled("Band", 71))), root=0)

x, y, z = mesh.X
v = uw.discretisation.MeshVariable("v", mesh, 3, degree=2)
p = uw.discretisation.MeshVariable("p", mesh, 1, degree=0, continuous=False)
stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
stokes.bodyforce = [0.0, 0.0, 0.0]
for wall in ("Bottom", "Top", "Left", "Right", "Front", "Back"):
    stokes.add_dirichlet_bc((y - 0.5, 0.0, 0.0), wall)
net.apply(stokes)
stokes.petsc_use_pressure_nullspace = True
stokes.tolerance = 1e-5
t1 = time.perf_counter(); info = net.solve(stokes); t_cold = time.perf_counter() - t1
# warm: a full solve again, from zero, with the tail and the rotation reused
t2 = time.perf_counter(); info2 = net.solve(stokes, zero_init_guess=True); t_warm = time.perf_counter() - t2
slips = net.slips(stokes)
peaks = {n: comm.allreduce(float(slips.get(n, 0.0)), op=max) for n in sorted(set(k for k, _ in net.prepared))}
if comm.rank == 0:
    imb = max(cells) / (sum(cells) / len(cells))
    print(f"[layout] {layout} tail={int(params.tail)} np={comm.size}: build {t_build:.1f}s; "
          f"regions {net.info['n_regions']} gathered {net.info['n_gathered']} moved {net.info['n_moved']}; "
          f"cells/rank {cells} (max/mean {imb:.2f}) band/rank {band}", flush=True)
    print(f"[layout] {layout} tail={int(params.tail)}: cold {t_cold:.1f}s warm {t_warm:.1f}s; "
          f"pc={info.get('velocity_pc')} newton={info.get('nonlinear_iterations')} "
          f"converged={info.get('converged')} vel_its={info.get('vel_its_last')} pres_its={info.get('pres_its_last')}; "
          f"warm newton={info2.get('nonlinear_iterations')} vel_its={info2.get('vel_its_last')} reused={info2.get('rotation_reused')}", flush=True)
    print(f"[layout] {layout} tail={int(params.tail)}: slips " + " ".join(f"{n}={s:.5f}" for n, s in peaks.items()), flush=True)
