"""C on the ladder with the split chain starting at the base spacing: the
test_1022 rig at 0.0025 with four nested SPLIT mids 0.04, 0.02, 0.01, 0.005,
contact only. Companion to ladder_depth_split.py.

The finest level is a placed transfinite ribbon (spacing S) with the spine cut
and split (or, for TI, left whole with the band's footprint painted
transversely isotropic). The mid levels are the same ribbon at 2:1 and 4:1
spacing, welded or split on the same spine vertices. Base: L0 + L1.

python ladder_c.py <outdir>
"""
import json
import os
import sys
import time

import numpy as np
import sympy
import underworld3 as uw
from underworld3.utilities import fault_contact
from underworld3.utilities.custom_mg import set_custom_fmg
from underworld3.utilities.place_surface import place_thin_volume

out = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/+Simulations/fault_split_mg")
os.makedirs(out, exist_ok=True)
H_BG = 1.0 / 8
WIDTH = 0.06
TRACE = np.array([[0.15, 0.42], [0.85, 0.58]])
TIP = 0.04                      # the cut stops this far inside the band, every level
FAULT = "Main"
a, b = TRACE[0], TRACE[-1]
t_hat = (b - a) / np.linalg.norm(b - a)
L = float(np.linalg.norm(b - a))
n_hat = np.array([-t_hat[1], t_hat[0]])


def banded(base, size):
    for clearance in (0.55, 0.7, 0.9):
        try:
            dm, _info = place_thin_volume(
                base.dm, [TRACE], width=WIDTH, label="Zone", label_value=31,
                size=size, clearance=clearance, assembly="fuse", mesher="ladder")
            return uw.discretisation.Mesh(
                dm, simplex=True, qdegree=2, boundaries=base.boundaries,
                coordinate_system_type=base.CoordinateSystem.coordinate_type)
        except RuntimeError:
            continue
    raise RuntimeError(f"band build failed at size {size}")


def spine_line(size):
    """the cut, sampled at the ladder's own rung vertices, tips TIP inside"""
    n_along = max(2, int(round(L / size)))
    step = L / n_along
    k0 = int(round(TIP / step))
    return a + (np.arange(k0, n_along - k0 + 1) * step)[:, None] * t_hat


def split_ribbon(mesh, size):
    return mesh.add_fault((FAULT, spine_line(size)), split_levels=False)


def contact_stokes(mesh, tag):
    x, y = mesh.X
    v = uw.discretisation.MeshVariable(f"v{tag}", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable(f"p{tag}", mesh, 1, degree=1, continuous=True)
    s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    s.bodyforce = [0.0, 0.0]
    for wall in ("Bottom", "Top", "Left", "Right"):
        s.add_dirichlet_bc((y - 0.5, 0.0), wall)
    s.add_fault_bc(0, boundary=FAULT)
    s.petsc_use_pressure_nullspace = True
    s.tolerance = 1e-5
    s.strategy = "robust"
    return s


def ti_stokes(mesh, tag, contrast=1e-3):
    x, y = mesh.X
    v = uw.discretisation.MeshVariable(f"v{tag}", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable(f"p{tag}", mesh, 1, degree=1, continuous=True)
    zone = uw.discretisation.MeshVariable(f"zone{tag}", mesh, 1, degree=0, continuous=False)
    mask = mesh.cells_labelled("Zone", 31)
    zone.array[:, 0, 0] = mask.astype(float)
    s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    s.constitutive_model = uw.constitutive_models.TransverseIsotropicFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    s.constitutive_model.Parameters.shear_viscosity_1 = 1.0 - (1.0 - contrast) * zone.sym[0]
    s.constitutive_model.Parameters.director = sympy.Matrix([[float(n_hat[0]), float(n_hat[1])]])
    s.bodyforce = [0.0, 0.0]
    for wall in ("Bottom", "Top", "Left", "Right"):
        s.add_dirichlet_bc((y - 0.5, 0.0), wall)
    s.petsc_use_pressure_nullspace = True
    s.tolerance = 1e-5
    s.strategy = "robust"
    return s, mask


def chain_density(velpc):
    out = []
    for l in range(velpc.getMGLevels()):
        A = velpc.getMGSmoother(l).getOperators()[0]
        out.append(round(A.getInfo()["nz_used"] / max(A.getSize()[0], 1), 1))
    return out


records = []
base = uw.meshing.UnstructuredSimplexBox(minCoords=(0, 0), maxCoords=(1, 1), cellSize=H_BG,
                                         regular=False, qdegree=2, refinement=1)
L01 = base._coarse_level_meshes()[:-1] + [base]
ribbons = {}
for size in (0.04, 0.02, 0.01, 0.005, 0.0025):
    t0 = time.perf_counter()
    ribbons[size] = banded(base, size)
    print(f"ribbon {size}: {ribbons[size].dm.getHeightStratum(0)[1]} cells "
          f"({time.perf_counter() - t0:.1f} s)", flush=True)
split = {size: split_ribbon(m, size) for size, m in ribbons.items()}

arms = []
for mids in ([0.04, 0.02, 0.01, 0.005],):
    arms.append((f"fine 0.0025: split mid {mids}", 0.0025, [split[m] for m in mids], "contact"))

for k, (name, s_fine, mids, kind) in enumerate(arms):
    tail = L01 + mids
    t0 = time.perf_counter()
    if kind == "contact":
        mesh = split[s_fine]
        stokes = contact_stokes(mesh, f"c{k}")
        set_custom_fmg(stokes, tail, field_id=0)
        info = fault_contact.solve_with_fault(stokes)
        ctx = (getattr(stokes, "_rotated_linear_cache", None) or {})["ctx"]
        velpc = ctx["ksp"].getPC().getFieldSplitSubKSP()[0].getPC()
        _c, jumps, normals = fault_contact.fault_pair_jumps(stokes, FAULT, stokes._rotated_freeslip_info)
        jn = np.einsum("ij,ij->i", jumps, normals)
        rec = {"arm": name, "kind": kind, "vel_its": ctx.get("vel_its_last"),
               "pres_its": ctx.get("pres_its_last"), "converged": bool(info.get("converged")),
               "velocity_pc": velpc.getType(), "nnz_per_row": chain_density(velpc),
               "slip": float(np.linalg.norm(jumps - jn[:, None] * normals, axis=1).max()),
               "leak": float(np.abs(jn).max()),
               "levels": len(tail) + 1, "cells": int(mesh.dm.getHeightStratum(0)[1]),
               "t_solve": round(time.perf_counter() - t0, 1)}
    else:
        mesh = ribbons[s_fine]
        stokes, mask = ti_stokes(mesh, f"t{k}")
        set_custom_fmg(stokes, tail, field_id=0, fac_zone=mask)
        stokes.solve()
        ksp = stokes.snes.getKSP()
        velksp = ksp.getPC().getFieldSplitSubKSP()[0]
        velpc = velksp.getPC()
        # iterations of the velocity sub-solve on the last outer application
        rec = {"arm": name, "kind": kind, "vel_its": velksp.getIterationNumber(),
               "outer_its": ksp.getIterationNumber(), "velocity_pc": velpc.getType(),
               "nnz_per_row": chain_density(velpc) if velpc.getType() == "mg" else None,
               "levels": len(tail) + 1, "cells": int(mesh.dm.getHeightStratum(0)[1]),
               "t_solve": round(time.perf_counter() - t0, 1)}
    records.append(rec)
    print("ARM", json.dumps(rec), flush=True)
json.dump(records, open(os.path.join(out, "ladder_depth_split4.json"), "w"), indent=1)
print("DONE")
